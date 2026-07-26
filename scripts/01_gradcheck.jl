# # 勾配検証：手書き離散随伴 vs 中心差分
#
# **最初に走らせるスクリプト。** ここが通らなければ他は全部無意味なので、
# Metal に行く前にまず CPU + Float64 で通すこと。
#
# 必ず `Float64` で走らせる。Apple GPU は倍精度を持たないので Metal では
# `Float32` 固定になり、そちらでは中心差分側の丸めが効いて 3〜4 桁しか
# 合わない（それ自体は正常な挙動）。
#
# ```
# julia --project=. scripts/01_gradcheck.jl
# ```

using SPHAdjoint
using KernelAbstractions
using Printf
using Random

const T = Float64
backend = CPU()

# ## パラメータ
#
# 検証用なので小さく。粒子 49 個、25 ステップ。

p = SPHParams{T}(h = 0.10, m = 0.010, rho0 = 1.0, c = 8.0, mu = 0.02,
                 dt = 1.0e-3, Lx = 1.0, Ly = 1.0, kw = 2.0e4, ngx = 9, ngy = 9)

# ## 初期配置
#
# 左下のブロック（格子＋微小擾乱）。完全な格子だと対称性で勾配の成分が
# 落ちてしまい検証が甘くなるので、わざと乱数を混ぜている。

rng = MersenneTwister(0)
nx, ny = 7, 7
X0 = zeros(T, 2, nx * ny)
V0 = zeros(T, 2, nx * ny)
k = 0
for jy in 0:ny-1, ix in 0:nx-1
    global k += 1
    X0[1, k] = 0.12 + 0.05 * ix + 0.004 * randn(rng)
    X0[2, k] = 0.12 + 0.05 * jy + 0.004 * randn(rng)
end
V0 .= 0.05 .* randn(rng, T, size(V0))
N = size(X0, 2)

theta0 = 3 .* rand(rng, T, p.ngy, p.ngx)
nsteps = 25

# ## 目的関数
#
# 終端状態のランダム線形汎関数 `J = Σ cX·X_T + Σ cV·V_T` にする。
# こうすると終端随伴の種が `cX`, `cV` そのものになり、しかも全成分に
# 感度が乗るので一度の逆行で全部を検査できる。

cX = randn(rng, T, 2, N)
cV = randn(rng, T, 2, N)

function run_forward(X0, V0, theta; want_tape = false)
    st = State(backend, X0, V0, p)
    th = KernelAbstractions.allocate(backend, T, p.ngy, p.ngx)
    copyto!(th, theta)
    tape = want_tape ? Tuple{typeof(st.X),typeof(st.V)}[] : nothing
    simulate!(st, th, p, backend, nsteps; tape)
    J = sum(Array(st.X) .* cX) + sum(Array(st.V) .* cV)
    return J, st, th, tape
end

@printf("N = %d 粒子, %d ステップ, 設計変数 %d 個\n\n", N, nsteps, length(theta0))

# ## 随伴で勾配を出す

J, st, th, tape = run_forward(X0, V0, theta0; want_tape = true)
ws = AdjointWorkspace(backend, N, p)
backward!(ws, tape, th, p, backend; seedX = cX, seedV = cV)

gth = Array(ws.gtheta)
gX = Array(ws.gX)
gV = Array(ws.gV)
@printf("J = %.12f\n\n", J)

# ## 中心差分と突き合わせる

function fd(f, x, i, hfd)
    xp = copy(x); xp[i] += hfd
    xm = copy(x); xm[i] -= hfd
    return (f(xp) - f(xm)) / (2hfd)
end

hfd = 1e-6

println("dJ/dtheta  (中心差分と比較)")
for idx in [CartesianIndex(1, 1), CartesianIndex(4, 5), CartesianIndex(6, 3)]
    g = fd(t -> run_forward(X0, V0, t)[1], theta0, idx, hfd)
    rel = abs(gth[idx] - g) / max(abs(g), 1e-30)
    @printf("  theta[%d,%d]: 随伴 % .9e   FD % .9e   rel %.2e\n",
            idx[1], idx[2], gth[idx], g, rel)
end

println("\ndJ/dX0  (中心差分と比較)")
for idx in [CartesianIndex(1, 1), CartesianIndex(2, 18), CartesianIndex(1, 41)]
    g = fd(x -> run_forward(x, V0, theta0)[1], X0, idx, hfd)
    rel = abs(gX[idx] - g) / max(abs(g), 1e-30)
    @printf("  X0[%d,%d]:   随伴 % .9e   FD % .9e   rel %.2e\n",
            idx[1], idx[2], gX[idx], g, rel)
end

println("\ndJ/dV0  (中心差分と比較)")
for idx in [CartesianIndex(1, 5), CartesianIndex(2, 30)]
    g = fd(v -> run_forward(X0, v, theta0)[1], V0, idx, hfd)
    rel = abs(gV[idx] - g) / max(abs(g), 1e-30)
    @printf("  V0[%d,%d]:   随伴 % .9e   FD % .9e   rel %.2e\n",
            idx[1], idx[2], gV[idx], g, rel)
end

# ## 期待される結果
#
# 相対誤差は `1e-6` 台まで落ちるはず（中心差分側の打ち切り・丸めが支配する）。
# `1e-2` を超えるようなら随伴か移植のどこかが壊れている。
#
# 参考として、同じ定式化を NumPy に書いて PyTorch の自動微分と比較した結果は
# `dJ/dX0` 2.6e-15, `dJ/dV0` 1.4e-15, `dJ/dθ` 1.0e-15
# （`tools/verify_adjoint.py` および `tools/verify_gather.py`）。
# つまり **式そのものは検証済み** なので、ここで合わなければ疑うべきは
# Julia / KernelAbstractions の作法まわり（README の「初回のチェックポイント」）。

println("\n相対誤差が 1e-6 前後なら OK。1e-2 を超えるなら README の")
println("「初回のチェックポイント」を確認してください。")
