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

# θ のスケールは外から `THETA_SCALE` で差し替えられる。
#
# 既定の 3 は `dt·α ≈ 3e-3`、つまり陰解法の減衰因子 `D = 1/(1+dt·α)` が
# ほぼ 1 のままで、**抗力を陰的にした部分がほとんど効いていない領域**しか
# 検証していないことになる（陽解法版と数値的に区別がつかない）。
# 大きい `α_max` を使うなら、`D` が 1 から大きく外れる領域でも
# 随伴が合うことを確認しておくこと（`scripts/03_optimize.jl` の設計場は
# `α_max` をこの何桁も上に取る）。
if !@isdefined(THETA_SCALE)
    THETA_SCALE = 3.0
end
theta0 = T(THETA_SCALE) .* rand(rng, T, p.ngy, p.ngx)
nsteps = 25

@printf("θ ∈ [%.3g, %.3g],  dt·α ∈ [%.3g, %.3g],  D = 1/(1+dt·α) ∈ [%.4g, %.4g]\n",
        minimum(theta0), maximum(theta0),
        p.dt * minimum(theta0), p.dt * maximum(theta0),
        1 / (1 + p.dt * maximum(theta0)), 1 / (1 + p.dt * minimum(theta0)))

# ## 目的関数
#
# 終端状態のランダム線形汎関数 `J = Σ cX·X_T + Σ cV·V_T` にする。
# こうすると終端随伴の種が `cX`, `cV` そのものになり、しかも全成分に
# 感度が乗るので一度の逆行で全部を検査できる。

cX = randn(rng, T, 2, N)
cV = randn(rng, T, 2, N)

# 勾配検証は近傍リストの再利用を使わない（`interval = 1` = 毎ステップ組み直し）。
# 再利用が結果を変えないことは test/runtests.jl の専用 testset で検証しており、
# ここで見たいのは随伴そのものの正しさだから。
#
# なおこの設定は壁ペナルティが強く 1 ステップの変位が大きいため、既定の
# `interval = 8` では変位の上限（skin·h）を破る。破ると前進と逆行で欠落する
# 近傍が食い違い、勾配が 1e-2 程度ずれる（そのときは警告が出る）。
function run_forward(X0, V0, theta; want_tape = false)
    st = State(backend, X0, V0, p; interval = 1)
    th = KernelAbstractions.allocate(backend, T, p.ngy, p.ngx)
    copyto!(th, theta)
    tape = want_tape ? Tape(backend, N, nsteps, p) : nothing
    simulate!(st, th, p, backend, nsteps; tape)
    J = sum(Array(st.X) .* cX) + sum(Array(st.V) .* cV)
    return J, st, th, tape
end

@printf("N = %d 粒子, %d ステップ, 設計変数 %d 個\n\n", N, nsteps, length(theta0))

# ## 随伴で勾配を出す

J, st, th, tape = run_forward(X0, V0, theta0; want_tape = true)
ws = AdjointWorkspace(backend, N, p)
backward!(ws, tape, th, p, backend; seedX = cX, seedV = cV, interval = 1)

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

# θ 側だけ相対ステップにする。`THETA_SCALE` を上げたときに絶対 1e-6 のままだと
# 摂動が θ の丸め（3e4 に対して相対 3e-11）に埋もれ、差分側が壊れる。
hfd_theta = hfd * max(1.0, THETA_SCALE)

println("dJ/dtheta  (中心差分と比較)")
for idx in [CartesianIndex(1, 1), CartesianIndex(4, 5), CartesianIndex(6, 3)]
    g = fd(t -> run_forward(X0, V0, t)[1], theta0, idx, hfd_theta)
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
# ### `THETA_SCALE` を上げたときの `dJ/dV0`
#
# 抗力が強いと初期速度の影響が終端までにほとんど減衰するので、`dJ/dV0` の
# **絶対値そのものが小さくなる**（`THETA_SCALE = 3e4` で 1e-5 台）。
# すると差分商 `(J⁺-J⁻)/2h` の丸め（`ε·|J|/2h ≈ 5e-10`）が相対誤差として
# 効いてきて、`hfd = 1e-6` のままだと 1e-6 台に見える。随伴の誤りではなく
# **差分側の限界**で、実測は次のとおり（`THETA_SCALE = 3e4`）：
#
# | `hfd` | 1e-6 | 1e-5 | 1e-4 | 1e-3 |
# |---|---:|---:|---:|---:|
# | `V0[1,5]` | 2.65e-06 | 5.69e-07 | 5.52e-08 | 3.14e-09 |
# | `V0[2,30]` | 1.08e-06 | 1.08e-06 | 1.23e-07 | 2.13e-08 |
#
# 疑わしいときは `hfd` を振ってみること。**単調に下がるなら差分側の丸め、
# 下げ止まるなら随伴を疑う。**
#
# 参考として、同じ定式化を NumPy に書いて PyTorch の自動微分と比較した結果は
# `dJ/dX0` 2.6e-15, `dJ/dV0` 1.4e-15, `dJ/dθ` 1.0e-15
# （`tools/verify_adjoint.py` および `tools/verify_gather.py`）。
# つまり **式そのものは検証済み** なので、ここで合わなければ疑うべきは
# Julia / KernelAbstractions の作法まわり（README の「初回のチェックポイント」）。

println("\n相対誤差が 1e-6 前後なら OK。1e-2 を超えるなら README の")
println("「初回のチェックポイント」を確認してください。")
