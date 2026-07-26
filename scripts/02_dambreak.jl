# # 前進のみ：2D ダムブレイク
#
# 勾配は使わず、まず流れが妥当に見えるか・どれだけ速いかを確認する。
# `01_gradcheck.jl` が通ったら次はこれ。
#
# ```
# julia --project=. scripts/02_dambreak.jl
# ```
#
# ## Metal で走らせる
#
# 冒頭の 3 行を差し替えるだけでよい。**カーネルは一行も変えない。**
#
# ```julia
# using Metal
# const T = Float32          # Apple GPU は Float64 を持たない
# backend = MetalBackend()
# ```
#
# ここで出る「ステップ/秒」が、あとで `04_interactive.jl` の `nsub`
# （1 フレームあたりの物理ステップ数）を決める根拠になる。

using SPHAdjoint
using KernelAbstractions
using Printf

const T = Float64
backend = CPU()

# ---- パラメータ -----------------------------------------------------------
dp = 0.012                      # 初期粒子間隔
p = SPHParams{T}(
    h    = 1.3 * dp,            # 平滑化長（サポート半径 2h）
    m    = dp^2 * 1.0,          # 質量 = 面積 × 参照密度
    rho0 = 1.0,
    c    = 20.0,                # 数値音速: 最大流速の 10 倍程度が目安
    mu   = 0.05,
    dt   = 1.5e-4,
    Lx   = 1.0,
    Ly   = 0.6,
    kw   = 5.0e4,
    ngx  = 33,
    ngy  = 21,
)

# ---- 初期配置: 左側の水柱 -------------------------------------------------
col_w, col_h = 0.30, 0.45
xs = dp:dp:col_w
ys = dp:dp:col_h
X0 = zeros(T, 2, length(xs) * length(ys))
k = 0
for y in ys, x in xs
    global k += 1
    X0[1, k] = x
    X0[2, k] = y
end
V0 = zeros(T, size(X0))
N = size(X0, 2)
@printf("粒子数 N = %d\n", N)

# 設計変数場は今回は全部 0（障害物なし）
theta = KernelAbstractions.zeros(backend, T, p.ngy, p.ngx)

st = State(backend, X0, V0, p)

# ---- 時間発展 -------------------------------------------------------------
nsteps = 6000
save_every = 50
frames = Matrix{T}[]

t0 = time()
for n in 1:nsteps
    step!(st, theta, p, backend)
    if n % save_every == 0
        push!(frames, Array(st.X))
    end
end
@printf("%d ステップ / %.2f 秒  (%.1f ステップ/秒)\n",
        nsteps, time() - t0, nsteps / (time() - t0))

# ---- 出力 -----------------------------------------------------------------
open("dambreak_frames.csv", "w") do io
    println(io, "frame,id,x,y")
    for (fi, F) in enumerate(frames), i in 1:N
        println(io, fi, ",", i, ",", F[1, i], ",", F[2, i])
    end
end
println("dambreak_frames.csv を書き出しました（", length(frames), " フレーム）")

# ---- 可視化（GLMakie があれば） -------------------------------------------
# using GLMakie
# pos = Observable(Point2f.(frames[1][1,:], frames[1][2,:]))
# fig = Figure(size=(900, 560))
# ax = Axis(fig[1,1], aspect=DataAspect(), limits=(0, p.Lx, 0, p.Ly))
# scatter!(ax, pos, markersize=6, color=:dodgerblue)
# display(fig)
# for F in frames
#     pos[] = Point2f.(F[1,:], F[2,:])
#     sleep(1/60)
# end
