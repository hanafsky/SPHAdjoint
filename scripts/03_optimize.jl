# # 流体トポロジー最適化デモ
#
# 背景ボクセル上の Brinkman 摩擦場を設計変数にして、ダムブレイクの水を
# できるだけ目標領域に導く障害物配置を求める。
#
# ```
# julia --project=. scripts/03_optimize.jl
# ```
#
# ## 考え方
#
# NBPH (Liu et al., Struct Multidisc Optim 2023) と同じ発想。
# 粒子は Lagrangian に動くが、**設計変数は固定 Eulerian 格子に置く**。
# こうすると FEM のトポ最適で使う道具がそのまま流用できる：
#
# - Borrvall–Petersson の凸補間 `α(ρ) = α_max·q(1-ρ)/(q+ρ)`
# - 線形重みの密度フィルタ（とその随伴）
# - 体積制約の二分法射影
#
# ## 注意
#
# CPU だと 1 反復に 1500 ステップの前進＋逆行が入るので、それなりに待つ。
# まず `nsteps` を 300 くらいに落として通ることを確認するのがおすすめ。

using SPHAdjoint
using KernelAbstractions
using Printf

# Metal で走らせるには、include する前に `const USE_METAL = true` を定義するか、
# 下のデフォルトを true にする。**カーネルは一行も変えない。**
#
# 粒子数が小さいと GPU はカーネル起動レイテンシ律速で CPU に負ける
# （`05_metal_bench.jl` の実測では N=216 で CPU の 0.5 倍、N≳900 で逆転）。
# なので Metal 側は解像度を上げた設定にしてある。
if !@isdefined(USE_METAL)
    const USE_METAL = false
end
if USE_METAL
    using Metal
end

const T = USE_METAL ? Float32 : Float64
backend = USE_METAL ? MetalBackend() : CPU()

dp = USE_METAL ? 0.008 : 0.020          # N = 1500 / 240
# 音速 CFL: dt < 0.25 h/c
dt = USE_METAL ? 1.5e-4 : 2.0e-4
p = SPHParams{T}(
    h = 1.3 * dp, m = dp^2 * 1.0, rho0 = 1.0, c = 15.0, mu = 0.05,
    dt = dt, Lx = 1.0, Ly = 0.5, kw = 5.0e4, ngx = 33, ngy = 17,
)

# ---- 初期配置 -------------------------------------------------------------
xs = dp:dp:0.24
ys = dp:dp:0.40
X0 = zeros(T, 2, length(xs) * length(ys))
k = 0
for y in ys, x in xs
    global k += 1
    X0[1, k] = x
    X0[2, k] = y
end
V0 = zeros(T, size(X0))
N = size(X0, 2)

# nsteps は「水が目標領域に届く時間」で決めること。届かないうちに打ち切ると
# J が指数関数の裾（先頭 1 粒子ぶん）になり、勾配が実質ノイズになって最適化が
# 動かない。全部流体の設計で測ったところ：
#
#   1500 步 (0.3s) → 2σ 内 0 個,   J = -1.4e-02   ← 信号なし
#   3000 步 (0.6s) → 2σ 内 61 個,  J = -2.8e+01
#   4000 步 (0.8s) → 2σ 内 101 個, J = -4.9e+01
#   5500 步 (1.1s) → 2σ 内 121 個, J = -6.1e+01   ← ピーク
#   7000 步 (1.4s) → 2σ 内 108 個, J = -5.2e+01   ← 跳ね返って悪化
#
# ピーク以降は水が壁で跳ね返って J が減るので、その手前を取る。
# 解像度を変えても同じ物理時間を見るよう、ステップ数は t_end から決める。
t_end = 0.80                   # 秒
nsteps = round(Int, t_end / p.dt)
target = (0.85, 0.10)          # 目標領域の中心
sigma = 0.10

# ---- 設計変数 -------------------------------------------------------------
# rd ∈ [0,1]:  rd=1 → 流体（α=0）、rd=0 → 固体（α=α_max）
# Borrvall–Petersson の凸補間:
#   α(ρ) = α_max · qp (1-ρ) / (qp + ρ)
alpha_max = T(2000.0)
qp = T(0.1)
volfrac = T(0.75)              # 流体として残す体積割合の下限
filt_r = 2                     # 密度フィルタ半径（格子点数）

alpha_of(rd) = alpha_max * qp * (1 - rd) / (qp + rd)
dalpha_of(rd) = -alpha_max * qp * (qp + 1) / (qp + rd)^2

"""線形重みの密度フィルタ（FEM のトポ最適と同じもの）。"""
function build_filter(ngy, ngx, r)
    W = Dict{Tuple{Int,Int},Vector{Tuple{Int,Int,T}}}()
    for j in 1:ngy, i in 1:ngx
        lst = Tuple{Int,Int,T}[]
        s = zero(T)
        for jj in max(1, j - r):min(ngy, j + r), ii in max(1, i - r):min(ngx, i + r)
            w = T(r + 1) - sqrt(T((ii - i)^2 + (jj - j)^2))
            if w > 0
                push!(lst, (jj, ii, w))
                s += w
            end
        end
        W[(j, i)] = [(jj, ii, w / s) for (jj, ii, w) in lst]
    end
    return W
end

apply_filter(W, x) = begin
    y = similar(x)
    for j in axes(x, 1), i in axes(x, 2)
        acc = zero(T)
        for (jj, ii, w) in W[(j, i)]
            acc += w * x[jj, ii]
        end
        y[j, i] = acc
    end
    y
end

# フィルタは対称なので転置も同じ重みで書ける
filter_adjoint(W, g) = begin
    out = zeros(T, size(g))
    for j in axes(g, 1), i in axes(g, 2)
        for (jj, ii, w) in W[(j, i)]
            out[jj, ii] += w * g[j, i]
        end
    end
    out
end

W = build_filter(p.ngy, p.ngx, filt_r)

"""体積制約 mean(rd) >= volfrac を満たすように [0,1] へ射影（二分法）。

制約は**下限**なので、すでに満たしていれば何もしない。ここを等式射影
（常に `mean == volfrac` にする）にすると、全部流体の出発点から
**ステップ幅に関係なく**毎回 25% を固体に変える巨大な一歩が入り、
最適化が一切動かなくなる（実際そうなっていた）。
"""
function project_volume(rd, volfrac)
    x = clamp.(rd, zero(T), one(T))
    sum(x) / length(x) >= volfrac && return x
    lo, hi = zero(T), one(T)        # 全部 1 にすれば必ず満たせるので上限は 1
    for _ in 1:60
        mid = (lo + hi) / 2
        if sum(clamp.(x .+ mid, zero(T), one(T))) / length(x) < volfrac
            lo = mid
        else
            hi = mid
        end
    end
    return clamp.(x .+ (lo + hi) / 2, zero(T), one(T))
end

# ---- 目的関数と勾配 -------------------------------------------------------
th_dev = KernelAbstractions.allocate(backend, T, p.ngy, p.ngx)
ws = AdjointWorkspace(backend, N, p)
# テープは反復ごとに作り直さず、一度だけ確保して使い回す。
# GPU では毎ステップの確保がそのまま効くので、ここが効率の要。
tape = Tape(backend, N, nsteps, p)

function objective_and_grad(rd)
    rdf = apply_filter(W, rd)
    theta = alpha_of.(rdf)
    copyto!(th_dev, theta)

    st = State(backend, X0, V0, p)
    reset!(tape)
    simulate!(st, th_dev, p, backend, nsteps; tape)

    J, seed = target_objective(st.X, target[1], target[2], sigma)
    backward!(ws, tape, th_dev, p, backend; seedX = seed, seedV = zero(seed))

    gtheta = Array(ws.gtheta)
    grdf = gtheta .* dalpha_of.(rdf)         # α(ρ) の連鎖律
    grd = filter_adjoint(W, grdf)            # フィルタの随伴
    return J, grd
end

"""目的関数だけ（前進のみ）。直線探索で候補を試すのに使う。"""
function objective_only(rd)
    rdf = apply_filter(W, rd)
    copyto!(th_dev, alpha_of.(rdf))
    st = State(backend, X0, V0, p)
    simulate!(st, th_dev, p, backend, nsteps)
    J, _ = target_objective(st.X, target[1], target[2], sigma)
    return J
end

# ---- 最適化ループ（射影付き勾配法） ---------------------------------------
rd = fill(T(1.0), p.ngy, p.ngx)   # 全部流体から出発
Jhist = T[]

@printf("N = %d 粒子, %d ステップ (%.2f 秒), 設計変数 %d 個\n",
        N, nsteps, nsteps * p.dt, length(rd))

# 出発点の健全性チェック。目標に水が全く届いていない設定だと勾配が意味を
# 持たないので、黙って回さずここで気付けるようにしておく。
let stc = State(backend, X0, V0, p),
    thc = KernelAbstractions.zeros(backend, T, p.ngy, p.ngx)
    simulate!(stc, thc, p, backend, nsteps)
    Xh = Array(stc.X)
    d = sqrt.((Xh[1, :] .- target[1]) .^ 2 .+ (Xh[2, :] .- target[2]) .^ 2)
    nhit = count(<(2 * sigma), d)
    @printf("初期設計（全部流体）: J = %+.4e, 目標の 2σ 内に %d / %d 粒子\n",
            -sum(exp.(-(d .^ 2) ./ (2 * sigma^2))), nhit, N)
    nhit == 0 && @warn "目標に水が届いていない。nsteps か target を見直すこと（勾配が裾のノイズになる）"
end

# 単純な「勾配方向へ固定幅」では動かない。設計変数が [0,1] しか無いので
# 0.25 も動かすと一発で流路を塞ぎ、J が 3 桁悪化したまま戻ってこない
# （実際にそうなった）。改善するまでステップを半分にするバックトラッキングを
# 入れ、**改善した候補しか採用しない**ようにする。候補の評価は前進のみで済む。
println("iter        J        |grad|      step   前進評価")

step_size = T(0.05)
J, g = objective_and_grad(rd)
push!(Jhist, J)
@printf("%4d  %+.6e  %.3e  %.4f  (初期)\n", 0, J, maximum(abs, g), step_size)

for it in 1:40
    global rd, J, g, step_size
    gn = maximum(abs, g)
    gn < 1e-30 && break

    accepted = false
    ntry = 0
    for _ in 1:6
        cand = project_volume(rd .- step_size .* g ./ gn, volfrac)
        ntry += 1
        Jc = objective_only(cand)
        if Jc < J                       # 改善した場合のみ採用（最小化）
            rd, J = cand, Jc
            step_size = min(step_size * T(1.3), T(0.30))
            accepted = true
            break
        end
        step_size *= T(0.5)             # 改善しなければ刻みを半分に
    end
    if !accepted
        @printf("収束（ステップ %.2e まで縮めても改善せず）\n", step_size)
        break
    end

    push!(Jhist, J)
    _, g = objective_and_grad(rd)
    @printf("%4d  %+.6e  %.3e  %.4f  %d 回\n", it, J, maximum(abs, g), step_size, ntry)
end

# ---- 出力 -----------------------------------------------------------------
open("design_field.csv", "w") do io
    println(io, "j,i,rho_design,alpha")
    rdf = apply_filter(W, rd)
    for j in 1:p.ngy, i in 1:p.ngx
        println(io, j, ",", i, ",", rdf[j, i], ",", alpha_of(rdf[j, i]))
    end
end
open("objective_history.csv", "w") do io
    println(io, "iter,J")
    for (i, J) in enumerate(Jhist)
        println(io, i, ",", J)
    end
end
println("\ndesign_field.csv / objective_history.csv を書き出しました")

# ---- 可視化（GLMakie があれば） -------------------------------------------
# using GLMakie
# rdf = apply_filter(W, rd)
# fig = Figure(size=(900, 400))
# ax = Axis(fig[1,1], aspect=DataAspect(), title="設計場 ρ (1=流体, 0=固体)")
# heatmap!(ax, range(0, p.Lx, p.ngx), range(0, p.Ly, p.ngy), permutedims(rdf),
#          colormap=:grays, colorrange=(0,1))
# display(fig)
