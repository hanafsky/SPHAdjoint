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

const T = Float64
backend = CPU()

dp = 0.020
p = SPHParams{T}(
    h = 1.3 * dp, m = dp^2 * 1.0, rho0 = 1.0, c = 15.0, mu = 0.05,
    dt = 2.0e-4, Lx = 1.0, Ly = 0.5, kw = 5.0e4, ngx = 33, ngy = 17,
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

nsteps = 1500
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

"""体積制約 mean(rd) >= volfrac を満たすように [0,1] へ射影（二分法）。"""
function project_volume(rd, volfrac)
    lo, hi = -1.0, 1.0
    for _ in 1:60
        mid = (lo + hi) / 2
        if sum(clamp.(rd .+ mid, 0, 1)) / length(rd) < volfrac
            lo = mid
        else
            hi = mid
        end
    end
    return clamp.(rd .+ (lo + hi) / 2, 0, 1)
end

# ---- 目的関数と勾配 -------------------------------------------------------
th_dev = KernelAbstractions.allocate(backend, T, p.ngy, p.ngx)
ws = AdjointWorkspace(backend, N, p)

function objective_and_grad(rd)
    rdf = apply_filter(W, rd)
    theta = alpha_of.(rdf)
    copyto!(th_dev, theta)

    st = State(backend, X0, V0, p)
    tape = Tuple{typeof(st.X),typeof(st.V)}[]
    simulate!(st, th_dev, p, backend, nsteps; tape)

    J, seed = target_objective(st.X, target[1], target[2], sigma)
    backward!(ws, tape, th_dev, p, backend; seedX = seed, seedV = zero(seed))

    gtheta = Array(ws.gtheta)
    grdf = gtheta .* dalpha_of.(rdf)         # α(ρ) の連鎖律
    grd = filter_adjoint(W, grdf)            # フィルタの随伴
    return J, grd
end

# ---- 最適化ループ（射影付き勾配法） ---------------------------------------
rd = fill(T(1.0), p.ngy, p.ngx)   # 全部流体から出発
Jhist = T[]

@printf("N = %d 粒子, %d ステップ, 設計変数 %d 個\n", N, nsteps, length(rd))
println("iter        J        |grad|      step")

step_size = T(0.25)
for it in 1:40
    global rd
    J, g = objective_and_grad(rd)
    push!(Jhist, J)
    gn = maximum(abs, g)
    gn < 1e-30 && break
    rd = project_volume(rd .- step_size .* g ./ gn, volfrac)
    @printf("%4d  %+.6e  %.3e  %.3f\n", it, J, gn, step_size)
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
