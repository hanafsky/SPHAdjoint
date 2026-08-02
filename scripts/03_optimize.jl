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

# 近傍リストの再構築間隔。**既定の 4 はこの設定では長すぎる。**
# 実測で 1 ステップの最大変位 6.96e-4 に対し許容は `skin·h` = 2.08e-3 なので
# K ≤ 2。既定のままだと近傍が欠落して勾配がずれ、直線探索が受理しなくなる
# （`|v|max = 3.28`、`c = 15` に対し `v/c = 0.22` で、既定が想定する `c/10`
# より速いのが原因）。前進と逆行の両方に同じ値を渡すこと。
if !@isdefined(NL_INTERVAL)
    NL_INTERVAL = 2
end

# ---- 設計変数 -------------------------------------------------------------
#
# **設計変数まわりは粒子側の精度（Metal では Float32）と切り離して常に Float64
# で持つ。** 設計変数は 561 個で CPU 上の処理なのでコストは無視できる一方、
# `α(ρ)` は `α_max` を掛けるので ρ の 1e-7 の丸めが `α_max·1e-7` に化ける。
# `α_max = 2e5` では全部流体（`rd ≡ 1`）ですら `α` が負に振れ、
# 速度を増幅する「負の抗力」になっていた（`apply_filter` のコメント参照）。
const TW = Float64
#
# rd ∈ [0,1]:  rd=1 → 流体（α=0）、rd=0 → 固体（α=α_max）
# Borrvall–Petersson の凸補間:
#   α(ρ) = α_max · qp (1-ρ) / (qp + ρ)
#
# `α_max` は外から `ALPHA_MAX` で差し替えられる。既定の 2000 は
# **抗力が陽解法だった頃の値**で、`dt < 2/α_max` の制限（この設定では
# `α_max < 1.3e4`）に収まるよう選んである。抗力を陰的にした（#4）今は
# この制限が無い——が、**上げても固体境界はシャープにならない**。
# むしろ薄くなる（`α_max` が大きいと ρ を 1e-3 下げるだけで必要な抗力が
# 買えるので、設計を 0/1 に分離する動機が消える）。到達する J も変わらず、
# 変わるのは収束の速さだけ（2e5 で 5.7 倍）。なお 2e6 まで上げると既定の
# `STEP0 = 0.05` では初手が受理されない。上げるなら初期幅も一緒に下げること。
# 既定を 2000 のままにしてあるのは
# **この設定が一番設計場が見える**から。実測は README の
# 「α_max を上げても設計はシャープにならない」を参照。
if !@isdefined(ALPHA_MAX)
    ALPHA_MAX = 2.0e3
end
alpha_max = TW(ALPHA_MAX)
#
# Borrvall–Petersson の罰則 `qp`。**q が大きいほど α(ρ) は線形に近く、
# 中間密度でも α が大きい**（q → ∞ で α = α_max(1-ρ)）。
#
# **真の固体（ρ=0）が欲しければ q を下げる**（実測は README の
# 「qp を下げると真の固体が出る」）。q を上げる方向は α_max を上げるのと
# 同型の機構で設計が薄くなる（中間密度で買える抗力が増え、浅掘りで足りる）。
# q=0.01 で初めて ρ=0 のセルが現れ、J も 4 ポイント良くなる。0.003 は収穫なし。
# FEM トポ最適の「中間密度を高価にすれば 0/1 に行く」はこの問題では成立しない
# （体積制約が下限で不活性 = 固体を使わされる圧力が無いから）。
# `QP` と `RD0` を外から与えれば continuation も組めるが、この問題サイズでは
# q=0.01 直行と大差なかった（J で 0.3%）。
if !@isdefined(QP)
    QP = 0.1
end
qp = TW(QP)
if !@isdefined(VOLFRAC)
    VOLFRAC = 0.75
end
volfrac = TW(VOLFRAC)          # 流体として残す体積割合の下限
filt_r = 2                     # 密度フィルタ半径（格子点数）

alpha_of(rd) = alpha_max * qp * (1 - rd) / (qp + rd)
dalpha_of(rd) = -alpha_max * qp * (qp + 1) / (qp + rd)^2

"""線形重みの密度フィルタ（FEM のトポ最適と同じもの）。

重みも `TW`（Float64）で持つ。`Float32` で正規化すると重みの和が 1 から 1e-7
ずれ、全部流体（`rd ≡ 1`）でも `rdf` が 1 をまたぐ。
"""
function build_filter(ngy, ngx, r)
    W = Dict{Tuple{Int,Int},Vector{Tuple{Int,Int,TW}}}()
    for j in 1:ngy, i in 1:ngx
        lst = Tuple{Int,Int,TW}[]
        s = zero(TW)
        for jj in max(1, j - r):min(ngy, j + r), ii in max(1, i - r):min(ngx, i + r)
            w = TW(r + 1) - sqrt(TW((ii - i)^2 + (jj - j)^2))
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
    y = zeros(TW, size(x))
    for j in axes(x, 1), i in axes(x, 2)
        acc = zero(TW)
        for (jj, ii, w) in W[(j, i)]
            acc += w * TW(x[jj, ii])
        end
        # 重みは正規化してあるので数学的には凸結合＝ [0,1] に入る。それでも
        # clamp するのは、**ここを外すと `α = α_max·q(1-ρ)/(q+ρ)` が負になる**
        # ——速度を増幅する「負の抗力」になり、しかも大きさが `α_max` に比例する
        # ——から。Float32 で重みを持っていた頃は全部流体（`rd ≡ 1`）でも
        # 561 セル中 87 セルで `rdf > 1` になり、5333 ステップで α_max=2e5 なら
        # 0.19% のエネルギー注入になっていた（出発点の J が α_max 依存でずれ、
        # 最適化が 1 歩も進まなくなる）。Float64 にした今は保険。
        y[j, i] = clamp(acc, zero(TW), one(TW))
    end
    y
end

# フィルタは対称なので転置も同じ重みで書ける
filter_adjoint(W, g) = begin
    out = zeros(TW, size(g))
    for j in axes(g, 1), i in axes(g, 2)
        for (jj, ii, w) in W[(j, i)]
            out[jj, ii] += w * TW(g[j, i])
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
    x = clamp.(rd, zero(TW), one(TW))
    sum(x) / length(x) >= volfrac && return x
    lo, hi = zero(TW), one(TW)      # 全部 1 にすれば必ず満たせるので上限は 1
    for _ in 1:60
        mid = (lo + hi) / 2
        if sum(clamp.(x .+ mid, zero(TW), one(TW))) / length(x) < volfrac
            lo = mid
        else
            hi = mid
        end
    end
    return clamp.(x .+ (lo + hi) / 2, zero(TW), one(TW))
end

# ---- 目的関数と勾配 -------------------------------------------------------
th_dev = KernelAbstractions.allocate(backend, T, p.ngy, p.ngx)
ws = AdjointWorkspace(backend, N, p)
# テープは反復ごとに作り直さず、一度だけ確保して使い回す。
# GPU では毎ステップの確保がそのまま効くので、ここが効率の要。
tape = Tape(backend, N, nsteps, p)

function objective_and_grad(rd)
    rdf = apply_filter(W, rd)
    copyto!(th_dev, T.(alpha_of.(rdf)))   # 設計側は Float64、デバイス側は T

    st = State(backend, X0, V0, p; interval = NL_INTERVAL)
    reset!(tape)
    simulate!(st, th_dev, p, backend, nsteps; tape)

    J, seed = target_objective(st.X, target[1], target[2], sigma)
    backward!(ws, tape, th_dev, p, backend; seedX = seed, seedV = zero(seed),
              interval = NL_INTERVAL)

    gtheta = Array(ws.gtheta)
    grdf = gtheta .* dalpha_of.(rdf)         # α(ρ) の連鎖律
    grd = filter_adjoint(W, grdf)            # フィルタの随伴
    return J, grd
end

"""目的関数だけ（前進のみ）。直線探索で候補を試すのに使う。"""
function objective_only(rd)
    rdf = apply_filter(W, rd)
    copyto!(th_dev, T.(alpha_of.(rdf)))
    st = State(backend, X0, V0, p; interval = NL_INTERVAL)
    simulate!(st, th_dev, p, backend, nsteps)
    J, _ = target_objective(st.X, target[1], target[2], sigma)
    return J
end

# ---- 最適化ループ（射影付き勾配法） ---------------------------------------
#
# 初期設計は外から `RD0` で差し替えられる。continuation（qp を段階的に上げ、
# 前段の解を次段の初期値にする）を外側のループで組むためのもの：
#
#     for q in (0.1, 1.0, 10.0)
#         global QP = q
#         include("scripts/03_optimize.jl")
#         global RD0 = rd            # 次段は今段の解から出発
#     end
if @isdefined(RD0)
    rd = copy(RD0)::Matrix{TW}
else
    rd = fill(TW(1.0), p.ngy, p.ngx)   # 全部流体から出発
end
Jhist = T[]

@printf("N = %d 粒子, %d ステップ (%.2f 秒), 設計変数 %d 個\n",
        N, nsteps, nsteps * p.dt, length(rd))

# 出発点の健全性チェック。目標に水が全く届いていない設定だと勾配が意味を
# 持たないので、黙って回さずここで気付けるようにしておく。
let stc = State(backend, X0, V0, p; interval = NL_INTERVAL),
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

t_start = time()
# 直線探索の初期ステップ幅。`α_max` を大きく取ると 1 反復目から
# 「どの向きに動かしても悪化する」ことがあり、そのときは初期幅の問題なのか
# 出発点（全部流体）が局所最適なのかを切り分けたくなるので外から縮められる。
if !@isdefined(STEP0)
    STEP0 = 0.05
end
step_size = TW(STEP0)
J, g = objective_and_grad(rd)
push!(Jhist, J)
@printf("%4d  %+.6e  %.3e  %.4f  (初期)\n", 0, J, maximum(abs, g), step_size)

# 反復上限。**条件を比較するときは「上限で打ち切られた」のか「直線探索が
# 収束した」のかを必ず見分けること。** 上限に張り付いたまま降下し続けている
# 条件と、収束した条件の J を並べると「まだ伸びしろがある方が悪い」に見える。
if !@isdefined(MAXIT)
    MAXIT = 40
end

for it in 1:MAXIT
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
            step_size = min(step_size * TW(1.3), TW(0.30))
            accepted = true
            break
        end
        step_size *= TW(0.5)            # 改善しなければ刻みを半分に
    end
    if !accepted
        @printf("収束（ステップ %.2e まで縮めても改善せず）\n", step_size)
        break
    end

    push!(Jhist, J)
    _, g = objective_and_grad(rd)
    @printf("%4d  %+.6e  %.3e  %.4f  %d 回\n", it, J, maximum(abs, g), step_size, ntry)
end

t_elapsed = time() - t_start

# ---- 設計場の離散性 -------------------------------------------------------
#
# 「固体境界がシャープか」を数字で見る。Sigmund の非離散性測度
#
#     Mnd = (1/n) Σ 4 ρ (1-ρ) × 100 [%]
#
# は ρ が 0/1 に張り付くと 0、全部 0.5 だと 100 になる。
#
# **フィルタ前 `rd` とフィルタ後 `rdf` の両方を出すこと。** `filt_r = 2` の
# 線形フィルタは境界に必ず中間値を作るので、`rdf` の Mnd はフィルタの半径に
# 支配されて `α_max` を変えてもあまり動かない。設計そのものが 0/1 に寄ったか
# は `rd` 側に出る。
#
# `volfrac` は**下限**制約なので、`mean(rd)` がこれに張り付いていなければ
# 「固体の予算を使い切っていない」＝ 固体にする動機が足りていない、と読める。
mnd(x) = 100 * sum(4 .* x .* (1 .- x)) / length(x)

let rdf = apply_filter(W, rd)
    println("\n---- 設計場 ----")
    @printf("α_max = %.3g   qp = %.3g   dt·α(ρ=%.2f) = %.3g   dt·α(ρ=0) = %.3g\n",
            alpha_max, qp, volfrac, p.dt * alpha_of(volfrac), p.dt * alpha_of(zero(TW)))
    @printf("Mnd(rd)  = %6.2f %%   min %.4f   ρ<0.1 のセル %d / %d\n",
            mnd(rd), minimum(rd), count(<(TW(0.1)), rd), length(rd))
    @printf("Mnd(rdf) = %6.2f %%   min %.4f   ρ<0.1 のセル %d / %d\n",
            mnd(rdf), minimum(rdf), count(<(TW(0.1)), rdf), length(rdf))
    @printf("mean(rd) = %.4f （体積制約の下限 %.2f）\n", sum(rd) / length(rd), volfrac)
    @printf("J: %+.6e → %+.6e （%.2f%% 改善）   %d 反復   %.1f 秒\n",
            Jhist[1], Jhist[end], 100 * (Jhist[1] - Jhist[end]) / abs(Jhist[1]),
            length(Jhist) - 1, t_elapsed)
end

# ---- 出力 -----------------------------------------------------------------
# 条件を変えて回すときに上書きし合わないよう、`OUT_SUFFIX` で名前を分けられる。
if !@isdefined(OUT_SUFFIX)
    OUT_SUFFIX = ""
end
open("design_field$(OUT_SUFFIX).csv", "w") do io
    println(io, "j,i,rho_design,rho_filtered,alpha")
    rdf = apply_filter(W, rd)
    for j in 1:p.ngy, i in 1:p.ngx
        println(io, j, ",", i, ",", rd[j, i], ",", rdf[j, i], ",", alpha_of(rdf[j, i]))
    end
end
open("objective_history$(OUT_SUFFIX).csv", "w") do io
    println(io, "iter,J")
    for (i, J) in enumerate(Jhist)
        println(io, i, ",", J)
    end
end
println("design_field$(OUT_SUFFIX).csv / objective_history$(OUT_SUFFIX).csv を書き出しました")

# ---- 可視化（GLMakie があれば） -------------------------------------------
# using GLMakie
# rdf = apply_filter(W, rd)
# fig = Figure(size=(900, 400))
# ax = Axis(fig[1,1], aspect=DataAspect(), title="設計場 ρ (1=流体, 0=固体)")
# heatmap!(ax, range(0, p.Lx, p.ngx), range(0, p.Ly, p.ngy), permutedims(rdf),
#          colormap=:grays, colorrange=(0,1))
# display(fig)
