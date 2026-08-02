# # 設計変数勾配の検証：03 の連鎖律 vs 中心差分
#
# `01_gradcheck.jl` が検証するのは **θ（格子上の α 場）に対する勾配**まで。
# `03_optimize.jl` はその先に
#
# ```
# rd → [密度フィルタ] → rdf → [α(ρ) = α_max·q(1-ρ)/(q+ρ)] → θ
# gθ → [dα/dρ の連鎖律] → grdf → [フィルタの随伴] → grd
# ```
#
# というスクリプト側の連鎖律を持っていて、ここは 01 では検証されない。
# `qp` を大きくすると `dα/dρ` の非線形性が強くなる（#24 の continuation で
# まさにそこを使う）ので、**設計変数 rd に対する勾配**を中心差分と直接
# 突き合わせる。
#
# ```
# julia --project=. scripts/10_design_gradcheck.jl
# ```
#
# ## 検証点の置き方（2 つ条件がある）
#
# 1. **内点に置く。** 既定の出発点（全部流体 `rd ≡ 1`）では `apply_filter` の
#    clamp が境界で片側にしか動けず、中心差分が原理的に合わない。
# 2. **水が目標に届く設計で検証する。** 最初 `rd ≈ 0.6` の一様乱数で試したら
#    抗力が強すぎて水が届かず、`J = -1.5e-07`（指数関数の裾）・勾配 3.5e-9 の
#    ノイズ同士を比較していた（3 点中 2 点は完全にゼロ）。それでは合っていても
#    意味がない。1 に近い内点（0.85〜0.99）なら流れが保たれ、J に信号が乗る。

using Random

# 03 を「最適化を回さず」読み込む: MAXIT = 0 なら直線探索ループが回らないので、
# 関数定義と初期評価だけが手に入る。
USE_METAL = false                 # 中心差分は Float64 で
MAXIT = 0
STEP0 = 0.05
if !@isdefined(QP)
    QP = 10.0                     # 非線形性が強い側で検証する
end
# α_max は検証用に落とす。**連鎖律の形は α_max に依存しない**ので検証としては
# 等価で、本番の 2000 のままだと「J に信号がある内点」がほぼ存在しない
# （qp=10 では rd=0.92 でも v_term = g/α ≈ 0.2 m/s で水が届かない。実測で
# J = -2e-07 まで潰れた）。100 なら rd≈0.9 で v_term ≈ 1 m/s、流れが減衰
# しつつ届くレンジに入る。
if !@isdefined(ALPHA_MAX)
    ALPHA_MAX = 100.0
end
OUT_SUFFIX = "_designgradcheck"

let rng = MersenneTwister(7)
    # 03 が読む格子サイズと合わせる（ngy=17, ngx=33 は 03 の SPHParams と同じ）
    global RD0 = 0.90 .+ 0.10 .* (rand(rng, 17, 33) .- 0.5)
end

include("03_optimize.jl")

using Printf

# ## 随伴勾配と中心差分

Jc, gc = objective_and_grad(rd)
@printf("\nqp = %.3g   J = %+.8e   |grad|max = %.3e\n", qp, Jc, maximum(abs, gc))
# 裾ノイズ（水が全く届かない設計では J ~ -1e-7・勾配 ~1e-8 になり、ノイズ同士の
# 比較になる）を弾く。J が 1e-3 もあれば FD の S/N は十分。
abs(Jc) < 1e-3 && error("J に信号が無い（|J| = $(abs(Jc))）。水が目標に届いていないので検証にならない")

# 検証点は「勾配が大きい上位 3 成分」を選ぶ。固定添字だと粒子の通り道の外
# （勾配が恒等的にゼロ）を引いて、0 と 0 を比較するだけになる。
idxs = partialsortperm(vec(abs.(gc)), 1:3; rev = true)

hfd = 1e-5
println("\ndJ/drd  (中心差分と比較、|grad| 上位 3 成分)")
worst = 0.0
for li in idxs
    idx = CartesianIndices(gc)[li]
    xp = copy(rd); xp[idx] += hfd
    xm = copy(rd); xm[idx] -= hfd
    gfd = (objective_only(xp) - objective_only(xm)) / (2hfd)
    rel = abs(gc[idx] - gfd) / max(abs(gfd), 1e-30)
    global worst = max(worst, rel)
    @printf("  rd[%2d,%2d]: 随伴 % .9e   FD % .9e   rel %.2e\n",
            idx[1], idx[2], gc[idx], gfd, rel)
end

println(worst < 1e-5 ? "\nOK（1e-5 以下）" :
        "\nNG: 相対誤差が 1e-5 を超えた。dalpha_of かフィルタ随伴を疑うこと")
