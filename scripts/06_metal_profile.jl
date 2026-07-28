# # Metal はどこで詰まっているか
#
# 近傍リストを実体化した（#1）あとの内訳。
#
# ```
# julia --project=. scripts/06_metal_profile.jl
# ```
#
# ## 素朴に並べても比較にならない
#
# 構築系（セルリスト + 近傍リスト）は `interval` ステップに 1 回しか走らない。
# 物理系（density / eos / accel / integrate）は毎ステップ走る。
# **同じ「1 回あたり μs」を並べると構築が過大に見える**ので、構築側は
# `1/interval` に償却してから足す。
#
# ## 積み上げが `step!` と合うのは N が大きいときだけ
#
# 同一セッション 3 回の実測：
#
# | | N=925 | N=15000 | N=60000 |
# |---|---:|---:|---:|
# | 積み上げ（鎖 + integrate + 構築/4） | 88〜119 | 173〜176 | 444〜462 |
# | `step!` 実測 | 76〜78 | 165〜167 | 473〜475 |
# | 差 | **-12〜-41** | -6〜-10 | **+13〜+31** |
#
# **N=60000 では `step!` のほうが 3〜7% 遅い。** 符号が正なのでカーネル起動の
# オーバーヘッドとして素直に読める。N=15000 でも差は 4〜6% で実用上一致。
#
# **N=925 では逆に積み上げのほうが 15〜53% 大きく、差の大きさも一定しない。**
# 小 N ではカーネルの実行時間そのものより起動レイテンシが大きいので、段階を
# 個別に測ると起動分が段階の数だけ乗る。**ただしこの説明は裏取りできていない**
# （そうであれば差は概ね一定のはずだが、-12〜-41 μs と 3 倍動く。単体測定の
# 絶対値自体が 1.5 倍ぶれるのでその中に埋もれている）。
#
# したがって：
#
# - **N ≥ 15000 なら**内訳を足して `step!` を説明してよい
# - **小 N の内訳は段階どうしの相対比較にだけ**使う。絶対値も信用しない
#   （N=925 の density は同じセッションで 21.3 μs と 31.7 μs の間を動く）
#
# ここで測るのは 3 つ。
#
# 1. **段階ごとの内訳（償却込み）** — 構築と物理のどちらが重いか
# 2. **density の実効帯域** — 近傍リストの**実測長**から読み出し量を出す
#    （セル走査時代のように候補数を見積もる必要はもう無い）
# 3. **粒子データをセル順に並べ替えると速くなるか** — #10 の判断材料

using SPHAdjoint
using KernelAbstractions
using Metal
using Printf

const T = Float32
const gpu = MetalBackend()

# M2 のメモリ帯域（公称）。実効帯域をこれで割って達成率を出す。
const PEAK_GBPS = 100.0

make_params(dp) = SPHParams{T}(
    h = 1.3 * dp, m = dp^2 * 1.0, rho0 = 1.0, c = 20.0, mu = 0.05,
    dt = 1.5e-4 * (dp / 0.012), Lx = 1.0, Ly = 0.6, kw = 5.0e4, ngx = 33, ngy = 21,
)

function water_column(dp; col_w = 0.30, col_h = 0.45)
    xs = dp:dp:col_w
    ys = dp:dp:col_h
    X0 = zeros(T, 2, length(xs) * length(ys))
    k = 0
    for y in ys, x in xs
        k += 1
        X0[1, k] = x
        X0[2, k] = y
    end
    return X0
end

"""カーネル 1 回あたりの時間。GPU は非同期なので reps 回まとめて撃って割る。

`warmup` は初回コンパイルを追い出すため（README の「性能を測るときのルール」2）。
`tries` 回まわして**最良値**を返す（同 3）。**単発だとぶれる**：この測定でも
「物理カーネル個別の和 / 鎖」の比が 0.95〜1.38x、並べ替えの効果が 1.00x〜1.43x
と実行ごとに動いた。最良値なら 5 回の測定で数 % に収まる。
"""
function time_stage(f, backend; reps = 48, warmup = 5, tries = 5)
    for _ in 1:warmup
        f()
    end
    KernelAbstractions.synchronize(backend)
    best = Inf
    for _ in 1:tries
        t0 = time()
        for _ in 1:reps
            f()
        end
        KernelAbstractions.synchronize(backend)
        best = min(best, (time() - t0) / reps)
    end
    return best
end

"""A と B を**交互に**測る。各試行の比を集め、`(A の最良, B の最良, 比の中央値)`。

README の「性能を測るときのルール」1。A をまとめて → B をまとめて、では
先に測ったほうが不利になる（キャッシュと GPU のクロック状態が違う）。
実際このスクリプトで、まったく同じ density の測定が先攻 59.2 μs / 後攻 37.3 μs
と 1.6 倍ずれたことがある。

**最良値どうしの比では足りない。** 測定区間まるごとが遅くなること（熱／クロックの
ドリフト）があり、そうなると 5 回の最良値も揃って遅い。実際 N=15000 の並べ替え
比較で、4 回の実行のうち 1 回だけ 1.54x が出た（他は 1.00〜1.03x）。
**各試行の中で A と B を続けて測り、その区間内の比を取る**（区間内なら条件が
揃っている）。それを試行数ぶん集めて中央値を採れば、遅い区間ごと弾ける。
"""
function time_ab(backend, fa, fb; reps = 48, warmup = 5, tries = 7)
    for _ in 1:warmup
        fa()
        fb()
    end
    KernelAbstractions.synchronize(backend)
    ba = Inf
    bb = Inf
    ratios = Float64[]
    for _ in 1:tries
        t0 = time()
        for _ in 1:reps
            fa()
        end
        KernelAbstractions.synchronize(backend)
        ta = (time() - t0) / reps

        t0 = time()
        for _ in 1:reps
            fb()
        end
        KernelAbstractions.synchronize(backend)
        tb = (time() - t0) / reps

        ba = min(ba, ta)
        bb = min(bb, tb)
        push!(ratios, ta / tb)
    end
    return ba, bb, sort(ratios)[(length(ratios)+1)÷2]
end

# ## 1. 段階ごとの内訳
#
# `step!` は `maybe_rebuild!`（`interval` に 1 回だけ実体が走る）+ 物理 4 本。
# 構築の中身は
#
# ```
# build!            : _cl_assign! → cumsum/starts/cursor → _cl_fill! → _cl_sort_cells!
# build_neighbors!  : build! → _nl_build!
# ```
#
# `_cl_sort_cells!` は #2（GPU の非決定性）で入れたもの。ここに出る数字が
# 「ソートのコストはステップ時間の数 %」という README の主張の根拠になる。

function breakdown(dp; backend = gpu, interval = 4)
    p = make_params(dp)
    X0 = water_column(dp)
    N = size(X0, 2)
    st = State(backend, X0, zeros(T, size(X0)), p; interval)
    theta = KernelAbstractions.zeros(backend, T, p.ngy, p.ngx)
    build_neighbors!(st.nl, st.cl, st.X, p, backend)
    cl = st.cl
    nl = st.nl
    cs = T(cl.cs)

    # ---- 構築系（interval ステップに 1 回） ----
    t_assign = time_stage(backend) do
        fill!(cl.counts, Int32(0))
        SPHAdjoint._cl_assign!(backend)(cl.cellof, cl.counts, st.X, cs, cl.nx, cl.ny;
                                        ndrange = N)
    end
    t_scan = time_stage(backend) do
        cumsum!(cl.cum, cl.counts)
        cl.starts .= cl.cum .- cl.counts
        cl.cursor .= cl.starts
    end
    ## cursor は消費されるので毎回巻き戻す（その copy 込みの時間）
    t_fill = time_stage(backend) do
        cl.cursor .= cl.starts
        SPHAdjoint._cl_fill!(backend)(cl.order, cl.cursor, cl.cellof; ndrange = N)
    end
    t_sort = time_stage(backend) do
        SPHAdjoint._cl_sort_cells!(backend)(cl.order, cl.starts, cl.counts;
                                            ndrange = length(cl.counts))
    end
    ## 近傍リスト本体。`check = true` は実運用と同じ（X が動いていないので
    ## 変位フラグは立たない）。
    t_nl = time_stage(backend) do
        SPHAdjoint._nl_build!(backend)(nl.indices, nl.counts, nl.flags, nl.Xref, st.X,
                                       cl.starts, cl.counts, cl.order,
                                       T(nl.rc^2), eps(T) * p.h * p.h, cs, cl.nx, cl.ny,
                                       nl.maxnb, nl.sm, nl.si, T(nl.dmax^2), true;
                                       ndrange = N)
    end

    # ---- 物理系（毎ステップ） ----
    t_rho = time_stage(backend) do
        SPHAdjoint.density_kernel!(backend)(st.rho, st.X, nl.counts, nl.indices,
                                            p.h, p.m, nl.sm, nl.si; ndrange = N)
    end
    t_eos = time_stage(backend) do
        SPHAdjoint.eos_kernel!(backend)(st.pterm, st.invrho, st.rho, p.c^2, p.rho0;
                                        ndrange = N)
    end
    t_acc = time_stage(backend) do
        SPHAdjoint.accel_kernel!(backend)(st.a, st.alpha, st.X, st.V, st.pterm,
                                          st.invrho, theta, nl.counts, nl.indices, p,
                                          nl.sm, nl.si; ndrange = N)
    end
    # ---- density / eos / accel を 1 回ずつ順に撃つ ----
    #
    # 個別に測った和との差が「同じカーネルを連続で撃つこと」のバイアス。
    #
    # **integrate はこの鎖に入れない。** X と V を進めてしまうので、鎖を 245 回
    # 回すと粒子が数 % ドリフトし、近傍リストも組み直されないまま古くなる。
    # そうなると `r2 < r2max` を外れる候補が増えて**鎖のほうが速く出る**——
    # 状態が壊れるほど速くなる測定になってしまう。上の 3 本は `rho` / `pterm` /
    # `invrho` / `a` / `alpha` にしか書かないので X と V は不変で、何回回しても
    # 同じ配置を測れる。
    t_chain = time_stage(backend) do
        SPHAdjoint.density_kernel!(backend)(st.rho, st.X, nl.counts, nl.indices,
                                            p.h, p.m, nl.sm, nl.si; ndrange = N)
        SPHAdjoint.eos_kernel!(backend)(st.pterm, st.invrho, st.rho, p.c^2, p.rho0;
                                        ndrange = N)
        SPHAdjoint.accel_kernel!(backend)(st.a, st.alpha, st.X, st.V, st.pterm,
                                          st.invrho, theta, nl.counts, nl.indices, p,
                                          nl.sm, nl.si; ndrange = N)
    end

    # ---- integrate は専用の State で ----
    #
    # `a` と `alpha` をゼロのままにしておけば `v' = 0`, `X += 0` で**状態が
    # 動かない**。演算量は本番と同じなので時間はそのまま使える。
    st_i = State(backend, X0, zeros(T, size(X0)), p; interval)
    t_int = time_stage(backend) do
        SPHAdjoint.integrate_kernel!(backend)(st_i.X, st_i.V, st_i.a, st_i.alpha, p.dt;
                                              ndrange = N)
    end

    # ---- step! 全体 ----
    #
    # 上の測定で状態が汚れているので新しい State で測る。**最初の 1 回は
    # `age = typemax` で必ず構築が走る**ので、ウォームアップで interval を
    # 何周かして定常状態にしてから測ること。reps も interval の倍数にして、
    # 測定区間に入る構築回数を `reps/interval` ちょうどにする。
    st2 = State(backend, X0, zeros(T, size(X0)), p; interval)
    reps = 12 * interval
    t_step = time_stage(backend; reps, warmup = 3 * interval) do
        step!(st2, theta, p, backend)
    end

    build_parts = (assign = t_assign, scan = t_scan, fill = t_fill,
                   sort = t_sort, nl_build = t_nl)
    phys_parts = (density = t_rho, eos = t_eos, accel = t_acc, integrate = t_int)
    t_build = sum(values(build_parts))
    t_phys = sum(values(phys_parts))
    t_amort = t_build / interval + t_phys

    nb = Array(nl.counts)
    ncell = cl.nx * cl.ny
    @printf("\nN = %d, セル数 = %d (%d×%d), 1 セルあたり平均 %.1f 粒子, 近傍数 平均 %.1f / 最大 %d\n",
            N, ncell, cl.nx, cl.ny, N / ncell, sum(nb) / N, maximum(nb))
    @printf("  %-12s %9s %11s %7s\n", "段階", "1 回 μs", "1 步 μs", "%")
    for (k, v) in pairs(build_parts)
        @printf("  %-12s %9.1f %11.1f %6.1f%%\n",
                k, v * 1e6, v * 1e6 / interval, 100 * (v / interval) / t_amort)
    end
    @printf("  %-12s %9.1f %11.1f %6.1f%%   ← %d ステップに 1 回\n",
            "構築 小計", t_build * 1e6, t_build * 1e6 / interval,
            100 * (t_build / interval) / t_amort, interval)
    for (k, v) in pairs(phys_parts)
        @printf("  %-12s %9.1f %11.1f %6.1f%%\n", k, v * 1e6, v * 1e6, 100 * v / t_amort)
    end
    @printf("  %-12s %9.1f %11.1f %6.1f%%\n", "物理 小計", t_phys * 1e6, t_phys * 1e6,
            100 * t_phys / t_amort)
    println("  ", "-"^44)
    t_phys3 = t_rho + t_eos + t_acc          # 鎖に入っている 3 本ぶん
    t_stack = t_chain + t_int + t_build / interval
    @printf("  %-12s %21.1f   ρ/eos/a を個別に測った和 %.1f との比 %.2fx\n",
            "ρ/eos/a 鎖", t_chain * 1e6, t_phys3 * 1e6, t_phys3 / t_chain)
    @printf("  %-12s %21.1f   = 鎖 + integrate + 構築/%d\n",
            "積み上げ", t_stack * 1e6, interval)
    @printf("  %-12s %21.1f   積み上げとの差 %+.1f μs\n",
            "step! 実測", t_step * 1e6, (t_step - t_stack) * 1e6)
    return (; N, build_parts, phys_parts, t_build, t_phys, t_chain, t_int,
            t_stack, t_amort, t_step, nb)
end

# ## 2. density の実効帯域
#
# セル走査時代は「3×3 セルの候補数」を一様密度から見積もっていたが、
# 近傍リストになった今は **`nl.counts` がそのまま走査数**なので推定は要らない。
#
# 1 近傍あたり読むのは `indices` の Int32 (4 B) と近傍の座標 (2×Float32 = 8 B)。
#
# **これは「コアレスせず毎回 DRAM を叩いた場合」の上限**で、実際は近傍の X が
# キャッシュに乗るので DRAM トラフィックはこれより小さい。したがって
# **達成率が 100% を超えることがある**。超えたら「帯域律速ではない」と読む
# （帯域が足りているのではなく、そもそも DRAM まで降りていない）。
#
# 逆に N が小さいと達成率は不当に低く出る。カーネル起動レイテンシが
# density の実時間を底上げするため（N=925 で数十 μs のうち大半が起動）。
# **この指標が意味を持つのは N が十分大きいときだけ。**

function bandwidth_check(N, nb, t_rho)
    total = sum(Int, nb)
    bytes = total * 12                  # indices 4 B + X 8 B
    gbps = bytes / t_rho / 1e9
    @printf("  リスト長の合計 %d（平均 %.1f/粒子）→ 読み出し %.2f MB\n",
            total, total / N, bytes / 1e6)
    @printf("  density %.1f μs → 実効帯域 %.1f GB/s（M2 の公称 %.0f GB/s に対し %.0f%%）\n",
            t_rho * 1e6, gbps, PEAK_GBPS, 100 * gbps / PEAK_GBPS)
    return gbps
end

# ## 3. 粒子をセル順に並べ替えたら速くなるか
#
# 粒子の格納順は初期配置のまま。近傍リスト経由でも `indices[…]` の指す先が
# メモリ上バラバラなら gather はコアレスしない。粒子データ自体をセル順に
# 並べ替えれば近傍が連続配置になるはず。
#
# ここでは「並べ替えた配置を作って density を測る」ことで**効果の上限**だけを
# 見る（並べ替え自体のコストと、テープ・随伴との対応付けは #10 の話）。

function sorted_layout_gain(dp)
    p = make_params(dp)
    X0 = water_column(dp)
    N = size(X0, 2)

    stu = State(gpu, X0, zeros(T, size(X0)), p)
    build_neighbors!(stu.nl, stu.cl, stu.X, p, gpu)

    ## order の順に並べ替えた初期配置を作る
    perm = Array(stu.cl.order)
    X0s = X0[:, perm]
    sts = State(gpu, X0s, zeros(T, size(X0)), p)
    build_neighbors!(sts.nl, sts.cl, sts.X, p, gpu)

    ## 交互に測る（片方をまとめて測ると先攻が 1.6 倍不利になることがある）
    tu, ts, rmed = time_ab(gpu,
        () -> SPHAdjoint.density_kernel!(gpu)(stu.rho, stu.X, stu.nl.counts,
                                              stu.nl.indices, p.h, p.m,
                                              stu.nl.sm, stu.nl.si; ndrange = N),
        () -> SPHAdjoint.density_kernel!(gpu)(sts.rho, sts.X, sts.nl.counts,
                                              sts.nl.indices, p.h, p.m,
                                              sts.nl.sm, sts.nl.si; ndrange = N))
    ## 答えが変わっていないことも確認（並べ替えても密度の集合は同じ）
    du = sort(Array(stu.rho))
    ds = sort(Array(sts.rho))
    @printf("  N=%-6d density: 現状 %7.1f μs → セル順 %7.1f μs  (最良比 %.2fx / 比の中央値 %.2fx)   密度の一致 %.2e\n",
            N, tu * 1e6, ts * 1e6, tu / ts, rmed,
            maximum(abs, du .- ds) / maximum(abs, du))
    return rmed
end

# ## 実行

const DPS = (0.012, 0.003, 0.0015)
const INTERVAL = 4

@printf("=== 1. 段階ごとの内訳（Metal, interval = %d）===\n", INTERVAL)
res = [breakdown(dp; interval = INTERVAL) for dp in DPS]

println("\n=== 2. density の実効帯域（Metal）===")
for (dp, r) in zip(DPS, res)
    @printf("dp=%.4f N=%d\n", dp, r.N)
    bandwidth_check(r.N, r.nb, r.phys_parts.density)
end

println("\n=== 3. 粒子をセル順に並べ替えた場合の density（Metal）===")
for dp in DPS
    sorted_layout_gain(dp)
end

# ## 分かったこと（2026-07-29, Apple M2 10 GPU コア, Float32, interval=4）
#
# 数字は**同一セッションで 3 回**回した幅。N=15000 / 60000 は数 % に収まるが、
# **N=925 は単体カーネルが 1.5 倍動く**（density 21〜32 μs）ので絶対値を見ないこと。
#
# **この幅は節 1 の測定経路に限った話。** 同じ density を節 3 の経路（別に作った
# `State`、別のウォームアップ）で測ると N=15000 で **37〜57 μs** に散る。
# 同一セッション・同一カーネル・同一 N でもこれだけ動くということで、
# **絶対値は測定の文脈に依存する**。節 3 が比だけを報告しているのはそのため。
#
# ### 重いのは accel と近傍リストの構築
#
# | 1 步あたり μs | N=925 | N=15000 | N=60000 |
# |---|---:|---:|---:|
# | 構築 小計（interval に 1 回） | 33〜41 | 60〜63 | 110〜125 |
# | ├ `_nl_build!` | 13〜21 | 29〜36 | **79〜80** |
# | └ セルリスト（assign+scan+fill+sort） | 20 | 27〜31 | 31〜45 |
# | density | 21〜32 | **37.3〜37.5** | **90** |
# | accel | 27〜42 | **66.2〜66.6** | **228〜231** |
# | `step!` 実測 | 76〜78 | **165〜167** | **473〜475** |
#
# **accel が単独で最大**（N=60000 で `step!` の 48%）。次が近傍リストの構築で、
# `interval` に 1 回に償却しても 17%。`_cl_sort_cells!` は 1 步あたり 4.7〜5.4 μs、
# `step!` の **1.0〜1.1%**（#2 で入れたときの見積り「ステップ時間の 1.2〜2.8%」の
# 下端に収まる）。
#
# ### density は帯域律速ではない
#
# N=60000 で「コアレスせず毎回 DRAM を叩いた場合」の上限に対して **220〜222%**。
# つまり近傍の X は実際にはキャッシュに乗っていて DRAM まで降りていない。
# **density をこれ以上速くしたいならメモリではなく演算側**（が、accel のほうが
# 2.5 倍重いのでそちらが先）。
#
# ### 粒子データのセル順並べ替えはもう効かない（#10）
#
# 比の中央値で **0.97〜1.02x**（3 つの N、3 回とも）。セル走査時代は 1.07〜1.17x
# あったが、**近傍リストを実体化した時点で効果が消えている**。リスト構築の中で
# 近傍を集めてしまうので、粒子の格納順がバラバラでも痛むのは構築の 1 回だけになり、
# それを 4 ステップに償却しているため。#10（定期再ソート）は、この測定を
# 見るかぎり前進のみでも旨みが薄い。
#
# **この結論に至るまでにハーネスのバグを 3 つ潰した**（どれも「効果あり」と
# 誤報する向きに効いた）。同じ轍を踏まないよう記録しておく：
#
# 1. **単発測定** — best-of を取っていなかった。同じ比較が 1.00x と 1.43x に割れた
# 2. **A をまとめて → B をまとめて** — 先攻が不利になる。まったく同じ density の
#    測定が 59.3 μs / 37.4 μs（1.6 倍）に割れた。`time_ab` で交互にした
# 3. **最良値どうしの比** — 交互にしても、測定区間まるごとが遅くなると最良値も
#    揃って遅い。4 回に 1 回 1.54x が出た。**各試行の中で比を取り、その中央値**を
#    採ると 0.97〜1.02x に収まる（`time_ab` の戻り値の 3 番目）
