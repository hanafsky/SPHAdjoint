# # Metal での正しさとスループットの測定
#
# `02_dambreak.jl` の CPU 版に対して、**カーネルを一行も変えずに** backend を
# `MetalBackend()` に差し替えて同じ問題を解き、
#
# 1. 結果が CPU と一致するか（README「初回のチェックポイント」1〜4 の実地確認）
# 2. 粒子数を振ったときのステップ/秒
#
# を測る。②が `04_interactive.jl` の `nsub`（1 フレームあたりの物理ステップ数）
# を決める根拠になる。
#
# ```
# julia --project=. scripts/05_metal_bench.jl
# ```
#
# 比較は **CPU も Float32 で**行う。Apple GPU は倍精度を持たないので、
# Float64 の CPU 版と突き合わせると丸めの差なのか実装の差なのか分からなくなる。

using SPHAdjoint
using KernelAbstractions
using Metal
using Printf
using Random

const T = Float32
const gpu = MetalBackend()
const cpu = CPU()

# ## 共通のセットアップ
#
# `dp`（初期粒子間隔）だけを変えて問題サイズを振る。`dt` は音速 CFL
# （`dt < 0.25 h/c`）を保つよう `dp` に比例させる。

function make_params(dp)
    return SPHParams{T}(
        h    = 1.3 * dp,
        m    = dp^2 * 1.0,
        rho0 = 1.0,
        c    = 20.0,
        mu   = 0.05,
        dt   = 1.5e-4 * (dp / 0.012),
        Lx   = 1.0,
        Ly   = 0.6,
        kw   = 5.0e4,
        ngx  = 33,
        ngy  = 21,
    )
end

"左側の水柱。02_dambreak.jl と同じ形。"
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

# ## 1. 正しさ：CPU と Metal で同じ答えが出るか
#
# セルリスト（counting sort と Int32 アトミック）→ 密度総和 → 1 ステップ前進、
# の順に突き合わせる。ここが合えば README のチェックポイント 1〜3 は解決。

function check_forward(dp; nsteps = 20)
    p = make_params(dp)
    X0 = water_column(dp)
    V0 = zeros(T, size(X0))
    N = size(X0, 2)

    stc = State(cpu, X0, V0, p)
    stg = State(gpu, X0, V0, p)
    thc = KernelAbstractions.zeros(cpu, T, p.ngy, p.ngx)
    thg = KernelAbstractions.zeros(gpu, T, p.ngy, p.ngx)

    ## --- セルリスト単体 ---
    build!(stc.cl, stc.X, p, cpu)
    build!(stg.cl, stg.X, p, gpu)
    ok_counts = Array(stg.cl.counts) == Array(stc.cl.counts)
    ## order はセル内の並び順が非決定的なのでソートして比較する
    ok_order = sort(Array(stg.cl.order)) == collect(Int32(1):Int32(N))
    ok_starts = Array(stg.cl.starts) == Array(stc.cl.starts)

    ## --- 密度（順序に依らない量なので直接比べられる） ---
    for (st, be) in ((stc, cpu), (stg, gpu))
        SPHAdjoint.density_kernel!(be)(st.rho, st.X, st.cl.starts, st.cl.counts,
                                       st.cl.order, p.h, p.m, T(st.cl.cs),
                                       st.cl.nx, st.cl.ny; ndrange = N)
        KernelAbstractions.synchronize(be)
    end
    erho = maximum(abs, Array(stg.rho) .- Array(stc.rho)) / maximum(abs, Array(stc.rho))

    ## --- nsteps 前進したあとの位置・速度 ---
    for _ in 1:nsteps
        step!(stc, thc, p, cpu)
        step!(stg, thg, p, gpu)
    end
    eX = maximum(abs, Array(stg.X) .- Array(stc.X))
    eV = maximum(abs, Array(stg.V) .- Array(stc.V))

    @printf("  N=%-6d counts:%s order:%s starts:%s  rho rel %.2e  |ΔX| %.2e  |ΔV| %.2e\n",
            N, ok_counts, ok_order, ok_starts, erho, eX, eV)
    return (; N, ok_counts, ok_order, ok_starts, erho, eX, eV)
end

# ## 2. 正しさ：随伴が Metal で動くか
#
# ここだけ scatter が残る（設計変数場 θ への浮動小数アトミック）。
# README のチェックポイント 4 がこれ。CPU の随伴と突き合わせる。

function check_adjoint(dp; nsteps = 15)
    p = make_params(dp)
    X0 = water_column(dp; col_w = 0.15, col_h = 0.15)
    V0 = 0.05f0 .* randn(MersenneTwister(0), T, size(X0))
    N = size(X0, 2)
    theta0 = 3 .* rand(MersenneTwister(1), T, p.ngy, p.ngx)

    function run(backend)
        st = State(backend, X0, V0, p)
        th = KernelAbstractions.allocate(backend, T, p.ngy, p.ngx)
        copyto!(th, theta0)
        tape = Tape(backend, N, nsteps, p)
        simulate!(st, th, p, backend, nsteps; tape)
        ## 目標領域は**粒子群と重なる位置**に置くこと。遠くに置くと exp が
        ## アンダーフローして J も勾配も 0 になり、「0 と 0 が一致した」だけの
        ## 空虚な検証になる。
        J, seed = target_objective(st.X, 0.15, 0.10, 0.08)
        ws = AdjointWorkspace(backend, N, p)
        backward!(ws, tape, th, p, backend; seedX = seed, seedV = zero(seed))
        return J, Array(ws.gtheta), Array(ws.gX), Array(ws.gV)
    end

    Jc, gtc, gXc, gVc = run(cpu)
    Jg, gtg, gXg, gVg = run(gpu)

    rel(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), eps(T))
    ## 勾配が実際に有意な大きさを持っていることを併記する（空虚な一致の検出）
    @printf("  N=%-5d J %.5f / %.5f\n", N, Jc, Jg)
    @printf("  |gtheta|max %.3e  rel %.2e\n", maximum(abs, gtc), rel(gtg, gtc))
    @printf("  |gX|max     %.3e  rel %.2e\n", maximum(abs, gXc), rel(gXg, gXc))
    @printf("  |gV|max     %.3e  rel %.2e\n", maximum(abs, gVc), rel(gVg, gVc))
    @printf("  gtheta の非ゼロ節点数 CPU %d / Metal %d（全 %d）\n",
            count(!iszero, gtc), count(!iszero, gtg), length(gtc))
    return (; N, rel_gtheta = rel(gtg, gtc), rel_gX = rel(gXg, gXc))
end

# ## 3. スループット
#
# `step!` はカーネルごとに `synchronize` するので、GPU では 1 ステップあたり
# 5 回の同期が入る。小さい N ではこの往復レイテンシが支配的になり、GPU は
# CPU に負ける。どこで逆転するかを見るのが目的。

function bench(backend, dp; nsteps, warmup = 20)
    p = make_params(dp)
    X0 = water_column(dp)
    st = State(backend, X0, zeros(T, size(X0)), p)
    theta = KernelAbstractions.zeros(backend, T, p.ngy, p.ngx)

    for _ in 1:warmup
        step!(st, theta, p, backend)
    end
    KernelAbstractions.synchronize(backend)

    t0 = time()
    for _ in 1:nsteps
        step!(st, theta, p, backend)
    end
    KernelAbstractions.synchronize(backend)
    el = time() - t0

    finite = all(isfinite, Array(st.X))
    return nsteps / el, finite
end

nsteps_for(N) = N <= 1_000 ? 400 : N <= 5_000 ? 200 : N <= 20_000 ? 100 : 40

# ## 4. 同期を外したらどうなるか
#
# `step!` / `build!` はカーネルを撃つたびに `KernelAbstractions.synchronize` を
# 呼んでいる。同じ backend のカーネルは同じキューに順序どおり積まれるので、
# **ホストが結果を読むまで同期は要らない**はず。1 ステップ 5 回の同期が
# レイテンシとして効いているなら、外すだけで速くなる。
#
# ここでは `src/` を触らず、同期を外した版をスクリプト側に書いて比較する。
# 速くなることが確認できてから `src/forward.jl` に反映する。

function build_nosync!(cl, X, p::SPHParams{T}, backend) where {T}
    N = size(X, 2)
    fill!(cl.counts, Int32(0))
    SPHAdjoint._cl_assign!(backend)(cl.cellof, cl.counts, X, T(cl.cs), cl.nx, cl.ny;
                                    ndrange = N)
    cumsum!(cl.cum, cl.counts)
    cl.starts .= cl.cum .- cl.counts
    cl.cursor .= cl.starts
    SPHAdjoint._cl_fill!(backend)(cl.order, cl.cursor, cl.cellof; ndrange = N)
    return cl
end

# 現在の src/forward.jl の step! は同期を全部外した版そのものなので、
# 「同期なし」はただの step! である（この比較は歴史的な検証として残している）。
step_nosync!(st, theta, p::SPHParams, backend) = step!(st, theta, p, backend)

function bench_nosync(backend, dp; nsteps, warmup = 20)
    p = make_params(dp)
    X0 = water_column(dp)
    st = State(backend, X0, zeros(T, size(X0)), p)
    theta = KernelAbstractions.zeros(backend, T, p.ngy, p.ngx)

    for _ in 1:warmup
        step_nosync!(st, theta, p, backend)
    end
    KernelAbstractions.synchronize(backend)

    t0 = time()
    for _ in 1:nsteps
        step_nosync!(st, theta, p, backend)
    end
    KernelAbstractions.synchronize(backend)   # ホストが読む直前に 1 回だけ
    el = time() - t0
    return nsteps / el, all(isfinite, Array(st.X))
end

"同期を外しても答えが変わらないことを確認する（順序保証が効いているか）。"
function check_nosync(dp; nsteps = 50)
    p = make_params(dp)
    X0 = water_column(dp)
    V0 = zeros(T, size(X0))

    st1 = State(gpu, X0, V0, p)
    st2 = State(gpu, X0, V0, p)
    th = KernelAbstractions.zeros(gpu, T, p.ngy, p.ngx)
    for _ in 1:nsteps
        step!(st1, th, p, gpu)
        step_nosync!(st2, th, p, gpu)
    end
    KernelAbstractions.synchronize(gpu)
    d = maximum(abs, Array(st1.X) .- Array(st2.X))
    @printf("  dp=%.4f N=%-6d 同期あり vs 同期なし  |ΔX| = %.2e\n",
            dp, size(X0, 2), d)
    return d
end

# ## 5. 同期をどこまで外すか
#
# 選択肢は 2 つ。
#
# - **A: ステップ末尾に 1 回だけ残す** — 既存のスクリプト／テストを一切変えずに
#   済む。ホストがいつ読んでも安全。
# - **B: 完全に外す** — `Array()` / `copyto!`（ホストへの読み出し）が同期点に
#   なっていることに依存する。
#
# B が成立するなら A より速いはず。両方測り、B の安全性も確かめる。

function step_sync1!(st, theta, p::SPHParams{T}, backend) where {T}
    step_nosync!(st, theta, p, backend)
    KernelAbstractions.synchronize(backend)
    return st
end

function bench_with(stepfn, backend, dp; nsteps, warmup = 20)
    p = make_params(dp)
    X0 = water_column(dp)
    st = State(backend, X0, zeros(T, size(X0)), p)
    theta = KernelAbstractions.zeros(backend, T, p.ngy, p.ngx)
    for _ in 1:warmup
        stepfn(st, theta, p, backend)
    end
    KernelAbstractions.synchronize(backend)
    t0 = time()
    for _ in 1:nsteps
        stepfn(st, theta, p, backend)
    end
    KernelAbstractions.synchronize(backend)
    return nsteps / (time() - t0)
end

# `Array()` が同期点になっているか。なっていなければ「1 ステップ古い状態」や
# 途中結果が見えるので、同期あり版との差が丸め誤差 (~1e-7) では済まない。

function check_host_read_syncs(dp; nsteps = 50)
    p = make_params(dp)
    X0 = water_column(dp)
    V0 = zeros(T, size(X0))
    ref = State(gpu, X0, V0, p)
    tst = State(gpu, X0, V0, p)
    th = KernelAbstractions.zeros(gpu, T, p.ngy, p.ngx)
    for _ in 1:nsteps
        step!(ref, th, p, gpu)
        step_nosync!(tst, th, p, gpu)
    end
    KernelAbstractions.synchronize(gpu)
    Xref = Array(ref.X)
    ## ここが肝: synchronize を挟まずにいきなりホストへ読み出す
    for _ in 1:5
        step_nosync!(tst, th, p, gpu)
        step!(ref, th, p, gpu)
    end
    Xnow = Array(tst.X)                     # 同期を明示的に呼ばない
    KernelAbstractions.synchronize(gpu)
    Xafter = Array(tst.X)
    @printf("  dp=%.4f 同期無しで読んだ値 vs そのあと同期して読んだ値: |Δ| = %.2e\n",
            dp, maximum(abs, Xnow .- Xafter))
    @printf("  dp=%.4f 同期無しで読んだ値 vs 同期あり版の同ステップ:   |Δ| = %.2e\n",
            dp, maximum(abs, Xnow .- Array(ref.X)))
    return nothing
end

# ## 実行

println("=== 1. 前進の一致（CPU Float32 vs Metal Float32）===")
for dp in (0.024, 0.012, 0.006)
    check_forward(dp)
end

println("\n=== 2. 随伴の一致（θ への浮動小数アトミックを含む）===")
check_adjoint(0.02)

println("\n=== 3. スループット ===")
@printf("%-8s %-8s %11s %11s %11s %8s\n",
        "dp", "N", "CPU", "Metal", "Metal(同期無)", "GPU/CPU")
for dp in (0.024, 0.012, 0.006, 0.003, 0.0015)
    local n = size(water_column(dp), 2)
    local ns = nsteps_for(n)
    local sc, okc = bench(cpu, dp; nsteps = ns)
    local sg, okg = bench(gpu, dp; nsteps = ns)
    local sn, okn = bench_nosync(gpu, dp; nsteps = ns)
    @printf("%-8.4f %-8d %11.1f %11.1f %11.1f %8.2fx%s\n", dp, n, sc, sg, sn, sn / sc,
            (okc && okg && okn) ? "" : "  ← 発散")
end

println("\n=== 4. 同期を外しても答えが変わらないか ===")
for dp in (0.012, 0.003)
    check_nosync(dp)
end

println("\n=== 5. 同期をどこまで外すか（A: 末尾 1 回 / B: 完全に外す）===")
@printf("%-8s %-8s %11s %11s %11s\n", "dp", "N", "現状(5回)", "A(1回)", "B(0回)")
for dp in (0.012, 0.003, 0.0015)
    local n = size(water_column(dp), 2)
    local ns = nsteps_for(n)
    local s0 = bench_with(step!, gpu, dp; nsteps = ns)
    local s1 = bench_with(step_sync1!, gpu, dp; nsteps = ns)
    local s2 = bench_with(step_nosync!, gpu, dp; nsteps = ns)
    @printf("%-8.4f %-8d %11.1f %11.1f %11.1f\n", dp, n, s0, s1, s2)
end

println("\n  ホスト読み出しが同期点になっているか:")
for dp in (0.012, 0.003)
    check_host_read_syncs(dp)
end

# ## 6. 随伴のスループット
#
# `backward!` は 1 ステップに 7 回同期していたので、同期除去の効きはここが一番
# 大きいはず。最適化 1 反復＝前進 nsteps + 逆行 nsteps なので、この数字が
# `03_optimize.jl` の待ち時間を直接決める。

function bench_adjoint(backend, dp; nsteps = 100, warmup = true)
    p = make_params(dp)
    X0 = water_column(dp)
    V0 = zeros(T, size(X0))
    N = size(X0, 2)
    theta = KernelAbstractions.zeros(backend, T, p.ngy, p.ngx)

    ## ウォームアップ（コンパイル時間を計測から外す）
    if warmup
        stw = State(backend, X0, V0, p)
        tw = Tape(backend, N, 3, p)
        simulate!(stw, theta, p, backend, 3; tape = tw)
        _, sw = target_objective(stw.X, 0.5, 0.1, 0.1)
        wsw = AdjointWorkspace(backend, N, p)
        backward!(wsw, tw, theta, p, backend; seedX = sw, seedV = zero(sw))
        GC.gc()
    end

    ## テープ無しの前進（テープの純粋なオーバーヘッドを見るための基準）
    stn = State(backend, X0, V0, p)
    t00 = time()
    simulate!(stn, theta, p, backend, nsteps)
    tn = time() - t00

    st = State(backend, X0, V0, p)
    tape = Tape(backend, N, nsteps, p)
    t0 = time()
    simulate!(st, theta, p, backend, nsteps; tape)
    tf = time() - t0

    _, seed = target_objective(st.X, 0.5, 0.1, 0.1)
    ws = AdjointWorkspace(backend, N, p)
    t1 = time()
    backward!(ws, tape, theta, p, backend; seedX = seed, seedV = zero(seed))
    tb = time() - t1

    return nsteps / tn, nsteps / tf, nsteps / tb, maximum(abs, Array(ws.gtheta))
end

println("\n=== 6. 随伴のスループット（テープ無し前進 / テープ有り前進 / 逆行）===")
@printf("%-8s %-8s %11s %11s %11s %11s\n",
        "dp", "N", "tape無し", "tape有り", "逆行", "|gtheta|max")
for dp in (0.012, 0.006, 0.003)
    local n = size(water_column(dp), 2)
    for (name, be) in (("CPU  ", cpu), ("Metal", gpu))
        local f0, ff, bb, gt = bench_adjoint(be, dp; nsteps = n > 5000 ? 50 : 100)
        @printf("%-8.4f %-8d %11.1f %11.1f %11.1f %11.3e  %s\n",
                dp, n, f0, ff, bb, gt, name)
    end
end

# ## 04_interactive.jl の nsub をどう決めるか
#
# 30fps で実時間 1 倍を出すには 1 フレームあたり `1/(30·dt)` ステップ必要。
# 「シミュレーション秒 / 実秒」は単に `step/s × dt` で読める。

println()
for dp in (0.012, 0.006)
    local p = make_params(dp)
    local need = 1 / (30 * p.dt)
    @printf("dp=%.4f: dt=%.2e → 30fps 実時間 1 倍には %.0f ステップ/フレーム必要\n",
            dp, p.dt, need)
end
