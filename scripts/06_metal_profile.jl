# # Metal はどこで詰まっているか
#
# `05_metal_bench.jl` で分かったこと：GPU/CPU 比が **N=3750 以降 3 倍で頭打ち**
# （3750→60000 と 16 倍粒子を増やしても 2.76x → 3.11x にしかならない）。
# 粒子数を増やしても恩恵が増えない以上、律速はカーネルの中身側にある。
#
# ```
# julia --project=. scripts/06_metal_profile.jl
# ```
#
# ここで測るのは 3 つ。
#
# 1. **段階ごとの内訳** — build（セルリスト）と物理カーネルのどちらが重いか
# 2. **メモリ帯域に対する達成率** — 帯域律速なのか、それ以前の問題なのか
# 3. **粒子をセル順に並べ替えると速くなるか** — 近傍アクセスが
#    コアレスしていない疑いの検証。これが効くなら並べ替えを実装する価値がある。

using SPHAdjoint
using KernelAbstractions
using Metal
using Printf

const T = Float32
const gpu = MetalBackend()
const cpu = CPU()

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

"""カーネル 1 回あたりの時間。GPU は非同期なので reps 回まとめて撃って割る。"""
function time_stage(f, backend; reps = 50, warmup = 5)
    for _ in 1:warmup
        f()
    end
    KernelAbstractions.synchronize(backend)
    t0 = time()
    for _ in 1:reps
        f()
    end
    KernelAbstractions.synchronize(backend)
    return (time() - t0) / reps
end

# ## 1. 段階ごとの内訳
#
# `step!` は build（`_cl_assign!` → プレフィックス和 → `_cl_fill!`）と
# 物理 3 本（density / accel / integrate）から成る。

function breakdown(dp; backend = gpu)
    p = make_params(dp)
    X0 = water_column(dp)
    N = size(X0, 2)
    st = State(backend, X0, zeros(T, size(X0)), p)
    theta = KernelAbstractions.zeros(backend, T, p.ngy, p.ngx)
    build!(st.cl, st.X, p, backend)
    cs = T(st.cl.cs)
    cl = st.cl

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
    ## cursor は消費されるので毎回巻き戻す（その coppy 込みの時間）
    t_fill = time_stage(backend) do
        cl.cursor .= cl.starts
        SPHAdjoint._cl_fill!(backend)(cl.order, cl.cursor, cl.cellof; ndrange = N)
    end
    t_rho = time_stage(backend) do
        SPHAdjoint.density_kernel!(backend)(st.rho, st.X, cl.starts, cl.counts, cl.order,
                                            p.h, p.m, cs, cl.nx, cl.ny; ndrange = N)
    end
    t_acc = time_stage(backend) do
        SPHAdjoint.eos_kernel!(backend)(st.pterm, st.invrho, st.rho, p.c^2, p.rho0;
                                        ndrange = N)
        SPHAdjoint.accel_kernel!(backend)(st.a, st.X, st.V, st.pterm, st.invrho, theta,
                                          cl.starts, cl.counts, cl.order, p, cs,
                                          cl.nx, cl.ny; ndrange = N)
    end
    t_int = time_stage(backend) do
        SPHAdjoint.integrate_kernel!(backend)(st.X, st.V, st.a, p.dt; ndrange = N)
    end
    ## step! は状態を進めるので、上の測定で汚れていない新しい State で測る
    st2 = State(backend, X0, zeros(T, size(X0)), p)
    t_step = time_stage(backend) do
        step!(st2, theta, p, backend)
    end

    ncell = cl.nx * cl.ny
    parts = (assign = t_assign, scan = t_scan, fill = t_fill,
             density = t_rho, accel = t_acc, integrate = t_int)
    @printf("\nN = %d, セル数 = %d (%d×%d), 1 セルあたり平均 %.1f 粒子\n",
            N, ncell, cl.nx, cl.ny, N / ncell)
    @printf("  %-10s %8s %6s\n", "段階", "μs", "%")
    tot = sum(values(parts))
    for (k, v) in pairs(parts)
        @printf("  %-10s %8.1f %5.1f%%\n", k, v * 1e6, 100 * v / tot)
    end
    @printf("  %-10s %8.1f  (段階の和 %.1f μs, 差分 = 起動オーバーヘッド)\n",
            "step! 全体", t_step * 1e6, tot * 1e6)
    return (; N, parts, t_step)
end

# ## 2. メモリ帯域に対する達成率
#
# density カーネルが 1 粒子あたり読むのは、近傍 n 個ぶんの座標 (2×4 バイト) と
# 添字。M2 のメモリ帯域は約 100 GB/s。ここに対して何割出ているかを見る。

function bandwidth_check(dp, t_rho)
    p = make_params(dp)
    X0 = water_column(dp)
    N = size(X0, 2)
    ## サポート半径 2h 内の平均近傍数（一様密度近似）: π(2h)²/dp²
    nnb = π * (2 * p.h)^2 / dp^2
    ## 走査するのは 3×3 セル（セル幅 2h）なので、実際に触るのは (6h)²/dp² 個
    nscan = (3 * 2 * p.h)^2 / dp^2
    bytes = N * nscan * 8                      # 近傍の X を 2×Float32 読む
    @printf("  近傍数 ≈ %.0f, 走査数 ≈ %.0f (3×3 セル), 読み出し ≈ %.1f MB\n",
            nnb, nscan, bytes / 1e6)
    @printf("  density %.1f μs → 実効帯域 %.1f GB/s (M2 のピークは約 100 GB/s → %.0f%%)\n",
            t_rho * 1e6, bytes / t_rho / 1e9, 100 * (bytes / t_rho / 1e9) / 100)
end

# ## 3. 粒子をセル順に並べ替えたら速くなるか
#
# 現状は粒子の格納順が初期配置のまま。近傍リストは `order[s+k]` 経由の間接参照
# なので、隣り合うスレッドがメモリ上バラバラの粒子を読む。粒子データ自体を
# セル順に並べ替えれば近傍が連続配置になり、コアレスするはず。
#
# 並べ替えの実装は本体には入れず、ここでは「並べ替えた配置を作って density を
# 測る」ことで**効果の上限**だけを見る（並べ替え自体のコストは別途）。

function sorted_layout_gain(dp)
    p = make_params(dp)
    X0 = water_column(dp)
    N = size(X0, 2)

    stu = State(gpu, X0, zeros(T, size(X0)), p)
    build!(stu.cl, stu.X, p, gpu)
    cs = T(stu.cl.cs)

    ## order の順に並べ替えた初期配置を作る
    perm = Array(stu.cl.order)
    X0s = X0[:, perm]
    sts = State(gpu, X0s, zeros(T, size(X0)), p)
    build!(sts.cl, sts.X, p, gpu)

    tu = time_stage(gpu) do
        SPHAdjoint.density_kernel!(gpu)(stu.rho, stu.X, stu.cl.starts, stu.cl.counts,
                                        stu.cl.order, p.h, p.m, cs, stu.cl.nx, stu.cl.ny;
                                        ndrange = N)
    end
    ts = time_stage(gpu) do
        SPHAdjoint.density_kernel!(gpu)(sts.rho, sts.X, sts.cl.starts, sts.cl.counts,
                                        sts.cl.order, p.h, p.m, cs, sts.cl.nx, sts.cl.ny;
                                        ndrange = N)
    end
    ## 答えが変わっていないことも確認（並べ替えても密度の集合は同じ）
    du = sort(Array(stu.rho))
    ds = sort(Array(sts.rho))
    @printf("  N=%-6d density: 現状 %7.1f μs → セル順 %7.1f μs  (%.2fx)   密度の一致 %.2e\n",
            N, tu * 1e6, ts * 1e6, tu / ts, maximum(abs, du .- ds) / maximum(abs, du))
    return tu / ts
end

# ## 実行

println("=== 1. 段階ごとの内訳（Metal）===")
res = [breakdown(dp) for dp in (0.012, 0.003, 0.0015)]

println("\n=== 2. density の実効帯域（Metal）===")
for (dp, r) in zip((0.012, 0.003, 0.0015), res)
    @printf("dp=%.4f N=%d\n", dp, r.N)
    bandwidth_check(dp, r.parts.density)
end

println("\n=== 3. 粒子をセル順に並べ替えた場合の density（Metal）===")
for dp in (0.012, 0.003, 0.0015)
    sorted_layout_gain(dp)
end
