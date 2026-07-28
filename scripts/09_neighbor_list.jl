# # CSR 近傍リストのプロトタイプ（issue #1）
#
# > **【履歴】このスクリプトは旧 API 前提で、現在の `src/` では走らない（#17）。**
# >
# > ここで比較していた「転置・行長固定」のレイアウトが **#1 で本番になった**
# > （`src/neighbors.jl` の `NeighborList`）。つまりこのファイルは
# > **プロトタイプが本番に昇格する前の姿**で、ベースラインとして呼んでいる
# > `density_kernel!(…, cl.starts, cl.counts, cl.order, …)` は今の
# > シグネチャと合わない。レイアウト比較そのものは #16 で `layout` キーワード
# > （`:slot` / `:particle`）として本番に入り、CPU / GPU で切り替わる。
# >
# > 結論（素直な CSR は構築がセルリストの 10 倍で割に合わない、転置なら構築
# > 422 μs で density 1.86x / accel 1.87x、損益分岐となる再構築間隔）は
# > `src/neighbors.jl` 冒頭のコメントに取り込み済み。当時の測定を再現したい
# > ときは、そのときのコードごと取り出すこと：
# >
# > ```console
# > git worktree add /tmp/sph-42b40c8 42b40c8
# > cp Manifest.toml /tmp/sph-42b40c8/      # Manifest は gitignore なので要コピー
# > JULIA_PKG_PRECOMPILE_AUTO=0 \
# >   julia --project=/tmp/sph-42b40c8 /tmp/sph-42b40c8/scripts/09_neighbor_list.jl
# > ```
# >
# > 手順は実際に通して確認済み（2026-07-29）。`JULIA_PKG_PRECOMPILE_AUTO=0` が
# > 要るのは、`GLMakie` が `[deps]` にあるせいでヘッドレスの precompile が
# > segfault するから（#13）。
# >
# > **今の実装の内訳を見たいなら `06_metal_profile.jl`**（新パイプラインに
# > 追従済み）を使うこと。
#
# ## なぜやるか
#
# N=60000 の 1 ステップ 687 μs のうち accel が約 410 μs（60%）。
# 3×3 セル走査は 61 候補を舐めて実近傍は 21 個（π/9 ≈ 35%）で、
# 65% は空振りの距離判定と無駄な gather になっている。
#
# ## 設計上の要点（先に算数をしておく）
#
# 「候補訪問数」で数えると、**毎ステップ再構築なら CSR は前進を遅くする**：
#
# ```
# 現状      : density 61 + accel 61                      = 122
# CSR 毎回  : count 61 + fill 61 + density 21 + accel 21 = 164
# ```
#
# 効果は skin による再利用から出る。K ステップ再利用すれば
#
# ```
# CSR(K=20) : (61+61)/20 + 21 + 21 ≈ 48        → 現状比 2.5 倍
# ```
#
# 随伴込みなら物理パスが 4 本（density / accel / pass1 / pass2）なので、
# 構築コストの償却はさらに効く。
#
# ここでは構築コストと物理カーネルの実測から**損益分岐となる再構築間隔**を出す。
#
# ```
# julia --project=. scripts/09_neighbor_list.jl
# ```
#
# ## 結論（先に書いておく）
#
# **レイアウトが全てだった。** 素直な CSR（offsets + indices、行ごとに可変長）は
# 構築が 1438 μs (N=60000) もかかり、セルリスト 138 μs の 10 倍で割に合わない。
#
# 行長を固定して**転置レイアウト**（粒子 i の m 番目の近傍を `indices[(m-1)*N+i]`
# に置く）に変えると：
#
# - 同じ m で全スレッドが連続アドレスを触るので**書き込みも読み出しもコアレス**する
# - 行長が既知なので **count パス・cumsum・offsets が丸ごと不要**になる
#   （2 パス CSR ではこの前段だけで 419 μs 使っていた）
#
# 結果 (N=60000, skin=20%):
#
# | | 2 パス CSR | 転置・行長固定 |
# |---|---:|---:|
# | 構築 | 1438 μs | **422 μs** |
# | density | 227 μs (0.79x) | **96 μs (1.86x)** |
# | accel | 300 μs (1.34x) | **213 μs (1.87x)** |
# | 前進 1 step (K=20) | 680 μs (1.05x) | **330 μs (2.14x)** |
# | 随伴込み (K=20) | 1208 μs (1.07x) | **639 μs (2.01x)** |
#
# issue #1 に書いた見積もり 1.5〜1.7 倍を上回る。近傍数は skin=20% で平均 27.9 /
# 最大 32 だったので MAXNB=64 で安全（48 でも足りる）。
#
# 残件: 再構築の判定。最大変位 > skin/2 をホストで見ると毎ステップ同期が入って
# しまう（同期除去の効果が消える）ので、固定間隔 K + 数十ステップに 1 回の
# 検算にするのが現実的。

using SPHAdjoint
using KernelAbstractions
using Metal
using Printf

const T = Float32
const gpu = MetalBackend()

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

function time_stage(f; reps = 50, warmup = 5)
    for _ in 1:warmup
        f()
    end
    KernelAbstractions.synchronize(gpu)
    t0 = time()
    for _ in 1:reps
        f()
    end
    KernelAbstractions.synchronize(gpu)
    return (time() - t0) / reps
end

best_time(f; n = 4) = minimum(time_stage(f) for _ in 1:n)

# ## セルリスト（セル幅を指定できる版）
#
# skin を付けるとカットオフが 2h + skin になり、3×3 走査が成立するには
# セル幅もそれ以上必要。本体の `CellList` はセル幅が 2h 固定なので、
# プロトタイプではセル幅を渡せる版をここに書く。

mutable struct CellListCS{IA}
    cellof::IA
    counts::IA
    cum::IA
    starts::IA
    cursor::IA
    order::IA
    nx::Int
    ny::Int
    cs::Float64
end

function CellListCS(backend, N, cs::Real, Lx, Ly)
    nx = max(1, ceil(Int, Float64(Lx) / cs) + 2)
    ny = max(1, ceil(Int, Float64(Ly) / cs) + 2)
    ncell = nx * ny
    z(n) = KernelAbstractions.zeros(backend, Int32, n)
    return CellListCS(z(N), z(ncell), z(ncell), z(ncell), z(ncell), z(N),
                      nx, ny, Float64(cs))
end

function build_cs!(cl::CellListCS, X, backend)
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

# ## CSR 構築
#
# 2 パス。まず各粒子の近傍数を数え、排他的プレフィックス和で offsets を作り、
# もう一度走査して indices に書く。
#
# **スレッド i は自分の行にしか書かない**ので、`_cl_fill!` と違ってアトミックが
# 要らない。行内は挿入ソートで粒子 id の昇順に保つ。これで総和順序が固定され、
# issue #2 の非決定性（同一設計で目的関数が 0.29% 揺らぐ）も同時に消える。
# 行長は 21〜40 程度なので挿入ソートで十分だし、skin 再利用でコストは償却される。

@kernel inbounds = true function _nl_count!(nbcount, @Const(X), @Const(starts),
                                            @Const(counts), @Const(order),
                                            rc2, r2min, cs, nx, ny)
    i = @index(Global)
    T = eltype(X)
    invcs = one(T) / T(cs)
    nx32 = Int32(nx); ny32 = Int32(ny)
    xi = X[1, i]; yi = X[2, i]
    cx = clamp(unsafe_trunc(Int32, floor(xi * invcs)) + Int32(2), Int32(1), nx32)
    cy = clamp(unsafe_trunc(Int32, floor(yi * invcs)) + Int32(2), Int32(1), ny32)
    cnt = Int32(0)
    for ddy in Int32(-1):Int32(1), ddx in Int32(-1):Int32(1)
        jx = cx + ddx; jy = cy + ddy
        if Int32(1) <= jx <= nx32 && Int32(1) <= jy <= ny32
            c = jx + (jy - Int32(1)) * nx32
            s = starts[c]; n = counts[c]
            for k in Int32(1):n
                j = order[s+k]
                dx = xi - X[1, j]; dy = yi - X[2, j]
                r2 = dx * dx + dy * dy
                if r2min < r2 < rc2
                    cnt += Int32(1)
                end
            end
        end
    end
    nbcount[i] = cnt
end

@kernel inbounds = true function _nl_fill!(indices, @Const(offsets), @Const(X),
                                           @Const(starts), @Const(counts), @Const(order),
                                           rc2, r2min, cs, nx, ny)
    i = @index(Global)
    T = eltype(X)
    invcs = one(T) / T(cs)
    nx32 = Int32(nx); ny32 = Int32(ny)
    xi = X[1, i]; yi = X[2, i]
    cx = clamp(unsafe_trunc(Int32, floor(xi * invcs)) + Int32(2), Int32(1), nx32)
    cy = clamp(unsafe_trunc(Int32, floor(yi * invcs)) + Int32(2), Int32(1), ny32)
    base = offsets[i]              # 0-based の排他的オフセット
    m = Int32(0)                   # これまでに書いた個数
    for ddy in Int32(-1):Int32(1), ddx in Int32(-1):Int32(1)
        jx = cx + ddx; jy = cy + ddy
        if Int32(1) <= jx <= nx32 && Int32(1) <= jy <= ny32
            c = jx + (jy - Int32(1)) * nx32
            s = starts[c]; n = counts[c]
            for k in Int32(1):n
                j = order[s+k]
                dx = xi - X[1, j]; dy = yi - X[2, j]
                r2 = dx * dx + dy * dy
                if r2min < r2 < rc2
                    # 挿入ソート: 昇順を保ったまま j を差し込む
                    pos = m
                    while pos > Int32(0) && indices[base+pos] > j
                        indices[base+pos+Int32(1)] = indices[base+pos]
                        pos -= Int32(1)
                    end
                    indices[base+pos+Int32(1)] = j
                    m += Int32(1)
                end
            end
        end
    end
end

mutable struct NeighborList{IA}
    offsets::IA     # (N+1,) Int32 排他的プレフィックス（0-based）
    indices::IA     # (総ペア数,) Int32
    nbcount::IA     # (N,) Int32 作業用
    cum::IA         # (N,) Int32 作業用
    cl::Any
    rc::Float64     # カットオフ = 2h + skin
    npairs::Int
end

function NeighborList(backend, N, p::SPHParams{T}, skin_ratio) where {T}
    rc = Float64(2 * p.h) * (1 + skin_ratio)
    cl = CellListCS(backend, N, rc, p.Lx, p.Ly)
    z(n) = KernelAbstractions.zeros(backend, Int32, n)
    return NeighborList(z(N + 1), z(1), z(N), z(N), cl, rc, 0)
end

function build_nl!(nl::NeighborList, X, p::SPHParams{T}, backend) where {T}
    N = size(X, 2)
    build_cs!(nl.cl, X, backend)
    cs = T(nl.cl.cs)
    rc2 = T(nl.rc^2)
    r2min = eps(T) * p.h * p.h
    _nl_count!(backend)(nl.nbcount, X, nl.cl.starts, nl.cl.counts, nl.cl.order,
                        rc2, r2min, cs, nl.cl.nx, nl.cl.ny; ndrange = N)
    cumsum!(nl.cum, nl.nbcount)
    # offsets[1] = 0, offsets[i+1] = cum[i]
    copyto!(view(nl.offsets, 2:N+1), nl.cum)
    total = Int(Array(view(nl.cum, N:N))[1])
    if length(nl.indices) < total
        nl.indices = KernelAbstractions.zeros(backend, Int32, total)
    end
    nl.npairs = total
    _nl_fill!(backend)(nl.indices, nl.offsets, X, nl.cl.starts, nl.cl.counts,
                       nl.cl.order, rc2, r2min, cs, nl.cl.nx, nl.cl.ny; ndrange = N)
    return nl
end

# ## CSR を使う物理カーネル
#
# skin 込みでリストを作ってあるので、サポート外の分は物理カーネル側で
# r² 判定して落とす（走査するのは 61 ではなく行長ぶんだけ）。

@kernel inbounds = true function density_csr!(rho, @Const(X), @Const(offsets),
                                              @Const(indices), h, mass)
    i = @index(Global)
    T = eltype(rho)
    A = SPHAdjoint.wnorm(T(h))
    invh = one(T) / T(h)
    r2max = T(4) * T(h) * T(h)
    xi = X[1, i]; yi = X[2, i]
    acc = zero(T)
    for k in (offsets[i]+Int32(1)):offsets[i+1]
        j = indices[k]
        dx = xi - X[1, j]; dy = yi - X[2, j]
        r2 = dx * dx + dy * dy
        if r2 < r2max
            q = (@fastmath sqrt(r2)) * invh
            u = one(T) - T(0.5) * q
            u2 = u * u
            acc += T(mass) * (A * u2 * u2 * (2 * q + one(T)))
        end
    end
    # 自己項（近傍リストは i を含まない）
    rho[i] = acc + T(mass) * SPHAdjoint.w_kern(zero(T), A)
end

@kernel inbounds = true function accel_csr!(a, @Const(X), @Const(V), @Const(pterm),
                                            @Const(invrho), @Const(theta),
                                            @Const(offsets), @Const(indices), p)
    i = @index(Global)
    T = eltype(a)
    h = p.h
    A = SPHAdjoint.wnorm(h)
    mass = p.m
    invh = one(T) / h
    r2max = T(4) * h * h
    Fc = -5 * A * invh * invh
    twomumass = 2 * mass * p.mu
    xi = X[1, i]; yi = X[2, i]
    vxi = V[1, i]; vyi = V[2, i]
    pti = pterm[i]
    iri = invrho[i]
    ax = zero(T); ay = zero(T)
    for k in (offsets[i]+Int32(1)):offsets[i+1]
        j = indices[k]
        dx = xi - X[1, j]; dy = yi - X[2, j]
        r2 = dx * dx + dy * dy
        if r2 < r2max
            q = (@fastmath sqrt(r2)) * invh
            u = one(T) - T(0.5) * q
            F = Fc * u * u * u
            Pij = mass * (pti + pterm[j])
            ax -= Pij * F * dx
            ay -= Pij * F * dy
            C = twomumass * iri * invrho[j] * F
            ax += C * (vxi - V[1, j])
            ay += C * (vyi - V[2, j])
        end
    end
    alpha, _, _ = SPHAdjoint.interp_alpha(theta, xi, yi, p)
    ax += p.gx - alpha * vxi + SPHAdjoint.wall_accel(xi, p.Lx, p.kw)
    ay += p.gy - alpha * vyi + SPHAdjoint.wall_accel(yi, p.Ly, p.kw)
    a[1, i] = ax
    a[2, i] = ay
end

# ## 実行

relerr(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), eps(Float32))

for dp in (0.003, 0.0015), skin_ratio in (0.0, 0.2)
    p = make_params(dp)
    X0 = water_column(dp)
    N = size(X0, 2)
    st = State(gpu, X0, zeros(T, size(X0)), p)
    theta = KernelAbstractions.zeros(gpu, T, p.ngy, p.ngx)
    ## 初期格子は近傍数が均一すぎるので、少し流してから測る
    for _ in 1:200
        step!(st, theta, p, gpu)
    end
    build!(st.cl, st.X, p, gpu)
    cl = st.cl
    cs = T(cl.cs)
    SPHAdjoint.density_kernel!(gpu)(st.rho, st.X, cl.starts, cl.counts, cl.order,
                                    p.h, p.m, cs, cl.nx, cl.ny; ndrange = N)
    SPHAdjoint.eos_kernel!(gpu)(st.pterm, st.invrho, st.rho, p.c^2, p.rho0; ndrange = N)
    at_ref = KernelAbstractions.zeros(gpu, T, 2, N)
    SPHAdjoint.accel_kernel!(gpu)(at_ref, st.X, st.V, st.pterm, st.invrho, theta,
                                  cl.starts, cl.counts, cl.order, p, cs,
                                  cl.nx, cl.ny; ndrange = N)
    KernelAbstractions.synchronize(gpu)
    rho_ref = copy(st.rho)
    a_ref = copy(at_ref)

    nl = NeighborList(gpu, N, p, skin_ratio)
    build_nl!(nl, st.X, p, gpu)
    KernelAbstractions.synchronize(gpu)

    rt = KernelAbstractions.zeros(gpu, T, N)
    at = KernelAbstractions.zeros(gpu, T, 2, N)

    t_build = best_time(() -> build_nl!(nl, st.X, p, gpu))
    t_rho = best_time() do
        density_csr!(gpu)(rt, st.X, nl.offsets, nl.indices, p.h, p.m; ndrange = N)
    end
    t_acc = best_time() do
        accel_csr!(gpu)(at, st.X, st.V, st.pterm, st.invrho, theta,
                        nl.offsets, nl.indices, p; ndrange = N)
    end
    ## 現状（セル走査）側
    t_rho0 = best_time() do
        SPHAdjoint.density_kernel!(gpu)(rt, st.X, cl.starts, cl.counts, cl.order,
                                        p.h, p.m, cs, cl.nx, cl.ny; ndrange = N)
    end
    t_acc0 = best_time() do
        SPHAdjoint.accel_kernel!(gpu)(at, st.X, st.V, st.pterm, st.invrho, theta,
                                      cl.starts, cl.counts, cl.order, p, cs,
                                      cl.nx, cl.ny; ndrange = N)
    end
    t_build0 = best_time(() -> build!(st.cl, st.X, p, gpu))

    ## 正しさ
    density_csr!(gpu)(rt, st.X, nl.offsets, nl.indices, p.h, p.m; ndrange = N)
    accel_csr!(gpu)(at, st.X, st.V, st.pterm, st.invrho, theta,
                    nl.offsets, nl.indices, p; ndrange = N)
    KernelAbstractions.synchronize(gpu)

    @printf("\n===== dp=%.4f N=%d skin=%.0f%% =====\n", dp, N, 100 * skin_ratio)
    @printf("  ペア総数 %d（1 粒子あたり平均 %.1f）\n", nl.npairs, nl.npairs / N)
    @printf("  正しさ: density rel %.1e   accel rel %.1e\n",
            relerr(Array(rt), Array(rho_ref)), relerr(Array(at), Array(a_ref)))
    @printf("  構築    : セルリストのみ %6.1f μs → CSR %6.1f μs\n",
            t_build0 * 1e6, t_build * 1e6)
    @printf("  density : 現状 %6.1f μs → CSR %6.1f μs  (%.2fx)\n",
            t_rho0 * 1e6, t_rho * 1e6, t_rho0 / t_rho)
    @printf("  accel   : 現状 %6.1f μs → CSR %6.1f μs  (%.2fx)\n",
            t_acc0 * 1e6, t_acc * 1e6, t_acc0 / t_acc)

    ## 1 ステップあたりの総コスト。K = 再構築間隔。
    ## 現状は毎ステップ build（セルリスト）が要る。
    now = t_build0 + t_rho0 + t_acc0
    @printf("  --- 前進 1 ステップ（現状 %.1f μs）---\n", now * 1e6)
    for K in (1, 5, 10, 20, 50)
        csr = t_build / K + t_rho + t_acc
        @printf("    K=%-3d 再構築: %6.1f μs  (%.2fx)\n", K, csr * 1e6, now / csr)
    end
    ## 随伴込み（物理パスが 4 本、構築は 1 回）
    now_adj = t_build0 + t_rho0 + t_acc0 + t_rho0 + t_acc0
    @printf("  --- 随伴込み 1 ステップ相当（現状 %.1f μs）---\n", now_adj * 1e6)
    for K in (1, 5, 10, 20, 50)
        csr = t_build / K + 2 * (t_rho + t_acc)
        @printf("    K=%-3d 再構築: %6.1f μs  (%.2fx)\n", K, csr * 1e6, now_adj / csr)
    end
end

# ## 転置・行長固定レイアウト（採用候補）
#
# 粒子 i の m 番目の近傍を `indices[(m-1)*N + i]` に置く。
# 同じ m で全スレッドが連続アドレスを触るのでコアレスする。
# 行長固定なので count パス・cumsum・offsets が要らない。

const MAXNB = Int32(64)

@kernel inbounds = true function _nl_fill_T!(indices, nbcount, ovf, @Const(X),
                                             @Const(starts), @Const(counts), @Const(order),
                                             rc2, r2min, cs, nx, ny, N)
    i = @index(Global)
    T = eltype(X)
    invcs = one(T) / T(cs)
    nx32 = Int32(nx); ny32 = Int32(ny); N32 = Int32(N)
    xi = X[1, i]; yi = X[2, i]
    cx = clamp(unsafe_trunc(Int32, floor(xi * invcs)) + Int32(2), Int32(1), nx32)
    cy = clamp(unsafe_trunc(Int32, floor(yi * invcs)) + Int32(2), Int32(1), ny32)
    m = Int32(0)
    for ddy in Int32(-1):Int32(1), ddx in Int32(-1):Int32(1)
        jx = cx + ddx; jy = cy + ddy
        if Int32(1) <= jx <= nx32 && Int32(1) <= jy <= ny32
            c = jx + (jy - Int32(1)) * nx32
            s = starts[c]; n = counts[c]
            for k in Int32(1):n
                j = order[s+k]
                dx = xi - X[1, j]; dy = yi - X[2, j]
                r2 = dx * dx + dy * dy
                if r2min < r2 < rc2
                    if m < MAXNB
                        indices[m * N32 + Int32(i)] = j
                        m += Int32(1)
                    else
                        ovf[1] = Int32(1)   # 溢れを黙って捨てない
                    end
                end
            end
        end
    end
    nbcount[i] = m
end

@kernel inbounds = true function density_csrT!(rho, @Const(X), @Const(nbcount),
                                               @Const(indices), h, mass, N)
    i = @index(Global)
    T = eltype(rho)
    A = SPHAdjoint.wnorm(T(h))
    invh = one(T) / T(h)
    r2max = T(4) * T(h) * T(h)
    N32 = Int32(N)
    xi = X[1, i]; yi = X[2, i]
    acc = zero(T)
    for m in Int32(0):(nbcount[i]-Int32(1))
        j = indices[m * N32 + Int32(i)]
        dx = xi - X[1, j]; dy = yi - X[2, j]
        r2 = dx * dx + dy * dy
        if r2 < r2max
            q = (@fastmath sqrt(r2)) * invh
            u = one(T) - T(0.5) * q
            u2 = u * u
            acc += T(mass) * (A * u2 * u2 * (2 * q + one(T)))
        end
    end
    # 近傍リストは自己を含まないので自己項をここで足す
    rho[i] = acc + T(mass) * SPHAdjoint.w_kern(zero(T), A)
end

@kernel inbounds = true function accel_csrT!(a, @Const(X), @Const(V), @Const(pterm),
                                             @Const(invrho), @Const(theta),
                                             @Const(nbcount), @Const(indices), p, N)
    i = @index(Global)
    T = eltype(a)
    h = p.h; A = SPHAdjoint.wnorm(h); mass = p.m
    invh = one(T) / h
    r2max = T(4) * h * h
    Fc = -5 * A * invh * invh
    twomumass = 2 * mass * p.mu
    N32 = Int32(N)
    xi = X[1, i]; yi = X[2, i]
    vxi = V[1, i]; vyi = V[2, i]
    pti = pterm[i]; iri = invrho[i]
    ax = zero(T); ay = zero(T)
    for m in Int32(0):(nbcount[i]-Int32(1))
        j = indices[m * N32 + Int32(i)]
        dx = xi - X[1, j]; dy = yi - X[2, j]
        r2 = dx * dx + dy * dy
        if r2 < r2max
            q = (@fastmath sqrt(r2)) * invh
            u = one(T) - T(0.5) * q
            F = Fc * u * u * u
            Pij = mass * (pti + pterm[j])
            ax -= Pij * F * dx
            ay -= Pij * F * dy
            C = twomumass * iri * invrho[j] * F
            ax += C * (vxi - V[1, j])
            ay += C * (vyi - V[2, j])
        end
    end
    alpha, _, _ = SPHAdjoint.interp_alpha(theta, xi, yi, p)
    ax += p.gx - alpha * vxi + SPHAdjoint.wall_accel(xi, p.Lx, p.kw)
    ay += p.gy - alpha * vyi + SPHAdjoint.wall_accel(yi, p.Ly, p.kw)
    a[1, i] = ax
    a[2, i] = ay
end

println("\n\n########## 転置・行長固定レイアウト ##########")
for (dp, skin) in ((0.003, 0.2), (0.0015, 0.2))
    p = make_params(dp); X0 = water_column(dp); N = size(X0, 2)
    st = State(gpu, X0, zeros(T, size(X0)), p)
    theta = KernelAbstractions.zeros(gpu, T, p.ngy, p.ngx)
    for _ in 1:200; step!(st, theta, p, gpu); end
    build!(st.cl, st.X, p, gpu); cl = st.cl; cs0 = T(cl.cs)
    SPHAdjoint.density_kernel!(gpu)(st.rho, st.X, cl.starts, cl.counts, cl.order,
                                    p.h, p.m, cs0, cl.nx, cl.ny; ndrange=N)
    SPHAdjoint.eos_kernel!(gpu)(st.pterm, st.invrho, st.rho, p.c^2, p.rho0; ndrange=N)
    aref = KernelAbstractions.zeros(gpu, T, 2, N)
    SPHAdjoint.accel_kernel!(gpu)(aref, st.X, st.V, st.pterm, st.invrho, theta,
                                  cl.starts, cl.counts, cl.order, p, cs0, cl.nx, cl.ny; ndrange=N)
    KernelAbstractions.synchronize(gpu)
    rho_ref = copy(st.rho); a_ref = copy(aref)

    rc = 2*Float64(p.h)*(1+skin)
    clT = CellListCS(gpu, N, rc, p.Lx, p.Ly)
    idxT = KernelAbstractions.zeros(gpu, Int32, N*Int(MAXNB))
    nbc  = KernelAbstractions.zeros(gpu, Int32, N)
    ovf  = KernelAbstractions.zeros(gpu, Int32, 1)
    rc2 = T(rc^2); r2min = eps(T)*p.h*p.h
    ## 型不安定なフィールド越しの参照を測定ループに持ち込まない（1 度これで
    ## count が 50 ms に見える誤測定をやった）
    X = st.X; V = st.V; pt = st.pterm; ir = st.invrho
    ss = clT.starts; cc = clT.counts; oo = clT.order
    cnx = clT.nx; cny = clT.ny; csT = T(clT.cs)
    buildT() = begin
        build_cs!(clT, X, gpu)
        _nl_fill_T!(gpu)(idxT, nbc, ovf, X, ss, cc, oo, rc2, r2min, csT, cnx, cny, N; ndrange=N)
    end
    buildT(); KernelAbstractions.synchronize(gpu)
    @assert Array(ovf)[1] == 0 "MAXNB を超えた"

    rt = KernelAbstractions.zeros(gpu, T, N); at = KernelAbstractions.zeros(gpu, T, 2, N)
    t_build = best_time(buildT)
    t_rho = best_time(() -> density_csrT!(gpu)(rt, X, nbc, idxT, p.h, p.m, N; ndrange=N))
    t_acc = best_time(() -> accel_csrT!(gpu)(at, X, V, pt, ir, theta, nbc, idxT, p, N; ndrange=N))
    t_b0 = best_time(() -> build!(st.cl, X, p, gpu))
    t_r0 = best_time(() -> SPHAdjoint.density_kernel!(gpu)(rt, X, cl.starts, cl.counts,
                              cl.order, p.h, p.m, cs0, cl.nx, cl.ny; ndrange=N))
    t_a0 = best_time(() -> SPHAdjoint.accel_kernel!(gpu)(at, X, V, pt, ir, theta,
                              cl.starts, cl.counts, cl.order, p, cs0, cl.nx, cl.ny; ndrange=N))
    density_csrT!(gpu)(rt, X, nbc, idxT, p.h, p.m, N; ndrange=N)
    accel_csrT!(gpu)(at, X, V, pt, ir, theta, nbc, idxT, p, N; ndrange=N)
    KernelAbstractions.synchronize(gpu)

    @printf("\n===== dp=%.4f N=%d skin=%.0f%% =====\n", dp, N, 100*skin)
    @printf("  近傍数 平均 %.1f / 最大 %d (MAXNB=%d)\n",
            sum(Array(nbc))/N, maximum(Array(nbc)), MAXNB)
    @printf("  正しさ: density rel %.1e  accel rel %.1e\n",
            relerr(Array(rt), Array(rho_ref)), relerr(Array(at), Array(a_ref)))
    @printf("  構築   : セルリスト %6.1f → 転置CSR %6.1f μs\n", t_b0*1e6, t_build*1e6)
    @printf("  density: 現状 %6.1f → CSR %6.1f μs (%.2fx)\n", t_r0*1e6, t_rho*1e6, t_r0/t_rho)
    @printf("  accel  : 現状 %6.1f → CSR %6.1f μs (%.2fx)\n", t_a0*1e6, t_acc*1e6, t_a0/t_acc)
    now = t_b0+t_r0+t_a0; nowadj = t_b0+2*(t_r0+t_a0)
    @printf("  前進 1 step 現状 %.1f μs:\n", now*1e6)
    for K in (1,5,10,20,50)
        c = t_build/K+t_rho+t_acc
        @printf("    K=%-3d %6.1f μs (%.2fx)\n", K, c*1e6, now/c)
    end
    @printf("  随伴込み 現状 %.1f μs:\n", nowadj*1e6)
    for K in (1,5,10,20,50)
        c = t_build/K+2*(t_rho+t_acc)
        @printf("    K=%-3d %6.1f μs (%.2fx)\n", K, c*1e6, nowadj/c)
    end
end
