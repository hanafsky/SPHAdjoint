# # カーネル変種の実験：Metal はどこまで速くなるか
#
# > **【履歴】このスクリプトは旧 API 前提で、現在の `src/` では走らない（#17）。**
# >
# > 本番カーネルがセル走査から近傍リスト走査に変わった（#1 / PR #14）ため、
# > ここでベースラインとして呼んでいる
# > `density_kernel!(…, cl.starts, cl.counts, cl.order, …)` が現在の
# > シグネチャと合わない。`accel_kernel!` はさらに古く、`pterm` / `invrho` を
# > 前計算する前の形（このスクリプトがその前計算を提案した側なので当然）。
# >
# > **呼び出しだけ直すのは意味がない。** 比較対象の変種カーネルは意図的に
# > 旧形式（`starts, counts, order`）で定義してあり、本番だけ差し替えると
# > 「セル走査の変種 vs リスト方式の本番」という土俵の違う比較になる。
# >
# > 結論（inbounds / r² 早期棄却 / 除算の前計算 / Int32 セル添字）は
# > `src/forward.jl` 冒頭のコメントに取り込み済み。当時の測定を再現したいときは、
# > **そのときのコードごと**取り出すこと：
# >
# > ```console
# > git worktree add /tmp/sph-3bed1cb 3bed1cb
# > cp Manifest.toml /tmp/sph-3bed1cb/      # Manifest は gitignore なので要コピー
# > JULIA_PKG_PRECOMPILE_AUTO=0 \
# >   julia --project=/tmp/sph-3bed1cb /tmp/sph-3bed1cb/scripts/07_kernel_tuning.jl
# > ```
# >
# > 手順は実際に通して確認済み（2026-07-29）。`JULIA_PKG_PRECOMPILE_AUTO=0` が
# > 要るのは、`GLMakie` が `[deps]` にあるせいでヘッドレスの precompile が
# > segfault するから（#13）。SPHAdjoint 自体は問題なく読める。
#
# `06_metal_profile.jl` の結論は「物理カーネル 2 本で 90%、実効帯域はピークの
# 33%」だった。33% で止まっているということは、帯域律速と断定できない。
# 現行カーネルには演算側の無駄が 3 つある：
#
# 1. **境界チェックが生きている**（Metal でデバイス側 BoundsError が出る =
#    チェックが有効な証拠）。全配列アクセスに Int64 比較と分岐が入っている。
# 2. **棄却前に sqrt と除算を払っている**。3×3 セルの走査候補 61 個のうち
#    近傍は 21 個（35%）なのに、`q = sqrt(r2)/h` を全員に計算してから
#    `q >= 2` で捨てている。r² のうちに捨てれば sqrt/除算は近傍だけで済む。
# 3. **ペアごとの除算**。圧力項 `p_i/ρ_i² + p_j/ρ_j²` で 2 回、粘性項
#    `1/(ρ_i ρ_j)` で 1 回。`pterm[i] = p_i/ρ_i²` と `invrho[i] = 1/ρ_i` を
#    粒子ごとに前計算（O(N)）すれば、ペアループ（O(61N)）から除算が消える。
#
# これに加えて README チェックポイント 3 の **Int32 セル添字**と、
# **ワークグループサイズ**を振る。変種は本体を触らずここで定義し、
# 効いたものだけを `src/` に反映する。
#
# ```
# julia --project=. scripts/07_kernel_tuning.jl
# ```

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

function time_stage(f, backend; reps = 100, warmup = 10)
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

"数回測って最小値を取る（GPU のばらつき対策。06 で測定ゆらぎに騙されかけた）。"
best_time(f, backend; n = 3) = minimum(time_stage(f, backend) for _ in 1:n)

# ## density の変種
#
# D1: inbounds のみ（計算は本体と同一）
# D2: D1 + r² 早期棄却 + 除算→逆数乗算
# D3: D2 + セル走査を Int32 に

@kernel inbounds = true function density_D1!(rho, @Const(X), @Const(starts), @Const(counts),
                                             @Const(order), h, mass, cs, nx, ny)
    i = @index(Global)
    T = eltype(rho)
    A = SPHAdjoint.wnorm(T(h))
    xi = X[1, i]
    yi = X[2, i]
    cx = clamp(unsafe_trunc(Int, floor(xi / T(cs))) + 2, 1, nx)
    cy = clamp(unsafe_trunc(Int, floor(yi / T(cs))) + 2, 1, ny)
    acc = zero(T)
    for ddy in -1:1, ddx in -1:1
        jx = cx + ddx
        jy = cy + ddy
        if 1 <= jx <= nx && 1 <= jy <= ny
            c = jx + (jy - 1) * nx
            s = starts[c]
            n = counts[c]
            for k in 1:n
                j = order[s+k]
                dx = xi - X[1, j]
                dy = yi - X[2, j]
                r = sqrt(dx * dx + dy * dy)
                acc += T(mass) * SPHAdjoint.w_kern(r / T(h), A)
            end
        end
    end
    rho[i] = acc
end

@kernel inbounds = true function density_D2!(rho, @Const(X), @Const(starts), @Const(counts),
                                             @Const(order), h, mass, cs, nx, ny)
    i = @index(Global)
    T = eltype(rho)
    A = SPHAdjoint.wnorm(T(h))
    invh = one(T) / T(h)
    invcs = one(T) / T(cs)
    r2max = T(4) * T(h) * T(h)
    xi = X[1, i]
    yi = X[2, i]
    cx = clamp(unsafe_trunc(Int, floor(xi * invcs)) + 2, 1, nx)
    cy = clamp(unsafe_trunc(Int, floor(yi * invcs)) + 2, 1, ny)
    acc = zero(T)
    for ddy in -1:1, ddx in -1:1
        jx = cx + ddx
        jy = cy + ddy
        if 1 <= jx <= nx && 1 <= jy <= ny
            c = jx + (jy - 1) * nx
            s = starts[c]
            n = counts[c]
            for k in 1:n
                j = order[s+k]
                dx = xi - X[1, j]
                dy = yi - X[2, j]
                r2 = dx * dx + dy * dy
                if r2 < r2max                       # ← sqrt の前に棄却
                    q = sqrt(r2) * invh
                    u = one(T) - T(0.5) * q
                    acc += T(mass) * (A * u^4 * (2 * q + one(T)))
                end
            end
        end
    end
    rho[i] = acc
end

@kernel inbounds = true function density_D3!(rho, @Const(X), @Const(starts), @Const(counts),
                                             @Const(order), h, mass, cs, nx, ny)
    i = @index(Global)
    T = eltype(rho)
    A = SPHAdjoint.wnorm(T(h))
    invh = one(T) / T(h)
    invcs = one(T) / T(cs)
    r2max = T(4) * T(h) * T(h)
    nx32 = Int32(nx)
    ny32 = Int32(ny)
    xi = X[1, i]
    yi = X[2, i]
    cx = clamp(unsafe_trunc(Int32, floor(xi * invcs)) + Int32(2), Int32(1), nx32)
    cy = clamp(unsafe_trunc(Int32, floor(yi * invcs)) + Int32(2), Int32(1), ny32)
    acc = zero(T)
    for ddy in Int32(-1):Int32(1), ddx in Int32(-1):Int32(1)
        jx = cx + ddx
        jy = cy + ddy
        if Int32(1) <= jx <= nx32 && Int32(1) <= jy <= ny32
            c = jx + (jy - Int32(1)) * nx32
            s = starts[c]
            n = counts[c]
            for k in Int32(1):n
                j = order[s+k]
                dx = xi - X[1, j]
                dy = yi - X[2, j]
                r2 = dx * dx + dy * dy
                if r2 < r2max
                    q = sqrt(r2) * invh
                    u = one(T) - T(0.5) * q
                    acc += T(mass) * (A * u^4 * (2 * q + one(T)))
                end
            end
        end
    end
    rho[i] = acc
end

# ## accel の変種
#
# A1: inbounds のみ
# A2: A1 + r² 早期棄却 + ペア除算の前計算化（pterm / invrho）
#
# A2 用の前計算カーネル。O(N) なのでコストはペアループの 1/61。

@kernel inbounds = true function eos_kernel!(pterm, invrho, @Const(rho), c2, rho0)
    i = @index(Global)
    T = eltype(pterm)
    ri = rho[i]
    inv = one(T) / ri
    invrho[i] = inv
    pterm[i] = T(c2) * (ri - T(rho0)) * inv * inv
end

@kernel inbounds = true function accel_A1!(a, @Const(X), @Const(V), @Const(rho), @Const(theta),
                                           @Const(starts), @Const(counts), @Const(order),
                                           p, cs, nx, ny)
    i = @index(Global)
    T = eltype(a)
    h = p.h
    A = SPHAdjoint.wnorm(h)
    mass = p.m
    c2 = p.c^2
    xi = X[1, i]; yi = X[2, i]
    vxi = V[1, i]; vyi = V[2, i]
    rhoi = rho[i]
    pi_ = c2 * (rhoi - p.rho0)
    cx = clamp(unsafe_trunc(Int, floor(xi / T(cs))) + 2, 1, nx)
    cy = clamp(unsafe_trunc(Int, floor(yi / T(cs))) + 2, 1, ny)
    ax = zero(T)
    ay = zero(T)
    for ddy in -1:1, ddx in -1:1
        jx = cx + ddx
        jy = cy + ddy
        if 1 <= jx <= nx && 1 <= jy <= ny
            cc = jx + (jy - 1) * nx
            s = starts[cc]
            n = counts[cc]
            for k in 1:n
                j = order[s+k]
                j == i && continue
                dx = xi - X[1, j]
                dy = yi - X[2, j]
                r2 = dx * dx + dy * dy
                r2 < eps(T) * h * h && continue
                r = sqrt(r2)
                q = r / h
                q >= T(2) && continue
                F = SPHAdjoint.f_kern(q, A, h)
                rhoj = rho[j]
                pj = c2 * (rhoj - p.rho0)
                Pij = mass * (pi_ / rhoi^2 + pj / rhoj^2)
                ax -= Pij * F * dx
                ay -= Pij * F * dy
                C = 2 * mass * p.mu / (rhoi * rhoj) * F
                ax += C * (vxi - V[1, j])
                ay += C * (vyi - V[2, j])
            end
        end
    end
    alpha, _, _ = SPHAdjoint.interp_alpha(theta, xi, yi, p)
    ax += p.gx - alpha * vxi + SPHAdjoint.wall_accel(xi, p.Lx, p.kw)
    ay += p.gy - alpha * vyi + SPHAdjoint.wall_accel(yi, p.Ly, p.kw)
    a[1, i] = ax
    a[2, i] = ay
end

@kernel inbounds = true function accel_A2!(a, @Const(X), @Const(V), @Const(pterm),
                                           @Const(invrho), @Const(theta),
                                           @Const(starts), @Const(counts), @Const(order),
                                           p, cs, nx, ny)
    i = @index(Global)
    T = eltype(a)
    h = p.h
    A = SPHAdjoint.wnorm(h)
    mass = p.m
    invh = one(T) / h
    invcs = one(T) / T(cs)
    r2max = T(4) * h * h
    r2min = eps(T) * h * h
    Fc = -5 * A * invh * invh               # f_kern(q) = Fc·u³
    twomumass = 2 * mass * p.mu
    xi = X[1, i]; yi = X[2, i]
    vxi = V[1, i]; vyi = V[2, i]
    pti = pterm[i]
    iri = invrho[i]
    cx = clamp(unsafe_trunc(Int, floor(xi * invcs)) + 2, 1, nx)
    cy = clamp(unsafe_trunc(Int, floor(yi * invcs)) + 2, 1, ny)
    ax = zero(T)
    ay = zero(T)
    for ddy in -1:1, ddx in -1:1
        jx = cx + ddx
        jy = cy + ddy
        if 1 <= jx <= nx && 1 <= jy <= ny
            cc = jx + (jy - 1) * nx
            s = starts[cc]
            n = counts[cc]
            for k in 1:n
                j = order[s+k]
                dx = xi - X[1, j]
                dy = yi - X[2, j]
                r2 = dx * dx + dy * dy
                if r2min < r2 < r2max           # ← sqrt の前に棄却（j==i も r2≈0 で落ちる）
                    q = sqrt(r2) * invh
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
        end
    end
    alpha, _, _ = SPHAdjoint.interp_alpha(theta, xi, yi, p)
    ax += p.gx - alpha * vxi + SPHAdjoint.wall_accel(xi, p.Lx, p.kw)
    ay += p.gy - alpha * vyi + SPHAdjoint.wall_accel(yi, p.Ly, p.kw)
    a[1, i] = ax
    a[2, i] = ay
end

@kernel inbounds = true function accel_A3!(a, @Const(X), @Const(V), @Const(pterm),
                                           @Const(invrho), @Const(theta),
                                           @Const(starts), @Const(counts), @Const(order),
                                           p, cs, nx, ny)
    i = @index(Global)
    T = eltype(a)
    h = p.h
    A = SPHAdjoint.wnorm(h)
    mass = p.m
    invh = one(T) / h
    invcs = one(T) / T(cs)
    r2max = T(4) * h * h
    r2min = eps(T) * h * h
    Fc = -5 * A * invh * invh
    twomumass = 2 * mass * p.mu
    nx32 = Int32(nx)
    ny32 = Int32(ny)
    xi = X[1, i]; yi = X[2, i]
    vxi = V[1, i]; vyi = V[2, i]
    pti = pterm[i]
    iri = invrho[i]
    cx = clamp(unsafe_trunc(Int32, floor(xi * invcs)) + Int32(2), Int32(1), nx32)
    cy = clamp(unsafe_trunc(Int32, floor(yi * invcs)) + Int32(2), Int32(1), ny32)
    ax = zero(T)
    ay = zero(T)
    for ddy in Int32(-1):Int32(1), ddx in Int32(-1):Int32(1)
        jx = cx + ddx
        jy = cy + ddy
        if Int32(1) <= jx <= nx32 && Int32(1) <= jy <= ny32
            cc = jx + (jy - Int32(1)) * nx32
            s = starts[cc]
            n = counts[cc]
            for k in Int32(1):n
                j = order[s+k]
                dx = xi - X[1, j]
                dy = yi - X[2, j]
                r2 = dx * dx + dy * dy
                if r2min < r2 < r2max
                    q = sqrt(r2) * invh
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
        end
    end
    alpha, _, _ = SPHAdjoint.interp_alpha(theta, xi, yi, p)
    ax += p.gx - alpha * vxi + SPHAdjoint.wall_accel(xi, p.Lx, p.kw)
    ay += p.gy - alpha * vyi + SPHAdjoint.wall_accel(yi, p.Ly, p.kw)
    a[1, i] = ax
    a[2, i] = ay
end

# ## セットアップと正しさの確認

function setup(dp)
    p = make_params(dp)
    X0 = water_column(dp)
    N = size(X0, 2)
    st = State(gpu, X0, zeros(T, size(X0)), p)
    theta = KernelAbstractions.zeros(gpu, T, p.ngy, p.ngx)
    ## 適当に流してから測る（初期格子配置は近傍数が均一すぎて楽観的になる）
    for _ in 1:200
        step!(st, theta, p, gpu)
    end
    build!(st.cl, st.X, p, gpu)
    cs = T(st.cl.cs)
    SPHAdjoint.density_kernel!(gpu)(st.rho, st.X, st.cl.starts, st.cl.counts, st.cl.order,
                                    p.h, p.m, cs, st.cl.nx, st.cl.ny; ndrange = N)
    KernelAbstractions.synchronize(gpu)
    return (; p, st, theta, cs, N)
end

relerr(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), eps(Float32))

# ## 実行

for dp in (0.003, 0.0015)
    S = setup(dp)
    p, st, theta, cs, N = S.p, S.st, S.theta, S.cs, S.N
    cl = st.cl
    println("\n========== dp = $dp, N = $N ==========")

    ## --- density ---
    rho0v = copy(st.rho)
    rt = KernelAbstractions.zeros(gpu, T, N)
    dvars = [
        ("D0 現状",        () -> SPHAdjoint.density_kernel!(gpu)(rt, st.X, cl.starts, cl.counts, cl.order, p.h, p.m, cs, cl.nx, cl.ny; ndrange = N)),
        ("D1 inbounds",    () -> density_D1!(gpu)(rt, st.X, cl.starts, cl.counts, cl.order, p.h, p.m, cs, cl.nx, cl.ny; ndrange = N)),
        ("D2 +早期棄却",   () -> density_D2!(gpu)(rt, st.X, cl.starts, cl.counts, cl.order, p.h, p.m, cs, cl.nx, cl.ny; ndrange = N)),
        ("D3 +Int32",      () -> density_D3!(gpu)(rt, st.X, cl.starts, cl.counts, cl.order, p.h, p.m, cs, cl.nx, cl.ny; ndrange = N)),
    ]
    t0d = 0.0
    println("--- density ---")
    for (name, f) in dvars
        t = best_time(f, gpu)
        f(); KernelAbstractions.synchronize(gpu)
        e = relerr(Array(rt), Array(rho0v))
        t0d = name[1:2] == "D0" ? t : t0d
        @printf("  %-14s %8.1f μs  %5.2fx  rel %.1e\n", name, t * 1e6, t0d / t, e)
    end

    ## --- accel ---
    at = KernelAbstractions.zeros(gpu, T, 2, N)
    SPHAdjoint.accel_kernel!(gpu)(at, st.X, st.V, st.rho, theta, cl.starts, cl.counts,
                                  cl.order, p, cs, cl.nx, cl.ny; ndrange = N)
    KernelAbstractions.synchronize(gpu)
    a0v = copy(at)
    pterm = KernelAbstractions.zeros(gpu, T, N)
    invrho = KernelAbstractions.zeros(gpu, T, N)
    c2 = p.c^2
    avars = [
        ("A0 現状",       () -> SPHAdjoint.accel_kernel!(gpu)(at, st.X, st.V, st.rho, theta, cl.starts, cl.counts, cl.order, p, cs, cl.nx, cl.ny; ndrange = N)),
        ("A1 inbounds",   () -> accel_A1!(gpu)(at, st.X, st.V, st.rho, theta, cl.starts, cl.counts, cl.order, p, cs, cl.nx, cl.ny; ndrange = N)),
        ("A2 +棄却+除算", () -> begin
            eos_kernel!(gpu)(pterm, invrho, st.rho, c2, p.rho0; ndrange = N)
            accel_A2!(gpu)(at, st.X, st.V, pterm, invrho, theta, cl.starts, cl.counts, cl.order, p, cs, cl.nx, cl.ny; ndrange = N)
        end),
        ("A3 +Int32",     () -> begin
            eos_kernel!(gpu)(pterm, invrho, st.rho, c2, p.rho0; ndrange = N)
            accel_A3!(gpu)(at, st.X, st.V, pterm, invrho, theta, cl.starts, cl.counts, cl.order, p, cs, cl.nx, cl.ny; ndrange = N)
        end),
    ]
    t0a = 0.0
    println("--- accel（A2 は eos 前計算込み）---")
    for (name, f) in avars
        t = best_time(f, gpu)
        f(); KernelAbstractions.synchronize(gpu)
        e = relerr(Array(at), Array(a0v))
        t0a = name[1:2] == "A0" ? t : t0a
        @printf("  %-14s %8.1f μs  %5.2fx  rel %.1e\n", name, t * 1e6, t0a / t, e)
    end

    ## --- ワークグループサイズ（最良変種で。このカーネルのレジスタ使用量では
    ##     Metal の上限が 512。1024 は "should not exceed 512" で蹴られる） ---
    println("--- ワークグループサイズ（D3 / A3, デフォルト vs 固定）---")
    for wg in (64, 128, 256, 512)
        td = best_time(gpu) do
            density_D3!(gpu, wg)(rt, st.X, cl.starts, cl.counts, cl.order, p.h, p.m, cs, cl.nx, cl.ny; ndrange = N)
        end
        ta = best_time(gpu) do
            eos_kernel!(gpu, wg)(pterm, invrho, st.rho, c2, p.rho0; ndrange = N)
            accel_A3!(gpu, wg)(at, st.X, st.V, pterm, invrho, theta, cl.starts, cl.counts, cl.order, p, cs, cl.nx, cl.ny; ndrange = N)
        end
        @printf("  wg=%-5d  density %8.1f μs   accel %8.1f μs\n", wg, td * 1e6, ta * 1e6)
    end
end
