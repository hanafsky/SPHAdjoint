# # @fastmath は効くか
#
# 文献調査で「期待値 1 割前後・条件付き」とされた fast math の実測。
#
# 予想としては効果は小さいはず：07 の改修でペアループから除算を消し、
# sqrt も r² 早期棄却の後（実近傍 21 個だけ）に移してあるので、fastmath の
# 主な標的（除算・sqrt の精密版）がもう薄い。残るは FMA 縮約と fast sqrt。
# ただし予想は当てにならない（wg 調整は外れ、Int32 は当たりだった）ので測る。
#
# Julia の `@fastmath` は構文レベルの置き換えなので、カーネル本体の演算部を
# `@fastmath begin ... end` で包む。`interp_alpha` など関数呼び出しの中身には
# 波及しない（そこは別途）。
#
# ```
# julia --project=. scripts/08_fastmath.jl
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

best_time(f, backend; n = 4) = minimum(time_stage(f, backend) for _ in 1:n)

# ## fastmath 版カーネル（現行 src と同一の構造、演算部だけ @fastmath）

@kernel inbounds = true function density_F!(rho, @Const(X), @Const(starts),
                                            @Const(counts), @Const(order),
                                            h, mass, cs, nx, ny)
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
                ## `^` は @fastmath 下で pow_fast になり、Metal のバックエンド
                ## コンパイラ (AGXMetalG14G) が XPC 断で落ちる。乗算に展開して
                ## おき、@fastmath は算術演算だけに絞る。
                dx = xi - X[1, j]
                dy = yi - X[2, j]
                r2 = @fastmath dx * dx + dy * dy
                if r2 < r2max
                    q = @fastmath sqrt(r2) * invh
                    u = one(T) - T(0.5) * q
                    u2 = u * u
                    acc += @fastmath T(mass) * (A * u2 * u2 * (2 * q + one(T)))
                end
            end
        end
    end
    rho[i] = acc
end

@kernel inbounds = true function accel_F!(a, @Const(X), @Const(V), @Const(pterm),
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
                r2 = @fastmath dx * dx + dy * dy
                if r2min < r2 < r2max
                    q = @fastmath sqrt(r2) * invh
                    u = one(T) - T(0.5) * q
                    F = @fastmath Fc * u * u * u
                    Pij = mass * (pti + pterm[j])
                    ax = @fastmath ax - Pij * F * dx
                    ay = @fastmath ay - Pij * F * dy
                    C = @fastmath twomumass * iri * invrho[j] * F
                    ax = @fastmath ax + C * (vxi - V[1, j])
                    ay = @fastmath ay + C * (vyi - V[2, j])
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

# ## 実行

relerr(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), eps(Float32))

function run_case(dp)
    p = make_params(dp)
    X0 = water_column(dp)
    N = size(X0, 2)
    st = State(gpu, X0, zeros(T, size(X0)), p)
    theta = KernelAbstractions.zeros(gpu, T, p.ngy, p.ngx)
    for _ in 1:200
        step!(st, theta, p, gpu)
    end
    build!(st.cl, st.X, p, gpu)
    cl = st.cl
    cs = T(cl.cs)
    SPHAdjoint.density_kernel!(gpu)(st.rho, st.X, cl.starts, cl.counts, cl.order,
                                    p.h, p.m, cs, cl.nx, cl.ny; ndrange = N)
    SPHAdjoint.eos_kernel!(gpu)(st.pterm, st.invrho, st.rho, p.c^2, p.rho0; ndrange = N)
    KernelAbstractions.synchronize(gpu)

    rt = KernelAbstractions.zeros(gpu, T, N)
    at = KernelAbstractions.zeros(gpu, T, 2, N)
    rho_ref = copy(st.rho)
    SPHAdjoint.accel_kernel!(gpu)(at, st.X, st.V, st.pterm, st.invrho, theta,
                                  cl.starts, cl.counts, cl.order, p, cs,
                                  cl.nx, cl.ny; ndrange = N)
    KernelAbstractions.synchronize(gpu)
    a_ref = copy(at)

    td0 = best_time(gpu) do
        SPHAdjoint.density_kernel!(gpu)(rt, st.X, cl.starts, cl.counts, cl.order,
                                        p.h, p.m, cs, cl.nx, cl.ny; ndrange = N)
    end
    td1 = best_time(gpu) do
        density_F!(gpu)(rt, st.X, cl.starts, cl.counts, cl.order,
                        p.h, p.m, cs, cl.nx, cl.ny; ndrange = N)
    end
    density_F!(gpu)(rt, st.X, cl.starts, cl.counts, cl.order,
                    p.h, p.m, cs, cl.nx, cl.ny; ndrange = N)
    KernelAbstractions.synchronize(gpu)
    ed = relerr(Array(rt), Array(rho_ref))

    ta0 = best_time(gpu) do
        SPHAdjoint.accel_kernel!(gpu)(at, st.X, st.V, st.pterm, st.invrho, theta,
                                      cl.starts, cl.counts, cl.order, p, cs,
                                      cl.nx, cl.ny; ndrange = N)
    end
    ta1 = best_time(gpu) do
        accel_F!(gpu)(at, st.X, st.V, st.pterm, st.invrho, theta,
                      cl.starts, cl.counts, cl.order, p, cs, cl.nx, cl.ny; ndrange = N)
    end
    accel_F!(gpu)(at, st.X, st.V, st.pterm, st.invrho, theta,
                  cl.starts, cl.counts, cl.order, p, cs, cl.nx, cl.ny; ndrange = N)
    KernelAbstractions.synchronize(gpu)
    ea = relerr(Array(at), Array(a_ref))

    @printf("N=%-6d density: 現行 %7.1f μs → fastmath %7.1f μs (%.2fx)  rel %.1e\n",
            N, td0 * 1e6, td1 * 1e6, td0 / td1, ed)
    @printf("         accel:   現行 %7.1f μs → fastmath %7.1f μs (%.2fx)  rel %.1e\n",
            ta0 * 1e6, ta1 * 1e6, ta0 / ta1, ea)
    return nothing
end

for dp in (0.003, 0.0015)
    run_case(dp)
end
