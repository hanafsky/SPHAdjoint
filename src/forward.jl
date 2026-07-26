# ---------------------------------------------------------------------------
# 前進計算
#
#   密度総和      ρ_i = Σ_j m W(q_ij)                （自己項を含む）
#   線形状態方程式 p_i = c² (ρ_i - ρ0)
#   圧力加速度    a^p_i = -Σ_j m (p_i/ρ_i² + p_j/ρ_j²) F_ij d_ij
#   Morris 粘性   a^v_i = Σ_j m (2μ/(ρ_i ρ_j)) F_ij (v_i - v_j)
#   重力          g
#   Brinkman 抗力 -α(x_i) v_i                        ← 設計変数
#   平滑壁        kw relu(penetration)²
#
# 状態量は (2, N) の列優先レイアウト。GPU で座標がコアレスするようこの向きにしている。
# ---------------------------------------------------------------------------

@kernel function density_kernel!(rho, @Const(X), @Const(starts), @Const(counts),
                                 @Const(order), h, mass, cs, nx, ny)
    i = @index(Global)
    T = eltype(rho)
    A = wnorm(T(h))
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
                acc += T(mass) * w_kern(r / T(h), A)
            end
        end
    end
    rho[i] = acc
end

@kernel function accel_kernel!(a, @Const(X), @Const(V), @Const(rho), @Const(theta),
                               @Const(starts), @Const(counts), @Const(order),
                               p, cs, nx, ny)
    i = @index(Global)
    T = eltype(a)
    h = p.h
    A = wnorm(h)
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
                F = f_kern(q, A, h)

                rhoj = rho[j]
                pj = c2 * (rhoj - p.rho0)
                Pij = mass * (pi_ / rhoi^2 + pj / rhoj^2)

                # 圧力
                ax -= Pij * F * dx
                ay -= Pij * F * dy

                # 粘性（Morris）: F < 0 なので相対速度に対して散逸的
                C = 2 * mass * p.mu / (rhoi * rhoj) * F
                ax += C * (vxi - V[1, j])
                ay += C * (vyi - V[2, j])
            end
        end
    end

    alpha, _, _ = interp_alpha(theta, xi, yi, p)
    ax += p.gx - alpha * vxi + wall_accel(xi, p.Lx, p.kw)
    ay += p.gy - alpha * vyi + wall_accel(yi, p.Ly, p.kw)

    a[1, i] = ax
    a[2, i] = ay
end

@kernel function integrate_kernel!(X, V, @Const(a), dt)
    i = @index(Global)
    v1 = V[1, i] + dt * a[1, i]
    v2 = V[2, i] + dt * a[2, i]
    V[1, i] = v1
    V[2, i] = v2
    X[1, i] += dt * v1
    X[2, i] += dt * v2
end

"""
    step!(st, theta, p, backend)

semi-implicit Euler を 1 ステップ。V を先に更新し、その新しい V で X を進める。
"""
function step!(st, theta, p::SPHParams{T}, backend) where {T}
    N = size(st.X, 2)
    build!(st.cl, st.X, p, backend)
    cs = T(st.cl.cs)
    density_kernel!(backend)(st.rho, st.X, st.cl.starts, st.cl.counts, st.cl.order,
                             p.h, p.m, cs, st.cl.nx, st.cl.ny; ndrange = N)
    KernelAbstractions.synchronize(backend)
    accel_kernel!(backend)(st.a, st.X, st.V, st.rho, theta, st.cl.starts, st.cl.counts,
                           st.cl.order, p, cs, st.cl.nx, st.cl.ny; ndrange = N)
    KernelAbstractions.synchronize(backend)
    integrate_kernel!(backend)(st.X, st.V, st.a, p.dt; ndrange = N)
    KernelAbstractions.synchronize(backend)
    return st
end
