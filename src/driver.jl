# ---------------------------------------------------------------------------
# 前進・随伴のドライバ
# ---------------------------------------------------------------------------

mutable struct State{AT,VT,CL}
    X::AT       # (2, N)
    V::AT
    a::AT
    rho::VT     # (N,)
    cl::CL
end

function State(backend, X0::AbstractMatrix, V0::AbstractMatrix, p::SPHParams{T}) where {T}
    N = size(X0, 2)
    X = KernelAbstractions.allocate(backend, T, 2, N)
    V = KernelAbstractions.allocate(backend, T, 2, N)
    a = KernelAbstractions.zeros(backend, T, 2, N)
    rho = KernelAbstractions.zeros(backend, T, N)
    copyto!(X, T.(X0))
    copyto!(V, T.(V0))
    return State(X, V, a, rho, CellList(backend, N, p))
end

"""
    simulate!(st, theta, p, backend, nsteps; tape=nothing)

`nsteps` ステップ前進する。`tape` に `Vector` を渡すと各ステップ**開始前**の
`(X, V)` のコピーを積む（随伴に必要）。

!!! note "メモリ"
    全ステップ保存なので、N 粒子 × nsteps ステップで `4 * 2 * N * nsteps` バイト
    （Float32）。N=10⁴, nsteps=10³ で約 320 MB。これを超える規模では
    「k ステップごとにチェックポイントし、逆行時に前進を再実行する」
    2 段チェックポイントに切り替えること（diffSPH も数百ステップが実用上限）。
"""
function simulate!(st::State, theta, p::SPHParams, backend, nsteps::Integer; tape = nothing)
    for _ in 1:nsteps
        if tape !== nothing
            push!(tape, (copy(st.X), copy(st.V)))
        end
        step!(st, theta, p, backend)
    end
    return st
end

struct AdjointWorkspace{AT,VT,MT}
    gX::AT
    gV::AT
    gXs::AT
    gVs::AT
    abar::AT
    grho::VT
    galpha::VT
    gtheta::MT
end

function AdjointWorkspace(backend, N::Integer, p::SPHParams{T}) where {T}
    z2() = KernelAbstractions.zeros(backend, T, 2, N)
    z1() = KernelAbstractions.zeros(backend, T, N)
    return AdjointWorkspace(z2(), z2(), z2(), z2(), z2(), z1(), z1(),
                            KernelAbstractions.zeros(backend, T, p.ngy, p.ngx))
end

"""
    backward!(ws, tape, theta, p, backend; seedX, seedV)

終端の随伴 `seedX = ∂J/∂X_T`, `seedV = ∂J/∂V_T` を与えて時間を逆行し、
`ws.gX`（=∂J/∂X_0）、`ws.gV`（=∂J/∂V_0）、`ws.gtheta`（=∂J/∂θ）を埋める。
"""
function backward!(ws::AdjointWorkspace, tape, theta, p::SPHParams{T}, backend;
                   seedX, seedV) where {T}
    N = size(ws.gX, 2)
    copyto!(ws.gX, seedX)
    copyto!(ws.gV, seedV)
    fill!(ws.gtheta, zero(T))

    scratch = State(backend, zeros(T, 2, N), zeros(T, 2, N), p)

    for n in length(tape):-1:1
        Xn, Vn = tape[n]
        copyto!(scratch.X, Xn)
        copyto!(scratch.V, Vn)

        # 前進で使った近傍リストと密度を再構成（テープには X, V しか積まない）
        build!(scratch.cl, scratch.X, p, backend)
        cs = T(scratch.cl.cs)
        density_kernel!(backend)(scratch.rho, scratch.X, scratch.cl.starts,
                                 scratch.cl.counts, scratch.cl.order,
                                 p.h, p.m, cs, scratch.cl.nx, scratch.cl.ny; ndrange = N)
        KernelAbstractions.synchronize(backend)

        # X' = X + dt V'   →  ḡV' = ḡV + dt ḡX
        _adj_integrate!(backend)(ws.gV, ws.gX, p.dt; ndrange = N)
        KernelAbstractions.synchronize(backend)
        # V' = V + dt a    →  ā = dt ḡV'
        _scale2!(backend)(ws.abar, ws.gV, p.dt; ndrange = N)
        KernelAbstractions.synchronize(backend)

        adj_pass1_kernel!(backend)(ws.gXs, ws.gVs, ws.grho, ws.galpha,
                                   ws.abar, scratch.X, scratch.V, scratch.rho, theta,
                                   scratch.cl.starts, scratch.cl.counts, scratch.cl.order,
                                   p, cs, scratch.cl.nx, scratch.cl.ny; ndrange = N)
        KernelAbstractions.synchronize(backend)

        adj_design_kernel!(backend)(ws.gtheta, ws.gXs, ws.galpha, scratch.X, theta, p;
                                    ndrange = N)
        KernelAbstractions.synchronize(backend)

        adj_pass2_kernel!(backend)(ws.gXs, ws.grho, scratch.X, scratch.cl.starts,
                                   scratch.cl.counts, scratch.cl.order,
                                   p.h, p.m, cs, scratch.cl.nx, scratch.cl.ny; ndrange = N)
        KernelAbstractions.synchronize(backend)

        _axpy2!(backend)(ws.gX, ws.gXs; ndrange = N)
        _axpy2!(backend)(ws.gV, ws.gVs; ndrange = N)
        KernelAbstractions.synchronize(backend)
    end
    return ws
end

# --- 目的関数の例 ----------------------------------------------------------

"""
    target_objective(X, xc, yc, sigma)

「目標領域にどれだけ水が届いたか」を測る平滑な目的関数
`J = -Σ_i exp(-|x_i - x_c|²/(2σ²))`（最小化すると水が集まる）と、
その終端随伴 `∂J/∂X_T` を返す。自由表面がある問題では、こうした
積分量・平滑量を目的にしないと勾配がギザギザになる。
"""
function target_objective(X::AbstractMatrix{T}, xc, yc, sigma) where {T}
    Xh = Array(X)
    N = size(Xh, 2)
    seed = zeros(T, 2, N)
    J = zero(T)
    s2 = T(sigma)^2
    for i in 1:N
        dx = Xh[1, i] - T(xc)
        dy = Xh[2, i] - T(yc)
        e = exp(-(dx * dx + dy * dy) / (2 * s2))
        J -= e
        seed[1, i] = e * dx / s2
        seed[2, i] = e * dy / s2
    end
    return J, seed
end
