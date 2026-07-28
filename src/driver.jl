# ---------------------------------------------------------------------------
# 前進・随伴のドライバ
# ---------------------------------------------------------------------------

mutable struct State{AT,VT,CL,NL}
    X::AT       # (2, N)
    V::AT
    a::AT
    rho::VT     # (N,)
    pterm::VT   # (N,)  p/ρ²（eos_kernel! の前計算。ペアループから除算を消す）
    invrho::VT  # (N,)  1/ρ
    alpha::VT   # (N,)  Brinkman 抗力係数（陰的な速度更新で使う）
    cl::CL      # セルリスト（セル幅は近傍リストのカットオフに合わせる）
    nl::NL      # 近傍リスト
end

"""
    State(backend, X0, V0, p; skin = 0.2, interval = 4, maxnb = 64)

キーワード引数は近傍リストの設定（[`NeighborList`](@ref) を参照）。
セルリストのセル幅は近傍リストのカットオフ `2h(1+skin)` に合わせて作る
（3×3 走査が成立するにはセル幅がカットオフ以上である必要がある）。
"""
function State(backend, X0::AbstractMatrix, V0::AbstractMatrix, p::SPHParams{T};
               skin = 0.2, interval = 4, maxnb = 64, layout = :auto) where {T}
    N = size(X0, 2)
    X = KernelAbstractions.allocate(backend, T, 2, N)
    V = KernelAbstractions.allocate(backend, T, 2, N)
    a = KernelAbstractions.zeros(backend, T, 2, N)
    z1() = KernelAbstractions.zeros(backend, T, N)
    copyto!(X, T.(X0))
    copyto!(V, T.(V0))
    nl = NeighborList(backend, N, p; skin, interval, maxnb, layout)
    cl = CellList(backend, N, p; cs = nl.rc)
    return State(X, V, a, z1(), z1(), z1(), z1(), cl, nl)
end

# --- テープ ---------------------------------------------------------------
#
# 随伴には各ステップ開始前の (X, V) が要る。以前は毎ステップ `copy(st.X)` で
# 新しい配列を確保して `Vector` に push していたが、GPU では確保コストが
# そのまま効いて Metal で 3.2 倍遅くなっていた（CPU ではほぼ無償）。
# `(2, N, nsteps)` の配列を最初に一本確保し、スライスへ書き込む形にする。

mutable struct Tape{AT}
    X::AT       # (2, N, capacity)
    V::AT
    n::Int      # 実際に積んだステップ数
end

"""
    Tape(backend, N, nsteps, p)

`nsteps` ステップ分の `(X, V)` を保持する領域を**一度だけ**確保する。

!!! note "メモリ"
    `2 * sizeof(T) * 2 * N * nsteps` バイト。Float32 で N=10⁴, nsteps=10³ なら
    約 160 MB（X と V で計 320 MB）。これを超える規模では「k ステップごとに
    チェックポイントし、逆行時に前進を再実行する」2 段チェックポイントに
    切り替えること（diffSPH も数百ステップが実用上限）。
"""
function Tape(backend, N::Integer, nsteps::Integer, ::SPHParams{T}) where {T}
    return Tape(KernelAbstractions.allocate(backend, T, 2, N, nsteps),
                KernelAbstractions.allocate(backend, T, 2, N, nsteps), 0)
end

Base.length(tp::Tape) = tp.n
capacity(tp::Tape) = size(tp.X, 3)

"""積んだ内容を捨てる（確保済みの領域は再利用する）。"""
reset!(tp::Tape) = (tp.n = 0; tp)

@kernel function _tape_store!(dst, @Const(src), k)
    i = @index(Global)
    dst[1, i, k] = src[1, i]
    dst[2, i, k] = src[2, i]
end

@kernel function _tape_load!(dst, @Const(src), k)
    i = @index(Global)
    dst[1, i] = src[1, i, k]
    dst[2, i] = src[2, i, k]
end

"""
    simulate!(st, theta, p, backend, nsteps; tape=nothing)

`nsteps` ステップ前進する。`tape` に [`Tape`](@ref) を渡すと各ステップ
**開始前**の `(X, V)` を積む（随伴に必要）。テープは追記されるので、
同じテープを使い回すときは [`reset!`](@ref) すること。
"""
function simulate!(st::State, theta, p::SPHParams, backend, nsteps::Integer; tape = nothing)
    N = size(st.X, 2)
    if tape !== nothing
        need = tape.n + nsteps
        need <= capacity(tape) || throw(ArgumentError(
            "テープの容量が足りない: $(capacity(tape)) ステップ分しか確保していないが " *
            "$need ステップ必要。Tape(backend, N, nsteps, p) を大きく取り直すか reset! を。"))
    end
    for _ in 1:nsteps
        if tape !== nothing
            tape.n += 1
            _tape_store!(backend)(tape.X, st.X, tape.n; ndrange = N)
            _tape_store!(backend)(tape.V, st.V, tape.n; ndrange = N)
        end
        step!(st, theta, p, backend)
    end
    # 内側のループは一切同期しない（`step!` の docstring 参照）。公開 API の
    # 境界としてここで 1 回だけ待ち、戻った時点で状態が確定しているようにする。
    KernelAbstractions.synchronize(backend)
    # 近傍リストの異常（行溢れ・変位超過）はここでまとめて報告する。
    # 毎ステップ見ると同期が入ってしまうので、境界での事後検出にしている。
    check_neighbor_flags(st.nl)
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
function backward!(ws::AdjointWorkspace, tape::Tape, theta, p::SPHParams{T}, backend;
                   seedX, seedV, scratch = nothing,
                   skin = 0.2, interval = 4, maxnb = 64, layout = :auto) where {T}
    N = size(ws.gX, 2)
    copyto!(ws.gX, seedX)
    copyto!(ws.gV, seedV)
    fill!(ws.gtheta, zero(T))

    # 近傍リストの設定は前進と揃えること。相互作用集合が同一であれば離散随伴は
    # 厳密なままだが、変位の上限を破っている設定では前進と逆行で欠落する近傍が
    # 食い違い、勾配がずれる。
    # `scratch` を渡せば毎回の確保を避けられる（最適化ループでは効く）。
    if scratch === nothing
        scratch = State(backend, zeros(T, 2, N), zeros(T, 2, N), p;
                        skin, interval, maxnb, layout)
    end

    for n in length(tape):-1:1
        _tape_load!(backend)(scratch.X, tape.X, n; ndrange = N)
        _tape_load!(backend)(scratch.V, tape.V, n; ndrange = N)

        # 前進で使った近傍リストと密度・EOS 前計算を再構成
        # （テープには X, V しか積まない）
        #
        # 再構築のタイミングは前進と揃っていなくてよい。物理カーネルが
        # r² < (2h)² で絞るので、リストが上位集合であれば相互作用集合は同一に
        # なり、離散随伴は厳密なまま（詳細は src/neighbors.jl 冒頭のコメント）。
        maybe_rebuild!(scratch.nl, scratch.cl, scratch.X, p, backend)
        density_kernel!(backend)(scratch.rho, scratch.X, scratch.nl.counts,
                                 scratch.nl.indices, p.h, p.m,
                                 scratch.nl.sm, scratch.nl.si; ndrange = N)
        eos_kernel!(backend)(scratch.pterm, scratch.invrho, scratch.rho,
                             p.c^2, p.rho0; ndrange = N)

        # 前進で使った a_rest と α を再現する（テープには X, V しか積まない）
        accel_kernel!(backend)(scratch.a, scratch.alpha, scratch.X, scratch.V,
                               scratch.pterm, scratch.invrho, theta,
                               scratch.nl.counts, scratch.nl.indices, p,
                               scratch.nl.sm, scratch.nl.si; ndrange = N)

        # X' = X + dt V'   →  ḡV' = ḡV + dt ḡX
        _adj_integrate!(backend)(ws.gV, ws.gX, p.dt; ndrange = N)
        # 陰的な抗力の随伴。ā, ᾱ を作り、ḡV を D 倍する（詳細は _adj_drag!）
        _adj_drag!(backend)(ws.abar, ws.gV, ws.galpha, scratch.V, scratch.a,
                            scratch.alpha, p.dt; ndrange = N)

        adj_pass1_kernel!(backend)(ws.gXs, ws.gVs, ws.grho,
                                   ws.abar, scratch.X, scratch.V,
                                   scratch.pterm, scratch.invrho, theta,
                                   scratch.nl.counts, scratch.nl.indices,
                                   p, scratch.nl.sm, scratch.nl.si; ndrange = N)

        adj_design_kernel!(backend)(ws.gtheta, ws.gXs, ws.galpha, scratch.X, theta, p;
                                    ndrange = N)

        adj_pass2_kernel!(backend)(ws.gXs, ws.grho, scratch.X, scratch.nl.counts,
                                   scratch.nl.indices, p.h, p.m,
                                   scratch.nl.sm, scratch.nl.si; ndrange = N)

        _axpy2!(backend)(ws.gX, ws.gXs; ndrange = N)
        _axpy2!(backend)(ws.gV, ws.gVs; ndrange = N)
    end
    # 逆行ループも同期しない（1 ステップに 7 回入っていた）。戻る直前に 1 回だけ。
    KernelAbstractions.synchronize(backend)
    check_neighbor_flags(scratch.nl)
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
