# ---------------------------------------------------------------------------
# 一様セルリスト（counting sort）
#
# セル幅 = サポート半径 2h。近傍は 3×3 セルを走査すれば足りる。
# 32bit 整数アトミックだけで組んであるので Metal / CUDA / ROCm / CPU すべてで動く
# （Apple GPU は 64bit アトミックを持たないため Int32 固定）。
# ---------------------------------------------------------------------------

mutable struct CellList{IA}
    cellof::IA      # (N,)        粒子 i の属するセル番号
    counts::IA      # (ncell,)    セルごとの粒子数
    cum::IA         # (ncell,)    counts の累積和（作業用）
    starts::IA      # (ncell,)    排他的プレフィックス（0-based オフセット）
    cursor::IA      # (ncell,)    充填用カーソル
    order::IA       # (N,)        セル順に並べ替えた粒子 id
    nx::Int
    ny::Int
    cs::Float64     # セル幅
end

function CellList(backend, N::Integer, p::SPHParams{T}) where {T}
    cs = Float64(support(p))
    # 壁をわずかに越える粒子を吸収するため 1 セル分の余裕を左右に付ける
    nx = max(1, ceil(Int, Float64(p.Lx) / cs) + 2)
    ny = max(1, ceil(Int, Float64(p.Ly) / cs) + 2)
    ncell = nx * ny
    z(n) = KernelAbstractions.zeros(backend, Int32, n)
    return CellList(z(N), z(ncell), z(ncell), z(ncell), z(ncell), z(N), nx, ny, cs)
end

@inline function cell_index(x::T, y::T, cs::T, nx, ny) where {T}
    cx = clamp(unsafe_trunc(Int, floor(x / cs)) + 2, 1, nx)   # +2: 左に 1 セル余裕
    cy = clamp(unsafe_trunc(Int, floor(y / cs)) + 2, 1, ny)
    return Int32(cx + (cy - 1) * nx)
end

@kernel function _cl_assign!(cellof, counts, @Const(X), cs, nx, ny)
    i = @index(Global)
    c = cell_index(X[1, i], X[2, i], cs, nx, ny)
    cellof[i] = c
    Atomix.@atomic counts[c] += Int32(1)
end

@kernel function _cl_fill!(order, cursor, @Const(cellof))
    i = @index(Global)
    c = cellof[i]
    # Atomix の @atomic x += v は「新しい値」を返す。cursor は排他的オフセットで
    # 初期化してあるので、返り値がそのまま 1-based の格納位置になる。
    pos = (Atomix.@atomic cursor[c] += Int32(1))
    order[pos] = Int32(i)
end

function build!(cl::CellList, X, p::SPHParams{T}, backend) where {T}
    N = size(X, 2)
    fill!(cl.counts, Int32(0))
    _cl_assign!(backend)(cl.cellof, cl.counts, X, T(cl.cs), cl.nx, cl.ny; ndrange = N)
    KernelAbstractions.synchronize(backend)

    # 排他的プレフィックス和。GPUArrays が accumulate! を提供していない
    # バックエンドでは、この 3 行を CPU 往復に差し替えればよい（ncell は小さい）。
    cumsum!(cl.cum, cl.counts)
    cl.starts .= cl.cum .- cl.counts
    cl.cursor .= cl.starts

    _cl_fill!(backend)(cl.order, cl.cursor, cl.cellof; ndrange = N)
    KernelAbstractions.synchronize(backend)
    return cl
end
