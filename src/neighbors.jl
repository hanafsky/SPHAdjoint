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

"""
    CellList(backend, N, p; cs = support(p))

セル幅 `cs` の一様格子。既定はサポート半径 `2h`。近傍リスト（[`NeighborList`](@ref)）
用に skin を足したカットオフで作るときは `cs` を大きくして渡す。
3×3 セル走査が成立するにはセル幅がカットオフ以上である必要がある。
"""
function CellList(backend, N::Integer, p::SPHParams{T}; cs = Float64(support(p))) where {T}
    cs = Float64(cs)
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

@kernel inbounds = true function _cl_assign!(cellof, counts, @Const(X), cs, nx, ny)
    i = @index(Global)
    c = cell_index(X[1, i], X[2, i], cs, nx, ny)
    cellof[i] = c
    Atomix.@atomic counts[c] += Int32(1)
end

@kernel inbounds = true function _cl_fill!(order, cursor, @Const(cellof))
    i = @index(Global)
    c = cellof[i]
    # Atomix の @atomic x += v は「新しい値」を返す。cursor は排他的オフセットで
    # 初期化してあるので、返り値がそのまま 1-based の格納位置になる。
    pos = (Atomix.@atomic cursor[c] += Int32(1))
    order[pos] = Int32(i)
end

"""
    build!(cl, X, p, backend)

セルリストを組み直す。

!!! note "同期しない"
    ここでは `synchronize` を呼ばない。同一 backend のカーネルと配列演算は
    同じキューに順序どおり積まれるので、ホストが結果を読むまで同期は不要
    （`Array()` / ホストへの `copyto!` 自体が同期点になる）。CPU backend は
    そもそも同期実行で `synchronize(::CPU)` は no-op。

    カーネルごとに同期していた版に比べ、Metal では 1 ステップあたりの
    往復レイテンシが消えて **N=925 で 7.3 倍**速い（`scripts/05_metal_bench.jl`）。
"""
function build!(cl::CellList, X, p::SPHParams{T}, backend) where {T}
    N = size(X, 2)
    fill!(cl.counts, Int32(0))
    _cl_assign!(backend)(cl.cellof, cl.counts, X, T(cl.cs), cl.nx, cl.ny; ndrange = N)

    # 排他的プレフィックス和。GPUArrays が accumulate! を提供していない
    # バックエンドでは、この 3 行を CPU 往復に差し替えればよい（ncell は小さい）。
    cumsum!(cl.cum, cl.counts)
    cl.starts .= cl.cum .- cl.counts
    cl.cursor .= cl.starts

    _cl_fill!(backend)(cl.order, cl.cursor, cl.cellof; ndrange = N)
    return cl
end

# ---------------------------------------------------------------------------
# 近傍リスト（転置・行長固定）
#
# ■ なぜセル走査を直接使わないか
#   セル幅 2h の 3×3 走査は候補 61 個を舐めて実近傍は 21 個（π/9 ≈ 35%）。
#   65% は空振りの距離判定と無駄な gather になる。物理カーネルは前進 2 本 +
#   随伴 2 本の計 4 本あるので、この無駄が 4 回繰り返される。
#
# ■ レイアウトが全て（scripts/09_neighbor_list.jl の実測）
#   素直な CSR（offsets + 可変長 indices）は構築が 1438 μs (N=60000) と、
#   セルリスト 138 μs の 10 倍かかって割に合わなかった。
#   行長を固定して**転置**（粒子 i の m 番目の近傍を `indices[(m-1)*N+i]` に置く）
#   に変えると、
#     - 同じ m で全スレッドが連続アドレスを触るので読み書きがコアレスする
#     - 行長が既知なので count パス・cumsum・offsets が丸ごと不要になる
#   構築 422 μs、density 1.86x、accel 1.87x、1 ステップ総計で約 2 倍。
#
# ■ skin による再利用の正当性
#   物理カーネルは `r² < (2h)²` で絞るので、リストが真の近傍の**上位集合**で
#   ありさえすれば、毎ステップ構築した場合と厳密に同じ相互作用集合になる。
#   カットオフを rc = 2h + s（s = skin·2h）で作ると、各粒子の変位が δ ≤ s/2 の
#   間は上位集合が保たれる（r_ij(n) < 2h ⟹ r_ij(n₀) < 2h + 2δ ≤ rc）。
#   許される再構築間隔は K ≤ skin·h/(dt·v_max)。ダムブレイク（dp=0.012,
#   dt=1.5e-4, v_max≈2.8）では K ≤ 7.4 だったので、既定は余裕を見て 4 にした
#   （K=8 は限界ぎりぎりで実際に警告が出た）。速度を詰めたければ上げてよいが、
#   変位超過の警告が出ないことを必ず確認すること。
#
#   これは**前進と随伴で再構築のタイミングが揃っている必要がない**ことも意味する。
#   相互作用集合が同一なら、計算される関数が同じなので離散随伴は厳密なまま。
# ---------------------------------------------------------------------------

mutable struct NeighborList{IA,AT}
    indices::IA     # (N*maxnb,) Int32  粒子 i の m 番目の近傍 = indices[(m-1)*N+i]
    counts::IA      # (N,) Int32        粒子 i の近傍数
    flags::IA       # (2,) Int32        [1] 行溢れ, [2] 変位超過
    Xref::AT        # (2, N)            前回構築時の座標（変位チェック用）
    maxnb::Int      # 1 粒子あたりの上限
    rc::Float64     # カットオフ 2h(1+skin)
    dmax::Float64   # 許される変位 = skin·h
    interval::Int   # 再構築間隔
    age::Int        # 前回構築からのステップ数
end

"""
    NeighborList(backend, N, p; skin = 0.2, interval = 4, maxnb = 64)

`skin` はサポート半径に対する余裕の比、`interval` は再構築間隔。
`maxnb` は 1 粒子あたりの近傍数の上限（溢れたら `flags[1]` が立ち、
`simulate!` / `backward!` の終わりで警告される）。

既定値の根拠は上のコメントを参照。2D・dp 間隔の一様配置なら skin=0.2 で
近傍数は平均 28 / 最大 32 程度なので maxnb=64 は十分な余裕がある。
"""
function NeighborList(backend, N::Integer, p::SPHParams{T};
                      skin = 0.2, interval = 4, maxnb = 64) where {T}
    rc = Float64(support(p)) * (1 + skin)
    dmax = Float64(p.h) * skin
    return NeighborList(KernelAbstractions.zeros(backend, Int32, N * maxnb),
                        KernelAbstractions.zeros(backend, Int32, N),
                        KernelAbstractions.zeros(backend, Int32, 2),
                        KernelAbstractions.zeros(backend, T, 2, N),
                        Int(maxnb), rc, dmax, Int(interval), typemax(Int))
end

@kernel inbounds = true function _nl_build!(indices, nbcount, flags, Xref,
                                            @Const(X), @Const(starts), @Const(counts),
                                            @Const(order), rc2, r2min, cs, nx, ny,
                                            N, maxnb, dmax2, check)
    i = @index(Global)
    T = eltype(X)
    invcs = one(T) / T(cs)
    nx32 = Int32(nx); ny32 = Int32(ny); N32 = Int32(N); mx = Int32(maxnb)
    xi = X[1, i]
    yi = X[2, i]

    # 前回構築からの変位を検査してから Xref を更新する。ここで見ているのは
    # 「区間の最後の時点での変位」なので、超過は事後検出になる（同期を毎ステップ
    # 入れないための割り切り。interval を CFL から保守的に取れば起きない）。
    if check
        ddx = xi - Xref[1, i]
        ddy = yi - Xref[2, i]
        if ddx * ddx + ddy * ddy > T(dmax2)
            flags[2] = Int32(1)
        end
    end
    Xref[1, i] = xi
    Xref[2, i] = yi

    cx = clamp(unsafe_trunc(Int32, floor(xi * invcs)) + Int32(2), Int32(1), nx32)
    cy = clamp(unsafe_trunc(Int32, floor(yi * invcs)) + Int32(2), Int32(1), ny32)
    m = Int32(0)
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
                # 自己（と重心一致粒子）は r2min で除外。上限は skin 込みの rc。
                if r2min < r2 < rc2
                    if m < mx
                        indices[m*N32+Int32(i)] = j
                        m += Int32(1)
                    else
                        flags[1] = Int32(1)   # 溢れは黙って捨てず記録する
                    end
                end
            end
        end
    end
    nbcount[i] = m
end

"""
    build_neighbors!(nl, cl, X, p, backend)

セルリストを組み直してから近傍リストを構築する。`nl.age` を 0 に戻す。
"""
function build_neighbors!(nl::NeighborList, cl::CellList, X, p::SPHParams{T},
                          backend) where {T}
    N = size(X, 2)
    build!(cl, X, p, backend)
    check = nl.age != typemax(Int)      # 初回は前回座標が無いので検査しない
    _nl_build!(backend)(nl.indices, nl.counts, nl.flags, nl.Xref, X,
                        cl.starts, cl.counts, cl.order,
                        T(nl.rc^2), eps(T) * p.h * p.h, T(cl.cs), cl.nx, cl.ny,
                        N, nl.maxnb, T(nl.dmax^2), check; ndrange = N)
    nl.age = 0
    return nl
end

"""
    maybe_rebuild!(nl, cl, X, p, backend)

間隔に達していたら組み直す。ホスト側のカウンタだけで判断するので同期は要らない。

`interval = 1` が「毎ステップ組み直す」を意味するように、判定してから加算する
（先に加算すると 1 ステップぶんずれて `interval = 1` が 2 ステップに 1 回になる）。
"""
function maybe_rebuild!(nl::NeighborList, cl::CellList, X, p::SPHParams, backend)
    if nl.age >= nl.interval
        build_neighbors!(nl, cl, X, p, backend)   # age = 0 に戻る
    end
    nl.age += 1
    return nl
end

"""`flags` を読んで異常を報告する。ホスト読み出しなのでここで同期が入る。"""
function check_neighbor_flags(nl::NeighborList)
    f = Array(nl.flags)
    f[1] != 0 && @warn "近傍リストの行が溢れた（maxnb=$(nl.maxnb)）。近傍が欠落している。\
                        NeighborList の maxnb を増やすこと。"
    f[2] != 0 && @warn "近傍リストの再構築間隔が長すぎる（変位が skin/2 を超えた）。\
                        近傍が欠落している可能性がある。interval を小さくするか skin を大きくすること。"
    fill!(nl.flags, Int32(0))
    return nothing
end
