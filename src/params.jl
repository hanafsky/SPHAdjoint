# ---------------------------------------------------------------------------
# パラメータと設計変数場
# ---------------------------------------------------------------------------

"""
    SPHParams{T}

2D WCSPH のパラメータ。`T` は `Float64`（CPU 検証用）か `Float32`（Metal 用）。

!!! warning "Metal は Float64 非対応"
    Apple GPU は倍精度を持たない。Metal バックエンドでは必ず `Float32` を使うこと。
    勾配検証は CPU + `Float64` で行う。
"""
Base.@kwdef struct SPHParams{T}
    h::T                      # 平滑化長（サポート半径は 2h）
    m::T                      # 粒子質量
    rho0::T                   # 参照密度
    c::T                      # 数値音速
    mu::T                     # 粘性係数
    gx::T = zero(T)           # 重力 x
    gy::T = T(-9.81)          # 重力 y
    dt::T                     # 時間刻み
    Lx::T                     # 領域 x
    Ly::T                     # 領域 y
    kw::T                     # 壁ペナルティ係数
    ngx::Int                  # 設計格子 節点数 x
    ngy::Int                  # 設計格子 節点数 y
end

@inline support(p::SPHParams) = 2 * p.h
@inline dxg(p::SPHParams) = p.Lx / (p.ngx - 1)
@inline dyg(p::SPHParams) = p.Ly / (p.ngy - 1)

# --- 平滑な壁（relu² ペナルティ。力が C1 なので勾配が連続） -----------------

@inline function wall_accel(x::T, L::T, kw::T) where {T}
    lo = max(zero(T), -x)
    hi = max(zero(T), x - L)
    return kw * (lo^2 - hi^2)
end

@inline function wall_daccel(x::T, L::T, kw::T) where {T}
    lo = max(zero(T), -x)
    hi = max(zero(T), x - L)
    return kw * (-2 * lo - 2 * hi)
end

# --- 設計変数場 θ（背景格子の節点値）の双線形補間 ---------------------------
#
# θ は「Brinkman 抗力係数」。粒子は a_i -= α(x_i) v_i を受ける。
# α が大きい領域＝固体、α≈0＝流路。これで SIMP がそのまま乗る。

"""格子セル添字と補間係数。clip が効いている粒子は座標勾配を 0 にする。"""
@inline function interp_cell(x::T, y::T, p::SPHParams{T}) where {T}
    hx = dxg(p)
    hy = dyg(p)
    gx = x / hx
    gy = y / hy
    ax = ((gx > zero(T)) & (gx < T(p.ngx - 1))) ? one(T) : zero(T)
    ay = ((gy > zero(T)) & (gy < T(p.ngy - 1))) ? one(T) : zero(T)
    gxc = clamp(gx, zero(T), T(p.ngx - 1) - T(1e-6))
    gyc = clamp(gy, zero(T), T(p.ngy - 1) - T(1e-6))
    i0 = clamp(unsafe_trunc(Int, floor(gxc)), 0, p.ngx - 2)
    j0 = clamp(unsafe_trunc(Int, floor(gyc)), 0, p.ngy - 2)
    fx = gxc - T(i0)
    fy = gyc - T(j0)
    return i0 + 1, j0 + 1, fx, fy, ax, ay   # 1-based 添字で返す
end

"""α_i とその空間勾配 (∂α/∂x, ∂α/∂y) を同時に返す。θ は (ngy, ngx)。"""
@inline function interp_alpha(theta, x::T, y::T, p::SPHParams{T}) where {T}
    i, j, fx, fy, ax, ay = interp_cell(x, y, p)
    t00 = theta[j, i]
    t01 = theta[j, i+1]
    t10 = theta[j+1, i]
    t11 = theta[j+1, i+1]
    a = (1 - fx) * (1 - fy) * t00 + fx * (1 - fy) * t01 +
        (1 - fx) * fy * t10 + fx * fy * t11
    dadx = ((1 - fy) * (t01 - t00) + fy * (t11 - t10)) / dxg(p) * ax
    dady = ((1 - fx) * (t10 - t00) + fx * (t11 - t01)) / dyg(p) * ay
    return a, dadx, dady
end
