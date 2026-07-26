# ---------------------------------------------------------------------------
# Wendland C2 カーネル（2D）
#
#   q = r/h,  A = 7/(4π h²)
#   W(q)        = A (1-q/2)⁴ (2q+1)
#   ∇_i W_ij    = F(q) d_ij          d_ij = x_i - x_j
#   F(q)        = -5A (1-q/2)³ / h²
#   dF/dq       = G(q) = (15A/2) (1-q/2)² / h²
#
# ∇W を「F(q)·d」の形で持つのが要点。r で割る操作が消えるので r→0 の特異性が
# 無く、随伴側でも 0/0 を踏まない。密度総和の随伴に現れる ∂W/∂x も同じ F で
# 書けるため、前進で使う量をそのまま再利用できる。
# ---------------------------------------------------------------------------

@inline wnorm(h::T) where {T} = T(7) / (T(4) * T(π) * h^2)

@inline function w_kern(q::T, A::T) where {T}
    if q < T(2)
        u = one(T) - q / 2
        return A * u^4 * (2q + one(T))
    end
    return zero(T)
end

"""∇_i W_ij = F(q) * (x_i - x_j)"""
@inline function f_kern(q::T, A::T, h::T) where {T}
    if q < T(2)
        u = one(T) - q / 2
        return -5 * A * u^3 / h^2
    end
    return zero(T)
end

"""dF/dq"""
@inline function g_kern(q::T, A::T, h::T) where {T}
    if q < T(2)
        u = one(T) - q / 2
        return T(7.5) * A * u^2 / h^2
    end
    return zero(T)
end
