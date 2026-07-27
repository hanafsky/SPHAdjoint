# ---------------------------------------------------------------------------
# 手書き離散随伴
#
# ■ なぜ手書きか
#   Enzyme.jl の GPU 対応は事実上 CUDA 中心で、Metal 上を通すのは現状かなり険しい。
#   一方 SPH の随伴は、対称な近傍関係を使うと「すべて gather」に書き直せる。
#   gather に落ちれば scatter 用の浮動小数アトミックが要らず、前進カーネルと
#   まったく同じ KernelAbstractions のコードが Metal でもそのまま走る。
#
# ■ gather 化の要点
#   随伴には x̄_j += ... のような近傍への書き戻しが現れるが、近傍関係は対称
#   （j が i の近傍 ⟺ i が j の近傍）なので、スレッド i が自分の近傍 j を
#   舐めながら「i が近傍側を演じる分」も一緒に足せる。ペアごとの寄与を
#     da   = ā_i - ā_j
#     ad_d = (ā_j - ā_i)·d_ij
#     ad_v = (ā_i - ā_j)·(v_i - v_j)
#   の形にまとめると、すべての項が i と j のデータだけで閉じる。
#
# ■ 検証
#   同じ式を NumPy で実装し、PyTorch の自動微分と相対誤差 1e-15、
#   中心差分と 9 桁一致することを確認済み（verify_adjoint.py）。
#
# 唯一 scatter が残るのは設計変数 θ への書き戻し（粒子 → 格子節点）で、
# ここだけ Atomix のアトミック加算を使う。
# ---------------------------------------------------------------------------

# 前進側と同じチューニングを適用してある（inbounds / r² 早期棄却 / 除算の前計算 /
# Int32 セル走査。詳細は src/forward.jl 冒頭のコメントと scripts/07_kernel_tuning.jl）。
# pterm = p/ρ², invrho = 1/ρ は前進と同じ eos_kernel! で作る。
@kernel inbounds = true function adj_pass1_kernel!(gX, gV, grho, galpha,
                                   @Const(abar), @Const(X), @Const(V),
                                   @Const(pterm), @Const(invrho),
                                   @Const(theta), @Const(nbcount), @Const(indices),
                                   p, sm, si)
    i = @index(Global)
    T = eltype(gX)
    h = p.h
    A = wnorm(h)
    mass = p.m
    c2 = p.c^2
    invh = one(T) / h
    r2max = T(4) * h * h
    Fc = -5 * A * invh * invh              # f_kern(q) = Fc·u³
    Gc = T(7.5) * A * invh * invh          # g_kern(q) = Gc·u²
    twomumass = 2 * mass * p.mu
    sm32 = Int32(sm); si32 = Int32(si)

    xi = X[1, i]; yi = X[2, i]
    vxi = V[1, i]; vyi = V[2, i]
    abx = abar[1, i]; aby = abar[2, i]
    pti = pterm[i]
    iri = invrho[i]

    gx = zero(T); gy = zero(T)
    gvx = zero(T); gvy = zero(T)
    grh = zero(T)
    sumS = zero(T)

    for m in Int32(0):(nbcount[i]-Int32(1))
        j = indices[nl_index(m, Int32(i), sm32, si32)]
        dx = xi - X[1, j]
        dy = yi - X[2, j]
        r2 = dx * dx + dy * dy
        if r2 < r2max
            r = sqrt(r2)
            q = r * invh
            u = one(T) - T(0.5) * q
            F = Fc * u * u * u
            G = Gc * u * u

            Pij = mass * (pti + pterm[j])
            Cbase = twomumass * iri * invrho[j]
            C = Cbase * F

            dax = abx - abar[1, j]
            day = aby - abar[2, j]
            dvx = vxi - V[1, j]
            dvy = vyi - V[2, j]

            ad_d = -(dax * dx + day * dy)  # (ā_j - ā_i)·d_ij
            ad_v = dax * dvx + day * dvy   # (ā_i - ā_j)·(v_i - v_j)

            # 圧力項の d_ij 経由（i 側と j 側をまとめて）
            gx -= Pij * F * dax
            gy -= Pij * F * day

            # F_ij 経由: ∂F/∂x_i = G(q) d_ij/(h r)
            Fbar = Pij * ad_d + Cbase * ad_v
            w = Fbar * G * invh / r
            gx += w * dx
            gy += w * dy

            # 粘性項の速度依存
            gvx += C * dax
            gvy += C * day

            # 粘性項の ρ_i 依存
            grh -= C * ad_v * iri

            # 圧力項の P_ij 経由（p̄ と ρ̄ にあとでまとめて配る）
            sumS += F * ad_d
        end
    end

    # P_ij = m(p_i/ρ_i² + p_j/ρ_j²) 経由の ρ̄ と、p = c²(ρ-ρ0) 経由の ρ̄
    #   pi_/ρ_i³ = pterm_i·(1/ρ_i),  1/ρ_i² = (1/ρ_i)²
    grh += (-2 * mass * pti * iri) * sumS
    grh += c2 * mass * iri * iri * sumS

    # Brinkman 抗力
    alpha, _, _ = interp_alpha(theta, xi, yi, p)
    gvx -= alpha * abx
    gvy -= alpha * aby
    galpha[i] = -(abx * vxi + aby * vyi)

    # 壁
    gx += abx * wall_daccel(xi, p.Lx, p.kw)
    gy += aby * wall_daccel(yi, p.Ly, p.kw)

    gX[1, i] = gx
    gX[2, i] = gy
    gV[1, i] = gvx
    gV[2, i] = gvy
    grho[i] = grh
end

# 密度総和の随伴:  ∂ρ_i/∂x_i = Σ_j m F_ij d_ij,  ∂ρ_j/∂x_i = m F_ij d_ij
#   ⇒ x̄_i += Σ_j m F_ij d_ij (ρ̄_i + ρ̄_j)      （完全に gather）
@kernel inbounds = true function adj_pass2_kernel!(gX, @Const(grho), @Const(X),
                                   @Const(nbcount), @Const(indices), h, mass, sm, si)
    i = @index(Global)
    T = eltype(gX)
    A = wnorm(T(h))
    invh = one(T) / T(h)
    r2max = T(4) * T(h) * T(h)
    Fc = -5 * A * invh * invh
    sm32 = Int32(sm); si32 = Int32(si)
    xi = X[1, i]; yi = X[2, i]
    gri = grho[i]

    gx = zero(T); gy = zero(T)
    for m in Int32(0):(nbcount[i]-Int32(1))
        j = indices[nl_index(m, Int32(i), sm32, si32)]
        dx = xi - X[1, j]
        dy = yi - X[2, j]
        r2 = dx * dx + dy * dy
        if r2 < r2max
            # sqrt だけ fastmath（理由は forward.jl の density_kernel! を参照。
            # このカーネルは density と同じ sqrt+多項式構造なので同様に効く）
            q = (@fastmath sqrt(r2)) * invh
            u = one(T) - T(0.5) * q
            w = T(mass) * (Fc * u * u * u) * (gri + grho[j])
            gx += w * dx
            gy += w * dy
        end
    end
    gX[1, i] += gx
    gX[2, i] += gy
end

# 設計変数場への書き戻し。ここだけ scatter なのでアトミックを使う。
@kernel inbounds = true function adj_design_kernel!(gtheta, gX, @Const(galpha), @Const(X),
                                    @Const(theta), p)
    i = @index(Global)
    T = eltype(gX)
    xi = X[1, i]; yi = X[2, i]
    ga = galpha[i]

    _, dadx, dady = interp_alpha(theta, xi, yi, p)
    gX[1, i] += ga * dadx
    gX[2, i] += ga * dady

    ii, jj, fx, fy, _, _ = interp_cell(xi, yi, p)
    # 多次元添字のアトミックは実装差があるので線形添字に落とす（列優先）
    ny = size(gtheta, 1)
    l00 = jj + (ii - 1) * ny
    Atomix.@atomic gtheta[l00] += ga * (1 - fx) * (1 - fy)
    Atomix.@atomic gtheta[l00+ny] += ga * fx * (1 - fy)
    Atomix.@atomic gtheta[l00+1] += ga * (1 - fx) * fy
    Atomix.@atomic gtheta[l00+ny+1] += ga * fx * fy
end

@kernel inbounds = true function _axpy2!(dst, @Const(src))
    i = @index(Global)
    dst[1, i] += src[1, i]
    dst[2, i] += src[2, i]
end

@kernel inbounds = true function _adj_integrate!(gV, @Const(gX), dt)
    i = @index(Global)
    gV[1, i] += dt * gX[1, i]
    gV[2, i] += dt * gX[2, i]
end

@kernel inbounds = true function _scale2!(dst, @Const(src), s)
    i = @index(Global)
    dst[1, i] = s * src[1, i]
    dst[2, i] = s * src[2, i]
end
