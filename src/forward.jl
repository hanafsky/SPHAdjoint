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
#
# ■ ペアループの書き方（scripts/07_kernel_tuning.jl の実測に基づく）
#   Metal (M2, N=60000) で density 1.6 倍 / accel 2.6 倍を積み上げた 4 点：
#   1. `inbounds = true`   デバイス側境界チェックの除去（+30%）。
#   2. r² で早期棄却        3×3 セルの走査候補のうち近傍は約 35% しかない。
#      sqrt と除算を棄却の**後**に置き、候補全員に払わない。
#   3. 除算の前計算         p_i/ρ_i², 1/ρ_i は粒子ごとに 1 回（eos_kernel!）。
#      ペアループ内は乗算のみにする。
#   4. セル走査を Int32     Metal は 64bit 整数をエミュレーションする。
#      セル添字演算を Int32 に落とす（N < 2^31 が前提。粒子数 20 億は当面来ない）。
# ---------------------------------------------------------------------------

@kernel inbounds = true function density_kernel!(rho, @Const(X), @Const(nbcount),
                                                 @Const(indices), h, mass, sm, si)
    i = @index(Global)
    T = eltype(rho)
    A = wnorm(T(h))
    invh = one(T) / T(h)
    r2max = T(4) * T(h) * T(h)
    sm32 = Int32(sm); si32 = Int32(si)
    xi = X[1, i]
    yi = X[2, i]

    acc = zero(T)
    for m in Int32(0):(nbcount[i]-Int32(1))
        j = indices[nl_index(m, Int32(i), sm32, si32)]
        dx = xi - X[1, j]
        dy = yi - X[2, j]
        r2 = dx * dx + dy * dy
        # 近傍リストは skin 込みで作ってあるので、ここで真のサポートに絞る。
        # この判定があるおかげで、リストが上位集合でありさえすれば毎ステップ
        # 構築した場合と厳密に同じ相互作用集合になる。
        if r2 < r2max
            # q < 2 が保証されるので w_kern の分岐なし版をインライン。
            # ここの 2 点はどちらも実測で確認した高速化（N=60000 の density）:
            # ・`u^4` と書かない。Float^Int は補正付きの pow_body に落ちて
            #   乗算 2 回の 2.25 倍のコストになる（`^2`/`^3` は乗算に展開
            #   されるので無害、`^4` 以上が罠）。
            # ・sqrt だけ fastmath（+14%）。Metal の精密 Float32 sqrt は
            #   補正付き命令列、fast 版は HW 命令 1 発。誤差は数 ulp で
            #   GPU 非決定性の床と同水準。CPU のネイティブ fsqrt は元々
            #   IEEE 準拠なので CPU 側の結果は変わらない。
            # accel 側は gather 律速で sqrt を速くしても効かない（実測
            # 0.98〜1.06x）ため IEEE のままにしてある。
            q = (@fastmath sqrt(r2)) * invh
            u = one(T) - T(0.5) * q
            u2 = u * u
            acc += T(mass) * (A * u2 * u2 * (2 * q + one(T)))
        end
    end
    # 近傍リストは自己を含まないので、自己項をここで足す
    rho[i] = acc + T(mass) * A
end

"""EOS とペアループ用の前計算。`pterm = p/ρ²`, `invrho = 1/ρ`。"""
@kernel inbounds = true function eos_kernel!(pterm, invrho, @Const(rho), c2, rho0)
    i = @index(Global)
    T = eltype(pterm)
    ri = rho[i]
    inv = one(T) / ri
    invrho[i] = inv
    pterm[i] = T(c2) * (ri - T(rho0)) * inv * inv
end

@kernel inbounds = true function accel_kernel!(a, alpha, @Const(X), @Const(V), @Const(pterm),
                                               @Const(invrho), @Const(theta),
                                               @Const(nbcount), @Const(indices), p, sm, si)
    i = @index(Global)
    T = eltype(a)
    h = p.h
    A = wnorm(h)
    mass = p.m
    invh = one(T) / h
    r2max = T(4) * h * h
    Fc = -5 * A * invh * invh          # f_kern(q) = Fc·u³
    twomumass = 2 * mass * p.mu
    sm32 = Int32(sm); si32 = Int32(si)

    xi = X[1, i]; yi = X[2, i]
    vxi = V[1, i]; vyi = V[2, i]
    pti = pterm[i]
    iri = invrho[i]

    ax = zero(T)
    ay = zero(T)
    for m in Int32(0):(nbcount[i]-Int32(1))
        j = indices[nl_index(m, Int32(i), sm32, si32)]
        dx = xi - X[1, j]
        dy = yi - X[2, j]
        r2 = dx * dx + dy * dy
        # 自己と重心一致粒子は近傍リスト構築時に除外済み。ここはサポート境界のみ。
        if r2 < r2max
            q = sqrt(r2) * invh
            u = one(T) - T(0.5) * q
            F = Fc * u * u * u

            # 圧力（pterm = p/ρ² を前計算済み）
            Pij = mass * (pti + pterm[j])
            ax -= Pij * F * dx
            ay -= Pij * F * dy

            # 粘性（Morris）: F < 0 なので相対速度に対して散逸的
            C = twomumass * iri * invrho[j] * F
            ax += C * (vxi - V[1, j])
            ay += C * (vyi - V[2, j])
        end
    end

    # Brinkman 抗力はここに入れない。速度更新を陰的にするため、係数だけ返して
    # integrate_kernel! で `v' = (v + dt a) / (1 + dt α)` として解く。
    # 陽解法だと dt < 2/α_max の制限があり、固体を硬くしたくて α_max を上げると
    # すぐ発散していた。陰的にすれば無条件安定。
    al, _, _ = interp_alpha(theta, xi, yi, p)
    alpha[i] = al
    ax += p.gx + wall_accel(xi, p.Lx, p.kw)
    ay += p.gy + wall_accel(yi, p.Ly, p.kw)

    a[1, i] = ax
    a[2, i] = ay
end

# semi-implicit Euler。Brinkman 抗力だけ陰的に解く:
#   v' = (v + dt a_rest) / (1 + dt α)
# 陽解法（v' = v + dt(a_rest - α v)）は dt < 2/α_max の制限があるが、これは無条件安定。
@kernel inbounds = true function integrate_kernel!(X, V, @Const(a), @Const(alpha), dt)
    i = @index(Global)
    T = eltype(X)
    D = one(T) / (one(T) + dt * alpha[i])
    v1 = (V[1, i] + dt * a[1, i]) * D
    v2 = (V[2, i] + dt * a[2, i]) * D
    V[1, i] = v1
    V[2, i] = v2
    X[1, i] += dt * v1
    X[2, i] += dt * v2
end

"""
    step!(st, theta, p, backend)

semi-implicit Euler を 1 ステップ。V を先に更新し、その新しい V で X を進める。

!!! note "同期しない"
    カーネルは順序どおりキューに積まれるだけで、`synchronize` は呼ばない
    （理由は `build!` の docstring を参照）。`step!` を回したあとホスト側で
    `Array(st.X)` すればそこが同期点になるので、通常は明示的な同期は要らない。
    生ポインタを触るなど自前で同期が必要な場合だけ
    `KernelAbstractions.synchronize(backend)` を呼ぶこと。
"""
function step!(st, theta, p::SPHParams{T}, backend) where {T}
    N = size(st.X, 2)
    # 近傍リストは skin のぶんだけ余裕を持たせてあるので毎ステップは組み直さない
    # （判定はホスト側のカウンタだけなので同期は入らない）
    maybe_rebuild!(st.nl, st.cl, st.X, p, backend)
    density_kernel!(backend)(st.rho, st.X, st.nl.counts, st.nl.indices,
                             p.h, p.m, st.nl.sm, st.nl.si; ndrange = N)
    eos_kernel!(backend)(st.pterm, st.invrho, st.rho, p.c^2, p.rho0; ndrange = N)
    accel_kernel!(backend)(st.a, st.alpha, st.X, st.V, st.pterm, st.invrho, theta,
                           st.nl.counts, st.nl.indices, p, st.nl.sm, st.nl.si; ndrange = N)
    integrate_kernel!(backend)(st.X, st.V, st.a, st.alpha, p.dt; ndrange = N)
    return st
end
