using SPHAdjoint
using KernelAbstractions
using Test
using Random

const T = Float64
const backend = CPU()

# ---------------------------------------------------------------------------
# 1. カーネル関数の整合性
#    ∇W を F(q)·d の形で持っている前提が本当に正しいかを差分で確認する。
# ---------------------------------------------------------------------------
@testset "カーネル W / F / G の整合性" begin
    h = 0.1
    A = SPHAdjoint.wnorm(h)

    # F(q) = (dW/dr)/r  を差分で検証
    for r in [0.01, 0.05, 0.1, 0.15, 0.19]
        δ = 1e-7
        dWdr = (SPHAdjoint.w_kern((r + δ) / h, A) - SPHAdjoint.w_kern((r - δ) / h, A)) / (2δ)
        @test SPHAdjoint.f_kern(r / h, A, h) ≈ dWdr / r rtol = 1e-6
    end

    # G(q) = dF/dq
    for q in [0.1, 0.5, 1.0, 1.5, 1.9]
        δ = 1e-7
        dFdq = (SPHAdjoint.f_kern(q + δ, A, h) - SPHAdjoint.f_kern(q - δ, A, h)) / (2δ)
        @test SPHAdjoint.g_kern(q, A, h) ≈ dFdq rtol = 1e-6
    end

    # サポート境界で滑らかに 0 に落ちること（勾配が壊れないための条件）
    @test SPHAdjoint.w_kern(2.0, A) == 0
    @test SPHAdjoint.f_kern(2.0, A, h) == 0
    @test SPHAdjoint.g_kern(2.0, A, h) == 0
    @test abs(SPHAdjoint.f_kern(1.999, A, h)) < 1e-6 * abs(SPHAdjoint.f_kern(1.0, A, h))
end

# ---------------------------------------------------------------------------
# 2. セルリストの正しさ
#    counting sort とアトミックが正しく動いているかを、全対全の総当たり計算と
#    突き合わせて確認する。README に挙げた「引っかかりそうな箇所」1〜3 は
#    ここが落ちれば一発で分かる。
# ---------------------------------------------------------------------------
@testset "セルリスト・近傍リスト vs 総当たり" begin
    rng = MersenneTwister(42)
    p = SPHParams{T}(h = 0.06, m = 0.004, rho0 = 1.0, c = 10.0, mu = 0.02,
                     dt = 1e-4, Lx = 1.0, Ly = 0.6, kw = 1e4, ngx = 9, ngy = 7)
    N = 400
    X0 = vcat(rand(rng, T, 1, N) .* p.Lx, rand(rng, T, 1, N) .* p.Ly)

    st = State(backend, X0, zeros(T, 2, N), p)
    build!(st.cl, st.X, p, backend)

    # セルリストが全粒子をちょうど 1 回ずつ含むこと
    ord = sort(Array(st.cl.order))
    @test ord == collect(Int32(1):Int32(N))
    @test sum(Array(st.cl.counts)) == N

    # 近傍リストが「カットオフ内の粒子を過不足なく含む」こと（総当たりと比較）
    build_neighbors!(st.nl, st.cl, st.X, p, backend)
    KernelAbstractions.synchronize(backend)
    Xh = Array(st.X)
    nbc = Array(st.nl.counts)
    idx = Array(st.nl.indices)
    rc2 = T(st.nl.rc^2)
    r2min = eps(T) * p.h * p.h
    ok = true
    for i in 1:N
        got = sort([idx[SPHAdjoint.nl_index(Int32(m-1), Int32(i), Int32(st.nl.sm), Int32(st.nl.si))]
                    for m in 1:nbc[i]])
        want = Int32[]
        for j in 1:N
            r2 = (Xh[1, i] - Xh[1, j])^2 + (Xh[2, i] - Xh[2, j])^2
            (r2min < r2 < rc2) && push!(want, Int32(j))
        end
        got == sort(want) || (ok = false; break)
    end
    @test ok
    @test all(nbc .<= st.nl.maxnb)

    # セル内が粒子 id の昇順に整列していること（GPU の総和順序を決定的にする条件）。
    # CPU では元々ビット再現するのでこのテスト自体は GPU の回帰を直接は捕まえないが、
    # ソートカーネルが壊れれば落ちる。
    ordh = Array(st.cl.order)
    sth = Array(st.cl.starts)
    cnth = Array(st.cl.counts)
    @test all(issorted(@view ordh[sth[c]+1:sth[c]+cnth[c]]) for c in eachindex(cnth))
    @test Array(st.nl.flags) == Int32[0, 0]      # 溢れも変位超過も無い

    # 密度が総当たりと一致すること（自己項を含む）
    SPHAdjoint.density_kernel!(backend)(
        st.rho, st.X, st.nl.counts, st.nl.indices, p.h, p.m,
        st.nl.sm, st.nl.si; ndrange = N)
    KernelAbstractions.synchronize(backend)

    A = SPHAdjoint.wnorm(p.h)
    rho_ref = zeros(T, N)
    for i in 1:N, j in 1:N
        r = hypot(Xh[1, i] - Xh[1, j], Xh[2, i] - Xh[2, j])
        rho_ref[i] += p.m * SPHAdjoint.w_kern(r / p.h, A)
    end

    @test Array(st.rho) ≈ rho_ref rtol = 1e-12
end

# ---------------------------------------------------------------------------
# 近傍リストの再利用が結果を変えないこと
#   物理カーネルが r² < (2h)² で絞るので、リストが上位集合であれば
#   毎ステップ構築した場合と厳密に同じ相互作用集合になる（= ビット一致）。
# ---------------------------------------------------------------------------
@testset "近傍リストの再利用が結果を変えない" begin
    rng = MersenneTwister(7)
    # 実際の設定と同じ密度にすること（格子間隔 dp、h = 1.3 dp）。
    # これより密にすると 1 粒子あたりの近傍数が maxnb を超えて溢れる
    # （そのときは flags[1] が立って警告が出る）。
    dp = 0.02
    p = SPHParams{T}(h = 1.3 * dp, m = dp^2, rho0 = 1.0, c = 10.0, mu = 0.02,
                     dt = 1e-4, Lx = 1.0, Ly = 0.6, kw = 1e4, ngx = 9, ngy = 7)
    xs = dp:dp:0.24
    ys = dp:dp:0.20
    N = length(xs) * length(ys)
    X0 = zeros(T, 2, N)
    k = 0
    for y in ys, x in xs
        k += 1
        X0[1, k] = x + 0.1 * dp * randn(rng)
        X0[2, k] = y + 0.1 * dp * randn(rng)
    end
    V0 = 0.05 .* randn(rng, T, 2, N)
    th = KernelAbstractions.zeros(backend, T, p.ngy, p.ngx)

    # interval=1（毎ステップ構築）と interval=8（既定）で 40 ステップ回す
    a = State(backend, X0, V0, p; interval = 1)
    b = State(backend, X0, V0, p; interval = 8)
    simulate!(a, th, p, backend, 40)
    simulate!(b, th, p, backend, 40)

    # 本質的な不変量: 実効的な相互作用集合（r < 2h）が完全に一致すること。
    # リストが真の近傍の上位集合であれば、物理カーネルの r² 判定で同じ集合に
    # 絞られる。これが成り立つ限り離散随伴も厳密なまま。
    r2max = (2 * p.h)^2
    effective(st) = begin
        nbc = Array(st.nl.counts)
        idx = Array(st.nl.indices)
        Xs = Array(st.X)
        [Set(j for m in 1:nbc[i]
             for j in (idx[SPHAdjoint.nl_index(Int32(m-1), Int32(i), Int32(st.nl.sm), Int32(st.nl.si))],)
             if (Xs[1, i] - Xs[1, j])^2 + (Xs[2, i] - Xs[2, j])^2 < r2max) for i in 1:N]
    end
    @test effective(a) == effective(b)

    # 数値としては行内の並び順が違うぶん総和順序が変わるので、ビット一致では
    # なく ulp レベルで一致する（実測: X はビット一致、V の差は 1 ulp）。
    @test Array(a.X) ≈ Array(b.X) rtol = 1e-14
    @test Array(a.V) ≈ Array(b.V) rtol = 1e-14
    @test Array(b.nl.flags) == Int32[0, 0]
end

# ---------------------------------------------------------------------------
# 3. 随伴 vs 中心差分
#    本体。目的関数は終端状態のランダム線形汎関数にして全成分を一度に検査する。
# ---------------------------------------------------------------------------
@testset "随伴 vs 中心差分" begin
    rng = MersenneTwister(0)
    p = SPHParams{T}(h = 0.10, m = 0.010, rho0 = 1.0, c = 8.0, mu = 0.02,
                     dt = 1.0e-3, Lx = 1.0, Ly = 1.0, kw = 2.0e4, ngx = 7, ngy = 7)
    nx, ny = 5, 5
    N = nx * ny
    X0 = zeros(T, 2, N)
    k = 0
    for jy in 0:ny-1, ix in 0:nx-1
        k += 1
        X0[1, k] = 0.20 + 0.06 * ix + 0.004 * randn(rng)
        X0[2, k] = 0.20 + 0.06 * jy + 0.004 * randn(rng)
    end
    V0 = 0.05 .* randn(rng, T, 2, N)
    theta0 = 3 .* rand(rng, T, p.ngy, p.ngx)
    nsteps = 12
    cX = randn(rng, T, 2, N)
    cV = randn(rng, T, 2, N)

    # 近傍リストの再利用は別の testset で検証しているので、ここでは interval=1
    # （毎ステップ組み直し）に固定して随伴そのものだけを見る。
    # なおこの設定は壁ペナルティが強く 1 ステップの変位が大きいため、既定の
    # interval=8 だと変位の上限を破って警告が出る（勾配自体は保たれるが、
    # 保証が効かない領域に入る）。
    function run_forward(X0, V0, theta; want_tape = false)
        st = State(backend, X0, V0, p; interval = 1)
        th = KernelAbstractions.allocate(backend, T, p.ngy, p.ngx)
        copyto!(th, theta)
        tape = want_tape ? Tape(backend, N, nsteps, p) : nothing
        simulate!(st, th, p, backend, nsteps; tape)
        J = sum(Array(st.X) .* cX) + sum(Array(st.V) .* cV)
        return J, st, th, tape
    end

    J, st, th, tape = run_forward(X0, V0, theta0; want_tape = true)
    @test isfinite(J)
    @test all(isfinite, Array(st.X))

    ws = AdjointWorkspace(backend, N, p)
    backward!(ws, tape, th, p, backend; seedX = cX, seedV = cV, interval = 1)
    gth = Array(ws.gtheta)
    gX = Array(ws.gX)
    gV = Array(ws.gV)
    @test all(isfinite, gth)
    @test all(isfinite, gX)

    hfd = 1e-6
    fd(f, x, i) = begin
        xp = copy(x); xp[i] += hfd
        xm = copy(x); xm[i] -= hfd
        (f(xp) - f(xm)) / (2hfd)
    end

    for idx in [CartesianIndex(1, 1), CartesianIndex(4, 3), CartesianIndex(5, 6)]
        g = fd(t -> run_forward(X0, V0, t)[1], theta0, idx)
        @test gth[idx] ≈ g rtol = 1e-5 atol = 1e-9
    end
    for idx in [CartesianIndex(1, 1), CartesianIndex(2, 9), CartesianIndex(1, 20)]
        g = fd(x -> run_forward(x, V0, theta0)[1], X0, idx)
        @test gX[idx] ≈ g rtol = 1e-5 atol = 1e-8
    end
    for idx in [CartesianIndex(1, 4), CartesianIndex(2, 15)]
        g = fd(v -> run_forward(X0, v, theta0)[1], V0, idx)
        @test gV[idx] ≈ g rtol = 1e-5 atol = 1e-8
    end
end

# ---------------------------------------------------------------------------
# 4. 設計場が効いていること（抗力が入ると粒子が遅くなる）
# ---------------------------------------------------------------------------
@testset "Brinkman 抗力が効いている" begin
    p = SPHParams{T}(h = 0.06, m = 0.004, rho0 = 1.0, c = 10.0, mu = 0.02,
                     dt = 1e-4, Lx = 1.0, Ly = 0.6, kw = 1e4, ngx = 9, ngy = 7)
    N = 64
    X0 = zeros(T, 2, N)
    for k in 1:N
        X0[1, k] = 0.2 + 0.02 * ((k - 1) % 8)
        X0[2, k] = 0.3 + 0.02 * ((k - 1) ÷ 8)
    end
    V0 = fill(T(1.0), 2, N)

    # |v| が音速を超えるほど激しい設定なので、近傍リストの再利用は使わない
    # （変位が上限を破って警告が出る。ここで見たいのは抗力の効果だけ）
    ke(th) = begin
        st = State(backend, X0, V0, p; interval = 1)
        thd = KernelAbstractions.allocate(backend, T, p.ngy, p.ngx)
        copyto!(thd, fill(T(th), p.ngy, p.ngx))
        simulate!(st, thd, p, backend, 200)
        sum(abs2, Array(st.V))
    end

    @test ke(500.0) < ke(0.0)      # 抗力ありのほうが運動エネルギーが小さい
end

# ---------------------------------------------------------------------------
# 5. 抗力が陰的なので α_max を上げても発散しないこと
#    陽解法版は dt < 2/α_max の制限があり、α_max を上げるとすぐ発散した。
#    通常の SPH 設定（h = 1.3 dp）で見る。testset 4 は h = 3 dp の過剰平滑化な
#    設定で近傍数が maxnb 近くまで来るため、安定性の検証には向かない。
# ---------------------------------------------------------------------------
@testset "抗力の陰解法で α_max を上げても発散しない" begin
    dp = 0.02
    p = SPHParams{T}(h = 1.3 * dp, m = dp^2, rho0 = 1.0, c = 15.0, mu = 0.05,
                     dt = 2.0e-4, Lx = 1.0, Ly = 0.5, kw = 5.0e4, ngx = 33, ngy = 17)
    xs = dp:dp:0.24
    ys = dp:dp:0.20
    N = length(xs) * length(ys)
    X0 = zeros(T, 2, N)
    k = 0
    for y in ys, x in xs
        k += 1
        X0[1, k] = x
        X0[2, k] = y
    end
    @test count(j -> X0[1, j] == 0 && X0[2, j] == 0, 1:N) == 0   # 配置が壊れていない

    explicit_limit = 2 / p.dt          # 陽解法ならここで発散していた（= 1e4）
    function run_alpha(alpha_max)
        st = State(backend, X0, zeros(T, 2, N), p; interval = 2)
        th = KernelAbstractions.allocate(backend, T, p.ngy, p.ngx)
        copyto!(th, fill(T(alpha_max), p.ngy, p.ngx))
        simulate!(st, th, p, backend, 300)
        return maximum(abs, Array(st.V)), all(isfinite, Array(st.X))
    end

    v_soft, ok_soft = run_alpha(explicit_limit / 10)   # 陽解法でも通る領域
    v_hard, ok_hard = run_alpha(explicit_limit * 10)   # 陽解法なら発散する領域
    @test ok_soft
    @test ok_hard
    @test v_hard < v_soft                              # 硬いほど止まる
end

