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
@testset "セルリスト vs 総当たり密度" begin
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

    SPHAdjoint.density_kernel!(backend)(
        st.rho, st.X, st.cl.starts, st.cl.counts, st.cl.order,
        p.h, p.m, T(st.cl.cs), st.cl.nx, st.cl.ny; ndrange = N)
    KernelAbstractions.synchronize(backend)

    # 総当たり
    A = SPHAdjoint.wnorm(p.h)
    Xh = Array(st.X)
    rho_ref = zeros(T, N)
    for i in 1:N, j in 1:N
        r = hypot(Xh[1, i] - Xh[1, j], Xh[2, i] - Xh[2, j])
        rho_ref[i] += p.m * SPHAdjoint.w_kern(r / p.h, A)
    end

    @test Array(st.rho) ≈ rho_ref rtol = 1e-12
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

    function run_forward(X0, V0, theta; want_tape = false)
        st = State(backend, X0, V0, p)
        th = KernelAbstractions.allocate(backend, T, p.ngy, p.ngx)
        copyto!(th, theta)
        tape = want_tape ? Tuple{typeof(st.X),typeof(st.V)}[] : nothing
        simulate!(st, th, p, backend, nsteps; tape)
        J = sum(Array(st.X) .* cX) + sum(Array(st.V) .* cV)
        return J, st, th, tape
    end

    J, st, th, tape = run_forward(X0, V0, theta0; want_tape = true)
    @test isfinite(J)
    @test all(isfinite, Array(st.X))

    ws = AdjointWorkspace(backend, N, p)
    backward!(ws, tape, th, p, backend; seedX = cX, seedV = cV)
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

    ke(th) = begin
        st = State(backend, X0, V0, p)
        thd = KernelAbstractions.allocate(backend, T, p.ngy, p.ngx)
        copyto!(thd, fill(T(th), p.ngy, p.ngx))
        simulate!(st, thd, p, backend, 200)
        sum(abs2, Array(st.V))
    end

    @test ke(500.0) < ke(0.0)      # 抗力ありのほうが運動エネルギーが小さい
end
