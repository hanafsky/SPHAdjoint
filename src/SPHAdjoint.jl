"""
    SPHAdjoint

Julia + KernelAbstractions.jl による微分可能 2D WCSPH。

前進も随伴も同一の KernelAbstractions カーネルで書いてあるので、
CPU / Metal / CUDA / ROCm を `backend` の差し替えだけで切り替えられる。

随伴は Enzyme ではなく**手書きの離散随伴**。理由は `src/adjoint.jl` の
冒頭コメントを参照（Enzyme の GPU 対応は事実上 CUDA 中心で、Metal 上は
現状かなり険しい。一方 SPH の随伴は全部 gather に書き直せるので、
手書きなら前進とまったく同じコードが Metal で走る）。

使い方の最小例:

```julia
using SPHAdjoint, KernelAbstractions
backend = CPU()                       # Metal なら MetalBackend()
p  = SPHParams{Float64}(h=0.03, m=9e-4, rho0=1.0, c=15.0, mu=0.02,
                        dt=2e-4, Lx=1.0, Ly=1.0, kw=2e4, ngx=17, ngy=17)
st = State(backend, X0, V0, p)
tape = Tape(backend, size(X0, 2), 500, p)   # (2, N, 500) を一度だけ確保
simulate!(st, theta, p, backend, 500; tape)
J, seed = target_objective(st.X, 0.8, 0.15, 0.08)
ws = AdjointWorkspace(backend, size(st.X,2), p)
backward!(ws, tape, theta, p, backend; seedX=seed, seedV=zero(seed))
# ws.gtheta が設計変数場の勾配
```

!!! warning "精度"
    Apple GPU は Float64 を持たない。Metal では `Float32` 固定になるため、
    勾配検証（`scripts/01_gradcheck.jl`）は必ず CPU + `Float64` で行うこと。
"""
module SPHAdjoint

using KernelAbstractions
using Atomix

export SPHParams, State, CellList, NeighborList, AdjointWorkspace, Tape
export simulate!, step!, build!, build_neighbors!, backward!, target_objective, reset!

include("kernels.jl")
include("params.jl")
include("neighbors.jl")
include("forward.jl")
include("adjoint.jl")
include("driver.jl")

end # module
