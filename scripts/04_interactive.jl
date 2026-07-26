# # インタラクティブ版：固体形状をマウスで塗りながら流れを見る
#
# ```
# julia --project=. scripts/04_interactive.jl
# ```
#
# 設計変数場 θ（Brinkman 摩擦場）がそもそも背景の一様格子なので、
# 「固体形状を編集する」が「ヒートマップを塗る」に落ちる。任意形状の
# ジオメトリを編集するより圧倒的に楽で、しかも最適化と同じ表現になる。
#
# ## なぜ計算中に形を変えても壊れないか
#
# 固体を「硬い壁」ではなく Brinkman 抗力 `-α(x)v` で表しているため、
# 新しく固体にした領域の中にいる粒子は *減速するだけ* で、弾かれたり
# 飛んだりしない。ペナルティ表現の実用上の利点で、インタラクションが
# 気持ちよくなる理由でもある。
#
# ## アーキテクチャ上の要点
#
# シミュレーションを別スレッド／別タスクで回さず、**Makie の tick イベントの
# 中で回す**こと。GPU コマンドの発行元タスクと GL コンテキストを同じに
# しておくのが一番安全で、実際これで足りる。
#
# `nsub` は `02_dambreak.jl` で測ったステップ/秒から決める。

using SPHAdjoint
using KernelAbstractions
using GLMakie

# Metal で走らせる場合はここだけ差し替える:
#   using Metal
#   const T = Float32
#   backend = MetalBackend()
const T = Float64
backend = CPU()

# ---- パラメータ -----------------------------------------------------------
dp = 0.016
p = SPHParams{T}(
    h = 1.3 * dp, m = dp^2 * 1.0, rho0 = 1.0, c = 15.0, mu = 0.05,
    dt = 2.0e-4, Lx = 1.0, Ly = 0.5, kw = 5.0e4, ngx = 65, ngy = 33,
)
alpha_max = T(1500.0)          # 固体セルの抗力係数
nsub = 20                      # 1 フレームあたりの物理ステップ数

# dt < 2/alpha_max でないと陽的な抗力項が発散する。下のチェックで気づけるように。
@assert p.dt < 2 / alpha_max "dt が大きすぎます（抗力項が陽解法で不安定）"

# ---- 初期粒子配置 ---------------------------------------------------------
function initial_particles()
    xs = dp:dp:0.24
    ys = dp:dp:0.40
    X = zeros(T, 2, length(xs) * length(ys))
    k = 0
    for y in ys, x in xs
        k += 1
        X[1, k] = x
        X[2, k] = y
    end
    return X, zeros(T, size(X))
end
X0, V0 = initial_particles()
N = size(X0, 2)
@info "粒子数 N = $N,  設計格子 $(p.ngx)×$(p.ngy)"

st = State(backend, X0, V0, p)
th_dev = KernelAbstractions.allocate(backend, T, p.ngy, p.ngx)

# ---- 設計場のキャンバス ---------------------------------------------------
# Makie の heatmap は M[i,j] を (x,y) と解釈するので、キャンバスは (ngx, ngy)。
# ソルバ側の θ は (ngy, ngx) なので転置して渡す。
# 値は 1 = 流体, 0 = 固体。
canvas = fill(T(1), p.ngx, p.ngy)

function push_design!()
    theta = permutedims(alpha_max .* (1 .- canvas))   # (ngy, ngx)
    copyto!(th_dev, theta)
end
push_design!()

gxs = range(0, p.Lx, length = p.ngx)
gys = range(0, p.Ly, length = p.ngy)

# ---- 図 -------------------------------------------------------------------
fig = Figure(size = (1240, 620))
ax = Axis(fig[1, 1], aspect = DataAspect(), limits = (0, p.Lx, 0, p.Ly),
          title = "左ドラッグ=固体を置く / 右ドラッグ=消す")

canvas_obs = Observable(canvas)
heatmap!(ax, gxs, gys, canvas_obs, colormap = :grays, colorrange = (0, 1),
         interpolate = true)

pos_obs = Observable(Point2f.(X0[1, :], X0[2, :]))
spd_obs = Observable(zeros(Float32, N))
scatter!(ax, pos_obs, color = spd_obs, colormap = :viridis, colorrange = (0, 2.5),
         markersize = 6)

# ---- コントロール ---------------------------------------------------------
panel = fig[1, 2] = GridLayout(tellheight = false)
run_toggle = Toggle(panel[1, 1], active = true)
panel[1, 2] = Label(fig, "計算する", halign = :left)
brush_sl = Slider(panel[2, 1:2], range = 1:8, startvalue = 2)
panel[3, 1:2] = Label(fig, @lift("ブラシ半径 = $($(brush_sl.value))"))
reset_btn = Button(panel[4, 1:2], label = "粒子をリセット")
clear_btn = Button(panel[5, 1:2], label = "固体を全消去")
info_lbl = Label(panel[6, 1:2], "t = 0.000 s")
colsize!(fig.layout, 2, Relative(0.16))

# ---- 塗り -----------------------------------------------------------------
painting = Ref(false)
paint_value = Ref(T(0))          # 0 = 固体を置く, 1 = 消す

function paint_at!(mp)
    r = brush_sl.value[]
    i0 = round(Int, mp[1] / p.Lx * (p.ngx - 1)) + 1
    j0 = round(Int, mp[2] / p.Ly * (p.ngy - 1)) + 1
    changed = false
    for j in max(1, j0 - r):min(p.ngy, j0 + r), i in max(1, i0 - r):min(p.ngx, i0 + r)
        if (i - i0)^2 + (j - j0)^2 <= r^2
            canvas[i, j] = paint_value[]
            changed = true
        end
    end
    if changed
        notify(canvas_obs)
        push_design!()
    end
end

on(events(fig).mousebutton, priority = 2) do ev
    if ev.action == Mouse.press && (ev.button == Mouse.left || ev.button == Mouse.right)
        mp = mouseposition(ax)
        if 0 <= mp[1] <= p.Lx && 0 <= mp[2] <= p.Ly
            paint_value[] = ev.button == Mouse.left ? T(0) : T(1)
            painting[] = true
            paint_at!(mp)
            return Consume(true)
        end
    elseif ev.action == Mouse.release
        painting[] = false
    end
    return Consume(false)
end

on(events(fig).mouseposition, priority = 2) do _
    if painting[]
        mp = mouseposition(ax)
        if 0 <= mp[1] <= p.Lx && 0 <= mp[2] <= p.Ly
            paint_at!(mp)
        end
        return Consume(true)
    end
    return Consume(false)
end

on(reset_btn.clicks) do _
    A, B = initial_particles()
    copyto!(st.X, A)
    copyto!(st.V, B)
end

on(clear_btn.clicks) do _
    fill!(canvas, T(1))
    notify(canvas_obs)
    push_design!()
end

# ---- メインループ（Makie の tick の中で回す） -----------------------------
simtime = Ref(0.0)

on(events(fig).tick) do _
    run_toggle.active[] || return
    for _ in 1:nsub
        step!(st, th_dev, p, backend)
    end
    simtime[] += nsub * p.dt

    Xh = Array(st.X)          # Metal でも 10⁴ 粒子で 80 KB。毎フレーム転送して問題ない
    Vh = Array(st.V)
    pos_obs[] = Point2f.(Xh[1, :], Xh[2, :])
    spd_obs[] = Float32.(sqrt.(Vh[1, :] .^ 2 .+ Vh[2, :] .^ 2))
    info_lbl.text[] = "t = $(round(simtime[], digits=3)) s"
end

display(fig)

# ---------------------------------------------------------------------------
# 【推奨する改造: 抗力項を陰的にする】
#
# 上の @assert のとおり、陽解法だと dt < 2/α_max の制約がつく。固体を「硬く」
# したくて α_max を上げると、すぐ発散する。速度更新の抗力部分だけ陰的にすると
# 無条件安定になり、α_max をいくらでも上げられる:
#
#     v' = (v + dt·a_rest) / (1 + dt·α)          a_rest = 抗力以外の加速度
#
# 微分も割り算ひとつぶんなので随伴の追加はごく軽い。ただし
# **現在の随伴は陽解法版で検証済み**なので、変更したら gradcheck.jl を
# 通し直すこと（src/forward.jl の integrate_kernel! と
# src/adjoint.jl の抗力項の両方を直す必要がある）。
#
# 【発展: 感度場をその場で表示する】
#
# 前進だけならテープは要らないが、ここに「感度を見る」ボタンを足すと強力:
# 現在の状態から数百ステップだけ tape 付きで回し、backward! して
# ∂J/∂ρ_design をもう一枚のヒートマップに重ねる。
# 「自分で塗る」→「どこに塗るべきだったかを見る」が同じ画面で回る。
# ---------------------------------------------------------------------------
