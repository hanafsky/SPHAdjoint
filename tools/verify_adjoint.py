"""
2D WCSPH の前進計算と「手書き離散随伴」を実装し、PyTorch の自動微分および
有限差分と突き合わせて随伴式の正しさを検証するための使い捨てハーネス。

ここで確認した式をそのまま Julia + KernelAbstractions に移植する。
（この環境には Julia バイナリが無いため、数式の検証だけを Python で行う）

物理:
  密度総和      rho_i = sum_j m W(q_ij)
  線形状態方程式 p_i   = c^2 (rho_i - rho0)
  圧力加速度    a^p_i = -sum_j m (p_i/rho_i^2 + p_j/rho_j^2) F_ij d_ij
  Morris粘性    a^v_i = sum_j m (2 mu/(rho_i rho_j)) F_ij (v_i - v_j)
  重力          g
  Brinkman抗力  -alpha(x_i) v_i     <- alpha が設計変数場（背景格子上）
  平滑壁        k_w * relu(penetration)^2

カーネル (Wendland C2, 2D):
  q = r/h,  A = 7/(4 pi h^2)
  W(q)  = A (1-q/2)^4 (2q+1)                 (0<=q<2)
  dW/dq = -5A q (1-q/2)^3
  grad_i W_ij = F(q) d_ij,  F(q) = -5A (1-q/2)^3 / h^2
     -> r で割る操作が消えるので r->0 の特異性が無い。AD にも随伴にも優しい。
  dF/dq = G(q) = (15A/2) (1-q/2)^2 / h^2
"""

import numpy as np

# ----------------------------------------------------------------------------
# パラメータ
# ----------------------------------------------------------------------------


class Params:
    def __init__(self):
        self.h = 0.10
        self.m = 0.010
        self.rho0 = 1.0
        self.c = 8.0
        self.mu = 0.02
        self.g = np.array([0.0, -9.81])
        self.dt = 1.0e-3
        self.Lx = 1.0
        self.Ly = 1.0
        self.kw = 2.0e4
        self.ngx = 9
        self.ngy = 9

    @property
    def A(self):
        return 7.0 / (4.0 * np.pi * self.h**2)


# ----------------------------------------------------------------------------
# カーネル関数
# ----------------------------------------------------------------------------


def W_of_q(q, A):
    u = 1.0 - 0.5 * q
    return np.where(q < 2.0, A * u**4 * (2.0 * q + 1.0), 0.0)


def F_of_q(q, A, h):
    """grad_i W_ij = F(q) * d_ij"""
    u = 1.0 - 0.5 * q
    return np.where(q < 2.0, -5.0 * A * u**3 / h**2, 0.0)


def G_of_q(q, A, h):
    """dF/dq"""
    u = 1.0 - 0.5 * q
    return np.where(q < 2.0, 7.5 * A * u**2 / h**2, 0.0)


# ----------------------------------------------------------------------------
# 設計変数場（背景格子）: 双線形補間
# ----------------------------------------------------------------------------


def interp_setup(X, P):
    dx = P.Lx / (P.ngx - 1)
    dy = P.Ly / (P.ngy - 1)
    gx = np.clip(X[:, 0] / dx, 0.0, P.ngx - 1 - 1e-9)
    gy = np.clip(X[:, 1] / dy, 0.0, P.ngy - 1 - 1e-9)
    i0 = np.floor(gx).astype(int)
    j0 = np.floor(gy).astype(int)
    i0 = np.clip(i0, 0, P.ngx - 2)
    j0 = np.clip(j0, 0, P.ngy - 2)
    fx = gx - i0
    fy = gy - j0
    return i0, j0, fx, fy, dx, dy


def interp_alpha(theta, X, P):
    """theta: (ngy, ngx) の節点値。alpha_i を返す。"""
    i0, j0, fx, fy, dx, dy = interp_setup(X, P)
    t00 = theta[j0, i0]
    t01 = theta[j0, i0 + 1]
    t10 = theta[j0 + 1, i0]
    t11 = theta[j0 + 1, i0 + 1]
    a = (
        (1 - fx) * (1 - fy) * t00
        + fx * (1 - fy) * t01
        + (1 - fx) * fy * t10
        + fx * fy * t11
    )
    return a


def interp_alpha_vjp(theta, X, P, abar):
    """alpha の随伴 abar から theta 勾配と X 勾配を返す。"""
    i0, j0, fx, fy, dx, dy = interp_setup(X, P)
    t00 = theta[j0, i0]
    t01 = theta[j0, i0 + 1]
    t10 = theta[j0 + 1, i0]
    t11 = theta[j0 + 1, i0 + 1]

    # clip が効いている粒子は座標に対する勾配が 0（AD と揃える）
    gxr = X[:, 0] / dx
    gyr = X[:, 1] / dy
    act_x = ((gxr > 0.0) & (gxr < P.ngx - 1 - 1e-9)).astype(float)
    act_y = ((gyr > 0.0) & (gyr < P.ngy - 1 - 1e-9)).astype(float)

    gtheta = np.zeros_like(theta)
    np.add.at(gtheta, (j0, i0), abar * (1 - fx) * (1 - fy))
    np.add.at(gtheta, (j0, i0 + 1), abar * fx * (1 - fy))
    np.add.at(gtheta, (j0 + 1, i0), abar * (1 - fx) * fy)
    np.add.at(gtheta, (j0 + 1, i0 + 1), abar * fx * fy)

    dadx = (-(1 - fy) * t00 + (1 - fy) * t01 - fy * t10 + fy * t11) / dx * act_x
    dady = (-(1 - fx) * t00 - fx * t01 + (1 - fx) * t10 + fx * t11) / dy * act_y
    gX = np.stack([abar * dadx, abar * dady], axis=1)
    return gtheta, gX


# ----------------------------------------------------------------------------
# 壁（平滑ペナルティ）
# ----------------------------------------------------------------------------


def wall_accel(X, P):
    s_lo = np.maximum(0.0, -X)
    s_hi = np.maximum(0.0, X - np.array([P.Lx, P.Ly]))
    return P.kw * (s_lo**2 - s_hi**2)


def wall_accel_dX(X, P):
    s_lo = np.maximum(0.0, -X)
    s_hi = np.maximum(0.0, X - np.array([P.Lx, P.Ly]))
    return P.kw * (-2.0 * s_lo - 2.0 * s_hi)


# ----------------------------------------------------------------------------
# 前進: 加速度（全対全。検証用なので近傍探索はしない）
# ----------------------------------------------------------------------------


def accel(X, V, theta, P, save=None):
    N = X.shape[0]
    d = X[:, None, :] - X[None, :, :]  # d_ij = x_i - x_j
    r2 = np.sum(d * d, axis=2)
    r = np.sqrt(r2 + 1e-30)
    q = r / P.h

    Wq = W_of_q(q, P.A)
    Fq = F_of_q(q, P.A, P.h)

    rho = P.m * np.sum(Wq, axis=1)
    p = P.c**2 * (rho - P.rho0)

    eye = np.eye(N, dtype=bool)

    # 圧力項
    Pij = P.m * (p[:, None] / rho[:, None] ** 2 + p[None, :] / rho[None, :] ** 2)
    coef_p = np.where(eye, 0.0, Pij * Fq)
    a_p = -np.einsum("ij,ijk->ik", coef_p, d)

    # 粘性項
    Cij = 2.0 * P.m * P.mu / (rho[:, None] * rho[None, :]) * Fq
    Cij = np.where(eye, 0.0, Cij)
    dv = V[:, None, :] - V[None, :, :]
    a_v = np.einsum("ij,ijk->ik", Cij, dv)

    alpha = interp_alpha(theta, X, P)
    a = a_p + a_v + P.g[None, :] - alpha[:, None] * V + wall_accel(X, P)

    if save is not None:
        save.update(
            dict(d=d, r=r, q=q, Fq=Fq, rho=rho, p=p, Pij=Pij, Cij=Cij,
                 dv=dv, alpha=alpha, eye=eye)
        )
    return a


# ----------------------------------------------------------------------------
# 随伴: 加速度の VJP
# ----------------------------------------------------------------------------


def accel_vjp(X, V, theta, P, S, abar):
    """abar: (N,2)。 (gX, gV, gtheta) を返す。"""
    N = X.shape[0]
    d, r, q = S["d"], S["r"], S["q"]
    Fq, rho, p = S["Fq"], S["rho"], S["p"]
    Pij, Cij, dv, alpha, eye = S["Pij"], S["Cij"], S["dv"], S["alpha"], S["eye"]

    gX = np.zeros_like(X)
    gV = np.zeros_like(V)
    grho = np.zeros(N)
    gp = np.zeros(N)
    gF = np.zeros((N, N))

    # --- 壁 ---
    gX += abar * wall_accel_dX(X, P)

    # --- Brinkman 抗力: a_i -= alpha_i v_i ---
    gV += -alpha[:, None] * abar
    galpha = -np.sum(abar * V, axis=1)
    gth, gXa = interp_alpha_vjp(theta, X, P, galpha)
    gX += gXa
    gtheta = gth

    # --- 粘性項: a^v_i = sum_j Cij (v_i - v_j),  Cij = 2 m mu/(rho_i rho_j) F_ij ---
    ad_v = np.einsum("ik,ijk->ij", abar, dv)  # (i,j) 成分ごとの abar . dv_ij
    Csum = np.sum(Cij, axis=1)
    gV += Csum[:, None] * abar
    gV -= np.einsum("ij,ik->jk", Cij, abar)
    base_v = 2.0 * P.m * P.mu / (rho[:, None] * rho[None, :])
    gF += np.where(eye, 0.0, ad_v * base_v)
    grho += -np.sum(np.where(eye, 0.0, ad_v * Cij), axis=1) / rho
    grho += -np.sum(np.where(eye, 0.0, ad_v * Cij), axis=0) / rho

    # --- 圧力項: a^p_i = -sum_j Pij F_ij d_ij ---
    ad_d = np.einsum("ik,ijk->ij", abar, d)  # abar_i . d_ij
    coef_p = np.where(eye, 0.0, Pij * Fq)
    # d_ij を通じた x への直接寄与
    gX += -np.sum(coef_p, axis=1)[:, None] * abar
    gX += np.einsum("ij,ik->jk", coef_p, abar)
    # F を通じた寄与
    gF += np.where(eye, 0.0, -Pij * ad_d)
    # Pij を通じた寄与
    gPij = np.where(eye, 0.0, -Fq * ad_d)
    gp += P.m * np.sum(gPij, axis=1) / rho**2
    gp += P.m * np.sum(gPij, axis=0) / rho**2
    grho += -2.0 * P.m * p / rho**3 * np.sum(gPij, axis=1)
    grho += -2.0 * P.m * p / rho**3 * np.sum(gPij, axis=0)

    # --- p = c^2 (rho - rho0) ---
    grho += P.c**2 * gp

    # --- F_ij = F(q_ij),  dF/dx_i = G(q) d/(h r) ---
    Gq = G_of_q(q, P.A, P.h)
    w = np.where(eye, 0.0, gF * Gq / (P.h * r))
    contrib = w[:, :, None] * d
    gX += np.sum(contrib, axis=1)
    gX -= np.sum(contrib, axis=0)

    # --- 密度総和 rho_i = sum_j m W_ij,  d rho_i/d x_i = sum_j m F_ij d_ij ---
    wr = np.where(eye, 0.0, grho[:, None] * P.m * Fq)
    contrib2 = wr[:, :, None] * d
    gX += np.sum(contrib2, axis=1)
    gX -= np.sum(contrib2, axis=0)

    return gX, gV, gtheta


# ----------------------------------------------------------------------------
# 時間積分（semi-implicit Euler）と随伴
# ----------------------------------------------------------------------------


def simulate(X0, V0, theta, P, nsteps, keep=True):
    X, V = X0.copy(), V0.copy()
    tape = []
    for _ in range(nsteps):
        S = {}
        a = accel(X, V, theta, P, save=S)
        if keep:
            tape.append((X.copy(), V.copy(), S))
        V = V + P.dt * a
        X = X + P.dt * V
    return X, V, tape


def simulate_adjoint(tape, theta, P, gXT, gVT):
    gX, gV = gXT.copy(), gVT.copy()
    gtheta = np.zeros_like(theta)
    for X, V, S in reversed(tape):
        # X' = X + dt V'
        gV1 = gV + P.dt * gX
        # V' = V + dt a
        gabar = P.dt * gV1
        gV = gV1.copy()
        dX, dV, dth = accel_vjp(X, V, theta, P, S, gabar)
        gX = gX + dX
        gV = gV + dV
        gtheta += dth
    return gX, gV, gtheta


# ----------------------------------------------------------------------------
# PyTorch による参照実装（自動微分の ground truth）
# ----------------------------------------------------------------------------


def torch_simulate(X0, V0, theta, P, nsteps, cX, cV):
    import torch

    t = lambda z: torch.tensor(z, dtype=torch.float64)
    A = P.A
    X = X0.clone()
    V = V0.clone()
    N = X.shape[0]
    eye = torch.eye(N, dtype=torch.bool)
    Lxy = t(np.array([P.Lx, P.Ly]))
    dxg = P.Lx / (P.ngx - 1)
    dyg = P.Ly / (P.ngy - 1)

    for _ in range(nsteps):
        d = X[:, None, :] - X[None, :, :]
        r = torch.sqrt((d * d).sum(2) + 1e-30)
        q = r / P.h
        u = torch.clamp(1.0 - 0.5 * q, min=0.0)
        Wq = A * u**4 * (2.0 * q + 1.0)
        Fq = -5.0 * A * u**3 / P.h**2

        rho = P.m * Wq.sum(1)
        p = P.c**2 * (rho - P.rho0)

        Pij = P.m * (p[:, None] / rho[:, None] ** 2 + p[None, :] / rho[None, :] ** 2)
        coef_p = torch.where(eye, torch.zeros_like(Fq), Pij * Fq)
        a_p = -torch.einsum("ij,ijk->ik", coef_p, d)

        Cij = 2.0 * P.m * P.mu / (rho[:, None] * rho[None, :]) * Fq
        Cij = torch.where(eye, torch.zeros_like(Cij), Cij)
        dv = V[:, None, :] - V[None, :, :]
        a_v = torch.einsum("ij,ijk->ik", Cij, dv)

        gx = torch.clamp(X[:, 0] / dxg, 0.0, P.ngx - 1 - 1e-9)
        gy = torch.clamp(X[:, 1] / dyg, 0.0, P.ngy - 1 - 1e-9)
        i0 = torch.clamp(torch.floor(gx).long(), 0, P.ngx - 2)
        j0 = torch.clamp(torch.floor(gy).long(), 0, P.ngy - 2)
        fx = gx - i0
        fy = gy - j0
        alpha = (
            (1 - fx) * (1 - fy) * theta[j0, i0]
            + fx * (1 - fy) * theta[j0, i0 + 1]
            + (1 - fx) * fy * theta[j0 + 1, i0]
            + fx * fy * theta[j0 + 1, i0 + 1]
        )

        s_lo = torch.clamp(-X, min=0.0)
        s_hi = torch.clamp(X - Lxy, min=0.0)
        a_w = P.kw * (s_lo**2 - s_hi**2)

        a = a_p + a_v + t(P.g)[None, :] - alpha[:, None] * V + a_w
        V = V + P.dt * a
        X = X + P.dt * V

    return (X * cX).sum() + (V * cV).sum()


# ----------------------------------------------------------------------------
# 検証
# ----------------------------------------------------------------------------


def main():
    import torch

    rng = np.random.default_rng(0)
    P = Params()

    # 初期配置: 左下のブロック（格子＋微小擾乱）
    nx, ny = 7, 7
    xs = np.linspace(0.12, 0.12 + (nx - 1) * 0.05, nx)
    ys = np.linspace(0.12, 0.12 + (ny - 1) * 0.05, ny)
    XX, YY = np.meshgrid(xs, ys)
    X0 = np.stack([XX.ravel(), YY.ravel()], axis=1)
    X0 += 0.004 * rng.standard_normal(X0.shape)
    V0 = 0.05 * rng.standard_normal(X0.shape)
    N = X0.shape[0]
    theta = 3.0 * rng.random((P.ngy, P.ngx))

    nsteps = 25
    cX = rng.standard_normal((N, 2))
    cV = rng.standard_normal((N, 2))

    print(f"N = {N} 粒子, {nsteps} ステップ, 設計変数 {theta.size} 個")

    # --- 手書き随伴 ---
    XT, VT, tape = simulate(X0, V0, theta, P, nsteps)
    gX_adj, gV_adj, gth_adj = simulate_adjoint(tape, theta, P, cX, cV)

    # --- PyTorch 自動微分 ---
    tX0 = torch.tensor(X0, dtype=torch.float64, requires_grad=True)
    tV0 = torch.tensor(V0, dtype=torch.float64, requires_grad=True)
    tth = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    J = torch_simulate(
        tX0, tV0, tth, P, nsteps,
        torch.tensor(cX, dtype=torch.float64),
        torch.tensor(cV, dtype=torch.float64),
    )
    J.backward()
    gX_ad = tX0.grad.numpy()
    gV_ad = tV0.grad.numpy()
    gth_ad = tth.grad.numpy()

    Jnp = np.sum(XT * cX) + np.sum(VT * cV)
    print(f"J(numpy) = {Jnp:.12f}   J(torch) = {J.item():.12f}")

    def relerr(a, b):
        return np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30)

    print(f"  dJ/dX0    相対誤差 = {relerr(gX_adj, gX_ad):.3e}")
    print(f"  dJ/dV0    相対誤差 = {relerr(gV_adj, gV_ad):.3e}")
    print(f"  dJ/dtheta 相対誤差 = {relerr(gth_adj, gth_ad):.3e}")

    k = np.unravel_index(np.argmax(np.abs(gX_adj - gX_ad)), gX_adj.shape)
    print(f"  最大誤差箇所 X0{k}: 随伴={gX_adj[k]:+.9e} AD={gX_ad[k]:+.9e}")

    print("\nステップ数を変えた場合の相対誤差:")
    for ns in [1, 2, 5, 10, 25]:
        cXs, cVs = cX, cV
        Xa, Va, tp = simulate(X0, V0, theta, P, ns)
        gXa, gVa, gta = simulate_adjoint(tp, theta, P, cXs, cVs)
        tX = torch.tensor(X0, dtype=torch.float64, requires_grad=True)
        tV = torch.tensor(V0, dtype=torch.float64, requires_grad=True)
        tt = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
        Js = torch_simulate(tX, tV, tt, P, ns,
                            torch.tensor(cXs, dtype=torch.float64),
                            torch.tensor(cVs, dtype=torch.float64))
        Js.backward()
        print(f"  {ns:3d} steps:  X0 {relerr(gXa, tX.grad.numpy()):.3e}   "
              f"V0 {relerr(gVa, tV.grad.numpy()):.3e}   "
              f"theta {relerr(gta, tt.grad.numpy()):.3e}")

    # --- 有限差分による独立チェック（theta の数成分だけ） ---
    print("\n有限差分チェック (dJ/dtheta, 中心差分, eps=1e-6):")
    eps = 1e-6
    idxs = [(0, 0), (3, 4), (5, 2), (P.ngy - 1, P.ngx - 1)]
    for (j, i) in idxs:
        tp = theta.copy(); tp[j, i] += eps
        tm = theta.copy(); tm[j, i] -= eps
        Xp, Vp, _ = simulate(X0, V0, tp, P, nsteps, keep=False)
        Xm, Vm, _ = simulate(X0, V0, tm, P, nsteps, keep=False)
        Jp = np.sum(Xp * cX) + np.sum(Vp * cV)
        Jm = np.sum(Xm * cX) + np.sum(Vm * cV)
        fd = (Jp - Jm) / (2 * eps)
        print(f"  theta[{j},{i}]: 随伴 = {gth_adj[j,i]:+.9e}   FD = {fd:+.9e}")

    print("\n有限差分チェック (dJ/dX0, 中心差分):")
    for k in [0, 17, 40]:
        for c in [0, 1]:
            Xp = X0.copy(); Xp[k, c] += eps
            Xm = X0.copy(); Xm[k, c] -= eps
            Ap, Bp, _ = simulate(Xp, V0, theta, P, nsteps, keep=False)
            Am, Bm, _ = simulate(Xm, V0, theta, P, nsteps, keep=False)
            Jp = np.sum(Ap * cX) + np.sum(Bp * cV)
            Jm = np.sum(Am * cX) + np.sum(Bm * cV)
            fd = (Jp - Jm) / (2 * eps)
            print(f"  X0[{k},{c}]: 随伴 = {gX_adj[k,c]:+.9e}   FD = {fd:+.9e}")


if __name__ == "__main__":
    main()
