"""
Julia カーネル (src/adjoint.jl) が実装している「gather 形式」の随伴を、
カーネルと一行ずつ対応する形で Python に写経し、PyTorch の自動微分と比較する。

verify_adjoint.py が検証したのは行列/scatter 形式。GPU 用に gather へ書き直す
過程で代数ミスが入っていないかは別に確かめる必要がある（そこがこの移植で
一番危ないところ）。
"""

import numpy as np
import torch

from verify_adjoint import (
    Params, W_of_q, F_of_q, G_of_q, interp_alpha, interp_alpha_vjp,
    wall_accel, wall_accel_dX, accel, torch_simulate,
)


def accel_vjp_gather(X, V, theta, P, abar):
    """src/adjoint.jl の adj_pass1 / adj_pass2 / adj_design をそのまま写経。"""
    N = X.shape[0]
    A, h, mass, c2 = P.A, P.h, P.m, P.c**2

    # 前進で計算済みの量を再構成（Julia 側も density_kernel! を再実行している）
    d_all = X[:, None, :] - X[None, :, :]
    r_all = np.sqrt(np.sum(d_all**2, axis=2) + 1e-30)
    rho = mass * np.sum(W_of_q(r_all / h, A), axis=1)
    p_ = c2 * (rho - P.rho0)

    gX = np.zeros_like(X)
    gV = np.zeros_like(V)
    grho = np.zeros(N)
    galpha = np.zeros(N)

    # ---- pass 1 -----------------------------------------------------------
    for i in range(N):
        xi, yi = X[i]
        vxi, vyi = V[i]
        abx, aby = abar[i]
        rhoi, pi_ = rho[i], p_[i]
        gx = gy = gvx = gvy = grh = sumS = 0.0

        for j in range(N):
            if j == i:
                continue
            dx = xi - X[j, 0]
            dy = yi - X[j, 1]
            r2 = dx * dx + dy * dy
            if r2 < np.finfo(float).eps * h * h:
                continue
            r = np.sqrt(r2)
            q = r / h
            if q >= 2.0:
                continue
            F = float(F_of_q(np.array(q), A, h))
            G = float(G_of_q(np.array(q), A, h))

            rhoj, pj = rho[j], p_[j]
            Pij = mass * (pi_ / rhoi**2 + pj / rhoj**2)
            Cbase = 2 * mass * P.mu / (rhoi * rhoj)
            C = Cbase * F

            dax = abx - abar[j, 0]
            day = aby - abar[j, 1]
            dvx = vxi - V[j, 0]
            dvy = vyi - V[j, 1]

            ad_d = -(dax * dx + day * dy)
            ad_v = dax * dvx + day * dvy

            gx -= Pij * F * dax
            gy -= Pij * F * day

            Fbar = Pij * ad_d + Cbase * ad_v
            w = Fbar * G / (h * r)
            gx += w * dx
            gy += w * dy

            gvx += C * dax
            gvy += C * day
            grh -= C * ad_v / rhoi
            sumS += F * ad_d

        grh += (-2 * mass * pi_ / rhoi**3) * sumS
        grh += c2 * (mass / rhoi**2) * sumS

        alpha = float(interp_alpha(theta, X[i:i + 1], P)[0])
        gvx -= alpha * abx
        gvy -= alpha * aby
        galpha[i] = -(abx * vxi + aby * vyi)

        wd = wall_accel_dX(X[i:i + 1], P)[0]
        gx += abx * wd[0]
        gy += aby * wd[1]

        gX[i] = (gx, gy)
        gV[i] = (gvx, gvy)
        grho[i] = grh

    # ---- design (alpha -> theta, x) ---------------------------------------
    gtheta, gXa = interp_alpha_vjp(theta, X, P, galpha)
    gX += gXa

    # ---- pass 2:  x̄_i += sum_j m F_ij d_ij (ρ̄_i + ρ̄_j) --------------------
    for i in range(N):
        xi, yi = X[i]
        gri = grho[i]
        gx = gy = 0.0
        for j in range(N):
            if j == i:
                continue
            dx = xi - X[j, 0]
            dy = yi - X[j, 1]
            r2 = dx * dx + dy * dy
            if r2 < np.finfo(float).eps * h * h:
                continue
            r = np.sqrt(r2)
            q = r / h
            if q >= 2.0:
                continue
            w = mass * float(F_of_q(np.array(q), A, h)) * (gri + grho[j])
            gx += w * dx
            gy += w * dy
        gX[i, 0] += gx
        gX[i, 1] += gy

    return gX, gV, gtheta


def simulate_gather(X0, V0, theta, P, nsteps):
    X, V = X0.copy(), V0.copy()
    tape = []
    for _ in range(nsteps):
        tape.append((X.copy(), V.copy()))
        a = accel(X, V, theta, P)
        V = V + P.dt * a
        X = X + P.dt * V
    return X, V, tape


def backward_gather(tape, theta, P, gXT, gVT):
    gX, gV = gXT.copy(), gVT.copy()
    gtheta = np.zeros_like(theta)
    for X, V in reversed(tape):
        gV = gV + P.dt * gX          # ḡV' = ḡV + dt ḡX
        abar = P.dt * gV             # ā   = dt ḡV'
        dX, dV, dth = accel_vjp_gather(X, V, theta, P, abar)
        gX = gX + dX
        gV = gV + dV
        gtheta += dth
    return gX, gV, gtheta


def main():
    rng = np.random.default_rng(0)
    P = Params()
    nx, ny = 7, 7
    xs = np.linspace(0.12, 0.12 + (nx - 1) * 0.05, nx)
    ys = np.linspace(0.12, 0.12 + (ny - 1) * 0.05, ny)
    XX, YY = np.meshgrid(xs, ys)
    X0 = np.stack([XX.ravel(), YY.ravel()], axis=1) + 0.004 * rng.standard_normal((nx * ny, 2))
    V0 = 0.05 * rng.standard_normal(X0.shape)
    N = X0.shape[0]
    theta = 3.0 * rng.random((P.ngy, P.ngx))
    nsteps = 25
    cX = rng.standard_normal((N, 2))
    cV = rng.standard_normal((N, 2))

    XT, VT, tape = simulate_gather(X0, V0, theta, P, nsteps)
    gX, gV, gth = backward_gather(tape, theta, P, cX, cV)

    tX0 = torch.tensor(X0, dtype=torch.float64, requires_grad=True)
    tV0 = torch.tensor(V0, dtype=torch.float64, requires_grad=True)
    tth = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    J = torch_simulate(tX0, tV0, tth, P, nsteps,
                       torch.tensor(cX, dtype=torch.float64),
                       torch.tensor(cV, dtype=torch.float64))
    J.backward()

    def relerr(a, b):
        return np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30)

    print("gather 形式（= Julia カーネルと同じ式）vs PyTorch 自動微分")
    print(f"  dJ/dX0    相対誤差 = {relerr(gX, tX0.grad.numpy()):.3e}")
    print(f"  dJ/dV0    相対誤差 = {relerr(gV, tV0.grad.numpy()):.3e}")
    print(f"  dJ/dtheta 相対誤差 = {relerr(gth, tth.grad.numpy()):.3e}")


if __name__ == "__main__":
    main()
