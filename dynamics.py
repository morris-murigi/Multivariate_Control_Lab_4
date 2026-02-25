"""
dynamics.py – Continuous-time EOM and forward-Euler step for BEE 5214 Lab 4.

All equations follow Section 4.1 / 4.2 of the lab sheet exactly.
"""

import numpy as np
from params import DroneParams


# ──────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _thrust_and_torques(x: np.ndarray, p: DroneParams):
    """
    Compute total thrust T and body torques [τφ, τθ, τψ] from the state.

    Uses X-configuration torque equations (Lab §4.1):
        τφ = (l/√2)·kT·(−ω₁² + ω₂² + ω₃² − ω₄²)
        τθ = (l/√2)·kT·(+ω₁² + ω₂² − ω₃² − ω₄²)
        τψ = kQ·(+ω₁² − ω₂² + ω₃² − ω₄²)
    """
    w2  = x[4:8] ** 2          # [ω₁², ω₂², ω₃², ω₄²]
    le  = p.l / np.sqrt(2.0)

    T       = p.kT * np.sum(w2)
    tau_phi = p.kT * le * (-w2[0] + w2[1] + w2[2] - w2[3])
    tau_th  = p.kT * le * ( w2[0] + w2[1] - w2[2] - w2[3])
    tau_psi = p.kQ       * ( w2[0] - w2[1] + w2[2] - w2[3])
    return T, tau_phi, tau_th, tau_psi


# ──────────────────────────────────────────────────────────────────────────────
#  Continuous-time derivative
# ──────────────────────────────────────────────────────────────────────────────

def state_derivative(x: np.ndarray, V: np.ndarray, p: DroneParams) -> np.ndarray:
    """
    Continuous-time state derivative  ẋ = f(x, V).

    Parameters
    ----------
    x : (20,)  full state vector
    V : (4,)   motor input voltages  [V1, V2, V3, V4]
    p : DroneParams

    Returns
    -------
    dx : (20,)
    """
    dx = np.zeros(20)

    # Unpack state
    i_m              = x[0:4]            # motor currents
    w_m              = x[4:8]            # propeller speeds
    p_r, q_r, r_r    = x[8], x[9], x[10]  # body angular rates
    phi, th, psi     = x[11], x[12], x[13]
    vx, vy, vz       = x[14], x[15], x[16]

    T, tau_phi, tau_th, tau_psi = _thrust_and_torques(x, p)

    # ── Motor electrical  ẋ = (1/L)(−R·i − kb·ω + V)  ─────────────────────
    dx[0:4] = (1.0 / p.L) * (-p.R * i_m - p.kb * w_m + V)

    # ── Motor mechanical  ẋ = (1/Jm)(kt·i − bm·ω − kQ·ω²)  ────────────────
    dx[4:8] = (1.0 / p.Jm) * (p.kt * i_m - p.bm * w_m - p.kQ * w_m**2)

    # ── Trig shorthand  ─────────────────────────────────────────────────────
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth,  sth  = np.cos(th),  np.sin(th)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    # ── Translational dynamics  m·v̇ = f(T, angles) − drag  ─────────────────
    dx[14] = (T * (cphi * sth * cpsi + sphi * spsi) - p.kdx * vx) / p.m
    dx[15] = (T * (cphi * sth * spsi - sphi * cpsi) - p.kdy * vy) / p.m
    dx[16] = (T * (cphi * cth)        - p.m * p.g   - p.kdz * vz) / p.m

    # ── Position kinematics  ─────────────────────────────────────────────────
    dx[17] = vx
    dx[18] = vy
    dx[19] = vz

    # ── Rotational dynamics (gyroscopic coupling)  ───────────────────────────
    dx[8]  = (tau_phi + (p.Iy - p.Iz) * q_r * r_r - p.kphi   * p_r) / p.Ix
    dx[9]  = (tau_th  + (p.Iz - p.Ix) * p_r * r_r - p.ktheta * q_r) / p.Iy
    dx[10] = (tau_psi + (p.Ix - p.Iy) * p_r * q_r - p.kpsi   * r_r) / p.Iz

    # ── Euler angle kinematics  ──────────────────────────────────────────────
    tan_th = np.tan(th)
    dx[11] = p_r + q_r * sphi * tan_th + r_r * cphi * tan_th
    dx[12] = q_r * cphi - r_r * sphi
    dx[13] = (q_r * sphi + r_r * cphi) / (cth + 1e-9)   # guard against θ = ±90°

    return dx


# ──────────────────────────────────────────────────────────────────────────────
#  Forward-Euler step
# ──────────────────────────────────────────────────────────────────────────────

def euler_step(x: np.ndarray, V: np.ndarray, p: DroneParams) -> np.ndarray:
    """
    One forward-Euler step:  x_{k+1} = x_k + Ts · f(x_k, V_k)

    Voltage is clipped to [0, V_max] before integration.
    """
    V_safe = np.clip(V, 0.0, p.V_max)
    return x + p.Ts * state_derivative(x, V_safe, p)
