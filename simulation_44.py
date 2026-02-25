"""
simulation_44.py – Section 4.4: CKF + PI Altitude + Attitude + Forward Velocity.
"""
import numpy as np
from params import DroneParams
from dynamics import euler_step
from ckf import CKF
from controllers import AttitudeController, AltitudeController, VelocityController, voltages_from_wrench

def _build_ckf(p: DroneParams, x0: np.ndarray) -> CKF:
    n = 20
    Q = np.diag(np.concatenate([[5e-3]*4, [5e-1]*4, [1e-4]*3, [1e-5]*3, [1e-3]*3, [1e-4]*3]))
    R_meas = np.diag([1e-4, 1e-4, 1e-4, 2e-3, 5e-3])
    P0 = np.diag(np.concatenate([[1e-2]*4, [1e2]*4, [1e-4]*3, [1e-5]*3, [1e-3]*3, [1e-4]*3]))
    
    def f(x, u): return euler_step(x, u, p)
    def h(x): return np.array([x[11], x[12], x[13], x[19], x[14]])
    return CKF(f, h, n, 5, Q, R_meas, x0, P0)

def run_simulation_44(t_end=15.0, z_ref=1.0, vx_ref=0.5, phi_ref=0.0, psi_ref=0.0, verbose=True):
    p, Ts = DroneParams(), DroneParams().Ts
    N = int(t_end / Ts)
    t = np.arange(N) * Ts

    x = np.zeros(20)
    x[0:4] = p.i_hover              # FIX: Initialize motor currents!
    x[4:8] = p.omega_hover          
    rng = np.random.default_rng(0)
    x[11] = np.deg2rad(2.0)        

    ctrl_alt, ctrl_vel, ctrl_att = AltitudeController(p), VelocityController(p), AttitudeController(p)
    ckf = _build_ckf(p, x.copy())

    X_true, X_est, V_hist = np.zeros((N, 20)), np.zeros((N, 20)), np.zeros((N, 4))
    milestones = set(int(N * q) for q in [0.2, 0.4, 0.6, 0.8, 1.0])

    for k in range(N):
        x_hat = ckf.state
        T_cmd = ctrl_alt.update(z_ref, x_hat[19])
        th_ref = ctrl_vel.update(vx_ref, x_hat[14])
        tau = ctrl_att.update(phi_ref, th_ref, psi_ref, x_hat[11], x_hat[12], x_hat[13])
        
        V = voltages_from_wrench(T_cmd, tau, p)
        x = euler_step(x, V, p)
        
        noise = rng.normal(0, [0.01, 0.01, 0.01, 0.03, 0.05])
        y = np.array([x[11], x[12], x[13], x[19], x[14]]) + noise
        
        ckf.predict(V)
        ckf.update(y)

        X_true[k], X_est[k], V_hist[k] = x, ckf.state, V

        if verbose and k in milestones:
            pct = int(100 * (k + 1) / N)
            print(f"    [4.4]  {pct:3d}%  t={t[k]:.2f}s  Z={x[19]:.3f} m  vx={x[14]:+.3f} m/s")

    return t, X_true, X_est, V_hist