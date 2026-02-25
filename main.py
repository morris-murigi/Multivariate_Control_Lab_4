"""
main.py – Entry point for BEE 5214 Lab 4 simulations.

Runs both §4.3 and §4.4, prints a performance summary, and shows all plots.
Plots are also saved as PNG files in the current directory.

Usage
-----
    python main.py            # run both
    python main.py --43-only  # §4.3 only
    python main.py --44-only  # §4.4 only
    python main.py --no-save  # skip saving PNGs
"""

import sys
import time
import numpy as np
from params import DroneParams
from simulation_43 import run_simulation_43
from simulation_44 import run_simulation_44
from plotting import plot_43, plot_44


# ──────────────────────────────────────────────────────────────────────────────
#  Performance summary helpers
# ──────────────────────────────────────────────────────────────────────────────

def _settle_time(t, signal, ref, band_frac=0.02):
    """Time for |signal − ref| to stay within band_frac*|ref| (or 0.05 m)."""
    band = max(abs(ref) * band_frac, 0.05)
    within = np.abs(signal - ref) < band
    # Find last time it leaves the band
    outside = np.where(~within)[0]
    if len(outside) == 0:
        return t[0]
    return float(t[outside[-1]])


def _overshoot_pct(signal, ref):
    """Peak overshoot percentage relative to step size."""
    if abs(ref) < 1e-9:
        return 0.0
    peak = np.max(np.abs(signal))
    return 100.0 * max(0.0, peak - abs(ref)) / abs(ref)


def _steady_state_error(signal, ref, last_frac=0.1):
    """Mean error over last 10% of simulation."""
    n  = len(signal)
    ss = signal[int(0.9 * n):]
    return float(np.mean(np.abs(ss - ref)))


def print_summary_43(t, X_true):
    print("\n  ── §4.3 Performance ──────────────────────────────────────────")
    for name, idx in [('Roll φ', 11), ('Pitch θ', 12), ('Yaw ψ', 13)]:
        sig = np.degrees(X_true[:, idx])
        ts  = _settle_time(t, sig, 0.0, band_frac=0.02)
        sse = _steady_state_error(sig, 0.0)
        print(f"    {name:<10}: settle = {ts:.2f} s  |  SS error = {sse:.4f}°")
    print()


def print_summary_44(t, X_true, z_ref, vx_ref):
    print("\n  ── §4.4 Performance ──────────────────────────────────────────")
    # Altitude
    Z   = X_true[:, 19]
    ts  = _settle_time(t, Z, z_ref)
    osh = _overshoot_pct(Z, z_ref)
    sse = _steady_state_error(Z, z_ref)
    print(f"    Altitude Z  : settle = {ts:.2f} s  |  overshoot = {osh:.1f}%  "
          f"|  SS error = {sse:.4f} m")
    # Velocity
    vx  = X_true[:, 14]
    ts  = _settle_time(t, vx, vx_ref)
    sse_pct = 100.0 * _steady_state_error(vx, vx_ref) / abs(vx_ref)
    print(f"    Velocity vx : settle = {ts:.2f} s  "
          f"|  SS error = {sse_pct:.1f}% of ref")
    # Pitch (attitude SS error)
    th_ss = np.mean(np.abs(np.degrees(X_true[int(0.9 * len(t)):, 12])))
    print(f"    Pitch θ SS  : {th_ss:.4f}° residual")
    print()


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args      = sys.argv[1:]
    run_43    = '--44-only' not in args
    run_44    = '--43-only' not in args
    save_png  = '--no-save' not in args

    p = DroneParams()
    print("=" * 62)
    print("   BEE 5214 Lab 4 — CKF + PI Quadcopter Simulation")
    print("=" * 62)
    print(f"   {p}")
    print(f"   ω_hover = {p.omega_hover:.1f} rad/s  |  V_hover ≈ {p.V_hover:.2f} V")
    print()

    # ── Section 4.3 ──────────────────────────────────────────────────────────
    if run_43:
        print("▶  §4.3  CKF + PI Attitude Hover  (10 s)")
        print("   Note: CKF propagates 40 sigma points per step – may take ~30 s")
        t0 = time.perf_counter()
        t43, X43, Xest43, V43 = run_simulation_43(t_end=10.0)
        elapsed = time.perf_counter() - t0
        print(f"   Done in {elapsed:.1f} s")
        print_summary_43(t43, X43)
        plot_43(t43, X43, Xest43, V43, save=save_png, prefix='43')

    # ── Section 4.4 ──────────────────────────────────────────────────────────
    if run_44:
        z_ref, vx_ref = 1.0, 0.5
        print("▶  §4.4  CKF + PI Altitude + Velocity  (15 s)")
        print("   Note: CKF propagates 40 sigma points per step – may take ~45 s")
        t0 = time.perf_counter()
        t44, X44, Xest44, V44 = run_simulation_44(t_end=15.0,
                                                   z_ref=z_ref,
                                                   vx_ref=vx_ref)
        elapsed = time.perf_counter() - t0
        print(f"   Done in {elapsed:.1f} s")
        print_summary_44(t44, X44, z_ref, vx_ref)
        plot_44(t44, X44, Xest44, V44,
                z_ref=z_ref, vx_ref=vx_ref,
                save=save_png, prefix='44')

    if save_png:
        print("   PNG files saved to current directory.")


if __name__ == '__main__':
    main()
