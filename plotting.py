"""
plotting.py – Matplotlib plots for BEE 5214 Lab 4 simulation results.

plot_43(t, X_true, X_est, V_hist)  → required plots for §4.3
plot_44(t, X_true, X_est, V_hist)  → required plots for §4.4
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi':    130,
    'axes.grid':     True,
    'grid.alpha':    0.35,
    'grid.linestyle':'--',
    'lines.linewidth': 1.6,
    'font.size':     10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
})

COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
MOTOR_LABELS = ['M1 (FR-CW)', 'M2 (FL-CCW)', 'M3 (RL-CW)', 'M4 (RR-CCW)']


# ──────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _motor_subplots(t, data, est, title, ylabel_fmt, unit, save_path=None):
    """Generic 4-panel motor plot (true vs CKF estimate)."""
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.01)
    for j in range(4):
        ax = axes[j]
        ax.plot(t, data[:, j], color=COLORS[j], label='True')
        if est is not None:
            ax.plot(t, est[:, j], '--', color=COLORS[j],
                    alpha=0.65, label='CKF est.')
        ax.set_ylabel(ylabel_fmt.format(j + 1, unit))
        ax.legend(fontsize=8, loc='upper right')
        ax.set_title(MOTOR_LABELS[j], fontsize=9, loc='left')
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  Section 4.3 plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_43(t, X_true, X_est, V_hist,
            save: bool = False, prefix: str = '43'):
    """
    Produce all required §4.3 plots:
      (i)   Motor voltages
      (ii)  Motor currents
      (iii) Propeller speeds
      (iv)  Altitude Z(t)
      (v)   Roll, pitch, yaw
    """
    figs = []

    # (i) Motor voltages ──────────────────────────────────────────────────────
    fig = _motor_subplots(t, V_hist, None,
                          '§4.3 – Motor Voltages',
                          'V{} ({})', 'V',
                          f'{prefix}_motor_voltages.png' if save else None)
    figs.append(fig)

    # (ii) Motor currents ─────────────────────────────────────────────────────
    fig = _motor_subplots(t, X_true[:, 0:4], X_est[:, 0:4],
                          '§4.3 – Motor Currents',
                          'i{} ({})', 'A',
                          f'{prefix}_motor_currents.png' if save else None)
    figs.append(fig)

    # (iii) Propeller speeds ──────────────────────────────────────────────────
    fig = _motor_subplots(t, X_true[:, 4:8], X_est[:, 4:8],
                          '§4.3 – Propeller Speeds',
                          'ω{} ({})', 'rad/s',
                          f'{prefix}_propeller_speeds.png' if save else None)
    figs.append(fig)

    # (iv) Altitude ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, X_true[:, 19], label='True Z')
    ax.plot(t, X_est[:,  19], '--', label='CKF estimate', alpha=0.8)
    ax.axhline(0.0, color='k', linestyle=':', linewidth=1.2, label='Ground (0 m)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude Z (m)')
    ax.set_title('§4.3 – Altitude Response')
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(f'{prefix}_altitude.png', bbox_inches='tight')
    figs.append(fig)

    # (v) Roll, pitch, yaw ────────────────────────────────────────────────────
    angle_names = ['Roll φ', 'Pitch θ', 'Yaw ψ']
    idx_true    = [11, 12, 13]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fig.suptitle('§4.3 – Euler Angle Response', fontsize=12,
                 fontweight='bold', y=1.01)
    for j, (name, idx) in enumerate(zip(angle_names, idx_true)):
        ax = axes[j]
        ax.plot(t, np.degrees(X_true[:, idx]), color=COLORS[j], label='True')
        ax.plot(t, np.degrees(X_est[:,  idx]), '--', color=COLORS[j],
                alpha=0.65, label='CKF est.')
        ax.axhline(0.0, color='k', linestyle=':', linewidth=1.0)
        ax.set_ylabel(f'{name} (°)')
        ax.legend(fontsize=8, loc='upper right')
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    if save:
        fig.savefig(f'{prefix}_euler_angles.png', bbox_inches='tight')
    figs.append(fig)

    plt.show()
    return figs


# ──────────────────────────────────────────────────────────────────────────────
#  Section 4.4 plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_44(t, X_true, X_est, V_hist,
            z_ref:  float = 1.0,
            vx_ref: float = 0.5,
            save:   bool  = False,
            prefix: str   = '44'):
    """
    Produce all required §4.4 plots:
      (i)   Motor voltages
      (ii)  Motor currents
      (iii) Propeller speeds
      (iv)  Altitude Z(t)
      (v)   Forward velocity vx(t)
      (vi)  Pitch angle θ(t)
    """
    figs = []

    # (i) Motor voltages ──────────────────────────────────────────────────────
    fig = _motor_subplots(t, V_hist, None,
                          '§4.4 – Motor Voltages',
                          'V{} ({})', 'V',
                          f'{prefix}_motor_voltages.png' if save else None)
    figs.append(fig)

    # (ii) Motor currents ─────────────────────────────────────────────────────
    fig = _motor_subplots(t, X_true[:, 0:4], X_est[:, 0:4],
                          '§4.4 – Motor Currents',
                          'i{} ({})', 'A',
                          f'{prefix}_motor_currents.png' if save else None)
    figs.append(fig)

    # (iii) Propeller speeds ──────────────────────────────────────────────────
    fig = _motor_subplots(t, X_true[:, 4:8], X_est[:, 4:8],
                          '§4.4 – Propeller Speeds',
                          'ω{} ({})', 'rad/s',
                          f'{prefix}_propeller_speeds.png' if save else None)
    figs.append(fig)

    # (iv) Altitude ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, X_true[:, 19], label='True Z')
    ax.plot(t, X_est[:,  19], '--', label='CKF estimate', alpha=0.8)
    ax.axhline(z_ref, color='k', linestyle=':', linewidth=1.2,
               label=f'z_ref = {z_ref} m')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude Z (m)')
    ax.set_title('§4.4 – Altitude Response')
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(f'{prefix}_altitude.png', bbox_inches='tight')
    figs.append(fig)

    # (v) Forward velocity ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, X_true[:, 14], label='True vx')
    ax.plot(t, X_est[:,  14], '--', label='CKF estimate', alpha=0.8)
    ax.axhline(vx_ref, color='k', linestyle=':', linewidth=1.2,
               label=f'vx_ref = {vx_ref} m/s')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Forward velocity vx (m/s)')
    ax.set_title('§4.4 – Forward Velocity Response')
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(f'{prefix}_forward_velocity.png', bbox_inches='tight')
    figs.append(fig)

    # (vi) Pitch ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, np.degrees(X_true[:, 12]), label='True θ')
    ax.plot(t, np.degrees(X_est[:,  12]), '--', label='CKF estimate', alpha=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pitch θ (°)')
    ax.set_title('§4.4 – Pitch Angle Response (set by velocity controller)')
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(f'{prefix}_pitch.png', bbox_inches='tight')
    figs.append(fig)

    plt.show()
    return figs
