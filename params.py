"""
params.py – Physical constants and drone parameters for BEE 5214 Lab 4.

State vector layout (length 20):
  x[ 0: 4] = [i1, i2, i3, i4]      motor currents      (A)
  x[ 4: 8] = [ω1, ω2, ω3, ω4]      propeller speeds    (rad/s)
  x[ 8:11] = [p,  q,  r ]           body angular rates  (rad/s)
  x[11:14] = [φ,  θ,  ψ ]           Euler angles        (rad)
  x[14:17] = [vx, vy, vz]           translational vels  (m/s)
  x[17:20] = [X,  Y,  Z ]           world position      (m)

Motor order (X configuration, refer to Fig 4.1):
  M1 – front-right, CW    M2 – front-left,  CCW
  M3 – rear-left,   CW    M4 – rear-right,  CCW
"""

import numpy as np


class DroneParams:
    # ------------------------------------------------------------------ Motor
    R  = 0.1       # Winding resistance       (Ω)
    L  = 0.002     # Winding inductance       (H)
    kb = 0.01      # Back-EMF constant        (Vs/rad)
    kt = 0.01      # Torque constant          (Nm/A)
    bm = 0.00009   # Motor viscous damping    (Ns/rad)
    Jm = 0.0004    # Rotor + propeller inertia (kg·m²)
    kQ = 0.000002  # Propeller drag coeff     (Ns²/rad²)
    kT = 0.000018  # Propeller thrust coeff   (Ns²/rad²)

    # ------------------------------------------------------------------ Body
    m  = 1.5       # Total mass               (kg)
    Ix = 0.02      # Roll  inertia            (kg·m²)
    Iy = 0.02      # Pitch inertia            (kg·m²)
    Iz = 0.04      # Yaw   inertia            (kg·m²)
    g  = 9.81      # Gravitational accel      (m/s²)
    l  = 0.1       # Arm length               (m)

    # -------------------------------------------------------- Damping coeffs
    kdx    = 0.15   # Translational damping x (Ns/m)
    kdy    = 0.15   # Translational damping y
    kdz    = 0.25   # Translational damping z
    kphi   = 0.002  # Angular damping – roll  (Ns/rad)
    ktheta = 0.002  # Angular damping – pitch
    kpsi   = 0.003  # Angular damping – yaw

    # ------------------------------------------------------- Discretisation
    Ts = 0.004      # Sampling period          (s)

    # -------------------------------------------------------- Actuation limits
    V_max = 12.0    # Maximum motor voltage    (V)
    i_max = 20.0    # Maximum motor current    (A)

    # ================================================ Derived hover quantities
    @property
    def omega_hover(self) -> float:
        """Propeller speed needed to hover  (rad/s).
        Solved from: 4 * kT * ω₀² = m·g
        """
        return float(np.sqrt(self.m * self.g / (4.0 * self.kT)))

    @property
    def i_hover(self) -> float:
        """Steady-state motor current at hover  (A)."""
        w = self.omega_hover
        return (self.bm * w + self.kQ * w**2) / self.kt

    @property
    def V_hover(self) -> float:
        """Steady-state motor voltage at hover  (V)."""
        w = self.omega_hover
        i = self.i_hover
        return self.R * i + self.kb * w

    # ====================================================== Allocation matrix
    @property
    def A_X(self) -> np.ndarray:
        """
        Allocation matrix  A_X  (4×4).

        Maps  [ω₁², ω₂², ω₃², ω₄²]  →  [T, τφ, τθ, τψ]

        Derived from X-config geometry (l_eff = l / √2):
          T   =  kT*(ω₁² + ω₂² + ω₃² + ω₄²)
          τφ  =  kT·l_eff*(-ω₁² + ω₂² + ω₃² - ω₄²)
          τθ  =  kT·l_eff*(+ω₁² + ω₂² - ω₃² - ω₄²)
          τψ  =  kQ*(+ω₁² - ω₂² + ω₃² - ω₄²)
        """
        le = self.l / np.sqrt(2.0)
        kT, kQ = self.kT, self.kQ
        return np.array([
            [ kT,      kT,      kT,      kT    ],   # Thrust
            [-kT*le,   kT*le,   kT*le,  -kT*le ],   # Roll τφ
            [ kT*le,   kT*le,  -kT*le,  -kT*le ],   # Pitch τθ
            [ kQ,     -kQ,      kQ,     -kQ    ],   # Yaw τψ
        ])

    @property
    def A_X_inv(self) -> np.ndarray:
        """Pseudo-inverse of the allocation matrix."""
        return np.linalg.pinv(self.A_X)

    def __repr__(self):
        return (f"DroneParams(m={self.m} kg, ω_hover={self.omega_hover:.1f} rad/s, "
                f"V_hover={self.V_hover:.2f} V)")
