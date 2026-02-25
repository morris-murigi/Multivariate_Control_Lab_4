"""
controllers.py – PI controllers and motor voltage mixer for BEE 5214 Lab 4.
"""
import numpy as np
from params import DroneParams

class PIController:
    def __init__(self, Kp: float, Ki: float, Ts: float, out_min=-np.inf, out_max=np.inf, int_min=-np.inf, int_max=np.inf):
        self.Kp, self.Ki, self.Ts = Kp, Ki, Ts
        self.out_min, self.out_max, self.int_min, self.int_max = out_min, out_max, int_min, int_max
        self._integ = 0.0

    def reset(self) -> None: self._integ = 0.0

    def update(self, error: float) -> float:
        self._integ = float(np.clip(self._integ + error * self.Ts, self.int_min, self.int_max))
        return float(np.clip(self.Kp * error + self.Ki * self._integ, self.out_min, self.out_max))
class AttitudeController:
    """
    Three-axis PI attitude controller.
    Goldilocks Tuning: Balances noise-rejection with Forward Euler stability.
    """
    def __init__(self, p: DroneParams,
                 Kp_phi=0.25,  Ki_phi=0.01,  # The perfect middle ground
                 Kp_th=0.25,   Ki_th=0.01,   
                 Kp_psi=0.15,  Ki_psi=0.005): 

        tau_lim = 0.5   
        int_lim = 0.25  

        self.phi_ctrl = PIController(Kp_phi, Ki_phi, p.Ts, -tau_lim, tau_lim, -int_lim, int_lim)
        self.th_ctrl  = PIController(Kp_th,  Ki_th,  p.Ts, -tau_lim, tau_lim, -int_lim, int_lim)
        self.psi_ctrl = PIController(Kp_psi, Ki_psi, p.Ts, -tau_lim, tau_lim, -int_lim, int_lim)

    def update(self, phi_ref, th_ref, psi_ref, phi, th, psi):
        return np.array([self.phi_ctrl.update(phi_ref - phi), self.th_ctrl.update(th_ref - th), self.psi_ctrl.update(psi_ref - psi)])
    def reset(self): self.phi_ctrl.reset(); self.th_ctrl.reset(); self.psi_ctrl.reset()

class AltitudeController:
    def __init__(self, p: DroneParams, Kp=2.0, Ki=0.5):
        T_max, int_lim = 2.0 * p.m * p.g, p.m * p.g * 0.8
        self._ctrl = PIController(Kp, Ki, p.Ts, out_min=-p.m * p.g, out_max= p.m * p.g, int_min=-int_lim, int_max= int_lim)
        self._mg, self._T_max = p.m * p.g, T_max

    def update(self, z_ref: float, z_hat: float) -> float:
        return float(np.clip(self._mg + self._ctrl.update(z_ref - z_hat), 0.0, self._T_max))
    def reset(self): self._ctrl.reset()

class VelocityController:
    def __init__(self, p: DroneParams, Kp=0.25, Ki=0.04):
        th_max = np.deg2rad(20)   
        self._ctrl = PIController(Kp, Ki, p.Ts, out_min=-th_max, out_max=th_max, int_min=-th_max, int_max=th_max)

    def update(self, vx_ref: float, vx_hat: float) -> float: return self._ctrl.update(vx_ref - vx_hat)
    def reset(self): self._ctrl.reset()

def voltages_from_wrench(T_cmd: float, tau: np.ndarray, p: DroneParams) -> np.ndarray:
    omega2_des = np.maximum(p.A_X_inv @ np.array([T_cmd, tau[0], tau[1], tau[2]]), 0.0)      
    omega_des  = np.sqrt(omega2_des)
    return np.clip(p.R * ((p.bm * omega_des + p.kQ * omega2_des) / p.kt) + p.kb * omega_des, 0.0, p.V_max)