"""
ckf.py – Cubature Kalman Filter (CKF) for BEE 5214 Lab 4.

Algorithm (Ienkaran Arasaratnam & Simon Haykin, 2009):
  For n states, generate 2n cubature points:
    ξᵢ = √n · eᵢ  (i-th column of √n · I)

  Time update:
    Xᵢ = chol(P) · ξᵢ + x̂         propagate → Xᵢ* = f(Xᵢ, u)
    x̂⁻ = mean(Xᵢ*)
    P⁻ = cov(Xᵢ*) + Q

  Measurement update:
    Yᵢ = h(Xᵢ*)
    ŷ = mean(Yᵢ)
    Pyy, Pxy = cross-covariances
    K = Pxy · Pyy⁻¹
    x̂ = x̂⁻ + K·(y − ŷ)
    P = P⁻ − K·Pyy·Kᵀ
"""

import numpy as np


class CKF:
    """
    Cubature Kalman Filter – general n-state, m-measurement implementation.

    Parameters
    ----------
    f    : callable  x_next = f(x, u)  – discrete-time propagation function
    h    : callable  y = h(x)          – measurement function
    n    : int       state dimension
    m    : int       measurement dimension
    Q    : (n, n)    process noise covariance
    R    : (m, m)    measurement noise covariance
    x0   : (n,)      initial state estimate
    P0   : (n, n)    initial error covariance
    """

    def __init__(self,
                 f, h,
                 n: int, m: int,
                 Q: np.ndarray, R: np.ndarray,
                 x0: np.ndarray, P0: np.ndarray):
        self.f  = f
        self.h  = h
        self.n  = n
        self.m  = m
        self.Q  = Q.copy()
        self.R  = R.copy()
        self.x  = x0.copy()
        self.P  = P0.copy()

        self._num_pts = 2 * n       # number of cubature points
        self._Xi_star = None        # cached propagated points (set in predict)

    # ----------------------------------------------------------------- helpers
    def _cholesky(self, M: np.ndarray) -> np.ndarray:
        """Numerically stable Cholesky: symmetrise and add jitter if needed."""
        M_sym = 0.5 * (M + M.T)
        try:
            return np.linalg.cholesky(M_sym)
        except np.linalg.LinAlgError:
            return np.linalg.cholesky(M_sym + 1e-8 * np.eye(self.n))

    def _generate_sigma_points(self) -> np.ndarray:
        """Return (2n, n) array of cubature sigma points."""
        n   = self.n
        S   = self._cholesky(self.P)
        Xi  = np.empty((2 * n, n))
        sq  = np.sqrt(float(n))
        for i in range(n):
            delta        = sq * S[:, i]
            Xi[i]        = self.x + delta
            Xi[n + i]    = self.x - delta
        return Xi

    # ----------------------------------------------------------------- predict
    def predict(self, u: np.ndarray) -> None:
        """
        CKF time-update step.

        u : (4,)  current control input (motor voltages) – passed to f(x, u)
        """
        n  = self.n
        N2 = self._num_pts

        # Generate and propagate sigma points
        Xi       = self._generate_sigma_points()
        Xi_star  = np.array([self.f(Xi[i], u) for i in range(N2)])

        # Predicted mean
        x_pred   = Xi_star.mean(axis=0)

        # Predicted covariance
        dX       = Xi_star - x_pred                     # (2n, n)
        P_pred   = (dX.T @ dX) / N2 + self.Q

        self.x       = x_pred
        self.P       = P_pred
        self._Xi_star = Xi_star     # cache for measurement update

    # ------------------------------------------------------------------ update
    def update(self, y: np.ndarray) -> None:
        """
        CKF measurement-update step.

        y : (m,)  actual measurement vector
        """
        n  = self.n
        N2 = self._num_pts

        Xi_star = self._Xi_star

        # Predicted measurements
        Yi      = np.array([self.h(Xi_star[i]) for i in range(N2)])
        y_pred  = Yi.mean(axis=0)

        # Cross-covariances
        dY = Yi - y_pred          # (2n, m)
        dX = Xi_star - self.x     # (2n, n)

        Pyy = (dY.T @ dY) / N2 + self.R   # (m, m)
        Pxy = (dX.T @ dY) / N2             # (n, m)

        # Kalman gain
        K = Pxy @ np.linalg.inv(Pyy)       # (n, m)

        # Innovation – wrap angular channels to (−π, π)
        innov = y - y_pred
        for idx in [0, 1, 2]:              # φ, θ, ψ are the first 3 measurements
            if idx < len(innov):
                innov[idx] = (innov[idx] + np.pi) % (2.0 * np.pi) - np.pi

        # State and covariance update
        self.x = self.x + K @ innov
        self.P = self.P - K @ Pyy @ K.T

        # Enforce symmetry and positive definiteness
        self.P = 0.5 * (self.P + self.P.T) + 1e-9 * np.eye(n)

    # --------------------------------------------------------------- accessors
    @property
    def state(self) -> np.ndarray:
        """Current state estimate  x̂  (copy)."""
        return self.x.copy()

    @property
    def covariance(self) -> np.ndarray:
        """Current error covariance  P  (copy)."""
        return self.P.copy()
