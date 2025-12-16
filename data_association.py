from entities import *
from utils import *
import numpy as np
from typing import Optional

# assumes you already have:
# - wrap_angle(a: float) -> float
# - landmark_slice(j: int) -> slice
# - get_landmark_xy(mu: np.ndarray, j: int) -> np.ndarray  (returns [lx, ly].copy())
# - Observation(r: float, b: float, landmark_id: Optional[int] = None)
# - SLAMState(mu: np.ndarray, Sigma: np.ndarray, inited: np.ndarray)

class DataAssociation:
    """
    Mahalanobis-distance data association (range-bearing).
    For each initialized landmark j:
      - predict measurement z_hat(j)
      - compute innovation nu = z - z_hat
      - compute S = H Sigma H^T + R
      - compute d^2 = nu^T S^{-1} nu
    Choose smallest d^2; accept if < gate_d2 else None.
    """

    def __init__(self, R: np.ndarray, gate_d2: float = 16.0):
        """
        Inputs:
          R: 2x2 measurement noise covariance (range,bearing)
          gate_d2: acceptance threshold on Mahalanobis distance squared
                   (default ~ chi-square 2 dof @ 0.99)
        """
        self.R = np.array(R, dtype=float)
        self.gate_d2 = float(gate_d2)

    def associate(self, state, z) -> Optional[int]:
        """
        Inputs:
          state: SLAMState (mu, Sigma, inited)
          z: Observation (r,b)
        Output:
          matched landmark index j, or None (treat as new landmark)
        """
        inited_idx = np.where(state.inited)[0]
        if len(inited_idx) == 0:
            return None

        mu = state.mu
        Sigma = state.Sigma
        dim = len(mu)

        z_meas = np.array([float(z.r), float(z.b)], dtype=float)

        best_j = None
        best_d2 = float("inf")

        for j in inited_idx:
            j = int(j)
            z_hat, H = self._predict_measurement_and_H(mu, j, dim)

            nu = z_meas - z_hat
            nu[1] = wrap_angle(nu[1])  # bearing residual wrap

            S = H @ Sigma @ H.T + self.R

            try:
                sol = np.linalg.solve(S, nu)  # S^{-1} nu
            except np.linalg.LinAlgError:
                continue

            d2 = float(nu.T @ sol)
            if d2 < best_d2:
                best_d2 = d2
                best_j = j

        if best_j is None:
            return None
        return best_j if best_d2 < self.gate_d2 else None

    def _predict_measurement_and_H(self, mu: np.ndarray, j: int, dim: int):
        """
        Range-bearing measurement model for landmark j.
        Returns:
          z_hat: np.array([r_hat, b_hat])
          H: (2 x dim) Jacobian wrt full SLAM state
        """
        x, y, th = float(mu[0]), float(mu[1]), float(mu[2])
        lx, ly = get_landmark_xy(mu, j)

        dx = float(lx) - x
        dy = float(ly) - y
        q = dx*dx + dy*dy
        q = max(q, 1e-12)
        r_hat = float(np.sqrt(q))
        b_hat = float(wrap_angle(np.arctan2(dy, dx) - th))

        z_hat = np.array([r_hat, b_hat], dtype=float)

        Hr = np.array([
            [-dx / r_hat, -dy / r_hat, 0.0],
            [ dy / q,     -dx / q,     -1.0],
        ], dtype=float)  # 2x3

        Hl = np.array([
            [ dx / r_hat,  dy / r_hat],
            [-dy / q,      dx / q],
        ], dtype=float)  # 2x2

        H = np.zeros((2, dim), dtype=float)
        H[:, 0:3] = Hr
        k = 3 + 2*j
        H[:, k:k+2] = Hl

        return z_hat, H
