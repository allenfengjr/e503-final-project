# -----------------------------
# EKF-SLAM core (3 steps: predict, associate, update/init)
# -----------------------------
from entities import *
from utils import *
import numpy as np

import numpy as np

# assumes these exist in your codebase:
# - wrap_angle(a: float) -> float  (recommended range: (-pi, pi])
# - landmark_slice(j: int) -> slice
# - SLAMState(mu: np.ndarray, Sigma: np.ndarray, inited: np.ndarray)
# - Control(v: float, omega: float)
# - Observation(r: float, b: float, landmark_id: Optional[int] = None)
# - NoiseParams(sigma_v, sigma_omega, sigma_r, sigma_b)



class EKFSLAM:
    def __init__(self, n_landmarks: int, noise):
        self.N = n_landmarks
        self.dim = 3 + 2 * n_landmarks
        self.state = SLAMState(
            mu=np.zeros(self.dim, dtype=float),
            Sigma=np.eye(self.dim, dtype=float) * 1e-3,
            inited=np.zeros(n_landmarks, dtype=bool),
        )
        self.noise = noise

        # covariance hyper-parameters (constant)
        self.Q = np.diag([noise.sigma_v**2, noise.sigma_omega**2])  # control/motion noise
        self.R = np.diag([noise.sigma_r**2, noise.sigma_b**2])      # measurement noise
        # landmark matching
        self._step_k = 0
        self.pending = {}  # key -> {"count": int, "last_k": int, "z": Observation}
        self.pending_confirm = 2   # need 2 hits to init
        self.pending_ttl = 2       # keep pending for 2 steps

        # quantization for pending keys (tune if needed)
        self.pending_r_bin = 0.5               # meters
        self.pending_b_bin = np.deg2rad(10.0)  # radians


    def set_initial_estimate(self, mu0: np.ndarray, Sigma0: np.ndarray):
        self.state.mu = mu0.copy()
        self.state.Sigma = Sigma0.copy()

    def predict(self, u, dt: float) -> None:
        """
        Step 1: prediction.
        Inputs:
          u: Control(v, omega)
          dt: time step
        Updates:
          self.state.mu, self.state.Sigma
        """
        mu = self.state.mu
        Sigma = self.state.Sigma

        x, y, th = mu[0], mu[1], mu[2]
        v, w = float(u.v), float(u.omega)

        # --- predict mean with unicycle model ---
        x2 = x + v * dt * np.cos(th)
        y2 = y + v * dt * np.sin(th)
        th2 = wrap_angle(th + w * dt)

        mu[0], mu[1], mu[2] = x2, y2, th2

        # --- Jacobians wrt robot state and control ---
        Fx = np.array([
            [1.0, 0.0, -v * dt * np.sin(th)],
            [0.0, 1.0,  v * dt * np.cos(th)],
            [0.0, 0.0, 1.0],
        ], dtype=float)

        Fu = np.array([
            [dt * np.cos(th), 0.0],
            [dt * np.sin(th), 0.0],
            [0.0, dt],
        ], dtype=float)

        # --- embed into full SLAM dimension ---
        F = np.eye(self.dim, dtype=float)
        F[0:3, 0:3] = Fx

        G = np.zeros((self.dim, 2), dtype=float)
        G[0:3, :] = Fu

        Sigma = F @ Sigma @ F.T + G @ self.Q @ G.T
        self.state.Sigma = symmetrize(Sigma)

    def init_landmark(self, j: int, z) -> None:
        """
        Initialize landmark j using current robot estimate and one observation (r,b).
        Inputs:
          j: landmark slot index [0..N-1]
          z: Observation(r, b)
        Updates:
          mu landmark entries and Sigma blocks (including cross-cov)
        """
        mu = self.state.mu
        Sigma = self.state.Sigma

        x, y, th = mu[0], mu[1], mu[2]
        r, b = float(z.r), float(z.b)

        phi = th + b
        lx = x + r * np.cos(phi)
        ly = y + r * np.sin(phi)

        set_landmark_xy(mu, j, np.array([lx, ly], dtype=float))


        # --- covariance initialization (simple but principled) ---
        # landmark depends on robot pose (x,y,theta) and measurement (r,b)
        Jr = np.array([
            [1.0, 0.0, -r * np.sin(phi)],
            [0.0, 1.0,  r * np.cos(phi)],
        ], dtype=float)  # 2x3

        Jz = np.array([
            [np.cos(phi), -r * np.sin(phi)],
            [np.sin(phi),  r * np.cos(phi)],
        ], dtype=float)  # 2x2

        # cross-cov with entire state through robot uncertainty
        Sigma_r_all = Sigma[0:3, :]                  # 3 x dim
        Sigma_l_all = Jr @ Sigma_r_all               # 2 x dim

        k = 3 + 2 * j
        Sigma[k:k+2, :] = Sigma_l_all
        Sigma[:, k:k+2] = Sigma_l_all.T

        # add measurement noise effect to landmark's own 2x2 block
        Sigma_rr = Sigma[0:3, 0:3]
        Sigma_ll = Jr @ Sigma_rr @ Jr.T + Jz @ self.R @ Jz.T
        Sigma[k:k+2, k:k+2] = Sigma_ll

        self.state.Sigma = symmetrize(Sigma)

    def update_landmark(self, j: int, z) -> None:
        """
        Measurement update for an already-initialized landmark j.
        Inputs:
          j: landmark slot index
          z: Observation(r,b)
        Updates:
          mu, Sigma
        """
        mu = self.state.mu
        Sigma = self.state.Sigma

        x, y, th = mu[0], mu[1], mu[2]
        k = 3 + 2 * j
        lx, ly = get_landmark_xy(mu, j)


        dx = lx - x
        dy = ly - y
        q = dx*dx + dy*dy
        q = max(q, 1e-12)  # avoid divide-by-zero
        r_hat = np.sqrt(q)
        b_hat = wrap_angle(np.arctan2(dy, dx) - th)

        # innovation (bearing difference must be wrapped)
        nu = np.array([
            float(z.r) - r_hat,
            wrap_angle(float(z.b) - b_hat),
        ], dtype=float)

        # Jacobians wrt robot and landmark
        Hr = np.array([
            [-dx / r_hat, -dy / r_hat, 0.0],
            [ dy / q,     -dx / q,     -1.0],
        ], dtype=float)  # 2x3

        Hl = np.array([
            [ dx / r_hat,  dy / r_hat],
            [-dy / q,      dx / q],
        ], dtype=float)  # 2x2

        # build full H (2 x dim)
        H = np.zeros((2, self.dim), dtype=float)
        H[:, 0:3] = Hr
        H[:, k:k+2] = Hl

        S = H @ Sigma @ H.T + self.R
        # solve instead of explicit inverse for stability
        K = Sigma @ H.T @ np.linalg.inv(S)

        mu = mu + K @ nu
        mu[2] = wrap_angle(mu[2])

        I = np.eye(self.dim, dtype=float)
        # Joseph form (more stable)
        Sigma = (I - K @ H) @ Sigma @ (I - K @ H).T + K @ self.R @ K.T

        self.state.mu = mu
        self.state.Sigma = symmetrize(Sigma)

    def process_observations(self, obs: List[Observation], assoc) -> None:
        """
        Step 2+3:
        - Global one-to-one data association using Mahalanobis distance (computed on a snapshot).
        - Update matched landmarks.
        - Unmatched observations go to a confirmation buffer (pending). Only confirmed ones are initialized.
        """
        if len(obs) == 0:
            return

        # ---------- helpers ----------
        def pending_key(z: Observation):
            rbin = int(np.floor(float(z.r) / self.pending_r_bin))
            b = wrap_angle(float(z.b))
            bbin = int(np.floor(b / self.pending_b_bin))
            return (rbin, bbin)

        # ---------- step counter + cleanup stale pending ----------
        self._step_k += 1
        k_now = self._step_k

        dead = [key for key, item in self.pending.items() if (k_now - item["last_k"]) > self.pending_ttl]
        for key in dead:
            del self.pending[key]

        # ---------- snapshot for ASSOCIATION (important) ----------
        mu0 = self.state.mu.copy()
        Sigma0 = self.state.Sigma.copy()

        inited_idx = np.where(self.state.inited)[0].astype(int)
        M = len(obs)
        J = len(inited_idx)

        # ---------- build all candidate matches (i obs, j landmark) ----------
        candidates = []  # list of (d2, i_obs, j_slot)

        if J > 0:
            for i, z in enumerate(obs):
                z_meas = np.array([float(z.r), float(z.b)], dtype=float)

                for j_slot in inited_idx:
                    # predict measurement and Jacobian wrt full state (on snapshot!)
                    z_hat, H = assoc._predict_measurement_and_H(mu0, int(j_slot), len(mu0))

                    nu = z_meas - z_hat
                    nu[1] = wrap_angle(nu[1])

                    S = H @ Sigma0 @ H.T + assoc.R
                    try:
                        sol = np.linalg.solve(S, nu)  # S^{-1} nu
                    except np.linalg.LinAlgError:
                        continue

                    d2 = float(nu.T @ sol)
                    if d2 < assoc.gate_d2:  # gating
                        candidates.append((d2, i, int(j_slot)))

        # ---------- global one-to-one assignment (greedy by best d2) ----------
        candidates.sort(key=lambda x: x[0])

        obs_to_j = [None] * M
        used_obs = set()
        used_j = set()

        for d2, i, j in candidates:
            if i in used_obs or j in used_j:
                continue
            obs_to_j[i] = j
            used_obs.add(i)
            used_j.add(j)

        # ---------- apply updates first (best matches first) ----------
        for d2, i, j in candidates:
            if obs_to_j[i] == j:
                self.update_landmark(j, obs[i])

        # ---------- handle unmatched: pending-confirm then init ----------
        for i, z in enumerate(obs):
            if obs_to_j[i] is not None:
                continue

            key = pending_key(z)
            if key not in self.pending:
                self.pending[key] = {"count": 1, "last_k": k_now, "z": z}
                continue

            item = self.pending[key]
            # require consecutive steps for confirmation
            if item["last_k"] == k_now - 1:
                item["count"] += 1
            else:
                item["count"] = 1

            item["last_k"] = k_now
            item["z"] = z

            if item["count"] < self.pending_confirm:
                continue

            # confirmed new landmark -> allocate free slot
            free = np.where(~self.state.inited)[0]
            if len(free) == 0:
                # no space -> just keep pending (or drop); here we drop to avoid endless growth
                del self.pending[key]
                continue

            j_new = int(free[0])
            self.init_landmark(j_new, z)
            self.state.inited[j_new] = True

            del self.pending[key]



    def get_est_pose(self) -> Pose2D:
        mu = self.state.mu
        return Pose2D(x=float(mu[0]), y=float(mu[1]), theta=float(mu[2]))

    def get_est_landmarks(self) -> np.ndarray:
        L = np.full((self.N, 2), np.nan, dtype=float)
        for j in range(self.N):
            if self.state.inited[j]:
                xy = get_landmark_xy(self.state.mu, j)  # 你已有的get_landmark_xy
                L[j, :] = xy
        return L

