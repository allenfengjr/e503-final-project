# -----------------------------
# Simulator (God view, generates noisy data)
# -----------------------------

from entities import *
from utils import *
import numpy as np

class Simulator:
    def __init__(self, cfg: SimConfig, x0_true: Pose2D, rng: np.random.Generator, include_landmark_id_in_obs: bool=False):
        self.cfg = cfg
        self.x_true = x0_true
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.include_landmark_id_in_obs = include_landmark_id_in_obs # debug only

    def step(self, u_cmd) -> None:
        """
        Advance TRUE robot state by 1 time step.

        Inputs:
          u_cmd: Control(v, omega) -- the command you send
        Output:
          updates self.x_true in-place

        Noise:
          Adds Gaussian noise to v and omega (execution noise).
        """
        dt = float(self.cfg.dt)
        nv = self.rng.normal(0.0, float(self.cfg.noise.sigma_v))
        nw = self.rng.normal(0.0, float(self.cfg.noise.sigma_omega))

        v = float(u_cmd.v) + nv
        w = float(u_cmd.omega) + nw

        x = float(self.x_true.x)
        y = float(self.x_true.y)
        th = float(self.x_true.theta)

        # Unicycle discrete update (simple Euler)
        x = x + v * dt * np.cos(th)
        y = y + v * dt * np.sin(th)
        th = wrap_angle(th + w * dt)

        self.x_true.x = x
        self.x_true.y = y
        self.x_true.theta = th

    def sense(self) -> List:
        """
        Generate noisy range-bearing observations from TRUE pose.

        Output:
          List[Observation], each has (r, b) for landmarks inside FOV.

        Noise:
          Adds Gaussian noise to range and bearing.
        """
        obs: List = []
        x = float(self.x_true.x)
        y = float(self.x_true.y)
        th = float(self.x_true.theta)

        max_r = float(self.cfg.fov.max_range)
        fov = float(self.cfg.fov.fov_angle)

        for lm in self.cfg.world_landmarks:
            dx = float(lm.x) - x
            dy = float(lm.y) - y

            r_true = np.hypot(dx, dy)
            if r_true > max_r:
                continue

            b_true = wrap_angle(np.arctan2(dy, dx) - th)
            if abs(b_true) > 0.5 * fov:
                continue

            nr = self.rng.normal(0.0, float(self.cfg.noise.sigma_r))
            nb = self.rng.normal(0.0, float(self.cfg.noise.sigma_b))

            r_meas = float(max(1e-3, r_true + nr))
            b_meas = float(wrap_angle(b_true + nb))

            obs.append(
                Observation(
                    r=r_meas,
                    b=b_meas,
                    landmark_id=(lm.id if self.include_landmark_id_in_obs else None), # if enabled, we pass id, else leave for
                )
            )

        return obs

    def get_true_pose(self) -> Pose2D:
        return Pose2D(self.x_true.x, self.x_true.y, self.x_true.theta)


    def get_true_landmarks(self) -> np.ndarray:
        # (N,2)
        L = np.zeros((self.cfg.n_landmarks, 2), dtype=float)
        for lm in self.cfg.world_landmarks:
            L[lm.id, :] = [lm.x, lm.y]
        return L