# main.py
import numpy as np

# --- import your modules/classes ---
from utils import *
from entities import Pose2D, Control, Landmark, Observation, NoiseParams, FOVParams, SimConfig, Frame2D
from simulator import Simulator
from ekf_slam import EKFSLAM
from controller import Controller
from data_association import DataAssociation
from visualization import animate_2d_with_mse

# If you kept everything in one file, just remove these imports and use the definitions directly.


def build_landmarks_circle_in_out(
    n: int,
    center=(0.0, 0.0),
    r_in: float = 6.0,
    r_out: float = 10.0,
):
    """
    Create n landmarks evenly spaced by angle.
    Alternating inside/outside radii to place them on both sides of the robot's circle path.
    """
    cx, cy = center
    lms = []
    for i in range(n):
        ang = 2.0 * np.pi * i / n
        rr = r_in if (i % 2 == 0) else r_out
        x = cx + rr * np.cos(ang)
        y = cy + rr * np.sin(ang)
        lms.append(Landmark(id=i, x=float(x), y=float(y)))
    return lms


def main():
    rng = np.random.default_rng(0)

    # -----------------------------
    # World / trajectory settings
    # -----------------------------
    center = (0.0, 0.0)
    R_path = 8.0                 # robot reference circle radius
    n_landmarks = 20

    # Landmarks: 10 inside (radius 6), 10 outside (radius 10), alternating by angle
    world_landmarks = build_landmarks_circle_in_out(
        n=n_landmarks, center=center, r_in=0.75 * R_path, r_out=1.25 * R_path
    )

    # -----------------------------
    # Noise + FOV + sim config
    # -----------------------------
    dt = 0.1
    noise = NoiseParams(
        sigma_v=0.10,                 # motion: v noise std
        sigma_omega=0.05,             # motion: omega noise std
        sigma_r=0.10,                 # meas: range noise std
        sigma_b=np.deg2rad(5.0),      # meas: bearing noise std
    )
    fov = FOVParams(
        max_range=30.0,               # big enough to see most landmarks
        fov_angle=1 * np.pi,        # 90 degrees
    )
    cfg = SimConfig(
        dt=dt,
        n_landmarks=n_landmarks,
        noise=noise,
        fov=fov,
        world_landmarks=world_landmarks,
    )

    # -----------------------------
    # True initial pose (on the circle at angle 0)
    # -----------------------------
    x0_true = Pose2D(x=center[0] + R_path, y=center[1], theta=np.pi / 2)  # facing +y
    sim = Simulator(cfg=cfg, x0_true=x0_true, rng=rng, include_landmark_id_in_obs=False)

    # -----------------------------
    # EKF-SLAM + association
    # -----------------------------
    slam = EKFSLAM(n_landmarks=n_landmarks, noise=noise)

    # initial estimate: no noise
    mu0 = np.zeros(3 + 2 * n_landmarks, dtype=float)
    mu0[0] = x0_true.x
    mu0[1] = x0_true.y
    mu0[2] = wrap_angle(x0_true.theta)

    Sigma0 = np.eye(3 + 2 * n_landmarks, dtype=float) * 1e-3
    Sigma0[0, 0] = 1e-9
    Sigma0[1, 1] = 1e-9
    Sigma0[2, 2] = 1e-9

    slam.set_initial_estimate(mu0, Sigma0)

    assoc = DataAssociation(R=slam.R, gate_d2=9.21)  # Mahalanobis gating

    # -----------------------------
    # Controller (constant speed, only steer)
    # -----------------------------
    v_const = 1.0
    ref_ang_speed = v_const / R_path  # so reference point moves around circle at ~same speed
    controller = Controller(
        v_const=v_const,
        k_omega=2.5,
        omega_max=1.5,
        center=center,
        radius=R_path,
        ref_ang_speed=ref_ang_speed,
        phase=0.0,
    )

    # -----------------------------
    # Run loop + record frames
    # -----------------------------
    frames = []
    mse = []

    T_steps = 800
    for k in range(T_steps):
        t = k * dt

        est_pose = slam.get_est_pose()
        u_cmd = controller.compute_control(est_pose, t)

        sim.step(u_cmd)
        obs = sim.sense()

        slam.predict(u_cmd, dt)
        slam.process_observations(obs, assoc)

        est_pose = slam.get_est_pose()
        true_pose = sim.get_true_pose()
        true_L = sim.get_true_landmarks()
        est_L = slam.get_est_landmarks()

        frames.append(
            Frame2D(
                t=t,
                true_pose=true_pose,
                est_pose=est_pose,
                true_landmarks=true_L,
                est_landmarks=est_L,
                inited=slam.state.inited.copy(),
                Sigma=slam.state.Sigma.copy(),
                observations=obs,
            )
        )

        dx = true_pose.x - est_pose.x
        dy = true_pose.y - est_pose.y
        mse.append(dx * dx + dy * dy)

    mse = np.array(mse, dtype=float)

    # -----------------------------
    # Visualization
    # -----------------------------
    animate_2d_with_mse(frames, interval_ms=30, show_obs_rays=False)
    # plot_mse(mse, window=50)


if __name__ == "__main__":
    main()
