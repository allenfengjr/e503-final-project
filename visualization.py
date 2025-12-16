from __future__ import annotations
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from entities import *


def _set_equal_aspect(ax):
    ax.set_aspect("equal", adjustable="box")


def animate_2d_with_mse(
    frames: List[Frame2D],
    interval_ms: int = 50,
    show_obs_rays: bool = True,
    mse_window: int = 20,
    fixed_margin: float = 2.0,
):
    """
    Left: 2D SLAM visualization
    Right: moving-window MSE vs time step (draws progressively)
    Fixes:
      - no jitter: axis limits fixed once (no autoscale each frame)
      - MSE shown side-by-side and updated per frame
    """

    T = len(frames)
    if T == 0:
        return None

    # --- Precompute histories ---
    true_xy = np.array([[fr.true_pose.x, fr.true_pose.y] for fr in frames], dtype=float)
    est_xy  = np.array([[fr.est_pose.x,  fr.est_pose.y]  for fr in frames], dtype=float)

    L_true = frames[0].true_landmarks

    # per-step squared position error (no smoothing yet)
    err2 = (true_xy[:, 0] - est_xy[:, 0]) ** 2 + (true_xy[:, 1] - est_xy[:, 1]) ** 2

    # moving-window mean (same length as T by using prefix handling)
    w = max(1, int(mse_window))
    mse = np.empty(T, dtype=float)
    csum = np.cumsum(err2)
    for k in range(T):
        a = max(0, k - w + 1)
        total = csum[k] - (csum[a - 1] if a > 0 else 0.0)
        mse[k] = total / (k - a + 1)

    # --- Figure: 2 panels ---
    fig, (ax_map, ax_mse) = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Left: map ----
    _set_equal_aspect(ax_map)
    ax_map.set_title("EKF-SLAM Convergence (2D)")
    ax_map.set_xlabel("x")
    ax_map.set_ylabel("y")

    sc_true_lm = ax_map.scatter(L_true[:, 0], L_true[:, 1], marker="x", label="True landmarks")
    sc_true_now = ax_map.scatter([], [], color="r", s=60, zorder=5, label="True now")
    sc_est_now  = ax_map.scatter([], [], color="r", s=25, alpha=0.6, zorder=5, label="Est now")


    (ln_true_traj,) = ax_map.plot([], [], label="True traj")
    (ln_est_traj,)  = ax_map.plot([], [], label="Est traj")
    sc_est_lm = ax_map.scatter([], [], marker="o", label="Est landmarks")

    obs_lines = []
    ax_map.legend(loc="upper right")

    # FIXED AXIS LIMITS (prevents jitter)
    all_x = np.concatenate([true_xy[:, 0], est_xy[:, 0], L_true[:, 0]])
    all_y = np.concatenate([true_xy[:, 1], est_xy[:, 1], L_true[:, 1]])
    xmin, xmax = float(all_x.min() - fixed_margin), float(all_x.max() + fixed_margin)
    ymin, ymax = float(all_y.min() - fixed_margin), float(all_y.max() + fixed_margin)
    ax_map.set_xlim(xmin, xmax)
    ax_map.set_ylim(ymin, ymax)

    # ---- Right: MSE ----
    ax_mse.set_title(f"Moving-window MSE (window={w})")
    ax_mse.set_xlabel("time step")
    ax_mse.set_ylabel("MSE")

    (ln_mse,) = ax_mse.plot([], [])
    ax_mse.set_xlim(0, T - 1)

    # set y-limits once using full mse range (prevents jitter)
    y_min = float(np.min(mse))
    y_max = float(np.max(mse))
    pad = 0.05 * (y_max - y_min + 1e-12)
    ax_mse.set_ylim(y_min - pad, y_max + pad)

    # optional: show current value
    txt = ax_mse.text(
        0.02, 0.95, "", transform=ax_mse.transAxes, va="top"
    )

    def init():
        ln_true_traj.set_data([], [])
        ln_est_traj.set_data([], [])
        sc_est_lm.set_offsets(np.zeros((0, 2)))
        ln_mse.set_data([], [])
        txt.set_text("")
        sc_true_now.set_offsets(np.zeros((0, 2)))
        sc_est_now.set_offsets(np.zeros((0, 2)))
        return [ln_true_traj, ln_est_traj, sc_est_lm, sc_true_lm, ln_mse, sc_true_now, sc_est_now, txt]

    def update(k: int):
        nonlocal obs_lines

        # trajectories up to k
        ln_true_traj.set_data(true_xy[: k + 1, 0], true_xy[: k + 1, 1])
        ln_est_traj.set_data(est_xy[: k + 1, 0], est_xy[: k + 1, 1])

        fr = frames[k]
        L_est = fr.est_landmarks
        mask = fr.inited & np.isfinite(L_est[:, 0]) & np.isfinite(L_est[:, 1])
        sc_est_lm.set_offsets(L_est[mask, :] if np.any(mask) else np.zeros((0, 2)))
        sc_true_now.set_offsets(np.array([[true_xy[k,0], true_xy[k,1]]]))
        sc_est_now.set_offsets(np.array([[est_xy[k,0],  est_xy[k,1]]]))


        # observation rays (remove old, draw new)
        for l in obs_lines:
            l.remove()
        obs_lines = []

        if show_obs_rays and fr.observations:
            x0, y0, th0 = fr.est_pose.x, fr.est_pose.y, fr.est_pose.theta
            for z in fr.observations:
                x1 = x0 + z.r * np.cos(th0 + z.b)
                y1 = y0 + z.r * np.sin(th0 + z.b)
                (l,) = ax_map.plot([x0, x1], [y0, y1], linewidth=1.0, alpha=0.35)
                obs_lines.append(l)

        # MSE up to k
        ln_mse.set_data(np.arange(k + 1), mse[: k + 1])
        txt.set_text(f"MSE={mse[k]:.4f}")

        artists = [ln_true_traj, ln_est_traj, sc_est_lm, sc_true_lm, ln_mse, txt] + obs_lines
        return artists

    ani = FuncAnimation(
        fig, update, frames=T, init_func=init, interval=interval_ms, blit=False
    )
    plt.tight_layout()
    plt.show()
    return ani
