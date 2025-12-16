# -----------------------------
# Controller (decides u_cmd to follow a known reference path)
# -----------------------------
from entities import *
from utils import *
import numpy as np

from typing import Tuple
import numpy as np

# assumes these exist:
# Pose2D(x,y,theta), Control(v,omega), wrap_angle(a)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

class Controller:
    """
    Very simple path-following controller:
    - Keep v constant
    - Compute a reference point on a circle at time t
    - Turn toward that reference point using omega = k * heading_error
    """

    def __init__(
        self,
        v_const: float,
        k_omega: float,
        omega_max: float,
        # reference circle parameters
        center: Tuple[float, float] = (0.0, 0.0),
        radius: float = 8.0,
        # reference angular speed (rad/s) for moving target point on circle
        ref_ang_speed: float = 0.15,
        # phase offset (rad)
        phase: float = 0.0,
    ):
        self.v_const = float(v_const)
        self.k_omega = float(k_omega)
        self.omega_max = float(omega_max)

        self.cx, self.cy = float(center[0]), float(center[1])
        self.radius = float(radius)
        self.ref_ang_speed = float(ref_ang_speed)
        self.phase = float(phase)

    def reference_point(self, t: float) -> Tuple[float, float]:
        """
        Return (x_ref, y_ref) on a circle.
        This is the "absolute correct trajectory" target point at time t.
        """
        ang = self.ref_ang_speed * float(t) + self.phase
        x_ref = self.cx + self.radius * np.cos(ang)
        y_ref = self.cy + self.radius * np.sin(ang)
        return (float(x_ref), float(y_ref))

    def compute_control(self, est_pose, t: float):
        """
        Keep speed constant, adjust heading only.
        Inputs:
          est_pose: Pose2D, the robot's estimated pose (x,y,theta)
          t: time (seconds)
        Output:
          Control(v, omega)
        """
        x_ref, y_ref = self.reference_point(t)

        dx = x_ref - float(est_pose.x)
        dy = y_ref - float(est_pose.y)

        desired_heading = float(np.arctan2(dy, dx))
        heading_error = wrap_angle(desired_heading - float(est_pose.theta))

        omega = self.k_omega * heading_error
        omega = clamp(omega, -self.omega_max, self.omega_max)

        return Control(v=self.v_const, omega=float(omega))
