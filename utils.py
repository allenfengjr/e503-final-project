'''
Helper functions
'''
from entities import Pose2D
import numpy as np
def wrap_angle(a: float) -> float:
    # map it to (-pi,pi]
    return (a + np.pi) % (2*np.pi) - np.pi


def pose_to_array(p: Pose2D) -> np.ndarray:
    return np.array([p.x, p.y, p.theta], dtype=float)


def array_to_pose(x: np.ndarray) -> Pose2D:
    return Pose2D(float(x[0]), float(x[1]), float(x[2]))


def landmark_slice(j: int) -> slice:
    k = 3 + 2*j
    return slice(k, k+2)

def get_landmark_xy(mu, j):
    s = slice(3+2*j, 3+2*j+2)
    return mu[s].copy()

def set_landmark_xy(mu, j, xy):
    s = slice(3+2*j, 3+2*j+2)
    mu[s] = xy

def symmetrize(S: np.ndarray) -> np.ndarray:
    return 0.5 * (S + S.T)