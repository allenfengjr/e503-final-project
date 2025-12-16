'''
This file includes all data entities
'''
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

@dataclass
class Pose2D:
    '''
    Robot position
    '''
    x: float
    y: float
    theta: float # rad


@dataclass
class Control:
    '''
    Robot motion
    '''
    v: float # m/s
    omega: float # rad/s


@dataclass
class Landmark:
    '''
    Landmark
    '''
    id: int
    x: float
    y: float


@dataclass
class Observation:
    '''
    One measurement
    '''
    r: float # range
    b: float # bearing (rad)
    # Need to accosiate
    landmark_id: Optional[int] = None


@dataclass
class SLAMState:
    '''
    In this project, we apply a simpler setting,
    #landmark is known while id(landmark) remains unknown
    '''
    # mu = [x, y, theta, l1x, l1y, ..., lNx, lNy]
    mu: np.ndarray
    Sigma: np.ndarray
    inited: np.ndarray # (N,) bool


@dataclass
class NoiseParams:
    # hyper-parameters
    # Motion noise
    sigma_v: float
    sigma_omega: float
    # Measurement noise
    sigma_r: float
    sigma_b: float


@dataclass
class FOVParams:
    max_range: float # view distance
    fov_angle: float # view rad


@dataclass
class SimConfig:
    dt: float
    n_landmarks: int
    noise: NoiseParams
    fov: FOVParams
    world_landmarks: List[Landmark] = field(default_factory=list)


@dataclass
class Frame2D:
    # A single visualization snapshot
    t: float
    true_pose: Pose2D
    est_pose: Pose2D
    true_landmarks: np.ndarray      # (N,2)
    est_landmarks: np.ndarray       # (N,2)
    inited: np.ndarray              # (N,)
    Sigma: np.ndarray               # full covariance
    observations: List[Observation] # raw measurements at this step


# -----------------------------
# Data association (rule logic)
# -----------------------------
class DataAssociation:
    def __init__(self, dist_thresh: float):
        self.dist_thresh = dist_thresh

    def associate(self, slam: SLAMState, z: Observation) -> Optional[int]:
        """
        Return landmark index j if matched, else None (new landmark).
        """
        # TODO: simplest NN in Euclidean distance (no EKF math required)
        return None


