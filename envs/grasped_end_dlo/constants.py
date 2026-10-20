"""Constants shared by grasped-end DLO environments."""

from __future__ import annotations

import numpy as np

MING_LI_SCALE = 30

CABLE_LENGTH = 0.4
CABLE_THICKNESS = 0.005
CABLE_SHEAR_MODULUS = 6e7 / MING_LI_SCALE
CABLE_YOUNGS_MODULUS = 5e7 / MING_LI_SCALE
CABLE_DR_LENGTH = (0.36, 0.44)
CABLE_DR_THICKNESS = (0.0018, 0.0022)
CABLE_DR_SHEAR_MODULUS = (5e7 / MING_LI_SCALE, 7e7 / MING_LI_SCALE)
CABLE_DR_YOUNGS_MODULUS = (4e7 / MING_LI_SCALE, 6e7 / MING_LI_SCALE)

PIPE_INNER_RADIUS = 0.01
PIPE_DR_INNER_RADIUS = (0.01, 0.02)
PIPE_OUTER_RADIUS = 0.0435 / 2
PIPE_LENGTH = 0.125 / 2
PIPE_RESOLUTION = 20

RESET_POS_MIN = (-0.05, -0.05, -0.05)
RESET_POS_MAX = (0.05, 0.05, 0.05)
RESET_ROT_MIN = (-np.deg2rad(15), -np.deg2rad(15), -np.deg2rad(15))
RESET_ROT_MAX = (np.deg2rad(15), np.deg2rad(15), np.deg2rad(15))

OBSERVATION_LAYOUT = (
    "keypoint_to_target_x_keypoint_frame",
    "keypoint_to_target_y_keypoint_frame",
    "keypoint_to_target_z_keypoint_frame",
    "target_z_axis_x_keypoint_frame",
    "target_z_axis_y_keypoint_frame",
    "target_z_axis_z_keypoint_frame",
)
