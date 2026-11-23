"""Geometric handoff pose for the frozen-actor alignment experiment."""

import numpy as np

from .control import PUSHER_RADIUS_M


ALIGNMENT = dict(lateral_tolerance_m=0.005, height_tolerance_m=0.006,
                 ready_gap_m=0.012, aim_gap_m=0.006, rear_plane_margin_m=0.008,
                 rear_plane_tolerance_m=0.003, max_steps=12)


def alignment_target(env, info):
    """Return an out-of-contact waypoint, or None when SAC can take over."""
    tcp, cube, goal = (info[key] for key in ("tcp", "cube", "goal"))
    axis = np.array([*(goal[:2] - cube[:2]), 0.])
    axis /= np.linalg.norm(axis)
    side = np.array([-axis[1], axis[0], 0.])
    rotation = env.data.xmat[env.cube].reshape(3, 3)
    offset = tcp - cube
    local = offset @ rotation
    height = np.clip(cube[2], env.control.low[2], env.control.high[2])
    gap = np.linalg.norm(np.maximum(np.abs(local[:2]) - env.half_side, 0.)) - PUSHER_RADIUS_M
    lateral = abs(offset @ side)

    if offset @ axis < 0 and lateral <= ALIGNMENT["lateral_tolerance_m"]:
        if gap <= ALIGNMENT["ready_gap_m"] and abs(tcp[2] - height) <= ALIGNMENT["height_tolerance_m"]:
            return None
        # Once aligned, advance only by the available clearance and continue
        # to center laterally. The distance to the footprint is 1-Lipschitz.
        approach = max(0., gap - ALIGNMENT["aim_gap_m"])
        target = tcp + approach * axis - (offset @ side) * side
        target[2] = height
        return target

    # Pick the rear-facing box face with strongest alignment to the goal
    # direction. Move to its exterior half-plane before traversing sideways.
    local_axis = axis @ rotation
    index = int(np.abs(local_axis[:2]).argmax())
    sign = np.sign(local_axis[index])
    rear_normal = -sign * rotation[:, index]
    setback = env.half_side + PUSHER_RADIUS_M + ALIGNMENT["rear_plane_margin_m"]
    if -sign * local[index] < setback - ALIGNMENT["rear_plane_tolerance_m"]:
        return tcp + (setback - (-sign * local[index])) * rear_normal

    target = cube - (setback / abs(local_axis[index])) * axis
    target[2] = height
    return target
