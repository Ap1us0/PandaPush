"""Vectorized goal rewards shared by step() and SB3's replay buffers."""

import numpy as np


def goal_distance(achieved, desired):
    return np.linalg.norm(np.asarray(achieved) - np.asarray(desired), axis=-1)


def _info_value(info, key):
    if isinstance(info, dict):
        return np.asarray(info.get(key, 0))
    return np.asarray([entry.get(key, 0) for entry in info])


def goal_reward(achieved, desired, radius, reward_type, info, sparse_safety_cost=False,
                force_penalty_weight=0., force_threshold_n=25., dense_failure_cost=0.):
    distance = goal_distance(achieved, desired)
    if reward_type == "sparse":
        reward = -(distance >= radius).astype(np.float32)
        if sparse_safety_cost and info is not None:
            cost = _info_value(info, "safety_cost")
            reward = np.where(cost > 0, -cost, reward).astype(np.float32)
        return reward

    # Keep the contact point slightly behind the cube along the direction of travel.
    # The absolute cost cannot be farmed by moving repeatedly toward and away from it.
    direction = np.asarray(desired, dtype=float) - np.asarray(achieved, dtype=float)
    direction[..., 2] = 0
    direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-7)
    if isinstance(info, dict):
        tcp = info["tcp"]
        offset = np.asarray(info["push_offset"])
    else:
        tcp = np.asarray([entry["tcp"] for entry in info])
        offset = np.asarray([entry["push_offset"] for entry in info])
    approach = goal_distance(tcp, achieved - offset[..., None] * direction)
    outside = distance >= radius
    reward = -7.0 * distance - 2.5 * np.minimum(approach, 0.18) * outside + 0.35 * ~outside
    if force_penalty_weight:
        peak_force = _info_value(info, "peak_normal_force_n")
        excess = np.clip((peak_force - force_threshold_n) / force_threshold_n, 0., 3.)
        reward -= force_penalty_weight * excess
    if dense_failure_cost:
        reward -= dense_failure_cost * _info_value(info, "safety_cost")
    return np.asarray(reward, dtype=np.float32)

