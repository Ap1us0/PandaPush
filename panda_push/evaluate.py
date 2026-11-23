"""Matched-seed evaluation of SAC, uniform actions, or a geometric baseline."""

import argparse
import csv
import json
from pathlib import Path
from time import sleep

import numpy as np

from .env import PHYSICS, PandaPushEnv
from .rewards import goal_distance

ENV_KEYS = ("layout", "physics", "horizon", "success_radius", "success_hold")
GROUP_REWARDS = {"sac_dense": "dense", "sac_sparse": "sparse", "sac_her": "sparse"}


def env_options(config):
    options = {"reward_type": GROUP_REWARDS[config["group"]], **{key: config[key] for key in ENV_KEYS}}
    optional = ("sparse_safety_cost", "contact_speed_mps", "free_speed_mps", "settle_speed_mps",
                "release_hold_steps", "release_clearance_m",
                "center_speed_mps", "center_radius_m", "center_steps", "center_min_alignment",
                "center_goal_projection", "approach_projection_radius_m",
                "force_penalty_weight",
                "force_threshold_n", "dense_failure_cost")
    options.update({key: config[key] for key in optional if key in config})
    return options


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class GeometricPolicy:
    """Separate controller for feasibility tests, never used during SAC learning."""

    def __init__(self, radius):
        self.radius = radius
        self.stage = "align"

    def __call__(self, observation, info, rng):
        tcp, cube, goal = (info[key] for key in ("tcp", "cube", "goal"))
        difference = goal - cube
        if np.linalg.norm(difference) < self.radius * 0.47:
            return np.zeros(3)
        direction = difference[:2] / max(1e-7, np.linalg.norm(difference[:2]))
        setback = info["push_offset"]
        behind = cube[:2] - direction * (setback + 0.018)
        if self.stage == "push":
            offset = cube[:2] - tcp[:2]
            along = np.dot(offset, direction)
            if np.linalg.norm(offset - along * direction) > 0.020 or along < 0.016:
                self.stage = "align"
        if self.stage == "align" and np.linalg.norm(tcp[:2] - behind) < 0.012:
            self.stage = "lower"
        z = max(goal[2] + 0.003, 0.260)
        if self.stage == "lower" and abs(tcp[2] - z) < 0.005:
            self.stage = "push"
        if self.stage == "align":
            xy = behind
            z = 0.315
        elif self.stage == "lower":
            xy = behind
        else:
            xy = cube[:2] - direction * (setback - 0.025)
        return np.clip((np.array([*xy, z]) - tcp) / 0.025, -1, 1)


def learned_policy(model):
    return lambda observation, info, rng: model.predict(observation, deterministic=True)[0]


def policy_for(name, env, model=None):
    if name == "random":
        return lambda observation, info, rng: rng.uniform(-1, 1, 3)
    if name == "rule":
        return GeometricPolicy(env.success_radius)
    if model is None:
        raise ValueError("SAC evaluation requires a checkpoint")
    return learned_policy(model)


def run_episode(env, seed, policy, on_step=None):
    rng = np.random.default_rng(seed + 1000003)
    observation, info = env.reset(seed=seed)
    initial_distance = min_distance = info["distance_m"]
    start = info["cube"]
    reward_total = cube_path = tcp_path = peak = max_depth = action_sq = 0.
    action_sum = np.zeros(3)
    contact_steps = ineffective = rejected = 0
    first_success = None
    contact_after_success = False
    if on_step:
        on_step(0, info, np.zeros(3))
    for step in range(1, env.horizon + 1):
        previous_tcp = info["tcp"]
        action = policy(observation, info, rng)
        observation, reward, terminated, truncated, info = env.step(action)
        action_sum += action
        action_sq += float(np.dot(action, action))
        reward_total += reward
        min_distance = min(min_distance, info["distance_m"])
        cube_path += info["cube_delta_m"]
        tcp_path += goal_distance(info["tcp"], previous_tcp)
        peak = max(peak, info["peak_normal_force_n"])
        max_depth = max(max_depth, info["penetration_m"])
        contact_steps += int(info["cube_contact"])
        ineffective += int(info["cube_contact"] and info["cube_delta_m"] < 0.001)
        rejected += int(info["ik_rejected"])
        contact_after_success |= first_success is not None and info["cube_contact"]
        if info["is_success"] and first_success is None:
            first_success = step
        if on_step:
            on_step(step, info, action)
        if terminated or truncated:
            break
    if info["is_success"]:
        failure = ""
    elif info["off_table"]:
        failure = "off_table"
    elif info["unsafe"]:
        failure = "unsafe"
    elif first_success is not None:
        failure = "left_goal"
    elif contact_steps == 0:
        failure = "no_contact"
    elif cube_path < 0.015:
        failure = "little_motion"
    else:
        failure = "missed_goal"
    return {
        "seed": seed, "steps": step, "success": info["is_success"], "first_success_step": first_success,
        "return": reward_total, "initial_error_m": initial_distance,
        "final_error_m": info["distance_m"], "minimum_error_m": min_distance,
        "contact_steps": contact_steps, "ineffective_contact_steps": ineffective,
        "peak_force_n": peak, "max_penetration_m": max_depth, "ik_rejected_steps": rejected,
        "joint_limit": info["joint_limit"], "failure": failure,
        "cube_displacement_m": info["cube_displacement_m"], "cube_path_m": cube_path, "tcp_path_m": tcp_path,
        "start_x": start[0], "start_y": start[1], "goal_x": float(info["goal"][0]), "goal_y": float(info["goal"][1]),
        "end_x": info["cube"][0], "end_y": info["cube"][1],
        "mass_kg": info["mass_kg"], "table_mu": info["table_mu"], "cube_mu": info["cube_mu"],
        "side_m": info["side_m"], "action_dx": action_sum[0] / step, "action_dy": action_sum[1] / step,
        "action_dz": action_sum[2] / step, "action_rms": float(np.sqrt(action_sq / (3 * step))),
        "contact_after_success": contact_after_success,
    }


def summarize(rows):
    def mean(key):
        return float(np.mean([row[key] for row in rows]))

    contacts = sum(row["contact_steps"] for row in rows)
    first = [row["first_success_step"] for row in rows if row["success"]]
    failures = sorted({row["failure"] for row in rows} - {""})
    return {
        "episodes": len(rows), "successes": sum(row["success"] for row in rows),
        "success_rate": mean("success"), "mean_return": mean("return"),
        "mean_final_error_m": mean("final_error_m"), "mean_minimum_error_m": mean("minimum_error_m"),
        "episode_contact_rate": float(np.mean([row["contact_steps"] > 0 for row in rows])),
        "control_contact_rate": contacts / sum(row["steps"] for row in rows),
        "ineffective_contact_fraction": sum(row["ineffective_contact_steps"] for row in rows) / max(contacts, 1),
        "mean_success_steps": float(np.mean(first)) if first else None,
        "mean_cube_displacement_m": mean("cube_displacement_m"),
        "mean_cube_path_m": mean("cube_path_m"), "mean_tcp_path_m": mean("tcp_path_m"),
        "mean_peak_force_n": mean("peak_force_n"), "max_penetration_m": max(row["max_penetration_m"] for row in rows),
        "joint_limit_episodes": sum(row["joint_limit"] for row in rows),
        "mean_action_dx": mean("action_dx"), "mean_action_dy": mean("action_dy"),
        "mean_action_dz": mean("action_dz"), "mean_action_rms": mean("action_rms"),
        "failures": {name: sum(row["failure"] == name for row in rows) for name in failures},
    }


def evaluate(env, seeds, name, model=None):
    return [run_episode(env, seed, policy_for(name, env, model)) for seed in seeds]


def load_sac(checkpoint, env):
    import torch
    from stable_baselines3 import SAC

    torch.set_num_threads(1)
    return SAC.load(checkpoint, env=env, device="cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("sac", "rule", "random"), default="sac")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--group", choices=GROUP_REWARDS, default="sac_dense")
    parser.add_argument("--layout", choices=("fixed", "random"))
    parser.add_argument("--physics", choices=PHYSICS)
    parser.add_argument("--seed", type=int, default=61000)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0 or (args.policy == "sac" and args.checkpoint is None):
        parser.error("Evaluation requires positive episode count, nonnegative seed and a checkpoint for SAC")
    config_path = args.config
    if config_path is None:
        config_path = args.checkpoint.parent / "config.json" if args.checkpoint else Path("configs/default.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.setdefault("group", args.group)
    if args.layout:
        config["layout"] = args.layout
    if args.physics:
        config["physics"] = args.physics
    options = env_options(config)
    env = PandaPushEnv(**options, render_mode="human" if args.viewer else None)
    try:
        model = load_sac(args.checkpoint, env) if args.policy == "sac" else None
        if args.viewer:
            rows = [run_episode(env, seed, policy_for(args.policy, env, model),
                                on_step=lambda *_: sleep(1 / env.metadata["render_fps"]))
                    for seed in range(args.seed, args.seed + args.episodes)]
        else:
            rows = evaluate(env, range(args.seed, args.seed + args.episodes), args.policy, model)
        metrics = {
            "policy": args.policy, "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "environment": options, "first_seed": args.seed, **summarize(rows),
        }
        print(json.dumps(metrics, indent=2, allow_nan=False))
        if args.output:
            args.output.mkdir(parents=True, exist_ok=False)
            write_rows(args.output / "episodes.csv", rows)
            write_json(args.output / "summary.json", metrics)
        if args.viewer:
            sleep(1.5)
    finally:
        env.close()


if __name__ == "__main__":
    main()

