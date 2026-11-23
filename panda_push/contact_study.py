"""Run the PandaPush contact diagnostics on matched scenario seeds."""

import argparse
import json
from pathlib import Path

import numpy as np

from .env import PandaPushEnv
from .evaluate import env_options, load_sac, learned_policy, run_episode, summarize, write_json, write_rows

DIAGNOSTIC_PROFILES = (
    "nominal", "size_54mm", "size_57mm", "size_60mm",
    "table_mu_083", "table_mu_092", "table_mu_100",
    "cube_mu_068", "cube_mu_077", "cube_mu_085",
    "high_friction", "mass_170g", "combined_fixed", "ood",
)


def evaluate_contact(env, model, first_seed, episodes):
    rows = []
    policy = learned_policy(model)
    for seed in range(first_seed, first_seed + episodes):
        terminal = {}
        max_substep_tcp_speed = 0.
        speed_limit_steps = 0
        force_pair = ""
        peak = 0.

        def observe(step, info, action):
            nonlocal terminal, max_substep_tcp_speed, speed_limit_steps, force_pair, peak
            if step:
                terminal = info
                if info["peak_normal_force_n"] > peak:
                    peak = info["peak_normal_force_n"]
                    force_pair = info["force_pair"]
                speed_limit_steps += int(info.get("speed_limited", False))
                max_substep_tcp_speed = max(max_substep_tcp_speed, info.get("contact_tcp_speed_mps", 0.))

        row = run_episode(env, seed, policy, on_step=observe)
        row.update({key: terminal[key] for key in (
            "safety_reason", "safety_substep", "depth_pair", "depth_substep",
            "collision_pair", "collision_depth_m", "orientation_rad",
        )})
        row.update(terminal_depth_m=terminal["penetration_m"], terminal_force_n=terminal["peak_normal_force_n"],
                   terminal_force_pair=terminal["force_pair"], terminal_force_substep=terminal["force_substep"],
                   peak_force_pair=force_pair, speed_limit_steps=speed_limit_steps,
                   peak_contact_tcp_speed_mps=max_substep_tcp_speed)
        rows.append(row)
    return rows


def contact_summary(rows):
    result = summarize(rows)
    force = np.array([row["peak_force_n"] for row in rows])
    contacted = force[force > 1e-5]
    failed = [row for row in rows if row["safety_reason"]]
    return {
        **result, "safety_failures": len(failed), "safety_rate": len(failed) / len(rows),
        "penetration_failures": sum("penetration" in row["safety_reason"] for row in failed),
        "p95_peak_force_n": float(np.percentile(force, 95)),
        "mean_peak_force_given_contact_n": float(contacted.mean()) if len(contacted) else None,
        "mean_speed_limited_steps": float(np.mean([row["speed_limit_steps"] for row in rows])),
        "p95_contact_tcp_speed_mps": float(np.percentile([row["peak_contact_tcp_speed_mps"] for row in rows], 95)),
    }


def run_profiles(checkpoint, output, profiles, first_seed, episodes):
    config = json.loads((checkpoint.parent / "config.json").read_text(encoding="utf-8"))
    protocol = dict(checkpoint=str(checkpoint), profiles=list(profiles),
                    first_seed=first_seed, episodes_per_profile=episodes)
    protocol_path = output / "protocol.json"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "complete.json").exists():
        raise FileExistsError(f"Evaluation already completed at {output}")
    if protocol_path.exists():
        archived = json.loads(protocol_path.read_text(encoding="utf-8"))
        if archived != protocol:
            raise ValueError(f"Evaluation protocol differs at {output}")
    elif any(output.iterdir()):
        raise FileExistsError(f"Evaluation directory contains other artifacts: {output}")
    else:
        write_json(protocol_path, protocol)
    model = None
    all_rows = []
    profile_rows = []
    for profile in profiles:
        options = env_options(config)
        options["physics"] = profile
        env = PandaPushEnv(**options, contact_diagnostics=True)
        try:
            if model is None:
                model = load_sac(checkpoint, env)
            rows = evaluate_contact(env, model, first_seed, episodes)
        finally:
            env.close()
        all_rows.extend(dict(profile=profile, **row) for row in rows)
        summary = contact_summary(rows)
        profile_rows.append(dict(profile=profile, **{key: value for key, value in summary.items() if key != "failures"}))
        print(f"{profile:<20} success={summary['success_rate']:.1%}  safety={summary['safety_rate']:.1%}  "
              f"mean peak force={summary['mean_peak_force_n']:.2f} N", flush=True)
        write_rows(output / "episodes.csv", all_rows)
        write_rows(output / "profiles.csv", profile_rows)
    write_json(output / "complete.json", {"profiles": len(profile_rows), "episodes": len(all_rows)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", default=DIAGNOSTIC_PROFILES)
    parser.add_argument("--seed", type=int, default=92000)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0 or len(set(args.profiles)) != len(args.profiles):
        parser.error("Use a positive episode count, a nonnegative seed, and distinct profiles")
    run_profiles(args.checkpoint, args.output, args.profiles, args.seed, args.episodes)


if __name__ == "__main__":
    main()

