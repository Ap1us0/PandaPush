"""Summarize matched 10,000-step experiments with training seeds as replicates."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .evaluate import GROUP_REWARDS, env_options, write_rows


def summarize_study(root, test_root, output, seeds, budget, groups):
    runs = []
    controls = None
    first_test_seed = test_episodes = None
    for seed in seeds:
        for group in groups:
            run = root / group / f"seed{seed}_{budget // 1000}k"
            evaluation = test_root / f"{group}_seed{seed}"
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            train = json.loads((run / "training_summary.json").read_text(encoding="utf-8"))
            test = json.loads((evaluation / "summary.json").read_text(encoding="utf-8"))
            shared = {key: val for key, val in config.items() if key not in {"seed", "group"}}
            if controls is None:
                controls = shared
                first_test_seed, test_episodes = test["first_seed"], test["episodes"]
            if (shared != controls or config["seed"] != seed or config["group"] != group
                    or train["end_step"] != budget or train["start_step"] != 0
                    or test["first_seed"] != first_test_seed or test["episodes"] != test_episodes
                    or test["environment"] != env_options(config)
                    or Path(test["checkpoint"]).parent.resolve() != run.resolve()):
                raise ValueError(f"Run or evaluation does not match the study protocol: {evaluation}")
            failure = test["failures"]
            dev = pd.read_csv(run / "development.csv")
            reaches = dev.loc[dev["success_rate"] >= 0.50, "step"]
            runs.append({
                "group": group, "training_seed": seed, "steps": budget, "wall_seconds": train["wall_s"],
                "train_mean_return": train["mean_training_return"], "test_mean_return": test["mean_return"],
                "test_first_seed": test["first_seed"], "test_episodes": test["episodes"],
                "test_successes": test["successes"], "test_success_pct": 100 * test["success_rate"],
                "test_final_error_cm": 100 * test["mean_final_error_m"],
                "test_contact_episode_pct": 100 * test["episode_contact_rate"],
                "test_safety_failure_pct": 100 * (failure.get("unsafe", 0) + failure.get("off_table", 0)) / test["episodes"],
                "test_cube_displacement_cm": 100 * test["mean_cube_displacement_m"],
                "first_dev_step_at_50_pct": int(reaches.iloc[0]) if len(reaches) else None,
            })
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "runs.csv", runs)
    data = pd.DataFrame(runs)
    aggregate = data.groupby("group", sort=False).agg(
        training_runs=("training_seed", "count"),
        mean_success_pct=("test_success_pct", "mean"),
        std_success_pct=("test_success_pct", "std"),
        mean_final_error_cm=("test_final_error_cm", "mean"),
        mean_test_contact_pct=("test_contact_episode_pct", "mean"),
        mean_test_safety_failure_pct=("test_safety_failure_pct", "mean"),
        mean_cube_displacement_cm=("test_cube_displacement_cm", "mean"),
        mean_wall_seconds=("wall_seconds", "mean"),
        runs_reaching_dev_50_pct=("first_dev_step_at_50_pct", "count"),
    )
    aggregate.to_csv(output / "by_group.csv")
    comparison = data.pivot(index="training_seed", columns="group", values="test_success_pct")
    comparison["her_minus_sparse_pp"] = comparison["sac_her"] - comparison["sac_sparse"]
    comparison.to_csv(output / "paired_her.csv")
    print(aggregate.to_string())
    print(comparison.to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-root", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 47])
    parser.add_argument("--budget", type=int, default=10000)
    parser.add_argument("--groups", nargs="+", choices=GROUP_REWARDS, default=list(GROUP_REWARDS))
    args = parser.parse_args()
    if args.budget <= 0 or args.budget % 1000 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use distinct training seeds and a positive budget in multiples of 1000")
    if not {"sac_sparse", "sac_her"} <= set(args.groups) or len(set(args.groups)) != len(args.groups):
        parser.error("Select both sparse groups and use each group once")
    test_root = args.test_root or args.root / f"test_{args.budget // 1000}k"
    summarize_study(args.root, test_root, args.output, args.seeds, args.budget, args.groups)


if __name__ == "__main__":
    main()

