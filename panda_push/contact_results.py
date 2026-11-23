"""Validate and summarize the matched PandaPush contact-safety experiments."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ARMS = {
    "control": ("Control", "#38587d", (0., 0.)),
    "speed_limit": ("Contact speed limit", "#118576", (0.30, 0.)),
    "force_penalty": ("Contact force penalty", "#b4602b", (0., 0.12)),
}
PROFILES = {"nominal": "Nominal", "size_ood": "Size shift", "friction_ood": "Friction shift", "ood": "Combined shift"}
SEEDS = (11, 29, 47)
METRICS = ("success_pct", "mean_peak_force_n", "p95_peak_force_n", "safety_pct",
           "penetration_pct", "finger_collision_pct", "mean_final_error_cm",
           "contact_pct", "p95_contact_tcp_speed_mps")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def event_metrics(rows):
    reasons = rows["safety_reason"].fillna("")
    peak = rows["peak_force_n"]
    finger_pair = rows["collision_pair"].fillna("")
    return {
        "episodes": len(rows),
        "success_pct": 100 * rows["success"].mean(),
        "mean_peak_force_n": peak.mean(),
        "p95_peak_force_n": peak.quantile(0.95),
        "safety_pct": 100 * reasons.ne("").mean(),
        "penetration_pct": 100 * reasons.str.contains("penetration").mean(),
        "finger_collision_pct": 100 * (reasons.str.contains("robot_collision")
                                      & finger_pair.str.contains("left_finger")
                                      & finger_pair.str.contains("right_finger")).mean(),
        "mean_final_error_cm": 100 * rows["final_error_m"].mean(),
        "contact_pct": 100 * rows["contact_steps"].gt(0).mean(),
        "p95_contact_tcp_speed_mps": rows["peak_contact_tcp_speed_mps"].quantile(0.95),
    }


def read_runs(root):
    shared = None
    scenarios = None
    episodes = []
    times = []
    learning = []
    for arm, (_, _, intervention) in ARMS.items():
        for seed in SEEDS:
            run = root / "training" / arm / f"seed{seed}"
            test = root / "test" / arm / f"seed{seed}"
            config = load_json(run / "config.json")
            training = load_json(run / "training_summary.json")
            protocol = load_json(test / "protocol.json")
            completed = load_json(test / "complete.json")
            controls = {key: value for key, value in config.items()
                        if key not in {"seed", "contact_speed_mps", "force_penalty_weight"}}
            if shared is None:
                shared = controls
            if (config["seed"] != seed or training["start_step"] != 0
                    or training["trained_steps"] != 50000 or training["end_step"] != 50000
                    or controls != shared
                    or (config["contact_speed_mps"], config["force_penalty_weight"]) != intervention
                    or protocol["profiles"] != list(PROFILES) or protocol["first_seed"] != 97000
                    or protocol["episodes_per_profile"] != 100
                    or Path(protocol["checkpoint"]).resolve() != (run / "final.zip").resolve()
                    or completed != {"profiles": 4, "episodes": 400}):
                raise ValueError(f"Non-matching run protocol at {run}")

            rows = pd.read_csv(test / "episodes.csv")
            scenario_columns = ["profile", "seed", "start_x", "start_y", "goal_x", "goal_y",
                                "mass_kg", "table_mu", "cube_mu", "side_m"]
            index = pd.MultiIndex.from_product([list(PROFILES), range(97000, 97100)], names=["profile", "seed"])
            if not rows.set_index(["profile", "seed"]).index.equals(index):
                raise ValueError(f"Non-matching evaluation seed order at {test}")
            paired = rows[scenario_columns]
            if scenarios is None:
                scenarios = paired
            if not scenarios.equals(paired):
                raise ValueError(f"Non-matching sampled physical scenarios at {test}")
            episodes.append(rows.assign(arm=arm, training_seed=seed))
            times.append({"arm": arm, "training_seed": seed, "steps": training["trained_steps"],
                          "wall_s": training["wall_s"]})
            learning.append(pd.read_csv(run / "development.csv").assign(arm=arm, training_seed=seed))
    return pd.concat(episodes, ignore_index=True), pd.DataFrame(times), pd.concat(learning, ignore_index=True)


def draw_test_summary(runs, output):
    metrics = [("success_pct", "Success (%)"), ("mean_peak_force_n", "Mean episode peak (N)"),
               ("safety_pct", "Safety termination (%)")]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.3), layout="constrained")
    positions = np.arange(len(PROFILES))
    for axis, (metric, label) in zip(axes, metrics, strict=True):
        lower, upper = 0., 0.
        for offset, (arm, (legend, color, _)) in zip((-0.19, 0., 0.19), ARMS.items(), strict=True):
            x = positions + offset
            group = runs[(runs["arm"] == arm) & (runs["profile"] != "all")]
            stat = group.groupby("profile")[metric].agg(["mean", "std"]).reindex(PROFILES)
            lower = min(lower, (stat["mean"] - stat["std"]).min())
            upper = max(upper, (stat["mean"] + stat["std"]).max())
            axis.errorbar(x, stat["mean"], yerr=stat["std"], color=color, marker="o",
                          linestyle="none", capsize=3, label=legend, zorder=3)
        axis.set_xticks(positions, PROFILES.values(), rotation=22, ha="right")
        axis.set_ylabel(label)
        if metric == "success_pct":
            upper = max(upper, 100)
        padding = 0.035 * (upper - lower)
        axis.set_ylim(lower - padding, upper + padding)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.25, zorder=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False, fontsize=9)
    fig.suptitle("PandaPush held-out tests | mean and sample SD across three training seeds")
    fig.savefig(output, dpi=155)
    plt.close(fig)


def draw_learning(learning, output):
    specs = [("success_rate", 100, "Development success (%)"),
             ("mean_peak_force_n", 1, "Mean episode peak (N)")]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), layout="constrained")
    for axis, (metric, scale, label) in zip(axes, specs, strict=True):
        for arm, (legend, color, _) in ARMS.items():
            group = learning[learning["arm"] == arm]
            for _, run in group.groupby("training_seed"):
                axis.plot(run["step"], run[metric] * scale, color=color, alpha=0.19)
            center = group.groupby("step")[metric].mean() * scale
            axis.plot(center.index, center, color=color, marker="o", ms=3, label=legend)
        axis.set(xlabel="Training control steps", ylabel=label)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.25)
    axes[0].set_ylim(-4, 104)
    axes[0].legend(loc="lower right", frameon=False, fontsize=8)
    fig.savefig(output, dpi=155)
    plt.close(fig)


def diagnostic_report(root, output):
    base = root / "diagnostics"
    episodes = pd.read_csv(base / "episodes.csv")
    if len(episodes) != 1400 or not episodes.groupby("profile")["seed"].nunique().eq(100).all():
        raise ValueError("Expected 14 diagnostic profiles with 100 matched episodes each")
    summary = []
    for profile, rows in episodes.groupby("profile", sort=False):
        metrics = event_metrics(rows)
        # This archived diagnostic sweep predates the TCP velocity measurement.
        del metrics["p95_contact_tcp_speed_mps"]
        previous = rows[rows["safety_reason"].notna()]["first_success_step"].notna().sum()
        summary.append({"profile": profile, **metrics, "safety_after_reaching": previous})
    data = pd.DataFrame(summary).set_index("profile")
    data.to_csv(output / "diagnostic_factors.csv")
    factors = (
        ("Cube side (mm)", ["nominal", "size_54mm", "size_57mm", "size_60mm"], [48, 54, 57, 60]),
        ("Table friction coefficient", ["nominal", "table_mu_083", "table_mu_092", "table_mu_100"],
         [0.60, 0.83, 0.92, 1.00]),
        ("Cube friction coefficient", ["nominal", "cube_mu_068", "cube_mu_077", "cube_mu_085"],
         [0.45, 0.68, 0.77, 0.85]),
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 5.7), layout="constrained")
    for column, (label, profiles, x) in enumerate(factors):
        rows = data.loc[profiles]
        axes[0, column].plot(x, rows["safety_pct"], "o-", color="#a63f36")
        axes[1, column].plot(x, rows["mean_peak_force_n"], "o-", color="#38587d")
        axes[1, column].set_xlabel(label)
        for axis in axes[:, column]:
            axis.set_xticks(x)
            axis.grid(alpha=0.25)
            axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Safety termination (%)")
    axes[1, 0].set_ylabel("Mean episode peak (N)")
    for axis in axes[0]:
        axis.set_ylim(0, 14)
    for axis in axes[1]:
        axis.set_ylim(10, 29)
    fig.savefig(output / "diagnostic_factors.png", dpi=155)
    plt.close(fig)
    return episodes


def safety_events(episodes, key_columns, output):
    events = episodes.loc[episodes["safety_reason"].notna()].copy()
    detail = ["safety_reason", "collision_pair", "depth_pair"]
    events.groupby([*key_columns, *detail], dropna=False).size().rename("episodes").to_csv(output)


def summarize_study(root, output, diagnosis_only):
    output.mkdir(parents=True, exist_ok=True)
    diagnostic_rows = diagnostic_report(root, output)
    safety_events(diagnostic_rows, ["profile"], output / "diagnostic_safety_pairs.csv")
    if diagnosis_only:
        return
    episodes, times, learning = read_runs(root)
    by_run = []
    for (arm, seed), rows in episodes.groupby(["arm", "training_seed"], sort=False):
        by_run.append({"arm": arm, "training_seed": seed, "profile": "all", **event_metrics(rows)})
        for profile, subset in rows.groupby("profile", sort=False):
            by_run.append({"arm": arm, "training_seed": seed, "profile": profile, **event_metrics(subset)})
    runs = pd.DataFrame(by_run)
    runs.to_csv(output / "metrics_by_run.csv", index=False)
    times.to_csv(output / "training_times.csv", index=False)
    summary = runs.groupby(["arm", "profile"], sort=False)[list(METRICS)].agg(["mean", "std"])
    summary.columns = ["_".join(columns) for columns in summary.columns]
    summary.to_csv(output / "metrics_by_arm.csv")
    control = runs[runs["arm"] == "control"].set_index(["training_seed", "profile"])[list(METRICS)]
    deltas = []
    for arm in ("speed_limit", "force_penalty"):
        treatment = runs[runs["arm"] == arm].set_index(["training_seed", "profile"])[list(METRICS)]
        deltas.append((treatment - control).reset_index().assign(arm=arm))
    pd.concat(deltas, ignore_index=True).to_csv(output / "paired_deltas.csv", index=False)
    safety_events(episodes, ["arm", "profile"], output / "test_safety_pairs.csv")
    draw_test_summary(runs, output / "heldout_comparison.png")
    draw_learning(learning, output / "development_curves.png")
    print(summary.loc[(slice(None), "all"), ["success_pct_mean", "mean_peak_force_n_mean", "safety_pct_mean"]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/contact_safety"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--diagnosis-only", action="store_true")
    args = parser.parse_args()
    summarize_study(args.root, args.output or args.root / "figures", args.diagnosis_only)

