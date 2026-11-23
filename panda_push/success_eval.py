"""Evaluate archived SAC policies on matched physical scenarios, one model per process.

Model/config paths in the manifest are relative to the project root (or absolute).
The first arm is the paired reference unless the manifest supplies "baseline".
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
from shutil import copyfile

import pandas as pd

from .contact_results import METRICS, event_metrics
from .contact_study import evaluate_contact
from .env import PandaPushEnv
from .evaluate import env_options, load_sac, write_json, write_rows


PROJECT = Path(__file__).resolve().parents[1]
SCENE = ["profile", "seed", "start_x", "start_y", "goal_x", "goal_y",
         "mass_kg", "table_mu", "cube_mu", "side_m"]
SHARED = ("layout", "horizon", "success_radius", "success_hold")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_manifest(path):
    manifest = load_json(path)
    profiles = manifest["profiles"]
    if not profiles or len(set(profiles)) != len(profiles) or "all" in profiles:
        raise ValueError("Expected nonempty, unique profile names")
    if manifest["eval_seed"] < 0 or manifest["episodes"] < 1:
        raise ValueError("Invalid evaluation seed or episode count")
    pairs = [(record["arm"], record["seed"]) for record in manifest["models"]]
    if not pairs or len(pairs) != len(set(pairs)):
        raise ValueError("Expected distinct (arm, training seed) pairs")
    seeds = {seed for _, seed in pairs}
    for arm, seed in pairs:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", arm) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"Invalid model identifier: {arm}, {seed}")
        if {s for a, s in pairs if a == arm} != seeds:
            raise ValueError(f"Unpaired training seeds for {arm}")
    if "baseline" in manifest and manifest["baseline"] not in dict(pairs):
        raise ValueError("Baseline arm is missing")
    return manifest


def protocol_for(manifest, model):
    checkpoint = PROJECT / model["checkpoint"]
    config_path = PROJECT / model["config"]
    config = load_json(config_path)
    if config["seed"] != model["seed"]:
        raise ValueError(f"Training seed disagrees with config at {config_path}")
    return {
        "model": model, "checkpoint_sha256": sha256(checkpoint),
        "config_sha256": sha256(config_path), "evaluation_env": env_options(config),
        "eval_seed": manifest["eval_seed"], "episodes": manifest["episodes"],
        "profiles": manifest["profiles"], "deterministic": True,
    }


def test_dir(path, record):
    return path.parent / "test" / record["arm"] / f"seed{record['seed']}"


def reuse_identical_rollouts(path, manifest, record, protocol, output):
    """Reuse a completed earlier arm only when the model and settings match exactly."""
    keys = ("seed", "checkpoint", "config")
    for other in manifest.get("models", ()):
        if other == record:
            break
        if any(other[key] != record[key] for key in keys):
            continue
        source = test_dir(path, other)
        marker = source / "complete.json"
        if not marker.exists():
            continue
        completed = load_json(marker)
        episodes = source / "episodes.csv"
        expected_count = len(manifest["profiles"]) * manifest["episodes"]
        if (load_json(source / "protocol.json") != dict(protocol, model=other)
                or completed["episodes"] != expected_count
                or completed["episodes_sha256"] != sha256(episodes)):
            raise ValueError(f"Identical-model evaluation archive changed at {source}")
        staged = output / "episodes.tmp"
        copyfile(episodes, staged)
        staged.replace(output / "episodes.csv")
        write_json(output / "complete.json", dict(completed, reused_from_arm=other["arm"]))
        print(f"{record['arm']} seed{record['seed']} reused identical rollouts from {other['arm']}", flush=True)
        return True
    return False


def run_model(path, manifest, record):
    root = test_dir(path, record)
    root.mkdir(parents=True, exist_ok=True)
    protocol = protocol_for(manifest, record)
    protocol_path = root / "protocol.json"
    if protocol_path.exists():
        if load_json(protocol_path) != protocol:
            raise ValueError(f"Model or protocol changed at {root}")
    elif any(root.iterdir()):
        raise FileExistsError(f"Unrecognized evaluation outputs at {root}")
    else:
        write_json(protocol_path, protocol)

    episodes_path = root / "episodes.csv"
    complete = root / "complete.json"
    if complete.exists():
        if load_json(complete)["episodes_sha256"] != sha256(episodes_path):
            raise ValueError(f"Episode archive changed at {root}")
        print(f"Already evaluated: {root}", flush=True)
        return
    if reuse_identical_rollouts(path, manifest, record, protocol, root):
        return

    frames = []
    model = None
    for profile in manifest["profiles"]:
        options = dict(protocol["evaluation_env"], physics=profile, contact_diagnostics=True)
        env = PandaPushEnv(**options)
        try:
            if model is None:
                model = load_sac(PROJECT / record["checkpoint"], env)
            rows = evaluate_contact(env, model, manifest["eval_seed"], manifest["episodes"])
        finally:
            env.close()
        frames.extend(dict(profile=profile, **row) for row in rows)
        print(f"{record['arm']} seed{record['seed']} completed {profile} ({len(rows)} episodes)", flush=True)
    temp = root / "episodes.tmp"
    write_rows(temp, frames)
    temp.replace(episodes_path)
    write_json(complete, {
        "episodes": len(frames), "checkpoint_steps": model.num_timesteps,
        "episodes_sha256": sha256(episodes_path),
    })


def read_matched(path, manifest):
    frames, provenance = [], []
    scenarios = settings = None
    seed_range = range(manifest["eval_seed"], manifest["eval_seed"] + manifest["episodes"])
    expected = pd.MultiIndex.from_product([manifest["profiles"], seed_range], names=["profile", "seed"])
    for record in manifest["models"]:
        root = test_dir(path, record)
        protocol = protocol_for(manifest, record)
        if load_json(root / "protocol.json") != protocol:
            raise ValueError(f"Model or protocol changed at {root}")
        complete = load_json(root / "complete.json")
        output = root / "episodes.csv"
        if complete["episodes_sha256"] != sha256(output) or complete["episodes"] != len(expected):
            raise ValueError(f"Incomplete or altered evaluation at {root}")
        rows = pd.read_csv(output)
        scene = rows[SCENE]
        shared = {key: protocol["evaluation_env"][key] for key in SHARED}
        if not rows.set_index(["profile", "seed"]).index.equals(expected):
            raise ValueError(f"Unexpected evaluation seeds or profiles at {root}")
        if scenarios is None:
            scenarios, settings = scene, shared
        if shared != settings or not scenarios.equals(scene):
            raise ValueError(f"Unmatched physical scenario or success criteria at {root}")
        frames.append(rows.assign(arm=record["arm"], training_seed=record["seed"]))
        provenance.append(dict(**record, checkpoint_steps=complete["checkpoint_steps"],
                               checkpoint_sha256=protocol["checkpoint_sha256"],
                               config_sha256=protocol["config_sha256"],
                               episodes_sha256=complete["episodes_sha256"],
                               reused_from_arm=complete.get("reused_from_arm")))
    return pd.concat(frames, ignore_index=True), provenance


def draw_chart(runs, manifest, output, precision=False):
    import matplotlib.pyplot as plt

    metrics = (("success_pct", "Success (%)"), ("mean_peak_force_n", "Mean episode peak force (N)"),
               ("safety_pct", "Safety termination (%)"))
    if precision:
        metrics = (("mean_final_error_cm", "Mean terminal error (cm)"),
                   ("within_1cm_pct", "Within 1 cm (%)"), ("within_2cm_pct", "Within 2 cm (%)"))
    arms = list(runs["arm"].unique())
    names = {"all": "All", "nominal": "Nominal", "size_ood": "Cube size",
             "friction_ood": "Friction", "ood": "Combined"}
    profiles = ["all", *manifest["profiles"]]
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4.3), layout="constrained")
    for axis, (metric, label) in zip(axes, metrics, strict=True):
        for i, arm in enumerate(arms):
            arm_runs = runs.loc[runs.arm == arm]
            values = arm_runs.pivot(index="profile", columns="training_seed", values=metric).loc[profiles]
            xs = [j + 0.5 * ((i + 0.5) / len(arms) - 0.5) for j in range(len(profiles))]
            color = colors[i % len(colors)]
            for seed in values.columns:
                axis.scatter(xs, values[seed], alpha=0.35, color=color, s=13)
            axis.scatter(xs, values.mean(axis=1), marker="D", s=27, color=color, label=arm.replace("_", " "))
        axis.set_xticks(range(len(profiles)), [names.get(p, p) for p in profiles], rotation=25, ha="right")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.22)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    if precision:
        for axis in axes[1:]:
            axis.set_ylim(-4, 104)
    else:
        axes[0].set_ylim(-4, 104)
        if "success_target_pct" in manifest:
            target = manifest["success_target_pct"]
            axes[0].axhline(target, color="#6b7481", linestyle=":", linewidth=1, zorder=1)
    fig.suptitle("Held-out PandaPush scenarios | dots: training seeds, diamonds: seed means")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=len(arms), frameon=False)
    fig.savefig(output, dpi=160)
    plt.close(fig)


PRECISION_METRICS = ("within_1cm_pct", "within_2cm_pct", "p95_final_error_cm")


def evaluation_metrics(rows, precision=False):
    metrics = dict(successes=int(rows["success"].sum()),
                   safety_stops=int(rows["safety_reason"].notna().sum()),
                   max_peak_force_n=rows["peak_force_n"].max(),
                   **event_metrics(rows))
    if precision:
        error = rows["final_error_m"]
        safe = rows["safety_reason"].isna()
        metrics.update(within_1cm_pct=100 * (safe & error.lt(0.01)).mean(),
                       within_2cm_pct=100 * (safe & error.lt(0.02)).mean(),
                       p95_final_error_cm=100 * error.quantile(0.95))
    return metrics


def aggregate(path, manifest):
    episodes, provenance = read_matched(path, manifest)
    output = path.parent / "summary"
    output.mkdir(exist_ok=True)
    records = []
    precision = manifest.get("precision_metrics", False)
    metric_names = (*METRICS, *PRECISION_METRICS) if precision else METRICS
    for (arm, seed), rows in episodes.groupby(["arm", "training_seed"], sort=False):
        for profile, group in [("all", rows), *rows.groupby("profile", sort=False)]:
            records.append(dict(arm=arm, training_seed=seed, profile=profile,
                                **evaluation_metrics(group, precision)))
    runs = pd.DataFrame(records)
    runs.to_csv(output / "metrics_by_run.csv", index=False)
    summary = runs.groupby(["arm", "profile"], sort=False)[list(metric_names)].agg(["mean", "std"])
    summary.columns = ["_".join(cols) for cols in summary.columns]
    by_arm = runs.groupby(["arm", "profile"], sort=False)
    counts = by_arm[["episodes", "successes", "safety_stops"]].sum()
    summary = counts.join(summary).join(by_arm["max_peak_force_n"].max())
    summary.to_csv(output / "metrics_by_arm.csv")
    baseline = manifest.get("baseline", manifest["models"][0]["arm"])
    indexed = runs.set_index(["arm", "training_seed", "profile"])[list(metric_names)]
    pairs = [(indexed.loc[arm] - indexed.loc[baseline]).reset_index().assign(arm=arm, baseline=baseline)
             for arm in runs.arm.unique() if arm != baseline]
    if pairs:
        pd.concat(pairs, ignore_index=True).to_csv(output / "paired_deltas.csv", index=False)
    failures = episodes.loc[episodes["failure"].notna()]
    failures.groupby(["arm", "training_seed", "profile", "failure"]).size().rename("count").to_csv(output / "failures.csv")
    pd.DataFrame(provenance).to_csv(output / "provenance.csv", index=False)
    draw_chart(runs, manifest, output / "heldout_comparison.png")
    if precision:
        draw_chart(runs, manifest, output / "precision_comparison.png", precision=True)
    print(summary.xs("all", level="profile")[["successes", "episodes", "safety_stops",
                                              "success_pct_mean", "mean_peak_force_n_mean", "max_peak_force_n"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--arm", required=True)
    run.add_argument("--seed", required=True, type=int)
    report = commands.add_parser("aggregate")
    report.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    manifest = read_manifest(args.manifest)
    if args.command == "run":
        record = next((record for record in manifest["models"]
                       if (record["arm"], record["seed"]) == (args.arm, args.seed)), None)
        if record is None:
            parser.error("Model does not appear in the manifest")
        run_model(args.manifest, manifest, record)
    else:
        aggregate(args.manifest, manifest)


if __name__ == "__main__":
    main()

