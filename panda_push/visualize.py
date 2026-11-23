"""Export plots and verified replays from PandaPush experiment artifacts."""

import argparse
import csv
import json
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from PIL import Image, ImageDraw, ImageFont

from .env import PandaPushEnv
from .evaluate import load_sac, policy_for, run_episode, write_json, write_rows

GROUP_LABELS = {"sac_dense": "Dense SAC", "sac_sparse": "Sparse SAC", "sac_her": "Sparse SAC + HER"}
GROUP_COLORS = {"sac_dense": "#226596", "sac_sparse": "#ba662a", "sac_her": "#158166"}


def experiment_label(config):
    label = GROUP_LABELS[config["group"]]
    if config["group"] != "sac_dense" and config.get("sparse_safety_cost"):
        label += " (safety cost)"
    return label


def tidy(axis, labels, grid=True):
    axis.set(**labels)
    axis.spines[["right", "top"]].set_visible(False)
    if grid:
        axis.grid(alpha=0.23, linewidth=0.65)
        axis.set_axisbelow(True)


def training(runs, output):
    output.mkdir(parents=True, exist_ok=True)
    comparison = []
    for run in runs:
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        log = pd.read_csv(run / "train.monitor.csv", skiprows=1)
        development = pd.read_csv(run / "development.csv")
        development["group"] = config["group"]
        development["seed"] = config["seed"]
        development["experiment"] = experiment_label(config)
        development["run"] = str(run)
        comparison.append(development)
        fig, axes = plt.subplots(3, 1, figsize=(8.6, 7.2), sharex=True, layout="constrained")
        episode_steps = development["step"].min() + log["l"].cumsum()
        axes[0].plot(episode_steps, log["r"], color="#afc8d9", linewidth=0.7, alpha=0.65)
        axes[0].plot(episode_steps, log["r"].rolling(25, min_periods=1).mean(),
                     label="Train, rolling 25 episodes", color="#226596")
        axes[0].plot(development["step"], development["mean_return"], "o--",
                     ms=3, label="Development, deterministic", color="#ba662a")
        axes[0].legend(loc="best", frameon=False)
        axes[1].plot(development["step"], 100 * development["success_rate"], "o-", color="#158166", ms=3)
        axes[2].plot(development["step"], 100 * development["mean_final_error_m"], "o-", color="#226596", ms=3)
        axes[2].axhline(3, color="#707887", linestyle=":", linewidth=1)
        for axis, title in zip(axes, ("Episode return", "Success (%)", "Final cube error (cm)"), strict=True):
            tidy(axis, {"ylabel": title})
        axes[1].set_ylim(-4, 104)
        axes[2].set_xlabel("Environment control steps")
        fig.suptitle(f"{experiment_label(config)}  |  seed {config['seed']}  |  {config['layout']} layout")
        tag = "_safety_cost" if config["group"] != "sac_dense" and config.get("sparse_safety_cost") else ""
        fig.savefig(output / f"{config['group']}{tag}_{run.name}_training.png", dpi=155)
        plt.close(fig)
    data = pd.concat(comparison, ignore_index=True)
    # Resumed segments share a seed. Later-loaded segments supply the boundary checkpoint.
    data = data.drop_duplicates(subset=["experiment", "seed", "step"], keep="last")
    data.to_csv(output / "development_all_runs.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.2), sharex=True, layout="constrained")
    for label, subset in data.groupby("experiment", sort=False):
        color = GROUP_COLORS[subset["group"].iloc[0]]
        line_style = "--" if "safety cost" in label else "-"
        for axis, key in zip(axes, ("success_rate", "mean_final_error_m"), strict=True):
            by_step = subset.groupby("step")[key]
            for _, series in subset.groupby("seed"):
                series = series.sort_values("step")
                axis.plot(series["step"], 100 * series[key], color=color, alpha=0.19, linewidth=1)
            center = by_step.mean()
            n = subset["seed"].nunique()
            axis.plot(center.index, 100 * center, color=color, marker="o", ms=2.5,
                      linestyle=line_style, label=f"{label} (n={n} runs)")
    tidy(axes[0], {"ylabel": "Development success (%)", "ylim": (-4, 104)})
    tidy(axes[1], {"ylabel": "Mean final cube error (cm)", "xlabel": "Environment control steps"})
    axes[0].legend(frameon=False, loc="best")
    fig.savefig(output / "sac_comparison.png", dpi=168)
    plt.close(fig)


def robustness(inputs, output):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in inputs:
        metrics = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        policy = metrics["checkpoint"] or metrics["policy"]
        config_path = Path(policy).parent / "config.json"
        if metrics["policy"] == "sac":
            config = json.loads(config_path.read_text(encoding="utf-8"))
            trained = json.loads((config_path.parent / "training_summary.json").read_text(encoding="utf-8"))
            label = (f"{experiment_label(config)} / {trained['end_step'] // 1000}k "
                     f"/ {config['physics']} train / seed {config['seed']}")
        else:
            label = policy
        rows.append({
            "policy": label, "profile": metrics["environment"]["physics"],
            "success_pct": 100 * metrics["success_rate"], "final_error_cm": 100 * metrics["mean_final_error_m"],
            "ineffective_contact_pct": 100 * metrics["ineffective_contact_fraction"],
            "cube_path_cm": 100 * metrics["mean_cube_path_m"], "tcp_path_cm": 100 * metrics["mean_tcp_path_m"],
            "mean_success_step": metrics["mean_success_steps"], "episodes": metrics["episodes"],
            "summary": str(path / "summary.json"),
        })
    data = pd.DataFrame(rows)
    data.to_csv(output / "robustness.csv", index=False)
    labels = {"nominal": "Nominal", "idr": "Moderate variation", "ood": "Heavy + high friction + larger",
              "heavy": "Heavy only", "slick": "Low friction"}
    profiles = [name for name in labels if name in data["profile"].values]
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.3), sharex=True, layout="constrained")
    for label, series in data.groupby("policy", sort=False):
        series = series.set_index("profile").reindex(profiles)
        axes[0].plot(profiles, series["success_pct"], "o-", label=label)
        axes[1].plot(profiles, series["final_error_cm"], "o-")
    axes[1].set_xticks(range(len(profiles)), [labels[name] for name in profiles])
    tidy(axes[0], {"ylabel": "Success (%)", "ylim": (-4, 104)})
    tidy(axes[1], {"ylabel": "Mean final cube error (cm)", "xlabel": "Physics profile"})
    axes[0].legend(frameon=False, loc="best")
    fig.savefig(output / "physics_generalization.png", dpi=168)
    plt.close(fig)


def trace_plot(trace, radius, output):
    cube = np.array([[row["cube_x"], row["cube_y"]] for row in trace])
    tcp = np.array([[row["tcp_x"], row["tcp_y"]] for row in trace])
    goal = [trace[0]["goal_x"], trace[0]["goal_y"]]
    contact = np.array([row["contact"] for row in trace])
    steps = [row["step"] for row in trace]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), layout="constrained")
    axes[0].add_patch(Circle(goal, radius, color="#1b8d61", alpha=0.19, label="Goal region"))
    axes[0].plot(*tcp.T, color="#226596", linestyle="--", label="Pusher center")
    axes[0].plot(*cube.T, color="#c14a24", label="Cube center")
    axes[0].scatter(*cube[0], marker="s", color="#c14a24", label="Cube start", s=25)
    axes[0].scatter(*cube[-1], marker="X", color="#c14a24", label="Cube end", s=40)
    tidy(axes[0], {"xlabel": "World X (m)", "ylabel": "World Y (m)", "aspect": "equal"})
    axes[0].legend(frameon=False, fontsize=8)
    error = 100 * np.array([row["error_m"] for row in trace])
    axes[1].plot(steps, error, color="#c14a24", label="Cube error")
    axes[1].scatter(np.asarray(steps)[contact], error[contact], color="#226596", s=13, label="Cube contact")
    axes[1].axhline(radius * 100, color="#1b8d61", linestyle=":", label="Threshold")
    tidy(axes[1], {"xlabel": "Control step", "ylabel": "Cube error (cm)"})
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(output, dpi=166)
    plt.close(fig)


def record_case(env, checkpoint, model, policy, row, name, output):
    seed = int(row["seed"])
    path = output / f"{name}.mp4"
    fps = env.metadata["render_fps"]
    writer = imageio_ffmpeg.write_frames(
        str(path.resolve()), (960, 540), fps=fps, codec="libx264", pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p", quality=8, macro_block_size=2, output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    trace = []
    font = ImageFont.load_default(size=21)
    status_font = ImageFont.load_default(size=16)
    frame = None

    def draw(step, info, action):
        nonlocal frame
        image = Image.fromarray(env.render())
        hud = ImageDraw.Draw(image)
        hud.rounded_rectangle((22, 448, 938, 525), radius=9, fill="#152231")
        hud.text((40, 457), f"{name.replace('_', ' ')}   |   scenario {seed}   |   step {step:02d}/{env.horizon}",
                 fill="white", font=font)
        text = f"Cube error {100 * info['distance_m']:.2f} cm   |   hold {info['hold_steps']}/{env.success_hold}"
        hud.text((40, 493), text, fill="white", font=status_font)
        status = f"Contact {info['peak_normal_force_n']:.1f} N" if info["cube_contact"] else "No robot-cube contact"
        hud.text((440, 493), status, fill="#ffb082", font=status_font)
        hud.text((765, 493), "Goal region", fill="#66e8b2", font=status_font)
        frame = image.tobytes()
        for _ in range(fps if step == 0 else 1):
            writer.send(frame)
        trace.append({
            "step": step, "time_s": step / fps, "error_m": info["distance_m"],
            "hold_steps": info["hold_steps"], "contact": info["cube_contact"],
            "normal_force_n": info["peak_normal_force_n"],
            "cube_x": info["cube"][0], "cube_y": info["cube"][1], "cube_z": info["cube"][2],
            "tcp_x": info["tcp"][0], "tcp_y": info["tcp"][1], "tcp_z": info["tcp"][2],
            "goal_x": info["goal"][0], "goal_y": info["goal"][1], "goal_z": info["goal"][2],
            "action_dx": action[0], "action_dy": action[1], "action_dz": action[2],
        })

    try:
        result = run_episode(env, seed, policy_for(policy, env, model), on_step=draw)
        if (result["success"] != (row["success"] == "True") or result["steps"] != int(row["steps"])
                or abs(result["final_error_m"] - float(row["final_error_m"])) > 1e-6):
            raise RuntimeError("Replay diverged from the saved evaluation")
        for _ in range(2 * fps):
            writer.send(frame)
    finally:
        writer.close()
    count, duration = imageio_ffmpeg.count_frames_and_secs(str(path.resolve()))
    if count != 3 * fps + result["steps"]:
        raise RuntimeError(f"Unexpected video frame count {count}")
    write_rows(output / f"{name}_trace.csv", trace)
    trace_plot(trace, env.success_radius, output / f"{name}_trajectory.png")
    return {"video": path.name, "checkpoint": checkpoint, "frames": count, "duration_s": duration, **result}


def cases(evaluation, output):
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((evaluation / "summary.json").read_text(encoding="utf-8"))
    with (evaluation / "episodes.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    checkpoint, policy = metadata["checkpoint"], metadata["policy"]
    env = PandaPushEnv(**metadata["environment"], render_mode="rgb_array")
    manifest = []
    try:
        model = load_sac(checkpoint, env) if checkpoint else None
        for success in (True, False):
            subset = [row for row in rows if (row["success"] == "True") == success]
            if not subset:
                continue
            if not success:
                # Prefer a failure showing real contact, and a frequent failure mode.
                moved = [row for row in subset if int(row["contact_steps"]) > 0]
                eligible = moved or subset
                common = max(sorted({row["failure"] for row in eligible}),
                             key=lambda category: sum(row["failure"] == category for row in eligible))
                subset = [row for row in eligible if row["failure"] == common]
            subset.sort(key=lambda row: float(row["final_error_m"]))
            row = subset[len(subset) // 2]
            name = f"{policy.upper()}_{'success' if success else 'failure'}"
            manifest.append(record_case(env, checkpoint, model, policy, row, name, output))
    finally:
        env.close()
    write_json(output / "cases.json", manifest)
    print(json.dumps(manifest, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("training")
    train_parser.add_argument("--runs", nargs="+", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    generalization = commands.add_parser("robustness")
    generalization.add_argument("--evaluations", nargs="+", type=Path, required=True)
    generalization.add_argument("--output", type=Path, required=True)
    videos = commands.add_parser("cases")
    videos.add_argument("--evaluation", type=Path, required=True)
    videos.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "training":
        training(args.runs, args.output)
    elif args.command == "robustness":
        robustness(args.evaluations, args.output)
    else:
        cases(args.evaluation, args.output)


if __name__ == "__main__":
    main()

