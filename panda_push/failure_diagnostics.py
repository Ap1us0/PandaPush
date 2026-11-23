"""Decompose push error and replay archived policy failures."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .env import PandaPushEnv
from .evaluate import env_options, learned_policy, load_sac, run_episode, write_rows
from .success_eval import load_json


def geometry(rows, radius):
    forward = rows[["goal_x", "goal_y"]].to_numpy() - rows[["start_x", "start_y"]].to_numpy()
    horizontal_distance = np.linalg.norm(forward, axis=1)
    axis = forward / horizontal_distance[:, None]
    to_target = rows[["goal_x", "goal_y"]].to_numpy() - rows[["end_x", "end_y"]].to_numpy()
    along = np.sum(to_target * axis, axis=1)
    lateral = axis[:, 0] * to_target[:, 1] - axis[:, 1] * to_target[:, 0]
    short, long, missed_side = along > radius, along < -radius, np.abs(lateral) > radius
    mode = np.select(
        [rows.success, rows.safety_reason.notna(), rows.contact_steps.eq(0),
         rows.first_success_step.notna(), rows.minimum_error_m.lt(radius),
         short & missed_side, long & missed_side, short, long, missed_side],
        ["success", "safety", "no_contact", "left_after_hold", "brief_entry",
         "short_and_lateral", "overshoot_and_lateral", "short", "overshoot", "lateral"],
        default="combined_error",
    )
    return rows.assign(axial_shortfall_cm=100 * along, signed_lateral_error_cm=100 * lateral,
                       angle_deg=np.degrees(np.arctan2(forward[:, 1], forward[:, 0])),
                       failure_mode=mode)


def trace_failures(rows, checkpoint, config):
    model = None
    for profile, group in rows.loc[~rows.success].groupby("profile", sort=False):
        env = PandaPushEnv(**dict(env_options(config), physics=profile))
        try:
            if model is None:
                model = load_sac(checkpoint, env)
            policy = learned_policy(model)
            for row in group.itertuples():
                trace = []

                def observe(step, info, _action):
                    trace.append({
                        "profile": profile, "seed": row.seed, "failure_mode": row.failure_mode,
                        "step": step, "error_m": info["distance_m"], "hold_steps": info["hold_steps"],
                        "contact": info["cube_contact"], "cube_delta_m": info["cube_delta_m"],
                        "peak_force_n": info["peak_normal_force_n"], "cube_x": info["cube"][0],
                        "cube_y": info["cube"][1], "cube_z": info["cube"][2],
                        "tcp_x": info["tcp"][0], "tcp_y": info["tcp"][1], "tcp_z": info["tcp"][2],
                    })

                result = run_episode(env, row.seed, policy, on_step=observe)
                if (result["success"] or result["steps"] != row.steps
                        or not np.isclose(result["final_error_m"], row.final_error_m, atol=1e-7, rtol=0)):
                    raise ValueError(f"Replay diverged for {profile} seed {row.seed}")
                yield from trace
        finally:
            env.close()


def hold_loss_events(trace, hold_steps):
    lost = trace.loc[trace.failure_mode == "left_after_hold"]
    events = []
    for (profile, seed), episode in lost.groupby(["profile", "seed"]):
        held = episode.loc[episode.hold_steps >= hold_steps].iloc[0]
        after = episode.loc[episode.step > held.step]
        first_exit = after.loc[after.hold_steps == 0].iloc[0]
        nearby = episode.loc[episode.step.between(first_exit.step - 1, first_exit.step)]
        events.append({
            "profile": profile, "seed": seed, "hold_step": held.step, "exit_step": first_exit.step,
            "contact_at_exit": bool(first_exit.contact), "contact_near_exit": bool(nearby.contact.any()),
            "contact_after_hold": bool(after.contact.any()), "max_near_exit_n": nearby.peak_force_n.max(),
            "closest_cm": 100 * episode.error_m.min(), "error_at_hold_cm": 100 * held.error_m,
            "final_cm": 100 * episode.error_m.iloc[-1],
        })
    return events


def plot_geometry(rows, radius, output):
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle

    fig, axis = plt.subplots(figsize=(7.4, 5.8), layout="constrained")
    axis.add_patch(Circle((0, 0), 100 * radius, facecolor="#dcece7", edgecolor="#5eada0", alpha=0.65))
    successes = rows.loc[rows.success]
    axis.scatter(successes.signed_lateral_error_cm, successes.axial_shortfall_cm, s=11,
                 c="#52a593", alpha=0.25, label="Success")
    kinds = {
        "lateral": ("Lateral miss", "#b36129", "o"),
        "short": ("Short push", "#9f2034", "^"),
        "short_and_lateral": ("Short and lateral", "#6d3a8c", "s"),
        "left_after_hold": ("Left after holding", "#1b5688", "D"),
    }
    other = ("Other failure", "#626573", "x")
    failures = rows.loc[~rows.success]
    labels = failures.failure_mode.where(failures.failure_mode.isin(kinds), "other")
    for kind, group in failures.groupby(labels, sort=False):
        label, color, marker = kinds.get(kind, other)
        axis.scatter(group.signed_lateral_error_cm, group.axial_shortfall_cm, label=label,
                     c=color, marker=marker, s=27, alpha=0.85)
    axis.set(xlabel="Signed lateral error (cm)", ylabel="Shortfall along push direction (cm)",
             title="Cube position at the end of the episode", aspect="equal")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.20)
    axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_json(args.config or args.checkpoint.parent / "config.json")
    rows = geometry(pd.read_csv(args.episodes), config["success_radius"])
    args.output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output / "failure_geometry.csv", index=False)
    counts = rows.groupby(["profile", "failure_mode"]).size().unstack(fill_value=0)
    counts.to_csv(args.output / "failure_counts.csv")
    trace = list(trace_failures(rows, args.checkpoint, config))
    if trace:
        write_rows(args.output / "failure_traces.csv", trace)
        exits = hold_loss_events(pd.DataFrame(trace), config["success_hold"])
        if exits:
            write_rows(args.output / "hold_loss_events.csv", exits)
    plot_geometry(rows, config["success_radius"], args.output / "failure_geometry.png")
    print(counts.to_string(), flush=True)


if __name__ == "__main__":
    main()
