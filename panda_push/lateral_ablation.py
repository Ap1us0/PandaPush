"""Replay a frozen SAC checkpoint with bounded early lateral steering."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .env import PandaPushEnv
from .control import PUSHER_RADIUS_M
from .evaluate import env_options, learned_policy, load_sac, run_episode, write_json, write_rows
from .failure_diagnostics import geometry
from .precontact_alignment import ALIGNMENT, alignment_target
from .success_eval import PROJECT, SCENE, evaluation_metrics, load_json, sha256


SOURCE = PROJECT / "results/success_seed29/followup"
ARMS = {
    "baseline": (False, False),
    "before_contact": (True, False),
    "early_contact": (False, True),
    "both": (True, True),
    "gated_before_contact": (False, False),
    "aligned_handoff": (False, False),
}
STEERING = dict(radius_m=0.080, correction_cap_m=0.006, gain=0.5, contact_steps=8)
GATED = dict(offset_m=0.023, predicted_offset_m=0.018, surface_gap_m=0.008,
             min_forward_m=0.001, correction_cap_m=0.0025, gain=0.35, max_steps=2)
HELDOUT = {"heldout": (176000, 96), "gated_a": (177000, 128), "gated_b": (178000, 128),
           "align_a": (179000, 128), "align_b": (180000, 128)}


def matched_probe(archived):
    """Match failures to distinct successes by layout and physical settings."""
    starts = archived[["start_x", "start_y"]].to_numpy()
    offset = archived[["goal_x", "goal_y"]].to_numpy() - starts
    features = np.column_stack([starts, offset, archived[["mass_kg", "side_m", "table_mu", "cube_mu"]]])
    features = (features - features.mean(axis=0)) / features.std(axis=0)
    available = archived.success.to_numpy().copy()
    pairs = []
    for failed in np.flatnonzero(~available):
        indices = np.flatnonzero(available)
        match = indices[np.square(features[indices] - features[failed]).sum(axis=1).argmin()]
        available[match] = False
        pairs.append((int(archived.seed.iloc[failed]), int(archived.seed.iloc[match])))
    return pairs


def steer_action(action, info, increment, weight):
    """Correct only the requested displacement perpendicular to the cube-to-goal ray."""
    toward = info["goal"][:2] - info["cube"][:2]
    forward = toward / np.linalg.norm(toward)
    offset = info["tcp"][:2] - info["cube"][:2]
    if (offset @ forward >= 0 or np.linalg.norm(offset) > STEERING["radius_m"]):
        return action, 0.
    side = np.array([-forward[1], forward[0]])
    action = np.clip(action.astype(float), -1., 1.)
    action /= max(1., np.linalg.norm(action))
    requested_side = increment * (action[:2] @ side)
    correction = np.clip(weight * STEERING["gain"] * (-offset @ side - requested_side),
                         -STEERING["correction_cap_m"], STEERING["correction_cap_m"])
    action[:2] += correction / increment * side
    return action, float(correction)


def gated_action(action, info, env):
    """Nudge an off-axis approach only when a forward command closes on the box."""
    toward = info["goal"][:2] - info["cube"][:2]
    forward = toward / np.linalg.norm(toward)
    side = np.array([-forward[1], forward[0]])
    offset = info["tcp"] - info["cube"]
    lateral = offset[:2] @ side
    if abs(lateral) <= GATED["offset_m"] or offset[:2] @ forward >= 0:
        return action, 0.

    # Measure clearance to the actual box footprint, including its corners.
    rotation = env.data.xmat[env.cube].reshape(3, 3)
    local = (offset @ rotation)[:2]
    gap = np.linalg.norm(np.maximum(np.abs(local) - env.half_side, 0.)) - PUSHER_RADIUS_M
    if gap > GATED["surface_gap_m"]:
        return action, 0.

    control = env.control
    target = control.action_target(np.clip(action, -1., 1.))
    free_budget = env.free_speed_mps * env.model.opt.timestep * env.physics_steps
    if free_budget:
        target = control.limit_target(target, free_budget)
    requested = target - info["tcp"]
    if requested[:2] @ forward <= GATED["min_forward_m"]:
        return action, 0.
    projected = ((target - info["cube"]) @ rotation)[:2]
    if np.linalg.norm(np.maximum(np.abs(projected) - env.half_side, 0.)) - PUSHER_RADIUS_M >= gap:
        return action, 0.

    direction = np.sign(lateral)
    excess = direction * (lateral + requested[:2] @ side) - GATED["predicted_offset_m"]
    if excess <= 0.:
        return action, 0.
    correction = -direction * min(GATED["correction_cap_m"], GATED["gain"] * excess)
    requested[:2] += correction * side
    return requested / control.increment_m, float(correction)


def replay(env, actor, seed, arm):
    before, early = ARMS[arm]
    first_contact = None
    trace = []
    correction = 0.
    gated_steps = 0
    alignment_steps = 0
    aligning = arm == "aligned_handoff"
    handoff_reason = ""
    aligned_command = False
    terminal = {}
    peak_tcp_speed = 0.

    def policy(observation, info, rng):
        nonlocal correction, gated_steps, aligning, alignment_steps, handoff_reason, aligned_command
        aligned_command = False
        if aligning:
            if first_contact is not None:
                handoff_reason = "contact"
            elif alignment_steps >= ALIGNMENT["max_steps"]:
                handoff_reason = "budget"
            else:
                target = alignment_target(env, info)
                if target is not None:
                    alignment_steps += 1
                    aligned_command = True
                    return (target - info["tcp"]) / env.control.increment_m
                handoff_reason = "aligned"
            aligning = False
        action = actor(observation, info, rng)
        correction = 0.
        if info["distance_m"] <= env.success_radius:
            return action
        if arm == "gated_before_contact" and first_contact is None and gated_steps < GATED["max_steps"]:
            action, correction = gated_action(action, info, env)
            gated_steps += int(correction != 0.)
        elif before and first_contact is None:
            action, correction = steer_action(action, info, env.control.increment_m, 1.)
        elif early and first_contact is not None and info["cube_contact"]:
            age = env.steps - first_contact
            if age < STEERING["contact_steps"]:
                weight = 1. - age / STEERING["contact_steps"]
                action, correction = steer_action(action, info, env.control.increment_m, weight)
        return action

    def observe(step, info, action):
        nonlocal first_contact, terminal, peak_tcp_speed
        terminal = info
        peak_tcp_speed = max(peak_tcp_speed, info.get("contact_tcp_speed_mps", 0.))
        if first_contact is None and info["cube_contact"]:
            first_contact = step
        trace.append(dict(seed=seed, step=step, error_m=info["distance_m"],
                          contact=info["cube_contact"], hold_steps=info["hold_steps"],
                          peak_force_n=info["peak_normal_force_n"],
                          cube_x=info["cube"][0], cube_y=info["cube"][1],
                          tcp_x=info["tcp"][0], tcp_y=info["tcp"][1],
                          requested_correction_m=correction, ik_rejected=info["ik_rejected"],
                          safety_reason=info.get("safety_reason", "")))
        if arm == "aligned_handoff":
            trace[-1].update(alignment_command=aligned_command, handoff_reason=handoff_reason)

    row = run_episode(env, seed, policy, on_step=observe)
    row.update(profile="ood", safety_reason=terminal["safety_reason"],
               collision_pair=terminal["collision_pair"], peak_contact_tcp_speed_mps=peak_tcp_speed,
               first_contact_step=first_contact,
               steering_steps=sum(item["requested_correction_m"] != 0. for item in trace))
    if arm == "aligned_handoff":
        row.update(alignment_steps=alignment_steps, handoff_reason=handoff_reason or "terminated")
    return row, trace


def run(output, dataset, arm):
    manifest = load_json(SOURCE / "manifest.json")
    record = next(row for row in manifest["models"] if row["arm"] == "seed29_tuned" and row["seed"] == 29)
    checkpoint, config_path = PROJECT / record["checkpoint"], PROJECT / record["config"]
    config = load_json(config_path)
    source_dir = SOURCE / "test/seed29_tuned/seed29"
    old = source_dir / "episodes.csv"
    old_hash = sha256(old)
    expected = load_json(source_dir / "complete.json")
    source_protocol = load_json(source_dir / "protocol.json")
    if (old_hash != expected["episodes_sha256"]
            or sha256(checkpoint) != source_protocol["checkpoint_sha256"]
            or sha256(config_path) != source_protocol["config_sha256"]):
        raise ValueError("Frozen source checkpoint, config, or rollouts changed")
    archived = pd.read_csv(old).query("profile == 'ood'")
    pairs = matched_probe(archived) if dataset == "probe" else []
    if dataset == "probe":
        seeds = sorted(seed for pair in pairs for seed in pair)
    else:
        first, count = HELDOUT[dataset]
        seeds = range(first, first + count)
    evaluation_env = dict(env_options(config), physics="ood", contact_diagnostics=True)
    root = output / dataset / arm
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "protocol.json", dict(dataset=dataset, arm=arm, model=record, seeds=list(seeds),
                                           source_sha256=old_hash, checkpoint_sha256=sha256(checkpoint),
                                           config_sha256=sha256(config_path), probe_pairs=pairs,
                                           steering={"gated_before_contact": GATED,
                                                     "aligned_handoff": ALIGNMENT}.get(arm, STEERING),
                                           evaluation_env=evaluation_env))
    env = PandaPushEnv(**evaluation_env)
    try:
        actor = learned_policy(load_sac(checkpoint, env))
        rows, traces = [], []
        for seed in seeds:
            row, trace = replay(env, actor, seed, arm)
            rows.append(row)
            traces.extend(trace)
    finally:
        env.close()
    episodes = pd.DataFrame(rows)
    episodes["safety_reason"] = episodes.safety_reason.mask(episodes.safety_reason.eq(""))
    episodes = geometry(episodes, config["success_radius"])
    if dataset == "probe":
        original = archived.set_index("seed").loc[episodes.seed].reset_index()
        coordinates = [key for key in SCENE if key not in ("profile", "seed")]
        if not np.allclose(episodes[coordinates], original[coordinates], atol=1e-10, rtol=0.):
            raise ValueError("Probe scenario differs from the archive")
        if arm == "baseline" and (not episodes.success.equals(original.success)
                                  or not np.allclose(episodes.final_error_m, original.final_error_m, atol=1e-7, rtol=0.)
                                  or not np.allclose(episodes.peak_force_n, original.peak_force_n, atol=1e-7, rtol=0.)):
            raise ValueError("Baseline probe diverged from the frozen rollouts")
    episodes.to_csv(root / "episodes.csv", index=False)
    write_rows(root / "traces.csv", traces)
    metrics = evaluation_metrics(episodes, precision=True)
    write_json(root / "metrics.json", metrics)
    print(f"{dataset} {arm}: {metrics['successes']}/{metrics['episodes']} successful", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=("probe", *HELDOUT), required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    args = parser.parse_args()
    run(args.output, args.dataset, args.arm)


if __name__ == "__main__":
    main()
