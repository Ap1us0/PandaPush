"""Screen PandaPush controller settings and freeze a held-out comparison."""

import argparse
from itertools import product
from pathlib import Path

import pandas as pd

from .evaluate import write_json
from .success_eval import (
    PROJECT, SCENE, SHARED, evaluation_metrics, load_json, protocol_for, read_manifest,
    read_matched, run_model, sha256, test_dir,
)

PHASES = ("archived", "approach", "intermediate", "continued", "focused", "layout")


def checkpoints(root, phase, seed):
    if phase == "archived":
        yield "100k", PROJECT / "results/success_progress/training" / f"seed{seed}" / "final.zip"
        if seed == 29:
            yield "50k", PROJECT / "results/contact_safety/training/speed_limit/seed29/final.zip"
    elif phase == "continued" and seed in (11, 47):
        run = root / "training" / f"seed{seed}"
        yield "150k", run / "final.zip"
    elif phase == "intermediate" and seed in (11, 47):
        steps = 120 if seed == 11 else 140
        yield f"{steps}k", root / "training" / f"seed{seed}" / f"checkpoint_{steps}000.zip"
    elif phase == "focused" and seed == 47:
        yield "130k_focused", root / "training" / "focused_seed47" / "final.zip"


def manifest_path(path):
    return path.relative_to(PROJECT).as_posix() if path.is_relative_to(PROJECT) else str(path)


def candidate(root, checkpoint, rank, tag, overrides, seed):
    arm = f"{rank}_{tag}"
    config = checkpoint.parent / "config.json"
    if overrides:
        config = root / "development" / "configs" / arm / f"seed{seed}.json"
    return dict(arm=arm, seed=seed, checkpoint=manifest_path(checkpoint), config=manifest_path(config))


def candidates(root, study, phase, seed):
    specs = study.get("checkpoints")
    if specs is None:
        sources = ((rank, checkpoint, study["variants"], {}) for rank, checkpoint in checkpoints(root, phase, seed))
    else:
        sources = (
            (spec["name"], PROJECT / spec["checkpoint"] if "checkpoint" in spec else root / spec["run_checkpoint"],
             spec["variants"], spec.get("base_overrides", {}))
            for spec in specs if spec["seed"] == seed and spec["phase"] == phase
        )
    for rank, checkpoint, variants, base_overrides in sources:
        for tag in variants:
            overrides = {**base_overrides, **study["variants"][tag]}
            yield candidate(root, checkpoint, rank, tag, overrides, seed), overrides


def screen(root, study, phase, seed):
    manifest = {"profiles": study["profiles"], **study["development"]}
    dev_path = root / "development" / "screen.json"
    for record, overrides in candidates(root, study, phase, seed):
        if overrides:
            config = PROJECT / record["config"]
            expected = {**load_json((PROJECT / record["checkpoint"]).parent / "config.json"), **overrides}
            if config.exists() and load_json(config) != expected:
                raise ValueError(f"Existing candidate settings differ at {config}")
            if not config.exists():
                config.parent.mkdir(parents=True, exist_ok=True)
                write_json(config, expected)
        run_model(dev_path, manifest, record)


def dev_rows(root, study):
    manifest = {"profiles": study["profiles"], **study["development"]}
    dev_path = root / "development" / "screen.json"
    seed_range = range(manifest["eval_seed"], manifest["eval_seed"] + manifest["episodes"])
    expected = pd.MultiIndex.from_product([manifest["profiles"], seed_range], names=["profile", "seed"])
    rows = []
    precision = "precision_reference" in study
    scene, settings = None, None
    for seed in study["seeds"]:
        for phase in PHASES:
            for record, _ in candidates(root, study, phase, seed):
                directory = test_dir(dev_path, record)
                path = directory / "episodes.csv"
                protocol = protocol_for(manifest, record)
                if (load_json(directory / "complete.json")["episodes_sha256"] != sha256(path)
                        or protocol != load_json(directory / "protocol.json")):
                    raise ValueError(f"Development archive integrity check failed at {directory}")
                episodes = pd.read_csv(path)
                shared = {key: protocol["evaluation_env"][key] for key in SHARED}
                if scene is None:
                    scene, settings = episodes[SCENE], shared
                if (not episodes.set_index(["profile", "seed"]).index.equals(expected)
                        or not scene.equals(episodes[SCENE]) or shared != settings):
                    raise ValueError(f"Development scenarios differ at {directory}")
                metrics = evaluation_metrics(episodes, precision)
                if precision:
                    ood = evaluation_metrics(episodes.loc[episodes.profile.eq("ood")], True)
                    metrics.update({f"ood_{key}": value for key, value in ood.items()})
                rows.append({**record, **metrics})
    return rows


def summarize(combo):
    seeds = len(combo)
    episodes = sum(row["episodes"] for row in combo)
    successes = sum(row["successes"] for row in combo)
    return {
        "options": [row["arm"] for row in combo],
        "successes": successes,
        "episodes": episodes,
        "success_pct": 100 * successes / episodes,
        "safety_stops": sum(row["safety_stops"] for row in combo),
        "mean_peak_force_n": sum(row["mean_peak_force_n"] for row in combo) / seeds,
        "p95_peak_force_n": sum(row["p95_peak_force_n"] for row in combo) / seeds,
        "max_peak_force_n": max(row["max_peak_force_n"] for row in combo),
    }


def rank_options(options, baseline, mean_margin, tail_margin):
    filtered = [
        (combo, scores) for combo, scores in options
        if scores["safety_stops"] <= baseline["safety_stops"]
        and scores["mean_peak_force_n"] <= baseline["mean_peak_force_n"] - mean_margin
        and scores["p95_peak_force_n"] <= baseline["p95_peak_force_n"] - tail_margin
    ]
    return sorted(filtered, key=lambda row: (
        -row[1]["successes"], row[1]["safety_stops"], row[1]["mean_peak_force_n"], row[1]["p95_peak_force_n"],
    ))


def select_peak_plans(rows, study, ranking):
    by_arm_seed = {(row["arm"], row["seed"]): row for row in rows}
    plans = {}
    for arm, tags in study["references"].items():
        models = tuple(by_arm_seed[tag, seed] for tag, seed in zip(tags, study["seeds"], strict=True))
        plans[arm] = (models, summarize(models))

    safety_budget = plans["peak_reference"][1]["safety_stops"]
    force_metrics = ("mean_peak_force_n", "p95_peak_force_n", "max_peak_force_n")
    for arm, reference in (("peak_matched", "peak_reference"), ("high_success_matched", "previous_high_success")):
        target = plans[reference][1]
        feasible = [
            (models, scores) for models, scores in ranking
            if scores["safety_stops"] <= safety_budget
            and all(scores[metric] <= target[metric] for metric in force_metrics)
        ]
        if not feasible:
            continue
        selected = min(feasible, key=lambda choice: (
            -choice[1]["successes"], choice[1]["safety_stops"],
            *(choice[1][metric] for metric in force_metrics),
        ))
        if all(selected[0] != plan[0] for plan in plans.values()):
            plans[arm] = selected
    return plans


def summarize_precision(models):
    metrics = ("mean_final_error_cm", "within_1cm_pct", "within_2cm_pct", "p95_final_error_cm")
    results = {}
    for scope, prefix in (("all", ""), ("ood", "ood_")):
        scoped_rows = [
            {**row, **{key.removeprefix(prefix): value for key, value in row.items() if key.startswith(prefix)}}
            if prefix else row for row in models
        ]
        results[scope] = summarize(scoped_rows)
        results[scope].update({key: sum(row[key] for row in scoped_rows) / len(models) for key in metrics})
    return results


def select_precision_plans(rows, study):
    by_arm_seed = {(row["arm"], row["seed"]): row for row in rows}
    reference = tuple(by_arm_seed[tag, seed] for tag, seed in zip(
        study["precision_reference"], study["seeds"], strict=True,
    ))
    baseline = summarize_precision(reference)
    ood_baseline = baseline["ood"]
    target_wins = study["heldout"]["success_target_pct"] * ood_baseline["episodes"] // 100 + 1
    min_dev_wins = min(target_wins, ood_baseline["successes"])
    force_metrics = ("mean_peak_force_n", "p95_peak_force_n", "max_peak_force_n", "safety_stops")
    feasible = []
    for models in product(*([row for row in rows if row["seed"] == seed] for seed in study["seeds"])):
        scores = summarize_precision(models)
        all_scores = scores["all"]
        if (scores["ood"]["successes"] >= min_dev_wins
                and all(scores[scope][metric] <= baseline[scope][metric]
                        for scope in scores for metric in force_metrics)
                and all_scores["mean_final_error_cm"] < baseline["all"]["mean_final_error_cm"]
                and all(all_scores[key] > baseline["all"][key] for key in ("within_1cm_pct", "within_2cm_pct"))):
            feasible.append((models, scores))
    if not feasible:
        raise ValueError("No development option meets the success, precision, and force constraints")
    best = min(feasible, key=lambda pair: (pair[1]["all"]["mean_final_error_cm"],
                                          pair[1]["ood"]["mean_final_error_cm"]))
    lower = min(feasible, key=lambda pair: (pair[1]["all"]["mean_peak_force_n"],
                                           pair[1]["ood"]["mean_peak_force_n"],
                                           pair[1]["all"]["mean_final_error_cm"]))
    plans = {"current": (reference, baseline), "centered": best}
    if lower != best:
        plans["lower_force_centering"] = lower
    return plans, int(min_dev_wins)


def select_focused_seed(rows, study):
    options = study["focused_seed"]
    seed, = study["seeds"]
    tag, = study["precision_reference"]
    reference = next(row for row in rows if row["arm"] == tag)
    baseline = summarize_precision((reference,))
    ood_baseline = baseline["ood"]
    target_wins = study["heldout"]["success_target_pct"] * ood_baseline["episodes"] // 100 + 1
    min_dev_wins = min(target_wins, ood_baseline["successes"])
    source = load_json(PROJECT / options["source_manifest"])
    current = tuple(record for record in source["models"] if record["arm"] == options["source_arm"])
    old = next(record for record in current if record["seed"] == seed)
    if (reference["checkpoint"] != old["checkpoint"]
            or load_json(PROJECT / reference["config"]) != load_json(PROJECT / old["config"])):
        raise ValueError("The development reference differs from the archived control configuration")

    guarded = []
    for row in rows:
        scores = summarize_precision((row,))
        if (row["arm"] != tag
                and all(scores[scope][key] <= baseline[scope][key]
                        for scope in scores for key in (
                            "mean_peak_force_n", "p95_peak_force_n", "max_peak_force_n", "safety_stops"))
                and all(scores[scope][key] >= baseline[scope][key]
                        for scope in scores for key in ("within_1cm_pct", "within_2cm_pct"))):
            guarded.append((row, scores))
    feasible = [item for item in guarded if item[1]["ood"]["successes"] >= min_dev_wins]
    exploratory = not feasible and options.get("allow_exploratory_validation", False)
    if exploratory:
        feasible = guarded
    if not feasible:
        raise ValueError(f"No development candidate meets the constraints for seed {seed}")

    def rank(item):
        scores = item[1]
        return (-scores["ood"]["successes"], scores["ood"]["mean_final_error_cm"],
                scores["all"]["mean_final_error_cm"], scores["ood"]["mean_peak_force_n"])

    def replace(item):
        row, scores = item
        return tuple(row if model["seed"] == seed else model for model in current), scores

    main = min(feasible, key=rank)
    plans = {"current": (current, baseline), f"seed{seed}_tuned": replace(main)}
    safer = [item for item in feasible
             if item[1]["ood"]["mean_peak_force_n"] < main[1]["ood"]["mean_peak_force_n"]
             and item[1]["ood"]["mean_peak_force_n"]
             <= baseline["ood"]["mean_peak_force_n"] - options["force_margin_n"]]
    if safer:
        plans[f"seed{seed}_force_margin"] = replace(min(safer, key=rank))
    return plans, {"min_dev_combined_successes": min_dev_wins,
                   "exploratory_validation": exploratory}


def confirmed_ood_rows(root, rows):
    path = root / "development_confirmation" / "manifest.json"
    manifest = read_manifest(path)
    confirmation, _ = read_matched(path, manifest)
    initial_path = root / "development" / "screen.json"
    confirmed_models = {(record["arm"], record["seed"]): record for record in manifest["models"]}
    compared = []
    for row in rows:
        record = confirmed_models.get((row["arm"], row["seed"]))
        if record is None:
            continue
        if any(record[key] != row[key] for key in ("checkpoint", "config")):
            raise ValueError(f"Development model changed for {row['arm']}")
        early = pd.read_csv(test_dir(initial_path, record) / "episodes.csv")
        later = confirmation.loc[confirmation.arm.eq(row["arm"]) & confirmation.training_seed.eq(row["seed"])]
        if set(early.seed).intersection(later.seed) or not later.profile.eq("ood").all():
            raise ValueError("Use disjoint scenario seeds and the combined profile for development confirmation")
        episodes = pd.concat([early.loc[early.profile.eq("ood")], later], ignore_index=True)
        metrics = evaluation_metrics(episodes, precision=True)
        compared.append({**row, **{f"ood_{key}": value for key, value in metrics.items()},
                         "max_peak_force_n": max(row["max_peak_force_n"], metrics["max_peak_force_n"])})
    if len(compared) != len(confirmed_models):
        raise ValueError("The development confirmation includes a model outside the initial screen")
    return compared


def select(root, study):
    output = root / "manifest.json"
    if output.exists():
        raise FileExistsError(f"Held-out comparison already frozen at {output}")
    rows = dev_rows(root, study)
    pd.DataFrame(rows).to_csv(root / "development" / "candidate_metrics.csv", index=False)
    if "focused_seed" in study:
        if study["focused_seed"].get("confirm_ood"):
            rows = confirmed_ood_rows(root, rows)
            pd.DataFrame(rows).to_csv(root / "development_confirmation" / "pooled_metrics.csv", index=False)
        plans, details = select_focused_seed(rows, study)
        freeze_plans(root, study, plans, "current", development_metrics_seed=study["seeds"][0], **details)
        return
    if "precision_reference" in study:
        plans, min_dev_wins = select_precision_plans(rows, study)
        freeze_plans(root, study, plans, "current", min_dev_combined_successes=min_dev_wins)
        return
    ranking = [(combo, summarize(combo)) for combo in product(
        *([row for row in rows if row["seed"] == seed] for seed in study["seeds"])
    )]
    if "references" in study:
        plans = select_peak_plans(rows, study, ranking)
        freeze_plans(root, study, plans, "peak_reference")
        return

    baseline = tuple(row for row in rows if row["arm"] == "100k_original")
    reference = summarize(baseline)
    feasible = rank_options(ranking, reference, 0, 0)
    guarded = rank_options(ranking, reference, study["force_mean_margin_n"], study["force_p95_margin_n"])
    conservative = rank_options(ranking, reference,
                                study["lower_force_mean_margin_n"], study["lower_force_p95_margin_n"])
    main = (guarded or feasible)[0]
    plans = {"baseline_100k": (baseline, reference), "tuned": main}
    if conservative and conservative[0][0] != main[0]:
        plans["lower_force"] = conservative[0]
    freeze_plans(root, study, plans, "baseline_100k", primary_margin_applied=bool(guarded),
                 best_under_zero_margin=feasible[0][1])


def freeze_plans(root, study, plans, baseline_arm, **details):
    selection = {arm: scores for arm, (_, scores) in plans.items()}
    selection.update(details)
    write_json(root / "development" / "selection.json", selection)
    models = [
        dict(arm=arm, **{key: row[key] for key in ("seed", "checkpoint", "config")})
        for arm, (combo, _) in plans.items() for row in combo
    ]
    write_json(root / "manifest.json", dict(**study["heldout"], profiles=study["profiles"],
                                            baseline=baseline_arm, models=models))
    print(selection, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--study", type=Path, default=Path("configs/success_90.json"))
    commands = parser.add_subparsers(dest="command", required=True)
    screen_args = commands.add_parser("screen")
    screen_args.add_argument("--seed", type=int, required=True)
    screen_args.add_argument("--phase", choices=PHASES, required=True)
    commands.add_parser("select")
    args = parser.parse_args()
    study = load_json(args.study)
    root = args.root.resolve()
    if args.command == "screen":
        if args.seed not in study["seeds"]:
            parser.error("Training seed not listed in study")
        screen(root, study, args.phase, args.seed)
    else:
        select(root, study)


if __name__ == "__main__":
    main()
