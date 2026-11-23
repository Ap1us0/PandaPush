"""Train or resume SB3 SAC with fixed development scenarios and archived metrics."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import torch
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from .env import PandaPushEnv
from .evaluate import GROUP_REWARDS, env_options, evaluate, summarize, write_json, write_rows
from .features import PushFeatures

SAC_KEYS = ("learning_rate", "gamma", "batch_size", "buffer_size", "learning_starts",
            "ent_coef", "train_freq", "gradient_steps")
MONITOR_KEYS = ("is_success", "distance_m", "cube_displacement_m", "unsafe", "joint_limit")


class ReplaySafetyInfo(gym.Wrapper):
    """Copy scalar safety metadata into HER without storing bulky contact diagnostics."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, {key: info[key] for key in (*MONITOR_KEYS, "safety_cost")}


class ExperimentCallback(BaseCallback):
    def __init__(self, config, eval_env, output):
        super().__init__()
        self.config, self.eval_env, self.output = config, eval_env, output
        self.last_eval_step = -1
        self.rows = []
        self.started = perf_counter()

    def record_evaluation(self):
        step = self.model.num_timesteps
        if self.last_eval_step == step:
            return
        first = self.config["eval_seed"]
        rows = evaluate(self.eval_env, range(first, first + self.config["eval_episodes"]), "sac", self.model)
        metrics = summarize(rows)
        write_rows(self.output / f"dev_{step:06d}.csv", rows)
        row = {"step": step, "elapsed_s": perf_counter() - self.started,
               **{key: value for key, value in metrics.items() if key != "failures"}}
        self.rows.append(row)
        write_rows(self.output / "development.csv", self.rows)
        self.last_eval_step = step
        print(f"step={step}  elapsed={row['elapsed_s']:.1f}s  "
              f"dev success={metrics['success_rate']:.1%}  "
              f"error={100 * metrics['mean_final_error_m']:.2f}cm  contact={metrics['episode_contact_rate']:.1%}",
              flush=True)

    def save(self, name):
        self.model.save(self.output / f"{name}.zip")
        self.model.save_replay_buffer(self.output / f"{name}_buffer.pkl")

    def _on_training_start(self):
        self.record_evaluation()

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        step = self.model.num_timesteps
        if step % self.config["eval_every"] == 0:
            self.record_evaluation()
        if step % self.config["checkpoint_every"] == 0:
            self.save(f"checkpoint_{step:06d}")

    def _on_training_end(self):
        self.save("final")
        # Capture the latest weights even when the previous callback used the same step counter.
        if self.last_eval_step == self.model.num_timesteps:
            self.rows.pop()
            self.last_eval_step = -1
        self.record_evaluation()


def train(config, output, resume):
    frequency = config["train_freq"]
    if any(config[key] <= 0 or config[key] % frequency for key in ("steps", "eval_every", "checkpoint_every")):
        raise ValueError("Training steps and intervals must be positive multiples of train_freq")
    if config["group"] == "sac_her" and config["learning_starts"] < config["horizon"]:
        raise ValueError("HER needs a complete episode before learning starts")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    use_her = config["group"] == "sac_her"
    copy_safety_info = use_her and bool(config.get("sparse_safety_cost"))
    options = env_options(config)
    training_options = dict(options, layout=config.get("train_layout", config["layout"]),
                            physics=config.get("train_physics", config["physics"]))
    raw_env = PandaPushEnv(**training_options)
    env = Monitor(ReplaySafetyInfo(raw_env) if copy_safety_info else raw_env,
                  str(output / "train"), info_keywords=MONITOR_KEYS)
    eval_env = PandaPushEnv(**options)
    try:
        if resume:
            model = SAC.load(resume, env=env, device="cpu", seed=config["seed"])
            model.load_replay_buffer(resume.with_name(resume.stem + "_buffer.pkl"), truncate_last_traj=True)
            if config.get("continuation_seed") is not None:
                model.set_random_seed(config["continuation_seed"])
        else:
            model = SAC(
                "MultiInputPolicy", env, device="cpu", seed=config["seed"],
                policy_kwargs={"features_extractor_class": PushFeatures, "net_arch": config["net_arch"]},
                replay_buffer_class=HerReplayBuffer if use_her else None,
                replay_buffer_kwargs={
                    "goal_selection_strategy": "future", "n_sampled_goal": 4, "copy_info_dict": copy_safety_info,
                } if use_her else None,
                **{key: config[key] for key in SAC_KEYS},
            )
        model.set_logger(configure(str(output), ["csv", "tensorboard"]))
        start = model.num_timesteps
        callback = ExperimentCallback(config, eval_env, output)
        model.learn(total_timesteps=config["steps"], callback=callback,
                    reset_num_timesteps=resume is None, log_interval=5)
        with (output / "train.monitor.csv").open(encoding="utf-8") as stream:
            next(stream)
            episodes = list(csv.DictReader(stream))
        write_json(output / "training_summary.json", {
            "start_step": start, "end_step": model.num_timesteps,
            "trained_steps": model.num_timesteps - start, "wall_s": perf_counter() - callback.started,
            "completed_episodes": len(episodes),
            "mean_training_return": sum(float(row["r"]) for row in episodes) / len(episodes) if episodes else None,
            "device": str(model.device), "torch": torch.__version__,
        })
    finally:
        env.close()
        eval_env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=GROUP_REWARDS)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--layout", choices=("fixed", "random"))
    parser.add_argument("--train-layout", choices=("random", "focused", "negative_y"),
                        help="Override training layout sampling while retaining the evaluation layout")
    parser.add_argument("--train-physics", choices=("contact_mix", "contact_mix_ood"),
                        help="Override training physics sampling while retaining the evaluation profiles")
    parser.add_argument("--physics", choices=("nominal", "idr"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--continuation-seed", type=int,
                        help="Start a fresh RNG stream while keeping the resumed model and replay buffer")
    parser.add_argument("--contact-speed-mps", type=float)
    parser.add_argument("--force-penalty-weight", type=float)
    parser.add_argument("--sparse-safety-cost", action="store_true", default=None,
                        help="Charge the remaining horizon when a sparse episode ends unsafely")
    args = parser.parse_args()
    overrides = ("group", "seed", "layout", "physics", "sparse_safety_cost",
                 "contact_speed_mps", "force_penalty_weight")
    if args.resume and (args.config or any(getattr(args, key) is not None for key in overrides)):
        parser.error("Resume uses the exact group, seed, and environment from the checkpoint configuration")
    if args.continuation_seed is not None and (
        args.resume is None or not 0 <= args.continuation_seed < 2**32
    ):
        parser.error("--continuation-seed requires --resume and a seed in [0, 2**32)")
    config_file = args.resume.parent / "config.json" if args.resume else args.config or Path("configs/default.json")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    config.setdefault("group", "sac_dense")
    config.setdefault("seed", 11)
    config.update({key: getattr(args, key) for key in overrides
                   if getattr(args, key) is not None})
    config.update(steps=args.steps, resumed_from=str(args.resume) if args.resume else None)
    if args.train_layout is not None:
        config["train_layout"] = args.train_layout
    if args.train_physics is not None:
        config["train_physics"] = args.train_physics
    config.pop("continuation_seed", None)
    if args.continuation_seed is not None:
        config["continuation_seed"] = args.continuation_seed
    train(config, args.output, args.resume)


if __name__ == "__main__":
    main()

