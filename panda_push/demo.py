"""View or render matched MuJoCo rollouts from the frozen seed-29 study."""

import argparse
from pathlib import Path
from time import sleep

import imageio_ffmpeg
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .env import PandaPushEnv
from .evaluate import env_options, learned_policy, load_sac
from .lateral_ablation import HELDOUT, SOURCE, replay
from .success_eval import PROJECT, load_json, sha256


STUDY = PROJECT / "results/success_seed29/aligned_handoff_ablation"
ARMS = ("baseline", "aligned_handoff")
DEFAULT_SEEDS = (179030, 180091)
VIDEO_SIZE = (1160, 446)
VIDEO_FPS = 16


class DemoEnv(PandaPushEnv):
    def __init__(self, *, record, **options):
        super().__init__(**options, render_mode="rgb_array" if record else "human")
        self.record = record
        self.frames = []

    def capture(self, *, after_reset=False):
        if self.record:
            frame = Image.fromarray(super().render())
            self.frames.append(frame.resize((550, 309), Image.Resampling.LANCZOS))
        else:
            if after_reset:
                super().render()
            sleep(1 / self.metadata["render_fps"])

    def reset(self, **options):
        result = super().reset(**options)
        self.capture(after_reset=True)
        return result

    def step(self, action):
        result = super().step(action)
        self.capture()
        return result


def run_case(seed, arm, record):
    manifest = load_json(SOURCE / "manifest.json")
    model = next(entry for entry in manifest["models"] if entry["arm"] == "seed29_tuned" and entry["seed"] == 29)
    batch = next(name for name in ("align_a", "align_b")
                 if HELDOUT[name][0] <= seed < sum(HELDOUT[name]))
    protocol = load_json(STUDY / batch / arm / "protocol.json")
    checkpoint, config = PROJECT / model["checkpoint"], PROJECT / model["config"]
    if (sha256(checkpoint) != protocol["checkpoint_sha256"]
            or sha256(config) != protocol["config_sha256"]):
        raise ValueError("The replay checkpoint or config differs from the archived rollout")
    options = dict(env_options(load_json(config)), physics="ood", contact_diagnostics=True)
    env = DemoEnv(record=record, **options)
    try:
        actor = learned_policy(load_sac(checkpoint, env))
        scored, trace = replay(env, actor, seed, arm)
        archive = pd.read_csv(STUDY / batch / arm / "episodes.csv").set_index("seed").loc[seed]
        if (scored["success"] != archive.success or scored["steps"] != archive.steps
                or abs(scored["final_error_m"] - archive.final_error_m) > 1e-6
                or abs(scored["peak_force_n"] - archive.peak_force_n) > 1e-4):
            raise ValueError(f"Demo replay diverged for {arm} in scenario {seed}")
        if not record:
            sleep(1.5)
        return env.frames, trace, scored
    finally:
        env.close()


def compose(seed, runs, step):
    base_font = ImageFont.load_default(size=17)
    label_font = ImageFont.load_default(size=20)
    title_font = ImageFont.load_default(size=28)
    image = Image.new("RGB", VIDEO_SIZE, "#12202d")
    draw = ImageDraw.Draw(image)
    draw.text((20, 13), f"PandaPush  /  Paired MuJoCo rollout  /  Scenario {seed}",
              fill="#f1f5f9", font=title_font)
    status = {
        DEFAULT_SEEDS[0]: "Recovered lateral miss",
        DEFAULT_SEEDS[1]: "Largest force peak in the alignment candidate",
    }.get(seed, "Same scene, different pre-contact control")
    draw.text((20, 50), status, fill="#a9c4da", font=base_font)
    for (frames, trace, scored), label, x in zip(
            runs, ("Frozen SAC", "Align, then SAC"), (20, 590), strict=True):
        index = min(step, len(frames) - 1)
        data = trace[index]
        draw.text((x, 79), label, fill="#f1f5f9", font=label_font)
        image.paste(frames[index], (x, 106))
        finished = index == len(frames) - 1
        if finished:
            result = "Success" if scored["success"] else "Miss"
            force = scored["peak_force_n"]
            metric = f"{result}  |  Error {100 * data['error_m']:.2f} cm  |  Episode peak {force:.1f} N"
            color = "#79eac0" if scored["success"] else "#ffb282"
        else:
            active_alignment = ((step == 0 and scored.get("alignment_steps", 0) > 0)
                                or data.get("alignment_command"))
            stage = "Align" if active_alignment else "SAC"
            metric = (f"{stage}  |  Step {index:02d}  |  Error {100 * data['error_m']:.2f} cm"
                      f"  |  Contact {data['peak_force_n']:.1f} N")
            color = "#cddbe7"
        draw.text((x, 421), metric, fill=color, font=base_font)
    return image


def render_video(output, seeds):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "mujoco_comparison.mp4"
    writer = imageio_ffmpeg.write_frames(
        str(path.resolve()), VIDEO_SIZE, fps=VIDEO_FPS, codec="libx264", pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p", quality=8, macro_block_size=2, output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    expected_frames = 0
    try:
        for seed in seeds:
            runs = [run_case(seed, arm, record=True) for arm in ARMS]
            finished_at = max(len(frames) for frames, _, _ in runs) - 1
            for step in range(finished_at + 1):
                frame = compose(seed, runs, step)
                repeat = 12 if step == 0 else 30 if step == finished_at else 1
                pixels = frame.tobytes()
                for _ in range(repeat):
                    writer.send(pixels)
                expected_frames += repeat
            if seed == seeds[0]:
                frame.save(output / "mujoco_comparison.png")
    finally:
        writer.close()
    frames, duration = imageio_ffmpeg.count_frames_and_secs(str(path.resolve()))
    if frames != expected_frames:
        raise RuntimeError(f"Expected {expected_frames} frames and encoded {frames}")
    print(f"{path}: {frames} frames, {duration:.2f} s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, help="Scenario from the two archived aligned-handoff batches")
    parser.add_argument("--viewer", choices=ARMS, help="Open a live MuJoCo window for one arm")
    parser.add_argument("--output", type=Path, default=Path("media"), help="Video and poster output directory")
    args = parser.parse_args()
    if args.seed is not None and not any(
            HELDOUT[name][0] <= args.seed < sum(HELDOUT[name]) for name in ("align_a", "align_b")):
        parser.error("Choose a scenario from 179000–179127 or 180000–180127")
    if args.viewer:
        run_case(args.seed if args.seed is not None else DEFAULT_SEEDS[0], args.viewer, record=False)
    else:
        render_video(args.output, (args.seed,) if args.seed is not None else DEFAULT_SEEDS)


if __name__ == "__main__":
    main()
