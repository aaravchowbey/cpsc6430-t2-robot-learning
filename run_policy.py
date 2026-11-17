"""Run a trained policy autonomously from frozen LIBERO episode start states."""

import argparse
import csv
from pathlib import Path
from time import perf_counter

# Configure the simulator's GL backend before importing LIBERO.
from replay_episode import localize_xml
import cv2
import h5py
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from train_policy import Policy, read_manifest, resize_view


LIVE_CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")


def rollout(task, row, data_dir, model, seed, max_steps, video_dir):
    with h5py.File(data_dir / row["dataset"], "r") as source:
        episode = source[f"data/{row['episode']}"]
        initial_state = episode["states"][0]
        xml = localize_xml(episode.attrs["model_file"])
    env = OffScreenRenderEnv(
        bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
        camera_heights=128, camera_widths=128, use_camera_obs=True,
    )
    writer = None
    try:
        env.seed(seed)
        env.reset()
        env.reset_from_xml_string(xml)
        env.sim.reset()
        obs = env.set_init_state(initial_state)
        if not np.allclose(env.get_sim_state(), initial_state, atol=1e-8, rtol=0):
            raise RuntimeError("Saved-state restoration did not match the recorded state")
        task_id = torch.tensor([int(row["task_id"])], dtype=torch.long)
        title = f"AUTONOMOUS {model.variant} | task {task_id.item()} | seed {seed}"
        if video_dir:
            video_dir.mkdir(parents=True, exist_ok=True)
            path = video_dir / f"{model.variant}_task{task_id.item()}_{row['episode']}_seed{seed}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (512, 320))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open video writer: {path}")

        def record(observation, step, success, repeat=1):
            if writer is None:
                return
            views = np.concatenate([observation[key][::-1] for key in LIVE_CAMERAS], axis=1)
            canvas = np.zeros((320, 512, 3), dtype=np.uint8)
            canvas[64:] = cv2.cvtColor(cv2.resize(views, (512, 256)), cv2.COLOR_RGB2BGR)
            for y, line in zip((23, 49), (title, f"{row['episode']} | step {step} | success={success}")):
                cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
            for _ in range(repeat):
                writer.write(canvas)

        elapsed = 0.0
        steps = 0
        success = bool(env.check_success())
        if success:
            raise RuntimeError(f"Initial state already satisfies task {task_id.item()}: {row['episode']}")
        record(obs, steps, success)
        for steps in range(1, max_steps + 1):
            start = perf_counter()
            images = np.stack([resize_view(obs[key]) for key in LIVE_CAMERAS])
            images = torch.from_numpy(images).permute(0, 3, 1, 2).unsqueeze(0)
            action = model(images, task_id)[0].numpy()
            elapsed += perf_counter() - start
            obs, _, done, _ = env.step(action)
            success = bool(env.check_success())
            record(obs, steps, success)
            if success or done:
                break
        record(obs, steps, success, repeat=40)
        outcome = "success" if success else "terminated" if done else "timeout"
        return outcome, steps, 1000 * elapsed / steps
    finally:
        if writer is not None:
            writer.release()
        env.close()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("splits/validation.csv"))
    parser.add_argument("--task", type=int, choices=(0, 1), help="Run only this task")
    parser.add_argument("--episode", help="Select one manifest episode for a given --task")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--per-task", type=int, default=1, help="Number of distinct episodes per task")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.per_task, args.max_steps, args.threads) < 1:
        parser.error("Rollouts, maximum steps and threads must be positive")
    if args.episode and (args.task is None or args.per_task != 1):
        parser.error("--episode requires --task and a single rollout")
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = Policy(checkpoint["variant"], checkpoint["state_dict"]["instructions"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    rows = read_manifest(args.manifest)
    if args.task is not None:
        rows = [row for row in rows if int(row["task_id"]) == args.task]
    if args.episode:
        rows = [row for row in rows if row["episode"] == args.episode]
    if not rows:
        parser.error("No episodes matched the selected manifest, task and episode")
    selected = []
    for task_id in sorted({int(row["task_id"]) for row in rows}):
        episodes = [row for row in rows if int(row["task_id"]) == task_id]
        if len(episodes) < args.per_task:
            parser.error(f"Task {task_id} has only {len(episodes)} distinct episode starts")
        selected.extend((task_id, args.seed + index, row) for index, row in enumerate(episodes[:args.per_task]))
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with args.results.open("x", newline="", encoding="utf-8") as result_file:
        results = csv.writer(result_file)
        results.writerow(("variant", "epoch", "smoke", "split", "task_id", "episode", "seed",
                          "outcome", "steps", "mean_inference_ms", "visual_tokens"))
        result_file.flush()
        for task_id, seed, row in selected:
            print(f"Running {model.variant} | task={task_id} | episode={row['episode']} | seed={seed}",
                  flush=True)
            outcome, steps, latency = rollout(
                suite.get_task(task_id), row, args.data_dir, model, seed, args.max_steps, args.video_dir,
            )
            results.writerow((model.variant, checkpoint["epoch"], checkpoint["smoke"], row["split"],
                              task_id, row["episode"], seed, outcome, steps, latency, model.visual_tokens))
            result_file.flush()
            print(f"{model.variant} task={task_id} episode={row['episode']} seed={seed} "
                  f"success={outcome == 'success'} outcome={outcome} steps={steps} "
                  f"mean_inference_ms={latency:.2f}", flush=True)


if __name__ == "__main__":
    main()
