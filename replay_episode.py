"""Replay LIBERO demonstrations in MuJoCo; this is not a learned policy."""
import argparse
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ["MUJOCO_GL"] = "glfw" if os.name == "nt" else os.environ.get("MUJOCO_GL", "glx")

import cv2
import h5py
import numpy as np
import robosuite
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def localize_xml(xml):
    """Replace the demonstration author's asset paths with local paths."""
    if isinstance(xml, bytes):
        xml = xml.decode("utf-8")
    root = ET.fromstring(xml)
    suite_root = Path(robosuite.__file__).parent
    assets_root = Path(get_libero_path("assets"))
    for element in root.iter():
        filename = element.get("file")
        if not filename:
            continue
        normalized = filename.replace("\\", "/")
        if "/robosuite/" in normalized:
            path = suite_root / normalized.rsplit("/robosuite/", 1)[1]
        elif "/assets/" in normalized:
            path = assets_root / normalized.split("/assets/", 1)[1]
        else:
            path = Path(filename)
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Cannot locate model asset: {filename} -> {path}")
        element.set("file", path.as_posix())
    return ET.tostring(root, encoding="unicode")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder containing the task HDF5 files")
    parser.add_argument("--task", type=int, choices=[0, 1], default=0)
    parser.add_argument("--episode", default="demo_0")
    parser.add_argument("--mode", choices=["states", "actions"], default="states")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    task = benchmark.get_benchmark_dict()["libero_goal"]().get_task(args.task)
    dataset = args.data_dir / f"{task.name}_demo.hdf5"
    with h5py.File(dataset, "r") as source:
        episode = source[f"data/{args.episode}"]
        states = episode["states"][()]
        actions = episode["actions"][()]
        xml = localize_xml(episode.attrs["model_file"])
    output = args.output or Path("output/replays") / f"{task.name}_{args.episode}_{args.mode}.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    env = None
    writer = None
    try:
        env = OffScreenRenderEnv(
            bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
            camera_heights=256, camera_widths=256, use_camera_obs=True,
        )
        env.seed(args.seed)
        env.reset()
        env.reset_from_xml_string(xml)
        env.sim.reset()
        obs = env.set_init_state(states[0])
        restored = env.get_sim_state()
        if not np.allclose(restored, states[0], atol=1e-8, rtol=0):
            raise RuntimeError("Saved-state restoration did not match the recorded state")
        print(f"DEMONSTRATION REPLAY | task={args.task}: {task.language}", flush=True)
        print(f"episode={args.episode} | seed={args.seed} | mode={args.mode} | restore=OK", flush=True)
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), 20, (512, 320))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {output}")

        def frame(observation, index, success):
            # MuJoCo's default OpenGL image convention has the vertical axis reversed.
            views = [observation[key][::-1] for key in ("agentview_image", "robot0_eye_in_hand_image")]
            canvas = np.zeros((320, 512, 3), dtype=np.uint8)
            canvas[64:] = cv2.cvtColor(np.concatenate(views, axis=1), cv2.COLOR_RGB2BGR)
            lines = [f"DEMO REPLAY ({args.mode}) | task {args.task} | {args.episode}",
                     f"seed {args.seed} | frame {index} | success={success}"]
            for y, text in zip((23, 49), lines):
                cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
            writer.write(canvas)

        success = bool(env.check_success())
        ever_success = success
        frame(obs, 0, success)
        count = len(states) - 1 if args.mode == "states" else len(actions)
        for index in range(count):
            if args.mode == "states":
                obs = env.set_init_state(states[index + 1])
            else:
                obs, _, _, _ = env.step(actions[index])
            success = bool(env.check_success())
            ever_success |= success
            frame(obs, index + 1, success)
        # Hold the measured final result so video players do not hide it at EOF.
        for _ in range(60):
            frame(obs, count, success)
        print(f"Final simulator success: {success} | success at any frame: {ever_success}", flush=True)
        print(f"Saved video: {output.resolve()}", flush=True)
        print("This is recorded demonstration replay, not autonomous policy execution.", flush=True)
    finally:
        if writer is not None:
            writer.release()
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
