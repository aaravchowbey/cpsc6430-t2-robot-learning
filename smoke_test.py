"""CPSC 6430 T2 - LIBERO smoke test (Checkpoint 1 practical-start evidence).

Creates a LIBERO-Goal environment, resets with a seed, prints observation
keys and the action space, takes one random step, and calls check_success().

Rendering note: this VM is headless and has no libEGL/libOSMesa, and
robosuite 1.4.1's default macros force MUJOCO_GL=egl. Setting MUJOCO_GL=glx
(the macros exemption list) makes robosuite fall back to the GLFW context for
imports, and we run with use_camera_obs=False so no GL context is created.
Camera observations are the documented next step (needs libosmesa6/egl).
"""
import os

os.environ.setdefault("MUJOCO_GL", "glx")  # must be set before robosuite import

import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

print("benchmark_root:", get_libero_path("benchmark_root"))
print("bddl_files:", get_libero_path("bddl_files"))
print("init_states:", get_libero_path("init_states"))
print("datasets:", get_libero_path("datasets"))

benchmark_dict = benchmark.get_benchmark_dict()
print("suites:", sorted(benchmark_dict.keys()))

task_suite = benchmark_dict["libero_goal"]()
print("libero_goal num tasks:", task_suite.get_num_tasks())

task = task_suite.get_task(0)
print("task 0 name:", task.name)
print("task 0 language:", task.language)
print("task 0 problem_folder:", task.problem_folder)
print("task 0 bddl_file:", task.bddl_file)

bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(
    bddl_file_name=bddl_file,
    use_camera_obs=False,  # headless VM: no GL/OSMesa yet (see docstring)
)
print("env created:", type(env).__name__)

env.seed(42)
obs = env.reset()
print("reset OK (seed=42)")

print("OBSERVATION KEYS:")
for k in sorted(obs.keys()):
    v = obs[k]
    shape = getattr(v, "shape", None)
    print(f"  {k}: shape={shape} dtype={getattr(v, 'dtype', type(v))}")

print("ACTION SPACE:", env.env.action_spec)  # wrapper doesn't expose it; inner robosuite env does
low, high = env.env.action_spec
print("action dim:", len(low), "| low:", low, "| high:", high)

action = np.random.uniform(low, high)
obs, reward, done, info = env.step(action)
print("one random step OK | reward:", reward, "| done:", done)

print("check_success():", env.check_success())

# also peek at a second task so the report can name two LIBERO-Goal tasks
task1 = task_suite.get_task(1)
print("task 1 name:", task1.name)
print("task 1 language:", task1.language)

env.close()
print("SMOKE TEST DONE")
