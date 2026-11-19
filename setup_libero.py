"""Configure the pinned LIBERO checkout for the project's Python virtual environment."""

import os
from pathlib import Path
import shutil
import sys
import sysconfig

import yaml


def main():
    if sys.prefix == sys.base_prefix:
        raise SystemExit("Run this setup with the project's virtual environment Python.")

    project = Path(__file__).resolve().parent
    checkout = project / "LIBERO"
    benchmark = checkout / "libero" / "libero"
    if not (benchmark / "bddl_files").is_dir():
        raise SystemExit("Run git submodule update --init from the project root first.")

    packages = Path(sysconfig.get_path("purelib"))
    if os.name == "nt":
        suite = packages / "robosuite"
        dll = packages / "mujoco" / "mujoco.dll"
        if not (suite / "utils").is_dir() or not dll.is_file():
            raise SystemExit("Install requirements.txt in the virtual environment first.")
        # robosuite 1.4.1 otherwise selects the Linux-only EGL backend on Windows.
        macros = suite / "macros_private.py"
        settings = macros.read_text(encoding="utf-8") if macros.exists() else ""
        override = "MUJOCO_GPU_RENDERING = False"
        if not settings.rstrip().endswith(override):
            with macros.open("a", encoding="utf-8") as file:
                file.write(f"\n{override}\n")
        shutil.copyfile(dll, suite / "utils" / "mujoco.dll")

    # The upstream editable install does not reliably expose the nested package.
    (packages / "project_libero.pth").write_text(
        f"import sys; sys.path.insert(0, {str(checkout)!a})\n", encoding="ascii"
    )
    config_dir = project / ".libero"
    config_dir.mkdir(exist_ok=True)
    paths = {
        "benchmark_root": benchmark,
        "bddl_files": benchmark / "bddl_files",
        "init_states": benchmark / "init_files",
        "assets": benchmark / "assets",
        "datasets": project / "data",
    }
    paths["datasets"].mkdir(exist_ok=True)
    (config_dir / "config.yaml").write_text(
        yaml.safe_dump({key: str(path) for key, path in paths.items()}), encoding="utf-8"
    )
    print(f"LIBERO configured. Set LIBERO_CONFIG_PATH to {config_dir} before running the simulator.")


if __name__ == "__main__":
    main()
