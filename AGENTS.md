# Repository Guidelines

## Project Structure & Module Organization

This repository implements convex MPC locomotion for quadruped robots in Python. Core control logic lives in `linear_mpc/`, including MPC solving, gait scheduling, swing-foot trajectories, and leg control. Shared math, robot data containers, kinematics, dynamics, and simulator helpers live in `utils/`. Runtime parameters are grouped in `config/`, with robot-specific values in `robot_configs.py` and controller settings in `linear_mpc_configs.py`.

Simulator entry points are in `scripts/`: `mujoco_aliengo.py` for MuJoCo and `isaacgym_a1.py` for Isaac Gym. Robot descriptions and mesh assets are under `robot/aliengo/` and `robot/a1/`. Notes and result media belong in `doc/`, including `doc/results/`.

## Build, Test, and Development Commands

Current macOS Apple Silicon workflow uses `uv` with Python 3.13 and the official MuJoCo Python package:

```bash
uv python install 3.13
uv sync
```

Run the MuJoCo demo without a viewer for smoke testing:

```bash
uv run python scripts/mujoco_aliengo.py --no-viewer --steps 200
```

Run the MuJoCo GUI demo on macOS:

```bash
uv run python scripts/mjpython_uv.py scripts/mujoco_aliengo.py
```

Run the Isaac Gym demo after installing Isaac Gym into the same environment:

```bash
python scripts/isaacgym_a1.py
```

Pinocchio, MuJoCo, Isaac Gym, and some ROS-related packages require separate system or simulator setup; keep environment notes in `README.md` when these steps change.

## MuJoCo Migration Notes

The MuJoCo Aliengo demo was migrated from the deprecated `mujoco_py` API to the official `mujoco` Python API. The entry point now uses `mujoco.MjModel.from_xml_path`, `mujoco.MjData`, `mujoco.mj_step`, named accessors such as `data.body("trunk")` and `data.geom("fl_foot")`, and `mujoco.viewer.launch_passive` for the GUI path. The script also supports `--steps N` and `--no-viewer` for finite automated smoke tests.

The project now has `pyproject.toml` and `uv.lock`; `requirements.txt` is kept as a lean compatibility list and no longer includes `mujoco-py` or the old ROS environment capture. On macOS with uv-managed standalone Python, plain `uv run mjpython ...` may fail because MuJoCo's trampoline cannot find `libpython3.13.dylib`; use `scripts/mjpython_uv.py`, which points the MuJoCo app at uv's real Python shared library.

Migration verification performed:

```bash
uv sync
uv run python -c "import mujoco, pinocchio; from pydrake.all import MathematicalProgram, PiecewisePolynomial"
uv run python -c "import mujoco; mujoco.MjModel.from_xml_path('robot/aliengo/aliengo.xml')"
uv run python scripts/mujoco_aliengo.py --no-viewer --steps 200
uv run python scripts/mjpython_uv.py scripts/mujoco_aliengo.py --steps 1000
```

## Coding Style & Naming Conventions

Use Python with 4-space indentation. Follow the existing module style: lowercase file names, `snake_case` functions and variables, and `CamelCase` classes such as `ModelPredictiveController` and `LinearMpcConfig`. Keep configuration values explicit in `config/` rather than hard-coding robot or MPC constants in scripts. Prefer NumPy arrays for vector and matrix operations, and keep frame/sign conventions documented near the relevant math.

## Testing Guidelines

There is currently no formal test suite. For changes to controller math, add focused tests under a new `tests/` directory using `pytest`, with names like `test_mpc.py` or `test_kinematics.py`. At minimum, run the affected simulator script and verify the robot initializes, steps, and produces plausible contact forces or motion. For utility math, include deterministic tests that do not require MuJoCo or Isaac Gym.

## Commit & Pull Request Guidelines

Recent commits use short, imperative summaries such as `align desired base height` and `add multi-robots support`. Keep commit messages concise and action-oriented.

Pull requests should describe the behavior change, list simulator or test commands run, and mention any environment assumptions. Include screenshots or GIFs for visible locomotion changes, and link related issues or notes when applicable. Avoid mixing simulator setup, controller behavior, and documentation cleanup in one PR unless the changes are tightly related.
