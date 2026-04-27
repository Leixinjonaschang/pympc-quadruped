# Repository Guidelines

## Project Structure & Module Organization

This repository implements convex MPC locomotion for quadruped robots in Python. Core control logic lives in `linear_mpc/`, including MPC solving, gait scheduling, swing-foot trajectories, and leg control. Shared math, robot data containers, kinematics, dynamics, and simulator helpers live in `utils/`. Runtime parameters are grouped in `config/`, with robot-specific values in `robot_configs.py` and controller settings in `linear_mpc_configs.py`.

Simulator entry points are in `scripts/`, with `mujoco_aliengo.py` as the active MuJoCo Aliengo demo. Robot descriptions and mesh assets are under `robot/aliengo/`. Notes and result media belong in `doc/`, including `doc/results/`.

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
uv run python scripts/mjpython_uv.py scripts/mujoco_aliengo.py --monitor-rate 20
```

The MuJoCo GUI shows foot trajectory visualization over the MPC horizon by default. Use `--foot-traj-rate HZ` to control redraw frequency, `--foot-traj-rate 0` to disable it, and `--foot-traj-samples N` to control samples across the horizon for each foot path. The viewer overlay shows demo monitoring information such as base state, gait/contact state, contact forces, real-time factor, and last MPC solve time. Use `--monitor-rate HZ` to control the overlay refresh rate, or `--monitor-rate 0` to disable it.

Pinocchio, MuJoCo, and some ROS-related packages require separate system or simulator setup; keep environment notes in `README.md` when these steps change.

## MuJoCo Migration Notes

The MuJoCo Aliengo demo was migrated from the deprecated `mujoco_py` API to the official `mujoco` Python API. The entry point now uses `mujoco.MjModel.from_xml_path`, `mujoco.MjData`, `mujoco.mj_step`, named accessors such as `data.body("trunk")` and `data.geom("fl_foot")`, `mujoco.viewer.launch_passive` for the GUI path, and `viewer.user_scn` for horizon foot-trajectory debug geometry. The script also supports `--steps N` and `--no-viewer` for finite automated smoke tests, plus `--monitor-rate HZ` for throttled viewer overlay updates and `--foot-traj-rate HZ` for throttled trajectory redraws.

The project now has `pyproject.toml` and `uv.lock`; `requirements.txt` is kept as a lean compatibility list and no longer includes `mujoco-py` or the old ROS environment capture. On macOS with uv-managed standalone Python, plain `uv run mjpython ...` may fail because MuJoCo's trampoline cannot find `libpython3.13.dylib`; use `scripts/mjpython_uv.py`, which points the MuJoCo app at uv's real Python shared library.

Migration verification performed:

```bash
uv sync
uv run python -c "import mujoco, pinocchio; from pydrake.all import MathematicalProgram, PiecewisePolynomial"
uv run python -c "import mujoco; mujoco.MjModel.from_xml_path('robot/aliengo/aliengo.xml')"
uv run python scripts/mujoco_aliengo.py --no-viewer --steps 200
uv run python scripts/mjpython_uv.py scripts/mujoco_aliengo.py --steps 1000 --monitor-rate 20
```

## Coding Style & Naming Conventions

Use Python with 4-space indentation. Follow the existing module style: lowercase file names, `snake_case` functions and variables, and `CamelCase` classes such as `ModelPredictiveController` and `LinearMpcConfig`. Keep configuration values explicit in `config/` rather than hard-coding robot or MPC constants in scripts. Prefer NumPy arrays for vector and matrix operations, and keep frame/sign conventions documented near the relevant math.

## Testing Guidelines

There is currently no formal test suite. For changes to controller math, add focused tests under a new `tests/` directory using `pytest`, with names like `test_mpc.py` or `test_kinematics.py`. At minimum, run the affected simulator script and verify the robot initializes, steps, and produces plausible contact forces or motion. For utility math, include deterministic tests that do not require MuJoCo.

## Commit & Pull Request Guidelines

Recent commits use short, imperative summaries such as `align desired base height` and `add multi-robots support`. Keep commit messages concise and action-oriented.

Pull requests should describe the behavior change, list simulator or test commands run, and mention any environment assumptions. Include screenshots or GIFs for visible locomotion changes, and link related issues or notes when applicable. Avoid mixing simulator setup, controller behavior, and documentation cleanup in one PR unless the changes are tightly related.

## Codebase Structure Notes

This project is organized as a small quadruped control stack: simulator adapters, robot state abstraction, linear MPC control, robot model assets, and documentation.

```text
pympc-quadruped/
├── AGENTS.md
│   └── Repository collaboration and development guidelines.
├── README.md
│   └── Project overview, installation, MuJoCo usage, and convention notes.
├── LICENSE
│   └── Open source license.
├── pyproject.toml
│   └── uv/Python 3.13 project metadata and dependencies: drake, mujoco, pinocchio, qpsolvers, etc.
├── uv.lock
│   └── uv dependency lockfile.
├── requirements.txt
│   └── Lean compatibility dependency list.
├── .gitignore
│   └── Git ignore rules.
├── .DS_Store
│   └── macOS-generated file with no project semantics.
│
├── config/
│   ├── cmd.py
│   │   └── Currently empty; reserved for command or motion input configuration.
│   ├── linear_mpc_configs.py
│   │   └── MPC parameters: control timestep, MPC update interval, horizon, gravity, friction coefficient, Q/R weights, and default velocity commands.
│   └── robot_configs.py
│       └── Robot parameter abstraction. `RobotConfig` is the base class; `AliengoConfig` defines mass, desired base height, inertia, max vertical force, swing height, and swing-leg PD gains.
│
├── linear_mpc/
│   ├── gait.py
│   │   └── Gait scheduling. The `Gait` enum defines standing, trotting, jumping, and pacing patterns, then generates MPC contact tables and per-leg swing/stance phases.
│   ├── mpc.py
│   │   └── Core linear MPC. `ModelPredictiveController` converts robot state into a 13D single-rigid-body state, generates reference trajectories, builds continuous/discrete state-space models, assembles QP costs and friction constraints, and solves contact forces with Drake or qpsolvers.
│   ├── swing_foot_trajectory_generator.py
│   │   └── Swing-foot trajectory generation. It plans footholds from gait timing, base velocity, desired velocity, and yaw rate, then uses Drake `PiecewisePolynomial` cubic Hermite curves for foot trajectories.
│   └── leg_controller.py
│       └── Joint torque synthesis. Stance legs map MPC contact force through `tau = J.T @ -f`; swing legs use task-space PD control to track foot trajectory targets.
│
├── utils/
│   ├── robot_data.py
│   │   └── Robot state abstraction. It receives simulator or sensor state, normalizes quaternion/pose/joint data, uses Pinocchio to compute foot positions, foot Jacobians, thigh positions, relative foot velocities, and includes a terrain-normal estimation sketch.
│   ├── kinematics.py
│   │   └── Kinematics and Lie-group math utilities: quaternion/rotation/Euler conversions, so3/se3 helpers, SE3 adjoints, exponential maps, and open-chain forward kinematics.
│   ├── dynamics.py
│   │   └── Small dynamics helper for constructing 3x3 CoM inertia matrices from URDF inertia entries.
│   ├── mujoco_simulation_utils.py
│   │   └── MuJoCo reset, sensor, and simulator-state helpers.
│   ├── mujoco_viewer_utils.py
│   │   └── MuJoCo viewer camera, overlay, and update-rate helpers.
│   └── mujoco_foot_trajectory_visualization.py
│       └── MuJoCo debug geometry for horizon foot trajectory visualization.
│
├── scripts/
│   ├── mujoco_aliengo.py
│   │   └── Aliengo MuJoCo entry point. It loads MJCF, reads MuJoCo state, updates `RobotData`, runs gait/MPC/swing-foot/leg-control logic, writes `data.ctrl`, shows swing-foot trajectory debug geometry in GUI mode, and supports `--no-viewer`, `--steps`, `--monitor-rate`, `--foot-traj-rate`, and `--foot-traj-samples`.
│   └── mjpython_uv.py
│       └── macOS + uv MuJoCo GUI wrapper. It locates MuJoCo's bundled `mjpython` app and sets `MJPYTHON_LIBPYTHON`.
│
├── robot/
│   └── aliengo/
│       ├── aliengo.xml
│       │   └── MuJoCo MJCF model with trunk, legs, foot geoms, sensors, actuators, and ground setup.
│       ├── urdf/aliengo.urdf
│       │   └── Aliengo URDF used by Pinocchio for kinematics and Jacobians.
│       └── meshes/
│           ├── trunk.stl
│           ├── hip.stl
│           ├── thigh.stl
│           ├── thigh_mirror.stl
│           └── calf.stl
│               └── Aliengo visual/collision mesh assets.
│
└── doc/
    ├── linear_mpc.md
    │   └── Linear MPC theory notes: single-rigid-body model, Euler-angle approximation, discretization, QP form, and friction-cone constraints.
    ├── state_estimation_kf.md
    │   └── State-estimation theory notes for IMU orientation filtering plus linear Kalman filtering. The implementation path is not currently active.
    └── results/trotting10_mujoco.gif
        └── MuJoCo trotting demo result media.
```

### Control Abstraction Layers

1. **Asset layer**: `robot/` contains MJCF, URDF, meshes, and textures. MuJoCo loads `robot/aliengo/aliengo.xml`; Pinocchio uses URDF files for kinematics.

2. **Configuration layer**: `config/` separates robot constants and MPC tuning from algorithm code. `LinearMpcConfig` controls time scales, QP weights, gravity, and friction. `RobotConfig` subclasses provide robot-specific physical and control parameters.

3. **Math and state layer**: `utils/kinematics.py` provides geometry utilities. `utils/robot_data.py` is the main adapter between raw simulator state and controller-ready quantities such as `R_base`, `pos_feet`, `pos_base_feet`, `Jv_feet`, and `base_vel_base_feet`.

4. **Control primitive layer**: `linear_mpc/gait.py` schedules contacts, `swing_foot_trajectory_generator.py` plans swing-foot targets, and `leg_controller.py` maps contact forces or foot tracking errors into joint torques.

5. **Optimization layer**: `linear_mpc/mpc.py` is the core controller. It uses a 13D single-rigid-body state `x = [roll, pitch, yaw, pos, omega, vel, gravity]` and a 12D input `u = [f_FL, f_FR, f_RL, f_RR]`, discretizes the dynamics over the horizon, and solves a constrained QP for contact forces.

6. **Simulator entry layer**: `scripts/mujoco_aliengo.py` initializes MuJoCo, reads state, calls the shared control pipeline, and writes actuator commands back to the simulator.

The main runtime data flow is:

```text
MuJoCo state
        ↓
RobotData.update()
        ↓
Pinocchio computes foot positions, velocities, and Jacobians
        ↓
Gait generates swing_states and gait_table
        ↓
MPC solves contact forces from current state and gait_table
        ↓
SwingFootTrajectoryGenerator computes swing-leg targets
        ↓
LegController generates 12 joint torques
        ↓
Simulator actuator / DOF force commands
```

`state_estimation` is documented in `doc/state_estimation_kf.md`, but the active code path currently uses simulator ground-truth state. `RobotData.update(..., state_estimation=True)` raises `NotImplementedError`.
