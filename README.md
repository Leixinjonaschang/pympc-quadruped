# pympc-quadruped
A Python implementation about quadruped locomotion using convex model predictive control (MPC).

![image](https://github.com/yinghansun/pympc-quadruped/blob/main/doc/results/trotting10_mujoco.gif)

## Installation

This project is now MuJoCo-only. It uses Python 3.13 with `uv`, the official
`mujoco` Python package, Drake, Pinocchio, and the Aliengo MuJoCo/URDF assets in
`robot/aliengo/`.

### Ubuntu

Install `uv` first if it is not already available, then create the project
environment:

~~~bash
$ cd ${path-to-pympc-quadruped}
$ uv python install 3.13
$ uv sync
~~~

### macOS

On macOS, especially Apple Silicon with uv-managed Python, use the same `uv`
environment setup:

~~~bash
$ cd ${path-to-pympc-quadruped}
$ uv python install 3.13
$ uv sync
~~~

## Get Started

Run commands through `uv run` so they execute inside the project environment.
For example, use `uv run python ...` instead of calling `python` directly.

### Ubuntu

Run a headless smoke test:

~~~bash
$ uv run python scripts/mujoco_aliengo.py --no-viewer --steps 200
~~~

Run the MuJoCo GUI demo:

~~~bash
$ uv run python scripts/mujoco_aliengo.py --monitor-rate 20
~~~

### macOS

Run a headless smoke test:

~~~bash
$ uv run python scripts/mujoco_aliengo.py --no-viewer --steps 200
~~~

Launch the GUI through the wrapper that points MuJoCo's bundled `mjpython` app
at uv's Python shared library:

~~~bash
$ uv run python scripts/mjpython_uv.py scripts/mujoco_aliengo.py --monitor-rate 20
~~~

The wrapper is needed because plain `uv run mjpython ...` may not find
`libpython3.13.dylib` from uv's standalone Python installation.

### Options

Common demo arguments:

- `--no-viewer`: run without opening the MuJoCo GUI.
- `--steps N`: stop after `N` simulation steps. Use `0` to run until interrupted.
- `--monitor-rate HZ`: refresh the text overlay at `HZ`. Use `0` to disable it.
- `--foot-traj-rate HZ`: redraw swing-foot trajectory visualization at `HZ`. Use `0` to disable it.
- `--foot-traj-samples N`: set the number of samples used across the MPC horizon for each visualized foot trajectory.

## Sign Convention

The controller uses a right-handed, z-up world frame, matching the MuJoCo
Aliengo model. Unless a variable name explicitly says otherwise, positions,
linear velocities, angular velocities, forces, and trajectories are expressed in
the world frame.

Quaternions from MuJoCo are stored as `[w, x, y, z]`, where `w` is the real
part. Pinocchio expects `[x, y, z, w]`, so `RobotData` converts the order before
running kinematics.

Variable names follow this pattern:

~~~text
<expressed_frame>_<quantity>_<reference>_<target>
~~~

The prefix says which frame the vector is expressed in. If the prefix is
omitted, the value is expressed in the world frame. The reference and target
suffixes describe the relative quantity. If the reference suffix is omitted, the
quantity is relative to the world frame.

Examples:

- `pos_base`: base position relative to world, expressed in world frame.
- `lin_vel_base`: base linear velocity relative to world, expressed in world frame.
- `ang_vel_base`: base angular velocity relative to world, expressed in world frame.
- `pos_feet`: foot positions relative to world, expressed in world frame.
- `pos_base_feet`: foot positions relative to base, expressed in world frame.
- `base_pos_base_feet`: foot positions relative to base, expressed in base frame.
- `base_vel_base_feet`: foot velocities relative to base, expressed in base frame.
- `base_pos_base_thighs`: thigh positions relative to base, expressed in base frame.
- `base_vel_base_des`: commanded base velocity expressed in base frame.

Leg indexing is always `FL, FR, RL, RR`, matching the order used by gait tables,
foot positions, foot Jacobians, contact forces, and torque commands.

MuJoCo actuator torques are written in joint order:

~~~text
FL_hip, FL_thigh, FL_calf,
FR_hip, FR_thigh, FR_calf,
RL_hip, RL_thigh, RL_calf,
RR_hip, RR_thigh, RR_calf
~~~
