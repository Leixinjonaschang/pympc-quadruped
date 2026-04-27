# pympc-quadruped
A Python implementation about quadruped locomotion using convex model predictive control (MPC).

![image](https://github.com/yinghansun/pympc-quadruped/blob/main/doc/results/trotting10_mujoco.gif)

## Installation

### 1. Create a Virtual Environment

On macOS Apple Silicon, use Python 3.13 with `uv`:

~~~
$ cd ${path-to-pympc-quadruped}
$ uv python install 3.13
$ uv sync
~~~

### 2. Run MuJoCo

The MuJoCo demo uses the official `mujoco` Python package from PyPI. No manual
MuJoCo 2.1 download or `mujoco-py` setup is required.

Run a headless smoke test:

~~~
$ uv run python scripts/mujoco_aliengo.py --no-viewer --steps 200
~~~

On macOS with uv-managed Python, launch the passive viewer through the wrapper
that points MuJoCo's `mjpython` app at uv's Python shared library:

~~~
$ uv run python scripts/mjpython_uv.py scripts/mujoco_aliengo.py
~~~

### 3. Install Dependences

Pinocchio is installed from PyPI through the `pin` dependency in `pyproject.toml`.

~~~
$ cd ${path-to-pympc-quadruped}
$ uv sync
~~~

## Sign Convention
