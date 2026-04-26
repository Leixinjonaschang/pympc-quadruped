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

### 2. Install Simulators
#### a. Mujoco

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

#### b. Isaac Gym
- Download Isaac Gym Preview Release from this [website](https://developer.nvidia.com/isaac-gym). 
- The tutorial for installation is in the `./isaacgym/docs/install.html`. Install it into this project's virtual environment if you use the Isaac Gym demo.
    ~~~
    $ cd ${path-to-issacgym}/python
    $ source ${path-to-pympc-quadruped}/.venv/bin/activate
    (.venv)$ pip install -e .
    ~~~
- Then you can trying to run examples in `./isaacgym/python/examples`. Note that if you follow the instructions above, you need to run the examples in the virtual environments.
    ~~~
    $ cd ${path-to-isaacgym}/python/examples
    $ source ${path-to-pympc-quadruped}/.venv/bin/activate
    (.venv)$ python 1080_balls_of_solitude.py
    ~~~
- For troubleshooting, check `./isaacgym/docs/index.html`

**Note**: If you meet the following issue like
`ImportError: libpython3.7m.so.1.0: cannot open shared object file: No such file or directory` when running the examples, you can try 
`export LD_LIBRARY_PATH=/path/to/conda/envs/your_env/lib` before executing your python script. If you are not using conda, the path shoud be `/path/to/libpython/directory`.

### 3. Install Dependences

**a) Install Pinocchio**

Pinocchio is installed from PyPI through the `pin` dependency in `pyproject.toml`.

**b) Install other dependences**

~~~
$ cd ${path-to-pympc-quadruped}
$ uv sync
~~~

## Sign Convention
