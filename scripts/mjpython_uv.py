import importlib.util
import os
import sys


def main():
    mujoco_spec = importlib.util.find_spec("mujoco")
    if mujoco_spec is None or mujoco_spec.origin is None:
        raise RuntimeError("The mujoco package is not installed in this environment.")

    module_dir = os.path.dirname(mujoco_spec.origin)
    mjpython_bin = os.path.join(
        module_dir, "MuJoCo_(mjpython).app", "Contents", "MacOS", "mjpython"
    )
    libpython = os.path.join(
        sys.base_prefix,
        "lib",
        f"libpython{sys.version_info.major}.{sys.version_info.minor}.dylib",
    )

    if not os.path.exists(mjpython_bin):
        raise RuntimeError(f"MuJoCo mjpython binary not found: {mjpython_bin}")
    if not os.path.exists(libpython):
        raise RuntimeError(f"Python shared library not found: {libpython}")

    env = os.environ.copy()
    env["MJPYTHON_LIBPYTHON"] = libpython
    os.execve(mjpython_bin, [sys.executable, *sys.argv[1:]], env)


if __name__ == "__main__":
    main()
