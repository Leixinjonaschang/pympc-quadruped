import mujoco
import numpy as np


LEG_NAMES = ("FL", "FR", "RL", "RR")


def center_viewer_on_robot(viewer, data):
    viewer.cam.lookat[:] = data.body("trunk").xpos


def get_viewer_update_interval(model, rate_hz):
    if rate_hz <= 0:
        return None

    sim_dt = model.opt.timestep
    return max(1, int(round(1.0 / (rate_hz * sim_dt))))


def format_vector(vec, precision=2):
    return " ".join(f"{value:.{precision}f}" for value in vec)


def update_viewer_monitor(
    viewer,
    data,
    robot_data,
    predictive_controller,
    gait,
    swing_states,
    contact_forces,
    base_vel_base_des,
    yaw_turn_rate_des,
    iter_counter,
    monitor_rate,
    real_time_factor,
):
    if viewer is None:
        return

    solve_time = predictive_controller.last_mpc_solve_time
    solve_time_text = "pending" if solve_time is None else f"{solve_time * 1000:.2f} ms"
    solve_iter = predictive_controller.last_mpc_solve_iteration
    solve_iter_text = "pending" if solve_iter is None else str(solve_iter)
    monitor_rate_text = "off" if monitor_rate <= 0 else f"{monitor_rate:.1f} Hz"

    contact_states = ["0" if swing_state > 0 else "1" for swing_state in swing_states]
    contact_text = " ".join(
        f"{leg}:{state}" for leg, state in zip(LEG_NAMES, contact_states)
    )

    leg_fz = contact_forces[2::3]
    leg_fz_text = " ".join(
        f"{leg}:{fz:.1f}" for leg, fz in zip(LEG_NAMES, leg_fz)
    )

    rpy_deg = np.rad2deg(robot_data.rpy_base)

    viewer.set_texts(
        (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            (
                "Sim time\n"
                "Real-time factor\n"
                "Control iter\n"
                "Monitor rate\n"
                "Gait / phase\n"
                "Cmd vel / yaw rate\n"
                "Base z\n"
                "Base rpy deg\n"
                "Base vel\n"
                "Base omega\n"
                "Contact\n"
                "Last MPC solve\n"
                "Last MPC iter\n"
                "Sum Fz\n"
                "Leg Fz"
            ),
            (
                f"{data.time:.3f} s\n"
                f"{real_time_factor:.2f}x\n"
                f"{iter_counter}\n"
                f"{monitor_rate_text}\n"
                f"{gait.name} / {gait.phase:.2f}\n"
                f"{format_vector(base_vel_base_des)} / {yaw_turn_rate_des:.2f}\n"
                f"{robot_data.pos_base[2]:.3f} m\n"
                f"{format_vector(rpy_deg, precision=1)}\n"
                f"{format_vector(robot_data.lin_vel_base)}\n"
                f"{format_vector(robot_data.ang_vel_base)}\n"
                f"{contact_text}\n"
                f"{solve_time_text}\n"
                f"{solve_iter_text}\n"
                f"{np.sum(leg_fz):.1f} N\n"
                f"{leg_fz_text}"
            ),
        )
    )
