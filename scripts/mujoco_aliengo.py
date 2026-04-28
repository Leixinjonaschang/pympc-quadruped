"""MuJoCo Aliengo demo for the linear MPC quadruped controller.

The script is intentionally kept as the top-level orchestration layer:
MuJoCo provides simulator state, RobotData converts it into controller-friendly
kinematic quantities, the gait scheduler decides contact phases, MPC computes
stance contact forces, swing-foot controllers generate foot targets, and the leg
controller maps those commands into joint torques.
"""

import os
import sys
# Keep the teaching scripts runnable from the repository root without requiring
# the project to be installed as a Python package.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time

import mujoco
import numpy as np

from linear_mpc.gait import Gait
from linear_mpc.leg_controller import LegController
from config.linear_mpc_configs import LinearMpcConfig
from linear_mpc.mpc import ModelPredictiveController
from utils.mujoco_foot_trajectory_visualization import update_viewer_foot_trajectories
from utils.mujoco_simulation_utils import (
    get_simulated_sensor_data,
    get_true_simulation_data,
    reset_robot_state,
)
from utils.mujoco_viewer_utils import (
    center_viewer_on_robot,
    get_viewer_update_interval,
    update_viewer_monitor,
)
from config.robot_configs import AliengoConfig
from utils.robot_data import RobotData
from linear_mpc.swing_foot_trajectory_generator import SwingFootTrajectoryGenerator


STATE_ESTIMATION = False


def parse_args():
    """Parse demo options used for smoke tests, GUI demos, and visualization."""
    parser = argparse.ArgumentParser(
        description="Run the Aliengo linear MPC demo with the official MuJoCo API."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Number of simulation steps to run. Use 0 to run until interrupted.",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run without opening the MuJoCo passive viewer.",
    )
    parser.add_argument(
        "--monitor-rate",
        type=float,
        default=5.0,
        help="Viewer monitor overlay refresh rate in Hz. Use 0 to disable it.",
    )
    parser.add_argument(
        "--foot-traj-rate",
        type=float,
        default=40.0,
        help=(
            "Viewer swing-foot trajectory redraw rate in Hz. "
            "Use 0 to disable it."
        ),
    )
    parser.add_argument(
        "--foot-traj-samples",
        type=int,
        default=64,
        help="Samples across the MPC horizon for each visualized foot trajectory.",
    )
    return parser.parse_args()


def run_control_loop(
    model,
    data,
    robot_config,
    robot_data,
    steps=0,
    viewer=None,
    monitor_rate=20.0,
    foot_traj_rate=20.0,
    foot_traj_samples=256,
):
    """Run the closed-loop locomotion controller.

    One loop iteration corresponds to one low-level control step. The MPC solver
    is updated at its own slower interval inside ModelPredictiveController, while
    the leg controller produces torques every iteration.
    """
    predictive_controller = ModelPredictiveController(LinearMpcConfig, robot_config)
    leg_controller = LegController(robot_config.Kp_swing, robot_config.Kd_swing)

    # The gait object encodes the contact schedule. TROTTING10 alternates
    # diagonal leg pairs and provides both swing phase and MPC contact tables.
    gait = Gait.TROTTING10
    swing_foot_trajs = [SwingFootTrajectoryGenerator(leg_idx) for leg_idx in range(4)]

    # Desired base velocity is expressed in the robot base frame. The MPC module
    # converts it to the world frame using the current base orientation.
    vel_base_des = np.array([LinearMpcConfig.cmd_xvel, LinearMpcConfig.cmd_yvel, 0.])
    yaw_turn_rate_des = LinearMpcConfig.cmd_yaw_turn_rate

    iter_counter = 0
    monitor_update_interval = get_viewer_update_interval(model, monitor_rate)
    foot_traj_update_interval = get_viewer_update_interval(model, foot_traj_rate)
    wall_start = time.monotonic()
    sim_start = data.time

    while steps == 0 or iter_counter < steps:
        if viewer is not None and not viewer.is_running():
            break

        # In this teaching demo we use MuJoCo ground-truth state. The
        # state-estimation path is left as a future extension in RobotData.
        if not STATE_ESTIMATION:
            sensor_data = get_true_simulation_data(model, data)
        else:
            sensor_data = get_simulated_sensor_data(data)

        # RobotData runs Pinocchio kinematics and exposes foot positions,
        # Jacobians, and base-frame quantities used by MPC and leg control.
        robot_data.update(
            pos_base=sensor_data[0],
            lin_vel_base=sensor_data[1],
            quat_base=sensor_data[2],
            ang_vel_base=sensor_data[3],
            q=sensor_data[4],
            qdot=sensor_data[5],
        )

        gait.set_iteration(predictive_controller.iterations_between_mpc, iter_counter)
        swing_states = gait.get_swing_state()
        gait_table = gait.get_gait_table()

        # The MPC computes optimal ground reaction forces for stance legs. Swing
        # legs receive zero contact force and are handled by foot tracking below.
        predictive_controller.update_robot_state(robot_data)
        contact_forces = predictive_controller.update_mpc_if_needed(
            iter_counter,
            vel_base_des,
            yaw_turn_rate_des,
            gait_table,
            solver='drake',
            debug=False,
            iter_debug=0,
        )

        pos_targets_swingfeet = np.zeros((4, 3))
        vel_targets_swingfeet = np.zeros((4, 3))

        # For swing legs, plan a foothold and sample the cubic Hermite swing
        # trajectory. Targets are returned in the base frame for the leg PD task.
        for leg_idx in range(4):
            if swing_states[leg_idx] > 0:   # leg is in swing state
                swing_foot_trajs[leg_idx].set_foot_placement(
                    robot_data, gait, vel_base_des, yaw_turn_rate_des
                )
                base_pos_base_swingfoot_des, base_vel_base_swingfoot_des = \
                    swing_foot_trajs[leg_idx].compute_traj_swingfoot(
                        robot_data, gait
                    )
                pos_targets_swingfeet[leg_idx, :] = base_pos_base_swingfoot_des
                vel_targets_swingfeet[leg_idx, :] = base_vel_base_swingfoot_des

        # The leg controller combines stance force tracking and swing-foot PD
        # tracking, then outputs one torque for each actuated joint.
        torque_cmds = leg_controller.update(
            robot_data,
            contact_forces,
            swing_states,
            pos_targets_swingfeet,
            vel_targets_swingfeet,
        )
        data.ctrl[:] = torque_cmds

        mujoco.mj_step(model, data)
        if viewer is not None:
            center_viewer_on_robot(viewer, data)
            # Debug geometry can be expensive, so it is redrawn at a separate
            # rate from the physics and low-level control loop.
            if (
                foot_traj_update_interval is not None
                and iter_counter % foot_traj_update_interval == 0
            ):
                update_viewer_foot_trajectories(
                    viewer,
                    robot_data,
                    swing_foot_trajs,
                    gait,
                    iter_counter,
                    predictive_controller.iterations_between_mpc,
                    predictive_controller.horizon,
                    predictive_controller.dt,
                    vel_base_des,
                    yaw_turn_rate_des,
                    swing_states,
                    contact_forces,
                    foot_traj_samples,
                )
            # The text overlay is also throttled independently from physics.
            if (
                monitor_update_interval is not None
                and iter_counter % monitor_update_interval == 0
            ):
                elapsed_wall_time = max(time.monotonic() - wall_start, 1e-9)
                real_time_factor = (data.time - sim_start) / elapsed_wall_time
                update_viewer_monitor(
                    viewer,
                    data,
                    robot_data,
                    predictive_controller,
                    gait,
                    swing_states,
                    contact_forces,
                    vel_base_des,
                    yaw_turn_rate_des,
                    iter_counter,
                    monitor_rate,
                    real_time_factor,
                )
            viewer.sync()
        iter_counter += 1
        
        time.sleep(0.0002)  # run the simulation slower for visualization

        # Long GUI demos periodically reset to avoid unbounded counter growth and
        # to clear any accumulated user-scene debug geometry.
        if iter_counter == 50000:
            reset_robot_state(model, data, robot_config)
            if viewer is not None:
                viewer.user_scn.ngeom = 0
            iter_counter = 0


def main():
    args = parse_args()
    cur_path = os.path.dirname(__file__)
    mujoco_xml_path = os.path.join(cur_path, '../robot/aliengo/aliengo.xml')
    model = mujoco.MjModel.from_xml_path(mujoco_xml_path)
    data = mujoco.MjData(model)

    robot_config = AliengoConfig

    # Initialize MuJoCo state before constructing RobotData so the first
    # controller update sees a valid floating-base pose and joint configuration.
    reset_robot_state(model, data, robot_config)
    mujoco.mj_step(model, data)

    urdf_path = os.path.join(cur_path, '../robot/aliengo/urdf/aliengo.urdf')
    robot_data = RobotData(urdf_path, state_estimation=STATE_ESTIMATION)

    if args.no_viewer:
        run_control_loop(
            model,
            data,
            robot_config,
            robot_data,
            steps=args.steps,
            viewer=None,
            monitor_rate=args.monitor_rate,
            foot_traj_rate=args.foot_traj_rate,
            foot_traj_samples=args.foot_traj_samples,
        )
    else:
        from mujoco import viewer as mujoco_viewer

        with mujoco_viewer.launch_passive(model, data) as viewer:
            run_control_loop(
                model,
                data,
                robot_config,
                robot_data,
                steps=args.steps,
                viewer=viewer,
                monitor_rate=args.monitor_rate,
                foot_traj_rate=args.foot_traj_rate,
                foot_traj_samples=args.foot_traj_samples,
            )

        
if __name__ == '__main__':
    main()
