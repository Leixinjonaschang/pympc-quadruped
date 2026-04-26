import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../config'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../linear_mpc'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils/'))

import argparse

import mujoco
import numpy as np

from gait import Gait
from leg_controller import LegController
from linear_mpc_configs import LinearMpcConfig
from mpc import ModelPredictiveController
from robot_configs import AliengoConfig
from robot_data import RobotData
from swing_foot_trajectory_generator import SwingFootTrajectoryGenerator


STATE_ESTIMATION = False

FOOT_GEOMS = ("fl_foot", "fr_foot", "rl_foot", "rr_foot")
THIGH_BODIES = ("FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh")


def center_viewer_on_robot(viewer, data):
    viewer.cam.lookat[:] = data.body("trunk").xpos


def reset(model, data, robot_config):
    mujoco.mj_resetData(model, data)
    # q_pos_init = np.array([
    #     0, 0, 0.116536,
    #     1, 0, 0, 0,
    #     0, 1.16, -2.77,
    #     0, 1.16, -2.77,
    #     0, 1.16, -2.77,
    #     0, 1.16, -2.77
    # ])
    q_pos_init = np.array([
        0, 0, robot_config.base_height_des,
        1, 0, 0, 0,
        0, 0.8, -1.6,
        0, 0.8, -1.6,
        0, 0.8, -1.6,
        0, 0.8, -1.6
    ])
    
    q_vel_init = np.array([
        0, 0, 0, 
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0
    ])

    data.qpos[:] = q_pos_init
    data.qvel[:] = q_vel_init
    mujoco.mj_forward(model, data)


def get_object_linear_velocity(model, data, obj_type, name):
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, obj_type, obj_id, velocity, 0)
    return velocity[3:6].copy()


def get_true_simulation_data(model, data):
    mujoco.mj_forward(model, data)
    pos_base = data.body("trunk").xpos.copy()
    vel_base = get_object_linear_velocity(
        model, data, mujoco.mjtObj.mjOBJ_BODY, "trunk"
    )
    quat_base = data.sensordata[0:4].copy()
    omega_base = data.sensordata[4:7].copy()
    pos_joint = data.sensordata[10:22].copy()
    vel_joint = data.sensordata[22:34].copy()
    touch_state = data.sensordata[34:38].copy()
    pos_foothold = [
        data.geom(foot_geom).xpos.copy()
        for foot_geom in FOOT_GEOMS
    ]
    vel_foothold = [
        get_object_linear_velocity(
            model, data, mujoco.mjtObj.mjOBJ_GEOM, foot_geom
        )
        for foot_geom in FOOT_GEOMS
    ]
    pos_thigh = [
        data.body(thigh_body).xpos.copy()
        for thigh_body in THIGH_BODIES
    ]

    true_simulation_data = [
        pos_base, 
        vel_base, 
        quat_base, 
        omega_base, 
        pos_joint, 
        vel_joint, 
        touch_state, 
        pos_foothold, 
        vel_foothold, 
        pos_thigh
    ]
    # print(true_simulation_data)
    return true_simulation_data


def get_simulated_sensor_data(data):
    imu_quat = data.sensordata[0:4].copy()
    imu_gyro = data.sensordata[4:7].copy()
    imu_accelerometer = data.sensordata[7:10].copy()
    pos_joint = data.sensordata[10:22].copy()
    vel_joint = data.sensordata[22:34].copy()
    touch_state = data.sensordata[34:38].copy()
            
    simulated_sensor_data = [
        imu_quat, 
        imu_gyro, 
        imu_accelerometer, 
        pos_joint, 
        vel_joint, 
        touch_state
        ]
    # print(simulated_sensor_data)
    return simulated_sensor_data


def initialize_robot(model, data, viewer, robot_config, robot_data):
    predictive_controller = ModelPredictiveController(LinearMpcConfig, robot_config)
    leg_controller = LegController(robot_config.Kp_swing, robot_config.Kd_swing)
    init_gait = Gait.STANDING
    vel_base_des = [0., 0., 0.]
    
    for iter_counter in range(800):

        if not STATE_ESTIMATION:
            sensor_data = get_true_simulation_data(model, data)
        else:
            sensor_data = get_simulated_sensor_data(data)

        robot_data.update(
            pos_base=sensor_data[0],
            lin_vel_base=sensor_data[1],
            quat_base=sensor_data[2],
            ang_vel_base=sensor_data[3],
            q=sensor_data[4],
            qdot=sensor_data[5]
        )

        init_gait.set_iteration(predictive_controller.iterations_between_mpc,iter_counter)
        swing_states = init_gait.get_swing_state()
        gait_table = init_gait.get_gait_table()

        predictive_controller.update_robot_state(robot_data)
        contact_forces = predictive_controller.update_mpc_if_needed(
            iter_counter, vel_base_des, 0., gait_table, solver='drake', debug=False, iter_debug=0)

        torque_cmds = leg_controller.update(robot_data, contact_forces, swing_states)
        data.ctrl[:] = torque_cmds

        mujoco.mj_step(model, data)
        if viewer is not None:
            center_viewer_on_robot(viewer, data)
            viewer.sync()


def parse_args():
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
    return parser.parse_args()


def run_control_loop(model, data, robot_config, robot_data, steps=0, viewer=None):
    predictive_controller = ModelPredictiveController(LinearMpcConfig, robot_config)
    leg_controller = LegController(robot_config.Kp_swing, robot_config.Kd_swing)

    gait = Gait.TROTTING10
    swing_foot_trajs = [SwingFootTrajectoryGenerator(leg_idx) for leg_idx in range(4)]

    vel_base_des = np.array([1.2, 0., 0.])
    yaw_turn_rate_des = 0.

    iter_counter = 0

    while steps == 0 or iter_counter < steps:
        if viewer is not None and not viewer.is_running():
            break

        if not STATE_ESTIMATION:
            sensor_data = get_true_simulation_data(model, data)
        else:
            sensor_data = get_simulated_sensor_data(data)

        robot_data.update(
            pos_base=sensor_data[0],
            lin_vel_base=sensor_data[1],
            quat_base=sensor_data[2],
            ang_vel_base=sensor_data[3],
            q=sensor_data[4],
            qdot=sensor_data[5]
        )

        gait.set_iteration(predictive_controller.iterations_between_mpc, iter_counter)
        swing_states = gait.get_swing_state()
        gait_table = gait.get_gait_table()

        predictive_controller.update_robot_state(robot_data)

        contact_forces = predictive_controller.update_mpc_if_needed(iter_counter, vel_base_des, 
            yaw_turn_rate_des, gait_table, solver='drake', debug=False, iter_debug=0) 

        pos_targets_swingfeet = np.zeros((4, 3))
        vel_targets_swingfeet = np.zeros((4, 3))

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

        torque_cmds = leg_controller.update(robot_data, contact_forces, swing_states, pos_targets_swingfeet, vel_targets_swingfeet)
        data.ctrl[:] = torque_cmds

        mujoco.mj_step(model, data)
        if viewer is not None:
            center_viewer_on_robot(viewer, data)
            viewer.sync()
        iter_counter += 1

        if iter_counter == 50000:
            reset(model, data, robot_config)
            iter_counter = 0


def main():
    args = parse_args()
    cur_path = os.path.dirname(__file__)
    mujoco_xml_path = os.path.join(cur_path, '../robot/aliengo/aliengo.xml')
    model = mujoco.MjModel.from_xml_path(mujoco_xml_path)
    data = mujoco.MjData(model)

    robot_config = AliengoConfig

    reset(model, data, robot_config)
    mujoco.mj_step(model, data)

    urdf_path = os.path.join(cur_path, '../robot/aliengo/urdf/aliengo.urdf')
    robot_data = RobotData(urdf_path, state_estimation=STATE_ESTIMATION)
    # initialize_robot(model, data, None, robot_config, robot_data)

    if args.no_viewer:
        run_control_loop(
            model, data, robot_config, robot_data, steps=args.steps, viewer=None
        )
    else:
        from mujoco import viewer as mujoco_viewer

        with mujoco_viewer.launch_passive(model, data) as viewer:
            run_control_loop(
                model, data, robot_config, robot_data, steps=args.steps, viewer=viewer
            )

        
if __name__ == '__main__':
    main()
