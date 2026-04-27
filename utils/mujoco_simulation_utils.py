import mujoco
import numpy as np


FOOT_GEOMS = ("fl_foot", "fr_foot", "rl_foot", "rr_foot")
THIGH_BODIES = ("FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh")


def reset_robot_state(model, data, robot_config):
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
