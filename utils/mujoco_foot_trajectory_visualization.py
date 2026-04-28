import mujoco
import numpy as np

from config.linear_mpc_configs import LinearMpcConfig
from linear_mpc.swing_foot_trajectory_generator import SwingFootTrajectoryGenerator


LEG_TRAJECTORY_COLORS = (
    np.array([0.1, 0.35, 1.0, 1.0], dtype=np.float32),
    np.array([1.0, 0.45, 0.05, 1.0], dtype=np.float32),
    np.array([1.0, 0.9, 0.1, 1.0], dtype=np.float32),
    np.array([0.55, 0.2, 0.9, 1.0], dtype=np.float32),
)
SUPPORT_POLYGON_COLOR = np.array([0., 0., 0., 1.], dtype=np.float32)
CONTACT_FORCE_COLOR = np.array([0.1, 0.8, 0.15, 1.], dtype=np.float32)
FOOTHOLD_MARKER_COLOR = np.array([1.0, 0.0, 0.85, 1.0], dtype=np.float32)
GEOM_IDENTITY = np.eye(3).reshape(-1)


def add_user_geom(viewer, geom_type, size, pos, rgba):
    scene = viewer.user_scn
    if scene.ngeom >= scene.maxgeom:
        return None

    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, geom_type, size, pos, GEOM_IDENTITY, rgba)
    scene.ngeom += 1
    return geom


def draw_trajectory_line(viewer, start, end, rgba, width=2.0):
    geom = add_user_geom(
        viewer,
        mujoco.mjtGeom.mjGEOM_LINE,
        np.zeros(3),
        np.zeros(3),
        rgba,
    )
    if geom is None:
        return

    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        width,
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
    )


def draw_trajectory_sphere(viewer, pos, radius, rgba):
    add_user_geom(
        viewer,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0., 0.], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        rgba,
    )


def draw_foothold_marker(viewer, pos):
    marker_pos = np.asarray(pos, dtype=np.float64).copy()
    marker_pos[2] += 0.035
    draw_trajectory_line(
        viewer,
        pos,
        marker_pos,
        FOOTHOLD_MARKER_COLOR,
        width=4.0,
    )
    draw_trajectory_sphere(
        viewer,
        marker_pos,
        0.02,
        FOOTHOLD_MARKER_COLOR,
    )


def draw_polyline(viewer, points, rgba, width=2.0):
    if len(points) < 2:
        return

    for point_start, point_end in zip(points[:-1], points[1:]):
        draw_trajectory_line(viewer, point_start, point_end, rgba, width)


def draw_current_foot_markers(viewer, robot_data, swing_states):
    for leg_idx, foot_position in enumerate(robot_data.pos_feet):
        rgba = LEG_TRAJECTORY_COLORS[leg_idx].copy()
        if swing_states[leg_idx] > 0:
            rgba[3] = 0.3
        draw_trajectory_sphere(viewer, foot_position, 0.015, rgba)


def draw_support_polygon(viewer, robot_data, swing_states):
    stance_feet = np.array(
        [
            foot_position
            for foot_position, swing_state in zip(robot_data.pos_feet, swing_states)
            if swing_state <= 0
        ],
        dtype=np.float64,
    )
    if len(stance_feet) < 2:
        return

    center_xy = np.mean(stance_feet[:, :2], axis=0)
    angles = np.arctan2(
        stance_feet[:, 1] - center_xy[1],
        stance_feet[:, 0] - center_xy[0],
    )
    ordered_feet = stance_feet[np.argsort(angles)]
    if len(ordered_feet) > 2:
        ordered_feet = np.vstack((ordered_feet, ordered_feet[0]))

    draw_polyline(viewer, ordered_feet, SUPPORT_POLYGON_COLOR, width=1.5)


def draw_contact_forces(viewer, robot_data, swing_states, contact_forces):
    for leg_idx, foot_position in enumerate(robot_data.pos_feet):
        if swing_states[leg_idx] > 0:
            continue

        force = np.asarray(contact_forces[3 * leg_idx:3 * leg_idx + 3])
        if np.linalg.norm(force) < 1e-6:
            continue

        draw_trajectory_line(
            viewer,
            foot_position,
            foot_position + force / 1000.0,
            CONTACT_FORCE_COLOR,
            width=3.0,
        )


def get_swing_state_at_time(gait, iter_counter, iterations_between_mpc, time_from_now):
    period_iterations = iterations_between_mpc * gait.num_segment
    phase = (
        (iter_counter + time_from_now / LinearMpcConfig.dt_control)
        % period_iterations
    ) / period_iterations

    swing_offsets = (
        gait.stance_offsets_normalized + gait.stance_durations_normalized
    ) % 1.0
    swing_durations = 1.0 - gait.stance_durations_normalized
    swing_states = np.zeros(4, dtype=np.float32)
    for leg_idx in range(4):
        swing_phase = phase - swing_offsets[leg_idx]
        if swing_phase < 0:
            swing_phase += 1.0

        if 0.0 < swing_phase <= swing_durations[leg_idx]:
            swing_states[leg_idx] = swing_phase / swing_durations[leg_idx]

    return swing_states


def get_yaw_rotation(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0.],
        [np.sin(theta), np.cos(theta), 0.],
        [0., 0., 1.],
    ])


def predict_base_pose(robot_data, base_vel_base_des, yaw_turn_rate_des, time_from_now):
    yaw = robot_data.rpy_base[2] + yaw_turn_rate_des * time_from_now
    R_base = get_yaw_rotation(yaw)
    vel_world_des = robot_data.R_base @ base_vel_base_des
    pos_base = np.array(robot_data.pos_base, dtype=np.float32) \
        + vel_world_des * time_from_now

    return pos_base, R_base, vel_world_des


def predict_foothold(
    robot_data,
    leg_idx,
    gait,
    base_vel_base_des,
    yaw_turn_rate_des,
    time_from_now,
):
    pos_base, R_base, vel_world_des = predict_base_pose(
        robot_data, base_vel_base_des, yaw_turn_rate_des, time_from_now
    )
    total_stance_time = gait.stance_time
    total_swing_time = gait.swing_time
    RotZ = get_yaw_rotation(yaw_turn_rate_des * 0.5 * total_stance_time)
    pos_thigh_corrected = RotZ @ robot_data.base_pos_base_thighs[leg_idx]

    foothold = pos_base \
        + R_base @ (
            pos_thigh_corrected + base_vel_base_des * total_swing_time
        ) \
        + 0.5 * total_stance_time * vel_world_des

    foothold[0] += (
        0.5 * pos_base[2] / LinearMpcConfig.gravity
    ) * (vel_world_des[1] * yaw_turn_rate_des)
    foothold[1] += (
        0.5 * pos_base[2] / LinearMpcConfig.gravity
    ) * (-vel_world_des[0] * yaw_turn_rate_des)
    foothold[2] = -0.0255

    return foothold


def create_visual_swing_plan(footpos_init, footpos_final, phase_start=0.0):
    footpos_init = np.asarray(footpos_init, dtype=np.float32).copy()
    footpos_final = np.asarray(footpos_final, dtype=np.float32).copy()
    return {
        "initial": footpos_init,
        "final": footpos_final,
        "phase_start": float(np.clip(phase_start, 0.0, 0.999)),
        "curve": SwingFootTrajectoryGenerator.create_swing_trajectory(
            footpos_init, footpos_final, 1.0
        ),
    }


def evaluate_visual_swing_plan(swing_plan, swing_phase):
    phase_start = swing_plan["phase_start"]
    curve_phase = (float(swing_phase) - phase_start) / (1.0 - phase_start)
    curve_phase = np.clip(curve_phase, 0.0, 1.0)
    return np.squeeze(swing_plan["curve"].value(curve_phase)).astype(np.float32)


def get_swing_windows_over_horizon(
    gait,
    iter_counter,
    iterations_between_mpc,
    horizon_time,
    leg_idx,
):
    period_iterations = iterations_between_mpc * gait.num_segment
    period_time = period_iterations * LinearMpcConfig.dt_control
    current_period_time = (
        iter_counter % period_iterations
    ) * LinearMpcConfig.dt_control

    swing_offset = (
        gait.stance_offsets_normalized[leg_idx]
        + gait.stance_durations_normalized[leg_idx]
    ) % 1.0
    swing_duration = 1.0 - gait.stance_durations_normalized[leg_idx]
    swing_duration_time = swing_duration * period_time
    swing_start_in_period = swing_offset * period_time

    num_periods = int(np.ceil((horizon_time + period_time) / period_time)) + 2
    windows = []
    for period_idx in range(-1, num_periods):
        start_time = (
            swing_start_in_period
            + period_idx * period_time
            - current_period_time
        )
        end_time = start_time + swing_duration_time
        if end_time <= 0.0 or start_time >= horizon_time:
            continue

        clipped_start = max(start_time, 0.0)
        clipped_end = min(end_time, horizon_time)
        if clipped_end <= clipped_start:
            continue

        phase_start = (clipped_start - start_time) / swing_duration_time
        phase_end = (clipped_end - start_time) / swing_duration_time
        windows.append(
            {
                "swing_start_time": start_time,
                "start_time": clipped_start,
                "end_time": clipped_end,
                "phase_start": phase_start,
                "phase_end": phase_end,
            }
        )

    return windows


def sample_horizon_foot_trajectories(
    robot_data,
    swing_foot_trajs,
    gait,
    iter_counter,
    iterations_between_mpc,
    horizon,
    horizon_dt,
    base_vel_base_des,
    yaw_turn_rate_des,
    num_samples,
):
    horizon_time = horizon * horizon_dt
    num_samples = max(2, int(num_samples))
    samples_per_second = num_samples / max(horizon_time, 1e-6)
    min_samples_per_swing = 24

    trajectory_segments = [[] for _ in range(4)]
    footholds = [[] for _ in range(4)]
    stance_positions = [
        np.asarray(foot_position, dtype=np.float32).copy()
        for foot_position in robot_data.pos_feet
    ]
    current_swing_states = get_swing_state_at_time(
        gait, iter_counter, iterations_between_mpc, 0.0
    )

    for leg_idx in range(4):
        current_plan = None
        if current_swing_states[leg_idx] > 0.0:
            current_plan = swing_foot_trajs[leg_idx].get_current_swing_plan(gait)

        for window in get_swing_windows_over_horizon(
            gait,
            iter_counter,
            iterations_between_mpc,
            horizon_time,
            leg_idx,
        ):
            is_current_swing = window["swing_start_time"] < 0.0
            if is_current_swing and current_plan is not None:
                swing_plan = create_visual_swing_plan(
                    robot_data.pos_feet[leg_idx],
                    current_plan["final"],
                    current_swing_states[leg_idx],
                )
                footholds[leg_idx].append(current_plan["final"])
            else:
                swing_start_time = max(0.0, window["swing_start_time"])
                footpos_final = predict_foothold(
                    robot_data,
                    leg_idx,
                    gait,
                    base_vel_base_des,
                    yaw_turn_rate_des,
                    swing_start_time,
                )
                swing_plan = create_visual_swing_plan(
                    stance_positions[leg_idx],
                    footpos_final,
                )
                footholds[leg_idx].append(footpos_final)

            segment_duration = window["end_time"] - window["start_time"]
            segment_samples = max(
                2,
                min_samples_per_swing,
                int(np.ceil(segment_duration * samples_per_second)) + 1,
            )
            sample_phases = np.linspace(
                window["phase_start"],
                window["phase_end"],
                segment_samples,
            )
            segment_points = np.asarray(
                [
                    evaluate_visual_swing_plan(swing_plan, float(swing_phase))
                    for swing_phase in sample_phases
                ],
                dtype=np.float32,
            )
            if len(segment_points) > 0:
                segment_points[0] = np.asarray(
                    swing_plan["initial"], dtype=np.float32
                )
                trajectory_segments[leg_idx].append(segment_points)

            stance_positions[leg_idx] = swing_plan["final"].copy()

    return trajectory_segments, footholds


def update_viewer_foot_trajectories(
    viewer,
    robot_data,
    swing_foot_trajs,
    gait,
    iter_counter,
    iterations_between_mpc,
    horizon,
    horizon_dt,
    base_vel_base_des,
    yaw_turn_rate_des,
    swing_states,
    contact_forces,
    num_samples,
):
    if viewer is None:
        return

    viewer.user_scn.ngeom = 0
    draw_current_foot_markers(viewer, robot_data, swing_states)
    draw_support_polygon(viewer, robot_data, swing_states)
    draw_contact_forces(viewer, robot_data, swing_states, contact_forces)

    foot_trajectory_segments, planned_footholds = sample_horizon_foot_trajectories(
        robot_data,
        swing_foot_trajs,
        gait,
        iter_counter,
        iterations_between_mpc,
        horizon,
        horizon_dt,
        base_vel_base_des,
        yaw_turn_rate_des,
        num_samples,
    )

    for leg_idx, trajectory_segments in enumerate(foot_trajectory_segments):
        if len(trajectory_segments) == 0:
            continue

        rgba = LEG_TRAJECTORY_COLORS[leg_idx]
        for points in trajectory_segments:
            draw_polyline(viewer, points, rgba, width=10.0)
        for planned_foothold in planned_footholds[leg_idx]:
            draw_foothold_marker(viewer, planned_foothold)
