# flake8: noqa

import numpy as np
import matplotlib.pyplot as plt

# Constants
LANDMARKS = np.array([[2, 3], [4, 5], [6, 1]])
INITIAL_POSITION = np.array([0, 0, 0])  # x, y, theta
NUMBER_OF_STEPS = 50
MOTION_NOISE = 0.1
SENSOR_NOISE = 0.1


# Functions
def generate_motion_command():
    return np.random.normal(1.0, 0.3), np.random.normal(np.pi / 18, np.pi / 36)


def motion_model(position, motion_command, dt):
    x, y, theta = position
    v, w = motion_command
    next_position = position + dt * np.array([v * np.cos(theta), v * np.sin(theta), w])
    next_position[2] = (next_position[2] + np.pi) % (2 * np.pi) - np.pi
    return next_position


def sense_landmarks(position):
    x, y, theta = position
    landmarks = []
    for l_x, l_y in LANDMARKS:
        distance = np.sqrt((l_x - x) ** 2 + (l_y - y) ** 2) + np.random.normal(
            0, SENSOR_NOISE
        )
        bearing = (
            np.arctan2(l_y - y, l_x - x) - theta + np.random.normal(0, SENSOR_NOISE)
        )
        landmarks.append([distance, bearing])
    return np.array(landmarks)


def ekf_slam(pose, pose_cov, motion_command, observations):
    n_landmarks = (pose_cov.shape[0] - 3) // 2
    dt = 1

    # Prediction step
    pose = motion_model(pose, motion_command, dt)
    J_pose = np.array(
        [
            [1, 0, -dt * motion_command[0] * np.sin(pose[2])],
            [0, 1, dt * motion_command[0] * np.cos(pose[2])],
            [0, 0, 1],
        ]
    )
    pose_cov[:3, :3] = J_pose @ pose_cov[:3, :3] @ J_pose.T + np.diag(
        [MOTION_NOISE**2, MOTION_NOISE**2, (MOTION_NOISE * 10) ** 2]
    )

    # Update step
    for i, observation in enumerate(observations):
        distance, bearing = observation

        if i >= n_landmarks:
            pose = np.hstack(
                (
                    pose,
                    [
                        pose[0] + distance * np.cos(bearing + pose[2]),
                        pose[1] + distance * np.sin(bearing + pose[2]),
                    ],
                )
            )
            pose_cov = np.vstack((pose_cov, np.zeros((2, pose_cov.shape[1]))))
            pose_cov = np.hstack((pose_cov, np.zeros((pose_cov.shape[0], 2))))
            pose_cov[-2, -2] = SENSOR_NOISE**2
            pose_cov[-1, -1] = SENSOR_NOISE**2
            n_landmarks += 1
        else:
            landmark_x, landmark_y = pose[3 + 2 * i], pose[4 + 2 * i]

            # Expected observation
            expected_distance = np.sqrt(
                (landmark_x - pose[0]) ** 2 + (landmark_y - pose[1]) ** 2
            )
            expected_bearing = (
                np.arctan2(landmark_y - pose[1], landmark_x - pose[0]) - pose[2]
            )

            # Calculate Jacobian
            dx, dy = landmark_x - pose[0], landmark_y - pose[1]
            q = dx**2 + dy**2
            sqrt_q = np.sqrt(q)
            H = (
                np.array(
                    [
                        [-sqrt_q * dx, -sqrt_q * dy, 0, sqrt_q * dx, sqrt_q * dy],
                        [dy, -dx, -q, -dy, dx],
                    ]
                )
                / q
            )

            # Kalman gain
            S = H @ pose_cov[:, 3 + 2 * i : 5 + 2 * i] @ H.T + np.diag(
                [SENSOR_NOISE**2, SENSOR_NOISE**2]
            )
            K = pose_cov[:, 3 + 2 * i : 5 + 2 * i] @ H.T @ np.linalg.inv(S)

            # Update state and covariance
            innovation = np.array(
                [distance - expected_distance, bearing - expected_bearing]
            )
            innovation[1] = (innovation[1] + np.pi) % (2 * np.pi) - np.pi
            pose[3 + 2 * i : 5 + 2 * i] += K @ innovation
            pose_cov[:, 3 + 2 * i : 5 + 2 * i] -= (
                K @ H @ pose_cov[:, 3 + 2 * i : 5 + 2 * i]
            )

    return pose, pose_cov


# Main loop
pose = INITIAL_POSITION
pose_cov = np.zeros((3, 3))
path = [pose[:2]]

for _ in range(NUMBER_OF_STEPS):
    motion_command = generate_motion_command()
    observations = sense_landmarks(pose)
    pose, pose_cov = ekf_slam(pose, pose_cov, motion_command, observations)
    path.append(pose[:2])

# Plotting
path = np.array(path)
plt.scatter(LANDMARKS[:, 0], LANDMARKS[:, 1], marker="*", c="r", s=200)
plt.scatter(pose[3::2], pose[4::2], c="g", marker="o")
plt.plot(path[:, 0], path[:, 1], c="b")
plt.xlabel("x")
plt.ylabel("y")
plt.title("2D SLAM using EKF")
plt.legend(["Path", "True Landmarks", "Estimated Landmarks"])
plt.grid(True)
plt.show()
