import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection

# ---------- Functions ----------
def plot_landmarks_3d(detection_result, ax):
    ax.cla()  # clear previous frame

    pose_landmarks_list = detection_result.pose_world_landmarks
    all_x, all_y, all_z = [], [], []

    for pose_landmarks in pose_landmarks_list:
        xs = [lm.x for lm in pose_landmarks]
        ys = [lm.y for lm in pose_landmarks]
        zs = [lm.z for lm in pose_landmarks]

        all_x.extend(xs)
        all_y.extend(ys)
        all_z.extend(zs)

        ax.scatter(xs, ys, zs, c="r", marker="o")

        for connection in solutions.pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            x_vals = [pose_landmarks[start_idx].x, pose_landmarks[end_idx].x]
            y_vals = [pose_landmarks[start_idx].y, pose_landmarks[end_idx].y]
            z_vals = [pose_landmarks[start_idx].z, pose_landmarks[end_idx].z]
            ax.plot(x_vals, y_vals, z_vals, c="b")

    # Set equal scaling
    max_range = max(
        max(all_x) - min(all_x),
        max(all_y) - min(all_y),
        max(all_z) - min(all_z)
    ) / 2.0
    mid_x = (max(all_x) + min(all_x)) * 0.5
    mid_y = (max(all_y) + min(all_y)) * 0.5
    mid_z = (max(all_z) + min(all_z)) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel("X (left-right)")
    ax.set_ylabel("Y (height)")
    ax.set_zlabel("Z (depth)")
    ax.view_init(elev=-55, azim=-165, roll=75)
    plt.draw()
    plt.pause(0.001)  # small pause to update the plot

# ---------- Initialize MediaPipe PoseLandmarker ----------
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions

base_options = python.BaseOptions(model_asset_path='./pose_landmarker_lite.task')
options = PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=mp.tasks.vision.RunningMode.VIDEO
)
detector = PoseLandmarker.create_from_options(options)

# ---------- Open Video ----------
video_path = "lance_wave.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_index = 0

# ---------- Prepare 3D plot ----------
plt.ion()  # interactive mode
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

# ---------- Process Video ----------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Pose detection for video
    timestamp_ms = int((frame_index / fps) * 1000)
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)

    # Update 3D animation
    plot_landmarks_3d(detection_result, ax)

    frame_index += 1

cap.release()
plt.ioff()
plt.show()
