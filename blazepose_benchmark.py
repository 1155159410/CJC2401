"""
This Python file is structured to mimic a Jupyter notebook. In the PyCharm IDE, 
the code following each '# %%' label is treated as a separate code cell.
"""

# %%
# @formatter:off
# Environment: Apple M1 Max, macOS 15.0, Python 3.10
!pip install ipython
!pip install matplotlib
!pip install mediapipe
!pip install opencv-python
!pip install tensorflow
!pip install tensorflow-metal
# @formatter:on

# %%
import os
import resource
import time

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt

# %%
LANDMARK_DICT = {
    'nose': 0,
    'left_eye_inner': 1,
    'left_eye': 2,
    'left_eye_outer': 3,
    'right_eye_inner': 4,
    'right_eye': 5,
    'right_eye_outer': 6,
    'left_ear': 7,
    'right_ear': 8,
    'mouth_left': 9,
    'mouth_right': 10,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_pinky': 17,
    'right_pinky': 18,
    'left_index': 19,
    'right_index': 20,
    'left_thumb': 21,
    'right_thumb': 22,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
    'left_heel': 29,
    'right_heel': 30,
    'left_foot_index': 31,
    'right_foot_index': 32
}

COCO_KEYPOINTS = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle'
]

# Maps bones to a matplotlib color name
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

color_map = {
    'c': (0, 191, 191),
    'm': (191, 0, 191),
    'y': (191, 191, 0)
}

KEYPOINT_DICT = {body_part: LANDMARK_DICT[body_part] for body_part in COCO_KEYPOINTS}
KEYPOINT_EDGE_INDS_TO_COLOR = {tuple(KEYPOINT_DICT[COCO_KEYPOINTS[i]] for i in k): color_map[v]
                               for k, v in KEYPOINT_EDGE_INDS_TO_COLOR.items()}

# %%
print(KEYPOINT_EDGE_INDS_TO_COLOR)

# %%
def save_image_with_prediction(idx,
                               raw_image,
                               keypoints_with_scores,
                               keypoint_threshold=0.11):
    raw_image = np.array(raw_image)
    keypoints_with_scores = keypoints_with_scores.copy()

    raw_height, raw_width, _ = raw_image.shape
    longest_side = max(raw_height, raw_width)

    """
    # Convert relative coordinates to actual coordinates
    keypoints_with_scores[..., :2] *= longest_side

    # Offset the coordinates based on the aspect ratio
    if raw_height > raw_width:
        keypoints_with_scores[..., 1] -= (longest_side - raw_width) // 2
    elif raw_height < raw_width:
        keypoints_with_scores[..., 0] -= (longest_side - raw_height) // 2
    """

    # Retrieve values from the output
    kpts_x = keypoints_with_scores[0, 0, :, 1].astype(int)
    kpts_y = keypoints_with_scores[0, 0, :, 0].astype(int)
    kpts_scores = keypoints_with_scores[0, 0, :, 2]

    # Pair up keypoints to form edges
    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
        if (kpts_scores[edge_pair[0]] > keypoint_threshold and
                kpts_scores[edge_pair[1]] > keypoint_threshold):
            x_start = kpts_x[edge_pair[0]]
            y_start = kpts_y[edge_pair[0]]
            x_end = kpts_x[edge_pair[1]]
            y_end = kpts_y[edge_pair[1]]

            cv2.line(raw_image, [x_start, y_start], [x_end, y_end], color, thickness=max(longest_side // 300, 1))

    # Plot the keypoints
    for i, coord in enumerate(zip(kpts_x, kpts_y)):
        if i in KEYPOINT_DICT.values() and kpts_scores[i] > keypoint_threshold:
            cv2.circle(raw_image, coord, radius=max(longest_side // 150, 2), color=(255, 20, 147), thickness=-1)

    # Convert RGB to BGR
    bgr_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)

    # Save the image
    cv2.imwrite(f"./output/{idx:08d}.png", bgr_image)

    return f"./output/{idx:08d}.png"

# %%
dataset_root_dir = "./dataset"
raw_images: list[np.ndarray] = []  # Each entry is a 3-channel RGB images [0, 255]
for dirpath, dirnames, filenames in os.walk(dataset_root_dir):
    dirnames.sort()
    filenames.sort()

    for filename in filenames:
        filepath = os.path.join(dirpath, filename)
        file_extension = os.path.splitext(filepath)[1].lower()

        image = tf.io.read_file(filepath)
        if file_extension in ('.jpg', '.jpeg'):
            image = tf.image.decode_jpeg(image)
        elif file_extension == '.png':
            image = tf.image.decode_png(image)
        else:
            continue

        # Ensure image is 3-channel
        image = image[..., :3]
        raw_images.append(image.numpy())

# %%
def run_inference(model, image: np.ndarray) -> np.ndarray:
    # Perform pose estimation on the image
    results = model.process(image)

    keypoints_with_scores = np.zeros((1, 1, 33, 3))

    # Check if any landmarks were detected
    if results.pose_landmarks:
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            # Each landmark has x, y, z coordinates (normalized)
            x = landmark.x * image.shape[1]  # Scale x to image width
            y = landmark.y * image.shape[0]  # Scale y to image height
            z = landmark.z  # z is already in relative depth
            confidence = landmark.visibility

            keypoints_with_scores[0, 0, i] = y, x, confidence

    return keypoints_with_scores

# %%
mp_pose = mp.solutions.pose
with mp_pose.Pose(static_image_mode=True,
                  min_detection_confidence=0.5,
                  model_complexity=1) as pose:
    start_time = time.time()
    results = [run_inference(pose, raw_image) for raw_image in raw_images]
    end_time = time.time()

print("Total time spent:", end_time - start_time)

# %%
memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes
print(f"Memory usage: {memory_usage / 1024 ** 3:.2f} GB")

# %%
def show_and_save(image_idx):
    output_path = save_image_with_prediction(image_idx,
                                             raw_images[image_idx],
                                             results[image_idx],
                                             keypoint_threshold=0)

    image = Image.open(output_path)

    # Display the resultant image using Matplotlib
    plt.imshow(image)
    plt.axis('off')  # Hide the axis
    # plt.show()

# %%
os.makedirs("./output", exist_ok=True)
for i in range(len(raw_images)):
    show_and_save(i)
plt.close('all')
