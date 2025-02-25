# %% Import necessary modules
import queue
import threading
import time
from datetime import datetime, timedelta
from typing import TypedDict

import cv2
import mediapipe as mp
import numpy as np
import torch
from numpy import ndarray
from sklearn.preprocessing import MinMaxScaler
from torch.nn import functional as F


# %% Frame Type Hints Class
class FrameInfo(TypedDict):
    rgb_frame: np.ndarray
    timestamp: float
    keypoints: np.ndarray
    posture_prob: np.ndarray
    correctness_prob: np.ndarray
    predicted_posture: str
    predicted_feedback: str


# %% Camera Class
class Camera:
    def __init__(self):
        self._cap = None
        self._thread = None  # Thread for capturing frames
        self._running = False  # Flag to control thread execution

    def start(self, buffer: queue.Queue[dict]) -> None:
        """
        Starts the camera thread and begins capturing frames.

        :param buffer: A queue where captured frames and timestamps will be placed.
        If the queue is full, new frames will be dropped.
        """

        def thread_func():
            while self._running:  # Continuously capture frames
                ret, frame = self._cap.read()  # Read a frame
                if not ret:  # If frame read fails
                    print("Failed to get frame")
                    break

                item = {
                    'rgb_frame': cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),  # Convert to RGB
                    'timestamp': time.time(),
                }

                try:
                    buffer.put(item, block=False)  # Add frame to buffer
                except queue.Full:
                    continue  # Drop the frame if the buffer is full

        if self._running:
            return  # Prevent multiple starts

        self._cap = cv2.VideoCapture(0)  # Open camera
        if not self._cap.isOpened():  # Check if camera is opened successfully
            raise RuntimeError("Camera open failed")

        self._running = True  # Start thread
        self._thread = threading.Thread(target=thread_func)
        self._thread.start()

    def stop(self):
        """
        Stops the camera thread and releases any resources.
        """
        self._running = False  # Stop thread

        if self._thread is not None:
            self._thread.join()  # Wait for thread to finish
            self._thread = None

        if self._cap is not None:
            self._cap.release()  # Release camera resources
            self._cap = None


# %% Posture Correction System Class
class PostureCorrectionSystem:
    POSTURE_NAMES = ['Down Dog', 'Plank', 'Side Plank', 'Warrior II']
    FEEDBACKS = ['Incorrect', 'Correct']

    def __init__(self):
        """
        Initializes the posture correction system by loading the necessary models and setting the device.
        """
        checkpoint_path = "./checkpoints/27.pth"

        # Set the device based on availability
        self.device = torch.device(
            'mps' if torch.backends.mps.is_available()
            else 'cuda' if torch.cuda.is_available()
            else 'cpu'
        )

        # Load the BlazePose model for keypoint detection
        self.blazepose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)

        # Load the pre-trained neural network model for posture classification
        self.model = torch.load(checkpoint_path, map_location=self.device)['model'].eval()

        self._thread = None
        self._running = False

    def __del__(self):
        self.blazepose.close()

    # noinspection PyUnresolvedReferences
    def first_stage(self, image_np: ndarray) -> list[list[float]]:
        """
        Detects and extracts keypoints from an image using BlazePose.
        Returns a list of keypoints (x, y, z, visibility) for each landmark.
        """
        keypoints = []
        result = self.blazepose.process(image_np)

        if result.pose_landmarks:
            for landmark in result.pose_landmarks.landmark:
                keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])

        return keypoints

    def second_stage(self, keypoints: ndarray) -> tuple[ndarray, ndarray]:
        """
        Passes the normalized keypoints through the neural network to classify the posture and correctness.
        Returns the probabilities for posture and correctness.
        """
        inputs = torch.tensor([keypoints], dtype=torch.float32).to(self.device)

        # Forward pass through the model
        with torch.no_grad():
            output: torch.Tensor = self.model(inputs)[0]

        # Split the output into posture and correctness logits
        posture_logits, correctness_logits = output.T

        # Apply softmax to posture logits and sigmoid to correctness logits
        posture_prob = F.softmax(posture_logits, dim=0)
        correctness_prob = torch.sigmoid(correctness_logits)

        return posture_prob.cpu().numpy(), correctness_prob.cpu().numpy()

    def process_image(self, image_np: ndarray, verbose: bool = False) -> dict:
        """
        Full pipeline to process an image:
        1. Extract keypoints using BlazePose.
        2. Normalize the keypoints.
        3. Classify posture and correctness.
        4. Print the results.
        """
        keypoints: list[list[float]] = self.first_stage(image_np)

        if not keypoints:
            print("No BlazePose landmarks detected in the image.")
            return {}

        # Normalize keypoints
        keypoints_np: np.ndarray = np.array(keypoints)
        normalized_keypoints = self.normalize_keypoints(keypoints_np)

        # Classify posture and correctness
        posture_prob, correctness_prob = self.second_stage(normalized_keypoints)

        predicted_posture_idx: int = int(posture_prob.argmax())
        predicted_feedback_idx: int = int(correctness_prob[predicted_posture_idx].round())

        predicted_posture: str = self.POSTURE_NAMES[predicted_posture_idx]
        predicted_feedback: str = self.FEEDBACKS[predicted_feedback_idx]

        # Print results
        if verbose:
            print(f"Posture classification probabilities: {posture_prob.tolist()}")
            print(f"Correctness probabilities: {correctness_prob.tolist()}")
            print(f"Predicted posture: {predicted_posture}")
            print(f"Predicted feedback: {predicted_feedback}")

        result = {
            'keypoints': keypoints_np,  # np.ndarray
            'posture_prob': posture_prob,  # np.ndarray
            'correctness_prob': correctness_prob,  # np.ndarray
            'predicted_posture': predicted_posture,  # str
            'predicted_feedback': predicted_feedback,  # str
        }

        return result

    def start_thread(self, in_queue: queue.Queue, out_queue: queue.Queue) -> None:
        def thread_func():
            while self._running:
                try:
                    item = in_queue.get(block=False)  # Get next item from input queue
                except queue.Empty:
                    continue

                rgb_frame = item['rgb_frame']  # Extract RGB image frame
                result = self.process_image(rgb_frame)  # Process the image
                item |= result

                out_queue.put(item)  # Send result to output queue

        if self._running:
            return  # Prevent multiple starts

        self._running = True
        self._thread = threading.Thread(target=thread_func)
        self._thread.start()

    def stop_thread(self):
        self._running = False

        if self._thread is not None:
            self._thread.join()
            self._thread = None

    @staticmethod
    def normalize_keypoints(keypoints: ndarray) -> ndarray:
        """
        Normalizes the keypoints so that x, y are scaled to [0, 1] and z is normalized to a unit vector.
        """
        keypoints = keypoints.copy()
        x, y, z, visibility = keypoints.T

        # Normalize x and y to the range [0, 1]
        scaler = MinMaxScaler()
        x[:] = scaler.fit_transform(x.reshape(-1, 1)).ravel()
        y[:] = scaler.fit_transform(y.reshape(-1, 1)).ravel()

        # Normalize z to a unit vector
        z /= np.linalg.norm(z)

        return keypoints


# %% Frame Processor Class
class FrameProcessor:
    KEYPOINT_EDGE_IDX_TO_COLOR = {(0, 2): (191, 0, 191), (0, 5): (0, 191, 191), (2, 7): (191, 0, 191),
                                  (5, 8): (0, 191, 191), (0, 11): (191, 0, 191), (0, 12): (0, 191, 191),
                                  (11, 13): (191, 0, 191), (13, 15): (191, 0, 191), (12, 14): (0, 191, 191),
                                  (14, 16): (0, 191, 191), (11, 12): (191, 191, 0), (11, 23): (191, 0, 191),
                                  (12, 24): (0, 191, 191), (23, 24): (191, 191, 0), (23, 25): (191, 0, 191),
                                  (25, 27): (191, 0, 191), (24, 26): (0, 191, 191), (26, 28): (0, 191, 191)}

    def __init__(self, rgb_frame) -> None:
        self.rgb_frame = rgb_frame.copy()
        self.frame_h, self.frame_w, _ = self.rgb_frame.shape

    @property
    def window_h(self):
        return self.rgb_frame.shape[0]

    @property
    def window_w(self):
        return self.rgb_frame.shape[1]

    @property
    def bgr_frame(self):
        return cv2.cvtColor(self.rgb_frame, cv2.COLOR_RGB2BGR)

    def draw_skeletons(self, keypoints: np.ndarray) -> None:
        longest_side = max(self.frame_h, self.frame_w)

        # Retrieve values from the output
        kpts_x = (keypoints[:, 0] * self.frame_w).astype(int)
        kpts_y = (keypoints[:, 1] * self.frame_h).astype(int)
        kpts_scores = keypoints[:, 3]

        # Pair up keypoints to form edges
        for edge_pair, color in self.KEYPOINT_EDGE_IDX_TO_COLOR.items():
            edge_pair = list(edge_pair)

            x_start, x_end = kpts_x[edge_pair]
            y_start, y_end = kpts_y[edge_pair]

            cv2.line(
                self.rgb_frame,
                [x_start, y_start],
                [x_end, y_end],
                color,
                thickness=max(longest_side // 300, 1)
            )

        # Plot the keypoints
        for i in set(sum(self.KEYPOINT_EDGE_IDX_TO_COLOR.keys(), ())):
            coord = kpts_x[i], kpts_y[i]
            cv2.circle(
                self.rgb_frame,
                coord,
                radius=max(longest_side // 150, 2),
                color=(255, 20, 147),
                thickness=-1
            )

    def put_fps(self, fps: int):
        text = f"FPS: {fps}"

        # Font, scale, thickness, and color
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.
        thickness = 3
        color = (180, 180, 180)  # Gray

        # Get text size to align it properly
        (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
        x = self.frame_w - text_width - 10  # Right-align with padding
        y = 10 + text_height  # Position near the top

        # Draw the text on the frame
        cv2.putText(self.rgb_frame, text, (x, y), font, scale, (0, 0, 0), thickness * 3)
        cv2.putText(self.rgb_frame, text, (x, y), font, scale, color, thickness)

    def extend_frame(self, ratio: float):
        extend_height = int(self.frame_h * ratio)
        self.rgb_frame = np.pad(
            self.rgb_frame,
            ((0, extend_height), (0, 0), (0, 0)),
            'constant',
            constant_values=0,
        )

    def put_text(self, text: str, color: tuple, left: bool = True):
        usable_height = self.window_h - self.frame_h
        if usable_height == 0:
            self.extend_frame(0.1)
            usable_height = self.window_h - self.frame_h

        margin = int(usable_height * 0.2)  # Margin from the edges
        max_text_height = usable_height - margin * 2  # Space after applying margins

        # Start with an arbitrary large scale and calculate its height
        font = cv2.FONT_HERSHEY_SIMPLEX
        base_font_scale = 1.
        base_thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, base_font_scale, base_thickness)

        # Compute the maximum scale that fits within the desire height
        font_scale = max_text_height / (text_height + baseline)
        font_scale = max(font_scale, 0.1)  # Ensure the font scale is not too small
        thickness = int(font_scale * 2)

        # Set text position
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x = margin if left else self.frame_w - text_width - margin
        y = self.frame_h + margin + text_height

        # Put text on frame
        cv2.putText(self.rgb_frame, text, (x, y), font, font_scale, color, thickness)


# %% Stopwatch Class
class Stopwatch:
    def __init__(self):
        self.current_posture: str = ''
        self.total_time: timedelta = timedelta()
        self.last_update: datetime = datetime.now()

    def update(self, posture: str, correctness: str) -> None:
        if posture != self.current_posture:
            self.current_posture = posture
            self.total_time = timedelta()  # Reset timer
            self.last_update = datetime.now()
        else:
            current_time = datetime.now()
            if correctness == 'Correct':
                self.total_time += current_time - self.last_update
            self.last_update = current_time

    def total_time_str(self) -> str:
        return f"{self.total_time:%M:%S}"
