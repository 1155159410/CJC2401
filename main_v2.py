# %% Import necessary modules
import warnings

# Suppress all Python warnings (global)
warnings.filterwarnings("ignore")

import queue
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import torch
from numpy import ndarray
from sklearn.preprocessing import MinMaxScaler
# noinspection PyPep8Naming
from torch.nn import functional as F


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
        self.blazepose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1)

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


# %% TODO
in_queue = queue.Queue(maxsize=30)
out_queue = queue.Queue()

camera = Camera()
posture_system = PostureCorrectionSystem()

camera.start(in_queue)
posture_system.start_thread(in_queue, out_queue)

while cv2.waitKey(1) != ord('q'):  # Up to 1000 loops per second
        item: dict = out_queue.get()

        frame_processor = FrameProcessor(item['rgb_frame'])
        if (keypoints := item.get('keypoints')) is not None:
            frame_processor.draw_skeletons(keypoints)
        rgb_frame = frame_processor.rgb_frame

    print("Delay", time.time() - timestamp)
    print()
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Posture Correction System", bgr_frame)

cv2.destroyAllWindows()
camera.stop()
print("Camera stopped")
posture_system.stop_thread()
print("System stopped")
del posture_system
print("System deleted")
quit()


