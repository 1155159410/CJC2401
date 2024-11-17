# %% Import necessary modules
import warnings

# Suppress all Python warnings (global)
warnings.filterwarnings("ignore")

from time import time

import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from numpy import ndarray
from sklearn.preprocessing import MinMaxScaler
from torch.nn import functional as F


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
        self.model = torch.load(checkpoint_path)['model'].to(self.device).eval()

    def __del__(self):
        self.blazepose.close()

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

    def process_image(self, image_np: ndarray):
        """
        Full pipeline to process an image:
        1. Extract keypoints using BlazePose.
        2. Normalize the keypoints.
        3. Classify posture and correctness.
        4. Print the results.
        """
        keypoints = self.first_stage(image_np)

        if not keypoints:
            print("No BlazePose landmarks detected in the image.")
            return

        # Normalize keypoints
        keypoints_np = np.array(keypoints)
        normalized_keypoints = self.normalize_keypoints(keypoints_np)

        # Classify posture and correctness
        posture_prob, correctness_prob = self.second_stage(normalized_keypoints)

        # Display results
        predicted_posture = self.POSTURE_NAMES[posture_prob.argmax()]
        predicted_feedback = self.FEEDBACKS[round(correctness_prob[posture_prob.argmax()])]

        print(f"Posture classification probabilities: {posture_prob.tolist()}")
        print(f"Correctness probabilities: {correctness_prob.tolist()}")
        print(f"Predicted posture: {predicted_posture}")
        print(f"Predicted feedback: {predicted_feedback}")

    @staticmethod
    def normalize_keypoints(keypoints: ndarray) -> ndarray:
        """
        Normalizes the keypoints so that x, y are scaled to [0, 1] and z is normalized to a unit vector.
        """
        x, y, z, visibility = keypoints.T

        # Normalize x and y to the range [0, 1]
        scaler = MinMaxScaler()
        x[:] = scaler.fit_transform(x.reshape(-1, 1)).ravel()
        y[:] = scaler.fit_transform(y.reshape(-1, 1)).ravel()

        # Normalize z to a unit vector
        z /= np.linalg.norm(z)

        return keypoints


# %% Load and process images
image_paths = [
    "./dataset/warrior_ii/positive/0326.png",
    "./dataset/plank/positive/frame_71.png",
    "./dataset/plank/positive/frame_293.png",
    "./dataset/downdog/negative/frame01255.png",
    "./dataset/warrior_ii/negative/frame_00185.png",
    "./dataset/plank/negative/frame_322.png",
    "./dataset/warrior_ii/negative/frame_00107.png",
    "./dataset/downdog/negative/frame_10149.png",
    "./dataset/side_plank/positive/frame_00100.png",
    "./dataset/side_plank/positive/frame_003.png",
    "./dataset/plank/negative/frame_3.png",
    "./dataset/warrior_ii/positive/0249.png",
    "./dataset/warrior_ii/negative/frame_00238.png",
    "./dataset/warrior_ii/negative/frame_00191.png",
    "./dataset/warrior_ii/negative/frame_00256.png",
    "./dataset/warrior_ii/positive/0155.png",
    "./dataset/side_plank/positive/frame_0032.png",
    "./dataset/plank/positive/frame_125.png",
    "./dataset/side_plank/positive/frame_00140.png",
    "./dataset/plank/positive/019.png",
]

# Load images into a list of NumPy arrays
image_list: list[ndarray] = []
for image_path in image_paths:
    image = Image.open(image_path)
    image_rgb = image.convert('RGB')
    image_np = np.array(image_rgb)
    image_list.append(image_np)

# %% Instantiate the Posture Correction System
posture_system = PostureCorrectionSystem()
print(flush=True)

# %% Process each image and display results
start_time = time()
for image_path, image_np in zip(image_paths, image_list):
    print(f"Processing image: '{image_path}'")
    posture_system.process_image(image_np)
    print()

end_time = time()
print(f"Total time to process {len(image_paths)} images: {end_time - start_time:.2f} seconds.")
