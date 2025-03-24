# %% Import necessary modules
import os
import pickle
import time

import mediapipe as mp
import numpy as np
import ollama
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import random_split
from tqdm import tqdm

# %% [PREPROC] Pickle the test dataset identical to `train.py`
postures: list[str] = ['downdog', 'plank', 'side_plank', 'warrior_ii']
dataset_base_path: str = "./dataset"

image_list: list[np.ndarray] = []  # List to store image NumPy arrays
label_list: list[list[int]] = []  # List to store labels: [posture_class, correctness]

for posture_class, posture_name in enumerate(postures):
    for correctness, subfolder_name in enumerate(['negative', 'positive']):
        folder_path: str = os.path.join(dataset_base_path, posture_name, subfolder_name)
        filenames: list[str] = sorted(os.listdir(folder_path))  # Sort in Python for consistency across OSes

        for filename in filenames:
            # Filter out non-image files
            if filename.split('.')[-1].lower() not in ['jpeg', 'jpg', 'png']:
                continue

            filepath: str = os.path.join(folder_path, filename)

            # Open the image
            image = Image.open(filepath)

            # Convert the image to RGB
            image_rgb = image.convert('RGB')

            # Convert the image to a NumPy array of type uint8
            image_np = np.array(image_rgb)

            # Append the NumPy array to the image data list
            image_list.append(image_np)

            metadata: dict = {
                'filepath': filepath,
                'flipped': False,
            }

            # Append the corresponding posture and label to the labels list
            label_list.append([posture_class, correctness, metadata])

for image_np in image_list[:]:
    image_list.append(np.fliplr(image_np))
for posture_class, correctness, metadata in label_list[:]:
    metadata = metadata.copy()
    metadata['flipped'] = True
    label_list.append([posture_class, correctness, metadata])

blazepose_results: list[list[list[float]]] = []  # List to store keypoints for all images

mp_pose = mp.solutions.pose
with mp_pose.Pose(static_image_mode=True,
                  model_complexity=1) as pose:
    for image_np in tqdm(image_list):
        # Perform pose estimation on the current image
        result = pose.process(image_np)

        # If no landmarks are detected, append an empty list for this image
        if not result.pose_landmarks:
            blazepose_results.append([])
            continue

        keypoints: list[list[float]] = []  # List to store keypoints for this image

        # Extract keypoints (x, y, z, visibility) for each detected landmark
        for landmark in result.pose_landmarks.landmark:
            x = landmark.x  # Horizontal axis (0 is the left)
            y = landmark.y  # Vertical axis (0 is the top)
            z = landmark.z
            confidence = landmark.visibility

            keypoints.append([x, y, z, confidence])

        # Append the keypoints for this image to the results list
        blazepose_results.append(keypoints)

for i in reversed(range(len(blazepose_results))):
    if not blazepose_results[i]:
        del blazepose_results[i]
        del image_list[i]
        del label_list[i]


class YogaPoseDataset(Dataset):
    def __init__(self, labels_):
        self.labels: list = labels_

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.labels[idx]


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, seed=2024):
    # Set the random seed for reproducibility
    torch.manual_seed(seed)

    # Calculate the lengths for each subset
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    test_size = dataset_size - train_size - val_size

    # Split the dataset using random_split
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    return train_dataset, val_dataset, test_dataset


dataset = YogaPoseDataset(label_list)
train_dataset, val_dataset, test_dataset = split_dataset(dataset)

assert len(test_dataset) == 1156
assert hash(tuple([tuple(label[:2]) for label in test_dataset])) == 2663417381329733251

metadata_list: list[dict] = [label[2] for label in test_dataset]
with open("test_dataset.pkl", 'wb') as f:
    pickle.dump(metadata_list, f)

# %% Load the test dataset
with open("test_dataset.pkl", 'rb') as f:
    metadata_list: list[dict] = pickle.load(f)

# %% Define the prompt
prompt = """
Identify the yoga pose in the given image and respond with ONE word of the following: [downdog, plank, side_plank, warrior_ii, none]. downdog: The body forms an inverted "V" shape with hands and feet on the ground, hips raised high. Arms and legs are straight, and the head aligns with the arms. Heels may be touching or slightly lifted off the ground. plank: The body is in a straight line from head to heels, supported by hands (or forearms) and toes. The back remains flat, core engaged, and hips should not sag or lift excessively. side_plank: The body is supported on one hand (or forearm) and the outer edge of one foot. The other foot may be stacked or staggered. The body remains in a straight line from head to heels, with the free arm often extended upward. warrior_ii: One leg is bent at the knee at approximately 90 degrees, while the other remains straight and extended behind. The arms are extended parallel to the ground, in line with the legs, and the torso remains upright. The head looks over the front hand. none: The pose does not match any of the above categories.
Besides reporting the posture type, also determine if the posture is correct, as a Boolean value True or False. For downdog, if an image clearly shows the practitioner forming an inverted V shape with hips lifted, legs extended, and arms in a straight line with the spine, then it is True; however, if the back is rounded, the arms and back do not form a straight line, or there is a noticeable bend in the knees, then it is False. For plank, if the posture displays a continuous straight line from the head to the heels with the forearms and upper arms perpendicular to the ground and no deviation in the hip alignment, then it is True; conversely, if the hips are raised too high or dropped too low, the arms are bent or not perpendicular, or the image resembles a variant like Purvottanasana, then it is False. For side_plank, if the image demonstrates a proper transition from a standard plank with the weight shifted onto one hand, the feet stacked, and the body forming a straight line with a support arm that is correctly bent (forming a right angle), then it is True; otherwise, if the feet are misaligned, the hips are off the straight line, or the support arm is straight instead of bent, then it is False. For warrior_ii, if the stance is wide with the front foot pointing forward, the front knee bent exactly at a 90-degree angle with the knee directly over the ankle while the back leg remains straight, and the torso stays upright with the arms extended parallel to the ground, then it is True; but if the torso is shifted, the arms sag or are raised improperly, or the knee angle deviates from 90 degrees, then it is False. Respond strictly in the format: 'POSTURE_TYPE; TRUE/FALSE'. No additional text or explanations.
""".strip()


# %% Define function for eval one image
def query_vlm(image_path: str, flipped: bool = False):
    # Flip the image if needed
    if flipped:
        image = Image.open(image_path)
        flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        tmp_name = str(time.time()).replace('.', '_') + '.png'
        flipped_image.save(tmp_name, format='PNG')
        image_path = tmp_name

    # Query the model
    response = ollama.chat(
        model='llama3.2-vision:11b',
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': [image_path],
        }]
    )

    if flipped:
        os.remove(image_path)

    return response


# %% Eval the whole test set and store the result
results = []
for item in tqdm(metadata_list):
    filepath, flipped = item.values()
    result = query_vlm(filepath, flipped)
    results.append(result)
with open("vlm_test_results.pkl", 'wb') as f:
    pickle.dump(results, f)
