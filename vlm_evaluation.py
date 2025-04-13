# %% Import necessary modules
import os
import pickle
import string
import time
from collections import Counter

import mediapipe as mp
import numpy as np
import ollama
import torch
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
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


# %% Eval the whole test set and store the results
results = []
for item in tqdm(metadata_list):
    filepath, flipped = item.values()
    result = query_vlm(filepath, flipped)
    results.append(result)
with open("vlm_test_results.pkl", 'wb') as f:
    pickle.dump(results, f)

# %% Read the stored results
with open("vlm_test_results.pkl", 'rb') as f:
    results = pickle.load(f)
with open("test_dataset.pkl", 'rb') as f:
    metadata_list: list[dict] = pickle.load(f)

# %% Generate possible answers
possible_ans = []
for validity in ('true', 'false'):
    for posture_name in ('downdog', 'plank', 'side_plank', 'warrior_ii'):
        possible_ans.append(f'{posture_name};{validity}')
for validity in ('true', 'false'):
    possible_ans.append(f'none;{validity}')

# %% Extract LLM responses
ans_count = Counter()
problematic_responses = []

true_indexes = []
pred_indexes = []

for i, result in enumerate(results):
    # Get the truth label of the sample
    filepath = metadata_list[i]['filepath']
    filepath = filepath.replace('positive', 'true').replace('negative', 'false')
    truth = ';'.join(filepath.split('/')[2:4])
    true_indexes.append(possible_ans.index(truth))

    # Preprocess response_msg
    response_msg: str = result.message.content
    response_msg = response_msg.lower().replace('; ', ';')
    # Remove unnecessary punctuations
    for c in string.punctuation:
        if c not in ('_', ';'):
            response_msg = response_msg.replace(c, '')

    # Split response_msg into words to look for keywords
    msg_words = set(response_msg.split())
    ans_existence = Counter(ans for ans in possible_ans if ans in msg_words)

    # Only responses with exactly 1 label is valid
    if ans_existence.total() == 1:
        ans_count += ans_existence
        ans: str = ans_existence.most_common()[0][0]
        pred_indexes.append(possible_ans.index(ans))
    else:
        problematic_responses.append(response_msg)
        pred_indexes.append(-1)

print("Valid responses:", ans_count.total())
print("Problematic responses:", len(problematic_responses))

# %% Plot the confusion matrix
class_names: list[str] = []
for validity in ('Correct', 'Incorrect'):
    for posture_name in ('Down Dog', 'Plank', 'Side Plank', 'Warrior II'):
        class_names.append(f"{posture_name} ({validity})")
class_names.append('Unexpected Response')

# Group unexpected responses
pred_indexes = [idx if 0 <= idx < 8 else 8 for idx in pred_indexes]

# Generate the full confusion matrix (9x9)
conf_matrix = confusion_matrix(true_indexes, pred_indexes)

# Crop the last row (true label: 'Unexpected Response')
conf_matrix = conf_matrix[:-1, :]  # Now shape is 8×9

# Plot the confusion matrix
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# Avoid using ConfusionMatrixDisplay's automatic label handling
im = ax.imshow(conf_matrix, cmap=plt.cm.Blues)

# Set x-axis (predicted) labels
ax.set_xticks(np.arange(len(class_names)))
ax.set_xticklabels(class_names, rotation=45, ha='right')

# Set y-axis (true) labels — only first 8
ax.set_yticks(np.arange(len(class_names) - 1))
ax.set_yticklabels(class_names[:-1])

# Compute font color (dark blue)
r, g, b = 20, 47, 103
font_color = (r / 255, g / 255, b / 255)

# Add counts inside cells
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        value = conf_matrix[i, j]
        ax.text(j, i, format(value, 'd'),
                ha='center', va='center',
                color='white' if value > conf_matrix.max() / 2 else font_color)

plt.title("Confusion Matrix on Test Set (w/o Normalization)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.colorbar(im, ax=ax)
plt.show()

# %% Plot the normalized confusion matrix
class_names: list[str] = []
for validity in ('Correct', 'Incorrect'):
    for posture_name in ('Down Dog', 'Plank', 'Side Plank', 'Warrior II'):
        class_names.append(f"{posture_name} ({validity})")
class_names.append('Unexpected Response')

# Group unexpected responses
pred_indexes = [idx if 0 <= idx < 8 else 8 for idx in pred_indexes]

# Generate the full confusion matrix (9x9)
conf_matrix = confusion_matrix(true_indexes, pred_indexes)

# Normalize the confusion matrix row-wise
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True).clip(min=1)

# Crop the last row (true label: 'Unexpected Response')
conf_matrix_normalized = conf_matrix_normalized[:-1, :]  # Now shape is 8×9

# Plot the confusion matrix
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# Avoid using ConfusionMatrixDisplay's automatic label handling
im = ax.imshow(conf_matrix_normalized, cmap=plt.cm.Blues)

# Set x-axis (predicted) labels
ax.set_xticks(np.arange(len(class_names)))
ax.set_xticklabels(class_names, rotation=45, ha='right')

# Set y-axis (true) labels — only first 8
ax.set_yticks(np.arange(len(class_names) - 1))
ax.set_yticklabels(class_names[:-1])

# Compute font color (dark blue)
r, g, b = 20, 47, 103
font_color = (r / 255, g / 255, b / 255)

# Add counts inside cells
for i in range(conf_matrix_normalized.shape[0]):
    for j in range(conf_matrix_normalized.shape[1]):
        value = conf_matrix_normalized[i, j]
        ax.text(j, i, format(value, '.2f'),
                ha='center', va='center',
                color='white' if value > conf_matrix_normalized.max() / 2 else font_color)

plt.title("Confusion Matrix on Test Set (w/ Normalization)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.colorbar(im, ax=ax)
plt.show()

# %% Plot time usage graph
time_records = [result['total_duration'] / 10 ** 9 for result in results]
plt.hist(time_records, bins=10_000, edgecolor='black')
plt.xlim([30, 80])  # Ignore outliers
plt.xlabel('Duration (seconds)')
plt.ylabel('Frequency')
plt.title('Histogram of Duration')
plt.show()
