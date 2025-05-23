# %% Terminal commands to set up the environment
# Environment: Apple M1 Max, macOS 15.0, Python 3.10
"""
pip install matplotlib
pip install mediapipe
pip install scikit-learn
pip install torch
pip install tqdm
"""

# %% Import necessary modules
import importlib
import os
import pickle
from collections import Counter

import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import random_split
from tqdm import tqdm

import models

# %% [PREPROC] Load dataset images into NumPy arrays
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

            # Append the corresponding posture and label to the labels list
            label_list.append([posture_class, correctness])

# %% [PREPROC] Apply horizontal flip as an augmentation
for image_np in image_list[:]:
    image_list.append(np.fliplr(image_np))
label_list *= 2

# %% [PREPROC] Pass images to BlazePose to extract keypoints
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

# %% [PREPROC] Filter out images that have no detected landmarks
for i in reversed(range(len(blazepose_results))):
    if not blazepose_results[i]:
        del blazepose_results[i]
        del image_list[i]
        del label_list[i]

# %% [PREPROC] Store valid BlazePose results and corresponding labels in a file
with open("blazepose_results.pkl", 'wb') as f:
    pickle.dump(blazepose_results, f)

with open("label_list.pkl", 'wb') as f:
    pickle.dump(label_list, f)

# %% Restore stored results and labels
with open("blazepose_results.pkl", 'rb') as f:
    blazepose_results = pickle.load(f)

with open("label_list.pkl", 'rb') as f:
    label_list = pickle.load(f)

# %% Normalize the BlazePose results: (x, y) using MinMaxScaler and (z) using L2 normalization
blazepose_results_np = np.array(blazepose_results)

for result_np in blazepose_results_np:
    x, y, z, confidence = result_np.T

    # Normalize x and y to the range [0, 1]
    scaler = MinMaxScaler()
    x[:] = scaler.fit_transform(x.reshape(-1, 1)).ravel()  # Normalize x
    y[:] = scaler.fit_transform(y.reshape(-1, 1)).ravel()  # Normalize y

    # Normalize the z column to a unit vector
    z /= np.linalg.norm(z)


# %% Define the Dataset class
class YogaPoseDataset(Dataset):
    def __init__(self, keypoints_np, labels_):
        self.keypoints_tensor = torch.tensor(keypoints_np, dtype=torch.float32)
        self.labels = torch.tensor(labels_)

    def __len__(self):
        return len(self.keypoints_tensor)

    def __getitem__(self, idx):
        keypoints = self.keypoints_tensor[idx]  # Shape (33, 4)
        label = self.labels[idx]  # Shape (2,): [class index, correctness boolean]
        return keypoints, label


# %% Define a function to split the dataset
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


# %% Create the dataset objects
dataset = YogaPoseDataset(blazepose_results_np, label_list)
train_dataset, val_dataset, test_dataset = split_dataset(dataset)

# %% Create a weighted sampler for the training dataset to address imbalance
# Convert labels from tensors to tuples
train_dataset_labels = [tuple(label.tolist()) for _, label in train_dataset]

# Count occurrences of each label
label_counts = Counter(train_dataset_labels)

# Assign inverse frequency as weight
weights = [label_counts[label_tuple] ** -1 for label_tuple in train_dataset_labels]

# Create the weighted sampler
train_sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# %% Create DataLoader for each set
train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# %% Get available acceleration device
device = torch.device('mps' if torch.backends.mps.is_available()
                      else 'cuda' if torch.cuda.is_available() else 'cpu')

# %% Instantiate the model [Configurable]
importlib.reload(models)
model = models.SharedMLP(dropout_p=0.1)
model = model.to(device)
print("No. of params (including bias):", sum(p.numel() for p in model.parameters()))


# %% Define the loss function
def loss_func(output_batch, label_batch):
    # Split the label batch into posture_class and correctness
    posture_class, correctness = label_batch.T

    # Loss for the posture class
    loss1 = nn.CrossEntropyLoss()(output_batch[:, :, 0], posture_class)

    # Loss for the correctness
    logits = output_batch[:, :, 1].gather(dim=1, index=posture_class.unsqueeze(dim=1))
    loss2 = nn.BCEWithLogitsLoss()(logits, correctness.unsqueeze(dim=1).to(torch.float32))

    # Combine both losses
    total_loss = loss1 + loss2

    return total_loss


# %% Define the optimizer and learning rate scheduler [Configurable]
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.00001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30)


# %% Helper function to convert raw model output to predicted labels
def logits_to_labels(output_batch: torch.tensor) -> torch.tensor:
    # Get the predicted posture class
    pred_posture_class = output_batch[:, :, 0].argmax(dim=1, keepdim=True)

    # Check the predicted correctness (gather the correctness logits and threshold at 0)
    pred_correctness = output_batch[:, :, 1].gather(dim=1, index=pred_posture_class) > 0

    # Stack the predicted posture class and correctness horizontally to form the predicted labels
    pred_labels = torch.hstack([pred_posture_class, pred_correctness])

    return pred_labels


# %% Helper function to count the number of correct predictions in a batch
def count_correct_predictions(output_batch: torch.tensor, label_batch: torch.tensor) -> int:
    # Convert raw model output to predicted labels
    pred_labels = logits_to_labels(output_batch)

    # Compare predicted labels with actual labels and count the number of fully correct predictions
    num_correct = torch.sum(torch.all(pred_labels == label_batch, dim=1)).item()

    return num_correct


# %% Helper function to save a checkpoint
def save_checkpoint():
    checkpoint_path = f"./checkpoint_epoch_{num_epochs:06d}.pth"
    torch.save({
        'num_epochs': num_epochs,
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
    }, checkpoint_path)
    print(f'\rCheckpoint saved at "{checkpoint_path}"')


# %% Initialize variables for the main loop
num_epochs = 0
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# %% ================ Main Training Loop ================
# Trim the record lists to avoid length inconsistency caused by a KeyboardInterrupt in the main loop
train_losses = train_losses[:num_epochs]
val_losses = val_losses[:num_epochs]
train_accuracies = train_accuracies[:num_epochs]
val_accuracies = val_accuracies[:num_epochs]

while scheduler.get_last_lr()[0] > 1e-5:
    """ Train """
    model.train()
    train_loss = 0
    train_correct = 0

    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = loss_func(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record loss
        train_loss += loss.item()

        # Record correct count
        train_correct += count_correct_predictions(outputs, labels)

    # Calculate average loss and accuracy for the epoch
    avg_train_loss = train_loss / len(train_loader)
    avg_train_accuracy = train_correct / len(train_loader.dataset)

    # Append and store the values
    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_accuracy)

    """ Validation """
    model.eval()
    val_loss = 0
    val_correct = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = loss_func(outputs, labels)

            # Record loss
            val_loss += loss.item()

            # Record correct count
            val_correct += count_correct_predictions(outputs, labels)

    avg_val_loss = val_loss / len(val_loader)
    avg_val_accuracy = val_correct / len(val_loader.dataset)

    # Append and store the values
    val_losses.append(avg_val_loss)
    val_accuracies.append(avg_val_accuracy)

    # Increment the number of epochs
    num_epochs += 1

    # Print epoch summary
    print(f"\rEpoch {num_epochs} | Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f} | "
          f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.4f}")

    # Step the scheduler
    scheduler.step(avg_val_loss)
    print("Next Epoch Learning Rate:", scheduler.get_last_lr()[0])

    # Save checkpoint every 20 epochs
    if num_epochs % 20 == 0:
        save_checkpoint()

    print()
else:
    save_checkpoint()

# %% Plot the loss and accuracy curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

# Plot the loss curves
axes[0].plot(train_losses, label='Training Loss')
axes[0].plot(val_losses, label='Validation Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Loss')
axes[0].legend()

# Plot the accuracy curves
axes[1].plot(train_accuracies, label='Training Accuracy')
axes[1].plot(val_accuracies, label='Validation Accuracy')
axes[1].set_title('Training and Validation Accuracy')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

# Adjust layout
plt.subplots_adjust(wspace=0.3)

plt.show()

# %% Load a checkpoint
checkpoint_path = "./checkpoints/27.pth"

# Load the checkpoint dictionary
checkpoint = torch.load(checkpoint_path)

# Restore the states and variables
num_epochs = checkpoint['num_epochs']
model = checkpoint['model']
optimizer = checkpoint['optimizer']
scheduler = checkpoint['scheduler']
train_losses = checkpoint['train_losses']
val_losses = checkpoint['val_losses']
train_accuracies = checkpoint['train_accuracies']
val_accuracies = checkpoint['val_accuracies']

print(f'\nCheckpoint loaded from "{checkpoint_path}"')

# %% Evaluate the model on the test set
model.eval()
test_loss = 0
test_correct = 0
test_true_labels = []
test_pred_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = loss_func(outputs, labels)

        # Record loss
        test_loss += loss.item()

        # Record correct count
        test_correct += count_correct_predictions(outputs, labels)

        # Store true and predicted labels
        test_true_labels.extend(labels.tolist())
        test_pred_labels.extend(logits_to_labels(outputs).tolist())

# Calculate the average test loss and accuracy
avg_test_loss = test_loss / len(test_loader)
avg_test_accuracy = test_correct / len(test_loader.dataset)

# Print the test results
print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.4f}")

# %% Plot the confusion matrix
# Construct class names for the confusion matrix
class_names: list[str] = []
for validity in ('Correct', 'Incorrect'):
    for posture_name in ('Down Dog', 'Plank', 'Side Plank', 'Warrior II'):
        class_names.append(f"{posture_name} ({validity})")


# Define a function to map a label to its corresponding class name's index
def label_to_index(label) -> int:
    posture_class, correctness = label
    return posture_class + (0 if correctness == 1 else 4)


# Convert true and predicted labels to indexes
test_true_indexes = list(map(label_to_index, test_true_labels))
test_pred_indexes = list(map(label_to_index, test_pred_labels))

# Generate confusion matrix
conf_matrix = confusion_matrix(test_true_indexes, test_pred_indexes)

# Plot the confusion matrix
fig, ax = plt.subplots(dpi=300)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.xticks(rotation=45, ha='right')
plt.title("Confusion Matrix on Test Set (w/o Normalization)")
plt.tight_layout()
plt.show()

# %% Plot the normalized confusion matrix
# Normalize the confusion matrix row-wise
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True)

# Plot the confusion matrix
fig, ax = plt.subplots(dpi=300)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_normalized, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='.2f')  # Show 2 decimal places
plt.xticks(rotation=45, ha='right')
plt.title("Confusion Matrix on Test Set (w/ Normalization)")
plt.tight_layout()
plt.show()
