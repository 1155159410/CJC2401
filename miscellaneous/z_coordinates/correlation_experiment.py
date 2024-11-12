import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

# Load images
img_near = mpimg.imread('./near.png')
img_far = mpimg.imread('./far.png')

# Create a figure and set of subplots for the images
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# Plot near.png on the left
axes[0].imshow(img_near)
axes[0].set_title('near.png')

# Plot far.png on the right
axes[1].imshow(img_far)
axes[1].set_title('far.png')

# Configure axes: show frame, hide ticks
for i in range(2):
    axes[i].axis('on')  # Show frame
    axes[i].set_xticks([])  # No x-axis ticks
    axes[i].set_yticks([])  # No y-axis ticks

# Display the image plots
plt.tight_layout()
plt.show()

# Process images with MediaPipe Pose
mp_pose = mp.solutions.pose
with mp_pose.Pose(static_image_mode=True) as pose:
    # Convert images to uint8 and process
    result_near = pose.process((img_near * 255).astype(np.uint8))
    result_far = pose.process((img_far * 255).astype(np.uint8))

# Extract z-coordinates from landmarks
z_near = np.array([landmark.z for landmark in result_near.pose_landmarks.landmark])
z_far = np.array([landmark.z for landmark in result_far.pose_landmarks.landmark])

# Create a figure and set of subplots for z-coordinate graphs
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Plot z-coordinates before normalization
axes[0].plot(z_near, label='near.png')
axes[0].plot(z_far, label='far.png')
axes[0].set_title('Before Normalization')

# Normalize z-coordinates
z_near_norm = z_near / np.linalg.norm(z_near)
z_far_norm = z_far / np.linalg.norm(z_far)

# Plot z-coordinates after normalization
axes[1].plot(z_near_norm, label='near.png')
axes[1].plot(z_far_norm, label='far.png')
axes[1].set_title('After Normalization')

# Configure axes labels, limits, and legends
for i in range(2):
    axes[i].axhline(y=0, color='black', linestyle='--')  # Add a reference line at y = 0
    axes[i].set_xlabel('Body Part Class (Index: 0-32)')
    axes[i].set_ylabel('Predicted Z-Coordinate')
    axes[i].set_xlim(0, 32)
    axes[i].legend()

# Adjust layout and display the plots
plt.tight_layout()
plt.subplots_adjust(wspace=0.4)
plt.show()
