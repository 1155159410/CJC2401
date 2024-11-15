import os

import matplotlib.pyplot as plt
from matplotlib.image import imread

# Image file names
image_files = [
    'Down Dog.jpg', 'Goddess.jpg',
    'Plank.jpg', 'Side Plank.jpg',
    'Tree.jpg', 'Warrior.jpeg'
]

# Create a 3x2 subplot grid with specified size and DPI
fig, axes = plt.subplots(3, 2, figsize=(12, 16), dpi=300)

# Display each image and set the title (without extension)
for idx, ax in enumerate(axes.flat):
    img = imread(image_files[idx])
    ax.imshow(img)
    title = os.path.splitext(image_files[idx])[0]
    ax.set_title(title, fontsize=28)
    ax.set_xticks([])  # Hide x-ticks
    ax.set_yticks([])  # Hide y-ticks

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()
