import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# File paths for each column
folders = ['./benchmark_output/posenet', './benchmark_output/movenet', './benchmark_output/blazepose']
filenames = ['00000340.png', '00000165.png', '00000207.png', '00000876.png']

# Labels for rows (pose names) and columns (model names)
row_labels = ['Down Dog', 'Plank', 'Tree', 'Warrior']
col_labels = ['PoseNet', 'MoveNet', 'BlazePose']

# Create a 4x3 grid of subplots (4 rows, 3 columns)
fig, axes = plt.subplots(4, 3, figsize=(12, 12))

# Loop through each row (images) and each column (folders/models)
for row in range(4):  # Loop through all 4 rows (poses)
    for col in range(3):  # Loop through each column
        img_path = f'{folders[col]}/{filenames[row]}'  # Transpose the path access
        img = mpimg.imread(img_path)

        # Plot the image in the corresponding subplot
        ax = axes[row, col]
        ax.imshow(img)
        ax.set_xticks([])  # Remove tick marks on x-axis
        ax.set_yticks([])  # Remove tick marks on y-axis

# Add column labels (model names) at the top of each column
for col in range(3):
    axes[0, col].set_title(col_labels[col], fontsize=14)

# Add row labels (pose names) on the left-hand side of each row
for row in range(4):
    axes[row, 0].text(-0.1, 0.5, row_labels[row],
                      fontsize=14, va='center', ha='right', transform=axes[row, 0].transAxes)

plt.tight_layout()
plt.show()
