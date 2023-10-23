import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv

from matplotlib.colors import ListedColormap

# Load your data from the CSV file
# Replace this with your own method of loading the matrix data
# In this example, we'll use random data
matrix_size = (5, 5)

frames = []

# Function to generate a random matrix for demonstration purposes
def generate_random_matrix():
    matrix_list = []
    with open("results.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            currentRow = list(map(lambda x: int(x), row))
            current_matrix = []
            for i in range(5):
                nested_row = []
                for j in range(5):
                    nested_row.append(currentRow[i * 5 + j])
                current_matrix.append(nested_row)
            matrix_list.append(np.array(current_matrix))
    return matrix_list

# Initialize the figure and axis with a white background
fig, ax = plt.subplots()
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Create an initial matrix for the first frame
matrix_list = generate_random_matrix()

custom_cmap = ListedColormap(['white'])  # Customize the colors as needed
# Display the initial matrix as text with white background
im = ax.matshow(matrix_list[0], cmap=custom_cmap)

# Initialize the text objects
texts = [[None for _ in range(matrix_size[1])] for _ in range(matrix_size[0])]

# Function to update the matrix values for each frame
def update(frame):
    # Generate a new random matrix for demonstration purposes
    new_matrix = matrix_list[frame]
    for i in range(5):
        for j in range(5):
            if texts[i][j]:
                texts[i][j].remove()  # Clear old text
            text = ax.text(j, i, new_matrix[i, j], ha='center', va='center', color='black')
            texts[i][j] = text
    ax.set_title(f'Frame {frame}')  # Add a title showing the current frame
    return im,

# Create the animation
ani = FuncAnimation(fig, update, frames=len(matrix_list), interval=1000, blit=True)

# Customize the plot (labels, titles, etc.)
ax.set_title(f'Frame 0')  # Initial title

# Save the animation as a GIF
ani.save('animated_matrix_values.gif', writer='pillow', fps=2)

# Show the animated chart
plt.show()
