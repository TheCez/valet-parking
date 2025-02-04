import numpy as np
import matplotlib.pyplot as plt

# Load the 2D grid
grid = np.load('grid.npy')

# Create the plot
plt.figure(figsize=(10, 10))
plt.imshow(grid, cmap='binary', interpolation='nearest')
plt.title('2D Grid Obstacle Map')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.colorbar(label='Obstacle Presence')
plt.tight_layout()
plt.savefig('grid_plot.png')
plt.show()
