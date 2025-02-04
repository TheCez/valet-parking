import numpy as np
import carla

def design_obstacles(x, y):
    ox, oy = [], []


    return ox, oy

def bounding_box_to_obstacle(world, grid_size=500, cell_size=1):
    


def main():
    """
    Main function to set up and visualize a map with obstacles.

    This function initializes starting and goal positions with their respective
    orientations, designs obstacles on the map, and plots the map with obstacles.

    Variables:
    - x, y: Dimensions of the map.
    - sx, sy, syaw0: Starting position coordinates and orientation in radians.
    - gx, gy, gyaw0: Goal position coordinates and orientation in radians.
    - ox, oy: Coordinates of the obstacles on the map.
    """

    print("start!")
    x, y = 51, 31
    sx, sy, syaw0 = 7.5, 6.0, np.deg2rad(90.0)
    gx, gy, gyaw0 = 42.5, 24.0, np.deg2rad(90.0)

    ox, oy = design_obstacles(x, y)

    # plot_map(x, y, ox, oy)

    plot_map(ox, oy)