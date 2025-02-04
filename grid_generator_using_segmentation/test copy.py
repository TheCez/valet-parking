import carla
import numpy as np
import cv2
import queue
import time
import matplotlib.pyplot as plt

def setup_semantic_camera(world, transform):
    camera_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')
    camera = world.spawn_actor(camera_bp, transform)
    return camera

# def process_semantic_data(image):
#     array = np.frombuffer(image.raw_data, dtype=np.uint8)
#     array = np.reshape(array, (image.height, image.width, 4))
    
#     semantic_tags = array[:, :, 2]

#     obstacle_tags = [14, 22, 24]  # Add as many tags as needed
#     obstacles_mask = np.isin(semantic_tags, obstacle_tags)
    
#     obstacle_grid = np.zeros((image.height, image.width), dtype=np.uint8)
#     obstacle_grid[obstacles_mask] = 1
    
#     # Apply morphological operations to thin the obstacles
#     kernel = np.ones((3,3), np.uint8)
#     obstacle_grid = cv2.dilate(obstacle_grid, kernel, iterations=1)
#     obstacle_grid = cv2.erode(obstacle_grid, kernel, iterations=2)
    
#     # Apply edge detection
#     edges = cv2.Canny(obstacle_grid, 50, 150)
    
#     # Thin the edges using skeletonization
#     from skimage.morphology import skeletonize
#     skeleton = skeletonize(edges > 0)
    
#     return skeleton.astype(np.uint8)

def process_semantic_data(image):
    # Convert raw data to numpy array
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    
    semantic_tags = array[:, :, 2]

    # Include relevant semantic tags
    obstacle_tags = [14, 22, 24]
    obstacles_mask = np.isin(semantic_tags, obstacle_tags)
    
    obstacle_grid = np.zeros((image.height, image.width), dtype=np.uint8)
    
    # Mark obstacles in the new array
    obstacle_grid[obstacles_mask] = 1  # Mark obstacles as 1 instead of 255
    # print(obstacle_grid)

    # Clean up noise and ensure proper data type for contours
    kernel = np.ones((3,3), np.uint8)
    obstacle_grid = cv2.morphologyEx(obstacle_grid, cv2.MORPH_CLOSE, kernel)
    
    return obstacle_grid



def capture_single_image(world, transform):
    camera = setup_semantic_camera(world, transform)
    image_queue = queue.Queue()
    camera.listen(image_queue.put)
    
    try:
        image = image_queue.get(timeout=5.0)
        obstacle_grid = process_semantic_data(image)
    finally:
        camera.stop()
        camera.destroy()
    
    return obstacle_grid

def main():
    client = None
    world = None
    
    try:
        client = carla.Client('localhost', 2000)
        world = client.get_world()
        
        spectator = world.get_spectator()
        transform = spectator.get_transform()
        
        obstacle_grid = capture_single_image(world, transform)
        
        # Save the obstacle grid
        np.save('obstacle_grid.npy', obstacle_grid)
        
        # Convert to ox, oy format
        ox, oy = [], []
        y, x = obstacle_grid.shape
        for i in range(x):
            for j in range(y):
                if obstacle_grid[j, i] == 1:
                    ox.append(i)
                    oy.append(j)
        
        # # Visualize using the requested format
        # plt.figure(figsize=(10, 8))
        # plt.cla()
        # plt.plot(ox, oy, "sk", markersize=1)
        # plt.axis("equal")
        # plt.title('Obstacle Grid')
        # # plt.gca().invert_yaxis()  # Invert y-axis to match the image orientation
        # plt.savefig('obstacle_grid.png')
        # plt.show()
        # print("Done!")

        plt.cla()
        plt.plot(ox, oy, "sk", markersize=1)
        plt.axis("equal")
    # plt.plot(x, y, linewidth=1.5, color='r')
        plt.show()
        
    finally:
        print("Finished processing")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Script interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
