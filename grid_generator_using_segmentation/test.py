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

def process_semantic_data(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    
    semantic_tags = array[:, :, 2]

    obstacle_tags = [14, 22, 24]  # Add as many tags as needed
    obstacles_mask = np.isin(semantic_tags, obstacle_tags)
    
    obstacle_grid = np.zeros((image.height, image.width), dtype=np.uint8)
    obstacle_grid[obstacles_mask] = 1
    
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
        
        # Visualize the obstacle grid
        plt.figure(figsize=(10, 8))
        plt.imshow(obstacle_grid, cmap='binary')
        plt.title('Obstacle Grid')
        plt.colorbar(label='Obstacle Presence')
        plt.savefig('obstacle_grid.png')
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
