import carla
import numpy as np
import cv2
import queue
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def setup_semantic_camera(world, transform):
    camera_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')
    camera = world.spawn_actor(camera_bp, transform)
    return camera

def process_semantic_data(image):
    # Get raw semantic data first (before conversion)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    
    semantic_tags = array[:, :, 2]

    obstacle_tags = [14, 22, 24]  # Add as many tags as needed
    obstacles_mask = np.isin(semantic_tags, obstacle_tags)
    
    # Create a new binary array to store obstacle information
    obstacle_grid = np.zeros((image.height, image.width), dtype=np.uint8)
    
    # Mark obstacles in the new array
    obstacle_grid[obstacles_mask] = 1  # Mark obstacles as 1 instead of 255
    
    # Convert to CityScapes palette for visualization
    image.convert(carla.ColorConverter.CityScapesPalette)
    
    # Get the converted image data
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    
    # Use the already converted visual array
    visual_array = array[:, :, :3]
    
    return visual_array, obstacle_grid


class SemanticVisualizer:
    def __init__(self, world, transform):
        self.world = world
        self.transform = transform
        self.semantic_camera = setup_semantic_camera(world, transform)
        self.image_queue = queue.Queue(maxsize=1)
        self.semantic_camera.listen(lambda image: self.image_queue.put(image, block=False))
        
        self.fig, self.axs = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.tight_layout()
        
        self.prev_semantic = None
        self.prev_obstacle = None
        
    def update(self, frame):
        try:
            image = self.image_queue.get(timeout=0.05)#, block=False)
            semantic_image, obstacle_grid = process_semantic_data(image)
            
            if self.prev_semantic is None:
                self.prev_semantic = semantic_image
                self.prev_obstacle = obstacle_grid
            
            # Apply alpha blending for smooth transitions
            alpha = 0.7
            blended_semantic = cv2.addWeighted(semantic_image, alpha, self.prev_semantic, 1 - alpha, 0)
            blended_obstacle = cv2.addWeighted(obstacle_grid, alpha, self.prev_obstacle, 1 - alpha, 0)
            
            self.axs[0].clear()
            self.axs[0].imshow(blended_semantic, interpolation='bilinear')
            self.axs[0].set_title('Semantic Segmentation')
            self.axs[0].axis('off')
            
            self.axs[1].clear()
            self.axs[1].imshow(blended_obstacle, cmap='binary', interpolation='bilinear')
            self.axs[1].set_title('Obstacle Grid')
            self.axs[1].axis('off')
            
            self.prev_semantic = semantic_image
            self.prev_obstacle = obstacle_grid
            
        except queue.Empty:
            pass
        
        self.fig.canvas.draw_idle()
        
    def animate(self):
        ani = animation.FuncAnimation(self.fig, self.update, interval=5, cache_frame_data=False)
        plt.show()

def main():
    client = None
    world = None
    visualizer = None
    
    try:
        client = carla.Client('localhost', 2000)
        world = client.get_world()
        
        spectator = world.get_spectator()
        transform = spectator.get_transform()
        
        visualizer = SemanticVisualizer(world, transform)
        visualizer.animate()
        
    finally:
        if visualizer is not None:
            if visualizer.semantic_camera is not None:
                visualizer.semantic_camera.stop()
                visualizer.semantic_camera.destroy()
        plt.close('all')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Script interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")