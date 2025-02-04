import carla
import numpy as np
import queue
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
    # Get raw semantic data
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    
    # Extract semantic tags and create obstacle grid
    semantic_tags = array[:, :, 2]
    obstacle_tags = [14, 22, 24]  # Example tags for obstacles
    obstacles_mask = np.isin(semantic_tags, obstacle_tags)
    
    obstacle_grid = np.zeros((image.height, image.width), dtype=np.uint8)
    obstacle_grid[obstacles_mask] = 255  # Mark obstacles as white
    
    # Convert to CityScapes palette for visualization
    image.convert(carla.ColorConverter.CityScapesPalette)
    visual_array = np.frombuffer(image.raw_data, dtype=np.uint8)
    visual_array = np.reshape(visual_array, (image.height, image.width, 4))[:, :, :3]
    
    return visual_array, obstacle_grid

class RealTimeSemanticVisualizer:
    def __init__(self, world, transform):
        self.world = world
        self.semantic_camera = setup_semantic_camera(world, transform)
        self.image_queue = queue.Queue(maxsize=1)
        self.semantic_camera.listen(lambda image: self.image_queue.put(image, block=False))
        
        self.fig, self.axs = plt.subplots(1, 2, figsize=(12, 6))
        self.semantic_ax = self.axs[0].imshow(np.zeros((600, 800, 3), dtype=np.uint8))
        self.obstacle_ax = self.axs[1].imshow(np.zeros((600, 800), dtype=np.uint8), cmap='binary')
        
        self.axs[0].set_title("Semantic Segmentation")
        self.axs[1].set_title("Obstacle Grid")
        for ax in self.axs:
            ax.axis("off")
        
    def update(self, frame):
        try:
            # Capture the latest image from the queue
            image = self.image_queue.get(timeout=0.05)  # Wait for up to 50ms
            semantic_image, obstacle_grid = process_semantic_data(image)
            
            # Update plots with new data
            self.semantic_ax.set_data(semantic_image)
            self.obstacle_ax.set_data(obstacle_grid)
            
        except queue.Empty:
            pass
        
        return [self.semantic_ax, self.obstacle_ax]

    def animate(self):
        ani = animation.FuncAnimation(self.fig, self.update, interval=50, blit=True)
        plt.show()

def main():
    client = None
    world = None
    visualizer = None  # Initialize visualizer to None
    
    try:
        client = carla.Client('localhost', 2000)
        # client.set_timeout(2.0)
        world = client.get_world()
        
        spectator = world.get_spectator()
        transform = spectator.get_transform()
        
        visualizer = RealTimeSemanticVisualizer(world, transform)
        visualizer.animate()
        
    finally:
        if visualizer is not None:  # Check if visualizer was initialized
            if visualizer.semantic_camera is not None:
                visualizer.semantic_camera.stop()
                visualizer.semantic_camera.destroy()
        plt.close('all')

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Script interrupted by user.")
