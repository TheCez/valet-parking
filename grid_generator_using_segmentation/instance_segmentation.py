import carla
import numpy as np
import cv2
import queue
import time

def setup_semantic_camera(world, transform):
    # Create the semantic segmentation camera blueprint
    camera_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
    
    # Set camera attributes
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')
    
    # Spawn the camera
    camera = world.spawn_actor(camera_bp, transform)
    
    return camera

def process_semantic_image(image):
    # Convert raw data to numpy array
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    
    # Convert to CityScapes palette
    image.convert(carla.ColorConverter.CityScapesPalette)
    
    # Get the converted image data
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    
    # Remove the alpha channel
    array = array[:, :, :3]
    
    return array

def main():
    client = None
    world = None
    semantic_camera = None
    
    try:
        # Connect to CARLA server
        client = carla.Client('localhost', 2000)
        # client.set_timeout(2.0)
        world = client.get_world()
        
        # Get spectator transform for camera placement
        spectator = world.get_spectator()
        transform = spectator.get_transform()
        print(f"Spectator transform: {transform}")
        
        # Setup camera
        semantic_camera = setup_semantic_camera(world, transform)
        
        # Create a queue to store and process the sensor data
        image_queue = queue.Queue()
        semantic_camera.listen(image_queue.put)
        
        # Create display window
        cv2.namedWindow('Semantic Segmentation', cv2.WINDOW_AUTOSIZE)
        
        while True:
            try:
                # Get the image from the queue
                image = image_queue.get(timeout=2.0)
                
                # Process the image
                processed_image = process_semantic_image(image)
                
                # Display the image
                cv2.imshow('Semantic Segmentation', processed_image)
                
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except queue.Empty:
                print("Queue is empty, waiting for more images...")
                
            time.sleep(0.1)  # Add a small delay to prevent high CPU usage
                
    finally:
        print("Cleaning up...")
        # Clean up
        if semantic_camera is not None:
            semantic_camera.stop()
            semantic_camera.destroy()
        cv2.destroyAllWindows()
        print("Cleanup complete.")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Script interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
