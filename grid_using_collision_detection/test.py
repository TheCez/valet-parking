import carla
import numpy as np
import time
import matplotlib.pyplot as plt
import queue

class CollisionDetector:
    def __init__(self, world, grid_size=500, cell_size=0.2):
        self.world = world
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
        self.center = grid_size // 2
        self.collision_points = []
        
    def collision_callback(self, event):
        # Get collision location
        collision_location = event.other_actor.get_location()
        
        # Convert world location to grid coordinates
        grid_x = int(self.center + collision_location.x / self.cell_size)
        grid_y = int(self.center + collision_location.y / self.cell_size)
        
        # Store collision point if within grid bounds
        if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
            self.collision_points.append((grid_x, grid_y))
            self.grid[grid_y, grid_x] = 1

def create_obstacle_grid(world, duration=10):
    try:
        # Create collision detector
        detector = CollisionDetector(world)
        
        # Create and spawn collision sensor
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        spawn_point = carla.Transform(carla.Location(z=2.0))
        collision_sensor = world.spawn_actor(blueprint, spawn_point)
        
        # Register collision callback
        collision_sensor.listen(lambda event: detector.collision_callback(event))
        
        # Use a pedestrian as probe
        walker_bp = world.get_blueprint_library().find('walker.pedestrian.0001')
        
        # Scan the environment
        for x in range(-25, 25, 1):
            for y in range(-25, 25, 1):
                location = carla.Location(x=float(x), y=float(y), z=0.5)
                transform = carla.Transform(location)
                try:
                    probe = world.spawn_actor(walker_bp, transform)
                    time.sleep(0.01)  # Small delay to allow collision detection
                    probe.destroy()
                except:
                    # If spawn fails due to collision, mark as obstacle
                    grid_x = int(detector.center + x / detector.cell_size)
                    grid_y = int(detector.center + y / detector.cell_size)
                    if 0 <= grid_x < detector.grid_size and 0 <= grid_y < detector.grid_size:
                        detector.grid[grid_y, grid_x] = 1
        
        if collision_sensor and collision_sensor.is_alive:
            collision_sensor.destroy()
            
        return detector.grid, detector.collision_points
        
    except Exception as e:
        if collision_sensor and collision_sensor.is_alive:
            collision_sensor.destroy()
        print(f"An error occurred: {e}")
        return None, None


def main():
    client = None
    world = None
    
    try:
        # Connect to CARLA
        client = carla.Client('localhost', 2000)
        # client.set_timeout(10.0)
        world = client.get_world()
        
        # Create obstacle grid
        print("Creating obstacle grid...")
        grid, collision_points = create_obstacle_grid(world)
        
        if grid is not None:
            # Save the grid
            np.save('obstacle_grid_collision.npy', grid)
            
            # Convert to ox, oy format for visualization
            ox = [point[0] for point in collision_points]
            oy = [point[1] for point in collision_points]
            
            # Visualize
            plt.figure(figsize=(10, 10))
            plt.plot(ox, oy, "sk", markersize=1)
            plt.grid(True)
            plt.axis("equal")
            plt.title("Obstacle Grid from Collision Detection")
            plt.savefig('collision_grid_collision.png')
            plt.show()
            
            print("Done! Grid saved as 'obstacle_grid.npy'")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        
    finally:
        print("Finished processing")

if __name__ == '__main__':
    main()
