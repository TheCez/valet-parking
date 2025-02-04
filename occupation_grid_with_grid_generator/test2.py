import numpy as np
import carla
import cv2
import random
import time
import subprocess
import threading

def create_2d_obstacle_grid(world, grid_size=500, cell_size=1):
    # Create an empty grid
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    
    # Calculate the grid center
    center = grid_size // 2
    
    # Get bounding boxes for static objects (e.g., walls)
    walls = world.get_level_bbs(carla.CityObjectLabel.Other)
    
    # Mark obstacles on the grid
    for bb in walls:
        mark_bounding_box(grid, bb, center, cell_size, value=1)
    
    return grid

def mark_ego_vehicle(grid, ego_vehicle, center, cell_size):
    # Create a copy of the grid
    new_grid = grid.copy()
    
    # Mark ego vehicle's bounding box
    ego_bb = ego_vehicle.bounding_box
    ego_transform = ego_vehicle.get_transform()
    mark_bounding_box(new_grid, ego_bb, center, cell_size, value=2, transform=ego_transform)
    
    return new_grid

def mark_bounding_box(grid, bb, center, cell_size, value, transform=None):
    if transform:
        bb_center = transform.transform(bb.location)
        rotation = transform.rotation
    else:
        bb_center = bb.location
        rotation = bb.rotation
    
    bb_extent = bb.extent
    
    # Calculate the corners of the rotated bounding box
    corners = get_bounding_box_corners(bb_center, bb_extent, rotation)
    
    # Convert corners to grid coordinates
    grid_corners = []
    for corner in corners:
        x = int(center + corner.x / cell_size)
        y = int(center + corner.y / cell_size)
        grid_corners.append((x, y))
    
    # Create a mask for the rotated bounding box
    mask = np.zeros(grid.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(grid_corners, dtype=np.int32)], value)
    
    # Apply the mask to the grid
    grid[mask == value] = value

def get_bounding_box_corners(center, extent, rotation):
    """
    Calculate the 4 corners of a rotated bounding box in world coordinates.
    """
    # Convert rotation to radians
    yaw_rad = np.radians(rotation.yaw)
    
    # Define the relative positions of corners in local space (2D projection)
    local_corners = [
        carla.Location(x=-extent.x, y=-extent.y),
        carla.Location(x= extent.x, y=-extent.y),
        carla.Location(x= extent.x, y= extent.y),
        carla.Location(x=-extent.x, y= extent.y),
    ]
    
    # Rotate and translate corners to world space
    world_corners = []
    for corner in local_corners:
        # Apply rotation (yaw only for 2D visualization)
        rotated_x = corner.x * np.cos(yaw_rad) - corner.y * np.sin(yaw_rad)
        rotated_y = corner.x * np.sin(yaw_rad) + corner.y * np.cos(yaw_rad)
        
        # Translate to world position (center of the bounding box)
        world_corner = carla.Location(
            x=center.x + rotated_x,
            y=center.y + rotated_y,
            z=center.z  # Z is ignored for 2D visualization
        )
        world_corners.append(world_corner)
    
    return world_corners

def visualize_grid_animated(grid, ego_vehicle, zoom_factor=2, context_size=200):
    color_map = np.array([[255, 255, 255], [0, 0, 0], [255, 0, 0]], dtype=np.uint8)
    
    cv2.namedWindow('Animated Obstacle Grid', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Animated Obstacle Grid', 800, 800)
    
    while True:
        # Mark ego vehicle on a copy of the grid
        current_grid = mark_ego_vehicle(grid, ego_vehicle, grid.shape[0] // 2, 1)
        
        # Apply the color map
        colored_grid = color_map[current_grid]
        
        # Find ego vehicle position
        ego_positions = np.where(current_grid == 2)
        
        if len(ego_positions[0]) > 0:
            center_y, center_x = ego_positions[0].mean(), ego_positions[1].mean()
            
            # Calculate the visible area
            start_y = max(0, int(center_y - context_size // 2))
            end_y = min(grid.shape[0], int(center_y + context_size // 2))
            start_x = max(0, int(center_x - context_size // 2))
            end_x = min(grid.shape[1], int(center_x + context_size // 2))
            
            # Adjust if too close to the edges
            if end_y >= grid.shape[0] - 10:
                shift = end_y - (grid.shape[0] - 10)
                start_y = max(0, start_y - shift)
                end_y = grid.shape[0]
            if start_y <= 10:
                shift = 10 - start_y
                end_y = min(grid.shape[0], end_y + shift)
                start_y = 0
            if end_x >= grid.shape[1] - 10:
                shift = end_x - (grid.shape[1] - 10)
                start_x = max(0, start_x - shift)
                end_x = grid.shape[1]
            if start_x <= 10:
                shift = 10 - start_x
                end_x = min(grid.shape[1], end_x + shift)
                start_x = 0
            
            context_grid = colored_grid[start_y:end_y, start_x:end_x]
            
            # Zoom in
            zoomed_grid = cv2.resize(context_grid, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_NEAREST)
            
            cv2.imshow('Animated Obstacle Grid', zoomed_grid)
        else:
            cv2.imshow('Animated Obstacle Grid', colored_grid)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        time.sleep(0.1)  # Adjust this to control animation speed

    cv2.destroyAllWindows()

# Main execution
client = carla.Client('localhost', 2000)
world = client.get_world()
# vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')
# spawn_points = world.get_map().get_spawn_points()
# ego_vehicle = world.spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))

# Get all actors in the world
all_actors = world.get_actors()

# Filter for vehicles
vehicles = all_actors.filter('vehicle.*')
print(vehicles)

ego_vehicle=vehicles[0]

# # Find the vehicle with the matching role_name
# target_vehicle_name = "hero"
# ego_vehicle = None

# for vehicle in vehicles:
#     if vehicle.attributes.get('role_name') == target_vehicle_name:
#         ego_vehicle = vehicle
#         break

# if ego_vehicle is not None:
#     print(f"Found vehicle: {ego_vehicle.id}")
# else:
#     print("Vehicle not found")

# Create the static obstacle grid
static_obstacle_grid = create_2d_obstacle_grid(world)




# Start visualization in a separate thread

vis_thread = threading.Thread(target=visualize_grid_animated, args=(static_obstacle_grid, ego_vehicle, 2, 200))
vis_thread.start()

# ego_vehicle.set_autopilot(True)
# # Run the manual_control.py script
# control_thread = threading.Thread(target=subprocess.run, args=(['python', 'manual_control.py', '--rolename="hero"'],))
# control_thread.start()

# # Get all actors in the world
# all_actors = world.get_actors()

# # Filter for vehicles
# vehicles = all_actors.filter('vehicle.*')

# # Find the vehicle with the matching name
# target_vehicle_name = "hero"
# ego_vehicle = None

# for vehicle in vehicles:
#     if vehicle.attributes.get('role_name') == target_vehicle_name:
#         ego_vehicle = vehicle
#         break
# if ego_vehicle:
#     print(f"Found vehicle: {ego_vehicle.id}")
# else:
#     print("Vehicle not found")







# # Move the ego vehicle
# while vis_thread.is_alive():
#     # Simple random movement
#     control = carla.VehicleControl()
#     control.throttle = random.uniform(0.3, 0.7)
#     control.steer = random.uniform(-0.3, 0.3)
#     ego_vehicle.apply_control(control)
#     time.sleep(0.1)

# ego_vehicle.destroy()
