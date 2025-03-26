import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.models.objects import BallObject
import numpy as np
from robosuite.environments.manipulation.pick_place import PickPlaceCan
import time
import torch

# Custom Environment
class CustomPickPlaceCan(PickPlaceCan):
    def _load_model(self):
        super()._load_model()
        ball = BallObject(name="red_ball", size=[0.05], rgba=[1, 0, 0, 1])
        ball_mjcf = ball.get_obj()
        ball_mjcf.set("pos", "0.1 0 1.1")  # Position of the red ball
        self.model.worldbody.append(ball_mjcf)

def inverse_square_gradient(action, obstacle_pos, safety_factor=0.05):
    """
    Torch-based implementation of inverse square gradient using torch.autograd.grad.
    
    Args:
        action (numpy.ndarray): Current action (position components)
        obstacle_pos (numpy.ndarray): Obstacle position
        safety_factor: Controls strength of repulsion
        
    Returns:
        numpy.ndarray: Gradient to push away from obstacle
    """
    # Convert numpy arrays to torch tensors
    action_tensor = torch.tensor(action, dtype=torch.float32, requires_grad=True)
    obstacle_tensor = torch.tensor(obstacle_pos, dtype=torch.float32)
    
    # Calculate distance between action and obstacle (position only)
    position_diff = action_tensor[:3] - obstacle_tensor[:3]
    dist = torch.linalg.norm(position_diff)
    
    # Only apply gradient if we're close to the obstacle
    # if dist.item() < 0.2:  # Threshold for applying avoidance
    # Prevent division by zero
    safe_dist = torch.clamp(dist, min=1e-6)
    
    # Inverse square repulsive potential
    repulsive_potential = safety_factor / (safe_dist ** 2)
    
    # Calculate gradient using autograd.grad
    grad = torch.autograd.grad(
        repulsive_potential, 
        action_tensor,
        create_graph=False
    )[0].numpy()
    
    # # Apply condition: if |y_action - obstacle[1]| > 0.1, set grad to 0
    # if abs(action[1] - obstacle_pos[1]) > 0.1:
    #     grad = np.zeros_like(action)
        
    # Negate to push away from obstacle and apply only to position components
    result = np.zeros_like(action)
    result[:3] = grad[:3]  # Negate for repulsion
    print("z = ", action[2])
    print("Grad torch:", grad[:3])
    return result
    # else:
    #     return np.zeros_like(action)


# Create environment with the OSC controller
env = CustomPickPlaceCan(
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    controller_configs={
        "type": "OSC_POSE",
        "input_max": 1,
        "input_min": -1,
        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        "kp": 150,
        "damping_ratio": 1,
        "impedance_mode": "fixed",
        "control_delta": True,
        "interpolation": None,
        "uncouple_pos_ori": True
    }
)

# Wrap with Gym wrapper
env = GymWrapper(env)
print("Action space dimension (wrapped):", env.action_space.shape[0])

# Reset environment to get a clean state
obs = env.reset()

# Get obstacle position (red ball)
ball_pos = env.unwrapped.sim.data.get_body_xpos("red_ball_main").copy()
print("Ball position:", ball_pos)

# Get initial EE position
initial_ee_pos = env.unwrapped.sim.data.get_site_xpos("gripper0_grip_site").copy()
print("Initial EE position:", initial_ee_pos)

# Define our target positions
first_target = np.array([0.1, initial_ee_pos[1], 1.1])  # x=0.1, z=1.5, keep current y
final_target = np.array([0.1, 2.0, 1.1])               # x=0.1, y=2.0, z=1.5

print("First target:", first_target)
print("Final target:", final_target)

# First part: Move to the first target position
print("\nMoving to first target position...")
step_size = 0.1  # Adjust based on controller sensitivity
max_steps = 1000

for i in range(max_steps):
    # Get current EE position
    ee_pos = env.unwrapped.sim.data.get_site_xpos("gripper0_grip_site").copy()
    
    # Compute vector to target
    direction = first_target - ee_pos
    distance = np.linalg.norm(direction)
    
    # Stop condition
    if distance < 0.05:
        print(f"Reached first target position at step {i}")
        print(f"Current position: {ee_pos}")
        break
        
    # Normalize direction and scale
    if distance > 0:
        direction = direction / distance * min(step_size, distance)
    
    # Create base action for OSC controller
    action = np.zeros(7)  # 7D action space for OSC_POSE
    action[:3] = direction  # First 3 dimensions are position
    
    # Apply obstacle avoidance
    action = action
    
    # Take step
    if i % 20 == 0:
        print(f"Step {i}:")
        print(f"  EE position: {ee_pos}")
        print(f"  Distance to target: {distance}")
    
    env.render()
    env.step(action)
    time.sleep(0.01)

# Second part: Move along y-axis to y=2
print("\nMoving along Y-axis to y=2...")

for i in range(max_steps):
    # Get current EE position
    ee_pos = env.unwrapped.sim.data.get_site_xpos("gripper0_grip_site").copy()
    
    # For y-axis movement, we only care about the y component
    y_distance = final_target[1] - ee_pos[1]
    distance_to_target = abs(y_distance)
    
    # Stop condition
    if distance_to_target < 0.05:
        print(f"Reached final target position at step {i}")
        print(f"Final position: {ee_pos}")
        break
    
    # Create action for y-axis movement only
    action = np.zeros(7)
    action[1] = np.sign(y_distance) * min(step_size, distance_to_target)  # Y-axis movement
    
    # Apply obstacle avoidance
    avoidance_grad = inverse_square_gradient(action, ball_pos, safety_factor=0.02)
    # inverse_square_gradient_notorch(action, ball_pos, safety_factor=0.02)
    action = action + avoidance_grad
    
    # Take step
    if i % 20 == 0:
        print(f"Step {i}:")
        print(f"  EE position: {ee_pos}")
        print(f"  Y-distance to target: {y_distance}")
    
    env.render()
    env.step(action)
    time.sleep(0.01)

# Get final position
final_ee_pos = env.unwrapped.sim.data.get_site_xpos("gripper0_grip_site")
print("\nResults:")
print("Initial position:", initial_ee_pos)
print("First target:", first_target)
print("Final target:", final_target)
print("Final position:", final_ee_pos)
print("Distance to final target:", np.linalg.norm(final_ee_pos - final_target))
print("Distance to obstacle:", np.linalg.norm(final_ee_pos - ball_pos))

print("Simulation complete")