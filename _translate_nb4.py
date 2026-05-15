#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Translate notebook 4: obstacle avoidance."""
import json

filepath = "C:/Users/gpx/Desktop/airgpt/io_airgpt/8-world_model/4-obstacle_avoidance.ipynb"
with open(filepath, 'r', encoding='utf-8') as f:
    nb = json.load(f)

translations = {
    "cell-0": """# 8.4 Experiment 2: Corridor Obstacle Avoidance (Inference)

> **AirSim Configuration**: Uses `settings.json`. If already running from notebook 3, restart AirSim to reset the drone position.

## Task Objective

Have the drone fly forward autonomously in an industrial inspection scenario, using depth perception to avoid buildings and facilities.

Difference from hover: hover means "stay still," obstacle avoidance means "fly and dodge" -- requiring simultaneous handling of forward progress and collision avoidance.""",
    "84107089": """![Obstacle Avoidance Scene](figures/avoidance_scene.png)

*Figure 8-8: Forward flight scene in AirSim -- the drone must navigate autonomously between buildings and facilities*

![Hover vs Obstacle Avoidance](figures/hover_vs_avoidance.png)

*Figure 8-9: Hover vs Obstacle Avoidance -- the avoidance task requires optimizing both forward progress and safety simultaneously*""",
    "cell-1": """import sys
sys.path.append('../external-libraries')

import numpy as np
import matplotlib.pyplot as plt
import airsim
import time
from PIL import Image

client = airsim.MultirotorClient()
client.confirmConnection()
DRONE = "Drone1"
print(f"Connected!")""",
    "cell-2": """## 8.4.1 Reward Function Design

The obstacle avoidance reward function must balance two objectives:

| Condition | Reward | Meaning |
|-----------|--------|---------|
| Forward progress | +1.0/step | Encourages advancement |
| Too close to obstacle | -1.0/step | Penalizes dangerous proximity |
| Collision | -100.0 | Severe penalty for collision |
| Out of bounds / crash | -100.0 | Severe penalty for loss of control |""",
    "cell-3": """START_POS = np.array([0.0, 0.0, -8.0])  # Starting position, 8m altitude
FORWARD_DIR = np.array([1.0, 0.0, 0.0])  # Forward direction (North)

def get_depth_image(client, drone_id):
    \"\"\"Get depth map, normalized to [0,1].\"\"\"
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True)
    ], vehicle_name=drone_id)
    if responses and responses[0].width > 0:
        depth = airsim.list_to_2d_float_array(
            responses[0].image_data_float, responses[0].width, responses[0].height
        )
        return np.clip(depth / 50.0, 0, 1)  # Normalized to 50m range
    return np.zeros((64, 64))

def get_rgb_image(client, drone_id):
    \"\"\"Get RGB image.\"\"\"
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ], vehicle_name=drone_id)
    if responses and responses[0].width > 0:
        img = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        return img.reshape(responses[0].height, responses[0].width, 3)
    return np.zeros((64, 64, 3), dtype=np.uint8)

print("Sensor functions defined")""",
    "cell-4": """## 8.4.2 Depth-Based Obstacle Avoidance Policy

Without pretrained DreamerV3 weights, we implement a reactive obstacle avoidance policy based on depth maps as a baseline and demonstration.

Core idea: Divide the depth map into left, center, and right regions, and steer toward whichever side has more clearance.""",
    "23d79936": """![Depth Map Illustration](figures/depth_scene.png)

*Figure 8-10: The depth map encodes "distance to objects ahead" as pixel intensity -- bright means far, dark means close*""",
    "cell-5": """def depth_based_policy(client, drone_id):
    \"\"\"Reactive obstacle avoidance policy based on depth map.\"\"\"
    depth = get_depth_image(client, drone_id)
    h, w = depth.shape

    # Divide depth map into left, center, right regions
    left_dist = depth[:, :w//3].mean()
    center_dist = depth[:, w//3:2*w//3].mean()
    right_dist = depth[:, 2*w//3:].mean()

    min_depth = depth[h//4:3*h//4, w//4:3*w//4].min()  # Minimum distance in center region

    # Basic forward motion
    pitch = 0.3  # Tilt forward
    roll = 0.0
    yaw_rate = 0.0
    throttle = 0.1

    # If obstacle ahead is too close, slow down and turn
    if min_depth < 0.15:  # Obstacle within ~7.5m
        pitch = 0.0  # Stop forward motion
        if left_dist > right_dist:
            yaw_rate = -0.5  # Turn left
        else:
            yaw_rate = 0.5   # Turn right
    elif min_depth < 0.3:  # Obstacle within ~15m
        pitch = 0.15  # Slow down
        if left_dist > right_dist:
            yaw_rate = -0.2
        else:
            yaw_rate = 0.2

    return np.array([pitch, roll, yaw_rate, throttle]), {
        'left': left_dist, 'center': center_dist, 'right': right_dist, 'min': min_depth
    }

print("Depth-based avoidance policy defined")""",
    "cell-6": """# Run obstacle avoidance experiment
client.reset()
client.enableApiControl(True, vehicle_name=DRONE)
client.armDisarm(True, vehicle_name=DRONE)
client.takeoffAsync(vehicle_name=DRONE).join()
client.moveToPositionAsync(START_POS[0], START_POS[1], START_POS[2], 3, vehicle_name=DRONE).join()
time.sleep(1)

positions = []
rewards = []
depth_snapshots = []
rgb_snapshots = []
prev_pos = START_POS.copy()

N_STEPS = 300
print(f"Starting obstacle avoidance flight, {N_STEPS} steps...")

for step in range(N_STEPS):
    action, depth_info = depth_based_policy(client, DRONE)

    client.moveByRollPitchYawrateThrottleAsync(
        float(action[1]) * 0.3,
        float(action[0]) * 0.3,
        float(action[2]) * 0.5,
        float(action[3]) * 0.5 + 0.5,
        duration=0.1, vehicle_name=DRONE
    ).join()

    state = client.getMultirotorState(vehicle_name=DRONE)
    pos = np.array([state.kinematics_estimated.position.x_val,
                    state.kinematics_estimated.position.y_val,
                    state.kinematics_estimated.position.z_val])

    # Compute reward
    collision = client.simGetCollisionInfo(vehicle_name=DRONE)
    if collision.has_collided:
        print(f"  Collision! step={step}")
        rewards.append(-100)
        positions.append(pos)
        break

    forward_progress = np.dot(pos - prev_pos, FORWARD_DIR)
    reward = forward_progress * 2.0
    if depth_info['min'] < 0.1:
        reward -= 1.0
    rewards.append(reward)
    positions.append(pos)
    prev_pos = pos.copy()

    # Save snapshot every 50 steps
    if step % 50 == 0:
        depth_snapshots.append(get_depth_image(client, DRONE))
        rgb_snapshots.append(get_rgb_image(client, DRONE))
        print(f"  step {step}: pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}), min_depth={depth_info['min']:.2f}")

positions = np.array(positions)
rewards = np.array(rewards)
total_forward = np.dot(positions[-1] - START_POS, FORWARD_DIR)
print(f"\\nFlight complete! Forward distance: {total_forward:.1f}m, Collision: {'Yes' if collision.has_collided else 'No'}")""",
    "cell-7": """# Visualize flight trajectory and depth maps
fig = plt.figure(figsize=(16, 8))

# Top-down trajectory view
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=1.5)
ax1.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start')
ax1.plot(positions[-1, 0], positions[-1, 1], 'r*', markersize=12, label='End')
ax1.set_xlabel('X (North)'); ax1.set_ylabel('Y (East)')
ax1.set_title('Flight Trajectory (Top View)'); ax1.legend(); ax1.grid(True, alpha=0.3)

# Altitude change
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(-positions[:, 2], 'b-')  # NED to normal altitude
ax2.set_xlabel('Time Step'); ax2.set_ylabel('Altitude (m)')
ax2.set_title('Flight Altitude'); ax2.grid(True, alpha=0.3)

# Depth map snapshots
if depth_snapshots:
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(depth_snapshots[0], cmap='viridis')
    ax3.set_title('Depth Map (Start)'); ax3.axis('off')

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.imshow(depth_snapshots[-1], cmap='viridis')
    ax4.set_title(f'Depth Map (Step {(len(depth_snapshots)-1)*50})'); ax4.axis('off')

plt.tight_layout()
plt.savefig('avoidance_result.png', dpi=150)
plt.show()""",
    "cell-8": """# Save RGB snapshot montage
if rgb_snapshots:
    n = min(len(rgb_snapshots), 4)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]
    for i in range(n):
        axes[i].imshow(rgb_snapshots[i][:,:,::-1])  # BGR to RGB
        axes[i].set_title(f'Step {i*50}')
        axes[i].axis('off')
    plt.suptitle('Drone POV Snapshots', fontsize=14)
    plt.tight_layout()
    plt.savefig('avoidance_snapshots.png', dpi=150)
    plt.show()""",
    "cell-9": """## 8.4.3 World Model "Imagination" Visualization

The most unique capability of world models is "imagining the future." Below we use a simplified world model to demonstrate this process: given the current state and an action sequence, the model predicts state changes over the next several steps.""",
    "cell-10": """import torch
from world_model_tools import SimpleWorldModel

# Demonstrate "imagination" with simplified world model
wm = SimpleWorldModel(state_dim=6, action_dim=4)

# Current state
state = client.getMultirotorState(vehicle_name=DRONE)
pos = state.kinematics_estimated.position
vel = state.kinematics_estimated.linear_velocity
current = torch.tensor([[pos.x_val, pos.y_val, pos.z_val,
                          vel.x_val, vel.y_val, vel.z_val]])

# Imagine 3 different action sequences
scenarios = {
    'Fly straight': torch.tensor([[0.3, 0.0, 0.0, 0.1]]),
    'Turn left': torch.tensor([[0.1, 0.0, -0.5, 0.1]]),
    'Turn right': torch.tensor([[0.1, 0.0, 0.5, 0.1]]),
}

fig, ax = plt.subplots(figsize=(8, 6))
colors = {'Fly straight': 'blue', 'Turn left': 'green', 'Turn right': 'red'}

with torch.no_grad():
    for name, action in scenarios.items():
        s = current.clone()
        traj = [s.numpy()[0, :3]]
        for _ in range(20):
            s, r = wm(s, action)
            traj.append(s.numpy()[0, :3])
        traj = np.array(traj)
        ax.plot(traj[:, 0], traj[:, 1], f'{colors[name][0]}-o',
                markersize=3, label=f'Imagined: {name}', alpha=0.7)

ax.plot(current[0, 0].item(), current[0, 1].item(), 'k*', markersize=15, label='Current position')
ax.set_xlabel('X (North)'); ax.set_ylabel('Y (East)')
ax.set_title('World Model "Imagination": Predicting Future Trajectories for Different Actions')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('imagination_visualization.png', dpi=150)
plt.show()
print("Note: This is an untrained simplified model, so trajectories are inaccurate.")
print("A trained DreamerV3 can accurately predict real environment dynamics.")""",
    "cell-11": """## 8.4.4 Summary

This section demonstrated:

1. **Depth maps** are the key sensor input for obstacle avoidance -- they directly tell the drone "how far ahead obstacles are"
2. A **reactive policy** based on depth maps can achieve basic obstacle avoidance, but lacks foresight
3. The core advantage of world models is **"imagination"** -- simulating consequences of different actions in the mind and choosing the safest path
4. A pretrained DreamerV3 can learn more sophisticated avoidance strategies (requires GPU training, covered in later sections)

In the next section (8.5), we'll train a DreamerV3 model from scratch for the hover task on a GPU machine.""",
}

for cell in nb.get('cells', []):
    cell_id = cell.get('id', '')
    if cell_id in translations:
        new_source = translations[cell_id]
        lines = new_source.split('\n')
        cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Translated: {filepath}")
