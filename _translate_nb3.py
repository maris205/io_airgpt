#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Translate notebook 3: hover experiment."""
import json, sys

filepath = "C:/Users/gpx/Desktop/airgpt/io_airgpt/8-world_model/3-hover_experiment.ipynb"
with open(filepath, 'r', encoding='utf-8') as f:
    nb = json.load(f)

translations = {
    "cell-0": """# 8.3 Experiment 1: Autonomous Hover (Inference)

> **AirSim Configuration**: Uses `settings.json` (single drone). Copy it to `~/Documents/AirSim/settings.json` and launch AirSim.

## Task Objective

Keep the drone hovering stably at position (0, 0, -5), resisting environmental disturbances.

This section uses pretrained world model weights for inference demonstration. If pretrained weights are unavailable, we provide a baseline policy based on a simplified world model for comparison.""",
    "cell-1": """import sys
sys.path.append('../external-libraries')
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
import airsim
import time
from PIL import Image

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
print(f"Connected! Drones: {client.listVehicles()}")""",
    "cell-2": """## 8.3.1 Reward Function Design

The reward function is the core of reinforcement learning -- it defines "what constitutes good behavior":

| Condition | Reward | Meaning |
|-----------|--------|---------|
| Staying near target position | +1.0/step | Encourages stable hover |
| Excessive attitude tilt | -0.1/step | Penalizes instability |
| Crash or out of bounds | -10.0 | Severe penalty for dangerous behavior |""",
    "6bd1621f": """![Hover Task Scene](figures/hover_scene.png)

*Figure 8-7: Hover task scene in AirSim -- the drone must maintain stability at the designated position*""",
    "cell-3": """# Define hover task parameters
TARGET = np.array([0.0, 0.0, -5.0])  # Target position (NED coordinates, z=-5 means 5m altitude)
DRONE = "Drone1"

def get_drone_state(client, drone_id):
    \"\"\"Get the complete drone state.\"\"\"
    state = client.getMultirotorState(vehicle_name=drone_id)
    pos = state.kinematics_estimated.position
    vel = state.kinematics_estimated.linear_velocity
    ori = state.kinematics_estimated.orientation
    return {
        'pos': np.array([pos.x_val, pos.y_val, pos.z_val]),
        'vel': np.array([vel.x_val, vel.y_val, vel.z_val]),
        'ori': np.array([ori.w_val, ori.x_val, ori.y_val, ori.z_val]),
    }

def compute_hover_reward(state, target):
    \"\"\"Compute hover reward.\"\"\"
    dist = np.linalg.norm(state['pos'] - target)
    tilt = abs(state['ori'][1]) + abs(state['ori'][2])  # roll + pitch
    reward = max(0, 1.0 - dist / 5.0) - 0.1 * tilt
    return float(reward), dist

print(f"Target position: {TARGET}")
print(f"Reward function: reward = max(0, 1 - dist/5) - 0.1 * tilt")""",
    "cell-4": """## 8.3.2 Baseline: Random Actions vs Simple PD Control

Before loading the world model, let's see how two baselines perform as reference points.""",
    "cell-5": """def run_episode(client, policy_fn, target, n_steps=200, label=""):
    \"\"\"Run one episode, recording trajectory and rewards.\"\"\"
    client.reset()
    client.enableApiControl(True, vehicle_name=DRONE)
    client.armDisarm(True, vehicle_name=DRONE)
    client.takeoffAsync(vehicle_name=DRONE).join()
    client.moveToPositionAsync(target[0], target[1], target[2], 3, vehicle_name=DRONE).join()
    time.sleep(1)

    positions = []
    rewards = []

    for step in range(n_steps):
        state = get_drone_state(client, DRONE)
        action = policy_fn(state, target)
        # Execute action
        client.moveByRollPitchYawrateThrottleAsync(
            float(action[1]) * 0.3,   # roll
            float(action[0]) * 0.3,   # pitch
            float(action[2]) * 0.5,   # yaw_rate
            float(action[3]) * 0.5 + 0.5,  # throttle
            duration=0.1, vehicle_name=DRONE
        ).join()

        state = get_drone_state(client, DRONE)
        reward, dist = compute_hover_reward(state, target)
        positions.append(state['pos'].copy())
        rewards.append(reward)

        if dist > 15:  # Flew too far
            print(f"  [{label}] Out of bounds, step={step}")
            break

    return np.array(positions), np.array(rewards)

# Policy 1: Random actions
def random_policy(state, target):
    return np.random.randn(4) * 0.3

# Policy 2: Simple PD control (proportional-derivative control based on position error)
def pd_policy(state, target):
    pos_err = target - state['pos']
    vel = state['vel']
    kp, kd = 0.5, 0.3
    control = kp * pos_err - kd * vel
    pitch = np.clip(control[0], -1, 1)
    roll = np.clip(control[1], -1, 1)
    throttle = np.clip(control[2] * 0.5, -1, 1)
    return np.array([pitch, roll, 0.0, throttle])

print("Running baseline comparison experiments...")
print("\\n--- Random Actions ---")
pos_random, rew_random = run_episode(client, random_policy, TARGET, 150, "Random")
print(f"  Mean reward: {rew_random.mean():.3f}, Final distance: {np.linalg.norm(pos_random[-1] - TARGET):.2f}m")

print("\\n--- PD Control ---")
pos_pd, rew_pd = run_episode(client, pd_policy, TARGET, 150, "PD")
print(f"  Mean reward: {rew_pd.mean():.3f}, Final distance: {np.linalg.norm(pos_pd[-1] - TARGET):.2f}m")""",
    "cell-6": """# Visualization comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# XY trajectory
axes[0].plot(pos_random[:, 0], pos_random[:, 1], 'r-', alpha=0.7, label='Random')
axes[0].plot(pos_pd[:, 0], pos_pd[:, 1], 'b-', alpha=0.7, label='PD Control')
axes[0].plot(TARGET[0], TARGET[1], 'k*', markersize=15, label='Target')
axes[0].set_xlabel('X (m)'); axes[0].set_ylabel('Y (m)')
axes[0].set_title('XY Plane Trajectory'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Altitude
axes[1].plot(pos_random[:, 2], 'r-', alpha=0.7, label='Random')
axes[1].plot(pos_pd[:, 2], 'b-', alpha=0.7, label='PD Control')
axes[1].axhline(y=TARGET[2], color='k', linestyle='--', label='Target altitude')
axes[1].set_xlabel('Time Step'); axes[1].set_ylabel('Z (m)')
axes[1].set_title('Altitude Change'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

# Reward
axes[2].plot(rew_random, 'r-', alpha=0.7, label='Random')
axes[2].plot(rew_pd, 'b-', alpha=0.7, label='PD Control')
axes[2].set_xlabel('Time Step'); axes[2].set_ylabel('Reward')
axes[2].set_title('Per-Step Reward'); axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hover_baseline_comparison.png', dpi=150)
plt.show()
print("Comparison plot saved: hover_baseline_comparison.png")""",
    "cell-7": """## 8.3.3 World Model Policy (Pretrained Weight Inference)

If pretrained DreamerV3 weights are available, load them and run inference directly.

If pretrained weights are not available (current state), we use an MPC (Model Predictive Control) policy based on a simplified world model to demonstrate the core idea: **try multiple actions in imagination and pick the best one**.""",
    "cell-8": """import torch
from world_model_tools import SimpleWorldModel
import os

# Check for pretrained weights
CHECKPOINT = 'models/hover_checkpoint/model.pt'
has_pretrained = os.path.exists(CHECKPOINT)

if has_pretrained:
    print(f"Found pretrained weights: {CHECKPOINT}")
    # TODO: Load DreamerV3 weights and run inference
    # model = DreamerV3.load(CHECKPOINT)
else:
    print("No pretrained weights found, using simplified MPC policy for demonstration")
    print("MPC policy: At each step, use the world model to 'imagine' consequences of multiple actions, then select the one with highest reward")

# Simplified MPC policy (online planning with world model)
def mpc_policy(state_dict, target, n_candidates=50, horizon=5):
    \"\"\"Model Predictive Control: sample multiple action sequences, evaluate with world model, select optimal.\"\"\"
    pos = state_dict['pos']
    vel = state_dict['vel']
    best_action = None
    best_reward = -float('inf')

    for _ in range(n_candidates):
        # Randomly sample an action
        action = np.random.randn(4) * 0.5
        # Simple physics model "imagination": predict position after executing this action
        pred_pos = pos + vel * 0.1 + np.array([action[0], action[1], action[3]]) * 0.05
        pred_dist = np.linalg.norm(pred_pos - target)
        pred_reward = max(0, 1.0 - pred_dist / 5.0)

        if pred_reward > best_reward:
            best_reward = pred_reward
            best_action = action

    return best_action

print("\\n--- MPC Policy (World Model Planning) ---")
pos_mpc, rew_mpc = run_episode(client, mpc_policy, TARGET, 150, "MPC")
print(f"  Mean reward: {rew_mpc.mean():.3f}, Final distance: {np.linalg.norm(pos_mpc[-1] - TARGET):.2f}m")""",
    "cell-9": """# Three-way policy comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for data, label, color in [
    (pos_random, 'Random', 'r'),
    (pos_pd, 'PD Control', 'b'),
    (pos_mpc, 'MPC (World Model)', 'g'),
]:
    axes[0].plot(data[:, 0], data[:, 1], f'{color}-', alpha=0.7, label=label)
axes[0].plot(TARGET[0], TARGET[1], 'k*', markersize=15, label='Target')
axes[0].set_title('XY Plane Trajectory'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

for data, label, color in [
    (rew_random, 'Random', 'r'),
    (rew_pd, 'PD Control', 'b'),
    (rew_mpc, 'MPC (World Model)', 'g'),
]:
    axes[1].plot(np.cumsum(data), f'{color}-', alpha=0.7, label=label)
axes[1].set_title('Cumulative Reward'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

# Distance to target
for data, label, color in [
    (pos_random, 'Random', 'r'),
    (pos_pd, 'PD Control', 'b'),
    (pos_mpc, 'MPC (World Model)', 'g'),
]:
    dists = np.linalg.norm(data - TARGET, axis=1)
    axes[2].plot(dists, f'{color}-', alpha=0.7, label=label)
axes[2].set_title('Distance to Target'); axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hover_3way_comparison.png', dpi=150)
plt.show()""",
    "cell-10": """# Capture screenshot of final hover state
responses = client.simGetImages([
    airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
], vehicle_name=DRONE)

if responses and responses[0].width > 0:
    img = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    img = img.reshape(responses[0].height, responses[0].width, 3)
    Image.fromarray(img[:,:,::-1]).save('hover_final_view.png')
    print("Hover state screenshot saved: hover_final_view.png")

state = get_drone_state(client, DRONE)
print(f"Final position: {state['pos']}")
print(f"Distance to target: {np.linalg.norm(state['pos'] - TARGET):.2f}m")""",
    "cell-11": """## 8.3.4 Summary

| Policy | Principle | Hover Performance |
|--------|-----------|-------------------|
| Random actions | No intelligence | Very poor, drifts away quickly |
| PD Control | Classical control theory | Good, but requires manual tuning |
| MPC (World Model) | Planning in imagination | Good, automatically learns control policy |
| DreamerV3 (Pretrained) | Deep world model | Best (requires pretrained weights) |

The MPC policy demonstrates the core value of world models: **instead of manually designing control rules, automatically select optimal actions by "imagining" the consequences of different actions**.

In the next section, we'll further validate this approach on a more complex obstacle avoidance task.""",
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
