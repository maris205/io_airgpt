#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Translate notebook 6: train avoidance."""
import json

filepath = "C:/Users/gpx/Desktop/airgpt/io_airgpt/8-world_model/6-train_avoidance.ipynb"
with open(filepath, 'r', encoding='utf-8') as f:
    nb = json.load(f)

translations = {
    "cell-0": """# 8.6 Experiment 4: Training Obstacle Avoidance from Scratch

> **Note**: This is a **training process notebook** and does not need to be run end-to-end in one sitting.
>
> **Important**: The obstacle avoidance task requires a depth map at each step, and each episode needs reset + takeoff + moveToPosition, so AirSim must remain running throughout. This is a short sanity check, not full-scale training.
>
> **Hardware**: CPU can train the small model, but runs best in an AirSim environment with GPU.
>
> **AirSim Configuration**: Uses `settings.json`. AirSim must be running.

Building on the hover training above, we train a world model and policy capable of autonomous obstacle avoidance.

Key differences in this task:
- Observation space includes **depth map** features (distance information)
- Reward function must balance **forward progress** and **safety** objectives
- Training is more demanding, requiring more data and longer training time""",
    "6c764ab0": """![DreamerV3 Training Loop](figures/dreamerv3_training_loop.png)

*Figure 8-13: The obstacle avoidance training follows the same loop, but with depth map observations and more complex AirSim interactions*

![Obstacle Avoidance Task Scene](figures/avoidance_scene.png)

*Figure 8-14: Obstacle avoidance task scene -- the drone must navigate autonomously between buildings and facilities*

![Depth Map](figures/depth_scene.png)

*Figure 8-15: The depth map is the key input for obstacle avoidance -- bright pixels indicate far objects, dark pixels indicate near objects*""",
    "cell-1": """import sys
sys.path.append('../external-libraries')
sys.path.append('.')

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import airsim
import os, time
from collections import deque

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

client = airsim.MultirotorClient()
client.confirmConnection()
DRONE = "Drone1"
FORWARD_DIR = np.array([1.0, 0.0, 0.0])
START_POS = np.array([0.0, 0.0, -8.0])""",
    "cbbc4b29": """## 8.6.0 Quick Sanity Check

If you don't have a GPU, or AirSim is unstable, use this section to:

1. Run 1-2 short episodes to verify the environment connection, reward function, and depth map are working
2. Understand the training process and pipeline

You don't need to run all 500 iterations, because the real bottleneck isn't the neural network -- it's AirSim's long-running stability.""",
    "cell-2": """## 8.6.1 State Representation for Obstacle Avoidance

The obstacle avoidance task requires distance information. We augment the state vector with depth map statistics (mean/min for left/center/right regions).""",
    "cell-3": """def get_depth_features(client, drone_id):
    \"\"\"Extract obstacle avoidance features from depth map (6-dim).\"\"\"
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True)
    ], vehicle_name=drone_id)
    if responses and responses[0].width > 0:
        depth = airsim.list_to_2d_float_array(
            responses[0].image_data_float, responses[0].width, responses[0].height
        )
        depth = np.clip(depth, 0, 50)
        h, w = depth.shape
        left = depth[:,:w//3]
        center = depth[:, w//3:2*w//3]
        right = depth[:, 2*w//3:]
        return np.array([
            left.mean() / 50, center.mean() / 50, right.mean() / 50,
            left.min() / 50, center.min() / 50, right.min() / 50,
        ], dtype=np.float32)
    return np.ones(6, dtype=np.float32)

def get_full_state(client, drone_id):
    \"\"\"Get full state: position(3) + velocity(3) + depth features(6) = 12-dim.\"\"\"
    ms = client.getMultirotorState(vehicle_name=drone_id)
    pos = ms.kinematics_estimated.position
    vel = ms.kinematics_estimated.linear_velocity
    motion = np.array([pos.x_val, pos.y_val, pos.z_val,
                       vel.x_val, vel.y_val, vel.z_val], dtype=np.float32)
    depth_feat = get_depth_features(client, drone_id)
    return np.concatenate([motion, depth_feat])

print(f"State dimension: 12 (position 3 + velocity 3 + depth features 6)")""",
    "8cabe524": """### Quick Verification

The following is not training -- it's confirming that:
- AirSim can return depth maps
- State vector is correct (12-dim)
- The drone doesn't crash on takeoff

If any of these fail, there's no point continuing to training.""",
    "fc7b37a2": """# Quick sanity check: takeoff -> read state -> read depth map -> done
client.reset()
client.enableApiControl(True, vehicle_name=DRONE)
client.armDisarm(True, vehicle_name=DRONE)
client.takeoffAsync(vehicle_name=DRONE).join()
client.moveToPositionAsync(*START_POS, 3, vehicle_name=DRONE).join()
time.sleep(0.5)

state = get_full_state(client, DRONE)
print(f"State shape: {state.shape}")
print(f"Position + Velocity: {state[:6]}")
print(f"Depth features: {state[6:]}")
print("Sanity check OK")""",
    "cell-4": """## 8.6.2 Data Collection and Training Loop

The process is similar to hover training, but with a larger state space and a different reward function.""",
    "cell-5": """class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    def add(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))
    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        return tuple(torch.tensor(np.array([b[i] for b in batch]), dtype=torch.float32)
                     for i in range(5))
    def __len__(self):
        return len(self.buffer)

buffer = ReplayBuffer()

def collect_avoidance_episode(client, policy_fn, buffer, max_steps=300):
    client.reset()
    client.enableApiControl(True, vehicle_name=DRONE)
    client.armDisarm(True, vehicle_name=DRONE)
    client.takeoffAsync(vehicle_name=DRONE).join()
    client.moveToPositionAsync(*START_POS, 3, vehicle_name=DRONE).join()
    time.sleep(0.5)

    prev_pos = START_POS.copy()
    total_reward = 0
    collided = False

    for step in range(max_steps):
        state = get_full_state(client, DRONE)
        action = policy_fn(state)

        client.moveByRollPitchYawrateThrottleAsync(
            float(action[1])*0.3, float(action[0])*0.3,
            float(action[2])*0.5, float(action[3])*0.5+0.5,
            duration=0.1, vehicle_name=DRONE
        ).join()

        next_state = get_full_state(client, DRONE)
        pos = next_state[:3]

        collision = client.simGetCollisionInfo(vehicle_name=DRONE)
        done = collision.has_collided or abs(pos[2]) < 0.5 or pos[2] < -30

        if done:
            reward = -100.0
            collided = collision.has_collided
        else:
            forward = np.dot(pos - prev_pos, FORWARD_DIR)
            reward = forward * 2.0
            if next_state[9] < 0.1:  # center_min_depth < 5m
                reward -= 1.0

        buffer.add(state, action, reward, next_state, float(done))
        total_reward += reward
        prev_pos = pos.copy()
        if done:
            break

    forward_dist = np.dot(pos - START_POS, FORWARD_DIR)
    return total_reward, step+1, forward_dist, collided

# Collect initial data
random_policy = lambda s: np.random.randn(4).astype(np.float32) * 0.3
print("Collecting initial data...")
for ep in range(30):
    r, steps, fwd, col = collect_avoidance_episode(client, random_policy, buffer)
    if (ep+1) % 10 == 0:
        print(f"  Ep {ep+1}: reward={r:.0f}, steps={steps}, forward={fwd:.1f}m, collision={col}")
print(f"Replay buffer size: {len(buffer)} samples")""",
    "cell-6": """from world_model_tools import SimpleWorldModel

# Obstacle avoidance world model (state 12-dim, action 4-dim)
world_model = SimpleWorldModel(state_dim=12, action_dim=4, hidden_dim=128).to(device)
wm_opt = torch.optim.Adam(world_model.parameters(), lr=3e-4)

# Policy network
policy = nn.Sequential(
    nn.Linear(12, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 4), nn.Tanh(),
).to(device)
pol_opt = torch.optim.Adam(policy.parameters(), lr=1e-4)

# Training loop
wm_losses, pol_rewards = [], []
print("Starting training...")

for iteration in range(500):
    # Train world model
    s, a, r, ns, d = buffer.sample(min(256, len(buffer)))
    s, a, r, ns = s.to(device), a.to(device), r.unsqueeze(1).to(device), ns.to(device)
    ps, pr = world_model(s, a)
    wm_loss = nn.MSELoss()(ps, ns) + nn.MSELoss()(pr, r)
    wm_opt.zero_grad(); wm_loss.backward(); wm_opt.step()
    wm_losses.append(wm_loss.item())

    # Train policy in imagination
    s0, _, _, _, _ = buffer.sample(min(64, len(buffer)))
    s0 = s0.to(device)
    total_r = torch.zeros(s0.shape[0], 1, device=device)
    st = s0
    for h in range(10):
        at = policy(st)
        st, rt = world_model(st, at)
        total_r += rt * (0.99 ** h)
    pol_loss = -total_r.mean()
    pol_opt.zero_grad(); pol_loss.backward(); pol_opt.step()
    pol_rewards.append(-pol_loss.item())

    if (iteration+1) % 100 == 0:
        print(f"  Iter {iteration+1}: wm_loss={wm_loss.item():.4f}, imagined_reward={-pol_loss.item():.2f}")

print("Training complete!")""",
    "cell-7": """# Training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(wm_losses); axes[0].set_title('World Model Loss'); axes[0].grid(True, alpha=0.3)
axes[1].plot(pol_rewards); axes[1].set_title('Imagined Reward'); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('avoidance_training_curves.png', dpi=150); plt.show()""",
    "cell-8": """## 8.6.3 Save Model and Validate with Inference""",
    "cell-9": """# Save model
os.makedirs('models/avoidance_checkpoint', exist_ok=True)
torch.save({
    'world_model': world_model.state_dict(),
    'policy': [p.data for p in policy.parameters()],
}, 'models/avoidance_checkpoint/model.pt')
print("Model saved")

# Validate with inference
def trained_avoidance_policy(state):
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        return policy(s).cpu().numpy()[0]

print("\\nRunning inference validation...")
r, steps, fwd, col = collect_avoidance_episode(
    client, trained_avoidance_policy, ReplayBuffer(1), max_steps=300
)
print(f"Result: reward={r:.0f}, steps={steps}, forward={fwd:.1f}m, collision={col}")""",
    "cell-10": """## 8.6.4 Evaluation Metrics

Key metrics for the obstacle avoidance task:

| Metric | Meaning | Target |
|--------|---------|--------|
| Collision rate | Fraction of episodes ending in collision | Lower is better |
| Forward distance | Distance traveled forward per episode | Higher is better |
| Survival time | Steps per episode | Longer is better |

A good obstacle avoidance policy should maintain a low collision rate while maximizing forward progress.""",
    "cell-11": """# Multi-run evaluation
n_eval = 5
results = []
print(f"Running {n_eval} evaluation episodes...")
for i in range(n_eval):
    r, steps, fwd, col = collect_avoidance_episode(
        client, trained_avoidance_policy, ReplayBuffer(1), 300
    )
    results.append({'reward': r, 'steps': steps, 'forward': fwd, 'collision': col})
    print(f"  Run {i+1}: forward={fwd:.1f}m, steps={steps}, collision={col}")

collision_rate = sum(1 for r in results if r['collision']) / n_eval
avg_forward = np.mean([r['forward'] for r in results])
avg_steps = np.mean([r['steps'] for r in results])
print(f"\\nEvaluation Results:")
print(f"  Collision rate: {collision_rate*100:.0f}%")
print(f"  Avg forward distance: {avg_forward:.1f}m")
print(f"  Avg survival time: {avg_steps:.0f} steps")""",
    "cell-12": """## 8.6.5 Summary

This completes the obstacle avoidance training pipeline. Comparing the two tasks:

| Aspect | Hover (8.5) | Obstacle Avoidance (8.6) |
|--------|-------------|--------------------------|
| State dimension | 6 (position + velocity) | 12 (+ depth features) |
| Objective | Stay still | Navigate forward while avoiding obstacles |
| Reward | Single target (distance to target) | Multi-objective (forward progress + safety) |
| Training difficulty | Low | High |
| AirSim load | Low | High (depth map each step) |

### Running Without AirSim

If you cannot run AirSim, this notebook still serves as a **training process reference**:
- The sanity check section above verifies the setup
- The training loop structure is complete
- You don't need to run all iterations

### Offline Data Solution

Since AirSim can be unstable over long runs, you can use an offline data approach:

1. Collect short batches of `(state, action, reward, next_state)` data in AirSim and save to `.npz` files
2. Train without connecting to AirSim by loading data from `.npz` files
3. This separates "data collection" from "training," improving stability

### Future Improvements

1. Use a CNN encoder to process raw depth maps instead of hand-crafted features
2. Use RSSM instead of MLP for the world model
3. Iterative data collection (collect -> train -> collect -> train... in a loop)
4. Curriculum Learning: start training in simple environments, then gradually increase difficulty
5. Expand training data to reduce dependence on AirSim stability""",
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
