#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Translate notebook 5: train hover."""
import json

filepath = "C:/Users/gpx/Desktop/airgpt/io_airgpt/8-world_model/5-train_hover.ipynb"
with open(filepath, 'r', encoding='utf-8') as f:
    nb = json.load(f)

translations = {
    "cell-0": """# 8.5 Experiment 3: Training the Hover Model from Scratch

> **Hardware Requirements**: The simplified model in this experiment has very few parameters (~10K), so CPU is sufficient -- no GPU needed. An Intel Ultra 5 / i5 class processor can complete it in a few minutes.
>
> **AirSim Configuration**: Uses `settings.json`. AirSim must be running.

This section trains a simplified world model from scratch to complete the hover task. We'll go through the complete training pipeline and observe an important phenomenon firsthand -- **Model Bias**, one of the most fundamental challenges in model-based reinforcement learning.""",
    "ab571ec1": """![DreamerV3 Training Loop](figures/dreamerv3_training_loop.png)

*Figure 8-11: This section walks through the complete training loop -- from data collection to training a policy in imagination*

![Hover Task Scene](figures/hover_scene.png)

*Figure 8-12: The hover task scene -- seemingly simple, but requires continuous precise control to counteract gravity and disturbances*""",
    "cell-1": """import sys
sys.path.append('../external-libraries')
sys.path.append('.')

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os, time
from collections import deque

device = torch.device('cpu')  # Simplified model runs fine on CPU
print(f"Device: {device}")
print(f"PyTorch version: {torch.__version__}")""",
    "cell-2": """## 8.5.1 Replay Buffer

Training a world model requires large amounts of interaction data. The Replay Buffer stores historical interactions for random sampling during training.""",
    "cell-3": """class ReplayBuffer:
    \"\"\"Simple replay buffer.\"\"\"
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
        actions = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32)
        rewards = torch.tensor(np.array([b[2] for b in batch]), dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32)
        dones = torch.tensor(np.array([b[4] for b in batch]), dtype=torch.float32).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

buffer = ReplayBuffer()
print(f"Replay buffer capacity: {buffer.buffer.maxlen}")""",
    "cell-4": """## 8.5.2 Data Collection

First, we collect interaction data in AirSim using a random policy to fill the replay buffer.""",
    "cell-5": """import airsim

client = airsim.MultirotorClient()
client.confirmConnection()
DRONE = "Drone1"
TARGET = np.array([0.0, 0.0, -5.0])

def collect_episode(client, policy_fn, buffer, max_steps=200):
    \"\"\"Collect data for one episode.\"\"\"
    client.reset()
    client.enableApiControl(True, vehicle_name=DRONE)
    client.armDisarm(True, vehicle_name=DRONE)
    client.takeoffAsync(vehicle_name=DRONE).join()
    client.moveToPositionAsync(TARGET[0], TARGET[1], TARGET[2], 3, vehicle_name=DRONE).join()
    time.sleep(0.5)

    total_reward = 0
    for step in range(max_steps):
        ms = client.getMultirotorState(vehicle_name=DRONE)
        pos = ms.kinematics_estimated.position
        vel = ms.kinematics_estimated.linear_velocity
        state = np.array([pos.x_val, pos.y_val, pos.z_val,
                          vel.x_val, vel.y_val, vel.z_val], dtype=np.float32)

        action = policy_fn(state)
        client.moveByRollPitchYawrateThrottleAsync(
            float(action[1])*0.3, float(action[0])*0.3,
            float(action[2])*0.5, float(action[3])*0.5+0.5,
            duration=0.1, vehicle_name=DRONE
        ).join()

        ms2 = client.getMultirotorState(vehicle_name=DRONE)
        p2 = ms2.kinematics_estimated.position
        v2 = ms2.kinematics_estimated.linear_velocity
        next_state = np.array([p2.x_val, p2.y_val, p2.z_val,
                               v2.x_val, v2.y_val, v2.z_val], dtype=np.float32)

        dist = np.linalg.norm(next_state[:3] - TARGET)
        collision = client.simGetCollisionInfo(vehicle_name=DRONE)
        done = collision.has_collided or dist > 15
        reward = max(0, 1.0 - dist / 5.0)
        if done:
            reward = -10.0

        buffer.add(state, action, reward, next_state, float(done))
        total_reward += reward
        if done:
            break

    return total_reward, step + 1

# Collect initial data
random_policy = lambda s: np.random.randn(4).astype(np.float32) * 0.3
print("Collecting initial data (random policy)...")
for ep in range(20):
    r, steps = collect_episode(client, random_policy, buffer)
    if (ep+1) % 5 == 0:
        print(f"  Episode {ep+1}: reward={r:.1f}, steps={steps}, buffer={len(buffer)}")
print(f"Data collection complete, replay buffer size: {len(buffer)}")""",
    "cell-6": """## 8.5.3 Training the World Model

Train the world model using collected data: learn the mapping (state, action) -> (next_state, reward).""",
    "cell-7": """from world_model_tools import SimpleWorldModel

world_model = SimpleWorldModel(state_dim=6, action_dim=4, hidden_dim=128).to(device)
wm_optimizer = torch.optim.Adam(world_model.parameters(), lr=3e-4)

wm_losses = []
print("Training world model...")
for epoch in range(200):
    states, actions, rewards, next_states, dones = buffer.sample(min(256, len(buffer)))
    states, actions = states.to(device), actions.to(device)
    rewards, next_states = rewards.to(device), next_states.to(device)

    pred_states, pred_rewards = world_model(states, actions)
    loss = nn.MSELoss()(pred_states, next_states) + nn.MSELoss()(pred_rewards, rewards)

    wm_optimizer.zero_grad()
    loss.backward()
    wm_optimizer.step()
    wm_losses.append(loss.item())

    if (epoch+1) % 50 == 0:
        print(f"  Epoch {epoch+1}: loss={loss.item():.6f}")

plt.figure(figsize=(8, 3))
plt.plot(wm_losses)
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('World Model Training Loss'); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()""",
    "cell-8": """## 8.5.4 Training the Policy Network (Actor)

Train the policy network in the world model's "imagination": roll out virtual trajectories using the world model and optimize the policy to maximize cumulative reward.""",
    "cell-9": """class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=6, action_dim=4, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh(),
        )
    def forward(self, state):
        return self.net(state)

policy = PolicyNetwork().to(device)
policy_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

policy_losses = []
print("Training policy network in imagination...")
for epoch in range(300):
    states, _, _, _, _ = buffer.sample(min(64, len(buffer)))
    states = states.to(device)

    # Roll out H steps in imagination
    H = 10
    total_reward = torch.zeros(states.shape[0], 1, device=device)
    s = states
    for h in range(H):
        a = policy(s)
        s, r = world_model(s, a)
        total_reward += r * (0.99 ** h)

    loss = -total_reward.mean()  # Maximize reward = minimize negative reward
    policy_optimizer.zero_grad()
    loss.backward()
    policy_optimizer.step()
    policy_losses.append(loss.item())

    if (epoch+1) % 100 == 0:
        print(f"  Epoch {epoch+1}: imagined_reward={-loss.item():.3f}")

plt.figure(figsize=(8, 3))
plt.plot([-l for l in policy_losses])
plt.xlabel('Epoch'); plt.ylabel('Imagined Reward')
plt.title('Policy Network Training (Cumulative Reward in Imagination)'); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()""",
    "cell-10": """## 8.5.5 Save Checkpoint and Validate with Inference""",
    "cell-11": """# Save model
os.makedirs('models/hover_checkpoint', exist_ok=True)
torch.save({
    'world_model': world_model.state_dict(),
    'policy': policy.state_dict(),
}, 'models/hover_checkpoint/model.pt')
print("Model saved: models/hover_checkpoint/model.pt")

# Run inference in AirSim with the trained policy
def trained_policy(state):
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        a = policy(s).cpu().numpy()[0]
    return a

print("\\nValidating trained policy in AirSim...")
r, steps = collect_episode(client, trained_policy, ReplayBuffer(1), max_steps=200)
print(f"Inference result: reward={r:.1f}, steps={steps}")""",
    "cell-12": """## 8.5.6 Observation: Model Bias

You may notice a counterintuitive phenomenon: **the trained policy performs well in "imagination" (reward keeps rising), but in real AirSim it may actually perform worse than a random policy!**

This is the most fundamental challenge in model-based RL -- **Model Bias**:

| Problem | Cause | Consequence |
|---------|-------|-------------|
| Insufficient data | Only a few hundred samples, world model is inaccurate | Model's predicted "future" doesn't match the real environment |
| Overfitting in imagination | Policy finds "shortcuts" in the inaccurate model | These shortcuts don't exist in the real environment |
| Distribution shift | Policy generates states outside the training data range | Model predictions are completely wrong for unseen states |

It's like someone who learned to fly in a dream -- the physics in the dream differ from reality, and upon waking they discover they can't actually fly.

### Solutions

DreamerV3 mitigates model bias through several mechanisms:

1. **Continuous data collection**: Constantly collect new data with the latest policy so the world model covers more states
2. **RSSM architecture**: More powerful sequence modeling than MLP, yielding more accurate predictions
3. **Short imagination horizon**: Only roll out short trajectories in imagination (15 steps) to reduce error accumulation
4. **Model ensembles**: Train multiple world models and use their disagreement to estimate uncertainty""",
    "5a272ed5": """## 8.5.7 Alternative Approach: MPC (Model Predictive Control)

Rather than training a policy network (which can easily overfit to an inaccurate model), we can use the world model for **online planning** at each step: sample multiple candidate actions, use the world model to predict short-term consequences of each, and select the one with the highest predicted reward.

This is the MPC (Model Predictive Control) approach -- instead of relying on an offline-trained policy, it plans in real time.""",
    "ef9919f2": """# MPC policy: sample N candidate actions per step, evaluate with world model, select optimal
def mpc_policy(state, world_model, n_candidates=100, horizon=3):
    \"\"\"Model Predictive Control: online planning.\"\"\"
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # (1, 6)
    best_action = None
    best_reward = -float('inf')

    # Sample N candidate actions
    candidate_actions = torch.randn(n_candidates, 4) * 0.5  # (N, 4)

    with torch.no_grad():
        for i in range(n_candidates):
            action = candidate_actions[i:i+1]  # (1, 4)
            s = state_t.clone()
            total_r = 0
            # Look ahead for horizon steps
            for h in range(horizon):
                s, r = world_model(s, action)
                total_r += r.item() * (0.99 ** h)
            if total_r > best_reward:
                best_reward = total_r
                best_action = candidate_actions[i].numpy()

    return np.clip(best_action, -1, 1)

# Run MPC policy in AirSim
print("MPC policy (world model online planning)...")
mpc_fn = lambda s: mpc_policy(s, world_model, n_candidates=80, horizon=3)
r_mpc, steps_mpc = collect_episode(client, mpc_fn, ReplayBuffer(1), max_steps=200)

# Compare with random policy
print("Random policy (baseline)...")
r_rand, steps_rand = collect_episode(client, random_policy, ReplayBuffer(1), max_steps=200)

print(f"\\n===== Comparison Results =====")
print(f"  Random policy:  reward={r_rand:.1f}, steps={steps_rand}")
print(f"  Trained policy: reward={r:.1f}, steps={steps}")
print(f"  MPC policy:     reward={r_mpc:.1f}, steps={steps_mpc}")
print(f"\\nMPC avoids the overfitting problem of the policy network through online planning")""",
    "26b69693": """## 8.5.8 Summary

Key takeaways from this section:

**Training Pipeline**: Data collection -> Train world model -> Train policy in imagination -> Validate with inference

**Key Finding -- Model Bias**:
- The policy's reward in "imagination" keeps rising, appearing to train successfully
- But in real AirSim, the trained policy may perform worse than random
- Reason: The world model isn't accurate enough, and the policy "overfits" to inaccurate imagination

**Two Ways to Use a World Model**:

| Approach | Principle | Model Accuracy Requirement |
|----------|-----------|---------------------------|
| Policy Network (Actor) | Train offline, deploy directly | High (needs accurate long-horizon prediction) |
| MPC (Online Planning) | Sample + evaluate candidate actions at each step | Lower (only needs short-horizon prediction) |

**From Simplified Model to DreamerV3**: The MLP world model used in this section is the most simplified version. The full DreamerV3 uses RSSM architecture, image encoder, value network, symlog transform, and other techniques that dramatically improve world model accuracy, making policy network training truly effective.""",
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
