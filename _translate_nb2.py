#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch translate notebook 2 cells."""
import json
import sys

filepath = sys.argv[1]
with open(filepath, 'r', encoding='utf-8') as f:
    nb = json.load(f)

translations = {
    "cell-1": """## 8.2.1 RSSM: Recurrent State-Space Model

RSSM is the core component of DreamerV3's world model. It elegantly combines two complementary modeling approaches:

### Fusion of Two "Brain Halves"

| Component | Analogy | Role | Origin |
|-----------|---------|------|--------|
| Deterministic state $h_t$ | Notebook | Precisely records key historical events | RNN/GRU |
| Stochastic state $s_t$ | Dice | Represents uncertainty about the future | Gaussian sampling |

Why do we need both types of state?

- **Deterministic only** (pure RNN): Cannot express uncertainty like "I'm not sure if what's ahead is a wall or a door"
- **Stochastic only** (pure SSM): No memory, cannot leverage historical information like "I saw a wall 3 seconds ago"
- **Both combined** (RSSM): Has both memory and the ability to express uncertainty

### RSSM Workflow

```
Input: latent features z_t + previous action a_{t-1}
  |
GRU updates deterministic state: h_t = GRU(h_{t-1}, [s_{t-1}, a_{t-1}])
  |
Prior prediction: p(s_t | h_t)          <- guess based on memory alone
Posterior correction: q(s_t | h_t, z_t) <- corrected with actual observation
  |
Full state: [h_t, s_t] -> used to predict rewards, images, etc.
```""",
    "7144f24f": """![RSSM Intuition Diagram](figures/rssm_intuition.png)

*Figure 8-4: RSSM combines "deterministic memory" with "stochastic uncertainty," making it better suited for real physical environments than pure RNN or pure probabilistic models*""",
    "cell-2": """## 8.2.2 The Four Components of DreamerV3

DreamerV3 consists of four modules that form a complete "learn-imagine-decide" loop:

### 1. Visual Encoder
- Input: Raw image $o_t$ (e.g., 64x64 RGB)
- Output: Latent features $z_t$ (e.g., 32-dim vector)
- Role: Compress high-dimensional images into a low-dimensional latent space

### 2. World Model (RSSM)
- Input: Latent features $z_t$ + action $a_t$
- Output: Predicted next state $s_{t+1}$, predicted reward $r_t$
- Role: Simulate environment dynamics in latent space

### 3. Policy Network (Actor)
- Input: Current state $[h_t, s_t]$
- Output: Optimal action $a_t$
- Role: Select the best action based on the world model's "imagination"

### 4. Value Network (Critic)
- Input: Current state $[h_t, s_t]$
- Output: Value estimate $V(s_t)$
- Role: Evaluate the long-term value of the current state to guide policy optimization

### Training Loop

```
Real environment interaction -> Collect data (o, a, r, o') -> Train world model
                                                                    |
                                                              Roll out imagined trajectories
                                                                    |
                                                              Train Actor + Critic
                                                                    |
                                                              Actor outputs action -> Back to real environment
```""",
    "1ebf1de6": """![DreamerV3 Architecture](figures/dreamerv3_architecture.png)

*Figure 8-5: DreamerV3's four components -- Encoder compresses information, World Model simulates the future, Actor selects actions, Critic evaluates value*""",
    "cell-3": """## 8.2.3 Key Innovations in DreamerV3

DreamerV3 introduces several important improvements over its predecessors (DreamerV1/V2):

### 1. Symlog Transform

Reward values can vary enormously (hover reward ~1, collision penalty -100). The symlog transform compresses large values while preserving small ones, making it easier for the network to learn:

$$\\text{symlog}(x) = \\text{sign}(x) \\cdot \\ln(|x| + 1)$$

### 2. Discretized Regression

Traditional methods directly predict continuous values (e.g., reward=3.7). DreamerV3 converts this into a classification problem: discretize the value range into bins and predict the probability of falling into each bin. This makes training more stable.

### 3. No Task-Specific Hyperparameter Tuning

A major selling point of DreamerV3 is "one set of hyperparameters for all tasks" -- from Atari games to robot control, there's no need to adjust learning rates, loss weights, or other hyperparameters for each task. This is very convenient for both teaching and practical applications.""",
    "cell-4": """## 8.2.4 Hands-On: Simplified RSSM Implementation

Below we implement a simplified RSSM in PyTorch and observe the behavior of deterministic and stochastic states.""",
    "cell-5": """import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class SimpleRSSM(nn.Module):
    \"\"\"Simplified RSSM: GRU (deterministic) + Gaussian sampling (stochastic)\"\"\"

    def __init__(self, obs_dim=32, action_dim=4, h_dim=64, s_dim=32):
        super().__init__()
        self.h_dim = h_dim
        self.s_dim = s_dim
        self.gru = nn.GRUCell(s_dim + action_dim, h_dim)
        # Prior network: prediction based on memory alone
        self.prior_net = nn.Sequential(
            nn.Linear(h_dim, 64), nn.ReLU(),
            nn.Linear(64, s_dim * 2),  # Outputs mean + logstd
        )
        # Posterior network: corrected with observation
        self.posterior_net = nn.Sequential(
            nn.Linear(h_dim + obs_dim, 64), nn.ReLU(),
            nn.Linear(64, s_dim * 2),
        )

    def step(self, h, s, action, obs):
        # 1. GRU updates deterministic state
        x = torch.cat([s, action], dim=-1)
        h_new = self.gru(x, h)

        # 2. Prior prediction (memory only)
        prior_params = self.prior_net(h_new)
        prior_mean, prior_logstd = prior_params.chunk(2, dim=-1)
        prior_std = torch.exp(prior_logstd.clamp(-5, 2))

        # 3. Posterior correction (with observation)
        post_params = self.posterior_net(torch.cat([h_new, obs], dim=-1))
        post_mean, post_logstd = post_params.chunk(2, dim=-1)
        post_std = torch.exp(post_logstd.clamp(-5, 2))

        # 4. Sample from posterior distribution
        s_new = post_mean + post_std * torch.randn_like(post_std)

        return h_new, s_new, {
            "prior_mean": prior_mean, "prior_std": prior_std,
            "post_mean": post_mean, "post_std": post_std,
        }

rssm = SimpleRSSM()
print(f"RSSM parameters: {sum(p.numel() for p in rssm.parameters()):,}")
print(f"Deterministic state h dimension: {rssm.h_dim}")
print(f"Stochastic state s dimension: {rssm.s_dim}")""",
    "cell-6": """# Observe RSSM behavior: prior vs posterior
rssm.eval()
with torch.no_grad():
    h = torch.zeros(1, 64)
    s = torch.zeros(1, 32)

    prior_stds = []
    post_stds = []

    for t in range(30):
        action = torch.randn(1, 4) * 0.3
        obs = torch.randn(1, 32)  # Simulated observation
        h, s, info = rssm.step(h, s, action, obs)
        prior_stds.append(info["prior_std"].mean().item())
        post_stds.append(info["post_std"].mean().item())

plt.figure(figsize=(8, 3))
plt.plot(prior_stds, 'r-o', markersize=3, label='Prior uncertainty (memory only)')
plt.plot(post_stds, 'b-o', markersize=3, label='Posterior uncertainty (with observation)')
plt.xlabel('Time Step')
plt.ylabel('Mean Std Dev')
plt.title('RSSM Prior vs Posterior Uncertainty')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print("The prior (red) typically has higher uncertainty than the posterior (blue) -- because observations provide additional information")""",
    "cell-7": """## 8.2.5 DreamerV3 and AirSim Data Pipeline

In subsequent experiments, DreamerV3 and AirSim interact in a closed loop:

```
+-----------------------------------------------------+
|                    Training Loop                      |
|                                                      |
|  AirSim --(o,a,r,o')--> Replay Buffer               |
|    ^                         |                       |
|    |                    Train World Model             |
|    |                         |                       |
|    |                    Roll out imagined trajectories|
|    |                         |                       |
|    |                    Train Actor-Critic            |
|    |                         |                       |
|    +------ Actor outputs action -----+               |
|                                                      |
+-----------------------------------------------------+
```

### Data Format

Each interaction step produces a tuple `(observation, action, reward, next_observation, done)`:

| Field | Source | Dimensions |
|-------|--------|-----------|
| observation | AirSim camera + IMU | Image 64x64x3 + state 6-dim |
| action | Policy network output | 4-dim (pitch, roll, yaw_rate, throttle) |
| reward | Reward function | Scalar |
| done | Collision/timeout detection | Boolean |""",
    "980d25ea": """![DreamerV3 Training Loop](figures/dreamerv3_training_loop.png)

*Figure 8-6: DreamerV3 training loop -- real interactions provide data, the world model provides imagination space, Actor-Critic optimizes within imagination*""",
    "cell-8": """## 8.2.6 Summary

Key takeaways from this section:

1. **RSSM** combines the memory capability of RNNs with the uncertainty representation of probabilistic models
2. DreamerV3 consists of **encoder, world model, policy network, and value network**
3. Key innovations: symlog transform, discretized regression, no task-specific hyperparameter tuning
4. The prior distribution represents "guesses based on memory alone," while the posterior represents "corrections informed by observation"
5. The data pipeline with AirSim forms a "interact -> train -> imagine -> decide" loop

In the next section, we'll use a pretrained DreamerV3 model to perform autonomous hover control in AirSim.""",
}

for cell in nb.get('cells', []):
    cell_id = cell.get('id', '')
    if cell_id in translations:
        new_source = translations[cell_id]
        cell['source'] = [line + '\n' for line in new_source.split('\n')[:-1]] + [new_source.split('\n')[-1]]

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Translated: {filepath}")
