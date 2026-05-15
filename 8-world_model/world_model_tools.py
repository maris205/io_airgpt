# -*- coding: utf-8 -*-
"""world_model_tools.py -- Shared utilities for the world model course.

Contains: world model implementations, DreamerV3 inference interface/API, helper tools.
"""
import sys
sys.path.append('../external-libraries')

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Simple World Model (used in Notebook 1)
# ============================================================

class SimpleWorldModel(nn.Module):
    """Simple world model: MLP that predicts the next state.

    Input: current state s (dim=state_dim) + action a (dim=action_dim)
    Output: next state s' (dim=state_dim) + reward r (dim=1)
    """

    def __init__(self, state_dim=6, action_dim=4, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.state_head = nn.Linear(hidden_dim, state_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        h = self.net(x)
        next_state = self.state_head(h)
        reward = self.reward_head(h)
        return next_state, reward


# ============================================================
# RSSM (used in Notebook 2)
# ============================================================

class SimpleRSSM(nn.Module):
    """Recurrent State-Space Model.

    Deterministic state h: GRU (memory)
    Stochastic state s: Gaussian distribution sampling (represents uncertainty)
    """

    def __init__(self, obs_dim=32, action_dim=4, h_dim=64, s_dim=32):
        super().__init__()
        self.h_dim = h_dim
        self.s_dim = s_dim

        self.gru = nn.GRUCell(obs_dim + action_dim, h_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(h_dim, 64), nn.ReLU(),
            nn.Linear(64, s_dim * 2),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(h_dim + obs_dim, 64), nn.ReLU(),
            nn.Linear(64, s_dim * 2),
        )

    def initial_state(self, batch_size=1):
        h = torch.zeros(batch_size, self.h_dim)
        s = torch.zeros(batch_size, self.s_dim)
        return h, s

    def forward_prior(self, h):
        params = self.prior_net(h)
        mean, logstd = params.chunk(2, dim=-1)
        std = torch.exp(logstd.clamp(-5, 2))
        s = mean + std * torch.randn_like(std)
        return s, mean, std

    def forward_posterior(self, h, obs):
        params = self.posterior_net(torch.cat([h, obs], dim=-1))
        mean, logstd = params.chunk(2, dim=-1)
        std = torch.exp(logstd.clamp(-5, 2))
        s = mean + std * torch.randn_like(std)
        return s, mean, std

    def step(self, h, s, action, obs):
        x = torch.cat([s, action], dim=-1)
        h_new = self.gru(x, h)
        s_prior, prior_mean, prior_std = self.forward_prior(h_new)
        s_post, post_mean, post_std = self.forward_posterior(h_new, obs)
        return h_new, s_post, {
            "prior_mean": prior_mean, "prior_std": prior_std,
            "post_mean": post_mean, "post_std": post_std,
        }
