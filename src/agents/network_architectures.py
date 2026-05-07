"""
network_architectures.py — Actor/Critic Networks for MAPPO
CNN for spatial obs, MLP for state, centralized critic.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


def orthogonal_init(layer: nn.Module, gain: float = np.sqrt(2)) -> nn.Module:
    """Orthogonal initialization for stable training."""
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    return layer


class SpatialEncoder(nn.Module):
    """
    CNN encoder for the local grid view (7x7 grid).
    Captures spatial patterns like nearby obstacles, robots, shelves.
    """

    def __init__(self, view_size: int = 7, num_cell_types: int = 6):
        super().__init__()
        self.view_size = view_size
        self.num_cell_types = num_cell_types

        # Embed each cell type
        self.cell_embed = nn.Embedding(num_cell_types + 1, 8)

        # Conv stack
        self.conv = nn.Sequential(
            orthogonal_init(nn.Conv2d(8, 32, kernel_size=3, padding=1)),
            nn.ReLU(),
            orthogonal_init(nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.output_dim = 64 * view_size * view_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, view_size*view_size) float in [0,1]
        B = x.shape[0]
        # Quantize to cell type indices
        cell_ids = (x * 5).long().clamp(0, self.num_cell_types)
        embedded = self.cell_embed(cell_ids)           # (B, 49, 8)
        embedded = embedded.view(B, self.view_size, self.view_size, 8)
        embedded = embedded.permute(0, 3, 1, 2)        # (B, 8, H, W)
        return self.conv(embedded)


class StateEncoder(nn.Module):
    """MLP encoder for robot state + order queue vectors."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            orthogonal_init(nn.Linear(input_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            orthogonal_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CommunicationEncoder(nn.Module):
    """Encodes communication messages from all agents."""

    def __init__(self, num_agents: int, comm_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.num_agents = num_agents
        self.comm_dim = comm_dim
        # Attention-based aggregation of neighbor messages
        self.msg_proj = orthogonal_init(nn.Linear(comm_dim, hidden_dim))
        self.attention = orthogonal_init(nn.Linear(hidden_dim, 1))
        self.output_dim = hidden_dim

    def forward(self, comm_flat: torch.Tensor) -> torch.Tensor:
        B = comm_flat.shape[0]
        msgs = comm_flat.view(B, self.num_agents, self.comm_dim)
        proj = F.relu(self.msg_proj(msgs))                          # (B, N, H)
        attn_weights = F.softmax(self.attention(proj), dim=1)       # (B, N, 1)
        aggregated = (attn_weights * proj).sum(dim=1)               # (B, H)
        return aggregated


class ActorNetwork(nn.Module):
    """
    Decentralized actor: takes local obs only → action distribution.
    Shared parameters across all agents (+ agent ID embedding).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 8,
        num_agents: int = 10,
        comm_dim: int = 16,
        view_size: int = 7,
        hidden_dim: int = 256,
        agent_embed_dim: int = 8,
    ):
        super().__init__()
        self.view_size = view_size
        self.view_flat_dim = view_size ** 2
        self.state_dim = obs_dim - self.view_flat_dim - num_agents * comm_dim
        self.comm_obs_dim = num_agents * comm_dim

        # Agent ID embedding
        self.agent_embed = nn.Embedding(num_agents, agent_embed_dim)

        # Encoders
        self.spatial_enc = SpatialEncoder(view_size)
        self.state_enc = StateEncoder(self.state_dim + agent_embed_dim)
        self.comm_enc = CommunicationEncoder(num_agents, comm_dim)

        # Fusion MLP
        fusion_dim = (
            self.spatial_enc.output_dim
            + self.state_enc.output_dim
            + self.comm_enc.output_dim
        )
        self.fusion = nn.Sequential(
            orthogonal_init(nn.Linear(fusion_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            orthogonal_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Action head
        self.action_head = orthogonal_init(nn.Linear(hidden_dim, action_dim), gain=0.01)

    def forward(
        self, obs: torch.Tensor, agent_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (action_logits, hidden_features)
        """
        B = obs.shape[0]

        # Split observation
        spatial_obs = obs[:, :self.view_flat_dim]
        state_obs = obs[:, self.view_flat_dim:-self.comm_obs_dim]
        comm_obs = obs[:, -self.comm_obs_dim:]

        # Agent ID embedding
        agent_emb = self.agent_embed(agent_ids)  # (B, embed_dim)

        # Encode
        spatial_feat = self.spatial_enc(spatial_obs)
        state_feat = self.state_enc(torch.cat([state_obs, agent_emb], dim=-1))
        comm_feat = self.comm_enc(comm_obs)

        # Fuse
        fused = torch.cat([spatial_feat, state_feat, comm_feat], dim=-1)
        hidden = self.fusion(fused)
        logits = self.action_head(hidden)

        return logits, hidden


class CentralizedCritic(nn.Module):
    """
    Centralized critic: takes ALL agents' observations concatenated → V(s).
    Only used during training (CTDE paradigm).
    """

    def __init__(
        self,
        obs_dim: int,
        num_agents: int = 10,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_agents = num_agents
        global_dim = obs_dim * num_agents

        self.net = nn.Sequential(
            orthogonal_init(nn.Linear(global_dim, hidden_dim * 2)),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            orthogonal_init(nn.Linear(hidden_dim * 2, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            orthogonal_init(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.ReLU(),
        )
        self.value_head = orthogonal_init(nn.Linear(hidden_dim // 2, 1), gain=1.0)

    def forward(self, all_obs: torch.Tensor) -> torch.Tensor:
        """
        all_obs: (batch, num_agents * obs_dim) — concatenated all agents' obs
        Returns: (batch, 1) — state value estimate
        """
        hidden = self.net(all_obs)
        return self.value_head(hidden)


class CommunicationHead(nn.Module):
    """
    Generates outgoing communication message from actor's hidden state.
    Each agent broadcasts a K-dim message to all others.
    """

    def __init__(self, hidden_dim: int = 256, comm_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            orthogonal_init(nn.Linear(hidden_dim, comm_dim * 2)),
            nn.ReLU(),
            orthogonal_init(nn.Linear(comm_dim * 2, comm_dim)),
            nn.Tanh(),  # bounded messages [-1, 1]
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)
