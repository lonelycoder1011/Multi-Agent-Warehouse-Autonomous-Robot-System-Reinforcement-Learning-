"""
comm_channel.py — Differentiable Inter-Agent Communication
Agents broadcast K-dim messages; others receive aggregated context.
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Dict, Optional


class CommunicationChannel:
    """
    Manages communication messages between agents.
    During rollouts, messages are numpy arrays.
    During training, gradients can flow through the comm head.
    """

    def __init__(self, num_agents: int, comm_dim: int = 16):
        self.num_agents = num_agents
        self.comm_dim = comm_dim
        self.messages = np.zeros((num_agents, comm_dim), dtype=np.float32)

    def reset(self) -> None:
        self.messages = np.zeros((self.num_agents, self.comm_dim), dtype=np.float32)

    def update(self, agent_idx: int, message: np.ndarray) -> None:
        """Store outgoing message for agent."""
        self.messages[agent_idx] = message[:self.comm_dim]

    def get_flat(self) -> np.ndarray:
        """Return all messages as flat array for obs concatenation."""
        return self.messages.flatten()

    def get_messages_for_agent(self, agent_idx: int) -> np.ndarray:
        """Get all OTHER agents' messages (exclude self)."""
        mask = np.ones(self.num_agents, dtype=bool)
        mask[agent_idx] = False
        return self.messages[mask].flatten()

    def broadcast_update(self, messages_dict: Dict[int, np.ndarray]) -> None:
        """Batch update messages from all agents."""
        for idx, msg in messages_dict.items():
            self.update(idx, msg)
