import os
import numpy as np
import torch
import torch.nn as nn
from types import SimpleNamespace

from agents.agent_base import BaseAgent
from environments.collector.state import EnvState


# Actions: 0=up, 1=right, 2=down, 3=left (matches env.py)
DIRS = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)], dtype=np.int32)


def preprocess(obs, grid_w=16, grid_h=16):
    """Turn the dict observation into a flat feature vector (24 floats).

    Features:
      [0:2]    self position (normalized)
      [2:4]    opponent position (normalized)
      [4:6]    offset to nearest item (dy, dx) normalized
      [6:10]   normalized distance to nearest item in each of UP/RIGHT/DOWN/LEFT
               half-planes (1.0 if no item in that direction)
      [10:14]  obstacle flag for the 4 neighbour tiles (1=blocked, 0=walkable)
      [14:23]  3x3 patch around the agent: 0=empty, 0.5=item, 1=wall/edge
      [23]     step counter / 1000
    """
    tile_map = obs["map_features"]["tile_type"]
    pos = obs["units"]["position"][0].astype(np.int32)
    opp = obs["units"]["position"][1].astype(np.int32)
    y, x = int(pos[0]), int(pos[1])
    H, W = tile_map.shape
    norm = np.array([grid_h, grid_w], dtype=np.float32)

    self_pos = pos.astype(np.float32) / norm
    opp_pos = opp.astype(np.float32) / norm

    # Nearest item (relative offset + cardinal distances)
    items = np.argwhere(tile_map == 2)
    if len(items) > 0:
        d = np.abs(items - pos).sum(axis=1)
        closest = items[np.argmin(d)].astype(np.float32)
        item_rel = (closest - pos.astype(np.float32)) / norm

        # Distance to nearest item in each direction
        dy = items[:, 0] - y
        dx = items[:, 1] - x
        manh = np.abs(dy) + np.abs(dx)
        max_d = float(H + W)
        card = np.ones(4, dtype=np.float32)
        if (dy < 0).any():
            card[0] = manh[dy < 0].min() / max_d   # UP
        if (dx > 0).any():
            card[1] = manh[dx > 0].min() / max_d   # RIGHT
        if (dy > 0).any():
            card[2] = manh[dy > 0].min() / max_d   # DOWN
        if (dx < 0).any():
            card[3] = manh[dx < 0].min() / max_d   # LEFT
    else:
        item_rel = np.zeros(2, dtype=np.float32)
        card = np.ones(4, dtype=np.float32)

    # Neighbouring obstacles
    obstacles = np.ones(4, dtype=np.float32)
    for i, (dy_, dx_) in enumerate(DIRS):
        ny, nx = y + int(dy_), x + int(dx_)
        if 0 <= ny < H and 0 <= nx < W:
            obstacles[i] = 1.0 if tile_map[ny, nx] == 1 else 0.0

    # 3x3 patch (out-of-map and walls = 1, items = 0.5, empty = 0)
    patch = np.ones(9, dtype=np.float32)
    k = 0
    for dy_ in (-1, 0, 1):
        for dx_ in (-1, 0, 1):
            ny, nx = y + dy_, x + dx_
            if dy_ == 0 and dx_ == 0:
                patch[k] = 0.0
            elif 0 <= ny < H and 0 <= nx < W:
                t = int(tile_map[ny, nx])
                patch[k] = 1.0 if t == 1 else (0.5 if t == 2 else 0.0)
            k += 1

    step_norm = np.array([float(obs["steps"]) / 1000.0], dtype=np.float32)

    return np.concatenate([self_pos, opp_pos, item_rel, card,
                           obstacles, patch, step_norm]).astype(np.float32)


class DQN_Network(nn.Module):
    def __init__(self, obs_size=24, n_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class Agent(BaseAgent):
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.config = config
        self.epsilon = getattr(config, "epsilon", 0.0)
        self.seed = getattr(config, "seed", 0)
        self.action_space = getattr(config, "action_space", 4)
        self.grid_w = getattr(config, "grid_w", 16)
        self.grid_h = getattr(config, "grid_h", 16)
        np.random.seed(self.seed)
        self.network = DQN_Network()

    def load(self) -> None:
        path = os.path.join(self.config.weights_dir, "dqn.pth")
        self.network.load_state_dict(torch.load(path, map_location="cpu"))
        self.network.eval()

    def act(self, observation: EnvState) -> int:
        # Small chance of random action at eval time, useful against deterministic opponents
        if self.epsilon > 0 and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_space))

        state = preprocess(observation, self.grid_w, self.grid_h)
        with torch.no_grad():
            q = self.network(torch.FloatTensor(state).unsqueeze(0))
        return int(q.argmax().item())
