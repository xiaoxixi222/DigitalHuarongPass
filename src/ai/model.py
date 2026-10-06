from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


MODEL_VERSION = "neural-solver-v1"
BOARD_SIZE = [4, 4]
NORMALIZATION = {
    "blank_row": "divide_by_3",
    "blank_col": "divide_by_3",
    "manhattan": "divide_by_48",
    "linear_conflict": "divide_by_24",
}


class HuarongNet(nn.Module):
    def __init__(self, input_size: int = 260):
        super().__init__()
        self.input_size = input_size
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(64, 4)
        self.q_head = nn.Linear(64, 4)

    def hidden(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.hidden(features)
        return self.policy_head(hidden), self.q_head(hidden)


def save_checkpoint(
    path: str | Path,
    model: HuarongNet,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_version": MODEL_VERSION,
        "input_size": model.input_size,
        "board_size": BOARD_SIZE,
        "normalization": NORMALIZATION,
        "action_order": ["UP", "DOWN", "LEFT", "RIGHT"],
        "state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[HuarongNet, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("model_version") != MODEL_VERSION:
        raise ValueError(f"unsupported model version: {payload.get('model_version')!r}")
    if payload.get("action_order") != ["UP", "DOWN", "LEFT", "RIGHT"]:
        raise ValueError("checkpoint action order does not match the runtime")
    if payload.get("board_size") != BOARD_SIZE:
        raise ValueError("checkpoint board size does not match the runtime")
    if payload.get("normalization") != NORMALIZATION:
        raise ValueError("checkpoint normalization does not match the runtime")
    model = HuarongNet(input_size=int(payload.get("input_size", 260)))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    metadata = dict(payload.get("metadata", {}))
    metadata.setdefault("model_version", payload["model_version"])
    return model, metadata
