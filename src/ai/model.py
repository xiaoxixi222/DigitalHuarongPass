from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn


MODEL_VERSION = "neural-solver-v2"
LEGACY_MODEL_VERSIONS = {"neural-solver-v1", MODEL_VERSION}
BOARD_SIZE = [4, 4]
DISTANCE_BUCKETS = 97
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
        hidden_size = 128 if input_size >= 400 else 64
        encoder_width = 256 if input_size >= 400 else 128
        self.encoder = nn.Sequential(
            nn.Linear(input_size, encoder_width),
            nn.ReLU(),
            nn.Linear(encoder_width, hidden_size),
            nn.ReLU(),
        )
        self.residual = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        # Zero initialization keeps legacy checkpoints numerically compatible
        # while allowing eighth-generation training to learn a residual path.
        self.residual_gate = nn.Parameter(torch.zeros(1))
        self.policy_head = nn.Linear(hidden_size, 4)
        self.q_head = nn.Linear(hidden_size, 4)
        self.distance_head = nn.Linear(hidden_size, 1)
        self.distance_bucket_head = nn.Linear(hidden_size, DISTANCE_BUCKETS)
        self.risk_head = nn.Linear(hidden_size, 4)

    def hidden(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(features)
        return encoded + self.residual_gate * self.residual(encoded)

    def forward_full(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.hidden(features)
        return {
            "policy_logits": self.policy_head(hidden),
            "q_values": self.q_head(hidden),
            "distance": self.distance_head(hidden),
            "distance_logits": self.distance_bucket_head(hidden),
            "risk_logits": self.risk_head(hidden),
        }

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.forward_full(features)
        return outputs["policy_logits"], outputs["q_values"]


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
    if payload.get("model_version") not in LEGACY_MODEL_VERSIONS:
        raise ValueError(f"unsupported model version: {payload.get('model_version')!r}")
    if payload.get("action_order") != ["UP", "DOWN", "LEFT", "RIGHT"]:
        raise ValueError("checkpoint action order does not match the runtime")
    if payload.get("board_size") != BOARD_SIZE:
        raise ValueError("checkpoint board size does not match the runtime")
    if payload.get("normalization") != NORMALIZATION:
        raise ValueError("checkpoint normalization does not match the runtime")
    model = HuarongNet(input_size=int(payload.get("input_size", 260)))
    model.load_state_dict(payload["state_dict"], strict=False)
    model.to(device)
    model.eval()
    metadata = dict(payload.get("metadata", {}))
    metadata.setdefault("model_version", payload["model_version"])
    return model, metadata


def save_ensemble_checkpoint(
    path: str | Path,
    models: Sequence[HuarongNet],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save every fold inside one portable checkpoint."""
    if not models:
        raise ValueError("an ensemble checkpoint requires at least one model")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fold_state_dicts = [
        {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        for model in models
    ]
    first = models[0]
    payload = {
        "model_version": MODEL_VERSION,
        "checkpoint_kind": "embedded-ensemble",
        "input_size": first.input_size,
        "board_size": BOARD_SIZE,
        "normalization": NORMALIZATION,
        "action_order": ["UP", "DOWN", "LEFT", "RIGHT"],
        "fold_state_dicts": fold_state_dicts,
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_ensemble_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[tuple[HuarongNet, ...], dict[str, Any]]:
    """Load an embedded ensemble without consulting sibling files."""
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("checkpoint_kind") != "embedded-ensemble":
        raise ValueError("checkpoint does not contain an embedded ensemble")
    if payload.get("model_version") not in LEGACY_MODEL_VERSIONS:
        raise ValueError(f"unsupported model version: {payload.get('model_version')!r}")
    if payload.get("action_order") != ["UP", "DOWN", "LEFT", "RIGHT"]:
        raise ValueError("checkpoint action order does not match the runtime")
    if payload.get("board_size") != BOARD_SIZE:
        raise ValueError("checkpoint board size does not match the runtime")
    if payload.get("normalization") != NORMALIZATION:
        raise ValueError("checkpoint normalization does not match the runtime")
    fold_state_dicts = payload.get("fold_state_dicts")
    if not isinstance(fold_state_dicts, list) or not fold_state_dicts:
        raise ValueError("checkpoint has no embedded fold state dicts")
    models: list[HuarongNet] = []
    for state_dict in fold_state_dicts:
        model = HuarongNet(input_size=int(payload.get("input_size", 260)))
        model.load_state_dict(state_dict, strict=False)
        model.to(device).eval()
        models.append(model)
    metadata = dict(payload.get("metadata", {}))
    metadata.setdefault("model_version", payload["model_version"])
    metadata.setdefault("folds", len(models))
    return tuple(models), metadata
