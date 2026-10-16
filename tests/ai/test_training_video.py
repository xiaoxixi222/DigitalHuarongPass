from pathlib import Path

from ai.agent import NeuralAgent
from ai.encoding import goal_state
from ai.model import HuarongNet
from ai.training_video import (
    encode_training_video,
    render_comparison_frames,
    render_training_metrics,
)


def test_render_training_metrics_writes_png(tmp_path: Path):
    output = render_training_metrics(
        {
            "history": [
                {"train_loss": 1.0, "validation_accuracy": 0.1},
                {"train_loss": 0.5, "validation_accuracy": 0.8},
            ],
            "evaluation": {"solve_rate": 0.5},
        },
        tmp_path / "metrics.png",
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_render_comparison_frames_writes_png(tmp_path: Path):
    state = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, -1, 15)
    count = render_comparison_frames(
        NeuralAgent(HuarongNet()),
        NeuralAgent(HuarongNet()),
        [state],
        tmp_path / "frames",
        max_steps=1,
    )

    assert count == 1
    assert (tmp_path / "frames" / "frame-000000.png").exists()


def test_encode_training_video_returns_false_without_encoder(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("ai.training_video.shutil.which", lambda name: None)

    assert encode_training_video(tmp_path, tmp_path / "out.mp4") is False

