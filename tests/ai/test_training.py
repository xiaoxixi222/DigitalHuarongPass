from pathlib import Path

from ai.train import run_training


def test_tiny_training_writes_checkpoint_and_report(tmp_path: Path):
    result = run_training(
        samples=24,
        epochs=1,
        eval_games=4,
        seed=3,
        output_root=tmp_path,
    )

    assert result.checkpoint.exists()
    assert result.report.exists()
    assert result.metrics.exists()
    assert "supervised" in result.metrics.read_text()
