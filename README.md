**清华附中计算机高研作品**

## Neural Solver

The project includes an optional PyTorch solver that learns from A* decisions and can be fine-tuned with DQN. Install the locked dependencies with:

```bash
uv sync
```

Train the supervised policy and generate a checkpoint plus a Markdown report:

```bash
uv run python -m ai.train --stage imitation --samples 2500 --epochs 30 --eval-games 100 --seed 42
```

Fine-tune the policy with environment rewards, then evaluate a saved checkpoint:

```bash
uv run python -m ai.train --stage dqn --checkpoint models/imitation.pt --samples 1000 --episodes 300 --eval-games 100 --seed 42
uv run python -m ai.train --stage eval-only --checkpoint models/best.pt --eval-games 200 --seed 42
```

Launch the game with the default model path:

```bash
uv run python -m render.window
```

Press `A` to toggle AI mode, `Space` to pause or advance one move while paused, `T` to show or hide the decision panel, `R` to generate a new board, and `Esc` to reset to the solved board. The latest report is written under `reports/` and includes training metrics, evaluation results, and representative decision traces.
