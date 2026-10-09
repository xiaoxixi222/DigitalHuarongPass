**清华附中计算机高研作品**

## Neural Solver

The project includes an optional PyTorch solver for the 4×4 board. The seventh-generation checkpoint learns from A* teacher trajectories generated offline; the game process loads only the neural ensemble and does not call A* during deployment. Install the locked dependencies with:

```bash
uv sync
```

Train the supervised policy and generate a checkpoint plus a Markdown report:

```bash
uv run python -m ai.train --stage imitation --samples 2500 --epochs 30 --eval-games 100 --seed 42
```

Train the seventh-generation model with deeper trajectory supervision:

```bash
uv run python -m ai.train --stage sixth-generation --samples 3000 --epochs 20 --folds 5 --train-max-depth 1000 --eval-games 100 --seed 20261008
```

This writes the seventh-generation checkpoint `models/第七代飞天数字华容道享受者.pt` and five deep fold checkpoints. The training data now samples random walks from depth 2 through 1000; this is a scramble budget, while the 4×4 puzzle's actual shortest routes remain much shorter. The checkpoint also averages the frozen fifth-generation folds so shallow behavior is retained while the new folds cover deeper states. Its metadata records `astar_guard=false` and enables a bounded neural beam planner (`beam_width=64`, `beam_horizon=80`); runtime inference uses only neural policy outputs, legal-action filtering, and the planner.

```bash
uv run python -m ai.train --stage eval-only --checkpoint models/第七代飞天数字华容道享受者.pt --eval-games 100 --seed 20261008
```

The latest independent run from the 1000-depth training data solved all 100 shallow games and all 100 depth-24 games, with 0% fallback and 0% illegal actions. The depth buckets solved 100/100 at depth 12, 100/100 at depth 16, 100/100 at depth 20, and 100/100 at depth 24. The planner is still bounded and does not provide a formal guarantee over every solvable permutation. The latest evaluation is `reports/training-20261007-153012.md` with metrics in the matching `.metrics.json` file.

For the higher-accuracy legacy model, train five state-disjoint folds and average them at inference time. The command writes the fold checkpoints, updates `models/第五代超级数字华容道之神.pt`, and records both pure-ensemble and A*-guarded results offline:

```bash
uv run python -m ai.train --stage cross-validation --samples 5000 --epochs 25 --folds 5 --eval-games 200 --seed 42
```

When `models/第五代超级数字华容道之神.pt` contains an ensemble manifest, the game loads all fold models automatically. The sixth-generation runtime path does not enable that legacy A* guard.

Fine-tune the policy with environment rewards, then evaluate a saved checkpoint:

```bash
uv run python -m ai.train --stage dqn --checkpoint models/imitation.pt --samples 1000 --episodes 300 --eval-games 100 --seed 42
uv run python -m ai.train --stage eval-only --checkpoint models/第五代超级数字华容道之神.pt --eval-games 200 --seed 42
```

Launch the game with the seventh-generation model explicitly:

```bash
uv run python -m render.window --model models/第七代飞天数字华容道享受者.pt
```

The fifth-generation `models/第五代超级数字华容道之神.pt` checkpoint remains available for comparison. It may contain A* guard metadata, but the game loader now disables the guard for interactive inference as well.

Press `A` to toggle AI mode, `Space` to pause or advance one move while paused, `T` to show or hide the decision panel, `R` to generate a new board, and `Esc` to reset to the solved board. The latest report is written under `reports/` and includes training metrics, evaluation results, and representative decision traces.
