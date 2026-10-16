# 第八代困难随机纯神经增强 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Execute this plan task-by-task in the current session with tests after each task.

**Goal:** Improve the eighth-generation pure-neural solver on uniformly random solvable 4×4 boards and retrain the embedded checkpoint.

**Architecture:** Add a uniform-random dataset generator and evaluate it separately from trajectory states. Update the v8 beam score and diversity selection using neural outputs only, then add hard-board DAgger and retrain the existing five-fold embedded ensemble.

**Tech Stack:** Python 3.13, PyTorch, pygame, pytest, existing weighted-A* teacher used offline only.

## Global Constraints

- Runtime v8 must not call A*, `heuristic_score`, or any handcrafted route solver.
- The shared artifact must remain `models/第八代数字华容道高速王.pt`.
- Cross-validation remains five state-disjoint folds.
- The interactive budget remains 100 moves at 300 ms per move.

### Task 1: Add uniform-random labeled states

**Files:**
- Modify: `src/ai/dataset.py`
- Test: `tests/ai/test_dataset.py`

**Interfaces:** Add `generate_uniform_examples(count: int, *, seed: int = 42) -> tuple[Example, ...]`. It repeatedly calls `board.generate_board(4, 4)`, flattens the board, rejects duplicate states, and labels each state with `_label_state`.

- [ ] Write a test that requests 12 examples, asserts unique states, legal expert actions, and replay to a valid next state.
- [ ] Run `uv run pytest -q tests/ai/test_dataset.py::test_generate_uniform_examples` and verify it fails before implementation.
- [ ] Implement the generator with a bounded retry budget of `max(100, count * 100)` and raise `RuntimeError` if the requested number cannot be labeled.
- [ ] Run the focused test and then `uv run pytest -q tests/ai/test_dataset.py`.

### Task 2: Make v8 beam scoring distance-aware and diverse

**Files:**
- Modify: `src/ai/agent.py`
- Test: `tests/ai/test_agent.py`

**Interfaces:** Keep `EighthGenerationNeuralAgent._plan(start) -> tuple[Action, ...] | None`; update its internal candidate score only from `_predict` outputs and exact state identity. Add a per-layer diversity cap of at least one candidate for each legal first action before filling remaining beam slots.

- [ ] Add a deterministic fake full model test where probability-only ranking chooses a dead-end branch but predicted distance chooses the shorter branch.
- [ ] Run the focused test and verify it fails with the old score.
- [ ] Normalize each state’s Q values, subtract normalized predicted distance from the accumulated score, subtract risk, and add a small action-log-probability term. Deduplicate candidate states per layer while retaining the best score.
- [ ] Fill the beam by first selecting the best candidate for each first action, then highest remaining scores up to `beam_width`.
- [ ] Run `uv run pytest -q tests/ai/test_agent.py` and verify the no-A* and anti-loop tests remain green.

### Task 3: Add hard-board DAgger and separate evaluation

**Files:**
- Modify: `src/ai/train.py`
- Modify: `src/ai/dataset.py`
- Test: `tests/ai/test_training.py`

**Interfaces:** Add `evaluate_uniform_random(agent, games, seed, max_steps=100) -> dict[str, Any]`; add `uniform_random_examples` and `hard_dagger_examples` metrics to `run_training`.

- [ ] Add a small training test asserting the eighth-generation metrics include `uniform_random_evaluation` and its game count.
- [ ] Run the focused test and verify it fails before implementation.
- [ ] Build training examples as a deduplicated combination of trajectory examples and an equal-sized uniform-random sample. Generate DAgger starts from uniform states and merge relabeled states before the second fold training pass.
- [ ] Evaluate both the existing held-out trajectory examples and 100 uniform-random examples with 100 moves, recording solve rate, solve rate within 30 seconds, illegal-action rate, beam failure rate, and repeated-state visits.
- [ ] Run `uv run pytest -q tests/ai/test_training.py`.

### Task 4: Retrain and verify artifacts

**Files:**
- Update: `models/第八代数字华容道高速王.pt`
- Create: `reports/training-*.md`, `reports/training-*.metrics.json`, `reports/第八代数字华容道高速王-训练指标.png`, `reports/第八代数字华容道高速王-训练过程.mp4`

- [ ] Run `uv run pytest -q`.
- [ ] Run `uv run python -m ai.train --stage eighth-generation --samples 6000 --epochs 25 --folds 5 --train-max-depth 1000 --eval-games 100 --video-output reports/第八代数字华容道高速王-训练过程.mp4 --video-games 4 --dagger-episodes 16 --seed 20261011`.
- [ ] Load only the resulting `.pt` and run an independent 100-game evaluation with A* and `heuristic_score` monkeypatched to raise.
- [ ] Inspect metrics for uniform-random success, 30-second success, illegal actions, beam failures, and repeated states; confirm the video is encoded and the checkpoint embeds five folds.
