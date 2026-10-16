# 第八代数字华容道高速王 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Add a cross-validated eighth-generation neural solver whose five folds, metadata, and runtime configuration are packaged in one shareable checkpoint, with reproducible training visualizations and an MP4 comparing early and late behavior.

**Architecture:** Keep the existing 260-feature state encoder and compatibility forward() API, then extend HuarongNet with policy, action value, distance, and failure risk outputs. Train state-disjoint folds on trajectory examples with symmetry augmentation and a DAgger-style model-state collection pass; deploy an embedded ensemble through a pure neural beam planner that never calls A*. Render metrics and fixed-board early/late traces with pygame, then encode frames with ffmpeg.

**Tech Stack:** Python 3.13, PyTorch, pygame, ffmpeg, pytest, uv.

## Global Constraints

- Published checkpoint: models/第八代数字华容道高速王.pt.
- Runtime eighth-generation inference never calls Board.get_road, astar_action, or heuristic_score.
- Five cross-validation folds have state-disjoint validation sets.
- Existing older checkpoint loading and tests remain compatible.
- Video failure never prevents checkpoint and metrics creation.
- Use uv run for tests and training commands.

---

### Task 1: Model heads and single-file checkpoint

**Files:** Modify src/ai/model.py; test tests/ai/test_model.py.

**Interfaces:** HuarongNet.forward_full(features) returns policy_logits, q_values, distance, distance_logits, and risk_logits. Preserve forward(features) returning policy_logits and q_values. Add save_ensemble_checkpoint(path, models, metadata) and load_ensemble_checkpoint(path, device).

- [ ] Write failing tests for output shapes and embedded-fold round trip.
- [ ] Run uv run pytest tests/ai/test_model.py -q; expect failure.
- [ ] Implement a residual multi-task MLP over the existing 260 features, a 0–96 distance classifier, scalar distance regressor, and four-action risk logits. Store all fold state dicts in one payload with no external paths.
- [ ] Run focused tests; expect pass.
- [ ] Commit: feat(ai): add eighth-generation multi-task checkpoint.

### Task 2: Trajectory augmentation and DAgger data

**Files:** Modify src/ai/dataset.py; test tests/ai/test_dataset.py.

**Interfaces:** Extend Example with optional next_state, distance_bucket, and risk_target fields; add augment_example(example) and collect_dagger_examples(agent, starts, max_steps, seed).

- [ ] Add tests for horizontal reflection action mapping and unique valid DAgger states.
- [ ] Run focused tests; expect failure.
- [ ] Implement deterministic rotations/reflections with transformed action labels, applied after train/validation split.
- [ ] Implement bounded DAgger collection that records model-visited states and labels them only through the offline teacher.
- [ ] Run focused tests; expect pass.
- [ ] Commit: feat(ai): add augmented trajectory and dagger data.

### Task 3: Multi-task five-fold training

**Files:** Modify src/ai/train.py; test tests/ai/test_training.py.

**Interfaces:** Add train_eighth_generation_fold, evaluate_eighth_generation, stage=eighth-generation, and CLI options video-output, video-games, and dagger-episodes.

- [ ] Add a tiny end-to-end test that requests two folds and asserts one checkpoint named 第八代数字华容道高速王.pt embeds all folds.
- [ ] Run focused training tests; expect failure.
- [ ] Implement masked policy loss, Huber distance loss, distance-bucket loss, pairwise action ranking, risk loss, and symmetry consistency.
- [ ] Train state-disjoint folds with independent seeds and select epochs using solve rate, action accuracy, distance accuracy, and loop rate.
- [ ] Mix bounded DAgger states after initial trajectory data; record per-fold and distance-bucket metrics.
- [ ] Run focused tests; expect pass.
- [ ] Commit: feat(ai): train eighth-generation cross-validation ensemble.

### Task 4: Pure-neural runtime

**Files:** Modify src/ai/agent.py; test tests/ai/test_agent.py.

- [ ] Add tests monkeypatching astar_action, Board.get_road, and heuristic_score to raise, then load an embedded checkpoint and make legal decisions.
- [ ] Run focused agent tests; expect failure.
- [ ] Average all embedded folds’ policy, Q, distance, and risk outputs.
- [ ] Score beam nodes only with neural log probability, predicted distance, value, and risk; maintain path-local visited states and a width-32, horizon-96 budget.
- [ ] Expose decision_source=eighth-generation-neural-beam and beam/loop metrics.
- [ ] Run focused tests; expect pass.
- [ ] Commit: feat(ai): run eighth-generation embedded neural beam.

### Task 5: Training curves and early/late MP4

**Files:** Create src/ai/training_video.py; modify src/ai/train.py; test tests/ai/test_training_video.py.

**Interfaces:** Add render_training_metrics(metrics, output_png), render_comparison_frames(early_agent, final_agent, states, frame_dir, fps), and encode_training_video(frame_dir, output_mp4, fps).

- [ ] Add tests for deterministic PNG frame creation and an unavailable ffmpeg returning False without raising.
- [ ] Run focused visualization tests; expect failure.
- [ ] Render a 1280x720 pygame frame with metric chart, early/final boards, actions, distance predictions, confidence, and solved status.
- [ ] Encode frame-%06d.png using ffmpeg; preserve frames and metrics when encoding fails.
- [ ] Integrate fixed test states, early checkpoint, final ensemble, metrics JSON/PNG, and reports/第八代数字华容道高速王-训练过程.mp4.
- [ ] Run focused tests; expect pass.
- [ ] Commit: feat(ai): render eighth-generation training video.

### Task 6: UI, README, and end-to-end validation

**Files:** Modify src/render/window.py, src/render/ai_panel.py, README.md; tests in tests/render.

- [ ] Put the single eighth-generation model first in the selector and display the exact Chinese name and pure-neural source.
- [ ] Document one-file sharing, training-only A*, training command, evaluation command, and video outputs.
- [ ] Run uv run pytest -q; expect pass.
- [ ] Run a small eighth-generation smoke training with two folds, one epoch, one DAgger episode, and one video game.
- [ ] Copy the generated checkpoint to an empty directory and load it without fold files.
- [ ] Commit: docs(render): expose eighth-generation shareable model.

