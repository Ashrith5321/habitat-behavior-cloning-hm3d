# PIRLNav ObjectNav IL Training — Session Context

## What This Is
Imitation Learning (BC) pipeline for ObjectNav using PIRLNav human demonstrations and HabitatSim/Lab 0.3.3. The agent learns to navigate to objects (chair/bed/plant/toilet/tv_monitor/sofa) by imitating human teleoperation sequences.

## Environment
- Conda env: `habitat` (Python 3.9, habitat-sim 0.3.3, habitat-lab 0.3.3, habitat-baselines 0.3.3)
- GPU: RTX 5090, display :2 available
- Run all scripts from `/home/ashed/` with `conda activate habitat`

## Data Paths
| What | Path |
|------|------|
| Episode files (80 scenes, PIRLNav HD) | `/home/ashed/data/datasets/objectnav/objectnav_hm3d_hd/train/content/*.json.gz` |
| Extracted HM3D training scenes (800 GLBs) | `/home/ashed/Documents/hm3d-train-habitat-v0.2/NNNNN-SCENE_ID/SCENE_ID.basis.glb` |
| HM3D minival scenes (10 scenes) | `/home/ashed/data/scene_datasets/hm3d/minival/` |
| IL checkpoints | `/home/ashed/checkpoints/il_objectnav_multiscene/` |

80 of the 800 extracted scenes have matching episode files — all 80 are used for training.

## Scripts (`/home/ashed/Documents/tests/`)
| Script | Purpose |
|--------|---------|
| `replay.py` | Replay one episode from reference data, save video |
| `replay_live.py` | Same but shows live matplotlib window |
| `policy_runner.py` | Run reference-replay as policy through `habitat.Env`, prints success/SPL/DTG |
| `train_il.py` | **BC training** — PointNavResNetPolicy on all 80 scenes |

## Key Design Decisions

### Dataset Loading (why manual, not habitat's loader)
PIRLNav episodes have extra fields (`reference_replay`, `is_thda`, `attempts`) that crash `ObjectGoalNavEpisode(**ep)`. We load manually:
- Strip unknown fields before constructing `ObjectGoalNavEpisode`
- Set `episode.scene_id` = absolute path to `.glb` file
- Set `episode.scene_dataset_config = "default"` (bypasses missing `hm3d_annotated_basis.scene_dataset_config.json`)
- Set `episode.goals` from `goals_by_category` using `episode.goals_key = f"{basename(scene_id)}_{object_category}"`

### Habitat config
```python
habitat.get_config("benchmark/nav/objectnav/objectnav_hm3d.yaml", overrides=[...])
# Note: .yaml extension required. Config file at:
# /home/ashed/habitat-lab/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml
```
Provides: rgb (480×640×3) + depth (480×640×1) + objectgoal (1,) + compass (1,) + gps (2,)

### Policy
`PointNavResNetPolicy` from `habitat_baselines.rl.ddppo.policy`:
- ResNet18 + GRU, 5.7M params
- `evaluate_actions(obs, hidden, prev_act, mask, expert_id, None)` — last arg is `rnn_build_seq_info=None` which triggers `single_forward` path in RNN encoder (valid when `x.size(0) == hidden.size(1)` after permute)
- hidden state shape: `(batch, num_recurrent_layers, hidden_size)` = `(1, 1, 512)`
- `policy.act(obs, hidden, prev_act, mask, deterministic=True)` for greedy inference

### Training (BC with inflection weighting)
- Loss: `-log π(expert_a | obs) × inflection_weight` per step, averaged per TBPTT chunk
- Inflection weight: `iw_coeff ≈ 3.4` at action-change steps, `1.0` elsewhere
- TBPTT chunk = 64 steps; hidden state detached between chunks
- Grad clip norm = 0.5, Adam lr = 2.5e-4

### Multi-scene training (critical for diversity)
Single-scene training (1134 eps, 1 scene) plateaus at loss ≈ 1.60 after ~50 episodes — the model saturates on one scene's visual statistics. Use all 80 scenes (~91k+ episodes total).

## Training Command
```bash
cd /home/ashed
conda activate habitat
python Documents/tests/train_il.py
```
Checkpoints → `/home/ashed/checkpoints/il_objectnav_multiscene/ckpt_epoch{N}.pth`

With 76,394 episodes across 80 scenes, use EPOCHS=3 (not 10). Diversity makes repetition less useful. 1 epoch ≈ many hours of simulator rendering time.

## Evaluation (next step after training)
Load a checkpoint and run the policy's own predictions through `habitat.Env` (not reference replay) to get real success/SPL metrics. Modify `policy_runner.py` to:
1. Load the checkpoint: `policy.load_state_dict(torch.load(ckpt)["state_dict"])`
2. Replace the replay loop with `policy.act(obs_t, hidden, prev_act, mask, deterministic=True).actions.item()`
3. Report `env.get_metrics()` per episode

## Loss Interpretation
| Loss | Meaning |
|------|---------|
| ~3.4 | Random init (uniform over 6 actions × IW) |
| ~1.6 | Single-scene saturation (epoch 1 end on 1 scene) |
| ~1.0 | Good BC fit |
| <0.5 | Strong fit (possible overfitting if only 1 scene) |

## Known Warnings (harmless, ignore)
- `SceneDatasetAttributes: Lighting Layout 'no_lights' not found` — harmless
- `SemanticScene .scn not found` — no semantic annotations, navigation still works
- `Gym has been unmaintained` — ignore
- `PluginManager duplicate plugin` — ignore
