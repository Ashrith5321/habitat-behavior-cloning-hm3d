# PIRLNav ObjectNav Imitation Learning — Full Technical README

## What This Project Is

This is a **Behavioral Cloning (BC)** pipeline that trains a robot navigation agent to find objects in 3D indoor environments. The agent is taught entirely from **human demonstrations**: real teleoperation recordings of people navigating through 3D houses and finding objects like chairs, beds, plants, toilets, TVs, and sofas.

The task is called **ObjectNav** (Object Goal Navigation): given a target object category (e.g. "chair"), navigate to any instance of that object in the scene.

---

## Why We Built This (Motivation)

Reinforcement Learning for navigation is slow and brittle — the agent must explore randomly for millions of steps before it stumbles onto a reward signal. **Imitation Learning (IL)** sidesteps this: instead of learning from scratch, the agent copies what humans already do well. PIRLNav (People in Real Life doing Navigation) is a dataset of exactly these human demonstrations, collected via VR teleoperation on photorealistic 3D house scans.

This project loads those demonstrations, replays them through a physics-capable 3D simulator, and trains a neural network policy to predict the same action a human would take at each step.

---

## The Full Software Stack

Training runs across three layers simultaneously. Understanding which layer does what is critical.

```
┌─────────────────────────────────────────────────────────┐
│  train_il.py  ←  YOUR TRAINING LOOP                    │
│  - controls what happens at each step                   │
│  - computes loss, calls backward(), saves checkpoints   │
├─────────────────────────────────────────────────────────┤
│  habitat-baselines  ←  POLICY ARCHITECTURE              │
│  - PointNavResNetPolicy: ResNet18 + GRU neural net      │
│  - provides evaluate_actions() API for BC training      │
├─────────────────────────────────────────────────────────┤
│  habitat-lab  ←  TASK + METRICS LAYER                   │
│  - habitat.Env: wraps the simulator as an RL env        │
│  - defines ObjectNav task (what counts as success)      │
│  - computes success, SPL, distance_to_goal metrics      │
│  - manages episode/scene switching                      │
├─────────────────────────────────────────────────────────┤
│  habitat-sim  ←  3D RENDERER + PHYSICS ENGINE           │
│  - loads HM3D .glb 3D scene files                       │
│  - renders RGB (480×640) + depth (480×640) images       │
│  - executes move_forward / turn_left / etc.             │
│  - returns updated agent position/rotation              │
└─────────────────────────────────────────────────────────┘
```

All four layers run every single step of training. The simulator renders a frame, habitat-lab wraps it in an observation dict, train_il.py feeds it to the neural network, and the network outputs a probability distribution over actions.

---

## Where Every Component Came From

### The Dataset: PIRLNav HD

- **Source:** Ramrakhya et al., "Habitat-Web: Learning Embodied Object-Search Strategies from Human Demonstrations at Scale" (CVPR 2022) — later extended as PIRLNav.
- **What it is:** ~76,000+ episodes of human VR teleoperation across 80 HM3D scenes, stored as `.json.gz` files.
- **Location:** `/home/ashed/data/datasets/objectnav/objectnav_hm3d_hd/train/content/`
- **Format:** Each `.json.gz` file = one scene, containing:
  - `episodes`: list of navigation tasks (start position, target object, human action sequence)
  - `reference_replay`: the actual human action sequence — `[{"action": "MOVE_FORWARD"}, ...]`
  - `goals_by_category`: where each object type is located in the scene
  - `category_to_task_category_id`: maps "chair" → integer ID used by the policy

### The 3D Scenes: HM3D

- **Source:** Ramakrishnan et al., "Habitat-Matterport 3D Dataset (HM3D)" (NeurIPS 2021).
- **What it is:** 800 photorealistic 3D house scans, stored as `.glb` (GL Transmission Format) files with `.basis` texture compression.
- **Location:** `/home/ashed/Documents/hm3d-train-habitat-v0.2/NNNNN-SCENE_ID/SCENE_ID.basis.glb`
- **Why basis:** Basis Universal texture compression — smaller files, GPU decompresses on upload. Requires the `basis` plugin in habitat-sim.
- **How many used:** Only 80 of the 800 scenes have matching PIRLNav episode files.

### The Simulator: habitat-sim 0.3.3

- **Source:** Facebook AI Research (now Meta AI). [github.com/facebookresearch/habitat-sim](https://github.com/facebookresearch/habitat-sim)
- **What it does:** Loads `.glb` scenes, places an agent with RGB + depth cameras, executes navigation actions, returns sensor readings. Runs headless (no window needed) at 30–100+ FPS.
- **Why habitat-sim and not something else:** The PIRLNav dataset was collected using habitat-sim, so the visual observations during training exactly match what was seen during data collection. Using a different renderer would cause a visual domain gap.

### The Task Layer: habitat-lab 0.3.3

- **Source:** Facebook AI Research. [github.com/facebookresearch/habitat-lab](https://github.com/facebookresearch/habitat-lab)
- **What it does:** Defines the ObjectNav task on top of habitat-sim. Provides `habitat.Env` (an OpenAI Gym-like interface), the success condition (agent within 1m of target object and calls STOP), and the SPL metric (Success weighted by Path Length).
- **Config used:** `benchmark/nav/objectnav/objectnav_hm3d.yaml` — this file ships with habitat-lab and defines sensors (RGB, depth, objectgoal, compass, GPS), action space, and reward (not used in IL).
- **Why we need it:** habitat-sim alone gives you pixels. habitat-lab gives you the `objectgoal` sensor (what object to find) and `success`/`SPL`/`DTG` metrics. Without it, you can't know if the agent succeeded.

### The Policy: PointNavResNetPolicy from habitat-baselines 0.3.3

- **Source:** Facebook AI Research. Part of habitat-baselines, originally designed for DD-PPO RL training.
- **Architecture:**
  - **Visual encoder:** ResNet18, pretrained from scratch (not ImageNet — habitat trains from scratch because the task distribution is very different from ImageNet). Takes RGB (480×640×3) + depth (480×640×1) → 512-dim visual embedding.
  - **Recurrent layer:** GRU (Gated Recurrent Unit) with hidden size 512. Carries memory across timesteps — the agent remembers where it has been.
  - **Action head:** Linear layer → 6-dim logits → softmax → probability over {STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, LOOK_UP, LOOK_DOWN}.
- **Total parameters:** 5.7 million (ResNet18 is intentionally small — navigation is compute-bound by the simulator, not the network).
- **Why ResNet18 not ViT or larger:** Speed. Each training step requires a full simulator render. Using a larger visual encoder would make the simulator the even harder bottleneck with no proportional benefit.
- **Why GRU not LSTM:** GRU has fewer parameters (no separate cell state), trains faster, and performs comparably on navigation horizons of 100–500 steps. LSTM would add ~25% more parameters for minimal gain.
- **Import path:** `from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy`

---

## The Training Algorithm: Behavioral Cloning

### What BC Is

Behavioral Cloning is supervised learning on (observation, expert_action) pairs. At each timestep:
1. The simulator renders what the agent currently sees
2. The human demonstration says what action was taken here
3. We train the network to assign high probability to that action

The loss is cross-entropy: `L = -log π(expert_action | observation)`

### Why Not Reinforcement Learning

RL requires the agent to explore randomly and receive rewards. For ObjectNav:
- Random exploration rarely reaches the goal → sparse rewards → very slow learning
- Millions of environment steps needed before any useful gradient signal
- Training on 80 scenes × 76k episodes with RL would take weeks

BC converges in hours because every single step has a training signal (the human action).

**The tradeoff:** BC agents suffer from distribution shift — if the agent makes a mistake and ends up in a state the human never visited, it has no training signal for recovering. This is a known limitation addressed by methods like DAgger, but BC is the correct starting point.

### The evaluate_actions API

The key function from habitat-baselines used for BC training:

```python
_, log_probs, _, new_hidden, _ = policy.evaluate_actions(
    obs_t,      # observation dict: {"rgb": tensor, "depth": tensor, ...}
    hidden,     # GRU hidden state: shape (1, num_layers, hidden_size)
    prev_act,   # previous action taken: shape (1, 1) long tensor
    mask,       # episode boundary mask: False at step 0 (resets GRU), True otherwise
    expert_id,  # ground-truth action: shape (1, 1) long tensor
    None,       # rnn_build_seq_info — None triggers single_forward path in RNN encoder
)
```

**Why `rnn_build_seq_info=None`:** habitat-baselines normally batches many environment steps together for vectorized RL training. When this argument is provided, the RNN encoder uses a complex sequence-packing path. Passing `None` instead triggers the simple `single_forward` path — process one step at a time. This is correct for our online BC setup where we feed one observation per step.

**The `mask` argument:** At step 0 of each episode, `mask=False` causes the GRU to zero its hidden state (fresh episode, no memory from before). At all subsequent steps, `mask=True` so memory accumulates.

**Hidden state shape:** `(batch=1, num_layers=1, hidden_size=512)`. Note the batch dimension is the first axis — this is habitat-baselines' convention, different from PyTorch's default `(num_layers, batch, hidden_size)`.

---

## Inflection Weighting — Why and How

### The Problem: Class Imbalance in Human Demonstrations

Human navigators mostly walk forward in long straight corridors. A typical episode looks like:

```
MOVE_FORWARD × 80, TURN_LEFT × 3, MOVE_FORWARD × 40, TURN_RIGHT × 2, STOP
```

If you train with equal weight on all steps, the network learns to always predict MOVE_FORWARD (≈70–80% accuracy without learning anything useful about the goal). Turns and stops, which carry the most navigational information, are drowned out.

### The Solution: Inflection Weighting

**Source:** Landi et al., "Perceive, Transform, and Act: Multi-Modal Attention Networks for Vision-and-Language Navigation" — and adopted in Habitat-Web / PIRLNav papers.

The idea: assign higher loss weight to **timesteps where the action changes** (inflection points) and lower weight to repetitive continuations.

```python
# Inflection weight coefficient ≈ total_steps / inflection_steps ≈ 3.28
# At each step where action[t] != action[t-1]:
weight = iw_coeff   # e.g., 3.28
# At repetitive continuation steps:
weight = 1.0
```

We compute `iw_coeff` from the actual data:

```python
def compute_iw_coeff(replays) -> float:
    total, inflections = 0, 0
    for replay in replays:
        actions = [s["action"] for s in replay[1:] if s["action"] != "STOP"]
        total += len(actions)
        inflections += sum(
            1 for i in range(1, len(actions)) if actions[i] != actions[i - 1]
        )
    return total / max(inflections, 1)
```

Result: ~3.28 for 80 scenes (roughly 1 in 3 steps is an inflection). This means the network pays 3× more attention to direction changes than to straight-ahead walking.

---

## Truncated Backpropagation Through Time (TBPTT)

### The Problem: Long Episodes Break Backprop

ObjectNav episodes can be 100–500 steps long. Backpropagating through 500 RNN steps:
- Requires storing all intermediate activations in GPU memory (500 × 512 tensors)
- Gradients vanish or explode over 500 steps
- GPU OOM for typical hidden sizes

### The Solution: TBPTT with Chunk Size 64

We process the episode in **chunks of 64 steps**. After each chunk:
1. Compute the loss for those 64 steps
2. Call `loss.backward()` — gradients flow through those 64 steps only
3. Clip gradients to norm 0.5 (prevents exploding gradients)
4. Call `optimizer.step()` — update weights
5. **Detach the hidden state** — `hidden = hidden.detach()` — so the next chunk's backward pass stops here

```python
if len(chunk_steps) == TBPTT_STEPS:
    hidden = flush_chunk()   # flush_chunk() returns hidden.detach()
    chunk_steps = []
```

**Why detach:** Without detaching, PyTorch would try to backprop through the entire episode history accumulated in `hidden`'s computation graph, which defeats the purpose.

**Why 64:** A power of 2 that's large enough to capture multi-step context (typical corridor straight run = 20–40 steps) but small enough to fit in GPU memory comfortably. This value comes from habitat-baselines' own TBPTT defaults.

**Why grad clip 0.5:** Prevents a single bad episode from making a large destructive update. 0.5 is the value used in the original habitat-baselines DD-PPO training.

---

## The Dataset Loading Problem (Why We Don't Use Habitat's Own Loader)

### What Habitat's Loader Would Do

Normally you'd use `habitat.datasets.make_dataset("ObjectNav-v1")` and point it at the data directory. This calls `ObjectGoalNavEpisode(**episode_dict)` for each episode.

### Why That Crashes with PIRLNav Data

`ObjectGoalNavEpisode` is defined using the `attr` library with `@attr.s(auto_attribs=True)` — a strict dataclass that **rejects any unknown keyword arguments** with:

```
TypeError: __init__() got an unexpected keyword argument 'reference_replay'
```

The PIRLNav episodes have extra fields that habitat's original ObjectNav episodes don't:
- `reference_replay` — the human action sequence (this is the whole point of PIRLNav!)
- `is_thda` — "Teleoperated Human Demonstration Annotation" flag
- `attempts` — how many tries the human took

### Our Fix: Manual Dataset Construction

We bypass the constructor entirely:

```python
dataset = ObjectNavDatasetV1.__new__(ObjectNavDatasetV1)  # skip __init__
dataset.episodes = []
dataset.goals_by_category = {}
# ... manually set required attributes
```

Then for each episode, we filter to only the fields `ObjectGoalNavEpisode` actually accepts:

```python
EPISODE_KNOWN = {
    "episode_id", "scene_id", "scene_dataset_config",
    "additional_obj_config_paths", "start_position", "start_rotation",
    "info", "goals", "start_room", "shortest_paths", "object_category",
}
clean = {k: v for k, v in ep.items() if k in EPISODE_KNOWN}
episode = ObjectGoalNavEpisode(**clean)
```

We save `reference_replay` separately in a parallel list `replays[]` indexed by episode index.

The same filter applies to goals. `ObjectGoal` also rejects unknown fields like `object_name_id`:

```python
GOAL_KNOWN = {
    "position", "radius", "object_id", "object_name",
    "object_category", "room_id", "room_name", "view_points",
}
```

### The Scene Path Problem

Each episode's `scene_id` in the JSON looks like a relative path: `hm3d/train/00744-1S7LAXRdDqK/1S7LAXRdDqK.basis.glb`. But habitat looks this up relative to a `data/` directory structure we don't have configured.

**Fix:** We replace `scene_id` with the absolute path to the `.glb` file:

```python
def _scene_glb(scene_id: str) -> str:
    # scene_id from JSON is like "1S7LAXRdDqK"
    for entry in os.scandir(SCENES_DIR):
        if entry.is_dir() and entry.name.endswith(f"-{scene_id}"):
            glb = os.path.join(entry.path, f"{scene_id}.basis.glb")
            if os.path.exists(glb):
                return glb
    raise FileNotFoundError(f"No extracted GLB for scene {scene_id}")
```

The directory structure on disk is: `00744-1S7LAXRdDqK/1S7LAXRdDqK.basis.glb` — the 5-digit prefix is the HM3D scene index. We scan for any directory ending in `-{scene_id}` to avoid hardcoding the prefix.

### The scene_dataset_config Problem

Habitat also looks for a `hm3d_annotated_basis.scene_dataset_config.json` file that maps scene IDs to their semantic annotations. We don't have this file.

**Fix:** Set `scene_dataset_config = "default"` on every episode. This tells habitat-sim to load the `.glb` as a plain mesh with no semantic scene annotations. Navigation still works perfectly — we lose per-object semantic IDs, but the `objectgoal` sensor (what category to find) still works via the category mappings we load manually.

---

## How the Goals System Works

### goals_by_category

In each scene's `.json.gz`, there's a dict like:
```json
{
  "goals_by_category": {
    "1S7LAXRdDqK_chair": [{"position": [x,y,z], "view_points": [...], ...}],
    "1S7LAXRdDqK_bed":   [{"position": [x,y,z], "view_points": [...], ...}]
  }
}
```

Each entry is a list of `ObjectGoal` instances — physical locations of that object type in that scene.

### episode.goals_key

`ObjectGoalNavEpisode` has a property `goals_key` that returns `f"{basename(scene_id)}_{object_category}"`, e.g. `"1S7LAXRdDqK_chair"`. We use this to look up the correct goals list:

```python
episode.goals = dataset.goals_by_category.get(episode.goals_key, [])
```

This is how the agent knows where in the scene the target object is (for computing DTG/SPL) — the goals contain the object's 3D position. The `objectgoal` sensor separately gives the agent the category ID (an integer from `category_to_task_category_id`) as input.

### ObjectViewLocation

Each goal also has `view_points` — positions from which the object is visible (within 1m, with line-of-sight). The success condition checks: is the agent at a view_point AND did it call STOP?

```python
for vp in g_dict.get("view_points") or []:
    state = AgentState(
        position=vp["agent_state"]["position"],
        rotation=vp["agent_state"].get("rotation"),
    )
    vps.append(ObjectViewLocation(agent_state=state, iou=vp.get("iou")))
```

`iou` here is the Intersection-over-Union of the object in the camera view from that viewpoint (used internally by habitat for quality filtering).

---

## The Observation Space

Defined by the config `objectnav_hm3d.yaml`. Each observation is a dict:

| Key | Shape | Dtype | What it is |
|-----|-------|-------|------------|
| `rgb` | (480, 640, 3) | uint8 | RGB camera, 90° HFOV |
| `depth` | (480, 640, 1) | float32 | Depth camera (0–10m normalized) |
| `objectgoal` | (1,) | int64 | Target object category ID |
| `compass` | (1,) | float32 | Agent heading relative to start (radians) |
| `gps` | (2,) | float32 | Agent x,z displacement from start (meters) |

The `objectgoal` sensor is what tells the agent **what to look for** — without it, the agent would have no idea whether its target is "chair" or "bed". The category ID is mapped via `category_to_task_category_id` to an integer the ResNet head can embed.

In `train_il.py`, we convert each observation to a GPU tensor:

```python
obs_t = {k: torch.tensor(v, device=device).unsqueeze(0) for k, v in obs.items()}
```

The `unsqueeze(0)` adds a batch dimension (batch size = 1 for online training).

---

## The Action Space

6 discrete actions:

| ID | Name | Effect |
|----|------|--------|
| 0 | STOP | End episode, trigger success check |
| 1 | MOVE_FORWARD | Move 0.25m forward |
| 2 | TURN_LEFT | Rotate 30° left |
| 3 | TURN_RIGHT | Rotate 30° right |
| 4 | LOOK_UP | Tilt camera 30° up |
| 5 | LOOK_DOWN | Tilt camera 30° down |

The JSON stores actions as uppercase strings (`"MOVE_FORWARD"`). habitat.Env expects lowercase (`"move_forward"`). We map between them with `EXPERT_ID` (string → int for loss) and `HAB_STR` (string → lowercase for simulator step).

---

## Multi-Scene Training — Why It Matters

### The Single-Scene Plateau Problem

We first trained on only one scene (1,134 episodes of scene `1S7LAXRdDqK`). After ~50 episodes, loss plateaued at ~1.60 and stopped improving, regardless of additional epochs.

**Why:** With one scene, the visual inputs are always the same walls, floors, and furniture. The network memorizes the scene's visual statistics without learning generalizable navigation. The GRU hidden state encodes "where am I in this specific apartment" rather than "how do I navigate toward objects generally."

### The Fix: 80-Scene Training

PIRLNav provides episodes for 80 different HM3D scenes. With 80 scenes:
- Every 100 episodes, the scene changes → radically different visual appearance
- The ResNet must learn visual features that generalize (edges, object shapes, depth cues) not scene-specific textures
- The GRU learns navigation strategies not scene layouts

**Practical constraint:** We only use scenes where we have the extracted `.glb` file. Of the 800 HM3D training scenes, we have 80 extracted. Of those 80, all have matching PIRLNav episode files.

### The MAX_EP_PER_SCENE Cap

Full 80-scene dataset: ~76,394 episodes, ~3+ days per epoch on our hardware.

With `MAX_EP_PER_SCENE = 100`: 80 × 100 = 8,000 episodes. This preserves full scene diversity while fitting in ~5 hours. The episodes chosen are the first 100 from each scene file (sorted order), which is a reasonable random sample since the JSON files aren't ordered by difficulty.

---

## The Training Loop Step by Step

For each episode:

1. **Parse the replay** (`action_seq`): Extract the human action sequence. Skip the first action if it's STOP (artifact of some recordings). Stop when STOP is encountered.

2. **Compute inflection weights**: Build a per-step weight array. Default 1.0, `iw_coeff` at direction changes.

3. **Initialize state**: Zero GRU hidden state (shape `(1, 1, 512)`), zero previous action tensor.

4. **Reset the environment**: `obs = env.reset()` — habitat.Env advances to the next episode in its iterator, loads the scene if needed, places the agent at the start position, returns the initial observation.

5. **Step through the replay**: For each expert action:
   - Convert obs to GPU tensors, add batch dim
   - Compute `mask` (False at t=0 → zero GRU, True otherwise → carry memory)
   - Call `evaluate_actions` → get log probability of the expert action
   - Append `(log_prob, weight)` to current chunk

6. **Flush chunk every 64 steps**: Stack log probs and weights, compute mean weighted loss, backward, clip grad, optimizer step, detach hidden.

7. **Execute the expert action in the simulator**: `obs = env.step(HAB_STR[action_str])` — the simulator moves the agent, returns new RGB+depth+sensors. This is what makes training "online": the agent actually moves through the scene.

8. **Final flush**: Process the last partial chunk.

9. **Get metrics**: `env.get_metrics()` returns `success` (did agent call STOP near the object?). During BC training on expert data, success rate should be ~1.0 if the replays are correct — it's a sanity check, not a training signal.

---

## The Checkpoint System

Checkpoints save three things:

```python
torch.save({
    "epoch": epoch,
    "episode": episode,
    "state_dict": policy.state_dict(),    # all network weights
    "optimizer": optimizer.state_dict(),  # Adam moment estimates (for resuming)
}, path)
```

Two checkpoint triggers:
- **Every 500 episodes**: `ckpt_ep{epoch}_{episode}.pth` — allows recovery if training crashes
- **End of epoch**: `ckpt_epoch{N}.pth` — the "official" checkpoint for evaluation

The optimizer state is saved so training can be resumed mid-epoch without losing Adam's momentum estimates (which take hundreds of steps to warm up).

---

## Loss Interpretation

| Loss value | What it means |
|-----------|---------------|
| ~3.4 | Random initialization — uniform distribution over 6 actions × IW coefficient |
| ~2.0 | Network is learning something but still mostly random |
| ~1.6 | Single-scene saturation — memorized one scene, not generalizing |
| ~1.0 | Good BC fit — network predicts correct action about 37% of the time |
| ~0.5 | Strong fit — network predicts correct action ~60% of the time |
| <0.3 | Likely overfitting on limited scenes |

Note: ~37% accuracy at loss=1.0 might seem low but remember 6 actions and inflection weighting — the loss is inflated by the 3× weight on direction changes. Actual action accuracy at loss=1.0 is typically 60–70%.

---

## Warnings That Appear During Training (All Harmless)

| Warning | Why It Appears | Safe to Ignore |
|---------|---------------|----------------|
| `SceneDatasetAttributes: Lighting Layout 'no_lights' not found` | HM3D scenes reference a lighting preset not present in our config | Yes — default lighting is used |
| `SemanticScene .scn not found` | Each HM3D scene has an optional semantic annotation file we don't have | Yes — navigation works without it |
| `Gym has been unmaintained` | habitat-lab uses an old OpenAI Gym API | Yes |
| `PluginManager duplicate plugin` | habitat-sim registers sensors multiple times at startup | Yes |

---

## File Structure

```
/home/ashed/Documents/tests/
├── train_il.py          ← BC training (this file)
├── policy_runner.py     ← Reference replay through habitat.Env (metrics test)
├── replay.py            ← Raw habitat-sim replay, saves video
├── replay_live.py       ← Same + live matplotlib window
└── README.md            ← This file

/home/ashed/data/datasets/objectnav/objectnav_hm3d_hd/train/content/
└── SCENE_ID.json.gz     ← PIRLNav episode files (80 files)

/home/ashed/Documents/hm3d-train-habitat-v0.2/
└── NNNNN-SCENE_ID/
    └── SCENE_ID.basis.glb   ← 3D scene files (800 directories, 80 match PIRLNav)

/home/ashed/checkpoints/il_objectnav_multiscene/
├── ckpt_ep1_500.pth     ← Mid-epoch checkpoints
├── ckpt_ep1_1000.pth
└── ckpt_epoch1.pth      ← End-of-epoch checkpoint
```

---

## Running Training

```bash
cd /home/ashed
conda activate habitat
python Documents/tests/train_il.py
```

**Current config (5-hour run):**
- 80 scenes × 100 episodes = 8,000 episodes
- 1 epoch
- Checkpoints every 500 episodes

**Full-dataset config (multi-day):**
- Set `MAX_EP_PER_SCENE = None`, `EPOCHS = 3`
- ~76,394 episodes, ~3–9 days depending on hardware

---

## After Training: Evaluating the Policy

To test whether the trained agent can actually find objects (not just imitate):

1. Load a checkpoint
2. Replace the reference replay with the policy's own predictions
3. Run through `habitat.Env` and report real `success`/`SPL`/`DTG`

```python
# In eval_policy.py (to be written):
policy.load_state_dict(torch.load("ckpt_epoch1.pth")["state_dict"])
policy.eval()

# Replace replay loop with:
with torch.no_grad():
    value, action, log_prob, hidden, _ = policy.act(
        obs_t, hidden, prev_act, mask, deterministic=True
    )
action_id = action.item()
```

Yes — you can give it any target category and it will navigate autonomously. The `objectgoal` sensor feeds the category ID as input, so the agent's behavior changes depending on whether it's told "chair" vs "bed".

---

## Key Papers and Sources

| What | Paper | Where used |
|------|-------|------------|
| PIRLNav dataset | Ramrakhya et al., CVPR 2023 | Episode data, reference replays |
| Habitat-Web / human demo training | Ramrakhya et al., CVPR 2022 | Training methodology, IW |
| HM3D scenes | Ramakrishnan et al., NeurIPS 2021 | 3D scene files |
| habitat-sim | Savva et al., ICCV 2019 | Simulator |
| habitat-lab | Szot et al., NeurIPS 2021 | Task/metrics layer |
| DD-PPO (policy arch) | Wijmans et al., ICLR 2020 | PointNavResNetPolicy architecture |
| Inflection Weighting | Anderson et al. / Habitat-Web | Loss weighting for imbalanced demonstrations |
| Truncated BPTT | Werbos 1990, standard ML | RNN training over long sequences |
