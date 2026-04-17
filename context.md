# COMPLETE CONTEXT FILE: PIRLNav + Habitat-Web Replay & Data Analysis
## For Any Agent, Human, or LLM to Understand and Execute Without Reading Any Repository

---

## TABLE OF CONTENTS
1. [Project Overview](#1-project-overview)
2. [Two Datasets — Which One You Have](#2-two-datasets)
3. [Environment Setup](#3-environment-setup)
4. [Data Locations on Disk](#4-data-locations-on-disk)
5. [Data Format — Complete Reference](#5-data-format)
6. [Action Space](#6-action-space)
7. [What an Inflection Point Is](#7-inflection-points)
8. [Headless Data Analysis — No Simulator Needed](#8-headless-analysis)
9. [Pure Visual Replay — habitat-sim Only](#9-pure-replay)
10. [Live Window Replay](#10-live-replay)
11. [Common Errors and Fixes](#11-errors)
12. [Key Numbers from the Papers](#12-paper-numbers)
13. [What Each File and Folder Does](#13-file-map)
14. [Dependency Stack](#14-dependency-stack)
15. [Quick Command Reference](#15-quick-commands)

---

## 1. PROJECT OVERVIEW

### What This Is
This project works with two related research datasets of **human demonstrations for ObjectNav** (Object Goal Navigation) — the task where a robot must find and navigate to a specified object (e.g. "find a chair") in an unknown environment.

A human sat at Amazon Mechanical Turk (AMT), saw a first-person view of a virtual house, and pressed keyboard keys (WASD-style) to search for a specified object. Every keypress was recorded as an "action". These recordings are the datasets.

### The Research Goal
Use these human demonstrations to train robots via **Imitation Learning (IL)** — the robot watches what humans did and learns to copy it. This outperforms Reinforcement Learning (RL) because human search strategies are sophisticated (peeking into rooms, checking corners, doing panoramic turns) and hard to engineer as reward functions.

### The Two Papers
- **Habitat-Web** (CVPR 2022): 80k human demos on **MP3D** (Matterport3D) scenes
- **PIRLNav** (CVPR 2023): 77k human demos on **HM3D** (Habitat-Matterport 3D) scenes
- Same first author (Ram Ramrakhya), same data format, different scene datasets

---

## 2. TWO DATASETS — WHICH ONE YOU HAVE

### Habitat-Web (MP3D)
- **Scenes**: MP3D (Matterport3D) — 56 training scenes, scene IDs look like `17DRP5sb8fy`
- **Demo count**: 80,217 episodes (70k MP3D + 10k Gibson)
- **Object categories**: 21 categories (chair, table, picture, cabinet, cushion, sofa, bed, chest_of_drawers, plant, sink, toilet, stool, towel, tv_monitor, shower, bathtub, counter, fireplace, gym_equipment, seating, clothes)
- **HuggingFace**: `https://huggingface.co/datasets/axel81/habitat-web`
- **Code repo**: `https://github.com/Ram81/habitat-imitation-baselines`
- **Scene source**: `https://niessner.github.io/Matterport/` (academic license, 1-3 day approval)
- **agent_state**: HAS position data at every step

### PIRLNav (HM3D) ← YOU ARE USING THIS
- **Scenes**: HM3D (Habitat-Matterport 3D) — 145 training scenes, IDs look like `00744-1S7LAXRdDqK`
- **Demo count**: 77k human demos (ObjectNav-HD) + 240k shortest path (ObjectNav-SP) + 70k frontier exploration (ObjectNav-FE)
- **Object categories**: 6 categories ONLY: chair, bed, plant, sofa, toilet, tv_monitor
- **HuggingFace**: `https://huggingface.co/datasets/axel81/pirlnav`
- **Code repo**: `https://github.com/Ram81/pirlnav`
- **Scene source**: HM3D dataset from Matterport (same license process)
- **agent_state**: NULL at every step (no per-step position data)

### Key Difference Summary
| Property | Habitat-Web (MP3D) | PIRLNav (HM3D) |
|---|---|---|
| Scene dataset | MP3D | HM3D |
| Scene format | `mp3d/SCENE_ID/SCENE_ID.glb` | `hm3d/train/NNNNN-SCENE_ID/SCENE_ID.basis.glb` |
| # categories | 21 | 6 |
| agent_state | Has positions | NULL |
| Download | `axel81/habitat-web` | `axel81/pirlnav` |
| Avg steps/ep | 243 | ~150-200 |

---

## 3. ENVIRONMENT SETUP

### What Is Already Installed on This Machine
```
conda env:       habitat  (at /home/ashed/miniconda3/envs/habitat)
habitat-sim:     0.3.3
habitat-lab:     0.3.3
habitat-baselines: 0.3.3
Python:          3.9.23
GPU:             NVIDIA GeForce RTX 5090
Display:         :2 (X display available — windows will open)
```

### How to Activate
```bash
conda activate habitat
```

### Required Python Packages (Install If Missing)
```bash
pip install imageio imageio-ffmpeg matplotlib numpy
```

### Verify Everything Works
```bash
conda activate habitat
python -c "import habitat_sim; print(habitat_sim.__version__)"
python -c "import imageio; print('imageio ok')"
python -c "import matplotlib; print('matplotlib ok')"
```

---

## 4. DATA LOCATIONS ON DISK

### PIRLNav Demo Data (Human Demonstrations)
```
/home/ashed/data/datasets/objectnav/
├── objectnav_hm3d_hd/          ← HUMAN DEMOS (77k) — use this
│   └── train/
│       └── content/
│           ├── 1S7LAXRdDqK.json.gz    ← scene 00744, 1134 episodes
│           ├── <other_scene>.json.gz
│           └── ...
├── objectnav_hm3d_fe/          ← frontier exploration demos (70k)
└── objectnav_hm3d_sp/          ← shortest path demos (240k) [if downloaded]
```

### HM3D Scene Assets (3D Meshes)
```
/home/ashed/data/scene_datasets/hm3d/
└── minival/                    ← small test split (10 scenes, extracted)
    ├── 00800-TEEsavR23oF/
    ├── 00801-HaxA7YrQdEC/
    ├── 00802-wcojb4TFT35/
    ├── 00803-k1cupFYWXJ6/
    ├── 00804-BHXhpBwSMLh/
    ├── 00805-SUHsP6z2gcJ/
    ├── 00806-tQ5s4ShP627/
    ├── 00807-rsggHU7g7dh/
    ├── 00808-y9hTuugGdiq/
    └── 00809-Qpor2mEya8F/

/home/ashed/Documents/hm3d-train-habitat-v0.2/
└── 00744-1S7LAXRdDqK/         ← THE SCENE YOUR DEMOS USE
    ├── 1S7LAXRdDqK.basis.glb     ← 3D mesh file
    └── 1S7LAXRdDqK.basis.navmesh ← navigation mesh (pathfinding)
```

### Test Scripts Location
```
/home/ashed/Documents/tests/
├── replay.py          ← basic replay (saves video only)
├── replay_live.py     ← replay with live matplotlib window
└── sanity_test.py     ← scene loading test
```

---

## 5. DATA FORMAT — COMPLETE REFERENCE

### File Format
Each scene's episodes are stored as a **gzipped JSON** file: `<scene_id>.json.gz`

### Top-Level Structure
```json
{
  "goals_by_category": {
    "1S7LAXRdDqK.basis.glb_chair": [...],
    "1S7LAXRdDqK.basis.glb_sofa":  [...]
  },
  "episodes": [ ... ],
  "category_to_task_category_id": { "chair": 0, "bed": 1, ... },
  "category_to_scene_annotation_category_id": { "chair": 0, ... }
}
```

### Single Episode Structure
```json
{
  "episode_id":     "A1FYMTQ3YSEOQ9:36W0OB37HYHHYJIPVEGYQHN4HPPZH4",
  "scene_id":       "hm3d/train/00744-1S7LAXRdDqK/1S7LAXRdDqK.basis.glb",
  "start_position": [-8.43357, -2.6084, -4.88854],
  "start_rotation": [0, 0.25317, 0, -0.96742],
  "object_category": "sofa",
  "start_room":     null,
  "shortest_paths": null,
  "is_thda":        null,
  "info": {
    "geodesic_distance": 5.67572,
    "euclidean_distance": 6.39182,
    "closest_goal_object_id": 387
  },
  "goals": [],
  "reference_replay": [
    { "action": "STOP",         "agent_state": null },  ← index 0, SKIP THIS
    { "action": "TURN_LEFT",    "agent_state": null },  ← index 1, first real action
    { "action": "TURN_LEFT",    "agent_state": null },
    ...
    { "action": "STOP",         "agent_state": null }   ← last entry, episode done
  ],
  "attempts": 1,
  "scene_dataset_config": "./data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
}
```

### IMPORTANT QUIRKS
1. **Leading STOP**: `reference_replay[0]` is always `{"action": "STOP"}` — this is metadata, NOT a real action. ALWAYS skip index 0.
2. **NULL agent_state**: In PIRLNav (HM3D), `agent_state` is always `null`. No per-step position data exists.
3. **In Habitat-Web (MP3D)**, `agent_state` HAS position/rotation data: `{ "position": [x,y,z], "rotation": [qx,qy,qz,qw], "sensor_data": {...} }`
4. **episode_id format**: `WORKER_ID:TASK_ID` — the part before `:` is the AMT worker who collected it.
5. **start_rotation format**: `[qx, qy, qz, qw]` — note this is NOT standard quaternion order. When passing to numpy-quaternion, reorder to `[qw, qx, qy, qz]`.

### How to Load an Episode
```python
import gzip, json

with gzip.open('/home/ashed/data/datasets/objectnav/objectnav_hm3d_hd/train/content/1S7LAXRdDqK.json.gz', 'rt') as f:
    data = json.load(f)

episodes = data['episodes']          # list of all episodes in this scene
episode  = episodes[1]               # pick episode index 1
actions  = [step['action'] for step in episode['reference_replay'][1:]]  # skip index 0
```

---

## 6. ACTION SPACE

### ObjectNav Action Space (6 actions, both datasets)
| String (in JSON) | Habitat-Sim key | ID | Description |
|---|---|---|---|
| `"STOP"` | — | 0 | End episode (success) |
| `"MOVE_FORWARD"` | `"move_forward"` | 1 | Move 0.25m forward |
| `"TURN_LEFT"` | `"turn_left"` | 2 | Rotate 30° left |
| `"TURN_RIGHT"` | `"turn_right"` | 3 | Rotate 30° right |
| `"LOOK_UP"` | `"look_up"` | 4 | Tilt camera 30° up |
| `"LOOK_DOWN"` | `"look_down"` | 5 | Tilt camera 30° down |

### Action Map (JSON string → habitat-sim key)
```python
ACTION_MAP = {
    "MOVE_FORWARD": "move_forward",
    "TURN_LEFT":    "turn_left",
    "TURN_RIGHT":   "turn_right",
    "LOOK_UP":      "look_up",
    "LOOK_DOWN":    "look_down",
}
```

### PickPlace Action Space (9 actions — only in Habitat-Web, not PIRLNav)
MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT, LOOK_UP, LOOK_DOWN, GRAB_RELEASE, NO_OP, STOP

---

## 7. INFLECTION POINTS

### Definition
An inflection point is any timestep `t` where the action changes from the previous timestep:
```
inflection at t  ⟺  actions[t] != actions[t-1]
```

### Example
```
actions = [TL, TL, TL, MF, MF, MF, TR, MF, MF, STOP]
indices =   0   1   2   3   4   5   6   7   8    9

Inflections at: 3 (TL→MF), 6 (MF→TR), 7 (TR→MF), 9 (MF→STOP)
Total inflections: 4
Total actions: 10
```

### Why Inflection Points Matter
The paper uses **Inflection Weighting (IW)** in the imitation learning loss:
```
IW_coefficient = total_actions / total_inflection_points
```
This coefficient multiplies the loss at every inflection timestep, forcing the model to pay extra attention to **decision moments** (where the human changed what they were doing) vs. repetitive actions (long runs of MOVE_FORWARD).

### How to Compute
```python
import numpy as np

def find_inflections(action_sequence):
    actions = np.array(action_sequence)
    if len(actions) <= 1:
        return np.array([])
    mask = actions[1:] != actions[:-1]
    return np.where(mask)[0] + 1   # indices in original array

def iw_coefficient(all_sequences):
    total_actions = sum(len(s) for s in all_sequences)
    total_inflections = sum(len(find_inflections(s)) for s in all_sequences)
    return total_actions / total_inflections
```

### Expected Values
- Human demo inflection rate: ~15-25% of steps are inflections
- IW coefficient: roughly 4-8 (meaning inflection steps get 4-8× more loss weight)
- Shortest path demos have LOWER inflection rate (long MOVE_FORWARD runs in corridors) → higher IW coefficient needed

---

## 8. HEADLESS DATA ANALYSIS — NO SIMULATOR NEEDED

This is pure Python, works without habitat-sim, no GPU, no scene files.

### Load All Episodes from All Scene Files
```python
import gzip, json, glob, numpy as np
from collections import Counter

def load_all_episodes(data_path, max_episodes=None):
    files = sorted(glob.glob(f"{data_path}/*.json.gz"))
    all_episodes = []
    for path in files:
        with gzip.open(path, 'rt') as f:
            data = json.load(f)
        all_episodes.extend(data.get('episodes', []))
        if max_episodes and len(all_episodes) >= max_episodes:
            return all_episodes[:max_episodes]
    return all_episodes

def extract_actions(episodes):
    return [[step['action'] for step in ep['reference_replay'][1:]]  # skip index 0
            for ep in episodes]
```

### Compute All Inflection Stats
```python
def analyze(action_sequences):
    total_actions = 0
    total_inflections = 0
    results = []

    for seq in action_sequences:
        n = len(seq)
        infl = [i for i in range(1, n) if seq[i] != seq[i-1]]
        total_actions += n
        total_inflections += len(infl)
        results.append({
            'n_actions': n,
            'n_inflections': len(infl),
            'inflection_rate': len(infl) / n if n > 0 else 0,
            'inflection_indices': infl,
        })

    iw_coeff = total_actions / total_inflections if total_inflections else 0
    print(f"Total actions:         {total_actions:,}")
    print(f"Total inflections:     {total_inflections:,}")
    print(f"IW Coefficient:        {iw_coeff:.4f}")
    print(f"Avg inflection rate:   {np.mean([r['inflection_rate'] for r in results]):.4f}")
    return results, iw_coeff
```

### Run It
```python
DATA_PATH = "/home/ashed/data/datasets/objectnav/objectnav_hm3d_hd/train/content/"
episodes = load_all_episodes(DATA_PATH)
sequences = extract_actions(episodes)
results, iw = analyze(sequences)
```

---

## 9. PURE VISUAL REPLAY — habitat-sim ONLY

### What This Does
Loads a HM3D 3D scene, places the agent at the recorded start position, and steps through every action from `reference_replay`, capturing RGB frames and saving as video.

### Full Working Script (`replay.py`)
```python
import gzip, json, os, sys
import numpy as np, quaternion, imageio, habitat_sim

EPISODE_FILE  = "/home/ashed/data/datasets/objectnav/objectnav_hm3d_hd/train/content/1S7LAXRdDqK.json.gz"
SCENE_FILE    = "/home/ashed/Documents/hm3d-train-habitat-v0.2/00744-1S7LAXRdDqK/1S7LAXRdDqK.basis.glb"
EPISODE_INDEX = 1
OUTPUT_VIDEO  = "/home/ashed/replay_ep1.mp4"
FPS           = 10

for p in [EPISODE_FILE, SCENE_FILE]:
    if not os.path.exists(p): sys.exit(f"Not found: {p}")

with gzip.open(EPISODE_FILE, 'rt') as f:
    data = json.load(f)
episode = data['episodes'][EPISODE_INDEX]
print(f"Target: {episode['object_category']} | Actions: {len(episode['reference_replay'])}")

sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = SCENE_FILE
sim_cfg.enable_physics = False

rgb_spec = habitat_sim.CameraSensorSpec()
rgb_spec.uuid = "rgb"
rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
rgb_spec.resolution = [480, 640]
rgb_spec.position = [0.0, 1.25, 0.0]

agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = [rgb_spec]
agent_cfg.action_space = {
    "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)),
    "turn_left":    habitat_sim.agent.ActionSpec("turn_left",    habitat_sim.agent.ActuationSpec(amount=30.0)),
    "turn_right":   habitat_sim.agent.ActionSpec("turn_right",   habitat_sim.agent.ActuationSpec(amount=30.0)),
    "look_up":      habitat_sim.agent.ActionSpec("look_up",      habitat_sim.agent.ActuationSpec(amount=30.0)),
    "look_down":    habitat_sim.agent.ActionSpec("look_down",    habitat_sim.agent.ActuationSpec(amount=30.0)),
}
ACTION_MAP = {"MOVE_FORWARD":"move_forward","TURN_LEFT":"turn_left","TURN_RIGHT":"turn_right","LOOK_UP":"look_up","LOOK_DOWN":"look_down"}

sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
agent = sim.get_agent(0)
state = habitat_sim.agent.AgentState()
state.position = np.array(episode['start_position'], dtype=np.float32)
qx, qy, qz, qw = episode['start_rotation']
state.rotation = np.quaternion(qw, qx, qy, qz)
agent.set_state(state)

frames = []
for t, step in enumerate(episode['reference_replay']):
    action = step['action']
    if t == 0 and action == 'STOP': continue
    if action == 'STOP': break
    ha = ACTION_MAP.get(action)
    if ha:
        obs = sim.step(ha)
        frames.append(obs['rgb'][:, :, :3])

imageio.mimsave(OUTPUT_VIDEO, frames, fps=FPS)
print(f"Done: {OUTPUT_VIDEO} ({len(frames)/FPS:.1f}s)")
sim.close()
```

### Run
```bash
conda activate habitat
cd ~/Documents/tests
python replay.py
```

---

## 10. LIVE WINDOW REPLAY

### What This Does
Same as pure replay but opens a matplotlib window on display `:2` showing the agent's view in real-time. Title bar turns orange at inflection points.

### Prerequisites
```bash
pip install matplotlib
```

### Key Parts of the Live Script
```python
import matplotlib
matplotlib.use('TkAgg')        # use TkAgg for display :2
import matplotlib.pyplot as plt

# Setup before replay loop:
plt.ion()
fig, ax = plt.subplots(figsize=(9, 6))
ax.axis('off')
im = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
title = ax.set_title("", color="white", fontsize=12)
plt.draw()
plt.pause(0.1)

# Inside replay loop after obs = sim.step(ha):
im.set_data(obs['rgb'][:, :, :3])
is_inflection = (prev_action is not None and action != prev_action)
title.set_text(f"Step {t} | {action}{'  ← INFLECTION' if is_inflection else ''}")
title.set_color("orange" if is_inflection else "white")
plt.draw()
plt.pause(1.0 / FPS)
prev_action = action

# After loop:
plt.ioff()
plt.show(block=True)   # keeps window open until user closes it
```

### If Window Doesn't Show / TkAgg Fails
```python
matplotlib.use('Qt5Agg')   # try this instead
```

---

## 11. COMMON ERRORS AND FIXES

| Error | Cause | Fix |
|---|---|---|
| Script runs silently, no output | File is empty (0 bytes) | Recreate with nano, verify with `wc -l script.py` |
| `Permission denied (publickey)` on HF clone | SSH key not set up for HuggingFace | Use HTTPS: `git clone https://huggingface.co/...` |
| `ModuleNotFoundError: habitat_sim` | Wrong Python / conda env not active | `conda activate habitat` |
| Scene not found | Wrong path or tar not extracted | Extract tar to correct folder, verify with `ls` |
| All warnings about `.scn` / semantic files | Semantic annotations not downloaded | Harmless for pure replay, ignore |
| Black video frames | Camera height wrong or agent in void | Try `rgb_spec.position = [0.0, 0.88, 0.0]` |
| Agent doesn't move (stuck in wall) | Start position inside geometry | Try different episode index |
| `plt.show()` closes immediately | Non-blocking call | Use `plt.show(block=True)` |
| Empty `agent_state` (null) | PIRLNav dataset design | Expected — positions not stored. Use start_position only |
| `Gym has been unmaintained...` warning | Outdated gym library | Harmless warning, ignore |
| `PluginManager duplicate plugin` warning | Magnum rendering init | Harmless warning, ignore |
| Leading STOP in replay | First entry is always STOP | Skip index 0: `replay[1:]` |

---

## 12. KEY NUMBERS FROM THE PAPERS

### Habitat-Web (MP3D) — Paper Numbers
| Metric | Value |
|---|---|
| Total ObjectNav demos | 80,217 |
| Average steps per episode | 243 |
| Total actions (human demos) | ~19.5M |
| Episode success rate (train, <500 steps) | 88.9% |
| Human SPL (val) | 42.5% |
| Best RL agent success (val) | 34.6% |
| Best IL agent success (70k demos, val) | 35.4% |
| IL "exchange rate" | 1 human demo ≈ 4 RL trajectories |

### PIRLNav (HM3D) — Paper Numbers
| Metric | Value |
|---|---|
| Human demos (HD) | 77k |
| Shortest path demos (SP) | 240k |
| Frontier exploration demos (FE) | 70k |
| Best IL-only success (val) | 64.1% |
| Best IL→RL success (val) | 70.4% |
| Previous state of art | 60.0% |

### Key Finding Both Papers Agree On
- IL on human demos > IL on shortest paths (humans explore, shortest paths don't)
- More demos = better performance (scaling hasn't saturated)
- Inflection weighting significantly helps training

---

## 13. WHAT EACH FILE AND FOLDER DOES

### Data Files
| Path | What It Is |
|---|---|
| `objectnav_hm3d_hd/train/content/*.json.gz` | Human demo episodes, one file per HM3D scene |
| `objectnav_hm3d_fe/train/content/*.json.gz` | Frontier exploration demo episodes |
| `objectnav_hm3d_sp/train/content/*.json.gz` | Shortest path demo episodes |
| `hm3d/train/NNNNN-SCENE_ID/SCENE_ID.basis.glb` | 3D mesh of the scene |
| `hm3d/train/NNNNN-SCENE_ID/SCENE_ID.basis.navmesh` | Navigation mesh for pathfinding |
| `hm3d_annotated_basis.scene_dataset_config.json` | Config file Habitat needs to load scenes |

### What Each Field in an Episode Means
| Field | Type | Meaning |
|---|---|---|
| `episode_id` | string | Unique ID: `WORKER_ID:TASK_ID` |
| `scene_id` | string | Relative path to the .glb file |
| `start_position` | [x, y, z] | Where agent spawns |
| `start_rotation` | [qx, qy, qz, qw] | Which direction agent faces at start |
| `object_category` | string | What object the human was searching for |
| `reference_replay` | list | THE ACTION SEQUENCE — one dict per timestep |
| `reference_replay[t].action` | string | Action name at timestep t |
| `reference_replay[t].agent_state` | dict or null | Agent pose at timestep t (null in PIRLNav) |
| `info.geodesic_distance` | float | Shortest navigable path length to goal (meters) |
| `info.euclidean_distance` | float | Straight-line distance to goal (meters) |
| `attempts` | int | How many times the worker tried this episode |
| `scene_dataset_config` | string | Path to HM3D config (relative to project root) |

---

## 14. DEPENDENCY STACK

```
YOUR SCRIPTS
    ↓
habitat-sim 0.3.3          — C++ simulator, Python bindings
    ↓ uses
HM3D .glb + .navmesh       — 3D scene assets (in ~/Documents/hm3d-train-habitat-v0.2/)
    ↓ renders via
NVIDIA RTX 5090 / EGL      — headless GPU rendering (display :2 also available)
```

### What habitat-sim Does
- Load .glb 3D mesh files
- Place agent at (x, y, z) with quaternion rotation
- Execute actions (move_forward, turn_left, etc.)
- Render RGB/depth images from agent's camera
- Collision detection via navmesh

### What habitat-lab Does (NOT used for pure replay)
- Task definitions (ObjectNav success criteria)
- Gym-style environment wrapper (`habitat.Env`)
- Training loops
- Metrics (SPL, success rate, distance_to_goal)

### For Pure Replay: Only habitat-sim Is Needed

---

## 15. QUICK COMMAND REFERENCE

### Activate Environment
```bash
conda activate habitat
```

### Check Environment is Active
```bash
echo $CONDA_DEFAULT_ENV   # should print: habitat
which python              # should show: /home/ashed/miniconda3/envs/habitat/bin/python
```

### List Available Demo Scene Files
```bash
ls /home/ashed/data/datasets/objectnav/objectnav_hm3d_hd/train/content/ | head -20
```

### Check How Many Episodes in a Scene File
```bash
python -c "
import gzip, json
with gzip.open('/home/ashed/data/datasets/objectnav/objectnav_hm3d_hd/train/content/1S7LAXRdDqK.json.gz','rt') as f:
    d = json.load(f)
print(f'{len(d[\"episodes\"])} episodes')
"
```

### Quick Inflection Count for One Episode
```bash
python -c "
import gzip, json
with gzip.open('/home/ashed/data/datasets/objectnav/objectnav_hm3d_hd/train/content/1S7LAXRdDqK.json.gz','rt') as f:
    d = json.load(f)
ep = d['episodes'][1]
actions = [s['action'] for s in ep['reference_replay'][1:]]
infl = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i-1])
print(f'Actions: {len(actions)} | Inflections: {infl} | Rate: {infl/len(actions):.3f}')
"
```

### Run Basic Replay (Saves Video)
```bash
cd ~/Documents/tests
python replay.py
# Output: /home/ashed/replay_ep1.mp4
```

### Run Live Replay (Opens Window + Saves Video)
```bash
cd ~/Documents/tests
python replay_live.py
# Opens matplotlib window on display :2
# Title turns orange at inflection points
# Output: /home/ashed/replay_live_ep1.mp4
```

### Run Headless Analysis (No Simulator)
```bash
python headless_analysis.py \
    --data_path /home/ashed/data/datasets/objectnav/objectnav_hm3d_hd/train/content/ \
    --output_dir /home/ashed/results/ \
    --max_episodes 100   # remove this line for full 77k run
```

### Test Scene Loading
```bash
python -c "
import habitat_sim
cfg = habitat_sim.SimulatorConfiguration()
cfg.scene_id = '/home/ashed/Documents/hm3d-train-habitat-v0.2/00744-1S7LAXRdDqK/1S7LAXRdDqK.basis.glb'
sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [habitat_sim.agent.AgentConfiguration()]))
print('Scene loaded OK')
sim.close()
"
```

### Watch Output Video (Local Machine)
```bash
# From your local machine:
scp ashed@<server-ip>:/home/ashed/replay_ep1.mp4 ~/Desktop/
```

---

## APPENDIX: EPISODE WORKER IDs

Episode IDs have format `WORKER_ID:TASK_ID`. The worker IDs starting with `A` are Amazon Mechanical Turk worker IDs (anonymized). Multiple episodes with the same worker ID prefix means the same person collected multiple demos. This can be used for inter-annotator analysis.

## APPENDIX: SCENE NAMING CONVENTIONS

| Dataset | Scene ID Format | Example |
|---|---|---|
| MP3D (Habitat-Web) | 11-char alphanumeric | `17DRP5sb8fy` |
| HM3D (PIRLNav) | `NNNNN-11chars` | `00744-1S7LAXRdDqK` |
| Gibson | Short alphanumeric | `Adrian` |

## APPENDIX: ROTATION CONVENTION

Habitat stores rotations as `[qx, qy, qz, qw]` (x, y, z, w order).
`numpy-quaternion` expects `[qw, qx, qy, qz]` (w first).

Always reorder when setting agent state:
```python
qx, qy, qz, qw = episode['start_rotation']
state.rotation = np.quaternion(qw, qx, qy, qz)   # reordered!
```

## APPENDIX: WHAT "HEADLESS" MEANS

Headless = rendering to memory/file without a visible window.
Habitat-sim uses **EGL** (GPU rendering without X display) by default.
On this machine, display `:2` exists so windows CAN open, but the simulator defaults to EGL unless a display is explicitly configured.
The `Renderer: NVIDIA GeForce RTX 5090` message = GPU rendering working correctly, headless or not.