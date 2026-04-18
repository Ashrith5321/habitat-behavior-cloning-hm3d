"""
Behavioral Cloning (IL) training for ObjectNav on PIRLNav human demonstrations.

Uses PointNavResNetPolicy (ResNet18 + GRU) from habitat-baselines.
Training is online: the agent follows expert actions in the simulator,
collects observations at each step, and trains with cross-entropy +
inflection weighting via truncated BPTT.

Run from /home/ashed/:
    cd /home/ashed
    conda activate habitat
    python Documents/tests/train_il.py
"""
import gzip
import json
import os

import torch
import torch.optim as optim
from gym import spaces

import habitat
from habitat.core.simulator import AgentState
from habitat.datasets.object_nav.object_nav_dataset import ObjectNavDatasetV1
from habitat.tasks.nav.object_nav_task import (
    ObjectGoal,
    ObjectGoalNavEpisode,
    ObjectViewLocation,
)
from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy

# ─── Paths ────────────────────────────────────────────────────────────────────
CONTENT_DIR  = "/home/ashed/data/datasets/objectnav/objectnav_hm3d_hd/train/content/"
SCENES_DIR   = "/home/ashed/Documents/hm3d-train-habitat-v0.2/"
CKPT_DIR     = "/home/ashed/checkpoints/il_objectnav_multiscene"

# ─── Training hyperparameters ─────────────────────────────────────────────────
DEVICE        = "cuda:0"
LR            = 2.5e-4
EPOCHS        = 1
TBPTT_STEPS   = 64       # Truncated BPTT chunk length
MAX_EP_PER_SCENE = 100   # 80 scenes × 100 eps = 8,000 total, ~5h run
CKPT_INTERVAL = 500      # Save checkpoint every N episodes per epoch

# ─── Policy architecture ──────────────────────────────────────────────────────
HIDDEN_SIZE = 512
BACKBONE    = "resnet18"
RNN_TYPE    = "GRU"     # GRU is faster/simpler; use LSTM for better long-horizon

# ─── Action mappings ──────────────────────────────────────────────────────────
EXPERT_ID = {
    "STOP": 0, "MOVE_FORWARD": 1, "TURN_LEFT": 2,
    "TURN_RIGHT": 3, "LOOK_UP": 4, "LOOK_DOWN": 5,
}
HAB_STR = {
    "STOP": "stop", "MOVE_FORWARD": "move_forward",
    "TURN_LEFT": "turn_left", "TURN_RIGHT": "turn_right",
    "LOOK_UP": "look_up", "LOOK_DOWN": "look_down",
}
NUM_ACTIONS = 6

# ─── Episode/goal field filters ───────────────────────────────────────────────
EPISODE_KNOWN = {
    "episode_id", "scene_id", "scene_dataset_config",
    "additional_obj_config_paths", "start_position", "start_rotation",
    "info", "goals", "start_room", "shortest_paths", "object_category",
}
GOAL_KNOWN = {
    "position", "radius", "object_id", "object_name",
    "object_category", "room_id", "room_name", "view_points",
}


# ─── Dataset helpers (mirrors policy_runner.py) ───────────────────────────────

def _build_goal(g_dict: dict) -> ObjectGoal:
    vps = []
    for vp in g_dict.get("view_points") or []:
        state = AgentState(
            position=vp["agent_state"]["position"],
            rotation=vp["agent_state"].get("rotation"),
        )
        vps.append(ObjectViewLocation(agent_state=state, iou=vp.get("iou")))
    clean = {k: v for k, v in g_dict.items() if k in GOAL_KNOWN}
    clean["view_points"] = vps
    return ObjectGoal(**clean)


def _scene_glb(scene_id: str) -> str:
    """Map a scene ID like '1S7LAXRdDqK' to its absolute .glb path."""
    for entry in os.scandir(SCENES_DIR):
        if entry.is_dir() and entry.name.endswith(f"-{scene_id}"):
            glb = os.path.join(entry.path, f"{scene_id}.basis.glb")
            if os.path.exists(glb):
                return glb
    raise FileNotFoundError(f"No extracted GLB for scene {scene_id}")


def build_dataset(max_ep_per_scene=None):
    """Load episodes from all available scene content files."""
    import glob

    content_files = sorted(glob.glob(os.path.join(CONTENT_DIR, "*.json.gz")))
    print(f"  Found {len(content_files)} scene episode files")

    dataset = ObjectNavDatasetV1.__new__(ObjectNavDatasetV1)
    dataset.content_scenes_path = "{data_path}/content/{scene}.json.gz"
    dataset.goals_by_category = {}
    dataset.category_to_task_category_id = {}
    dataset.category_to_scene_annotation_category_id = {}
    dataset.episodes = []
    replays = []

    skipped = 0
    ep_global = 0
    for content_file in content_files:
        scene_id = os.path.basename(content_file).replace(".json.gz", "")
        try:
            glb = _scene_glb(scene_id)
        except FileNotFoundError:
            skipped += 1
            continue

        with gzip.open(content_file, "rt") as f:
            raw = json.load(f)

        # Merge category mappings (same across PIRLNav scenes)
        dataset.category_to_task_category_id.update(
            raw.get("category_to_task_category_id", {})
        )
        dataset.category_to_scene_annotation_category_id.update(
            raw.get("category_to_scene_annotation_category_id", {})
        )
        dataset.goals_by_category.update(
            {k: [_build_goal(g) for g in v]
             for k, v in raw.get("goals_by_category", {}).items()}
        )

        episodes_raw = raw.get("episodes", [])
        if max_ep_per_scene:
            episodes_raw = episodes_raw[:max_ep_per_scene]

        for ep in episodes_raw:
            replays.append(ep.get("reference_replay", []))
            clean = {k: v for k, v in ep.items() if k in EPISODE_KNOWN}
            clean.update(
                episode_id=str(ep_global),
                scene_id=glb,
                scene_dataset_config="default",
                goals=[],
            )
            episode = ObjectGoalNavEpisode(**clean)
            episode.goals = dataset.goals_by_category.get(episode.goals_key, [])
            dataset.episodes.append(episode)
            ep_global += 1

    if skipped:
        print(f"  Skipped {skipped} scenes (no extracted GLB)")
    return dataset, replays


def compute_iw_coeff(replays) -> float:
    total, inflections = 0, 0
    for replay in replays:
        actions = [s["action"] for s in replay[1:] if s["action"] != "STOP"]
        total += len(actions)
        inflections += sum(
            1 for i in range(1, len(actions)) if actions[i] != actions[i - 1]
        )
    return total / max(inflections, 1)


def make_env(dataset) -> habitat.Env:
    config = habitat.get_config(
        "benchmark/nav/objectnav/objectnav_hm3d.yaml",
        overrides=[
            "habitat.environment.max_episode_steps=500",
            "habitat.simulator.habitat_sim_v0.gpu_device_id=0",
        ],
    )
    return habitat.Env(config=config, dataset=dataset)


# ─── Training ─────────────────────────────────────────────────────────────────

def train_episode(env, policy, optimizer, replay, iw_coeff, device):
    """
    Run one expert-driven episode in the simulator, computing BC loss at
    each step and updating the policy via truncated BPTT every TBPTT_STEPS.

    Returns (avg_loss_per_chunk, n_steps, success).
    """
    # Build clean action sequence (skip leading/trailing STOP)
    action_seq = []
    for t, step in enumerate(replay):
        a = step["action"]
        if t == 0 and a == "STOP":
            continue
        if a == "STOP":
            break
        action_seq.append(a)

    if not action_seq:
        return 0.0, 0, 0.0

    # Inflection weights: iw_coeff at direction changes, 1.0 otherwise
    weights = [1.0] * len(action_seq)
    
    for i in range(1, len(action_seq)):
        if action_seq[i] != action_seq[i - 1]:
            weights[i] = iw_coeff

    n_layers = policy.num_recurrent_layers
    h_size   = policy.recurrent_hidden_size
    hidden   = torch.zeros(1, n_layers, h_size, device=device)
    prev_act = torch.zeros(1, 1, dtype=torch.long, device=device)

    obs = env.reset()

    chunk_steps = []   # (log_prob, weight) pairs for current TBPTT chunk
    total_loss  = 0.0
    n_chunks    = 0

    def flush_chunk():
        nonlocal total_loss, n_chunks
        if not chunk_steps:
            return hidden.detach()
        log_probs_t = torch.stack([lp for lp, _ in chunk_steps])
        weights_t   = torch.tensor([w for _, w in chunk_steps], device=device)
        loss = -(log_probs_t * weights_t).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        n_chunks   += 1
        return hidden.detach()

    for t, action_str in enumerate(action_seq):
        if env.episode_over:
            break

        obs_t = {k: torch.tensor(v, device=device).unsqueeze(0)
                 for k, v in obs.items()}
        # mask=False at step 0 zeroes the RNN hidden state
        mask      = torch.tensor([[t > 0]], dtype=torch.bool, device=device)
        expert_id = torch.tensor([[EXPERT_ID[action_str]]], device=device)

        _, log_probs, _, new_hidden, _ = policy.evaluate_actions(
            obs_t, hidden, prev_act, mask, expert_id,
            None,  # rnn_build_seq_info=None → single_forward path
        )

        chunk_steps.append((log_probs.squeeze(), weights[t]))
        hidden  = new_hidden
        prev_act = expert_id.detach()

        if len(chunk_steps) == TBPTT_STEPS:
            hidden = flush_chunk()
            chunk_steps = []

        obs = env.step(HAB_STR[action_str])

    hidden = flush_chunk()  # Final partial chunk

    metrics  = env.get_metrics()
    avg_loss = total_loss / max(n_chunks, 1)
    return avg_loss, len(action_seq), float(metrics.get("success", 0.0))


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)

    print("Loading dataset (all scenes)...")
    dataset, replays = build_dataset(MAX_EP_PER_SCENE)
    n_ep = len(dataset.episodes)
    iw   = compute_iw_coeff(replays)
    print(f"  {n_ep} total episodes across scenes | IW coefficient: {iw:.3f}")

    print("Creating environment...")
    env = make_env(dataset)

    print("Initializing policy...")
    obs_space    = env.observation_space
    action_space = spaces.Discrete(NUM_ACTIONS)
    policy = PointNavResNetPolicy(
        observation_space=obs_space,
        action_space=action_space,
        hidden_size=HIDDEN_SIZE,
        num_recurrent_layers=1,
        rnn_type=RNN_TYPE,
        resnet_baseplanes=32,
        backbone=BACKBONE,
        normalize_visual_inputs=False,
    ).to(DEVICE)
    policy.train()

    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  {BACKBONE} + {RNN_TYPE} | {n_params:,} trainable params")
    print(f"  recurrent_hidden_size = {policy.recurrent_hidden_size}")
    print(f"  num_recurrent_layers  = {policy.num_recurrent_layers}")

    optimizer = optim.Adam(policy.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{EPOCHS}")
        epoch_loss = 0.0
        epoch_succ = 0.0

        for ep_idx in range(n_ep):
            # env cycles through episodes automatically via EpisodeIterator
            replay = replays[ep_idx]

            loss, steps, success = train_episode(
                env, policy, optimizer, replay, iw, DEVICE
            )
            epoch_loss += loss
            epoch_succ += success

            if (ep_idx + 1) % 100 == 0:
                print(
                    f"  [{ep_idx+1:5d}/{n_ep}] "
                    f"loss={epoch_loss/(ep_idx+1):.4f}  "
                    f"success={epoch_succ/(ep_idx+1):.2%}  "
                    f"steps={steps}"
                )

            if (ep_idx + 1) % CKPT_INTERVAL == 0:
                _save(policy, optimizer, epoch, ep_idx + 1, CKPT_DIR)

        avg_loss = epoch_loss / n_ep
        avg_succ = epoch_succ / n_ep
        print(f"\nEpoch {epoch} | loss={avg_loss:.4f} | success={avg_succ:.2%}")
        _save(policy, optimizer, epoch, n_ep, CKPT_DIR, tag=f"epoch{epoch}")

    env.close()
    print("\nTraining complete.")


def _save(policy, optimizer, epoch, episode, ckpt_dir, tag=None):
    name = f"ckpt_ep{epoch}_{episode}.pth" if tag is None else f"ckpt_{tag}.pth"
    path = os.path.join(ckpt_dir, name)
    torch.save(
        {
            "epoch": epoch,
            "episode": episode,
            "state_dict": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
    print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
