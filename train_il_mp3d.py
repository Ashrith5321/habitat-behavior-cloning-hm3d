"""
BC training on MP3D ObjectNav using shortest-path expert trajectories.

Single scene (17DRP5sb8fy), up to MAX_EPISODES episodes per epoch.
Action sequences come from episode['shortest_paths'][0] (integer IDs).

Run from /home/ashed/:
    cd /home/ashed
    conda activate habitat
    python Documents/tests/habitat-behavior-cloning-hm3d/train_il_mp3d.py
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
CONTENT_FILE = "/home/ashed/data/datasets/objectnav/train/content/17DRP5sb8fy.json.gz"
SCENE_GLB    = "/home/ashed/Documents/tests/my_data/mp3d/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb"
CKPT_DIR     = "/home/ashed/checkpoints/il_mp3d_17DRP5sb8fy"

# ─── Training hyperparameters ─────────────────────────────────────────────────
DEVICE        = "cuda:0"
LR            = 2.5e-4
EPOCHS        = 2
MAX_EPISODES  = 50000    # full scene (50k total episodes)
TBPTT_STEPS   = 64
CKPT_INTERVAL = 200

# ─── Policy architecture ──────────────────────────────────────────────────────
HIDDEN_SIZE = 512
BACKBONE    = "resnet18"
RNN_TYPE    = "GRU"
NUM_ACTIONS = 6

# ─── Action mappings ──────────────────────────────────────────────────────────
# shortest_paths uses integer IDs: 1=MOVE_FORWARD, 2=TURN_LEFT, 3=TURN_RIGHT,
# 4=LOOK_UP, 5=LOOK_DOWN, None=STOP
INT_TO_HAB = {
    1: "move_forward",
    2: "turn_left",
    3: "turn_right",
    4: "look_up",
    5: "look_down",
}

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


def _build_goal(g_dict):
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


def build_dataset():
    with gzip.open(CONTENT_FILE, "rt") as f:
        raw = json.load(f)

    dataset = ObjectNavDatasetV1.__new__(ObjectNavDatasetV1)
    dataset.content_scenes_path = "{data_path}/content/{scene}.json.gz"
    dataset.category_to_task_category_id = raw.get("category_to_task_category_id", {})
    dataset.category_to_scene_annotation_category_id = raw.get(
        "category_to_scene_annotation_category_id", {}
    )
    dataset.goals_by_category = {
        k: [_build_goal(g) for g in v]
        for k, v in raw.get("goals_by_category", {}).items()
    }
    dataset.episodes = []
    action_seqs = []

    for i, ep in enumerate(raw["episodes"][:MAX_EPISODES]):
        # Extract shortest path action sequence (strip trailing None = STOP)
        sp = ep.get("shortest_paths", [[]])[0] or []
        action_seq = [a for a in sp if a is not None and a in INT_TO_HAB]
        action_seqs.append(action_seq)

        clean = {k: v for k, v in ep.items() if k in EPISODE_KNOWN}
        clean["episode_id"]           = str(i)
        clean["scene_id"]             = SCENE_GLB
        clean["scene_dataset_config"] = "default"
        clean["goals"]                = []
        episode = ObjectGoalNavEpisode(**clean)
        episode.goals = dataset.goals_by_category.get(episode.goals_key, [])
        dataset.episodes.append(episode)

    return dataset, action_seqs


def compute_iw_coeff(action_seqs):
    total, inflections = 0, 0
    for seq in action_seqs:
        total += len(seq)
        inflections += sum(
            1 for i in range(1, len(seq)) if seq[i] != seq[i - 1]
        )
    return total / max(inflections, 1)


def make_env(dataset):
    config = habitat.get_config(
        "benchmark/nav/objectnav/objectnav_mp3d.yaml",
        overrides=[
            "habitat.environment.max_episode_steps=500",
            "habitat.simulator.habitat_sim_v0.gpu_device_id=0",
        ],
    )
    return habitat.Env(config=config, dataset=dataset)


def train_episode(env, policy, optimizer, action_seq, iw_coeff, device):
    if not action_seq:
        return 0.0, 0, 0.0

    weights = [1.0] * len(action_seq)
    for i in range(1, len(action_seq)):
        if action_seq[i] != action_seq[i - 1]:
            weights[i] = iw_coeff

    hidden   = torch.zeros(1, policy.num_recurrent_layers,
                           policy.recurrent_hidden_size, device=device)
    prev_act = torch.zeros(1, 1, dtype=torch.long, device=device)

    obs = env.reset()

    chunk_steps = []
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

    for t, action_id in enumerate(action_seq):
        if env.episode_over:
            break

        obs_t     = {k: torch.tensor(v, device=device).unsqueeze(0)
                     for k, v in obs.items()}
        mask      = torch.tensor([[t > 0]], dtype=torch.bool, device=device)
        expert_id = torch.tensor([[action_id]], device=device)

        _, log_probs, _, new_hidden, _ = policy.evaluate_actions(
            obs_t, hidden, prev_act, mask, expert_id, None,
        )

        chunk_steps.append((log_probs.squeeze(), weights[t]))
        hidden   = new_hidden
        prev_act = expert_id.detach()

        if len(chunk_steps) == TBPTT_STEPS:
            hidden = flush_chunk()
            chunk_steps = []

        obs = env.step(INT_TO_HAB[action_id])

    flush_chunk()

    metrics  = env.get_metrics()
    avg_loss = total_loss / max(n_chunks, 1)
    return avg_loss, len(action_seq), float(metrics.get("success", 0.0))


def _save(policy, optimizer, epoch, episode, tag=None):
    os.makedirs(CKPT_DIR, exist_ok=True)
    name = f"ckpt_ep{epoch}_{episode}.pth" if tag is None else f"ckpt_{tag}.pth"
    path = os.path.join(CKPT_DIR, name)
    torch.save({
        "epoch": epoch,
        "episode": episode,
        "state_dict": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)
    print(f"  Saved: {path}")


def main():
    print("Loading dataset...")
    dataset, action_seqs = build_dataset()
    n_ep = len(dataset.episodes)
    iw   = compute_iw_coeff(action_seqs)
    print(f"  {n_ep} episodes | IW coefficient: {iw:.3f}")

    print("Creating environment...")
    env = make_env(dataset)

    print("Initializing policy...")
    policy = PointNavResNetPolicy(
        observation_space=env.observation_space,
        action_space=spaces.Discrete(NUM_ACTIONS),
        hidden_size=HIDDEN_SIZE,
        num_recurrent_layers=1,
        rnn_type=RNN_TYPE,
        resnet_baseplanes=32,
        backbone=BACKBONE,
        normalize_visual_inputs=False,
    ).to(DEVICE)
    policy.train()

    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  {BACKBONE}+{RNN_TYPE} | {n_params:,} params")

    optimizer = optim.Adam(policy.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*60}\nEpoch {epoch}/{EPOCHS}")
        epoch_loss = 0.0
        epoch_succ = 0.0

        for ep_idx in range(n_ep):
            loss, steps, success = train_episode(
                env, policy, optimizer, action_seqs[ep_idx], iw, DEVICE
            )
            epoch_loss += loss
            epoch_succ += success

            if (ep_idx + 1) % 50 == 0:
                print(f"  [{ep_idx+1:4d}/{n_ep}] "
                      f"loss={epoch_loss/(ep_idx+1):.4f}  "
                      f"success={epoch_succ/(ep_idx+1):.2%}  "
                      f"steps={steps}")

            if (ep_idx + 1) % CKPT_INTERVAL == 0:
                _save(policy, optimizer, epoch, ep_idx + 1)

        print(f"\nEpoch {epoch} | loss={epoch_loss/n_ep:.4f} | "
              f"success={epoch_succ/n_ep:.2%}")
        _save(policy, optimizer, epoch, n_ep, tag=f"epoch{epoch}")

    env.close()
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
