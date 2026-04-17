"""
Evaluate a trained BC checkpoint using the policy's own predictions.

Shows live agent view via OpenCV window (habitat's observations_to_image).

Run from /home/ashed/:
    cd /home/ashed && conda activate habitat
    DISPLAY=:2 python Documents/tests/eval_policy.py
"""
import gzip
import json
import os

import cv2
import numpy as np
import torch
from gym import spaces

import habitat
from habitat.core.simulator import AgentState
from habitat.datasets.object_nav.object_nav_dataset import ObjectNavDatasetV1
from habitat.tasks.nav.object_nav_task import (
    ObjectGoal,
    ObjectGoalNavEpisode,
    ObjectViewLocation,
)
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy

# ─── Config ───────────────────────────────────────────────────────────────────
CKPT_PATH    = "/home/ashed/checkpoints/il_objectnav_multiscene/ckpt_ep1_500.pth"
CONTENT_FILE = "/home/ashed/data/datasets/objectnav/objectnav_hm3d_hd/train/content/1S7LAXRdDqK.json.gz"
SCENE_GLB    = "/home/ashed/Documents/hm3d-train-habitat-v0.2/00744-1S7LAXRdDqK/1S7LAXRdDqK.basis.glb"
MAX_EPISODES = 20
MAX_STEPS    = 500
DEVICE       = "cuda:0"
HIDDEN_SIZE  = 512
BACKBONE     = "resnet18"
RNN_TYPE     = "GRU"
NUM_ACTIONS  = 6

EPISODE_KNOWN = {
    "episode_id", "scene_id", "scene_dataset_config",
    "additional_obj_config_paths", "start_position", "start_rotation",
    "info", "goals", "start_room", "shortest_paths", "object_category",
}
GOAL_KNOWN = {
    "position", "radius", "object_id", "object_name",
    "object_category", "room_id", "room_name", "view_points",
}
ACTION_NAMES = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "LOOK_UP", "LOOK_DOWN"]
HAB_STR = {
    "STOP": "stop", "MOVE_FORWARD": "move_forward",
    "TURN_LEFT": "turn_left", "TURN_RIGHT": "turn_right",
    "LOOK_UP": "look_up", "LOOK_DOWN": "look_down",
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
    dataset.category_to_task_category_id = raw["category_to_task_category_id"]
    dataset.category_to_scene_annotation_category_id = raw.get(
        "category_to_scene_annotation_category_id", {}
    )
    dataset.goals_by_category = {
        k: [_build_goal(g) for g in v]
        for k, v in raw["goals_by_category"].items()
    }
    dataset.episodes = []

    for i, ep in enumerate(raw["episodes"][:MAX_EPISODES]):
        clean = {k: v for k, v in ep.items() if k in EPISODE_KNOWN}
        clean["episode_id"]           = str(i)
        clean["scene_id"]             = SCENE_GLB
        clean["scene_dataset_config"] = "default"
        clean["goals"]                = []
        episode = ObjectGoalNavEpisode(**clean)
        episode.goals = dataset.goals_by_category.get(episode.goals_key, [])
        dataset.episodes.append(episode)

    return dataset


def make_env(dataset):
    config = habitat.get_config(
        "benchmark/nav/objectnav/objectnav_hm3d.yaml",
        overrides=[
            "habitat.environment.max_episode_steps=500",
            "habitat.simulator.habitat_sim_v0.gpu_device_id=0",
        ],
    )
    return habitat.Env(config=config, dataset=dataset)


def load_policy(obs_space):
    policy = PointNavResNetPolicy(
        observation_space=obs_space,
        action_space=spaces.Discrete(NUM_ACTIONS),
        hidden_size=HIDDEN_SIZE,
        num_recurrent_layers=1,
        rnn_type=RNN_TYPE,
        resnet_baseplanes=32,
        backbone=BACKBONE,
        normalize_visual_inputs=False,
    ).to(DEVICE)

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    print(f"  Loaded checkpoint: epoch={ckpt['epoch']}  episode={ckpt['episode']}")
    return policy


def main():
    print("Loading dataset...")
    dataset = build_dataset()
    print(f"  {len(dataset.episodes)} episodes from scene 1S7LAXRdDqK")

    print("Creating environment...")
    env = make_env(dataset)

    print("Loading policy...")
    policy = load_policy(env.observation_space)

    cv2.namedWindow("habitat-eval", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("habitat-eval", 960, 480)

    results = []

    for ep_idx in range(len(dataset.episodes)):
        obs = env.reset()
        episode = env.current_episode

        hidden   = torch.zeros(1, policy.num_recurrent_layers,
                               policy.recurrent_hidden_size, device=DEVICE)
        prev_act = torch.zeros(1, 1, dtype=torch.long, device=DEVICE)

        print(f"\n{'='*60}")
        print(f"Episode {ep_idx} | Target: {episode.object_category}")

        for t in range(MAX_STEPS):
            # Render live view
            frame = observations_to_image(obs, {})
            cv2.imshow("habitat-eval", frame[:, :, ::-1])
            key = cv2.waitKey(1)
            if key == ord("q"):
                env.close()
                cv2.destroyAllWindows()
                return

            obs_t = {k: torch.tensor(v, device=DEVICE).unsqueeze(0)
                     for k, v in obs.items()}
            mask = torch.tensor([[t > 0]], dtype=torch.bool, device=DEVICE)

            with torch.no_grad():
                action_data = policy.act(
                    obs_t, hidden, prev_act, mask, deterministic=True
                )

            action     = action_data.actions
            hidden     = action_data.rnn_hidden_states
            action_id  = action.item()
            action_str = ACTION_NAMES[action_id]
            prev_act   = action.detach()

            if action_str == "STOP" or env.episode_over:
                break

            obs = env.step(HAB_STR[action_str])

        metrics = env.get_metrics()
        success = metrics.get("success", False)
        spl     = metrics.get("spl", 0.0)
        dtg     = metrics.get("distance_to_goal", float("nan"))
        results.append({"success": success, "spl": spl, "dtg": dtg})

        print(f"  Steps: {t+1} | Success: {success} | SPL: {spl:.4f} | DTG: {dtg:.3f}m")

    env.close()
    cv2.destroyAllWindows()

    print(f"\n{'='*60}")
    print(f"Results over {len(results)} episodes:")
    print(f"  Success rate: {sum(r['success'] for r in results) / len(results):.2%}")
    print(f"  Mean SPL:     {sum(r['spl'] for r in results) / len(results):.4f}")
    print(f"  Mean DTG:     {sum(r['dtg'] for r in results) / len(results):.3f}m")


if __name__ == "__main__":
    main()
