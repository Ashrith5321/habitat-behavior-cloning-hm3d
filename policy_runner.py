"""
Run reference-replay as a policy through habitat.Env to get proper ObjectNav metrics.

Run from /home/ashed/:
    cd /home/ashed && conda activate habitat && python Documents/tests/policy_runner.py
"""
import gzip
import json
import os

import habitat
from habitat.core.simulator import AgentState
from habitat.datasets.object_nav.object_nav_dataset import ObjectNavDatasetV1
from habitat.tasks.nav.object_nav_task import (
    ObjectGoal,
    ObjectGoalNavEpisode,
    ObjectViewLocation,
)

# ─── Paths ────────────────────────────────────────────────────────────────────
SCENE_GLB    = "/home/ashed/Documents/hm3d-train-habitat-v0.2/00744-1S7LAXRdDqK/1S7LAXRdDqK.basis.glb"
CONTENT_FILE = "/home/ashed/data/datasets/objectnav/objectnav_hm3d_hd/train/content/1S7LAXRdDqK.json.gz"
MAX_EPISODES = 5
# ──────────────────────────────────────────────────────────────────────────────

ACTION_MAP = {
    "STOP":         "stop",
    "MOVE_FORWARD": "move_forward",
    "TURN_LEFT":    "turn_left",
    "TURN_RIGHT":   "turn_right",
    "LOOK_UP":      "look_up",
    "LOOK_DOWN":    "look_down",
}

# Fields in ObjectGoalNavEpisode (anything else gets stripped)
EPISODE_KNOWN = {
    "episode_id", "scene_id", "scene_dataset_config",
    "additional_obj_config_paths", "start_position", "start_rotation",
    "info", "goals", "start_room", "shortest_paths", "object_category",
}

# Fields in ObjectGoal (anything else gets stripped)
GOAL_KNOWN = {
    "position", "radius", "object_id", "object_name",
    "object_category", "room_id", "room_name", "view_points",
}


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

    replays = []
    for i, ep in enumerate(raw["episodes"][:MAX_EPISODES]):
        replays.append(ep.get("reference_replay", []))

        clean = {k: v for k, v in ep.items() if k in EPISODE_KNOWN}
        clean["episode_id"]           = str(i)
        clean["scene_id"]             = SCENE_GLB
        clean["scene_dataset_config"] = "default"
        clean["goals"]                = []

        episode = ObjectGoalNavEpisode(**clean)
        episode.goals = dataset.goals_by_category.get(episode.goals_key, [])
        dataset.episodes.append(episode)

    return dataset, replays


def main():
    dataset, replays = build_dataset()
    print(f"Loaded {len(dataset.episodes)} episodes from scene 1S7LAXRdDqK")

    config = habitat.get_config(
        "benchmark/nav/objectnav/objectnav_hm3d.yaml",
        overrides=[
            "habitat.environment.max_episode_steps=500",
            "habitat.simulator.habitat_sim_v0.gpu_device_id=0",
        ],
    )

    results = []
    with habitat.Env(config=config, dataset=dataset) as env:
        for ep_idx in range(len(dataset.episodes)):
            obs     = env.reset()
            episode = env.current_episode
            replay  = replays[ep_idx]

            print(f"\n{'='*60}")
            print(f"Episode {ep_idx} | Target: {episode.object_category}")
            print(f"  Goals: {len(episode.goals)} | Start: {episode.start_position}")

            step = 0
            for t, action_step in enumerate(replay):
                if env.episode_over:
                    break

                action = action_step["action"]

                if t == 0 and action == "STOP":
                    continue

                if action == "STOP":
                    env.step("stop")
                    step += 1
                    break

                hab_action = ACTION_MAP.get(action)
                if hab_action:
                    env.step(hab_action)
                    step += 1

            metrics = env.get_metrics()
            success = metrics.get("success", False)
            spl     = metrics.get("spl", 0.0)
            dtg     = metrics.get("distance_to_goal", float("nan"))

            print(f"  Steps:            {step}")
            print(f"  Success:          {success}")
            print(f"  SPL:              {spl:.4f}")
            print(f"  Distance to goal: {dtg:.3f} m")
            results.append({"success": success, "spl": spl, "dtg": dtg})

    print(f"\n{'='*60}")
    print(f"Summary over {len(results)} episodes:")
    print(f"  Success rate: {sum(r['success'] for r in results) / len(results):.2%}")
    print(f"  Mean SPL:     {sum(r['spl'] for r in results) / len(results):.4f}")
    print(f"  Mean DTG:     {sum(r['dtg'] for r in results) / len(results):.3f} m")


if __name__ == "__main__":
    main()
