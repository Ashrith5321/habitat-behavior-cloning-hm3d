"""
Pure replay of PIRLNav episode 1 from 1S7LAXRdDqK.
Target: find a sofa in the scene.
"""
import gzip
import json
import os
import sys
import numpy as np
import quaternion
import imageio
import habitat_sim


# ═══ YOUR PATHS ══════════════════════════════════════════════════════════════
EPISODE_FILE  = "/home/ashed/data/datasets/objectnav/objectnav_hm3d_hd/train/content/1S7LAXRdDqK.json.gz"
SCENE_FILE    = "/home/ashed/Documents/hm3d-train-habitat-v0.2/00744-1S7LAXRdDqK/1S7LAXRdDqK.basis.glb"
EPISODE_INDEX = 1
OUTPUT_VIDEO  = "/home/ashed/replay_ep1.mp4"
FPS           = 10
# ════════════════════════════════════════════════════════════════════════════


# ── Verify paths ─────────────────────────────────────────────────────────────
for p, name in [(EPISODE_FILE, "Episode JSON"), (SCENE_FILE, "Scene GLB")]:
    if not os.path.exists(p):
        sys.exit(f"❌ {name} not found: {p}")
print("✅ All files present")


# ── Load episode ─────────────────────────────────────────────────────────────
with gzip.open(EPISODE_FILE, "rt") as f:
    data = json.load(f)

episode = data["episodes"][EPISODE_INDEX]
print(f"\nEpisode {EPISODE_INDEX}:")
print(f"  ID:       {episode['episode_id']}")
print(f"  Target:   {episode['object_category']}")
print(f"  Start:    {episode['start_position']}")
print(f"  Rotation: {episode['start_rotation']}")
print(f"  Actions:  {len(episode['reference_replay'])}")


# ── Simulator config ─────────────────────────────────────────────────────────
sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = SCENE_FILE
sim_cfg.enable_physics = False

rgb_spec = habitat_sim.CameraSensorSpec()
rgb_spec.uuid = "rgb"
rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
rgb_spec.resolution = [480, 640]
rgb_spec.position = [0.0, 1.25, 0.0]   # camera at eye level

agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = [rgb_spec]
agent_cfg.action_space = {
    "move_forward": habitat_sim.agent.ActionSpec(
        "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)),
    "turn_left": habitat_sim.agent.ActionSpec(
        "turn_left",    habitat_sim.agent.ActuationSpec(amount=30.0)),
    "turn_right": habitat_sim.agent.ActionSpec(
        "turn_right",   habitat_sim.agent.ActuationSpec(amount=30.0)),
    "look_up": habitat_sim.agent.ActionSpec(
        "look_up",      habitat_sim.agent.ActuationSpec(amount=30.0)),
    "look_down": habitat_sim.agent.ActionSpec(
        "look_down",    habitat_sim.agent.ActuationSpec(amount=30.0)),
}

ACTION_MAP = {
    "MOVE_FORWARD": "move_forward",
    "TURN_LEFT":    "turn_left",
    "TURN_RIGHT":   "turn_right",
    "LOOK_UP":      "look_up",
    "LOOK_DOWN":    "look_down",
}


# ── Start sim ────────────────────────────────────────────────────────────────
print("\nLoading simulator...")
sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

# Place agent at recorded start
agent = sim.get_agent(0)
state = habitat_sim.agent.AgentState()
state.position = np.array(episode["start_position"], dtype=np.float32)

# JSON stores rotation as [qx, qy, qz, qw], numpy-quaternion wants [qw, qx, qy, qz]
qx, qy, qz, qw = episode["start_rotation"]
state.rotation = np.quaternion(qw, qx, qy, qz)
agent.set_state(state)
print("Agent placed at start position")


# ── Replay ───────────────────────────────────────────────────────────────────
print(f"\nReplaying {len(episode['reference_replay'])} actions...")
frames = []

for t, step in enumerate(episode["reference_replay"]):
    action = step["action"]

    # Skip leading STOP quirk
    if t == 0 and action == "STOP":
        continue
    if action == "STOP":
        print(f"  [{t}] STOP — episode complete")
        break

    habitat_action = ACTION_MAP.get(action)
    if habitat_action is None:
        print(f"  [{t}] Unknown action '{action}', skipping")
        continue

    obs = sim.step(habitat_action)
    frames.append(obs["rgb"][:, :, :3])

    if t % 10 == 0:
        print(f"  [{t:3d}/{len(episode['reference_replay'])}] {action}")


# ── Save video ───────────────────────────────────────────────────────────────
if frames:
    print(f"\nSaving {len(frames)} frames to {OUTPUT_VIDEO}...")
    imageio.mimsave(OUTPUT_VIDEO, frames, fps=FPS)
    print(f"✅ Done! Video: {OUTPUT_VIDEO}")
    print(f"   Duration: {len(frames)/FPS:.1f} seconds")
else:
    print("⚠️  No frames captured")

sim.close()