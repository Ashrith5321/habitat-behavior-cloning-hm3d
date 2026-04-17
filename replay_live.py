"""
Live replay viewer - shows agent first-person view in a window
as the human demonstration plays back.
"""
import gzip
import json
import os
import sys
import numpy as np
import quaternion
import imageio
import habitat_sim
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# ═══ PATHS ═══════════════════════════════════════════════════════════════════
EPISODE_FILE  = "/home/ashed/data/datasets/objectnav/objectnav_hm3d_hd/train/content/1S7LAXRdDqK.json.gz"
SCENE_FILE    = "/home/ashed/Documents/hm3d-train-habitat-v0.2/00744-1S7LAXRdDqK/1S7LAXRdDqK.basis.glb"
EPISODE_INDEX = 10
OUTPUT_VIDEO  = "/home/ashed/replay_live_ep1.mp4"
FPS           = 10
# ════════════════════════════════════════════════════════════════════════════


# Load episode
for p in [EPISODE_FILE, SCENE_FILE]:
    if not os.path.exists(p):
        sys.exit(f"Not found: {p}")

with gzip.open(EPISODE_FILE, "rt") as f:
    data = json.load(f)

episode = data["episodes"][EPISODE_INDEX]
print(f"Target:  {episode['object_category']}")
print(f"Actions: {len(episode['reference_replay'])}")


# Simulator config
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


# Start simulator
print("Loading simulator...")
sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

agent = sim.get_agent(0)
state = habitat_sim.agent.AgentState()
state.position = np.array(episode["start_position"], dtype=np.float32)
qx, qy, qz, qw = episode["start_rotation"]
state.rotation = np.quaternion(qw, qx, qy, qz)
agent.set_state(state)
print("Agent placed at start")


# Setup live window
plt.ion()
fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor("black")
ax.axis("off")
im = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
title = ax.set_title("", color="white", fontsize=12, pad=8)
fig.canvas.manager.set_window_title(f"Habitat Replay - {episode['object_category']}")
plt.tight_layout()
plt.draw()
plt.pause(0.1)


# Replay loop
print("\nReplaying...")
frames = []
prev_action = None

for t, step in enumerate(episode["reference_replay"]):
    action = step["action"]

    if t == 0 and action == "STOP":
        continue
    if action == "STOP":
        print(f"  [{t}] STOP - done")
        break

    habitat_action = ACTION_MAP.get(action)
    if habitat_action is None:
        continue

    obs = sim.step(habitat_action)
    frame = obs["rgb"][:, :, :3]
    frames.append(frame)

    # Check inflection
    is_inflection = (prev_action is not None and action != prev_action)
    prev_action = action

    # Update window
    im.set_data(frame)
    label = f"Target: {episode['object_category'].upper()}  |  Step {t}/{len(episode['reference_replay'])}  |  {action}"
    if is_inflection:
        label += "  ← INFLECTION"
    title.set_text(label)
    title.set_color("orange" if is_inflection else "white")
    plt.draw()
    plt.pause(1.0 / FPS)

    if t % 10 == 0:
        print(f"  [{t:3d}] {action}{'  <- inflection' if is_inflection else ''}")


# Save video
print(f"\nSaving {len(frames)} frames...")
imageio.mimsave(OUTPUT_VIDEO, frames, fps=FPS)
print(f"Done! {OUTPUT_VIDEO}")

plt.ioff()
plt.show()
sim.close()