"""
Step A1: Load scene_baseline_aug.usda, add Franka 70cm in front of target mesh handle.
No grasp yet — just standby and report Franka world pose for sanity check.

Run inside docker isaac-sim 5.1.
"""
import argparse
import json
import numpy as np

from omni.isaac.kit import SimulationApp
sa = SimulationApp({"headless": True, "open_usd": None})

import omni.usd
from pxr import UsdGeom
from isaacsim.core.api import World
from isaacsim.robot.manipulators.examples.franka import Franka


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usda", required=True)
    parser.add_argument("--handles", required=True, help="handle_poses.json")
    parser.add_argument("--mesh", default="mesh_000")
    parser.add_argument("--standoff", type=float, default=0.7,
                        help="Franka base distance from handle along outward_normal")
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()

    with open(args.handles) as f:
        handles = json.load(f)
    if args.mesh not in handles:
        print(f"ERROR: {args.mesh} not in handles", flush=True)
        sa.close()
        return
    h = handles[args.mesh]
    handle = np.array(h["handle_center"])
    normal = np.array(h["outward_normal"])

    print(f"[mesh={args.mesh}] handle={handle.tolist()}", flush=True)
    print(f"[mesh={args.mesh}] outward_normal={normal.tolist()}", flush=True)

    omni.usd.get_context().open_stage(args.usda)
    for _ in range(5):
        sa.update()
    stage = omni.usd.get_context().get_stage()
    print(f"stage up_axis={UsdGeom.GetStageUpAxis(stage)}", flush=True)

    # Franka base: handle + standoff * outward_normal, floored to handle.z - 0.9
    franka_pos = handle + normal * args.standoff
    franka_pos[2] = max(0.0, float(handle[2]) - 0.9)

    # yaw so Franka +X faces -outward_normal (toward cabinet)
    yaw = float(np.arctan2(-normal[1], -normal[0]))
    half = yaw / 2.0
    quat_wxyz = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])

    print(f"Franka base position={franka_pos.tolist()}", flush=True)
    print(f"Franka base yaw={np.degrees(yaw):.2f}deg quat_wxyz={quat_wxyz.tolist()}", flush=True)

    world = World(stage_units_in_meters=1.0)
    franka = world.scene.add(Franka(
        prim_path="/World/franka",
        name="franka",
        position=franka_pos,
        orientation=quat_wxyz,
    ))
    world.reset()
    sa.update()

    for i in range(args.steps):
        world.step(render=False)
        if i % 30 == 0:
            try:
                ee_pos, ee_rot = franka.end_effector.get_world_pose()
                print(f"step {i}: ee_pos={list(ee_pos)}", flush=True)
            except Exception as e:
                print(f"step {i}: ee_pos err {e}", flush=True)

    print("A1 standby done", flush=True)
    sa.close()


if __name__ == "__main__":
    main()
