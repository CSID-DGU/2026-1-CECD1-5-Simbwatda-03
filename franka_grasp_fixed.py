"""
Step (2): Fixed-constraint pull.

Franka is placed near the cabinet handle. A PhysX FixedJoint is created
between Franka's panda_hand and the cabinet's part link, forming a closed
kinematic chain. A position target is then applied to the cabinet articulation,
which drags the Franka along (visual equivalent of a successful grasp+pull).
"""
import argparse
import json
import numpy as np

from omni.isaac.kit import SimulationApp
sa = SimulationApp({"headless": True, "open_usd": None})

import omni.usd
from pxr import UsdPhysics, Sdf
from isaacsim.core.api import World
from isaacsim.robot.manipulators.examples.franka import Franka
from omni.isaac.dynamic_control import _dynamic_control


def find_part_link(stage, mesh_name):
    """Find a prim path ending with <mesh>_part (a Xform) under the articulation."""
    candidates = []
    for prim in stage.Traverse():
        nm = prim.GetName()
        if nm == f"{mesh_name}_part":
            candidates.append(str(prim.GetPath()))
    return candidates[0] if candidates else None


def find_articulation_root(stage, mesh_name):
    for prim in stage.Traverse():
        pp = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI) and mesh_name in pp:
            return pp
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--usda", required=True)
    p.add_argument("--handles", required=True)
    p.add_argument("--mesh", default="mesh_000")
    p.add_argument("--standoff", type=float, default=0.15)
    p.add_argument("--base_z_offset", type=float, default=0.3)
    p.add_argument("--target_rev", type=float, default=0.8)
    p.add_argument("--target_pris", type=float, default=0.3)
    p.add_argument("--steps", type=int, default=300)
    args = p.parse_args()

    with open(args.handles) as f:
        handles = json.load(f)
    h = handles[args.mesh]
    handle = np.array(h["handle_center"])
    normal = np.array(h["outward_normal"])
    jtype = h["joint_type"]

    print(f"[mesh={args.mesh}] handle={handle.tolist()} jtype={jtype}", flush=True)

    omni.usd.get_context().open_stage(args.usda)
    for _ in range(5):
        sa.update()
    stage = omni.usd.get_context().get_stage()

    franka_pos = handle + normal * args.standoff
    franka_pos[2] = max(0.0, float(handle[2]) - args.base_z_offset)
    yaw = float(np.arctan2(-normal[1], -normal[0]))
    quat_base = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])

    print(f"Franka base pos={franka_pos.tolist()}", flush=True)

    world = World(stage_units_in_meters=1.0)
    franka = world.scene.add(Franka(
        prim_path="/World/franka",
        name="franka",
        position=franka_pos,
        orientation=quat_base,
    ))
    world.reset()
    sa.update()

    cabinet_root_path = find_articulation_root(stage, args.mesh)
    part_link_path = find_part_link(stage, args.mesh)
    print(f"cabinet root={cabinet_root_path}", flush=True)
    print(f"part link={part_link_path}", flush=True)

    if not (cabinet_root_path and part_link_path):
        print("ERROR: cabinet root or part link not found", flush=True)
        sa.close()
        return

    # settle
    for _ in range(60):
        world.step(render=False)
    ee_pos_before, _ = franka.end_effector.get_world_pose()
    print(f"ee before constraint: {list(ee_pos_before)}", flush=True)

    # candidate Franka hand prim paths to try
    hand_candidates = [
        "/World/franka/panda_hand",
        "/World/franka/panda_link8",
        "/World/franka/panda_leftfinger",
    ]
    hand_path = None
    for c in hand_candidates:
        if stage.GetPrimAtPath(c):
            hand_path = c
            break
    if hand_path is None:
        print("ERROR: panda hand prim not found", flush=True)
        sa.close()
        return
    print(f"hand prim={hand_path}", flush=True)

    # Create FixedJoint
    joint_path = "/World/grasp_constraint"
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(hand_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(part_link_path)])
    sa.update()
    print(f"FixedJoint created: {joint_path}", flush=True)

    # Apply target to cabinet articulation
    dc = _dynamic_control.acquire_dynamic_control_interface()
    cabinet_h = dc.get_articulation(cabinet_root_path)
    if cabinet_h == _dynamic_control.INVALID_HANDLE:
        print("ERROR: invalid cabinet articulation handle", flush=True)
        sa.close()
        return

    target = args.target_rev if jtype == "revolute" else args.target_pris
    dc.wake_up_articulation(cabinet_h)
    dc.set_dof_position_target(dc.get_articulation_dof(cabinet_h, 0), target)
    print(f"target set: {target}", flush=True)

    pos_before = float(dc.get_articulation_dof_states(cabinet_h, _dynamic_control.STATE_POS)["pos"][0])
    print(f"cabinet pos before: {pos_before:.4f}", flush=True)

    for i in range(args.steps):
        world.step(render=False)
        if i % 30 == 0:
            s = dc.get_articulation_dof_states(cabinet_h, _dynamic_control.STATE_POS)
            try:
                ep, _ = franka.end_effector.get_world_pose()
                ep = list(ep)
            except Exception:
                ep = None
            print(f"step {i}: cabinet={s['pos'][0]:.4f} ee={ep}", flush=True)

    final = float(dc.get_articulation_dof_states(cabinet_h, _dynamic_control.STATE_POS)["pos"][0])
    delta = abs(final - pos_before)
    status = "PASS" if delta > 0.05 else "FAIL"
    print(f"\nFINAL [{args.mesh}] jtype={jtype} target={target:.3f} cabinet_final={final:.4f} delta={delta:.4f} {status}", flush=True)

    sa.close()


if __name__ == "__main__":
    main()
