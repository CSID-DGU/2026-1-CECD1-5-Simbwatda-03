"""
Batch fixed-constraint grasp for all 12 baseline meshes.
Single docker run, single stage open, iterate over meshes:
  - move Franka base near handle
  - create FixedJoint between panda_hand and <mesh>_part
  - apply position target to cabinet articulation
  - settle, measure delta, classify PASS/FAIL
  - delete FixedJoint, reset cabinet to 0, move on
Prints a final PASS/FAIL table.
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
    for prim in stage.Traverse():
        if prim.GetName() == f"{mesh_name}_part":
            return str(prim.GetPath())
    return None


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
    p.add_argument("--standoff", type=float, default=0.15)
    p.add_argument("--base_z_offset", type=float, default=0.3)
    p.add_argument("--target_rev", type=float, default=0.8)
    p.add_argument("--target_pris", type=float, default=0.3)
    p.add_argument("--settle_steps", type=int, default=60)
    p.add_argument("--drive_steps", type=int, default=180)
    args = p.parse_args()

    with open(args.handles) as f:
        handles = json.load(f)
    mesh_names = sorted(handles.keys())

    omni.usd.get_context().open_stage(args.usda)
    for _ in range(5):
        sa.update()
    stage = omni.usd.get_context().get_stage()

    # Franka first placed at first mesh handle
    first = handles[mesh_names[0]]
    init_handle = np.array(first["handle_center"])
    init_normal = np.array(first["outward_normal"])
    init_pos = init_handle + init_normal * args.standoff
    init_pos[2] = max(0.0, float(init_handle[2]) - args.base_z_offset)
    init_yaw = float(np.arctan2(-init_normal[1], -init_normal[0]))
    init_quat = np.array([np.cos(init_yaw / 2), 0.0, 0.0, np.sin(init_yaw / 2)])

    world = World(stage_units_in_meters=1.0)
    franka = world.scene.add(Franka(
        prim_path="/World/franka",
        name="franka",
        position=init_pos,
        orientation=init_quat,
    ))
    world.reset()
    sa.update()

    hand_path = "/World/franka/panda_hand"
    if stage.GetPrimAtPath(hand_path) is None:
        print(f"ERROR: hand prim not found at {hand_path}", flush=True)
        sa.close()
        return

    dc = _dynamic_control.acquire_dynamic_control_interface()
    results = []
    joint_path = "/World/grasp_constraint"

    for mesh in mesh_names:
        print(f"\n--- {mesh} ---", flush=True)
        h = handles[mesh]
        handle = np.array(h["handle_center"])
        normal = np.array(h["outward_normal"])
        jtype = h["joint_type"]

        # move Franka base
        franka_pos = handle + normal * args.standoff
        franka_pos[2] = max(0.0, float(handle[2]) - args.base_z_offset)
        yaw = float(np.arctan2(-normal[1], -normal[0]))
        quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        try:
            franka.set_world_pose(position=franka_pos, orientation=quat)
        except Exception:
            franka.set_local_pose(translation=franka_pos, orientation=quat)

        # settle
        for _ in range(args.settle_steps):
            world.step(render=False)

        cabinet_root = find_articulation_root(stage, mesh)
        part_link = find_part_link(stage, mesh)
        if not (cabinet_root and part_link):
            print(f"  no cabinet root or part link", flush=True)
            results.append((mesh, jtype, "no_articulation", 0.0, 0.0))
            continue
        cab_h = dc.get_articulation(cabinet_root)
        if cab_h == _dynamic_control.INVALID_HANDLE:
            print(f"  invalid articulation handle", flush=True)
            results.append((mesh, jtype, "no_handle", 0.0, 0.0))
            continue

        # create FixedJoint
        if stage.GetPrimAtPath(joint_path):
            stage.RemovePrim(joint_path)
            sa.update()
        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(hand_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(part_link)])
        sa.update()

        # apply target
        target = args.target_rev if jtype == "revolute" else args.target_pris
        dc.wake_up_articulation(cab_h)
        dc.set_dof_position_target(dc.get_articulation_dof(cab_h, 0), target)

        pos0 = float(dc.get_articulation_dof_states(cab_h, _dynamic_control.STATE_POS)["pos"][0])
        for _ in range(args.drive_steps):
            world.step(render=False)
        pos1 = float(dc.get_articulation_dof_states(cab_h, _dynamic_control.STATE_POS)["pos"][0])
        delta = abs(pos1 - pos0)
        status = "PASS" if delta > 0.05 else "FAIL"
        print(f"  jtype={jtype} target={target:.3f} pos0={pos0:.4f} pos1={pos1:.4f} delta={delta:.4f} {status}", flush=True)
        results.append((mesh, jtype, status, target, delta))

        # reset: remove fixed joint, drive cabinet back to 0
        stage.RemovePrim(joint_path)
        sa.update()
        dc.set_dof_position_target(dc.get_articulation_dof(cab_h, 0), 0.0)
        for _ in range(args.settle_steps):
            world.step(render=False)

    # summary
    print(f"\n{'='*70}", flush=True)
    print(f"{'mesh':14s} {'jtype':10s} {'status':14s} {'target':>8s} {'delta':>8s}", flush=True)
    print(f"{'-'*70}", flush=True)
    p_count = 0
    n = 0
    for m, jt, st, tg, dl in results:
        print(f"{m:14s} {jt:10s} {st:14s} {tg:8.3f} {dl:8.4f}", flush=True)
        if st in ("PASS", "FAIL"):
            n += 1
            if st == "PASS":
                p_count += 1
    print(f"\nSummary: {p_count}/{n} PASS", flush=True)

    sa.close()


if __name__ == "__main__":
    main()
