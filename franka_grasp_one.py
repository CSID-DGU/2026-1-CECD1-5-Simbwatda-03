"""
Step A3v3: Franka grasp + pull/swing using RMPFlowController (more robust than CCD IK).

Phases:
  0  settle
  1  pre-grasp (15cm in front of handle)
  2  approach (handle)
  3  close gripper
  4  pull (revolute: 30deg arc; prismatic: linear along outward_normal)
PASS if |cabinet_pos delta| > 0.05.
"""
import argparse
import json
import numpy as np

from omni.isaac.kit import SimulationApp
sa = SimulationApp({"headless": True, "open_usd": None})

import omni.usd
from pxr import UsdPhysics
from isaacsim.core.api import World
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.dynamic_control import _dynamic_control


def rot_from_z(z_target, up_hint=np.array([0, 0, 1.0])):
    z = np.asarray(z_target, dtype=float)
    z = z / np.linalg.norm(z)
    if abs(np.dot(z, up_hint)) > 0.95:
        up_hint = np.array([1.0, 0, 0])
    x = np.cross(up_hint, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def mat_to_quat_wxyz(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        return np.array([0.25 * S,
                         (R[2, 1] - R[1, 2]) / S,
                         (R[0, 2] - R[2, 0]) / S,
                         (R[1, 0] - R[0, 1]) / S])
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return np.array([(R[2, 1] - R[1, 2]) / S,
                         0.25 * S,
                         (R[0, 1] + R[1, 0]) / S,
                         (R[0, 2] + R[2, 0]) / S])
    if R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return np.array([(R[0, 2] - R[2, 0]) / S,
                         (R[0, 1] + R[1, 0]) / S,
                         0.25 * S,
                         (R[1, 2] + R[2, 1]) / S])
    S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return np.array([(R[1, 0] - R[0, 1]) / S,
                     (R[0, 2] + R[2, 0]) / S,
                     (R[1, 2] + R[2, 1]) / S,
                     0.25 * S])


def axis_angle_rotate(v, axis, theta):
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(theta), np.sin(theta)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--usda", required=True)
    p.add_argument("--handles", required=True)
    p.add_argument("--mesh", default="mesh_000")
    p.add_argument("--standoff", type=float, default=0.5)
    p.add_argument("--base_z_offset", type=float, default=0.3)
    p.add_argument("--pre_grasp_dist", type=float, default=0.15)
    p.add_argument("--pull_dist", type=float, default=0.15)
    p.add_argument("--swing_deg", type=float, default=30.0)
    p.add_argument("--steps_per_phase", type=int, default=200)
    args = p.parse_args()

    with open(args.handles) as f:
        handles = json.load(f)
    h = handles[args.mesh]
    handle = np.array(h["handle_center"])
    normal = np.array(h["outward_normal"])
    hinge_origin = np.array(h["hinge_origin"])
    hinge_axis = np.array(h["hinge_axis"])
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

    print(f"Franka base pos={franka_pos.tolist()} yaw_deg={np.degrees(yaw):.2f}", flush=True)

    world = World(stage_units_in_meters=1.0)
    franka = world.scene.add(Franka(
        prim_path="/World/franka",
        name="franka",
        position=franka_pos,
        orientation=quat_base,
    ))
    world.reset()
    sa.update()

    controller = RMPFlowController(name="grasp_ctrl", robot_articulation=franka)
    art_ctrl = franka.get_articulation_controller()

    R_grip = rot_from_z(-normal)
    quat_grip = mat_to_quat_wxyz(R_grip)

    dc = _dynamic_control.acquire_dynamic_control_interface()
    cabinet_handle = None
    cabinet_root_path = None
    for prim in stage.Traverse():
        pp = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI) and args.mesh in pp:
            cabinet_root_path = pp
            cabinet_handle = dc.get_articulation(pp)
            if cabinet_handle != _dynamic_control.INVALID_HANDLE:
                break
    if cabinet_handle is None or cabinet_handle == _dynamic_control.INVALID_HANDLE:
        print(f"ERROR: cabinet articulation for {args.mesh} not found", flush=True)
        sa.close()
        return
    print(f"cabinet root={cabinet_root_path}", flush=True)

    def cabinet_pos():
        s = dc.get_articulation_dof_states(cabinet_handle, _dynamic_control.STATE_POS)
        return float(s["pos"][0])

    def ee_pos():
        try:
            pos, _ = franka.end_effector.get_world_pose()
            return [float(x) for x in pos]
        except Exception:
            return [None, None, None]

    def run_phase(target_pos, target_quat, steps, label):
        for _ in range(steps):
            try:
                actions = controller.forward(
                    target_end_effector_position=np.asarray(target_pos),
                    target_end_effector_orientation=np.asarray(target_quat),
                )
                art_ctrl.apply_action(actions)
            except Exception as e:
                print(f"  RMP err [{label}]: {e}", flush=True)
                break
            world.step(render=False)

    def gripper(close=True):
        target = 0.0 if close else 0.04
        pos = franka.get_joint_positions()
        pos[-2] = target
        pos[-1] = target
        franka.set_joint_positions(pos)
        for _ in range(40):
            world.step(render=False)

    for _ in range(30):
        world.step(render=False)
    print(f"phase 0 settle: cabinet={cabinet_pos():.4f} ee={ee_pos()}", flush=True)

    pre = handle + normal * args.pre_grasp_dist
    run_phase(pre, quat_grip, args.steps_per_phase, "pre_grasp")
    print(f"phase 1 pre-grasp: cabinet={cabinet_pos():.4f} ee={ee_pos()}", flush=True)

    run_phase(handle, quat_grip, args.steps_per_phase, "approach")
    print(f"phase 2 approach: cabinet={cabinet_pos():.4f} ee={ee_pos()}", flush=True)

    gripper(close=True)
    print(f"phase 3 close: cabinet={cabinet_pos():.4f} ee={ee_pos()}", flush=True)

    if jtype == "prismatic":
        pull_target = handle + normal * args.pull_dist
    else:
        v = handle - hinge_origin
        v_rot = axis_angle_rotate(v, hinge_axis, np.deg2rad(args.swing_deg))
        pull_target = hinge_origin + v_rot
    run_phase(pull_target, quat_grip, args.steps_per_phase * 2, "pull")
    print(f"phase 4 pull: cabinet={cabinet_pos():.4f} ee={ee_pos()}", flush=True)

    final = cabinet_pos()
    status = "PASS" if abs(final) > 0.05 else "FAIL"
    print(f"\nFINAL [{args.mesh}] jtype={jtype} cabinet_final={final:.4f} {status}", flush=True)

    sa.close()


if __name__ == "__main__":
    main()
