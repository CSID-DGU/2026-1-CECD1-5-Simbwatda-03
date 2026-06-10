"""
Headless validation: scene.usda를 Isaac Sim에서 로드하고
모든 articulation을 sequential로 sweep해서 PASS/FAIL 표를 출력.

Based on validate_scene_usda.py, extended to all articulations.

Run inside Isaac Sim docker:
  docker run --rm --gpus all -e ACCEPT_EULA=Y \\
    --entrypoint /isaac-sim/python.sh \\
    -v /home/gpuadmin/capstone:/home/gpuadmin/capstone \\
    -v /home/gpuadmin/docker/isaac-sim/cache/main:/isaac-sim/.cache \\
    -v /home/gpuadmin/docker/isaac-sim/data:/isaac-sim/.local/share/ov/data \\
    -v /home/gpuadmin/docker/isaac-sim/pkg:/isaac-sim/.local/share/ov/pkg \\
    -v /home/gpuadmin/docker/isaac-sim/cache/computecache:/isaac-sim/.nv/ComputeCache \\
    -v /home/gpuadmin/docker/isaac-sim/config:/isaac-sim/.nvidia-omniverse/config \\
    -v /home/gpuadmin/docker/isaac-sim/logs:/isaac-sim/.nvidia-omniverse/logs \\
    nvcr.io/nvidia/isaac-sim:5.1.0 \\
    /home/gpuadmin/capstone/USDNet/REACT3D/validate_scene_usda_all.py \\
    --usda /home/gpuadmin/capstone/REACT3D/data/example_input_scene/usda/scene_baseline.usda
"""
import argparse

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True, "open_usd": None})

import omni.usd
import pxr.UsdPhysics as UsdPhysics
from isaacsim.core.api import SimulationContext
from omni.isaac.dynamic_control import _dynamic_control


def get_joint_type_for_root(stage, root_path):
    """Return 'revolute' / 'prismatic' / None by scanning joints under root_path."""
    root = stage.GetPrimAtPath(root_path)
    if not root:
        return None
    for prim in root.GetAllChildren():
        for sub in (prim,) + tuple(prim.GetAllChildren()):
            if sub.IsA(UsdPhysics.RevoluteJoint):
                return "revolute"
            if sub.IsA(UsdPhysics.PrismaticJoint):
                return "prismatic"
    for sub in root.GetAllDescendants() if hasattr(root, "GetAllDescendants") else []:
        if sub.IsA(UsdPhysics.RevoluteJoint):
            return "revolute"
        if sub.IsA(UsdPhysics.PrismaticJoint):
            return "prismatic"
    # fallback via traverse
    for sub in stage.Traverse():
        sp = str(sub.GetPath())
        if not sp.startswith(root_path + "/"):
            continue
        if sub.IsA(UsdPhysics.RevoluteJoint):
            return "revolute"
        if sub.IsA(UsdPhysics.PrismaticJoint):
            return "prismatic"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usda", required=True)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--target_rev", type=float, default=0.5, help="Revolute target (rad)")
    parser.add_argument("--target_pris", type=float, default=0.2, help="Prismatic target (m)")
    parser.add_argument("--pass_threshold", type=float, default=0.05, help="Min |delta| to count PASS")
    args = parser.parse_args()

    print(f"\n{'='*70}", flush=True)
    print(f"Loading: {args.usda}", flush=True)
    print(f"{'='*70}\n", flush=True)

    omni.usd.get_context().open_stage(args.usda)
    for _ in range(5):
        simulation_app.update()

    stage = omni.usd.get_context().get_stage()
    if not stage:
        print("ERROR: Failed to open stage", flush=True)
        simulation_app.close()
        return

    art_roots = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            art_roots.append(str(prim.GetPath()))

    print(f"Found {len(art_roots)} articulation root(s)", flush=True)
    for p in art_roots:
        print(f"  {p}", flush=True)

    sim = SimulationContext(stage_units_in_meters=1.0)
    sim.initialize_physics()
    sim.play()
    simulation_app.update()

    dc = _dynamic_control.acquire_dynamic_control_interface()

    results = []
    for root_path in art_roots:
        handle = dc.get_articulation(root_path)
        if handle == _dynamic_control.INVALID_HANDLE:
            results.append((root_path, "n/a", "INVALID_HANDLE", 0, 0.0, 0.0))
            continue
        dof_count = dc.get_articulation_dof_count(handle)
        if dof_count == 0:
            results.append((root_path, "static", "no_dof", 0, 0.0, 0.0))
            continue

        jtype = get_joint_type_for_root(stage, root_path) or "unknown"
        target = args.target_rev if jtype == "revolute" else args.target_pris

        dc.wake_up_articulation(handle)
        states = dc.get_articulation_dof_states(handle, _dynamic_control.STATE_POS)
        pos_before = float(states["pos"][0])

        dc.set_dof_position_target(dc.get_articulation_dof(handle, 0), target)
        for _ in range(args.steps):
            sim.step()
            simulation_app.update()

        states = dc.get_articulation_dof_states(handle, _dynamic_control.STATE_POS)
        pos_after = float(states["pos"][0])
        delta = abs(pos_after - pos_before)
        status = "PASS" if delta > args.pass_threshold else "FAIL"
        results.append((root_path, jtype, status, dof_count, target, delta))

        # reset target to 0 and stabilize before next mesh
        dc.set_dof_position_target(dc.get_articulation_dof(handle, 0), 0.0)
        for _ in range(20):
            sim.step()
            simulation_app.update()

    sim.stop()

    print(f"\n{'='*70}", flush=True)
    print(f"Per-articulation sweep ({args.steps} steps, threshold={args.pass_threshold})", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  {'STATUS':9s} {'JTYPE':10s} {'TARGET':>8s} {'DELTA':>8s}  ROOT", flush=True)
    pass_count = 0
    movable_total = 0
    for path, jtype, status, dof, target, delta in results:
        print(f"  {status:9s} {jtype:10s} {target:8.4f} {delta:8.4f}  {path}", flush=True)
        if dof > 0:
            movable_total += 1
            if status == "PASS":
                pass_count += 1

    print(f"\nSummary: {pass_count}/{movable_total} movable articulation(s) PASS", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
