"""
Headless validation: scene.usda를 Isaac Sim에서 로드하고
articulation 구조 + joint 동작 여부를 CLI로 검증.

Run inside Isaac Sim container (same volume mounts as urdf_to_usda.py):
  docker run --rm --gpus all -e ACCEPT_EULA=Y \
    --entrypoint /isaac-sim/python.sh \
    -v /home/gpuadmin/capstone/USDNet:/workspace/USDNet \
    -v /home/gpuadmin/docker/isaac-sim/cache/main:/isaac-sim/.cache \
    -v /home/gpuadmin/docker/isaac-sim/data:/isaac-sim/.local/share/ov/data \
    -v /home/gpuadmin/docker/isaac-sim/pkg:/isaac-sim/.local/share/ov/pkg \
    -v /home/gpuadmin/docker/isaac-sim/cache/computecache:/isaac-sim/.nv/ComputeCache \
    -v /home/gpuadmin/docker/isaac-sim/config:/isaac-sim/.nvidia-omniverse/config \
    -v /home/gpuadmin/docker/isaac-sim/logs:/isaac-sim/.nvidia-omniverse/logs \
    nvcr.io/nvidia/isaac-sim:5.1.0 \
    /workspace/USDNet/REACT3D/validate_scene_usda.py \
    --usda /workspace/USDNet/REACT3D/data/example_input_scene/usda/scene_v2.usda
"""
import argparse

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True, "open_usd": None})

import numpy as np
import omni.usd
import pxr.UsdPhysics as UsdPhysics
import pxr.UsdGeom as UsdGeom
from isaacsim.core.api import SimulationContext
from omni.isaac.dynamic_control import _dynamic_control


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usda", required=True)
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()

    print(f"\n{'='*60}", flush=True)
    print(f"Loading: {args.usda}", flush=True)
    print(f"{'='*60}\n", flush=True)

    omni.usd.get_context().open_stage(args.usda)
    for _ in range(5):
        simulation_app.update()

    stage = omni.usd.get_context().get_stage()
    if not stage:
        print("ERROR: Failed to open stage", flush=True)
        simulation_app.close()
        return

    print(f"Stage opened OK. Up axis: {UsdGeom.GetStageUpAxis(stage)}", flush=True)

    # 구조 분석
    art_roots, revolute, prismatic = [], [], []
    empty_axis_joints = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            art_roots.append(str(prim.GetPath()))
        if prim.IsA(UsdPhysics.RevoluteJoint):
            revolute.append(prim)
        if prim.IsA(UsdPhysics.PrismaticJoint):
            prismatic.append(prim)

    print(f"\n[Structure]", flush=True)
    print(f"  ArticulationRoot prims : {len(art_roots)}", flush=True)
    for p in art_roots:
        print(f"    {p}", flush=True)
    print(f"  RevoluteJoints   : {len(revolute)}", flush=True)
    print(f"  PrismaticJoints  : {len(prismatic)}", flush=True)

    # axis 상태 출력
    print(f"\n[Joint axes]", flush=True)
    all_joints = revolute + prismatic
    for prim in sorted(all_joints, key=lambda p: str(p.GetPath())):
        axis = prim.GetAttribute("physics:axis").Get()
        lower = prim.GetAttribute("physics:lowerLimit").Get()
        upper = prim.GetAttribute("physics:upperLimit").Get()
        jtype = "revolute" if prim.IsA(UsdPhysics.RevoluteJoint) else "prismatic"
        ok = "OK" if axis else "EMPTY"
        print(f"  [{ok}] {str(prim.GetPath()).split('/')[-1]:30s} "
              f"[{jtype:9s}] axis={axis!r:4} limits=[{lower:.3f}, {upper:.3f}]", flush=True)
        if not axis:
            empty_axis_joints.append(str(prim.GetPath()))

    if empty_axis_joints:
        print(f"\n[WARNING] {len(empty_axis_joints)} joint(s) still have empty axis!", flush=True)
    else:
        print(f"\n[OK] All joints have non-empty physics:axis", flush=True)

    # physx articulation interface로 joint 동작 테스트
    print(f"\n[Simulation Test] {args.steps} steps ...", flush=True)
    try:
        sim = SimulationContext(stage_units_in_meters=1.0)
        sim.initialize_physics()
        sim.play()
        simulation_app.update()

        dc = _dynamic_control.acquire_dynamic_control_interface()

        # articulation handle 조회 (dynamic_control API)
        art_handles = {}
        for root_path in art_roots:
            handle = dc.get_articulation(root_path)
            if handle != _dynamic_control.INVALID_HANDLE:
                dof_count = dc.get_articulation_dof_count(handle)
                art_handles[root_path] = (handle, dof_count)
                print(f"  Articulation: {root_path}  dofs={dof_count}", flush=True)
            else:
                print(f"  Articulation: {root_path}  handle=INVALID", flush=True)

        if not art_handles:
            print("  No valid articulation handles — 구조 오류 또는 physics 미적용", flush=True)
        else:
            # 첫 번째 articulation에 position target 설정
            first_path, (handle, dof_count) = next(iter(art_handles.items()))
            print(f"\n  Testing {first_path} (dofs={dof_count})", flush=True)

            dof_states = dc.get_articulation_dof_states(handle, _dynamic_control.STATE_POS)
            pos_before = list(dof_states["pos"])
            print(f"  Positions before: {pos_before}", flush=True)

            # DOF 0에 target 0.5 rad/m 설정
            dc.wake_up_articulation(handle)
            for i in range(dof_count):
                dc.set_dof_position_target(
                    dc.get_articulation_dof(handle, i),
                    0.5 if i == 0 else 0.0
                )

            for _ in range(args.steps):
                sim.step()
                simulation_app.update()

            dof_states = dc.get_articulation_dof_states(handle, _dynamic_control.STATE_POS)
            pos_after = list(dof_states["pos"])
            print(f"  Positions after : {pos_after}", flush=True)

            delta = max(abs(a - b) for a, b in zip(pos_after, pos_before))
            if delta > 1e-4:
                print(f"\n[PASS] Joint moved (max delta={delta:.4f}). Articulation works.", flush=True)
            else:
                print(f"\n[FAIL] Joint did NOT move (max delta={delta:.6f}).", flush=True)

        sim.stop()

    except Exception as e:
        import traceback
        print(f"\n[ERROR] {e}", flush=True)
        traceback.print_exc()

    print(f"\n{'='*60}", flush=True)
    print("Validation done.", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
