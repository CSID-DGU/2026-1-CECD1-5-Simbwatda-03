"""
Isaac Sim Nucleus에서 Spot Arm asset을 로컬 캐시로 다운로드하고
asset path와 DOF 정보를 출력.

Run:
  docker run --rm --gpus '"device=1"' -e ACCEPT_EULA=Y \
    --entrypoint /isaac-sim/python.sh \
    -v /home/gpuadmin/capstone/USDNet:/workspace/USDNet \
    -v /home/gpuadmin/docker/isaac-sim/cache/main:/isaac-sim/.cache \
    -v /home/gpuadmin/docker/isaac-sim/data:/isaac-sim/.local/share/ov/data \
    -v /home/gpuadmin/docker/isaac-sim/pkg:/isaac-sim/.local/share/ov/pkg \
    -v /home/gpuadmin/docker/isaac-sim/cache/computecache:/isaac-sim/.nv/ComputeCache \
    -v /home/gpuadmin/docker/isaac-sim/config:/isaac-sim/.nvidia-omniverse/config \
    -v /home/gpuadmin/docker/isaac-sim/logs:/isaac-sim/.nvidia-omniverse/logs \
    nvcr.io/nvidia/isaac-sim:5.1.0 \
    /workspace/USDNet/REACT3D/download_spot_arm.py
"""
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True, "open_usd": None})

import omni.usd
import omni.client
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.dynamic_control import _dynamic_control
from isaacsim.core.api import SimulationContext
import pxr.UsdGeom as UsdGeom


def list_nucleus_dir(path):
    result, entries = omni.client.list(path)
    if result == omni.client.Result.OK:
        return [e.relative_path for e in entries]
    return []


def main():
    assets_root = get_assets_root_path()
    print(f"\nAssets root: {assets_root}", flush=True)

    # Boston Dynamics 폴더 탐색
    bd_path = assets_root + "/Isaac/Robots/BostonDynamics"
    print(f"\nListing: {bd_path}", flush=True)
    entries = list_nucleus_dir(bd_path)
    for e in entries:
        print(f"  {e}", flush=True)

    # Spot Arm 후보 경로들
    candidates = [
        "/Isaac/Robots/BostonDynamics/spot/spot_with_arm.usd",
        "/Isaac/Robots/BostonDynamics/spot/spot.usd",
    ]

    found_path = None
    for c in candidates:
        full = assets_root + c
        result, _ = omni.client.stat(full)
        if result == omni.client.Result.OK:
            print(f"\n[FOUND] {full}", flush=True)
            found_path = full
            break
        else:
            print(f"  not found: {full}", flush=True)

    if not found_path:
        print("\n[ERROR] Spot Arm asset not found. Listing all Spot variants:", flush=True)
        for sub in entries:
            sub_entries = list_nucleus_dir(bd_path + "/" + sub)
            for se in sub_entries:
                print(f"  {sub}/{se}", flush=True)
        simulation_app.close()
        return

    # asset 로드해서 DOF 확인
    print(f"\nLoading {found_path} ...", flush=True)
    omni.usd.get_context().open_stage(found_path)
    for _ in range(10):
        simulation_app.update()

    stage = omni.usd.get_context().get_stage()
    print(f"Stage loaded. Up axis: {UsdGeom.GetStageUpAxis(stage)}", flush=True)

    sim = SimulationContext(stage_units_in_meters=1.0)
    sim.initialize_physics()
    sim.play()
    simulation_app.update()

    dc = _dynamic_control.acquire_dynamic_control_interface()

    import pxr.UsdPhysics as UsdPhysics
    art_roots = [
        str(p.GetPath()) for p in stage.Traverse()
        if p.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    print(f"\nArticulationRoot prims: {art_roots}", flush=True)

    for root_path in art_roots:
        handle = dc.get_articulation(root_path)
        if handle == _dynamic_control.INVALID_HANDLE:
            continue
        dof_count = dc.get_articulation_dof_count(handle)
        print(f"\n  path: {root_path}", flush=True)
        print(f"  dofs: {dof_count}", flush=True)
        for i in range(dof_count):
            dof = dc.get_articulation_dof(handle, i)
            props = dc.get_dof_properties(dof)
            print(f"    [{i}] type={props.type}  lower={props.lower:.3f}  upper={props.upper:.3f}", flush=True)

    sim.stop()
    print("\nDone.", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
