"""
Print DOF names and indices for spot_with_arm.usd.

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
    /workspace/USDNet/REACT3D/spot_dof_info.py
"""
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.prims import define_prim
from isaacsim.storage.native import get_assets_root_path

my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 500, rendering_dt=1 / 50)
assets_root_path = get_assets_root_path()

spot_usd = assets_root_path + "/Isaac/Robots/BostonDynamics/spot/spot_with_arm.usd"
print(f"\nLoading: {spot_usd}", flush=True)

prim = define_prim("/World/Spot", "Xform")
prim.GetReferences().AddReference(spot_usd)

robot = SingleArticulation(prim_path="/World/Spot", name="spot_arm", position=np.array([0, 0, 0.8]))
my_world.reset()
robot.initialize()

print(f"\nDOF count: {robot.num_dof}", flush=True)
print(f"\nDOF names:", flush=True)
for i, name in enumerate(robot.dof_names):
    pos = robot.get_joint_positions()[i]
    print(f"  [{i:2d}] {name:40s}  pos={pos:.4f}", flush=True)

print("\nDone.", flush=True)
simulation_app.close()
