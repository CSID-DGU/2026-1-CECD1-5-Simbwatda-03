"""
Convert REACT3D URDF outputs to a single combined USDA for Isaac Sim 5.1.

Reads URDFs from urdf_fixed/ (absolute paths) and combines all robots into
one scene.usda. Each articulated mesh is imported at /World/<mesh_name> and
the static background at /World/remain_scene.

Run inside Isaac Sim container:
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
    /workspace/USDNet/REACT3D/urdf_to_usda.py \
    --urdf_dir /workspace/USDNet/REACT3D/data/example_input_scene/urdf_fixed \
    --output /workspace/USDNet/REACT3D/data/example_input_scene/usda/scene_v2.usda
"""
import argparse
import os
import tempfile
import shutil
import xml.etree.ElementTree as ET

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True, "open_usd": None})

import omni.usd
import pxr.UsdGeom as UsdGeom
import pxr.UsdPhysics as UsdPhysics
from isaacsim.asset.importer.urdf import _urdf


def parse_urdf_joint_axes(urdf_path):
    """URDF에서 각 joint의 axis xyz 벡터를 반환. {joint_name: [x, y, z]}"""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    axes = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        axis_elem = joint.find("axis")
        if axis_elem is not None:
            xyz = [float(v) for v in axis_elem.get("xyz", "1 0 0").split()]
            axes[name] = xyz
    return axes


def fix_joint_axes(stage, joint_axes):
    """
    physics:axis = "" 인 joint를 URDF axis 기반으로 수정.
    dominant axis (abs값 최대) → "X" / "Y" / "Z".
    """
    fixed = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            attr = prim.GetAttribute("physics:axis")
            if attr and attr.Get() == "":
                jname = prim.GetName()
                if jname in joint_axes:
                    xyz = joint_axes[jname]
                    dominant = max(range(3), key=lambda i: abs(xyz[i]))
                    token = ["X", "Y", "Z"][dominant]
                    attr.Set(token)
                    fixed += 1
    return fixed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    urdf_interface = _urdf.acquire_urdf_interface()

    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.import_inertia_tensor = True
    import_config.fix_base = False
    import_config.make_default_prim = False
    import_config.self_collision = False
    import_config.create_physics_scene = False

    omni.usd.get_context().new_stage()
    simulation_app.update()

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetMetadata("metersPerUnit", 1.0)

    world = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world)
    stage.DefinePrim("/World/physicsScene", "PhysicsScene")

    urdf_files = sorted([f for f in os.listdir(args.urdf_dir) if f.endswith(".urdf")])
    print(f"Found {len(urdf_files)} URDF files", flush=True)

    # 임시 디렉토리에 robot name을 mesh 이름으로 수정한 URDF를 저장
    tmpdir = tempfile.mkdtemp(prefix="react3d_urdf_")
    print(f"Temp dir: {tmpdir}", flush=True)

    all_joint_axes = {}

    try:
        for fname in urdf_files:
            mesh_name = fname.replace(".urdf", "")
            src_path = os.path.join(args.urdf_dir, fname)

            # URDF robot name을 mesh_name으로 변경
            tree = ET.parse(src_path)
            root = tree.getroot()
            root.set("name", mesh_name)

            # joint axis 정보 수집
            axes = parse_urdf_joint_axes(src_path)
            all_joint_axes.update(axes)

            # 임시 파일 저장 (ET.write는 xml_declaration 필요)
            tmp_path = os.path.join(tmpdir, fname)
            tree.write(tmp_path, xml_declaration=True, encoding="unicode")

            # dest_path = "/World": robot name이 subdirectory → /World/mesh_000
            dest_path = "/World"
            print(f"Importing {fname} (robot={mesh_name}) -> {dest_path}/{mesh_name}", flush=True)

            robot = urdf_interface.parse_urdf(tmpdir, fname, import_config)
            prim_path = urdf_interface.import_robot(tmpdir, fname, robot, import_config, dest_path)
            simulation_app.update()

            if prim_path:
                print(f"  OK: {prim_path}", flush=True)
            else:
                print(f"  FAILED: {fname}", flush=True)

        # axis post-process
        print(f"\nFixing empty physics:axis ...", flush=True)
        fixed = fix_joint_axes(stage, all_joint_axes)
        print(f"  Fixed {fixed} joint(s)", flush=True)

        for _ in range(5):
            simulation_app.update()

        stage.Export(args.output)
        print(f"\nSaved: {args.output}", flush=True)
        print("Done.", flush=True)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
