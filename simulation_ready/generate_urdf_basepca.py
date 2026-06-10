"""
Generate URDFs using base_pca.obj (PCA surface-normal aligned base).
No base_scene.obj, no inner_box_v3. Original inner_box.obj for translation only.
"""
import numpy as np
import os
from pathlib import Path
import shutil
import argparse


def generate_urdf_interactive(folder_path, output_urdf_path, obj_name):
    folder_path = Path(folder_path)

    urdf_parts = []
    urdf_parts.append("""
    <link name="world"/>""")

    origin_path = folder_path / "origin_world.npy"
    vector_path = folder_path / "vector_world.npy"
    base_obj_path = folder_path / "base_pca.obj"
    base_obj_fallback = folder_path / "base.obj"
    part_obj_path = folder_path / "part.dae"
    articulation_type_path = folder_path / "articulation_type.npy"
    inner_box_path = folder_path / "inner_box.obj"
    range_max_path = folder_path / "range_max.npy"

    if not base_obj_path.exists():
        base_obj_path = base_obj_fallback
    if not (origin_path.exists() and vector_path.exists() and part_obj_path.exists() and articulation_type_path.exists()):
        print(f"skip {folder_path}, lack of necessary files.")
        return

    origin = np.load(origin_path)
    vector = np.load(vector_path)
    vector = vector / np.linalg.norm(vector)

    joint_type = "revolute"
    lower_limit = "0"
    upper_limit = "2.0944"
    if articulation_type_path.exists():
        articulation_type = np.load(articulation_type_path)
        if isinstance(articulation_type, (bytes, np.bytes_)):
            articulation_type = articulation_type.decode("utf-8")
        if isinstance(articulation_type, np.ndarray):
            articulation_type = str(articulation_type.item())
        if articulation_type.lower() == "translation":
            joint_type = "prismatic"
            lower_limit = "0"
            upper_limit = "0.5"

    if range_max_path.exists():
        range_max = float(np.load(range_max_path).flat[0])
        upper_limit = str(round(range_max, 6))

    base_visuals = []
    if base_obj_path.exists():
        base_visuals.append(f"""    <visual>
        <geometry>
            <mesh filename="package://sp/{base_obj_path}" scale="1 1 1"/>
        </geometry>
    </visual>
    <collision>
        <geometry>
            <mesh filename="package://sp/{base_obj_path}" scale="1 1 1"/>
        </geometry>
    </collision>""")
    if inner_box_path.exists():
        base_visuals.append(f"""    <visual>
        <geometry>
            <mesh filename="package://sp/{inner_box_path}" scale="1 1 1"/>
        </geometry>
    </visual>""")

    if base_visuals:
        visuals_str = "\n".join(base_visuals)
        urdf_parts.append(f"""
<link name="{obj_name}_base">
{visuals_str}
</link>

<joint name="{obj_name}_joint_base" type="fixed">
    <parent link="world"/>
    <child link="{obj_name}_base"/>
</joint>
    """)
    else:
        urdf_parts.append(f"""
<link name="{obj_name}_base"/>

<joint name="{obj_name}_joint_base" type="fixed">
    <parent link="world"/>
    <child link="{obj_name}_base"/>
</joint>
    """)

    urdf_parts.append(f"""
<joint name="{obj_name}_joint" type="{joint_type}">
    <parent link="{obj_name}_base"/>
    <child link="{obj_name}_part"/>
    <origin xyz="{origin[0]} {origin[1]} {origin[2]}"/>
    <axis xyz="{vector[0]} {vector[1]} {vector[2]}"/>
    <limit lower="{lower_limit}" upper="{upper_limit}" effort="100.0" velocity="1.0"/>
</joint>
    """)

    urdf_parts.append(f"""
<link name="{obj_name}_part">
    <visual>
        <origin xyz="{-origin[0]} {-origin[1]} {-origin[2]}" rpy="0 0 0" />
        <geometry>
            <mesh filename="package://sp/{part_obj_path}" scale="1 1 1"/>
        </geometry>
    </visual>
</link>
        """)

    urdf_content = f"""<?xml version="1.0"?>
<robot name="generated_robot">
""" + "\n".join(urdf_parts) + """
</robot>
    """

    with open(output_urdf_path, "w") as file:
        file.write(urdf_content)
    print(f"URDF generated: {output_urdf_path}")


def generate_urdf_static(folder_path, output_urdf_path):
    folder = Path(folder_path)
    dae_files = [folder]
    if not dae_files:
        return
    urdf_parts = []
    urdf_parts.append('    <link name="world"/>')
    for i, dae_file in enumerate(dae_files):
        stem = dae_file.stem
        link_name = f"link_{i}"
        position_file = folder / f"{stem}_position.npy"
        if position_file.exists():
            position = np.load(position_file)
            x, y, z = position.tolist()
        else:
            x, y, z = 0.0, 0.0, 0.0
        urdf_parts.append(f"""
    <link name="{link_name}">
        <visual>
            <geometry>
                <mesh filename="package://sp/{dae_file}" scale="1 1 1"/>
            </geometry>
        </visual>
    </link>
        """)
        urdf_parts.append(f"""
    <joint name="{link_name}_joint_{i}" type="fixed">
        <parent link="world"/>
        <child link="{link_name}"/>
        <origin xyz="{x} {y} {z}" rpy="0 0 0"/>
    </joint>
        """)
    urdf_content = f"""<?xml version="1.0"?>
<robot name="static_objects_robot">
{''.join(urdf_parts)}
</robot>
"""
    with open(output_urdf_path, "w") as f:
        f.write(urdf_content)
    print(f"URDF generated: {output_urdf_path}")


parser = argparse.ArgumentParser()
parser.add_argument("--folder_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
args = parser.parse_args()

folder_path = args.folder_path
output_path = args.output_path

if os.path.exists(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path, exist_ok=True)

for subfolder in os.listdir(folder_path):
    obj_name = subfolder.replace(" ", "_")
    subfolder_path = os.path.join(folder_path, subfolder)
    if not os.path.isdir(subfolder_path) and subfolder != "remain_scene.dae":
        continue
    output_file = os.path.join(output_path, f"{obj_name}.urdf")
    if subfolder == "remain_scene.dae":
        output_file = os.path.join(output_path, "remain_scene.urdf")
        generate_urdf_static(subfolder_path, output_file)
    else:
        generate_urdf_interactive(subfolder_path, output_file, obj_name)
