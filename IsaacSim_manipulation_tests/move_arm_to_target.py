import asyncio
import numpy as np

from scipy.spatial.transform import Rotation as R

from pxr import UsdGeom
from omni.isaac.core.utils.prims import get_prim_at_path


async def move_robot_to_target():

    # =========================================================
    # 1. 목표 물체 가져오기
    # =========================================================

    TARGET_PRIM_PATH = "/World/Cube"

    cube_prim = get_prim_at_path(TARGET_PRIM_PATH)

    if not cube_prim.IsValid():
        print(f"❌ 목표 물체를 찾을 수 없습니다: {TARGET_PRIM_PATH}")
        return

    # =========================================================
    # 2. 목표 물체의 월드 좌표 추출
    # =========================================================

    target_geom = UsdGeom.Imageable(cube_prim)

    cube_world_matrix = (
        target_geom.ComputeLocalToWorldTransform(0.0)
    )

    world_target_pos = np.array([
        cube_world_matrix.GetRow(3)[0],
        cube_world_matrix.GetRow(3)[1],
        cube_world_matrix.GetRow(3)[2]
    ])

    print(
        f"📍 실시간 손잡이 위치 자동 감지 완료: "
        f"{world_target_pos}"
    )

    # =========================================================
    # 3. Franka 로봇의 현재 위치 및 회전 추출
    # =========================================================

    robot_world_pos, robot_world_quat = robot.get_world_pose()

    scipy_quat = [
        robot_world_quat[1],
        robot_world_quat[2],
        robot_world_quat[3],
        robot_world_quat[0]
    ]

    robot_rotation = R.from_quat(scipy_quat)

    inverse_robot_rotation = robot_rotation.inv()

    # =========================================================
    # 4. 월드 좌표를 로봇 기준 좌표로 변환
    # =========================================================

    translation_diff = world_target_pos - robot_world_pos

    local_target_pos = inverse_robot_rotation.apply(
        translation_diff
    )

    # =========================================================
    # 5. 역기구학(IK) 계산
    # =========================================================

    action, success = (
        kinematics_solver.compute_inverse_kinematics(
            target_position=local_target_pos
        )
    )

    if success:

        robot.get_articulation_controller().apply_action(
            action
        )

        print(
            "🎉 [IK 성공] 목표 위치로 로봇 팔이 이동합니다!"
        )

    else:

        print(
            "❌ IK 계산 실패: 목표 위치에 도달할 수 없습니다."
        )


# =============================================================
# 6. 스크립트 실행
# =============================================================

asyncio.ensure_future(move_robot_to_target())
