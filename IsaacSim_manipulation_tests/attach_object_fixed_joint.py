import numpy as np

import omni.usd

from pxr import UsdGeom, Gf

from omni.isaac.core.utils.prims import get_prim_at_path


def enable_fixed_joint_with_offset():

    # =========================================================
    # 1. Stage 및 Fixed Joint 경로 설정
    # =========================================================

    JOINT_PRIM_PATH = "/World/FixedJoint"

    joint_prim = get_prim_at_path(JOINT_PRIM_PATH)

    if not joint_prim.IsValid():
        print(
            f"❌ 조인트를 찾을 수 없습니다: {JOINT_PRIM_PATH}"
        )
        return

    try:

        # =====================================================
        # 2. 연결된 Body 0 및 Body 1의 경로 확인
        # =====================================================

        body0_targets = joint_prim.GetRelationship(
            "physics:body0"
        ).GetTargets()

        body1_targets = joint_prim.GetRelationship(
            "physics:body1"
        ).GetTargets()

        if not body0_targets or not body1_targets:
            print("❌ Fixed Joint의 연결 대상이 설정되지 않았습니다.")
            return

        body0_path = body0_targets[0].pathString
        body1_path = body1_targets[0].pathString

        body0_prim = get_prim_at_path(body0_path)
        body1_prim = get_prim_at_path(body1_path)

        if not body0_prim.IsValid() or not body1_prim.IsValid():
            print("❌ 연결 대상 프림을 찾을 수 없습니다.")
            return

        # =====================================================
        # 3. 현재 두 물체의 월드 변환 행렬 추출
        # =====================================================

        geom0 = UsdGeom.Imageable(body0_prim)
        geom1 = UsdGeom.Imageable(body1_prim)

        mat0 = geom0.ComputeLocalToWorldTransform(0.0)
        mat1 = geom1.ComputeLocalToWorldTransform(0.0)

        # =====================================================
        # 4. 상대 위치 및 회전 계산
        # =====================================================

        local_transform = mat1 * mat0.GetInverse()

        local_pos = local_transform.ExtractTranslation()
        local_rot = local_transform.ExtractRotationQuat()

        # =====================================================
        # 5. Fixed Joint 오프셋 설정
        # =====================================================

        joint_prim.GetAttribute(
            "physics:localPos0"
        ).Set(Gf.Vec3f(local_pos))

        joint_prim.GetAttribute(
            "physics:localRot0"
        ).Set(Gf.Quatf(local_rot))

        joint_prim.GetAttribute(
            "physics:localPos1"
        ).Set(Gf.Vec3f(0, 0, 0))

        joint_prim.GetAttribute(
            "physics:localRot1"
        ).Set(Gf.Quatf(1, 0, 0, 0))

        # =====================================================
        # 6. Fixed Joint 활성화
        # =====================================================

        joint_prim.GetAttribute(
            "physics:jointEnabled"
        ).Set(True)

        print("🎉 [완벽 성공] Fixed Joint 오프셋을 설정했습니다!")

        print(f"📍 계산된 로컬 오프셋 위치: {local_pos}")

        print(
            "💡 이제 로봇이 물체와 함께 움직이는지 확인하세요."
        )

    except Exception as e:

        print(f"❌ 오류 발생: {e}")


# =============================================================
# 7. 함수 실행
# =============================================================

enable_fixed_joint_with_offset()
