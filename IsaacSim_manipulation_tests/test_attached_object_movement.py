import asyncio
import numpy as np

from omni.isaac.core.utils.prims import get_prim_at_path


# =============================================================
# 설정
# =============================================================

JOINT_PRIM_PATH = "/World/FixedJoint"

STEPS = 60
sleep_time = 0.02


async def move_base_smoothly():

    # =========================================================
    # 1. 로봇의 초기 위치 및 방향 저장
    # =========================================================

    origin_pos, origin_quat = robot.get_world_pose()

    origin_pos = np.array(origin_pos, dtype=float)

    # =========================================================
    # 2. 목표 후진 위치 설정
    # =========================================================

    # 실제 테스트 환경에 맞게 수정해야 하는 예시 좌표
    BACKWARD_DISTANCE = 0.30

    target_back_pos = origin_pos.copy()

    # 로봇의 월드 X축 방향으로 후진하는 예시
    target_back_pos[0] -= BACKWARD_DISTANCE

    # =========================================================
    # STEP 1. 부드럽게 후진
    # =========================================================

    print(
        "📦 [시퀀스 1] "
        "로봇 베이스가 부드럽게 후진하며 문을 엽니다..."
    )

    for i in range(STEPS + 1):

        alpha = i / STEPS

        current_pos = (
            (1 - alpha) * origin_pos
            + alpha * target_back_pos
        )

        robot.set_world_pose(
            position=current_pos,
            orientation=origin_quat
        )

        await asyncio.sleep(sleep_time)

    # 열린 상태에서 안정화 대기
    await asyncio.sleep(1.0)

    # =========================================================
    # STEP 2. 부드럽게 전진하여 원래 위치로 복귀
    # =========================================================

    print(
        "📦 [시퀀스 2] "
        "로봇 베이스가 부드럽게 전진하며 문을 다시 닫습니다..."
    )

    for i in range(STEPS + 1):

        alpha = i / STEPS

        current_pos = (
            (1 - alpha) * target_back_pos
            + alpha * origin_pos
        )

        robot.set_world_pose(
            position=current_pos,
            orientation=origin_quat
        )

        await asyncio.sleep(sleep_time)

    # 완전히 복귀할 때까지 대기
    await asyncio.sleep(1.0)

    # =========================================================
    # STEP 3. Fixed Joint 해제
    # =========================================================

    print(
        "📦 [시퀀스 3] "
        "이동 테스트가 끝났습니다. Fixed Joint를 해제합니다."
    )

    joint_prim = get_prim_at_path(JOINT_PRIM_PATH)

    if joint_prim.IsValid():

        joint_prim.GetAttribute(
            "physics:jointEnabled"
        ).Set(False)

        print(
            "🎉 [최종 종료] "
            "이동 테스트가 끝나고 Fixed Joint를 해제했습니다!"
        )

    else:

        print(
            "❌ 오류: 해제할 조인트를 찾지 못했습니다."
        )


# =============================================================
# 비동기 실행
# =============================================================

asyncio.ensure_future(move_base_smoothly())
