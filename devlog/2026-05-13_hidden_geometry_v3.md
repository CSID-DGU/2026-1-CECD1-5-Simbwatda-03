# REACT3D — Hidden Geometry 깊이 문제 해결 + cuboid_world 복원 (2026-05-13)

이전 세션(2026-05-05-1358)에서 base.obj 겹침 해결 + scene-aware base body extraction 완료.
이번 세션은 hidden geometry(inner_box)가 깊이 없이 표면에 붙어있는 문제를 해결.

---

## 0. 시작 시점 상태

이전 세션 결과:
- **base.obj 겹침** ✅ 해결 — `base_scene.obj` (remain_scene crop)로 교체
- **URDF 구조** — base link에 `base_scene.obj` + `inner_box.obj` visual, part link에 `part.dae`만
- **inner_box 위치** ✅ base link로 이동 (고정, 문/서랍과 같이 안 움직임)

미해결:
- inner_box가 깊이 없이 표면에만 붙어 있음 → 문 열어도 캐비닛 내부 공간이 안 보임
- mesh_500 inner_box 이웃 겹침
- 033/034 segmentation 겹침

---

## 1. 문제 진단: inner_box에 깊이가 없는 원인

### 증상
viser에서 문을 열면 hidden geometry가 문 표면에 얇은 박스로만 존재. 캐비닛 내부 깊이가 없음.

### 원인 분석

`corners_to_mesh()` (utilities.py)는 두 가지 corner set을 받음:
- **`obb_world`**: part 표면의 tight OBB → 깊이 없음 (0.04m 정도)
- **`cuboid_world`**: `complete_point_cloud_true_mesh()`에서 생성한 depth-completed cuboid → 깊이 있음 (0.75m 정도)

`complete_point_cloud_true_mesh()` (visualization.py:1470-1527)가 핵심:
- scene mesh에 ray를 쏴서 part 뒤쪽 깊이를 계산
- `proj_max` = part 표면 너머로 scene mesh까지의 깊이
- cuboid_world corners = `[x_min/max, y_min/max, proj_min/proj_max]` → 실제 캐비닛 깊이 포함

**문제**: `inference.py`에서 `cuboid_world`를 디스크에 저장하지 않고 있었음. `obb_world.npy`만 저장.

`generate_hidden_geometry.py` (이전 세션에서 작성)는 `obb_world.npy`를 읽어 inner_box를 생성 → 깊이 없는 얇은 box만 생성됨.

---

## 2. 수정: cuboid_world.npy 저장 추가

### 수정 파일
`REACT3D/part2interactive/inference.py` (line 885-886)

### 변경 내용
```python
np.save(f"{curr_output_dir}/obb_world.npy", obb_world)  # mask coverage ratio
np.save(f"{curr_output_dir}/cuboid_world.npy", np.array(cuboid_world["corners"]))  # depth-completed cuboid
```

기존에 `obb_world`만 저장하던 곳에 `cuboid_world["corners"]`도 함께 저장.

---

## 3. 파이프라인 재실행

cuboid_world.npy가 새로 저장되려면 part2interactive부터 다시 돌려야 함.

### 실행 순서
```bash
# 1. part2interactive inference (docker usdnet 내부)
cd /workspace/REACT3D/part2interactive
python inference.py --scene_dir /workspace/REACT3D/data/example_input_scene

# 2. filter_duplicates
python filter_duplicates.py \
  --scene_output_dir /workspace/REACT3D/data/example_input_scene/scene_output \
  --scene_mesh_path /workspace/REACT3D/data/example_input_scene/scene_mesh.ply \
  --output_dir /workspace/REACT3D/data/example_input_scene/scene_output_filtered

# 3. generate_remain_scene
python generate_remain_scene.py \
  --scene_output_filtered_dir /workspace/REACT3D/data/example_input_scene/scene_output_filtered \
  --scene_mesh_path /workspace/REACT3D/data/example_input_scene/scene_mesh.ply

# 4. texture conversion (호스트, react3d conda env)
conda activate react3d
python /home/gpuadmin/capstone/USDNet/REACT3D/texture/convert_texture.py \
  --scene_output_filtered_dir /home/gpuadmin/capstone/USDNet/REACT3D/data/example_input_scene/scene_output_filtered

# 5. blender GLB → DAE
# 4개 mesh (mesh_951, mesh_408, mesh_1099, mesh_023)가 texture 변환 실패 → 개별 재실행으로 해결
```

### 텍스처 변환 실패 4개 mesh 수동 복구
`convert_texture.py`가 일부 mesh에서 silent fail. 개별적으로 `process_obj_file()` 호출 + blender DAE 변환으로 해결.

---

## 4. inner_box_v3.obj 생성

### 방식
기존 `generate_hidden_geometry.py` (obb_world 사용, v2)를 버리고, 원본 `corners_to_mesh()` 로직을 그대로 사용하되 `cuboid_world.npy`를 입력으로 사용.

### 결과
12개 mesh 전부 `inner_box_v3.obj` 생성 완료. 각각 164 lines (OBJ format).

cuboid_world 기반이므로 깊이가 있는 inner_box가 생성됨.

---

## 5. URDF 생성 + viser 디렉토리 구성

### URDF 생성
```bash
python generate_urdf_scene_ros.py \
  --folder_path /home/gpuadmin/capstone/USDNet/REACT3D/data/example_input_scene/scene_output_filtered \
  --output_path /home/gpuadmin/capstone/USDNet/REACT3D/data/example_input_scene/urdf_20260513_1530
```

12개 URDF 전부 생성 성공.

### URDF 구조 (현재)
```
world (fixed)
  └─ {mesh}_base (fixed joint)
       ├─ base_scene.obj (visual + collision) — remain_scene crop
       └─ inner_box_v3.obj (visual) — cuboid_world 기반 hidden geometry
           └─ {mesh}_part (revolute/prismatic)
                └─ part.dae (visual) — textured movable geometry
```

### URDF 경로 fallback 체인
```python
base_obj_path: base_scene.obj → base.obj
inner_box_path: inner_box_v3.obj → inner_box.obj
```

### viser 디렉토리
`REACT3D/data/hidden_v3_20260513_1530/`
- `hidden_v3_20260513_1530_urdfs/` → symlink to urdf_20260513_1530
- `hidden_v3_20260513_1530_meshes/` → symlink to scene_output_filtered

### 확인 명령
```bash
conda activate react3d && python /home/gpuadmin/capstone/USDNet/REACT3D/visualization/vis_interactive_urdf.py \
  --scene_dir /home/gpuadmin/capstone/USDNet/REACT3D/data/hidden_v3_20260513_1530 --port 8080
```

---

## 6. 비교용 viser 디렉토리 전체 목록

| 디렉토리 | 설명 | 날짜 |
|---|---|---|
| `data/example_input_scene` | 원본 (base.obj OBB shell, inner_box in part link) | 기존 |
| `data/nobase_20260506` | base.obj 제거, 빈 link | 2026-05-06 |
| `data/scene_base_20260505_1320` | base_scene.obj 사용, inner_box 없음 | 2026-05-05 |
| `data/hidden_geom_20260513_1430` | generate_hidden_geometry.py v2 (obb_world, 깊이 없음) | 2026-05-13 |
| `data/hidden_orig_20260513_1450` | 원본 corners_to_mesh inner_box (obb_world, 깊이 없음) | 2026-05-13 |
| `data/hidden_v3_20260513_1530` | **최종: cuboid_world 기반 inner_box_v3 + base_scene.obj** | 2026-05-13 |

---

## 7. 현재 상태 요약

| 항목 | 상태 |
|---|---|
| 문(rotation) 표면 겹침 | ✅ 해결 (base_scene.obj 사용) |
| inner_box 위치 (base vs part link) | ✅ base link에 고정 |
| inner_box 깊이 | ✅ cuboid_world 기반으로 깊이 복원 |
| cuboid_world.npy 저장 | ✅ inference.py에 추가 |
| 서랍 mesh_500 이웃 겹침 | ❌ 미해결 (inner_box가 이웃과 겹칠 수 있음) |
| 033/034 segmentation 겹침 | ❌ 상위 파이프라인 문제 |
| mesh_002 max-open 충돌 | ❌ 미분석 |
| 텍스처 해상도 | ❌ 128x128 유지 (512 미적용) |
| viser 시각 확인 | ⏳ 미확인 — 사용자가 확인 필요 |

---

## 8. 핵심 발견: cuboid_world vs obb_world

이번 세션의 핵심 디버깅 결과.

| | obb_world | cuboid_world |
|---|---|---|
| 생성 함수 | Open3D OBB | `complete_point_cloud_true_mesh()` |
| 깊이 | part 표면 두께만 (~0.04m) | scene mesh 뒤까지 ray-cast (~0.75m) |
| 저장 여부 (수정 전) | ✅ `obb_world.npy` | ❌ 저장 안 됨 |
| 저장 여부 (수정 후) | ✅ `obb_world.npy` | ✅ `cuboid_world.npy` |
| 용도 | filter_duplicates IoU 등 | inner_box / hidden geometry 생성 |

`corners_to_mesh()`는 원래 inference 시점에서 `cuboid_world`를 직접 받아 inner_box를 만들었음.
하지만 post-processing 단계에서는 `cuboid_world`가 디스크에 없어서 `obb_world`로 대체할 수밖에 없었음.
→ `cuboid_world.npy` 저장을 추가함으로써 post-processing에서도 동일한 깊이를 확보.

---

## 9. 다음 세션에서 할 것

1. **viser 시각 확인** — hidden_v3_20260513_1530에서 문 열었을 때 캐비닛 내부 벽 보이는지
2. **mesh_500 (서랍) inner_box 이웃 겹침** — cuboid_world 기반으로도 여전히 겹치는지 확인
3. **텍스처 해상도** — 128→512 (`generate_texture.sh` 수정)
4. **033/034 segmentation 겹침** — scene2part mask boolean subtraction
5. **devlog 통합** — 05-05, 05-06, 05-13 세션을 논문 novelty 관점에서 정리

---

## 10. 파일 변경 / 신규 목록

### 수정
- `REACT3D/part2interactive/inference.py` (line 886) — `cuboid_world.npy` 저장 추가
- `REACT3D/simulation_ready/generate_urdf_scene_ros.py` — inner_box_v3.obj fallback 추가, base link 구조 정리

### 신규 (이전 세션)
- `REACT3D/simulation_ready/extract_base_from_scene.py` — remain_scene crop → base_scene.obj
- `REACT3D/simulation_ready/generate_hidden_geometry.py` — inner_box post-processing (v2, obb_world 기반 — superseded)
- `REACT3D/simulation_ready/check_sweep_collision.py` — sweep volume 충돌 검증

### 데이터
- `scene_output_filtered/mesh_*/cuboid_world.npy` — 12개 전부 생성
- `scene_output_filtered/mesh_*/inner_box_v3.obj` — 12개 전부 생성 (cuboid_world 기반)
- `data/example_input_scene/urdf_20260513_1530/` — 최종 URDF 12개
- `data/hidden_v3_20260513_1530/` — viser용 symlink 디렉토리
