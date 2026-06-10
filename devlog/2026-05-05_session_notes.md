# REACT3D 코드 분석 및 개선 계획 — 2026-05-05 (~23:00 KST, 최종 업데이트 ~24:00)

## 오늘 한 것

### 1. 프로젝트 상태 확인
- 이전 세션 목록 확인 (4/23~4/29 USDNet 관련 세션 4개)
- Memory 파일 (project_overview, reference_paths 등) 확인
- Isaac Sim 컨테이너 상태 확인: `nvcr.io/nvidia/isaac-sim:5.1.0` 존재, 7번 기동됨
- `urdf_to_usda.py` → `scene.usda` (1.4GB) 생성 성공 확인 (5/1)

### 2. REACT3D 파이프라인 전체 분석

```
scene2part (RAM++ → GroundedSAM → multi-view lifting → mesh)
    ↓ scene_output/ (60+ mesh_XXX 폴더)
part2interactive (OPDMulti inference → joint params + base/part mesh)
    ↓ scene_output/ (각 폴더에 origin/vector/articulation_type/base.obj/part.obj 추가)
filter_duplicates (point-set IoU 기반 중복 제거)
    ↓ scene_output_filtered/ (12개 mesh 남음)
texture (xatlas UV → vertex color → texture image → GLB → Blender DAE)
    ↓ part.glb, part.dae, part.png 생성
simulation_ready (URDF 생성 + ROS launch/rviz)
    ↓ urdf/ 폴더에 mesh별 .urdf
```

### 3. 핵심 코드 분석

#### filter_duplicates.py
- `part.ply` 점 좌표를 `set(tuple(point))`로 변환, 정확히 같은 좌표만 교집합
- 1단계: pairwise IoU > 0.8 → 작은 쪽 제거
- 2단계: 작은 mesh 조합이 큰 mesh를 IoU ≥ 0.8로 cover → 큰 쪽 제거
- ~~한계: float 좌표 달라도 IoU=0~~ → 실제로는 scene mesh에서 잘라온 거라 좌표 동일, 이 부분은 문제 아님

#### corners_to_mesh() in utilities.py → **겹침 문제의 핵심 원인**
- OBB 8개 corner에서 6면 얇은 shell 생성 (thickness = 3% of shortest edge)
- **rotation(문)**: base.obj만 생성 (OBB shell 전체), inner_box 없음
- **translation(서랍)**: base.obj + inner_box.obj (앞면/윗면 제거한 5면 box)
- base.obj는 uniform vertex color만 있음
- **문제**: base.obj = part를 감싸는 OBB shell → part와 완전히 겹침, 이웃 part의 OBB와도 겹침

#### generate_urdf_scene_ros.py → URDF 구조
```
world (fixed)
  └─ {mesh}_base (fixed joint) ← base.obj (OBB shell)
       └─ {mesh}_part (revolute/prismatic) ← part.dae [+ inner_box.obj]
```

#### convert_texture.py → 텍스처 파이프라인
- 입력: `part.obj` (open3d vertex color mesh, 원본은 scene mesh에서 잘라온 vertex color)
- xatlas UV parametrize → vertex color → texture image (현재 128x128) → cv2.inpaint → GLB
- Blender에서 GLB → DAE + texture PNG

### 4. example_input_scene 데이터 현황
- 12개 mesh (mesh_000~mesh_970): 11개 rotation(문), 1개 translation(서랍: mesh_500)
- inner_box.obj는 mesh_500만 보유
- range_max 전부 ~1.64~1.72 rad (≈94~98°) — 의심스럽게 비슷
- axis 대부분 Z축 (±0.01 이내) — 수직 hinge 문들

---

## 유저가 보고한 3가지 문제 (URDF → viser 웹 뷰어, 배경 없이 part만)

1. **물체 겹침**: part들이 겹쳐 보임
2. **텍스처 이상**: 닫힌 상태에서 이상하게 보이고, 열어도 텍스처 자체가 깨짐
3. **문 뒤 막힘**: 문 열어도 뒤가 막혀 있음 → 후순위

---

## 근본 원인 분석 (수정됨)

### 물체 겹침 — 원인은 base.obj (OBB shell)

**part mesh 자체는 겹치지 않는다.** scene mesh에서 잘라온 거라 위치가 고정돼 있음.

**겹치는 건 base.obj (OBB shell).**
- base.obj = part를 감싸는 OBB 전체를 얇은 shell로 만든 것
- 이 OBB shell이 (1) 자기 part.dae와 완전히 겹치고, (2) 이웃 part의 OBB와도 겹침

실제 AABB overlap 측정 결과:
```
mesh_033 vs mesh_034: overlap vol=0.0020  (나란한 문)
mesh_1099 vs mesh_408: overlap vol=0.0023  (나란한 문)
mesh_500 vs mesh_709: overlap vol=0.0040  (서랍과 옆 문)
mesh_500 vs mesh_970: overlap vol=0.0024  (서랍과 옆 문)
```

→ **filter_duplicates 문제가 아니라, corners_to_mesh()의 base.obj 생성 방식 자체가 원인**

### 이건 논문의 구현 한계

| | 논문의 의도 | 실제 구현 |
|---|---|---|
| base body | 문틀, 캐비닛 본체 (parent body) | OBB shell (= part를 감싸는 박스) |
| hidden geometry | 문 뒤 공간, 서랍 내부 | inner_box (서랍만, 문은 없음) |
| background carving | 문이 빠진 자리 비우기 | 구현 없음 |

논문 아이디어(base body + hidden geometry + carving)는 맞지만, 구현이 OBB shell 하나로 퉁치고 있음.
→ **이 후단을 개선하는 것이 우리 contribution 포인트**

### 텍스처 이상 — 원인

**A. base.obj와 part.dae가 겹쳐서 base의 단색 면이 비침**
- base.obj = uniform vertex color (단색 OBB shell)
- part.dae = scan에서 온 실제 텍스처
- 초기 닫힌 상태에서 둘이 같은 공간에 있으니 단색이 비쳐서 이상하게 보임
- → base.obj 겹침을 해결하면 이 문제도 같이 해결됨

**B. part.dae 텍스처 자체도 저품질**
- 원본: scene mesh의 vertex color (저해상도)
- texture 생성: xatlas UV + **128x128** → 매우 저해상도
- 2D RGB frame에서 직접 projection 하면 훨씬 나을 수 있음 (`top_k_list.pkl`에 best-view 정보 있음)

---

## 수정된 개선 계획

### Priority 1: base.obj 겹침 해결 (물체 겹침 + 텍스처 A 동시 해결)

**핵심**: `corners_to_mesh()`가 만드는 base.obj를 바꿔야 함

**Option A: base.obj를 아예 제거/빈 link로**
- 가장 간단. URDF에서 base link에 visual 없이 빈 link만 두기
- joint origin/axis만 유지하면 articulation은 그대로 작동
- 단점: parent body 시각적 표현 없음

**Option B: base.obj에서 part 영역을 빼기**
- OBB shell에서 part mesh가 차지하는 면을 boolean subtraction
- 문이면: OBB에서 앞면(part가 있는 면)을 제거 → 문틀처럼 나머지 frame만 남김
- 구현 복잡도 중간

**Option C: remain_scene에서 parent body 추출**
- background mesh에서 part 주변의 실제 문틀/캐비닛 body를 잘라서 base로 쓰기
- 가장 정확하지만 구현 복잡

**추천**: 먼저 Option A로 즉시 겹침 해결 → Option B로 시각적 개선

### Priority 2: 텍스처 개선

**2A. 즉시: texture 해상도 128→512**
- `generate_texture.sh`에서 `--texture-size 128` → `512`로 변경
- 1분 작업, 즉시 효과

**2B. 중기: 2D RGB → part texture projection**
- `top_k_list.pkl`에 best-view frame 정보 있음
- best-view RGB frame + 2D mask + camera intrinsic/extrinsic으로 part mesh에 직접 texture project
- vertex color 대신 고해상도 texture map 생성

### Priority 3: 문 뒤 막힘 (후순위)
- base.obj를 hinge side frame만 남기기 (Option B의 확장)
- remain_scene에서 door opening carving
- motion-aware sweep volume carving

---

## 다음 세션에서 할 것

1. **base.obj 제거 또는 수정** — Option A (빈 link) 먼저 적용해서 겹침 해결 확인
2. **텍스처 해상도 올리기** — 128→512
3. **viser로 결과 비교** — 수정 전후 확인
4. **(여유 있으면) 2D RGB projection** 텍스처 프로토타입

---

## 파일 참조
- `REACT3D/part2interactive/filter_duplicates.py` — 중복 제거 (겹침 원인 아님, 확인됨)
- `REACT3D/part2interactive/utilities.py` — **`corners_to_mesh()` = 겹침 원인** (base/inner_box 생성)
- `REACT3D/part2interactive/inference.py` — 메인 추론, line 901에서 corners_to_mesh 호출
- `REACT3D/simulation_ready/generate_urdf_scene_ros.py` — URDF 생성
- `REACT3D/texture/convert_texture.py` — vertex color → xatlas UV → texture
- `REACT3D/texture/generate_texture.sh` — **`--texture-size 128`** ← 여기를 512로 바꾸면 즉시 개선
- `REACT3D/texture/blender_glb_to_dae.py` — GLB → DAE 변환
- `REACT3D/scene2part/s2p_6.py` — mask_mesh_refined.ply 생성
