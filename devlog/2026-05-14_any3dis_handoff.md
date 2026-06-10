# Any3DIS 통합 — 진행 상태 핸드오프 (2026-05-14)

이 세션에서 Any3DIS를 REACT3D scene2part 단계 대체로 통합하는 작업 시작. 다음 세션에서 이어서 진행.

---

## 0. 흐름 — 왜 이걸 하게 됐나

### 0.1 시작 상태 (이전 세션까지의 진행)
이전 세션들에서 REACT3D 시각화 문제 3가지를 다뤘음:
1. 물체 겹침 (base.obj OBB shell 때문) → `base_scene.obj` (remain_scene crop) 또는 `base_pca.obj` (PCA surface normal 기반)으로 해결
2. 텍스처 저해상도 (128x128) → 미해결
3. 문 뒤 막힘 (hidden geometry) → `inner_box_v3.obj` (cuboid_world 기반) 해결, 결과 `basepca_v2_20260513`

이전 세션 마지막 결과 (`basepca_v2_20260513`): 시각적으로 거의 자연스러움. 하지만 일부 part 끼리 여전히 겹치는 케이스 잔존:
- **1099 vs 408**: part 자체가 2.3cm 겹쳐 있음 (X 방향) → upstream segmentation 문제로 base 단에서 해결 불가
- **033/034**: 같은 평면을 두 part로 잘못 segment

### 0.2 사용자 진단: "segmentation 단계 자체를 고쳐야 한다"
사용자가 지적 — base/hidden geometry 후처리로는 한계. 실제로 part가 잘못 잘려 있으면 그 다음 단계가 다 영향받음. 그래서 scene2part 자체를 개선해야 함.

### 0.3 REACT3D scene2part 약점 분석
파이프라인을 단계별로 까봤더니 약점 세 가지:
1. **GroundedSAM 2D mask edge 부정확**: 인접 part 사이 boundary가 깔끔하지 않음
2. **Multi-view lifting 평균화**: 여러 frame mask를 평균내면 가장자리 노이즈 누적
3. **DBSCAN으로 인접 part 분리 실패**: 같은 평면 위 두 part가 한 cluster로 묶임

### 0.4 최신 연구 탐색 (2024~2026 탑 학회)
사용자 요청으로 web search:
- **Any3DIS (CVPR 2025)** — SAM2 video tracking으로 mask를 frame간 일관되게 추적, dynamic programming으로 superpoint 할당 최적화. ScanNet++ 17.9 → 22.2 AP. **REACT3D의 약점 1+2+3을 한 번에 공격**.
- GeoSAM2, PartSAM, P3-SAM 등도 후보지만 Any3DIS가 indoor scene과 직접 매칭
- SAM 3D (Meta 2025), Articulate3D (ICCV 2025) 등 다른 방향은 우리 데이터에 안 맞음
- **결론: Any3DIS 채택**

### 0.5 코드 발견: PhucNDA의 unofficial repo
- https://github.com/PhucNDA/Any3DIS_unofficial
- **PhucNDA = 논문 1저자 (Nguyen Da Phuc)** 본인 — "unofficial"은 회사(VinAI) 정식 release가 아니란 의미
- Open3DIS 위에 올려놓은 코드. SAM2 + 12D feature filter + DP 3D proposal optimization 다 포함

### 0.6 통합 계획 수립
플랜:
1. Any3DIS 환경 구축 (새 conda env)
2. REACT3D 데이터를 ScanNet++ 형식으로 변환
3. Any3DIS가 필요로 하는 superpoint + dc_features 직접 생성 (REACT3D에는 없는 데이터)
4. Any3DIS 실행 → 3D instance proposals
5. 결과를 REACT3D scene_output 형식으로 역변환
6. REACT3D part2interactive 이후 단계는 그대로 실행
7. viser에서 결과 비교 (`basepca_v2_20260513` vs 새 결과)

### 0.7 핵심 결정: 의존성 minimize
Any3DIS는 ScanNet++에서 ISBNet/Mask3D pretrained feature를 dc_features로 사용. 우리에겐 없음.
- **Mask3D 설치 시도 안 함** — MinkowskiEngine 빌드 까다롭고 (~60% 성공률), ScanNet200 학습이 우리 cabinet door fine-grained에 강하지 않을 거고, 설치 시간만 2-4시간
- **대안: 12D geometric features** (normal+color+position+curvature) — 30분 만에 만들고 cabinet door 같은 명확한 geometric 객체엔 효과적
- 사용자 동의 후 진행

---

## 1. 완료된 단계 (Phase 1~9 일부)

### Phase 1: conda env `any3dis` ✅
- python 3.10, torch 2.3.1+cu118, torch_scatter, SAM2 (local), pointnet2 (custom build), CLIP, trimesh, open3d, hydra-core

**막힌 곳 / 해결**:
- `clip` git install 실패 → setuptools<81로 다운그레이드 후 `--no-build-isolation`으로 해결 (pkg_resources 모듈 issue)
- pointnet2 빌드 시 CUDA 버전 불일치 (시스템 13.2 vs torch 11.8) → conda로 cuda-toolkit 설치 시도했더니 13.2가 깔림 → `conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit` (labeled channel 사용)로 11.8 정확히 설치
- pointnet2 빌드 시 `gcc: fatal error: cannot execute 'cc1plus'` → gcc 12 / g++ 11 버전 불일치 → `CC=gcc-11 CXX=g++-11` 환경변수 강제로 해결

### Phase 2: SAM2 checkpoint ✅
- `/home/gpuadmin/capstone/USDNet/REACT3D/data/checkpoints/sam2_hiera_large.pt` (857MB)

### Phase 3: 데이터 변환 (REACT3D → Any3DIS) ✅
스크립트: `REACT3D/any3dis_integration/prepare_any3dis_data.py`

**중요**: `aligned_pose`를 쓴다 (scene mesh와 좌표계 일치). raw `pose`를 쓰면 mapping이 잘못된 좌표계로 됨.

결과 위치:
```
REACT3D/data/any3dis_data/
  Scannetpp_2D_5interval/val/react3d_scene/
    color/frame_XXXXXX.jpg     # 1354 frames, resized to 960x720
    depth/frame_XXXXXX.png     # symlinked, 256x192 uint16
    pose/frame_XXXXXX.txt      # aligned_pose 4x4
    intrinsic/frame_XXXXXX.txt # scaled to 960x720
    intrinsic.txt              # global RGB intrinsic
  Scannetpp_3D/val/
    original_ply_files/react3d_scene.ply  # symlink to mesh_aligned_0.05.ply
  react3d_val.txt              # split file
```

### Phase 4: Superpoint 생성 ✅
스크립트: `REACT3D/any3dis_integration/generate_superpoints.py`

Felzenszwalb-Huttenlocher graph segmentation (normal+color edge weight).
- 출력: `REACT3D/data/any3dis_data/Scannetpp_3D/val/superpoints/react3d_scene.pth`
- **3864 superpoints**, median 134 vertices, max 6502

### Phase 5: dc_features 생성 ✅
스크립트: `REACT3D/any3dis_integration/generate_dc_features.py`

12D geometric features = normal(3) + color(3) + relative_position(3) + curvature(3). Mask3D pretrained 대신 사용 (Mask3D 셋업 어려움 + cabinet door엔 geometric으로 충분).
- 출력: `REACT3D/data/any3dis_data/Scannetpp_3D/val/dc_feat_scannetpp/react3d_scene.pth`

### Phase 6: Any3DIS config ✅
`/tmp/Any3DIS_unofficial/configs/react3d.yaml`
- img_dim: [960, 720]
- save_dir: `/home/gpuadmin/capstone/USDNet/REACT3D/data/any3dis_exp`

### Phase 7: Any3DIS 실행 ✅
```bash
cd /tmp/Any3DIS_unofficial && \
PYTHONPATH=./segmenter2d/segment-anything-2:./:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 \
conda run -n any3dis python tools/mask_generator.py --config configs/react3d.yaml
```
출력: `REACT3D/data/any3dis_exp/react3d_run1/mask2d_lifted/react3d_scene.pth` (20MB)

**첫 번째 실행**: 20개 proposal만 나옴, 모두 X=0~1m 좁은 영역에 몰림 → **이상함**.
**디버그**: 3D proposal을 한 frame에 projection 시도 — `aligned_pose`로는 image 밖(v=982-1199)으로 나가고, raw `pose`로는 안에 들어옴.
**원인 발견**: 우리가 `pose/{frame}.txt`에 raw `pose`를 썼는데, scene mesh는 `mesh_aligned_0.05.ply` (aligned coords). 좌표계 불일치로 Any3DIS의 mapping이 일관되긴 하지만 잘못된 좌표계에서 계산됨 → frame index가 잘못된 view를 가리킴.

**해결**: `prepare_any3dis_data.py`에서 `aligned_pose`를 쓰게 수정 → Any3DIS 재실행.

**두 번째 실행**: **224개 3D instance proposals** 생성. Scene 전체에 골고루 분포 (X=0.03~5.21m).
- 각각 RLE-encoded mask on 762893 vertices + id_bank (best views) + obj_bank
- 처리 시간: ~3분

### Phase 8: Reverse converter (Any3DIS → REACT3D scene_output) ✅
스크립트: `REACT3D/any3dis_integration/any3dis_to_react3d.py`

**1차 시도**: vertex projection만 했더니 → 모든 mesh에서 "no usable frames" (mask sum < 50 pixel). Projection이 sparse해서 시각화 가능한 픽셀이 너무 적음.

**2차 시도**: dilation 추가 + mesh triangle rasterization → 154개 mesh 변환 성공.

**3차 시도 (이후 part2interactive에서 실패 발견)**:
- 첫 part2interactive 돌렸을 때 `RuntimeError: RGB image size (720 960) and depth image size (192 256) mismatch` 에러
- REACT3D scene2part의 `s2p_5.py` 코드 분석 → 모든 데이터(RGB, depth, mask)를 **256x192로 resize**해서 저장하고 있었음
- intrinsic도 scale=3.75로 나눠서 256x192용으로 사용
- → reverse converter에서 RGB/mask도 256x192로 resize해서 mesh subfolder에 저장하도록 수정

**4차 시도 (2D mask 정밀화)**:
- part2interactive 돌리니 "Mask coverage ratio is too low: 0.0" 다발
- 시각화로 확인 — 우리 mask가 vertex projection만 해서 너무 sparse, OPDMulti는 dense한 mask 예측
- mesh triangle 자체를 `cv2.fillConvexPoly`로 채워서 rasterize → solid mask
- 다시 실행 → 일부 case 성공

**최종 변환 결과**:
- 224 proposals → **154개 mesh** 변환 (70개 skip — min_vertices 500 미만, 또는 어떤 view에서도 안 보임)
- 출력 구조: `<output_dir>/perception/vis_groups_final_mesh/real/mesh_XXX/`
- 각 mesh subfolder: `mask_mesh_refined.ply` + `top_k_list.pkl` + 5개의 `rgb_<frame>.png` (256x192)
- top-k views는 id_bank 무시하고 직접 모든 frame projection 후 coverage 큰 순으로 K=5개 선택 (id_bank에 잘못된 frame index가 있을 수 있어서)
- 위치: `REACT3D/data/scene_any3dis/perception/vis_groups_final_mesh/real/`

### Phase 9 일부: REACT3D part2interactive + filter_duplicates + remain_scene ✅
```bash
# Symlink 미리 작업됨
cd REACT3D/data/scene_any3dis/
ln -sf .../mesh_aligned_0.05.ply mesh_aligned_0.05.ply
ln -sf .../pose_intrinsic_imu_mvp.json pose_intrinsic_imu_mvp.json
ln -sf .../pose_intrinsic_imu.json pose_intrinsic_imu.json
```

```bash
# part2interactive inference (완료)
cd REACT3D/part2interactive && conda run -n react3d python inference.py \
  --scene-folder /home/gpuadmin/capstone/USDNet/REACT3D/data/scene_any3dis \
  --output /home/gpuadmin/capstone/USDNet/REACT3D/data/scene_any3dis/scene_output \
  --model opdm_rgb.pth
# 결과: 154 mesh 중 11개에서 articulation 검출 (9 rotation + 2 translation)
# OPDM이 cabinet door specific이라 천장/바닥/벽 같은 region 자동 reject (정상)

# filter_duplicates (완료)
cd REACT3D/part2interactive && conda run -n react3d python filter_duplicates.py \
  --input_dir /home/gpuadmin/capstone/USDNet/REACT3D/data/scene_any3dis/scene_output \
  --output_dir /home/gpuadmin/capstone/USDNet/REACT3D/data/scene_any3dis/scene_output_filtered \
  --iou_threshold 0.5
# 결과: 11 → 8 (3개 중복 제거)

# generate_remain_scene (완료)
cd REACT3D/part2interactive && conda run -n react3d python generate_remain_scene.py \
  --scene .../mesh_aligned_0.05.ply \
  --part_dir REACT3D/data/scene_any3dis/scene_output_filtered
# 결과: remain_scene.ply 생성됨
```

남은 8개 mesh:
```
mesh_009, mesh_012, mesh_018, mesh_034, mesh_039, mesh_043, mesh_055, mesh_164
```

**OPDMulti가 자동으로 reject한 것들의 패턴** (정상 동작):
- `mesh_046`: 바닥 region → "Mask coverage ratio is too low: 0.0" (articulated 아님)
- `mesh_002`: 천장 region → 다행히 우연히 통과 (mask가 컸음)
- 대부분 벽/바닥/천장 segmentation → OPDM이 "moving part 없음" 또는 IoU 0
- → Any3DIS class-agnostic이 모든 surface를 segment하는데, OPDM이 cabinet-specific이라 articulation 없는 region을 자동 reject. **이 동작 자체는 정상**, 결과 11개도 합리적

---

## 2. 다음 세션에서 할 일

### 우선 확인: texture conversion 상태
백그라운드 task `bfuvlqkbk`로 `convert_texture.py`가 돌아갔었다. 완료 여부 확인:
```bash
ls /home/gpuadmin/capstone/USDNet/REACT3D/data/scene_any3dis/scene_output_filtered/mesh_*/part.glb 2>/dev/null
# 8개 part.glb가 다 있으면 완료
```
없으면 재실행:
```bash
cd /home/gpuadmin/capstone/USDNet/REACT3D/texture && conda run -n react3d python convert_texture.py \
  --folder /home/gpuadmin/capstone/USDNet/REACT3D/data/scene_any3dis/scene_output_filtered \
  --texture-size 128 --remain-texture-size 512
```

### Step 1: Blender GLB → DAE 변환
```bash
cd /home/gpuadmin/capstone/USDNet/REACT3D/texture && \
blender --background --python blender_glb_to_dae.py -- \
  --folder /home/gpuadmin/capstone/USDNet/REACT3D/data/scene_any3dis/scene_output_filtered
```

### Step 2: base_pca.obj 생성 (이전 세션 수정사항 적용)
이전 세션에서 만든 PCA 기반 base body 생성기를 사용. OBB shell 대신 표면 정렬된 base.
```bash
conda run -n react3d python /home/gpuadmin/capstone/USDNet/REACT3D/simulation_ready/generate_base_from_surface.py \
  --scene_dir /home/gpuadmin/capstone/USDNet/REACT3D/data/scene_any3dis/scene_output_filtered
```

### Step 3: URDF 생성
```bash
mkdir -p /home/gpuadmin/capstone/USDNet/REACT3D/data/scene_any3dis/urdf_any3dis_$(date +%Y%m%d)
conda run -n react3d python /home/gpuadmin/capstone/USDNet/REACT3D/simulation_ready/generate_urdf_basepca.py \
  --folder_path /home/gpuadmin/capstone/USDNet/REACT3D/data/scene_any3dis/scene_output_filtered \
  --output_path /home/gpuadmin/capstone/USDNet/REACT3D/data/scene_any3dis/urdf_any3dis_$(date +%Y%m%d)
```

### Step 4: Viser symlink 디렉토리 만들기
```bash
DATE=$(date +%Y%m%d_%H%M)
TARGET=/home/gpuadmin/capstone/USDNet/REACT3D/data/any3dis_${DATE}
mkdir -p ${TARGET}/any3dis_${DATE}_urdfs ${TARGET}/any3dis_${DATE}_meshes
ln -sf /home/gpuadmin/capstone/USDNet/REACT3D/data/scene_any3dis/urdf_any3dis_$(date +%Y%m%d)/* \
  ${TARGET}/any3dis_${DATE}_urdfs/
for d in /home/gpuadmin/capstone/USDNet/REACT3D/data/scene_any3dis/scene_output_filtered/mesh_*; do
  ln -sf "$d" ${TARGET}/any3dis_${DATE}_meshes/
done
ln -sf /home/gpuadmin/capstone/USDNet/REACT3D/data/scene_any3dis/scene_output_filtered/remain_scene.ply \
  ${TARGET}/any3dis_${DATE}_meshes/
```

### Step 5: Viser 확인
```bash
conda activate react3d && python /home/gpuadmin/capstone/USDNet/REACT3D/visualization/vis_interactive_urdf.py \
  --scene_dir /home/gpuadmin/capstone/USDNet/REACT3D/data/any3dis_${DATE} --port 8080
```

이전 세션 결과 (`basepca_v2_20260513`)와 비교.

---

## 3. 현재 상태 요약

| 항목 | 상태 |
|---|---|
| any3dis conda env | ✅ |
| SAM2 checkpoint | ✅ |
| 데이터 변환 (aligned_pose 사용) | ✅ |
| Superpoint 3864개 | ✅ |
| dc_features 12D | ✅ |
| Any3DIS 실행 → 224 proposals | ✅ |
| Reverse converter (triangle rasterization) → 154 mesh | ✅ |
| part2interactive → 11 articulated parts | ✅ |
| filter_duplicates → 8 parts | ✅ |
| generate_remain_scene | ✅ |
| **texture (convert_texture.py)** | ⏳ 실행 중일 수 있음, 확인 필요 |
| Blender DAE 변환 | ❌ 다음 세션 |
| base_pca.obj 생성 | ❌ 다음 세션 |
| URDF 생성 | ❌ 다음 세션 |
| Viser 시각 확인 | ❌ 다음 세션 |

---

## 4. 핵심 발견 / 주의사항

### 4.1 aligned_pose 필수
- `mesh_aligned_0.05.ply`는 scene aligned coords
- raw `pose`를 Any3DIS에 주면 mapping이 다른 좌표계로 됨 → garbage 결과
- **반드시 `aligned_pose`를 pose 파일에 써야 함**

### 4.2 RGB/depth/mask 해상도
- REACT3D inference는 **256x192**를 가정 (intrinsic scale=3.75)
- top_k_list에 저장하는 RGB도 256x192로 리사이즈 필요
- 안 하면 Open3D RGBDImage 생성 에러

### 4.3 2D mask는 triangle rasterization
- vertex projection만 하면 sparse → IoU 낮음 → "mask coverage too low"
- `cv2.fillConvexPoly`로 mesh triangle 채워서 rasterize 해야 함

### 4.4 OPDMulti는 cabinet door specific
- Any3DIS class-agnostic이 154개 region 다 segment (벽, 바닥, 천장 포함)
- OPDM이 articulation 없는 region은 자동 reject → 11개만 통과
- **이건 정상 동작**. 다 통과시키려면 다른 articulation 모델 필요

### 4.5 224 vs 12 비교
- 원본 REACT3D scene2part: 12 parts (cabinet door 위주, GroundedSAM "door" prompt)
- Any3DIS class-agnostic: 224 parts (모든 surface region)
- 최종 articulated: Any3DIS 11 vs 원본 12 — **비슷**
- 개선 효과는 cabinet door **edge 정확도**와 **인접 part 분리**에서 봐야 함 (viser에서 확인)

---

## 5. 파일 변경 / 신규 목록

### 신규 (이번 세션)
- `REACT3D/any3dis_integration/prepare_any3dis_data.py` — REACT3D → Any3DIS 변환
- `REACT3D/any3dis_integration/generate_superpoints.py` — Felzenszwalb mesh segmentation
- `REACT3D/any3dis_integration/generate_dc_features.py` — 12D geometric feature
- `REACT3D/any3dis_integration/any3dis_to_react3d.py` — Any3DIS proposals → REACT3D scene_output
- `/tmp/Any3DIS_unofficial/configs/react3d.yaml` — Any3DIS config
- `/tmp/Any3DIS_unofficial/tracker_2d.txt` — 빈 파일 (Any3DIS 실행 추적용)

### 데이터 (생성됨)
- `REACT3D/data/any3dis_data/` — Any3DIS 입력 데이터 (137MB)
- `REACT3D/data/any3dis_exp/react3d_run1/mask2d_lifted/react3d_scene.pth` — Any3DIS 출력 (20MB)
- `REACT3D/data/checkpoints/sam2_hiera_large.pt` — SAM2 checkpoint (857MB)
- `REACT3D/data/scene_any3dis/` — Any3DIS 기반 REACT3D pipeline 진행 중

### 클론된 코드 (`/tmp/Any3DIS_unofficial`)
- https://github.com/PhucNDA/Any3DIS_unofficial (PhucNDA = 논문 1저자 본인의 코드, "unofficial"은 회사 release가 아니란 의미)

---

## 6. 한 줄 요약

**Any3DIS 통합 9단계 중 4단계 남음** (texture 확인 → blender DAE → base_pca → URDF → viser). 결과 비교는 이전 세션의 `basepca_v2_20260513`와 비교.
