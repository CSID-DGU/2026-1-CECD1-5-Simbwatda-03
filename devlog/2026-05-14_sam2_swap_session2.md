# Any3DIS 마무리 + SAM2 image swap 시작 (2026-05-14 세션 2)

이전 세션(2026-05-14_any3dis_handoff.md)에서 Any3DIS 통합의 9단계 중 4단계(texture→DAE→base_pca→URDF→viser)를 남기고 핸드오프함. 이번 세션은 그 마무리 + Any3DIS 결과 평가 + 다음 방향(SAM2 mask refinement)으로 pivot.

---

## 0. 시작 시점 / 핸드오프 상태

이전 세션에서 끝낸 것:
- Any3DIS 통합 (env, data, superpoint, dc_features, Any3DIS 실행 → 224 proposals)
- Reverse converter (`any3dis_to_react3d.py`) → 154개 mesh
- REACT3D part2interactive → 11 articulated parts
- filter_duplicates → 8 parts
- generate_remain_scene → remain_scene.ply

배경 task `bfuvlqkbk`로 `convert_texture.py`가 돌아가고 있었는데, 사용자가 어제 중간에 끄는 바람에 일부만 생성됨.

---

## 1. Texture 진행 — remain_scene 병목 발견

### 1.1 초기 상태 확인
- mesh_009, 018, 034, 039, 043, 055, 164: `part.glb` 있음 (7/8)
- mesh_012: 없음 — top_k_list.pkl이 없는 mesh라 texture 단계 자체에서 skip된 듯
- remain_scene.glb: 없음

### 1.2 처음부터 다시 돌리기
"안전하게 처음부터" 방침. tmux 세션 `texture`에서:
```bash
python texture/convert_texture.py --folder .../scene_output_filtered \
  --texture-size 128 --remain-texture-size 512
```

### 1.3 막힌 곳: tmux + conda run 출력 무
처음에 `conda run -n react3d python ...`을 tmux에서 돌렸더니 stdout이 끝까지 안 나옴 (사용자가 attach해도 빈 화면). `conda run`이 stdout을 버퍼링/스왈로우하는 듯.
- **해결**: `bash -c "source .../conda.sh && conda activate react3d && python ..."` 형태로 활성화 후 직접 실행

또 하나: `source /opt/conda/etc/profile.d/conda.sh` 경로가 틀렸음 — 시스템 conda가 `/home/gpuadmin/miniconda3/`에 있어서 tmux 세션이 바로 죽음. 정정.

### 1.4 진짜 막힌 곳: remain_scene 처리 7시간+ 안 끝남
- 8/10에서 멈춤 — remain_scene.ply(31MB, 762893 vertex) + 512 해상도로 xatlas UV parameterization
- 사용자가 7시간째 대기 중. tqdm은 "14초/it" 거짓말 (이전 part 평균)

### 1.5 결정: 죽이지 않고 남겨두기
처음엔 "tmux 죽이고 remain_scene 빼고 진행하자"고 제안했는데, 사용자가 "죽이지 말고 두고 다른 단계 진행"이라고 정정. 그래서 texture는 그대로 둔 채 7개 part.glb로만 다음 단계 진행.
- 새 memory에 `feedback_dont_kill_bg.md` 추가 — 백그라운드 작업은 함부로 죽이지 말 것

---

## 2. 남은 파이프라인 단계 진행 (7개 mesh)

### 2.1 Blender GLB → DAE (성공)
```bash
blender --background --python texture/blender_glb_to_dae.py -- \
  --folder .../scene_output_filtered
```
- 7개 mesh.glb → mesh.dae 변환 완료
- mesh_012는 part.glb 없어서 자동 skip

### 2.2 base_pca.obj 생성 (성공)
```bash
python simulation_ready/generate_base_from_surface.py --scene_dir .../scene_output_filtered
```
- 8개 mesh 모두 base_pca.obj 생성됨 (mesh_012도 part.obj는 있음)

### 2.3 URDF 생성 (성공, 7개)
```bash
python simulation_ready/generate_urdf_basepca.py \
  --folder_path .../scene_output_filtered \
  --output_path .../urdf_any3dis
```
- 7개 URDF 생성. mesh_012는 `part.dae` 없어서 skip — 예상대로

### 2.4 viser 디렉토리 세팅
**막힌 곳**: vis_interactive_urdf.py가 `{scene_name}_urdfs` / `{scene_name}_meshes` 명명 규칙을 가정. 처음에 `any3dis_urdfs` / `any3dis_meshes`로 만들었더니 `not found` 에러.
- **해결**: scene_dir basename(`any3dis_20260514`)과 일치하도록 `any3dis_20260514_urdfs/meshes`로 rename

### 2.5 viser 띄우기 — 포트 충돌 / 동시 실행 문제
3개나 동시에 띄워버려서 같은 포트 충돌. `pkill -f`로 정리 후 tmux 세션 `viser`에서 단일 실행. `http://localhost:8080` 정상 동작.

---

## 3. Any3DIS 결과 평가 — 사용자 분석

viser에서 새 결과(8 parts)와 이전 결과(`basepca_v2_20260513`, 12 parts) 비교. 사용자 관찰:
- **mesh_164 = 옛 mesh_951 + 709가 merged**, 심지어 rotation이 prismatic으로 바뀜
- **mesh_043 = 새로 생긴 정체 불명**
- **옛 mesh_000, 002, 197, 500, 970이 사라짐**

### 3.1 좌표 매칭 분석
URDF의 joint origin을 비교:

| 새 (Any3DIS) | 옛 (REACT3D) | 위치 매칭 | 비고 |
|---|---|---|---|
| mesh_009 (1.36, 1.57, 1.82) | mesh_034 | ✅ 위쪽 문 |
| mesh_018 (2.57, 1.55, 1.94) | mesh_408 | ✅ 위쪽 문 |
| mesh_034 (0.31, 1.48, 0.49) | mesh_023 | ✅ |
| mesh_039 (2.58, 1.56, 1.83) | mesh_033 | ✅ 위쪽 문 |
| mesh_055 (3.47, 1.58, 1.95) | mesh_1099 | ✅ 위쪽 문 |
| **mesh_164 (1.97, 1.35, 0.62) prismatic** | (옛 951, 709, 500, 970, 197 영역) | ❌ 5개가 1개로 merge |
| **mesh_043 (2.42, 0.17, 3.22)** | — | ❌ Z=3.22m 천장 region, spurious |

### 3.2 왜 이렇게 됐나
1. **인접 drawer 시각 동일**: SAM2 video tracking이 같은 cabinet의 연속 drawer를 같은 instance로 묶음
2. **Class-agnostic의 함정**: REACT3D는 GroundedSAM에 "door" prompt를 줘서 처음부터 door-shape만 뽑음. Any3DIS는 모든 surface segment → OPDM 혼자 articulation 판단해야 함 → 길쭉 wall region도 prismatic으로 오인
3. **12D geometric feature 한계**: 인접 drawer가 normal/color 거의 동일해서 feature 분리 안 됨. Mask3D pretrained라도 ScanNet200의 "cabinet" 클래스가 drawer 단위로 학습 안 됨

### 3.3 결론
Any3DIS는 task mismatch. "fine-grained adjacent drawer 분리"는 Any3DIS의 강점인 "open vocabulary 인스턴스 분리"와 정반대. REACT3D의 GroundedSAM detection 자체는 정상이었고, **약점은 mask edge 정밀도 / 인접 boundary**였음.

---

## 4. 방향 전환: SAM2 image mask refinement (옵션 F)

사용자 인사이트: "원래 detection이 더 잘하는데, 거기서 segmentation만 보완하면 되는 거 아니야?"

REACT3D 약점 재진단 (이전 세션 노트 기반):
- 1099 vs 408: 2.3cm overlap — 인접 문 mask 경계가 살짝씩 침범 (edge 정밀도)
- 033/034: 같은 평면을 두 part로 over-segment

→ **detection 단계는 건드리지 말고, SAM step만 SAM2로 swap**

### 4.1 옵션 비교 (사용자에게 제시)
| 옵션 | 비용 | 효과 |
|---|---|---|
| 1: SAM2 image predictor swap | 30분-1시간 | mask quality 향상, frame consistency는 그대로 |
| 2: SAM2 video propagation 통합 | 4-6시간 | multi-view consistency 진짜 확보 |

사용자가 **옵션 1**부터 선택. PoC 빠르게 보고 옵션 2 갈지 결정.

### 4.2 환경 충돌 문제
- react3d env: torch 2.1.2 (SAM2 요구 2.3+ 미달)
- any3dis env: torch 2.3.1 + SAM2 있음, GroundingDINO 없음

**해결**: 분할 실행. GroundingDINO는 이미 돌렸으므로 `grounded_sam/mask/*.json`에 bbox 다 저장돼 있음. **GroundingDINO 재실행 필요 없음**. any3dis env에서 box 읽어 SAM2로 mask만 재생성하면 됨.

### 4.3 SAM2 swap 스크립트 작성
`grounded_sam/sam2_redo_masks.py` (신규):
- 입력: `scene_dir/grounded_sam/mask/*.json` (GroundingDINO bbox)
- 입력: `scene_dir/images_2/frame_*.png`
- 출력: `scene_dir/grounded_sam2/mask_data/*.pkl` (SAM2 mask, 원본 포맷 호환)
- 출력: `scene_dir/grounded_sam2/mask/*.json` (box json 복사)
- SAM2 image predictor 사용 (sam2_hiera_l.yaml + sam2_hiera_large.pt)

```bash
PYTHONPATH=/tmp/Any3DIS_unofficial/segmenter2d/segment-anything-2 \
conda run -n any3dis python grounded_sam/sam2_redo_masks.py \
  --scene_dir .../example_input_scene \
  --sam2_checkpoint .../data/checkpoints/sam2_hiera_large.pt \
  --sam2_config sam2_hiera_l.yaml
```

### 4.4 결과
- **1196 frame을 2분 31초에 처리** (~7.9 it/s, GPU 한 장)
- skip=0, errors=0 — 깔끔하게 완료

---

## 5. scene2part 파이프라인 재실행 (진행 중)

### 5.1 디렉토리 swap 전략
s2p_1, s2p_3가 `data_dir/grounded_sam` 경로를 하드코딩으로 보고 있음. 두 가지 옵션 중 더 안전한 백업+swap 방식 선택.

```bash
SCENE=.../example_input_scene
mv $SCENE/grounded_sam $SCENE/grounded_sam_dino       # 원본 백업
mv $SCENE/perception $SCENE/perception_dino           # 결과도 백업
mv $SCENE/grounded_sam2 $SCENE/grounded_sam           # SAM2 활성화
```

### 5.2 s2p_1 ~ s2p_6 순차 실행
tmux 세션 `s2p`에서:
```bash
python s2p_1_fast.py --data_dir $SCENE --image_dir $SCENE/images_2 \
  --mesh_path $SCENE/mesh_aligned_0.05.ply --num_max_frames 2000 \
  --num_faces_simplified 80000 --min_cooccurrence_count 3
python s2p_2.py --data_dir $SCENE
python s2p_3_fast.py --data_dir $SCENE --image_dir $SCENE/images_2 --top_k 5
python s2p_4.py --data_dir $SCENE
python s2p_5.py --data_dir $SCENE --mesh_path $SCENE/mesh_aligned_0.05.ply \
  --image_dir $SCENE/images_2 --depth_dir $SCENE/depth
python s2p_6.py -d $SCENE
```

스크립트: `/tmp/run_s2p.sh`
로그: `/tmp/s2p_log.txt`

### 5.3 현재 상태 (세션 종료 시점)
- **s2p_1_fast.py 실행 중** — 약 40분째. 4192 pairs 처리 후 큰 sort/scatter 단계 (2143794 items) 통과.
- 총 6단계 중 1단계. 나머지 5단계 비슷한 분량이면 **총 2-3시간 예상**.

---

## 6. 다음 세션에서 할 일

### 6.1 우선 확인: s2p 진행 상태
```bash
tmux list-sessions | grep s2p     # 살아있나
grep -oE "===== s2p_[0-9_a-z]+\.py =====" /tmp/s2p_log.txt | tail -5  # 어디까지 왔나
ls /home/gpuadmin/capstone/USDNet/REACT3D/data/example_input_scene/perception/  # 만들어진 결과
```
- "===== DONE =====" 보이면 끝남
- s2p_5 끝나면 `perception/vis_groups_final_mesh/real/mesh_*/` 디렉토리들 생김

### 6.2 끝났으면 결과 백업 + 비교 파이프라인
```bash
SCENE=.../example_input_scene
mv $SCENE/perception $SCENE/perception_sam2          # SAM2 결과 보관
# 또는 원본 복귀 원하면:
# mv $SCENE/grounded_sam $SCENE/grounded_sam2
# mv $SCENE/grounded_sam_dino $SCENE/grounded_sam
# mv $SCENE/perception $SCENE/perception_sam2 && mv $SCENE/perception_dino $SCENE/perception
```

### 6.3 SAM2 결과로 후속 파이프라인 진행
1. part2interactive
2. filter_duplicates
3. generate_remain_scene
4. convert_texture (또 remain_scene 7시간 문제 주의)
5. blender DAE
6. base_pca
7. URDF 생성 — output을 `urdf_sam2_20260514` 같은 이름으로
8. viser 띄워서 **3개 비교**:
   - 옛 REACT3D (`basepca_v2_20260513`)
   - Any3DIS (`any3dis_20260514`)
   - SAM2 swap (`sam2_20260514`)

### 6.4 평가 기준
- **1099 vs 408 인접 mask 겹침**: 줄어들었나?
- **033/034 over-segmentation**: 합쳐졌나? 아니면 그대로?
- 새 spurious detection 없나?
- 옛 12개 part 다 살아남나?

PoC 결과 보고 옵션 2 (SAM2 video propagation) 갈지 결정.

### 6.5 Texture 처리 (background에서 계속 진행 중)
- 처음 세션의 texture tmux 세션이 remain_scene 처리 중. 만약 끝났다면 remain_scene.glb 사용 가능
- 안 끝났으면 그대로 두거나, 별도로 part만 다시 처리

---

## 7. 핵심 발견 / 주의사항

### 7.1 tmux + conda 디버깅
- `conda run -n env python ...`은 tmux 안에서 stdout buffer 문제 발생 가능
- 대신 `bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate env && python ..."` 사용
- 시스템 conda 경로 정확히 확인 (`which conda`) — `/opt/conda` 아니라 `/home/gpuadmin/miniconda3`

### 7.2 백그라운드 작업 죽이지 말 것
사용자 정정: 오래 걸리는 작업도 죽이지 말고 두기. 다른 단계와 병렬로 진행하면 됨. memory `feedback_dont_kill_bg.md`에 기록.

### 7.3 환경 분리 정책
- react3d (torch 2.1.2): REACT3D 원본 파이프라인 전체
- any3dis (torch 2.3.1): SAM2, Any3DIS
- 직접 통합 어려움(torch 버전 충돌) → **디스크 통한 분할 실행**이 안전
- GroundingDINO bbox는 JSON으로 저장돼 있어 두 env 사이 데이터 교환 가능

### 7.4 Any3DIS = task mismatch
- Open vocabulary 인스턴스 분리에 강점
- Fine-grained adjacent drawer 같은 mismatch 문제엔 약점
- "성능 좋은 모델 = 우리 task에 좋음" 안 통함. 항상 task fit 먼저 확인

### 7.5 viser 디렉토리 명명 규칙
- `vis_interactive_urdf.py`는 `{scene_name}_urdfs` / `{scene_name}_meshes` 강제
- scene_dir basename과 일치해야 함

### 7.6 grounded_sam output 포맷 (참고)
- `mask_data/{frame}.pkl`: pickle `{'mask': tensor[N, 1, H, W]}` (bool)
- `mask/{frame}.json`: `[{value:0, label:background}, {value, label, logit, box:[x1,y1,x2,y2]}, ...]`

---

## 8. 파일 변경 / 신규 목록

### 신규 (이번 세션)
- `REACT3D/grounded_sam/sam2_redo_masks.py` — SAM2 image predictor로 mask 재생성
- `/tmp/run_s2p.sh` — s2p 1~6 순차 실행 스크립트
- `/tmp/s2p_log.txt` — s2p 실행 로그
- 메모리: `.../memory/feedback_dont_kill_bg.md`

### 데이터 변경
- `example_input_scene/grounded_sam` → `grounded_sam_dino` (백업)
- `example_input_scene/perception` → `perception_dino` (백업)
- `example_input_scene/grounded_sam2` → `grounded_sam` (SAM2 결과 활성화)
- `example_input_scene/perception/` (새로 생성 중)
- `scene_any3dis/scene_output_filtered/`: 7개 part.dae, base_pca.obj, urdf 추가
- `data/any3dis_20260514/`: viser용 symlink 디렉토리

### 살아있는 백그라운드 프로세스
- tmux `s2p`: scene2part 실행 중 (s2p_1 ~ s2p_6)
- (이전 세션의 `texture` tmux도 remain_scene 계속 처리 중일 수 있음)

---

## 9. 한 줄 요약

**Any3DIS 결과는 cabinet drawer 분리에 task mismatch로 더 나빠짐 → SAM2 image predictor swap으로 mask refinement만 PoC 시도 중. SAM2 mask 재생성(2.5분) 완료, scene2part 재실행 진행 중 (s2p_1 단계).**
