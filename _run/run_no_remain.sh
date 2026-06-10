set -e
# Paths resolve relative to THIS script; binaries are overridable via env vars
# (REACT3D_ENV / BLENDER_BIN / CONDA_SH / REACT3D_PY / REACT3D_DATA_DIR / CUDA_VISIBLE_DEVICES).
export CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
C="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root (this script lives in _run/)
DATA="${REACT3D_DATA_DIR:-$C/_run/example_input_scene}"
BLENDER="${BLENDER_BIN:-blender}"
source "$CONDA_SH" && conda activate "${REACT3D_ENV:-react3d}"
PY="${REACT3D_PY:-python}"
echo "===== STAGE 1: scene2part ====="; bash $C/scene2part/scene2part.sh "$DATA"
echo "===== STAGE 2: part2interactive ====="; bash $C/part2interactive/part2interactive.sh "$DATA"
echo "===== STAGE 3: texture (skip ONLY remain_scene texturing; parts fully textured + glb->dae) ====="
FD=$DATA/scene_output_filtered
# Move remain_scene.ply aside so convert_texture skips the slow whole-scene texturing.
[ -f "$FD/remain_scene.ply" ] && mv "$FD/remain_scene.ply" "$FD/remain_scene.ply.skip" || true
$PY $C/texture/convert_texture.py --folder "$FD" --texture-size 128 --remain-texture-size 512
[ -f "$FD/remain_scene.ply.skip" ] && mv "$FD/remain_scene.ply.skip" "$FD/remain_scene.ply" || true
# blender glb->dae: REQUIRED — simulation_ready needs part.dae per part (only remain_scene.glb,
# which we never produced, would be the slow item; parts are tiny).
cd $C/texture && "$BLENDER" --background --python blender_glb_to_dae.py -- --folder "$FD"
echo "===== STAGE 3c: shared-base grouping (coplanar parts -> shared_bases/group_XXX_base.obj) ====="
# REQUIRED for grouped cabinet shells; reads remain_scene.ply (restored above) for real cabinet depth.
$PY $C/simulation_ready/generate_shared_base_groups.py --scene_dir "$FD"
echo "===== STAGE 4: simulation_ready ====="; bash $C/simulation_ready/simulation_ready.sh "$DATA"
echo "===== PIPELINE DONE ====="
