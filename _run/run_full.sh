set -e
# FULL pipeline INCLUDING remain_scene whole-scene texturing (slow).
# Differs from run_no_remain.sh ONLY in that remain_scene.ply is LEFT IN PLACE, so:
#   - convert_texture.py also textures remain_scene.ply -> remain_scene.glb  (texture/convert_texture.py:250)
#   - blender_glb_to_dae.py also converts remain_scene.glb  -> remain_scene.dae (texture/blender_glb_to_dae.py:60)
# Grouping (STAGE 3c) is included, same as the updated run_no_remain.sh.
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
echo "===== STAGE 3: texture (FULL — parts + remain_scene whole-scene texturing; SLOW) ====="
FD=$DATA/scene_output_filtered
# remain_scene.ply intentionally NOT moved aside -> whole remaining scene gets textured.
$PY $C/texture/convert_texture.py --folder "$FD" --texture-size 128 --remain-texture-size 512
# blender glb->dae: converts every part.glb AND remain_scene.glb -> remain_scene.dae
cd $C/texture && "$BLENDER" --background --python blender_glb_to_dae.py -- --folder "$FD"
echo "===== STAGE 3c: shared-base grouping (coplanar parts -> shared_bases/group_XXX_base.obj) ====="
$PY $C/simulation_ready/generate_shared_base_groups.py --scene_dir "$FD"
echo "===== STAGE 4: simulation_ready ====="; bash $C/simulation_ready/simulation_ready.sh "$DATA"
echo "===== PIPELINE DONE (WITH remain texturing) ====="
