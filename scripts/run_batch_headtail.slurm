#!/bin/bash
#SBATCH -J cpcomposer_h2t
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o logs/stdout.%j
#SBATCH -e logs/stderr.%j
#SBATCH --no-requeue
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

set -euo pipefail
mkdir -p logs

# ===== user settings =====
PROJECT_ROOT=${PROJECT_ROOT:-/path/to/CP-Composer}
PDB_DIR=${PDB_DIR:-${PROJECT_ROOT}/StructCenter74}
OUT_DIR=${OUT_DIR:-${PROJECT_ROOT}/out}
RECEPTOR_CHAIN=${RECEPTOR_CHAIN:-R}
LIGAND_CHAIN=${LIGAND_CHAIN:-L}
CUTOFF=${CUTOFF:-10.0}
N_SAMPLES=${N_SAMPLES:-100}
GPU_ID=${GPU_ID:-0}
# =========================

cd "$PROJECT_ROOT"

nvidia-smi
python - <<'PY'
import torch
print('torch version:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda device count:', torch.cuda.device_count())
PY

python scripts/batch_headtail_from_ligand.py \
  --pdb_dir "$PDB_DIR" \
  --out_dir "$OUT_DIR" \
  --receptor_chain "$RECEPTOR_CHAIN" \
  --ligand_chain "$LIGAND_CHAIN" \
  --cutoff "$CUTOFF" \
  --n_samples "$N_SAMPLES" \
  --gpu "$GPU_ID" \
  --project_root "$PROJECT_ROOT"
