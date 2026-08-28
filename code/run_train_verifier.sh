#!/usr/bin/bash
#SBATCH -J dcm-vtrain
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad
#SBATCH -t 1-0
#SBATCH -o logs/vtrain-%A.out


pwd; hostname; date
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || source ~/.bashrc
CONDA_ENV=${CONDA_ENV:-dcm}
conda activate "$CONDA_ENV" || { echo "[ERROR] conda activate $CONDA_ENV 실패"; exit 1; }

export TOKENIZERS_PARALLELISM=false
export TMPDIR="/tmp/${USER}-${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR" verifier_out logs
trap 'rm -rf "$TMPDIR"' EXIT

DATA=${DATA:-sft_data/verifier_raw.jsonl}
OUT=${OUT:-./verifier_out/v1}
EPOCHS=${EPOCHS:-1}
LR=${LR:-1e-4}

for f in train_verifier.py "$DATA" ds_config_zero2.json; do
    [ -f "$f" ] || { echo "[ERROR] 파일 없음: $f"; exit 1; }
done
python -c "from peft import LoraConfig; import transformers, peft; \
print('[preflight] tf', transformers.__version__, '| peft', peft.__version__)" || exit 1
echo "[preflight] data=$DATA  out=$OUT  epochs=$EPOCHS  lr=$LR"

NGPU=${NGPU:-${SLURM_GPUS_ON_NODE:-4}}
PORT=$((29000 + ${SLURM_JOB_ID:-0} % 900))
echo "[preflight] GPU ${NGPU}장 → 실효 배치 $((4 * NGPU * 8))"
torchrun --nproc_per_node="$NGPU" --master_port="$PORT" train_verifier.py \
    --data "$DATA" --out_dir "$OUT" \
    --epochs "$EPOCHS" --lr "$LR" \
    --per_device_batch 4 --grad_accum 8 \
    --max_len 1536 \
    --deepspeed ds_config_zero2.json

RC=$?
date
[ "$RC" -eq 0 ] || { echo "[ERROR] 학습 실패 (exit $RC)"; exit "$RC"; }

echo
echo "=== 다음: 가중 다수결로 평가 ==="
echo "  conda activate $CONDA_ENV"
echo "  python infer_verifier.py --verifier $OUT/merged \\"
echo "      --cands <후보파일> --out verif_eval"
echo
echo "=== 비교 기준 (my_val_filtered 486, temp 0.8) ==="
echo "  다수결 Maj@16 : 75.0%   ← 이걸 넘어야 의미가 있다"
echo "  천장 pass@16  : 84.9%   ← 완벽한 선택기의 상한"
echo "  → 갭 9.9%p 중 얼마를 먹었는지가 성적표다"
exit 0
