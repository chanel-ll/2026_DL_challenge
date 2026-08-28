#!/usr/bin/bash
#SBATCH -J dcm-veval
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad
#SBATCH -t 1-0
#SBATCH -o logs/veval-%A.out
#SBATCH -w ariel-v12


pwd; hostname; date
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || source ~/.bashrc
conda activate dcm || { echo "[ERROR] conda activate dcm 실패"; exit 1; }
python -c "import vllm; print('[preflight] vllm', vllm.__version__)" || exit 1

export TOKENIZERS_PARALLELISM=false
export TMPDIR="/tmp/${USER}-${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR" sft_data logs
trap 'rm -rf "$TMPDIR"' EXIT

VERIF=${VERIF:-./verifier_out/v1/merged}
VAL=${VAL:-my_val_big.csv}
CANDS=${CANDS:-sft_data/val_cands_n16.jsonl}
N=${N:-16}
TEMP=${TEMP:-0.8}
ALPHA=${ALPHA:-6.0}
SUBMIT=${SUBMIT:-}
OUTDIR=${OUTDIR:-./verif_eval}
mkdir -p "$OUTDIR"
GENMODEL=${GENMODEL:-Qwen/Qwen2.5-3B-Instruct}
SUBMIT_MAJ=${SUBMIT_MAJ:-}
SEED=${SEED:-0}
MAXSEQS=${MAXSEQS:-64}
NGPU=${NGPU:-${SLURM_GPUS_ON_NODE:-4}}

for f in gen_verifier_data.py infer_verifier.py baseline_infer.py "$VAL"; do
    [ -f "$f" ] || { echo "[ERROR] 파일 없음: $f"; exit 1; }
done
[ -d "$VERIF" ] || { echo "[ERROR] verifier 없음: $VERIF (merge 했는지 확인)"; exit 1; }
if [ ! -f "$CANDS" ] && [ "$N" -gt "$MAXSEQS" ]; then
    echo "[ERROR] N($N) > MAXSEQS($MAXSEQS) — vLLM 이 스케줄을 못 해 영원히 멈춘다."
    echo "        (job 389452 실측: 6시간 15분간 0/208 완료, 산출물 0)"
    echo "        권장: n 을 쪼개고 SEED 를 다르게 준 뒤 merge_cands.py 로 합칠 것"
    echo "          CANDS=sft_data/lb_cands_n64_s1.jsonl SEED=1 N=64 sbatch ..."
    echo "          python merge_cands.py --out sft_data/lb_cands_n128.jsonl \\"
    echo "              sft_data/lb_cands_n64.jsonl sft_data/lb_cands_n64_s1.jsonl"
    exit 1
fi
NROW=$(python -c "import pandas as pd; print(len(pd.read_csv('$VAL')))")
echo "=============================================="
echo "[preflight] verifier = $VERIF"
echo "[preflight] 평가셋   = $VAL (${NROW}문항)  n=$N temp=$TEMP  GPU ${NGPU}장"
echo "[preflight] 생성모델 = $GENMODEL"
echo "[preflight] alpha    = $ALPHA"
[ -n "$SUBMIT_MAJ" ] && echo "[preflight] 다수결 제출본도 함께 생성 → $SUBMIT_MAJ"
echo "=============================================="

if [ -f "$CANDS" ]; then
    echo "[skip] $CANDS 이미 존재 → 생성 건너뜀 (다시 뽑으려면 삭제할 것)"
else
    PIDS=()
    for S in $(seq 0 $((NGPU-1))); do
        CUDA_VISIBLE_DEVICES=$S python gen_verifier_data.py \
            --problems "$VAL" --out "$CANDS" --model "$GENMODEL" \
            --n "$N" --temperature "$TEMP" --seed "$SEED" \
            --max_num_seqs "$MAXSEQS" \
            --shard "$S" --num_shards "$NGPU" --tp 1 \
            > "logs/veval-gen-s${S}-${SLURM_JOB_ID}.log" 2>&1 &
        PIDS+=($!)
    done
    FAIL=0
    for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
    [ "$FAIL" -eq 0 ] || { echo "[ERROR] 후보 생성 실패. logs/veval-gen-s*.log 확인"; exit 1; }
    cat "${CANDS%.jsonl}".shard*.jsonl > "$CANDS"
    echo "→ $CANDS"
fi
date

SCORE_NGPU=${SCORE_NGPU:-$NGPU}
SCORE_MAXSEQS=${SCORE_MAXSEQS:-256}
SUBMIT_ARG=""
if [ -n "$SUBMIT" ]; then
    SUBMIT_ARG="--submit_csv $SUBMIT"
    [ "$SCORE_NGPU" -gt 1 ] && echo "[info] SUBMIT 지정 → 샤딩 끄고 1장으로 돈다"
    SCORE_NGPU=1
fi
NCAND=$(python -c "
import json,sys
print(sum(len(json.loads(l)['cands']) for l in open('$CANDS',encoding='utf-8') if l.strip()))")
echo "[preflight] 채점 후보 ${NCAND}개  GPU ${SCORE_NGPU}장  max_num_seqs=${SCORE_MAXSEQS}"

if [ "$SCORE_NGPU" -le 1 ]; then
    CUDA_VISIBLE_DEVICES=0 python infer_verifier.py \
        --verifier "$VERIF" --cands "$CANDS" --max_num_seqs "$SCORE_MAXSEQS" \
        --out_dir "$OUTDIR" --alpha "$ALPHA" --tp 1 $SUBMIT_ARG
    RC=$?
else
    SPIDS=()
    for S in $(seq 0 $((SCORE_NGPU-1))); do
        CUDA_VISIBLE_DEVICES=$S python infer_verifier.py \
            --verifier "$VERIF" --cands "$CANDS" --max_num_seqs "$SCORE_MAXSEQS" \
            --out_dir "$OUTDIR" --alpha "$ALPHA" --tp 1 \
            --shard "$S" --num_shards "$SCORE_NGPU" \
            > "logs/veva-s${S}-${SLURM_JOB_ID:-manual}.log" 2>&1 &
        SPIDS+=($!)
    done
    RC=0
    for pid in "${SPIDS[@]}"; do wait "$pid" || RC=1; done
    if [ "$RC" -eq 0 ]; then
        python - "$OUTDIR" "$SCORE_NGPU" <<'PY'
import glob, json, os, re, sys
outdir, nsh = sys.argv[1], int(sys.argv[2])
sh = sorted(glob.glob(os.path.join(outdir, "verif_shard*_*.jsonl")))
if len(sh) != nsh:
    raise SystemExit(f"[ERROR] 샤드 {len(sh)}개 (기대 {nsh}개). 합치지 않는다.")
base = re.sub(r"^verif_shard\d+_", "", os.path.basename(sh[0]))
merged = os.path.join(outdir, "verif_detail_" + base)
rows = []
for p in sh:
    rows += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
with open(merged, "w", encoding="utf-8") as f:
    for d in rows:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
if any(d["gold"] is not None for d in rows):
    with open(os.path.join(outdir, "verif_detail.jsonl"), "w", encoding="utf-8") as f:
        for d in rows:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
for p in sh:
    os.remove(p)
print(f"[merge] 샤드 {nsh}개 → {merged}  ({len(rows):,}행, 샤드 삭제)")
PY
        RC=$?
    fi
fi
[ "$RC" -eq 0 ] || { date; echo "[ERROR] 재순위화 실패 (exit $RC). logs/veva-s*.log 확인"; exit "$RC"; }

if [ -n "$SUBMIT_MAJ" ]; then
    echo "--- 같은 후보로 다수결(α=0) 제출본 추가 ---"
    CUDA_VISIBLE_DEVICES=0 python infer_verifier.py \
        --verifier "$VERIF" --cands "$CANDS" \
        --out_dir "$OUTDIR" --alpha 0.0 --tp 1 --submit_csv "$SUBMIT_MAJ"
    RC=$?
    [ "$RC" -eq 0 ] || { date; echo "[ERROR] 다수결 제출본 생성 실패"; exit "$RC"; }
fi
date

echo
echo "=== 판정 ==="
echo "  wmaj > maj  → 채택. 리더보드 831 에 같은 파이프라인을 돌린다"
echo "  wmaj ≤ maj  → 폐기하고 다수결 유지 (손실 0). alpha 를 2.0, 4.0 으로 재시도해볼 것:"
echo "      ALPHA=2.0 CANDS=$CANDS sbatch run_verifier_eval.sh   # 후보 재사용, 10분"
echo
echo "  ※ 기대치: 갭 11.1%p 중 20~40% 회수 = wmaj 77~79% (maj 74.8% 대비)"
exit 0
