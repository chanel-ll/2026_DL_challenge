#!/usr/bin/bash
#SBATCH -J dcm-vgen
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad
#SBATCH -t 1-0
#SBATCH -o logs/vgen-%A.out


pwd; hostname; date
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || source ~/.bashrc
CONDA_ENV=${CONDA_ENV:-dcm}
conda activate "$CONDA_ENV" || { echo "[ERROR] conda activate $CONDA_ENV 실패"; exit 1; }
python -c "import vllm; print('[preflight] vllm', vllm.__version__)" || exit 1

export TOKENIZERS_PARALLELISM=false
export TMPDIR="/tmp/${USER}-${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR" sft_data logs
trap 'rm -rf "$TMPDIR"' EXIT

NGPU=${NGPU:-${SLURM_GPUS_ON_NODE:-4}}
N=${N:-8}
TEMP=${TEMP:-0.8}
case "$TEMP" in
    ''|*[!0-9.]*)
        echo "[ERROR] TEMP='$TEMP' 가 숫자가 아니다 (환경변수 TEMP 와 충돌한 것으로 보인다)."
        echo "        TEMP=0.8 을 명시해서 다시 던질 것."; exit 1;;
esac
PROB=${PROB:-train_clean_noval.csv}
OUT=${OUT:-sft_data/verifier_raw.jsonl}
LIMIT=${LIMIT:-0}
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
MAXTOK=${MAXTOK:-1024}
SEED=${SEED:-0}
MINP=${MINP:-0}
REPPEN=${REPPEN:-1.0}

for f in gen_verifier_data.py baseline_infer.py "$PROB"; do
    [ -f "$f" ] || { echo "[ERROR] 파일 없음: $f"; exit 1; }
done
NROW=$(python -c "import pandas as pd; print(len(pd.read_csv('$PROB')))")
echo "=============================================="
echo "[preflight] problems = $PROB (${NROW}문항)"
echo "[preflight] n=$N  temp=$TEMP  → 후보 약 $((NROW * N))개"
echo "[preflight] model = $MODEL"
echo "[preflight] max_tokens = $MAXTOK   seed = $SEED"
echo "[preflight] min_p = $MINP   repetition_penalty = $REPPEN"
echo "[preflight] out = $OUT   GPU ${NGPU}장 샤딩"
[ "$LIMIT" -gt 0 ] && echo "[preflight] ⚠️ LIMIT=$LIMIT (스모크 모드)"
echo "=============================================="

OVERWRITE=${OVERWRITE:-0}
if [ -f "$OUT" ] && [ "$OVERWRITE" != "1" ]; then
    echo "[ERROR] OUT 이 이미 있다: $OUT"
    echo "        다른 job 의 산출물을 덮어쓰려는 것일 수 있다."
    echo "        OUT 을 job 마다 다르게 주거나, 정말 덮어쓰려면 OVERWRITE=1"
    exit 1
fi
LOCK="${OUT}.lock"
if [ -f "$LOCK" ] && [ "$OVERWRITE" != "1" ]; then
    echo "[ERROR] 같은 OUT 을 쓰는 job 이 이미 돌고 있다: $LOCK"
    cat "$LOCK"
    echo "        SEED 만 바꾸고 OUT 을 안 바꾼 것이 아닌지 확인할 것."
    exit 1
fi
mkdir -p "$(dirname "$OUT")"
echo "job ${SLURM_JOB_ID:-manual}  seed $SEED  n=$N  $(date)" > "$LOCK"
trap 'rm -rf "$TMPDIR" "$LOCK"' EXIT

JID="${SLURM_JOB_ID:-manual}"
WORK="${OUT%.jsonl}.j${JID}"
echo "[preflight] 샤드 작업 경로 = ${WORK}.shard*.jsonl  (끝나면 삭제)"

PIDS=()
for S in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$S python gen_verifier_data.py \
        --problems "$PROB" --out "${WORK}.jsonl" --model "$MODEL" \
        --n "$N" --temperature "$TEMP" --max_tokens "$MAXTOK" --seed "$SEED" \
        --min_p "$MINP" --repetition_penalty "$REPPEN" \
        --shard "$S" --num_shards "$NGPU" --tp 1 --limit "$LIMIT" \
        > "logs/vgen-s${S}-${JID}.log" 2>&1 &
    PIDS+=($!)
done
FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
date
[ "$FAIL" -eq 0 ] || { echo "[ERROR] 일부 샤드 실패. logs/vgen-s*-${JID}.log 확인"; exit 1; }

NSHARD=$(ls -1 "${WORK}".shard*.jsonl 2>/dev/null | wc -l)
if [ "$NSHARD" -ne "$NGPU" ]; then
    echo "[ERROR] 샤드 파일이 ${NSHARD}개다 (기대 ${NGPU}개). 합치지 않는다."
    exit 1
fi
cat "${WORK}".shard*.jsonl > "$OUT"
rm -f "${WORK}".shard*.jsonl
NLINE=$(wc -l < "$OUT")
echo "→ $OUT   ${NLINE}행 (샤드 ${NSHARD}개 병합, 작업파일 삭제)"

python - "$OUT" <<'PY'
import json, sys
from collections import Counter
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
c = w = 0; st = Counter(); lens = []
for r in rows:
    c += r["n_correct"]; w += r["n_wrong"]
    st["mixed" if r["n_correct"] and r["n_wrong"] else
       "all_correct" if r["n_wrong"] == 0 else "all_wrong"] += 1
    lens += [len(x["text"]) for x in r["cands"]]
tot = max(c + w, 1); n = len(rows)
ncand = [len(r["cands"]) for r in rows]
lens.sort()
print("\n" + "=" * 62)
print(f"문제        : {n:,}")
print(f"후보        : {tot:,}   문제당 {min(ncand)}~{max(ncand)}개")
print(f"후보 길이   : 중앙값 {lens[len(lens)//2]:,}자")
if all(r.get("gold") is None for r in rows):
    print("=" * 62)
    print("⚠️ gold 없음 (리더보드/테스트 후보) — 정오 통계는 의미가 없어 생략한다.")
    print("판단: **문제 수와 문제당 후보 수**만 보면 된다. 정답률·혼합비율이 아니다.")
    print("  · 여러 시드를 합칠 때 merge_cands.py 가 seed 독립성까지 검사한다")
else:
    print(f"정답/오답   : {c:,} / {w:,}  ({c/tot*100:.1f}% 정답률)")
    print(f"★혼합 문제  : {st['mixed']:,} ({st['mixed']/n*100:.1f}%)  ← verifier 학습에 쓰이는 것")
    print(f" 전원 정답  : {st['all_correct']:,} ({st['all_correct']/n*100:.1f}%)  음성 예시 없음")
    print(f" 전원 오답  : {st['all_wrong']:,} ({st['all_wrong']/n*100:.1f}%)  양성 예시 없음")
    print("=" * 62)
    print("판단:")
    print("  · 혼합 비율이 40% 미만이면 n 을 16 으로 올려 판별 가능한 쌍을 늘릴 것")
    print("  · 정답률이 50~70% 면 이상적. 90% 넘으면 음성 예시가 부족하다")
PY

echo
echo "=== 다음 ==="
echo "  1) 혼합 문제 비율 확인 (40% 이상이어야 학습 가치가 있다)"
echo "  2) train_verifier.py 로 LoRA 학습 (dcm_rl 환경)"
echo "  3) 가중 다수결로 my_val_filtered 평가 → baseline Maj@16 75.0 과 비교"
exit 0
