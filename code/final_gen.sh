#!/usr/bin/bash
set -u

SEED=${1:?사용: ./final_gen.sh <SEED> <NODE> [GPUS]}
NODE=${2:?사용: ./final_gen.sh <SEED> <NODE> [GPUS]}
GPUS=${3:-4}

PROB=${PROB:-test.csv}
MODEL=Qwen/Qwen2.5-3B-Instruct
N=64
TEMP=0.8
MAXTOK=4096
TAG=${TAG:-$(basename "$PROB" .csv)}
OUT=sft_data/${TAG}_cands_n64_s${SEED}.jsonl

[ -f "$PROB" ] || { echo "[ERROR] 문제 파일 없음: $PROB"; exit 1; }
[ -f run_gen_verifier.sh ] || { echo "[ERROR] code/ 안에서 실행할 것"; exit 1; }
[ -f "$OUT" ] && { echo "[ERROR] 이미 있음: $OUT  (지우거나 SEED 를 바꿀 것)"; exit 1; }
mkdir -p sft_data logs

echo "==============================================" >&2
echo " 문제      : $PROB" >&2
echo " 모델      : $MODEL" >&2
echo " n / temp  : $N / $TEMP" >&2
echo " max_tokens: $MAXTOK          <- 본선 설정" >&2
echo " seed      : $SEED" >&2
echo " 출력      : $OUT" >&2
echo " 노드/GPU  : $NODE / $GPUS 장" >&2
echo "==============================================" >&2

MODEL="$MODEL" PROB="$PROB" OUT="$OUT" \
N="$N" TEMP="$TEMP" MAXTOK="$MAXTOK" SEED="$SEED" \
sbatch --parsable --gres=gpu:"$GPUS" -w "$NODE" -t 1-0 run_gen_verifier.sh
