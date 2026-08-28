#!/usr/bin/bash
set -u

NODE=${1:?사용: ./final_score.sh <NODE> [N]}
NN=${2:-256}

ALPHA=2
VERIF=${VERIF:-./verifier_out/v1/merged}
PROB=${PROB:-test.csv}
TAG=${TAG:-$(basename "$PROB" .csv)}
CANDS=sft_data/${TAG}_cands_n${NN}.jsonl
OUTDIR=verif_eval/${TAG}_n${NN}

[ -f run_verifier_eval.sh ] || { echo "[ERROR] code/ 안에서 실행할 것"; exit 1; }
[ -f "$PROB" ]  || { echo "[ERROR] 문제 파일 없음: $PROB"; exit 1; }
[ -f "$CANDS" ] || { echo "[ERROR] 후보 파일 없음: $CANDS  (병합을 먼저 할 것)"; exit 1; }
if [ ! -d "$VERIF" ]; then
    echo "[ERROR] verifier 없음: $VERIF"
    echo "        서버에서 링크를 걸 것:"
    echo "          ln -s /nas2/data/chan12/qwen_math/verifier_out ./verifier_out"
    exit 1
fi
mkdir -p verif_eval logs

echo "==============================================" >&2
echo " verifier  : $VERIF" >&2
echo " 후보      : $CANDS   (n=$NN)" >&2
echo " alpha     : $ALPHA          <- 본선 설정" >&2
echo " 출력      : $OUTDIR" >&2
echo " 노드      : $NODE / GPU 4장 (샤딩 필수)" >&2
echo " SUBMIT    : 넘기지 않음 (샤딩 유지)" >&2
echo "==============================================" >&2

VERIF="$VERIF" CANDS="$CANDS" VAL="$PROB" \
N="$NN" ALPHA="$ALPHA" OUTDIR="$OUTDIR" \
sbatch --parsable --gres=gpu:4 -w "$NODE" -t 1-0 run_verifier_eval.sh
