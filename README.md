# 아주 소중한 딥러닝 챌린지 2026 — Qwen2.5-3B-Instruct 수학 추론


## 핵심 방법

```
① 생성    Qwen2.5-3B-Instruct (base) 로 문제당 후보 풀이 256개를 샘플링
          n=256 은 seed 4개 × n=64 를 병합해서 만든다

② 채점    같은 3B 를 LoRA 로 학습한 검증기가 후보마다
          "이 풀이가 맞는가?" 에 대한 P(Yes) 를 매긴다

③ 집계    가중 다수결

              answer = argmax_a  Σ_{i: aᵢ = a} P(Yes_i)^α ,   α = 2

          α=0 이면 평범한 다수결이 된다. α 가 검증기를 얼마나 믿을지를 정한다.
```

## 하이퍼파라미터

### 후보 생성 (① · vLLM)

| | 값 | |
|---|---|---|
| `model` | `Qwen/Qwen2.5-3B-Instruct` |
| `temperature` | `0.8` |
| `top_p` | `0.95` |
| `max_tokens` | `4096` |
| `n` | `64` × seed 4개 = **256** |
| `max_num_seqs` | `64` |
| `seed` | 서로 다른 4개 |

### 검증기 학습 (② · LoRA SFT)

| | 값 |
|---|---|
| 베이스 | `Qwen/Qwen2.5-3B-Instruct` |
| 학습 데이터 | `train_clean_noval.csv` 에 base 로 **문제당 16개** 생성 (`temp 0.8`, `max_tokens 1024`, `seed 42`) 후 gold 대조 라벨링 |
| LoRA `r` / `alpha` / `dropout` | `32` / `64` / `0.05` |
| LoRA target | `q,k,v,o,gate,up,down_proj` |
| `epochs` | `1` |
| `lr` | `1e-4` |
| `max_len` | `1536` |
| batch | `per_device 4` × `grad_accum 8` × GPU 4 = **128** |
| `max_pairs` | `3` — 혼합 문제당 (정답, 오답) 쌍 수 |
| `unmixed_frac` | `0.15` — 전원 정답 문제를 섞는 비율 (전원 오답은 항상 제외) |
| `hard_boost` | `2` — 다수결이 지는 문제를 2배로 표집 |
| `seed` | `42` |
| 분산 | DeepSpeed ZeRO-2 (`ds_config_zero2.json`) |

학습 후 LoRA 를 base 에 병합해 `verifier_out/v1/merged` 로 저장합니다 (스크립트가 수행).

### 집계 (③)

| | 값 |
|---|---|
| `alpha` | **`2`** — 스크립트 기본값은 6.0. `final_score.sh` 가 고정 |

---

## 1. 환경

24GB GPU 4장 기준입니다. `run_*.sh` 는 SLURM 용이며, SLURM 이 없으면 스크립트 안의
`python` 호출부를 그대로 셸에서 실행하면 됩니다.

```bash
git clone https://github.com/chanel-ll/2026_deep_challenge.git
cd 2026_deep_challenge

conda create -n dcm -y --override-channels -c conda-forge python=3.10
conda activate dcm
pip install -r requirements.lock.txt
pip check                                  # → No broken requirements found.
```

확인:

```bash
python -c "import vllm, transformers, peft, deepspeed; print(vllm.__version__, transformers.__version__)"
# → 0.6.3.post1 4.48.3
```

`run_*.sh` 는 내부에서 `conda activate dcm` 을 합니다. 환경 이름이 다르면 각 스크립트
상단의 그 줄을 고치십시오.

## 2. 데이터 연결

스크립트는 `code/` 안에서 실행하며 데이터를 **같은 디렉토리에서 상대경로로** 찾습니다.
데이터는 `dataset/` 에 분리해 두었으므로 한 번만 링크를 겁니다.

```bash
cd code
ln -sf ../dataset/*.csv .        # 심볼릭 링크가 안 되면: cp ../dataset/*.csv .
chmod +x *.sh
```

| `dataset/` | 설명 |
|---|---|
| `deep_chal_math_train.csv` | 대회 제공 학습 데이터 |
| `deep_chal_math_leaderboard_filtered.csv` | 대회 제공 리더보드 문항 831개 |
| `train_filtered_ids.csv` | 학습셋에서 걸러낸 오류 문항 id (자체 검수) |
| `bad_ids_extra.csv` | 추가로 발견한 라벨 오류 14건 (자체 검수) |

---

## 3. 재현 순서

`code/` 안에서 순서대로 실행합니다.

### 3-0. 데이터 준비

```bash
python make_val.py --n_big 2000 --seed 42
```

```
my_val_big.csv          2,000    자체 평가셋
train_clean_noval.csv  14,373    오류 문항 + 평가셋 제외  ← 학습에는 이것만 쓴다
```

### 3-1. 검증기 학습 데이터 생성

`train_clean_noval` 전체에 base 로 문제당 16개를 뽑고 gold 대조로 정오 라벨을 붙입니다.

```bash
MODEL=Qwen/Qwen2.5-3B-Instruct PROB=train_clean_noval.csv \
OUT=sft_data/verifier_raw.jsonl \
N=16 TEMP=0.8 MAXTOK=1024 SEED=42 \
sbatch --gres=gpu:4 run_gen_verifier.sh
```

```bash
python check_cands.py sft_data/verifier_raw.jsonl | head -8
# 문제 14,361   후보수 n=16   gold 있음 14,361/14,361
```

### 3-2. 검증기 학습

```bash
DATA=sft_data/verifier_raw.jsonl OUT=./verifier_out/v1 \
EPOCHS=1 LR=1e-4 \
sbatch --gres=gpu:4 run_train_verifier.sh
```

`verifier_out/v1/merged` 가 생기면 완료입니다.

### 3-3. 후보 생성

```bash
TAG=lb PROB=deep_chal_math_leaderboard_filtered.csv ./final_gen.sh 201 <노드>
TAG=lb PROB=deep_chal_math_leaderboard_filtered.csv ./final_gen.sh 202 <노드>
TAG=lb PROB=deep_chal_math_leaderboard_filtered.csv ./final_gen.sh 203 <노드>
TAG=lb PROB=deep_chal_math_leaderboard_filtered.csv ./final_gen.sh 204 <노드>
```

`final_gen.sh` 가 `N=64 TEMP=0.8 MAXTOK=4096` 을 고정합니다.
**`run_gen_verifier.sh` 를 직접 던지지 마십시오 — 기본 `MAXTOK` 이 1024 입니다.**

병합:

```bash
python merge_cands.py --out sft_data/lb_cands_n256.jsonl sft_data/lb_cands_n64_s20[1-4].jsonl
python check_cands.py sft_data/lb_cands_n256.jsonl | head -6
# 문제 831   후보수 n=256:831문제
```

### 3-4. 채점

```bash
TAG=lb PROB=deep_chal_math_leaderboard_filtered.csv ./final_score.sh <노드>
```

`final_score.sh` 가 `ALPHA=2` 를 고정하고 GPU 4장 샤딩을 유지합니다.

- `run_verifier_eval.sh` 를 직접 던지면 기본 `ALPHA` 가 **6.0** 입니다.
- GPU 를 1장만 주면 샤딩이 꺼집니다 (511,488후보 기준 1장 7시간 20분 vs 4장 1시간 50분).
- `SUBMIT` / `SUBMIT_MAJ` 를 넘기지 마십시오 — 샤딩이 강제로 꺼집니다.

### 3-5. 제출 CSV

```bash
python resubmit_alpha.py --detail 'verif_eval/lb_n256/verif_detail_*.jsonl' \
    --alpha 2 --out_prefix submissions/lb_n256
```

```
행수 831 (기대 831)  OK
id   OK
```

→ **`submissions/lb_n256_a2.csv`** 가 제출 파일입니다.

---

## 4. 새 test set 에 적용

문제 파일만 바꾸면 동일합니다. `final_gen.sh` / `final_score.sh` 의 기본 `PROB` 이
`test.csv` 라 `TAG` · `PROB` 을 생략할 수 있습니다.

```bash
cd code
cp <받은_파일> test.csv

./final_gen.sh 201 <노드>
./final_gen.sh 202 <노드>
./final_gen.sh 203 <노드>
./final_gen.sh 204 <노드>

python merge_cands.py --out sft_data/test_cands_n256.jsonl \
    sft_data/test_cands_n64_s20[1-4].jsonl

./final_score.sh <노드>

python resubmit_alpha.py --detail 'verif_eval/test_n256/verif_detail_*.jsonl' \
    --alpha 2 --out_prefix submissions/test_final
```

```bash
python make_maj_submission.py --cands sft_data/test_cands_n256.jsonl \
    --out submissions/test_maj.csv
```

---

## 5. 저장소 구조

```
.
├── README.md
├── requirements.lock.txt      검증된 조합 (정확한 재현용)
├── requirements.txt           직접 의존성만
├── environment.yml
├── dataset/                   대회 제공 데이터 + 자체 검수 오류 목록
└── code/
    ├── make_val.py                평가셋 분리 · 오류 문항 제외
    ├── gen_verifier_data.py       vLLM 후보 생성 + gold 대조 라벨링
    ├── run_gen_verifier.sh        위의 SLURM 래퍼 (GPU 샤딩)
    ├── train_verifier.py          검증기 LoRA 학습 (Yes/No 이진 판정)
    ├── run_train_verifier.sh      위의 SLURM 래퍼 + LoRA 병합
    ├── ds_config_zero2.json       DeepSpeed ZeRO-2 설정
    ├── infer_verifier.py          후보별 P(Yes) 산출
    ├── run_verifier_eval.sh       위의 SLURM 래퍼 (GPU 샤딩)
    ├── merge_cands.py             n=64 여러 개 → n=256 병합
    ├── check_cands.py             후보 파일 무결성 · 통계 확인
    ├── resubmit_alpha.py          detail → 제출 CSV (α 변경은 CPU)
    ├── make_maj_submission.py     검증기 없는 폴백 (순수 다수결)
    ├── final_gen.sh          ★   생성 설정 고정 진입점 (mt4096)
    └── final_score.sh        ★   채점 설정 고정 진입점 (α=2, 4장 샤딩)
```

