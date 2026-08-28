#!/usr/bin/env python3
import argparse
import json
import os
from collections import Counter, defaultdict

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

SYS = ("You are a strict mathematics grader. You will be shown a problem and a "
       "proposed solution. Judge whether the final answer is correct. "
       "Reply with exactly one word: Yes or No.")


def build_prompt_ids(tok, question, solution, max_len):
    user = (f"Problem:\n{question}\n\n"
            f"Proposed solution:\n{solution}\n\n"
            f"Is the final answer correct?")
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    if len(ids) > max_len:
        ids = ids[len(ids) - max_len:]
    return ids


def first_token_id(tok, word):
    ids = tok(word, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"'{word}' 토크나이즈 실패")
    return ids[0]


def vote(cands, weights=None):
    score = defaultdict(float)
    for i, c in enumerate(cands):
        a = c["answer"]
        if a is None:
            continue
        score[a] += 1.0 if weights is None else weights[i]
    if not score:
        return None
    return max(score.items(), key=lambda kv: kv[1])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier", required=True, help="merge 된 verifier 경로")
    ap.add_argument("--cands", required=True, help="gen_verifier_data.py 산출 jsonl")
    ap.add_argument("--out_dir", default="./verif_eval")
    ap.add_argument("--max_len", type=int, default=1536,
                    help="학습 때와 같은 값. 초과분은 앞에서 자른다")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="가중치 지수. w = P(Yes)^alpha. >1 이면 확신한 후보에 더 쏠린다")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    ap.add_argument("--max_num_seqs", type=int, default=256)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help=">0 이면 앞 N문제만")
    ap.add_argument("--submit_csv", default=None,
                    help="지정하면 id,answer 제출 파일을 쓴다. gold 가 없는 리더보드용")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.num_shards > 1 and args.submit_csv:
        raise SystemExit(
            "[ERROR] 샤딩 모드에서는 --submit_csv 를 못 쓴다 (제출 CSV 를 합칠 수 없다).\n"
            "        채점만 하고 resubmit_alpha.py 로 제출 파일을 만들 것.")
    rows = [json.loads(l) for l in open(args.cands, encoding="utf-8") if l.strip()]
    if args.limit > 0:
        rows = rows[:args.limit]
    if args.num_shards > 1:
        rows = rows[args.shard::args.num_shards]
    n_c = sum(len(r["cands"]) for r in rows)
    print(f"[load] 문제 {len(rows):,}  후보 {n_c:,}"
          + (f"   (샤드 {args.shard}/{args.num_shards})" if args.num_shards > 1 else ""))

    tok = AutoTokenizer.from_pretrained(args.verifier)
    yes_id, no_id = first_token_id(tok, "Yes"), first_token_id(tok, "No")
    print(f"[tok] Yes={yes_id}  No={no_id}")

    flat, owner = [], []
    for ri, r in enumerate(rows):
        for c in r["cands"]:
            flat.append(build_prompt_ids(tok, r["question"], c["text"], args.max_len))
            owner.append(ri)

    llm = LLM(model=args.verifier, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_len + 8,
              dtype="bfloat16", max_num_seqs=args.max_num_seqs)
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    outs = llm.generate([{"prompt_token_ids": ids} for ids in flat], sp)
    assert len(outs) == len(flat), f"출력 수 불일치: {len(outs)} vs {len(flat)}"

    import math
    scores, n_missing = [], 0
    for o in outs:
        lp = o.outputs[0].logprobs[0]
        ly = lp[yes_id].logprob if yes_id in lp else None
        ln = lp[no_id].logprob if no_id in lp else None
        if ly is None and ln is None:
            n_missing += 1
            scores.append(0.5)
        elif ly is None:
            scores.append(0.0)
        elif ln is None:
            scores.append(1.0)
        else:
            m = max(ly, ln)
            ey, en = math.exp(ly - m), math.exp(ln - m)
            scores.append(ey / (ey + en))
    if n_missing:
        print(f"[warn] Yes/No 가 상위 20 안에 없던 후보 {n_missing:,}개 → 0.5 처리")

    per = defaultdict(list)
    for k, ri in enumerate(owner):
        per[ri].append(scores[k])

    st = Counter()
    submit = []
    cal_c, cal_w = [], []
    detail = []
    for ri, r in enumerate(rows):
        cands, gold = r["cands"], r["gold"]
        w = [s ** args.alpha for s in per[ri]]

        v_maj = vote(cands)
        v_wmaj = vote(cands, w)
        submit.append((r["id"], v_wmaj if v_wmaj is not None else v_maj))
        best = max(range(len(cands)), key=lambda i: per[ri][i])
        v_bon = cands[best]["answer"]
        hit = any(c["answer"] == gold for c in cands)

        if gold is None:
            st["n"] += 1
            detail.append({"id": r["id"], "gold": None, "maj": v_maj,
                           "wmaj": v_wmaj, "bon": v_bon, "hit": None,
                           "scores": [round(s2, 4) for s2 in per[ri]],
                           "answers": [c["answer"] for c in cands]})
            continue
        st["n"] += 1
        st["maj"] += (v_maj == gold)
        st["wmaj"] += (v_wmaj == gold)
        st["bon"] += (v_bon == gold)
        st["pass"] += hit
        if v_maj != gold and v_wmaj == gold:
            st["flip_win"] += 1
        if v_maj == gold and v_wmaj != gold:
            st["flip_lose"] += 1

        for c, s in zip(cands, per[ri]):
            (cal_c if c["answer"] == gold else cal_w).append(s)

        detail.append({"id": r["id"], "gold": gold, "maj": v_maj,
                       "wmaj": v_wmaj, "bon": v_bon, "hit": hit,
                       "scores": [round(s, 4) for s in per[ri]],
                       "answers": [c["answer"] for c in cands]})

    stem = os.path.splitext(os.path.basename(args.cands))[0]
    vp = os.path.normpath(args.verifier).replace("\\", "/").rstrip("/").split("/")
    vtag = vp[-2] if vp[-1] in ("merged", "final", "") and len(vp) >= 2 else vp[-1]
    if args.num_shards > 1:
        detail_path = os.path.join(args.out_dir,
                                   f"verif_shard{args.shard}_{stem}__{vtag}.jsonl")
    else:
        detail_path = os.path.join(args.out_dir, f"verif_detail_{stem}__{vtag}.jsonl")
    with open(detail_path, "w", encoding="utf-8") as f:
        for d in detail:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    if args.num_shards == 1 and any(d["gold"] is not None for d in detail):
        with open(os.path.join(args.out_dir, "verif_detail.jsonl"), "w", encoding="utf-8") as f:
            for d in detail:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"[out] {detail_path}  ({len(detail):,}행)")

    if args.submit_csv:
        import csv
        os.makedirs(os.path.dirname(args.submit_csv) or ".", exist_ok=True)
        with open(args.submit_csv, "w", encoding="utf-8", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["id", "answer"])
            for pid, a in submit:
                wcsv.writerow([pid, a if a is not None else 0])
        nnull = sum(1 for _, a in submit if a is None)
        print(f"\n제출 파일: {args.submit_csv}  "
              f"({len(submit):,}행, 미결정 {nnull} → 0 처리)")

    n = st["n"]
    if st["pass"] == 0 and st["maj"] == 0:
        print("[info] gold 가 없어 채점을 건너뛰었다 (리더보드 모드)")
        return
    p = lambda k: st[k] / n * 100
    print("\n" + "=" * 66)
    print(f"Verifier 재순위화 — {n:,}문제, 후보 {n_c // max(n,1)}개/문제, alpha={args.alpha}")
    print("=" * 66)
    print(f"  maj  (평범한 다수결)   {p('maj'):6.2f}%   ← 넘어야 할 기준선")
    print(f"  wmaj (가중 다수결)     {p('wmaj'):6.2f}%   {p('wmaj') - p('maj'):+.2f}%p")
    print(f"  bon  (Best-of-N)       {p('bon'):6.2f}%   {p('bon') - p('maj'):+.2f}%p")
    print(f"  pass (천장)            {p('pass'):6.2f}%")
    gap = p("pass") - p("maj")
    got = p("wmaj") - p("maj")
    print(f"\n  갭 {gap:.2f}%p 중 {got:+.2f}%p 회수  "
          f"({got / gap * 100 if gap else 0:.1f}%)")
    print(f"\n  뒤집어 맞힘 {st['flip_win']:,}  /  망친 것 {st['flip_lose']:,}"
          f"   순증 {st['flip_win'] - st['flip_lose']:+,}")
    if cal_c and cal_w:
        mc, mw = sum(cal_c) / len(cal_c), sum(cal_w) / len(cal_w)
        print(f"\n  [보정] 정답 후보 평균 P(Yes) {mc:.3f}  /  오답 후보 {mw:.3f}"
              f"   분리도 {mc - mw:+.3f}")
        print("        분리도가 0.1 미만이면 verifier 가 사실상 아무것도 구분 못 한다")
    print("=" * 66)
    print("판단:")
    print("  · wmaj > maj 면 채택. 아니면 다수결로 되돌린다 (손실 0)")
    print("  · bon > wmaj 면 가중합보다 단일 선택이 낫다 — alpha 를 올려볼 것")
    print("  · flip_lose 가 flip_win 과 비슷하면 verifier 가 노이즈다")
    print(f"저장: {detail_path}")
    print("  → alpha 는 이 파일만으로 CPU 에서 바꿔볼 수 있다 (GPU 재채점 불필요):")
    print(f"      python alpha_diff.py --detail {detail_path} --ref 4.0")


if __name__ == "__main__":
    main()
