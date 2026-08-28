#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
from collections import Counter

import pandas as pd

_d = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("bi", os.path.join(_d, "baseline_infer.py"))
bi = importlib.util.module_from_spec(_s); _s.loader.exec_module(bi)


def load_problems(path, exclude_files, shard, num_shards):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    idcol = df.columns[0]
    before = len(df)
    for ef in exclude_files or []:
        if not os.path.exists(ef):
            print(f"[warn] 제외파일 없음: {ef}")
            continue
        ex = pd.read_csv(ef)
        ex.columns = [c.strip() for c in ex.columns]
        bad = set(ex[ex.columns[0]].astype(str).str.strip())
        df = df[~df[idcol].astype(str).str.strip().isin(bad)]
        print(f"[제외] {ef}: {before - len(df)}개 누적 제거")
    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        if i % num_shards != shard:
            continue
        if "answer" in df.columns:
            g = bi._clean_int(str(r["answer"]))
            if g is None:
                continue
        else:
            g = None
        rows.append((str(r[idcol]).strip(), str(r["question"]), g))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="train_clean_noval.csv")
    ap.add_argument("--out", default="sft_data/verifier_raw.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--min_p", type=float, default=0.0)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=8, help="문제당 후보 수")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--exclude_ids", action="append",
                    default=None, help="여러 번 지정 가능")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    ap.add_argument("--max_num_seqs", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help=">0 이면 앞 N문제만 (스모크)")
    ap.add_argument("--seed", type=int, default=0,
                    help="샘플링 시드. 큰 n 을 여러 job 으로 쪼갤 때 반드시 다르게 줄 것")
    args = ap.parse_args()

    excl = args.exclude_ids
    if excl is None:
        excl = ["bad_ids_extra.csv"]

    probs = load_problems(args.problems, excl, args.shard, args.num_shards)
    if args.limit:
        probs = probs[:args.limit]
    print(f"[shard {args.shard}/{args.num_shards}] 문제 {len(probs):,}개  n={args.n}  "
          f"temp={args.temperature}  → 후보 {len(probs)*args.n:,}개")

    if args.n > args.max_num_seqs:
        raise SystemExit(
            f"[ERROR] n({args.n}) > max_num_seqs({args.max_num_seqs}) — vLLM 이 멈춘다.\n"
            f"        해결 1: --max_num_seqs {args.n} 이상 (메모리 여유 필요)\n"
            f"        해결 2: n 을 쪼개서 --seed 를 다르게 준 뒤 merge_cands.py 로 합친다\n"
            f"                (권장. 8시간 제한에 걸려 전부 잃는 위험도 같이 없앤다)")

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem, max_num_seqs=args.max_num_seqs,
              seed=args.seed, trust_remote_code=True)
    tok = llm.get_tokenizer()

    prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": bi.SYSTEM_PROMPT},
             {"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True)
        for _, q, _ in probs
    ]
    plen = max(len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts)
    bi.kv_clamp(llm, plen, args.max_tokens, args.n)

    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                        min_p=args.min_p,
                        repetition_penalty=args.repetition_penalty,
                        max_tokens=args.max_tokens)
    print(f"[sampling] temp={args.temperature} top_p={args.top_p} "
          f"min_p={args.min_p} rep_penalty={args.repetition_penalty} "
          f"max_tokens={args.max_tokens}")

    outs = llm.generate(prompts, sp)

    base = args.out.replace(".jsonl", f".shard{args.shard}.jsonl")
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)

    stat = Counter()
    n_c = n_w = 0
    with open(base, "w", encoding="utf-8") as f:
        for i, o in enumerate(outs):
            pid, q, gold = probs[i]
            cands = []
            for c in o.outputs:
                t = c.text.strip()
                a = bi.extract_answer(t)
                cands.append({"text": t, "answer": a,
                              "correct": None if gold is None else bool(a == gold)})
            nc = sum(bool(c["correct"]) for c in cands)
            nw = len(cands) - nc
            n_c += nc; n_w += nw
            stat["all_correct" if nw == 0 else "all_wrong" if nc == 0 else "mixed"] += 1
            f.write(json.dumps({"id": pid, "question": q, "gold": gold,
                                "n_correct": nc, "n_wrong": nw,
                                "cands": cands}, ensure_ascii=False) + "\n")

    tot = max(n_c + n_w, 1)
    print(f"\n[shard {args.shard}] 저장 → {base}")
    if all(g is None for _, _, g in probs):
        print("  ⚠️ gold 없음 (리더보드/테스트 후보) — 정오 통계는 의미 없다. 아래를 읽지 말 것.")
        print(f"  후보      : {tot:,}   문제 {len(probs):,}")
        print("  검증할 것 : 문제 수와 문제당 후보 수. 정답률·혼합비율이 아니다.")
        print("              여러 시드를 합칠 때 merge_cands.py 가 독립성까지 검사한다.")
    else:
        print(f"  후보      : {tot:,}  (정답 {n_c:,} / 오답 {n_w:,} = {n_c/tot*100:.1f}% 정답률)")
        print(f"  ★혼합문제 : {stat['mixed']:,}  ← verifier 학습에 실제로 쓰이는 것")
        print(f"   전원정답 : {stat['all_correct']:,}  (음성 예시 없음)")
        print(f"   전원오답 : {stat['all_wrong']:,}  (양성 예시 없음)")


if __name__ == "__main__":
    main()
