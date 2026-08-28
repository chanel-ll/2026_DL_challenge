#!/usr/bin/env python3

import argparse
import os
import re
import json
from collections import Counter

import pandas as pd


SYSTEM_PROMPT = (
    "You are a careful mathematical problem solver. "
    "Solve the problem step by step. "
    "All answers are integers. "
    "Put ONLY the final integer answer inside \\boxed{}. "
    "For example, if the answer is 42, end your solution with \\boxed{42}."
)

def build_messages(question: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def kv_clamp(llm, plen, max_tokens, n, verbose=True):
    try:
        eng = llm.llm_engine
        cc, sc = eng.cache_config, eng.scheduler_config
        nblk, blk = cc.num_gpu_blocks, cc.block_size
    except AttributeError:
        print("[warn] KV 가드 건너뜀 (vLLM 내부 구조가 다르다)")
        return None

    need = -(-(plen + max_tokens) // blk)
    safe = nblk // need
    cur = sc.max_num_seqs
    if verbose:
        print(f"[kv] {nblk * blk:,}토큰 / 시퀀스당 {need:,}블록 → 무선점 시퀀스 {safe}개"
              f"  (n={n}, 현재 max_num_seqs={cur})")
    if n > safe:
        raise SystemExit(
            f"[ERROR] n={n} 하나도 KV 에 안 들어간다 (용량 {safe}).\n"
            f"        그룹 하나는 통째로 살아 있어야 하므로 이건 낮출 수가 없다.\n"
            f"        → n 을 {safe} 이하로 낮추거나, --max_tokens 를 줄이거나, --tp 를 올려라.")

    tgt = max(n, (safe // n) * n)
    if cur > tgt:
        sc.max_num_seqs = tgt
        for s in getattr(eng, "scheduler", []):
            try:
                s.scheduler_config.max_num_seqs = tgt
            except AttributeError:
                pass
        if verbose:
            print(f"[kv] max_num_seqs {cur} → {tgt} 로 낮춘다 "
                  f"(그룹 {tgt // n}개 동시). 안 낮추면 선점되고, 선점되면 죽는다")
    return tgt


def _clean_int(s: str):
    if s is None:
        return None
    s = s.strip()
    s = s.replace(",", "").replace(" ", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("$", "")
    s = s.replace("\\text", "").replace("{", "").replace("}", "")

    if re.fullmatch(r"[+-]?\d+", s):
        try:
            return int(s)
        except ValueError:
            return None

    m = re.fullmatch(r"([+-]?\d+)\^([+-]?\d+)", s)
    if m:
        base, exp = int(m.group(1)), int(m.group(2))
        if 0 <= exp <= 64 and abs(base) <= 10**6:
            try:
                val = base ** exp
            except (OverflowError, ValueError):
                return None
            if len(str(abs(val))) <= 20:
                return val

    m = re.search(r"-?\d+", s)
    if not m:
        return None
    try:
        return int(m.group())
    except ValueError:
        return None


def extract_answer(text: str):
    if not text:
        return None

    boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if boxed:
        val = _clean_int(boxed[-1])
        if val is not None:
            return val

    m = re.findall(r"(?:answer|answer is|정답은|정답|final answer)\s*[:=]?\s*(-?\d[\d,]*)",
                   text, flags=re.IGNORECASE)
    if m:
        val = _clean_int(m[-1])
        if val is not None:
            return val

    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        val = _clean_int(lines[-1])
        if val is not None:
            return val

    nums = re.findall(r"-?\d[\d,]*", text)
    if nums:
        return _clean_int(nums[-1])

    return None


DEPRECATED_INPUTS = {
    "deep_chal_math_leaderboard.csv":
        "deep_chal_math_leaderboard_filtered.csv  (8/3 공지: 1,000 → 831문항, 기존 파일 무효)",
    "deep_chal_math_train.csv":
        "train_clean.csv 또는 train_clean_noval.csv  (오류 627개 제외, make_val.py 로 생성)",
    "my_val.csv":
        "my_val_filtered.csv(486) 또는 my_val_big.csv(2000)  (오류 문항 14개 포함되어 있음)",
}


def check_input_path(path, allow_raw=False):
    name = os.path.basename(str(path))
    if name in DEPRECATED_INPUTS and not allow_raw:
        raise SystemExit(
            f"\n[중단] '{name}' 은 더 이상 쓰면 안 된다.\n"
            f"       → 대신: {DEPRECATED_INPUTS[name]}\n"
            f"       의도적으로 써야 하면 --allow_raw 를 붙여라.\n")


def majority_vote(answers):
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="입력 csv (id, question[, answer])")
    ap.add_argument("--out_dir", default="./baseline_out")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--mode", choices=["greedy", "sc", "both"], default="both")
    ap.add_argument("--n_sc", type=int, default=16, help="self-consistency 샘플 수")
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.8, help="SC 샘플링 온도")
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size (GPU 수)")
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    ap.add_argument("--has_answer", action="store_true",
                    help="입력에 정답이 채워져 있으면 정확도까지 계산")
    ap.add_argument("--limit", type=int, default=0, help="디버그용: 앞 N문제만 (0=전체)")
    ap.add_argument("--allow_raw", action="store_true",
                    help="폐기된 원본 csv 사용 강행 (8/3 공지 이전 재현 등)")
    args = ap.parse_args()

    check_input_path(args.input, args.allow_raw)
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    df.columns = [c.strip() for c in df.columns]
    if args.limit > 0:
        df = df.head(args.limit).copy()
    questions = df["question"].astype(str).tolist()
    ids = df["id"].astype(str).tolist()

    gold = None
    if args.has_answer and "answer" in df.columns and df["answer"].notna().any():
        gold = [_clean_int(str(a)) for a in df["answer"].tolist()]

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_tokens + 2048,
        dtype="bfloat16",
    )

    prompts = []
    for q in questions:
        msgs = build_messages(q)
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(p)

    plen = max(len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts)
    kv_clamp(llm, plen, args.max_tokens,
             args.n_sc if args.mode in ("sc", "both") else 1)

    results = {}

    if args.mode in ("greedy", "both"):
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
        outs = llm.generate(prompts, sp)
        assert len(outs) == len(prompts), f"출력 수 불일치: {len(outs)} vs {len(prompts)}"
        greedy_ans = []
        greedy_raw = []
        for o in outs:
            text = o.outputs[0].text
            greedy_raw.append(text)
            greedy_ans.append(extract_answer(text))
        results["greedy"] = greedy_ans
        with open(os.path.join(args.out_dir, "greedy_raw.jsonl"), "w", encoding="utf-8") as f:
            for i, (id_, text, a) in enumerate(zip(ids, greedy_raw, greedy_ans)):
                f.write(json.dumps({"id": id_, "answer": a, "raw": text}, ensure_ascii=False) + "\n")

    if args.mode in ("sc", "both"):

        sp = SamplingParams(
            temperature=args.temperature, top_p=args.top_p,
            max_tokens=args.max_tokens, n=args.n_sc,
        )
        outs = llm.generate(prompts, sp)
        sc_ans = []
        sc_detail = []
        for o in outs:
            cand = [extract_answer(c.text) for c in o.outputs]
            voted = majority_vote(cand)
            sc_ans.append(voted)
            sc_detail.append(cand)
        results["sc"] = sc_ans
        with open(os.path.join(args.out_dir, "sc_detail.jsonl"), "w", encoding="utf-8") as f:
            for id_, cand, v in zip(ids, sc_detail, sc_ans):
                f.write(json.dumps({"id": id_, "voted": v, "candidates": cand}, ensure_ascii=False) + "\n")

    def report(name, ans_list):
        n = len(ans_list)
        parsed = sum(1 for a in ans_list if a is not None)
        line = [f"[{name}]",
                f"  총 문제수      : {n}",
                f"  파싱 성공      : {parsed} ({parsed/n*100:.1f}%)",
                f"  파싱 실패(None): {n-parsed}"]
        if gold is not None:
            correct = sum(1 for a, g in zip(ans_list, gold) if a is not None and a == g)
            line.append(f"  정확도(EM)     : {correct}/{n} = {correct/n*100:.2f}%")
        return "\n".join(line)

    report_lines = ["=" * 50, "BASELINE 진단 리포트", "=" * 50]
    for name in ["greedy", "sc"]:
        if name in results:
            report_lines.append(report("GREEDY" if name == "greedy" else f"SELF-CONSISTENCY(n={args.n_sc})",
                                        results[name]))
            report_lines.append("")
    report_txt = "\n".join(report_lines)
    print(report_txt)
    with open(os.path.join(args.out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt)

    final_mode = "sc" if "sc" in results else "greedy"
    def save_submission(id_list, ans_list, path):
        answers = [str(a if a is not None else 0) for a in ans_list]
        df_out = pd.DataFrame({"id": id_list, "answer": answers})
        df_out.to_csv(path, index=False)
        return path

    final_ans = results[final_mode]
    sub_path = os.path.join(args.out_dir, f"submission_{final_mode}.csv")
    save_submission(ids, final_ans, sub_path)
    print(f"\n제출 파일 저장: {sub_path}  (final mode = {final_mode})")

    if "greedy" in results:
        save_submission(ids, results["greedy"],
                        os.path.join(args.out_dir, "submission_greedy.csv"))


if __name__ == "__main__":
    main()
