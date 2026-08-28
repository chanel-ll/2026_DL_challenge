#!/usr/bin/env python3
import argparse
import json
import os
import random

import torch


SYS = ("You are a strict mathematics grader. You will be shown a problem and a "
       "proposed solution. Decide whether the solution's final answer is correct. "
       "Reply with exactly one word: Yes or No.")


def build_examples(path, max_pairs_per_problem, unmixed_frac, seed, hard_boost=2):
    from collections import Counter
    rng = random.Random(seed)
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    ex = []
    st = Counter()

    for r in rows:
        cor = [c for c in r["cands"] if c["correct"]]
        wrg = [c for c in r["cands"] if not c["correct"]]

        if not cor:
            st["drop_all_wrong"] += 1
            continue

        if not wrg:
            if rng.random() > unmixed_frac:
                st["drop_all_correct"] += 1
                continue
            st["all_correct"] += 1
            ex.append({"question": r["question"], "solution": rng.choice(cor)["text"],
                       "label": "Yes", "hard": 0})
            continue

        cnt = Counter(c["answer"] for c in r["cands"] if c["answer"] is not None)
        modal = cnt.most_common(1)[0][0] if cnt else None
        maj_fails = (modal != r["gold"])
        st["mixed_majfail" if maj_fails else "mixed_majok"] += 1

        k = min(len(cor), len(wrg), max_pairs_per_problem)
        if maj_fails:
            k = min(len(cor), len(wrg), max_pairs_per_problem * hard_boost)

        wrg_sorted = sorted(wrg, key=lambda c: -cnt.get(c["answer"], 0))
        for c in rng.sample(cor, k):
            ex.append({"question": r["question"], "solution": c["text"],
                       "label": "Yes", "hard": int(maj_fails)})
        for c in wrg_sorted[:k]:
            ex.append({"question": r["question"], "solution": c["text"],
                       "label": "No", "hard": int(maj_fails)})

    rng.shuffle(ex)
    pos = sum(1 for e in ex if e["label"] == "Yes")
    hard = sum(e["hard"] for e in ex)
    st.update({"total": len(ex), "pos": pos, "neg": len(ex) - pos, "hard": hard})
    return ex, st


class VerifDataset(torch.utils.data.Dataset):

    def __init__(self, ex, tok, max_len):
        self.ex, self.tok, self.max_len = ex, tok, max_len

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        e = self.ex[i]
        user = (f"Problem:\n{e['question']}\n\n"
                f"Proposed solution:\n{e['solution']}\n\n"
                f"Is the final answer correct?")
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": user}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True)
        full = prompt + e["label"] + self.tok.eos_token

        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = self.tok(full, add_special_tokens=False)["input_ids"]

        if len(f_ids) > self.max_len:
            cut = len(f_ids) - self.max_len
            f_ids = f_ids[cut:]
            p_ids = p_ids[cut:] if len(p_ids) > cut else []

        labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
        return {"input_ids": f_ids, "labels": labels,
                "attention_mask": [1] * len(f_ids)}


def collate(batch, pad_id):
    m = max(len(b["input_ids"]) for b in batch)
    out = {k: [] for k in ("input_ids", "labels", "attention_mask")}
    for b in batch:
        p = m - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * p)
        out["labels"].append(b["labels"] + [-100] * p)
        out["attention_mask"].append(b["attention_mask"] + [0] * p)
    return {k: torch.tensor(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sft_data/verifier_raw.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--out_dir", default="./verifier_out/v1")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_len", type=int, default=1536)
    ap.add_argument("--per_device_batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_pairs", type=int, default=3,
                    help="혼합 문제당 (정답,오답) 쌍 수")
    ap.add_argument("--unmixed_frac", type=float, default=0.15,
                    help="전원 정답 문제를 섞는 비율. 점수 보정용 (전원 오답은 항상 제외)")
    ap.add_argument("--hard_boost", type=int, default=2,
                    help="다수결이 지는 문제에서 몇 배로 뽑을지. 여기가 선택 갭의 실체다")
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deepspeed", default=None)
    args = ap.parse_args()

    is_main = int(os.environ.get("RANK", 0)) == 0
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)

    ex, st = build_examples(args.data, args.max_pairs, args.unmixed_frac,
                            args.seed, args.hard_boost)
    if is_main:
        n_all = st['total']
        print(f"[data] 문제 선별")
        print(f"  ★다수결 패배 : {st['mixed_majfail']:,}  ← verifier 가 뒤집어야 할 표본")
        print(f"   다수결 승리 : {st['mixed_majok']:,}   (다수결이 이미 맞힘)")
        print(f"   전원 정답   : {st['all_correct']:,} 채택 / {st['drop_all_correct']:,} 제외")
        print(f"   전원 오답   : {st['drop_all_wrong']:,} **전량 제외** (라벨 오류 농축 위험)")
        print(f"[data] 학습 예시 {n_all:,}개  "
              f"(Yes {st['pos']:,} / No {st['neg']:,} = {st['pos']/max(n_all,1)*100:.1f}% 양성)")
        print(f"[data] 그중 다수결 패배 문제에서 온 것 {st['hard']:,} "
              f"({st['hard']/max(n_all,1)*100:.1f}%)")
        if n_all < 5000:
            print("[WARN] 예시가 5,000개 미만이다. 데이터 생성 시 n 을 올릴 것")
        if st['pos'] / max(n_all, 1) > 0.65 or st['pos'] / max(n_all, 1) < 0.35:
            print("[WARN] 클래스가 치우쳤다. unmixed_frac 을 낮출 것")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    if is_main:
        tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[lora] 학습 파라미터 {tr/1e6:.1f}M")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.per_device_batch,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=10,
            save_strategy="epoch",
            report_to=[],
            group_by_length=False,
            deepspeed=args.deepspeed,
            seed=args.seed,
        ),
        train_dataset=VerifDataset(ex, tok, args.max_len),
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )
    trainer.train()

    if is_main:
        merged = trainer.accelerator.unwrap_model(trainer.model).merge_and_unload()
        out = os.path.join(args.out_dir, "merged")
        merged.save_pretrained(out, safe_serialization=True)
        tok.save_pretrained(out)
        print(f"\n[merge] {out}")
        print("[다음] 평가셋 후보를 텍스트째 새로 뽑은 뒤 재순위화한다:")
        print("         sbatch -w ariel-v9 run_verifier_eval.sh")
        print("       (1단계 gen_verifier_data.py --problems my_val_big.csv --n 16")
        print("        2단계 infer_verifier.py --cands sft_data/val_cands_n16.jsonl)")


if __name__ == "__main__":
    main()
