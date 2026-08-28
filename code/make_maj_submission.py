#!/usr/bin/env python3
import argparse
import json
import os
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True, help="gen_verifier_data.py 산출 jsonl")
    ap.add_argument("--out", required=True, help="제출 CSV 경로")
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--ans_col", default="answer")
    args = ap.parse_args()

    rows, empty, total_c = [], 0, 0
    with open(args.cands, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            pid = str(d.get("id"))
            cands = d.get("cands") or []
            total_c += len(cands)

            votes = Counter()
            order = {}
            for i, c in enumerate(cands):
                a = c.get("answer")
                if a is None or a == "":
                    continue
                a = str(a)
                votes[a] += 1
                order.setdefault(a, i)

            if not votes:
                empty += 1
                rows.append((pid, "0"))
                continue

            best = min(votes.items(), key=lambda kv: (-kv[1], order[kv[0]]))[0]
            rows.append((pid, best))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        f.write(f"{args.id_col},{args.ans_col}\n")
        for pid, a in rows:
            f.write(f"{pid},{a}\n")

    n = len(rows)
    print(f"[out] {args.out}")
    print(f"  문제        : {n:,}")
    print(f"  후보 총계   : {total_c:,}  (문제당 평균 {total_c / max(n, 1):.1f})")
    print(f"  전원 파싱실패: {empty}  -> 0 으로 채움")
    print(f"  답이 0 인 문제: {sum(1 for _, a in rows if a == '0')}  "
          f"(0 은 정당한 정답이다)")
    digits = max((len(a.lstrip('-')) for _, a in rows), default=0)
    print(f"  최대 자릿수 : {digits}  (대회 최대 16)")


if __name__ == "__main__":
    main()
