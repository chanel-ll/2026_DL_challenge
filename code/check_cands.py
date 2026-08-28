#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter

args = None

CSV_CANDIDATES = [
    ("my_val_big.csv", "my_val_big (자체 val 2,000)"),
    ("my_val.csv", "my_val (자체 val 500)"),
    ("deep_chal_math_leaderboard_filtered.csv", "리더보드 831 (필터본)"),
    ("deep_chal_math_leaderboard.csv", "리더보드 (원본)"),
    ("deep_chal_math_train.csv", "train 전체"),
]


def load_id_sets():
    out = []
    for path, label in CSV_CANDIDATES:
        if not os.path.exists(path):
            continue
        ids = set()
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for key in ("id", "ID", "problem_id", "index"):
                    if key in row and row[key] not in (None, ""):
                        ids.add(str(row[key]))
                        break
        if ids:
            out.append((label, ids))
    return out


def scan(path, id_sets):
    n_row = 0
    n_scored = 0
    ids = set()
    dup = 0
    n_cands = Counter()
    has_gold = 0
    has_correct = 0
    pass_hit = 0
    maj_hit = 0
    sel_gap_ids = []
    cap_fail_ids = []
    bad = 0
    stale = 0
    uniq_n = []
    modal_share = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                bad += 1
                continue
            n_row += 1
            pid = str(d.get("id"))
            if pid in ids:
                dup += 1
            ids.add(pid)

            cands = d.get("cands") or []
            n_cands[len(cands)] += 1

            gold = d.get("gold")
            if gold not in (None, ""):
                has_gold += 1
            labeled = any("correct" in c for c in cands)
            if labeled:
                has_correct += 1
            if gold in (None, "") or not labeled:
                continue
            n_scored += 1

            nc = sum(1 for c in cands if c.get("correct"))
            if d.get("n_correct") is not None and d["n_correct"] != nc:
                stale += 1
            ok = nc > 0
            pass_hit += ok

            votes = Counter(str(c.get("answer")) for c in cands
                            if c.get("answer") not in (None, ""))
            picked = votes.most_common(1)[0][0] if votes else None
            hit = picked is not None and picked == str(gold)
            maj_hit += hit
            if votes:
                uniq_n.append(len(votes))
                modal_share.append(votes.most_common(1)[0][1] / max(len(cands), 1))

            if ok and not hit:
                sel_gap_ids.append(pid)
            if not ok:
                cap_fail_ids.append(pid)

    return dict(path=path, n_row=n_row, n_scored=n_scored, ids=ids, dup=dup, n_cands=n_cands,
                has_gold=has_gold, has_correct=has_correct, bad=bad, stale=stale,
                pass_hit=pass_hit, maj_hit=maj_hit,
                sel_gap=sel_gap_ids, cap_fail=cap_fail_ids,
                uniq_n=uniq_n, modal_share=modal_share)


def report(r, id_sets):
    n = r["n_row"]
    print("─" * 70)
    size_mb = os.path.getsize(r["path"]) / 1e6
    print(f"{r['path']}   ({size_mb:,.1f} MB)")
    if n == 0:
        print("  ⚠️ 후보 레코드 0 — 이 파일은 후보 jsonl 이 아니다")
        return
    ncs = sorted(r["n_cands"].items())
    nstr = ", ".join(f"n={k}:{v}문제" for k, v in ncs[:6])
    if len(ncs) > 6:
        nstr += f", … ({len(ncs)}종)"
    print(f"  문제 {n:,}개  (중복 id {r['dup']})   후보수 {nstr}")
    if r["bad"]:
        print(f"  ⚠️ 파싱 실패 줄 {r['bad']}")

    tag = "정체 불명"
    for label, s in id_sets:
        inter = len(r["ids"] & s)
        if inter:
            cov_f = inter / len(r["ids"]) * 100
            cov_s = inter / len(s) * 100
            tag = f"{label}  겹침 {inter:,} (파일의 {cov_f:.0f}%, 셋의 {cov_s:.0f}%)"
            if cov_f > 95:
                break
    print(f"  문제셋: {tag}")

    print(f"  gold 있음 {r['has_gold']:,}/{n:,}   correct 라벨 {r['has_correct']:,}/{n:,}")
    if r["has_gold"] == 0:
        print("  → gold 가 없다. 리더보드 test 후보로 보인다. **1단계에 못 쓴다**")
        return
    if r["has_correct"] == 0:
        print("  → correct 라벨이 없다. 채점을 다시 해야 한다")
        return

    ns = r["n_scored"]
    if ns == 0:
        print("  → 채점 가능한 문제 0")
        return
    if ns != n:
        print(f"  ⚠️ gold/라벨이 있는 {ns:,}문제만 분모로 쓴다 (전체 {n:,})")
    p = r["pass_hit"] / ns * 100
    m = r["maj_hit"] / ns * 100
    nmax = max(r["n_cands"]) if r["n_cands"] else 0
    print()
    print(f"  ★ pass@{nmax}   {r['pass_hit']:,}/{ns:,} = {p:6.2f}%   ← 진짜 능력 경계")
    print(f"  ★ maj@{nmax}    {r['maj_hit']:,}/{ns:,} = {m:6.2f}%   (verifier 없이 순수 다수결)")
    print(f"  ★ 선택 갭      {p - m:6.2f}%p = {len(r['sel_gap']):,}문제"
          "   ← 후보에 정답이 있는데 못 고른 것")
    print(f"  ★ 능력 실패    {len(r['cap_fail']):,}문제"
          "   ← 후보 어디에도 정답이 없는 것")
    if r["stale"]:
        print(f"  ⚠️ 저장된 n_correct 가 실제와 다른 레코드 {r['stale']:,}개 — 다시 세서 쓴다.")
        print("     (구버전 merge_cands.py 로 합친 파일이다. 위 숫자는 재계산본이라 맞다)")

    u, ms = sorted(r["uniq_n"]), sorted(r["modal_share"])
    if u:
        print(f"  · 다양성       유니크 답 중앙 {u[len(u)//2]}종 / {nmax}"
              f"   최빈답 점유 중앙 {ms[len(ms)//2]*100:.1f}%")
        print("                 ← 학습 후 이 값이 떨어지면 샤프닝. base 와만 비교할 것")

    if not args.failset:
        return

    base = os.path.splitext(os.path.basename(r["path"]))[0].replace(".", "_")
    for name, lst in (("selgap", r["sel_gap"]), ("capfail", r["cap_fail"])):
        out = f"failset_{base}_{name}.txt"
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(lst) + "\n")
        print(f"    → {out}  ({len(lst):,} id)")

    if not os.path.exists(args.src_csv):
        print(f"    ⚠️ --src_csv 없음: {args.src_csv} — csv 는 안 쓴다")
        return
    fail = set(r["sel_gap"]) | set(r["cap_fail"])
    rest = r["ids"] - fail
    with open(args.src_csv, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        fields = rd.fieldnames
        idkey = next((k for k in ("id", "ID", "problem_id", "index") if k in fields), None)
        if idkey is None:
            print(f"    ⚠️ {args.src_csv} 에 id 컬럼이 없다 — csv 는 안 쓴다")
            return
        src = list(rd)
    for tag, want in (("fail", fail), ("rest", rest)):
        rows = [x for x in src if str(x[idkey]) in want]
        out_csv = f"{base}_{tag}.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
            wr.writerows(rows)
        print(f"    → {out_csv}  ({len(rows):,}문제)")
        if len(rows) != len(want):
            print(f"    ⚠️ id {len(want):,}개 중 {len(rows):,}개만 csv 에서 찾았다")


def main():
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*",
                    help="비우면 sft_data/*.jsonl 을 전부 훑는다")
    ap.add_argument("--failset", action="store_true",
                    help="선택실패/능력실패 id 목록 + 재생성용 csv 를 쓴다")
    ap.add_argument("--src_csv", default="my_val_big.csv",
                    help="--failset 이 문제 행을 떼어올 원본 csv")
    args = ap.parse_args()

    files = args.files or sorted(glob.glob("sft_data/*.jsonl"))
    if not files:
        print("후보 jsonl 을 못 찾았다. 경로를 직접 넘겨라.")
        sys.exit(1)

    id_sets = load_id_sets()
    print("대조 문제셋:", ", ".join(f"{l}({len(s):,})" for l, s in id_sets) or "없음")

    for path in files:
        if not os.path.exists(path):
            print(f"  (없음) {path}")
            continue
        with open(path, encoding="utf-8") as f:
            head = f.readline()
        if '"cands"' not in head:
            continue
        report(scan(path, id_sets), id_sets)

    print("─" * 70)


if __name__ == "__main__":
    main()
