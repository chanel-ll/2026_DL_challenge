#!/usr/bin/env python3
import argparse
import json
import os
from collections import OrderedDict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="합칠 후보 jsonl 들")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow_shards", action="store_true",
                    help="샤드 파일(.shardN.jsonl)도 입력으로 받는다. 보통 필요 없다")
    ap.add_argument("--force", action="store_true",
                    help="seed 미변경(완전 동일) 판정이 나와도 파일을 쓴다")
    args = ap.parse_args()

    shards = [p for p in args.inputs if ".shard" in os.path.basename(p)]
    if shards and not args.allow_shards:
        raise SystemExit(
            "[ERROR] 샤드 파일이 입력에 섞였다 — 본 파일의 부분집합이라 중복이 된다:\n"
            + "\n".join(f"        {p}" for p in shards)
            + "\n        glob 의 * 가 '.shard0' 까지 먹은 것이다."
              " `s91[1-4].jsonl` 처럼 자릿수를 못 박거나 파일명을 직접 나열할 것.\n"
              "        정말 의도한 것이면 --allow_shards")

    merged = OrderedDict()
    per_file = []
    for path in args.inputs:
        sig, n_row = {}, 0
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            n_row += 1
            sig[d["id"]] = tuple(sorted(c.get("text", "") for c in d["cands"]))
            if d["id"] not in merged:
                merged[d["id"]] = d
            else:
                merged[d["id"]]["cands"].extend(d["cands"])
        per_file.append((path, sig))
        print(f"[load] {path}  {n_row:,}문제")

    n = len(merged)
    counts = [len(d["cands"]) for d in merged.values()]
    lo, hi = min(counts), max(counts)

    MIN_VARIETY = 3
    identical = []
    for i in range(len(per_file)):
        for j in range(i + 1, len(per_file)):
            (pa, sa), (pb, sb) = per_file[i], per_file[j]
            usable = [k for k in (set(sa) & set(sb))
                      if len(set(sa[k])) >= MIN_VARIETY and len(set(sb[k])) >= MIN_VARIETY]
            if not usable:
                continue
            same = sum(1 for k in usable if sa[k] == sb[k])
            identical.append((pa, pb, same, len(usable)))

    n_fix = 0
    for d in merged.values():
        nc = sum(1 for c in d["cands"] if c.get("correct"))
        nw = len(d["cands"]) - nc
        if d.get("n_correct") != nc or d.get("n_wrong") != nw:
            n_fix += 1
        d["n_correct"], d["n_wrong"] = nc, nw

    if n_fix:
        print(f"[fix] n_correct/n_wrong 재계산 — {n_fix:,}개 레코드가 낡은 값이었다")

    print("=" * 66)
    print(f"→ {args.out}")
    print(f"   문제 {n:,}개   후보 {lo}~{hi}개/문제   총 {sum(counts):,}개")
    if lo != hi:
        print(f"   ⚠️ 후보 수가 문제마다 다르다({lo}~{hi}). 일부 입력에 없는 문제가 있다")

    print("-" * 66)
    print("seed 분리 검사 (후보끼리 겹치는 것 자체는 정상 — 다수결이 그걸로 돈다)")
    print(f"  판정 대상: 파일 안에 서로 다른 풀이가 {MIN_VARIETY}종 이상인 문제만")
    print("  (전원 같은 풀이인 쉬운 문제는 독립 추출이어도 일치한다 — 정보가 없다)")
    bad = False
    if not identical:
        print("   ⚠️ 판정 가능한 문제가 없다. 두 파일에 공통 문제가 없거나 전부 단조롭다")
    for pa, pb, same, tot in identical:
        pct = same / max(tot, 1) * 100
        if pct > 80:
            bad = True
            print(f"   🔴 {pa}\n      {pb}\n"
                  f"      → 판정대상 {tot:,}문제 중 {same:,}개({pct:.1f}%) 에서 후보 집합이 "
                  f"**완전히 동일**하다")
        else:
            print(f"   🟢 완전 동일 {same:,}/{tot:,} ({pct:.1f}%) — 독립 추출이다")
    if bad:
        print()
        print("   **seed 를 안 바꿨거나 같은 파일을 두 번 넣은 것이다.** 파일 크기는 두 배인데")
        print("   실제 다양성은 그대로라, 겉보기 n 만 커지고 성능은 절반 n 과 같아진다.")
        print("   gen_verifier_data.py --seed 0 / --seed 1 처럼 다르게 줄 것.")
        if not args.force:
            raise SystemExit(
                "\n[ERROR] 오염된 병합이라 **파일을 쓰지 않았다.**\n"
                "        원인을 고치고 다시 합칠 것. 정말 이대로 쓰려면 --force")

    with open(args.out, "w", encoding="utf-8") as f:
        for d in merged.values():
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print("=" * 66)
    tag = os.path.basename(args.out).replace(".jsonl", "")
    print("다음: 채점만 하면 된다 (후보가 이미 있으므로 생성 단계는 건너뛴다)")
    print(f"  VERIF=./verifier_out/v1/merged CANDS={args.out} \\")
    print(f"      VAL=deep_chal_math_leaderboard_filtered.csv N=128 ALPHA=2 \\")
    print(f"      OUTDIR=verif_eval/{tag} \\")
    print(f"      sbatch --gres=gpu:1 -w ariel-vXX run_verifier_eval.sh")
    print()
    print("  ⚠️ SUBMIT / SUBMIT_MAJ 를 주지 말 것. 제출 파일은 채점이 끝난 뒤")
    print("     resubmit_alpha.py 로 만든다 — α 를 바꿀 때마다 GPU 를 다시 쓰지 않기 위해서다.")
    print("     (예전 힌트는 submission_verifier_a4_n128.csv 를 덮어쓰게 되어 있었다.")
    print("      그 파일들은 **리더보드 점수를 아는 기준선**이라 덮으면 대응이 끊긴다.)")
    print(f"  python resubmit_alpha.py --detail 'verif_eval/{tag}/verif_detail_*.jsonl' \\")
    print(f"      --alpha 0 --alpha 2 --out_prefix submissions/submission_{tag}")


if __name__ == "__main__":
    main()
