#!/usr/bin/env python3
import argparse
import glob
import json
import os
from collections import Counter, defaultdict


def pick_wmaj(answers, scores, alpha):
    acc = defaultdict(float)
    for a, s in zip(answers, scores):
        if a is None:
            continue
        acc[a] += (s ** alpha) if alpha else 1.0
    return max(acc.items(), key=lambda kv: kv[1])[0] if acc else None


def pick_mix(answers, scores, beta, gamma):
    cnt, tot = defaultdict(int), defaultdict(float)
    for a, s in zip(answers, scores):
        if a is None:
            continue
        cnt[a] += 1
        tot[a] += s
    if not cnt:
        return None
    return max(cnt, key=lambda a: cnt[a] ** beta * (tot[a] / cnt[a]) ** gamma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", action="append", required=True,
                    help="verif_detail*.jsonl (glob 가능). 여러 번 주면 점수를 평균한다")
    ap.add_argument("--alpha", action="append", type=float,
                    help="가중 다수결 `Σ p^α`. 여러 번 지정 가능. 0 은 다수결 대조군")
    ap.add_argument("--mix", action="append", metavar="BETA,GAMMA",
                    help="집계를 `count^BETA · mean_p^GAMMA` 로. 여러 번 지정 가능. "
                         "val 이 고른 값은 0.5,1")
    ap.add_argument("--out_prefix", required=True,
                    help="'{prefix}_a{alpha}.csv' 로 저장된다")
    ap.add_argument("--lb", default="deep_chal_math_leaderboard_filtered.csv")
    args = ap.parse_args()

    groups = []
    for pat in args.detail:
        paths = sorted(glob.glob(pat))
        if not paths:
            raise SystemExit(f"[ERROR] 없음: {pat}")
        g = {}
        for p in paths:
            for line in open(p, encoding="utf-8"):
                if line.strip():
                    d = json.loads(line)
                    g[str(d["id"])] = d
        print(f"[load] {pat}  →  {len(g):,}문항")
        groups.append(g)

    base_g = groups[0]
    if "scores" not in next(iter(base_g.values())) or "answers" not in next(iter(base_g.values())):
        raise SystemExit("[ERROR] detail 에 scores/answers 가 없다 (구버전 infer_verifier). "
                         "GPU 재채점이 필요하다")

    rows = []
    n_skip = 0
    for i, r in base_g.items():
        acc = list(r["scores"])
        ok = True
        for g in groups[1:]:
            o = g.get(i)
            if o is None or o["answers"] != r["answers"]:
                ok = False
                break
            acc = [a + b for a, b in zip(acc, o["scores"])]
        if not ok:
            n_skip += 1
            continue
        rows.append({**r, "scores": [a / len(groups) for a in acc]})
    if len(groups) > 1:
        print(f"[앙상블] verifier {len(groups)}개 점수 평균  →  {len(rows):,}문항"
              f"{f'  (후보 불일치로 제외 {n_skip:,})' if n_skip else ''}")
        if n_skip:
            print("   ⚠️ 제외분이 있다 = 서로 다른 --cands 로 채점한 파일을 섞었다는 뜻")
    has_gold = rows[0].get("gold") is not None

    rules = []
    for a in (args.alpha or []):
        rules.append((f"a{a:g}", lambda ansl, sc, a=a: pick_wmaj(ansl, sc, a)))
    for m in (args.mix or []):
        try:
            b, g = (float(x) for x in m.split(","))
        except ValueError:
            raise SystemExit(f"[ERROR] --mix 형식은 BETA,GAMMA 다 (받은 값: {m!r}). 예: --mix 0.5,1")
        rules.append((f"mix{b:g}g{g:g}",
                      lambda ansl, sc, b=b, g=g: pick_mix(ansl, sc, b, g)))
    if not rules:
        raise SystemExit("[ERROR] --alpha 나 --mix 를 최소 하나는 줄 것")

    import pandas as pd
    base = None
    print(f"\n{'규칙':>12}{'기준과 다름':>12}{'정확도':>10}")
    for tag, fn in rules:
        ans, ok = [], 0
        for r in rows:
            v = fn(r["answers"], r["scores"])
            if v is None:
                v = r.get("maj")
            ans.append({"id": r["id"], "answer": str(v) if v is not None else "0"})
            if has_gold:
                ok += (v == r["gold"])
        if base is None:
            base = [x["answer"] for x in ans]
            diff = 0
        else:
            diff = sum(1 for x, b in zip(ans, base) if x["answer"] != b)
        acc = f"{ok/len(rows)*100:9.2f}%" if has_gold else "        -"
        out = f"{args.out_prefix}_{tag}.csv"
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        pd.DataFrame(ans).to_csv(out, index=False)
        print(f"{tag:>12}{diff:>12,}{acc}   → {os.path.basename(out)}")

    if os.path.exists(args.lb):
        lb = pd.read_csv(args.lb)
        need, got = set(lb["id"].astype(str)), {str(r["id"]) for r in rows}
        print("-" * 40)
        print(f"  행수 {len(rows)} (기대 {len(lb)})  "
              f"{'OK' if len(rows) == len(lb) else '★불일치'}")
        print(f"  id   {'OK' if got == need else f'★누락{len(need-got)} 초과{len(got-need)}'}")
    print("\n※ 같은 후보·같은 점수라 α 차이만 남는다. 생성 노이즈 0.")
    print("  ⚠️ 제출은 하루 5회다. α 는 val 로 고르고, 리더보드 슬롯은 최종본에만 쓸 것.")


if __name__ == "__main__":
    main()
