#!/usr/bin/env python3
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="deep_chal_math_train.csv")
    ap.add_argument("--bad_ids", default="train_filtered_ids.csv")
    ap.add_argument("--n_big", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.train)
    df.columns = [c.strip() for c in df.columns]
    df["id"] = df["id"].astype(str)

    bad = set(pd.read_csv(args.bad_ids)["id"].astype(str))
    print(f"[bad] 오류 문항 {len(bad):,}개 (공지 8/3)")

    old = df.sample(500, random_state=args.seed)
    old_ids = set(old["id"])
    hit = old_ids & bad
    print(f"[old] my_val 500 중 오류 {len(hit)}개 → 유효 {500 - len(hit)}개")

    keep_old = old[~old["id"].isin(bad)]
    keep_old.to_csv("my_val_filtered.csv", index=False, encoding="utf-8")
    print(f"[out] my_val_filtered.csv : {len(keep_old)}개")

    pool = df[~df["id"].isin(bad) & ~df["id"].isin(old_ids)]
    need = args.n_big - len(keep_old)
    if need > len(pool):
        raise SystemExit(f"pool({len(pool)})이 부족하다. --n_big 을 낮춰라.")
    extra = pool.sample(need, random_state=args.seed + 1)

    big = pd.concat([keep_old, extra]).reset_index(drop=True)
    big.to_csv("my_val_big.csv", index=False, encoding="utf-8")
    print(f"[out] my_val_big.csv     : {len(big)}개 "
          f"(기존 {len(keep_old)} + 추가 {len(extra)})")

    clean = df[~df["id"].isin(bad)]
    clean_no_val = clean[~clean["id"].isin(set(big["id"]))]
    clean.to_csv("train_clean.csv", index=False, encoding="utf-8")
    clean_no_val.to_csv("train_clean_noval.csv", index=False, encoding="utf-8")
    print(f"[out] train_clean.csv       : {len(clean):,}개 (오류만 제외)")
    print(f"[out] train_clean_noval.csv : {len(clean_no_val):,}개 (오류 + 평가셋 제외)")
    print("\n※ distill 은 train_clean_noval.csv 를 써라. 평가셋이 학습에 새면 점수가 부풀려진다.")

    import math
    for n, label in [(len(keep_old), "my_val_filtered"), (len(big), "my_val_big")]:
        se = math.sqrt(0.75 * 0.25 / n) * 100
        print(f"  {label:<18} n={n:<5} 단일 SE≈{se:.2f}%p")


if __name__ == "__main__":
    main()
