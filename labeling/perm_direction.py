# D) 방향예측 순열검정 — "방향을 잘 골랐나(우연보다)?"
#   예측자들의 방향을 서로 섞어(상승/하락 비율 유지) 방향실현율 귀무분포 생성.
#   실제 실현율이 그보다 높으면 → 방향 선택에 실력. 같으면 → 시장 따라간 운.
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finalize_labels as F


def realized_both(sd, jak, pred_date):
    A = F.anchor_idx(sd, jak)
    if A < 0:
        return None
    D = pred_date if pd.notna(pred_date) else None
    D_idx = F.session_for_date(sd, D) if D else None
    date_valid = (D_idx is not None) and (D_idx > A)
    end = (D_idx if date_valid else A) + F.DIR_FWD
    if end >= sd["n"]:
        return None
    h = end - A
    R = (sd["closes"][end] / sd["closes"][A] - 1) * 100
    up = sd["mu"] * h + F.K * sd["sig"] * np.sqrt(h)
    dn = sd["mu"] * h - F.K * sd["sig"] * np.sqrt(h)
    return (R >= up), (R <= dn)


def run(files, label, B=3000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.concat([pd.read_csv(os.path.join(F.LABELED, f"{f}_final.csv"), encoding="utf-8-sig")
                    for f in files], ignore_index=True)
    df = df[df["Class"].isin([1, 2, 3])].copy()        # 방향 예측자
    sds = {s: F.load_stock(s) for s in df["종목명"].unique()}

    up_r, dn_r, dirs = [], [], []
    for r in df.itertuples():
        rb = realized_both(sds[r.종목명], r.작성일, r.예측_날짜)
        if rb is None:
            continue
        up_r.append(rb[0]); dn_r.append(rb[1]); dirs.append(r.예측_방향)
    up_r = np.array(up_r); dn_r = np.array(dn_r); dirs = np.array(dirs)
    n = len(dirs)

    def rate(dv):
        return np.where(dv == "상승", up_r, dn_r).mean()

    obs = rate(dirs)
    null = np.array([rate(rng.permutation(dirs)) for _ in range(B)])
    p = (null >= obs).mean()
    print(f"[{label}] n={n} 방향예측자 | 실제 방향실현율 = {obs*100:.1f}%")
    print(f"  방향섞음(우연) 평균 {null.mean()*100:.1f}% / 95%상한 {np.quantile(null,0.95)*100:.1f}%")
    print(f"  p-value = {p:.4f} → " +
          ("**유의: 방향 선택에 실력**" if p < 0.05 else "유의X: 방향실현도 우연(시장 따라감)과 무차이"))


if __name__ == "__main__":
    print("=== NVDA 평가셋 ===")
    run(["nvda_eval"], "NVDA test")
    print("\n=== train+test 합침 ===")
    run(["tsla_train", "nvda_eval"], "TSLA+NVDA")
