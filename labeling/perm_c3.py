# C3 순열검정 — "예측 날짜 적중이 우연(아무 날짜)보다 높은가?"
#   시장방향·예측방향·작성시점은 그대로 두고, 예측 '날짜 오프셋'만 섞어서
#   날짜적중률(W_hit)의 귀무분포를 만든 뒤 실제값과 비교 → 날짜 선택의 실력 검정.
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finalize_labels as F   # load_stock, session_for_date, crossed, HIT_HALF


def date_hit(sd, date, direction) -> bool:
    di = F.session_for_date(sd, date)
    if di is None:
        return False
    hb, he = di - (F.HIT_HALF + 1), di + F.HIT_HALF
    if hb < 0 or he >= sd["n"]:
        return False
    Rh = (sd["closes"][he] / sd["closes"][hb] - 1) * 100
    return bool(F.crossed(direction, Rh, sd["mu"], sd["sig"], he - hb))


def run(files, label, B=3000, seed=42):
    rng = np.random.default_rng(seed)
    sub = []
    for f in files:
        d = pd.read_csv(os.path.join(F.LABELED, f"{f}_final.csv"), encoding="utf-8-sig")
        d = d[(d["예측_방향"].isin(["상승", "하락"])) & (d["예측_날짜"].notna())
              & (d["시점관계"] == "예측")].copy()
        d["_stock"] = d["종목명"]
        sub.append(d)
    df = pd.concat(sub, ignore_index=True)
    sds = {s: F.load_stock(s) for s in df["_stock"].unique()}

    base = pd.to_datetime(df["작성일"]).dt.tz_localize(None).dt.normalize()
    pred = pd.to_datetime(df["예측_날짜"], errors="coerce").dt.normalize()
    off = (pred - base).dt.days
    keep = (off >= 0) & (off <= 30)        # 비현실적 먼 날짜(LLM 헐렁추출) 제거
    df, off = df[keep].reset_index(drop=True), off[keep].reset_index(drop=True)
    base = base[keep].reset_index(drop=True)
    offsets = off.to_numpy()
    base = base.to_numpy(); dirs = df["예측_방향"].to_numpy(); stocks = df["_stock"].to_numpy()

    # 사전계산: 각 행 × 각 후보 오프셋(0~30)의 날짜적중 여부 (한 번만)
    n = len(df)
    uniq = np.unique(offsets)
    col = {int(o): j for j, o in enumerate(uniq)}
    H = np.zeros((n, len(uniq)), dtype=bool)
    for i in range(n):
        bi, sd, di = pd.Timestamp(base[i]), sds[stocks[i]], dirs[i]
        for j, o in enumerate(uniq):
            H[i, j] = date_hit(sd, (bi + pd.Timedelta(days=int(o))).strftime("%Y-%m-%d"), di)
    idx = np.arange(n)
    acol = np.array([col[int(o)] for o in offsets])
    obs = H[idx, acol].mean()                                  # 실제 적중률
    null = np.array([H[idx, rng.permutation(acol)].mean() for _ in range(B)])  # 날짜 섞음
    p = (null >= obs).mean()
    print(f"[{label}] n={len(df)} 날짜예측자 | 실제 날짜적중률 = {obs*100:.1f}%")
    print(f"  우연(날짜섞음) 분포: 평균 {null.mean()*100:.1f}% / 95%상한 {np.quantile(null,0.95)*100:.1f}%")
    print(f"  p-value(실제≥우연) = {p:.4f}  → " +
          ("**유의: 날짜선택에 실력 있음**" if p < 0.05 else "유의X: 날짜적중도 우연과 구별 안 됨"))
    return obs, null, p


if __name__ == "__main__":
    print("=== NVDA 평가셋만 ===")
    run(["nvda_eval"], "NVDA test")
    print("\n=== train+test 합침(검정력↑) ===")
    run(["tsla_train", "nvda_eval"], "TSLA+NVDA")
