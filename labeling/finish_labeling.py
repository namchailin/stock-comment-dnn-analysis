# Fold A 라벨링 마무리 — 단일 프로세스. 서버 깜빡임(530)·일시오류에 재시도 내장.
#   train_A_TSLA·test_A_NVDA 를 라운드 반복: 남은 게 0 될 때까지(resume이 에러행 재시도).
#   워커 낮게(12), 라운드 사이 대기. 중복 프로세스 없이 이거 하나만 돌리면 됨.
import os
import sys
import time
import subprocess
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
# 인자로 파일 지정 가능: python3 finish_labeling.py test_A_NVDA
TARGETS = sys.argv[1:] or ["train_A_TSLA", "test_A_NVDA"]
WORKERS = "48"
MAX_ROUNDS = 15
WAIT = 120


def remaining(name):
    total = len(pd.read_csv(os.path.join(DATA, f"{name}.csv"), encoding="utf-8-sig"))
    lp = os.path.join(DATA, f"labeled_{name}.csv")
    if not os.path.exists(lp):
        return total
    d = pd.read_csv(lp, encoding="utf-8-sig", on_bad_lines="skip", engine="python")
    good = d[d["시점관계"].notna()]["commentId"].nunique() if "시점관계" in d.columns else 0
    return total - good


def main():
    for rnd in range(1, MAX_ROUNDS + 1):
        rem = {n: remaining(n) for n in TARGETS}
        print(f"\n=== 라운드 {rnd} | 남은: {rem} ===", flush=True)
        if sum(rem.values()) == 0:
            print("🎉 Fold A 라벨링 전부 완료", flush=True)
            return
        for name in TARGETS:
            if rem[name] > 0:
                subprocess.run(["python3", "run_label.py", "--file", name,
                                "--workers", WORKERS], cwd=HERE)
        if sum(remaining(n) for n in TARGETS) > 0:
            time.sleep(WAIT)       # 서버 깜빡임 등 → 잠깐 쉬고 재시도
    print("⚠️ 최대 라운드 도달 — 남은 것 있으면 재실행", flush=True)


if __name__ == "__main__":
    main()
