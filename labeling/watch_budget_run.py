# 예산 충전 감시 → 충전되면 Fold A 풀 라벨링 자동 실행.
#   30분마다 싼 테스트콜로 확인(예산초과면 400이라 과금 0). 성공하면:
#     train_A_TSLA(12,635→20,206) + test_A_NVDA(→9,084) = 약 16,655콜, 워커 20.
#   재개(resume) 기반이라 train_A 기존 12,635건은 재라벨 안 함.
import os
import time
import subprocess
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
load_dotenv(os.path.join(ROOT, ".env"))
client = OpenAI(api_key=os.environ["API_KEY"], base_url=os.environ["BASE_URL"])
MODEL = os.environ["MODEL"]

INTERVAL = 1800     # 30분
MAX_CHECKS = 48     # 최대 24시간
WORKERS = "12"      # 과부하·Cloudflare 보호 트리거 방지(낮게)


def stamp():
    return datetime.now().strftime("%m-%d %H:%M:%S")


def budget_ok():
    try:
        client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": "hi"}],
            max_tokens=5, temperature=0)
        return True, ""
    except Exception as e:
        return False, str(e)[:140]


def main():
    print(f"[{stamp()}] 예산 감시 시작 (30분 간격, 최대 {MAX_CHECKS}회)", flush=True)
    for i in range(1, MAX_CHECKS + 1):
        ok, msg = budget_ok()
        if ok:
            print(f"[{stamp()}] ✅ 예산 확보 → 풀 라벨링 시작 (워커 {WORKERS})", flush=True)
            for f in ("train_A_TSLA", "test_A_NVDA"):
                print(f"[{stamp()}] ▶ {f}", flush=True)
                subprocess.run(["python3", "run_label.py", "--file", f,
                                "--workers", WORKERS], cwd=HERE)
            print(f"[{stamp()}] 🎉 Fold A 라벨링 완료 — finalize_labels.py 실행 단계", flush=True)
            return
        print(f"[{stamp()}] 체크 {i}/{MAX_CHECKS}: 예산 미충전 ({msg}). 30분 후 재확인", flush=True)
        time.sleep(INTERVAL)
    print(f"[{stamp()}] {MAX_CHECKS}회 미충전 — 감시 종료(수동 재시작 필요)", flush=True)


if __name__ == "__main__":
    main()
