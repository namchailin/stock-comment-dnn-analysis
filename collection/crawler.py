"""토스증권 종목 피드 댓글 크롤러 (내부 API 직접 호출 방식).

동작 개요
---------
1. Playwright로 종목 커뮤니티 페이지를 열어 로그인/쿠키/헤더 세션을 확보한다.
2. 페이지가 자동으로 호출하는 comments API 요청 1건을 가로채 URL/헤더를 확보한다.
3. 그 세션으로 comments API를 lastCommentId 커서를 바꿔가며 반복 호출한다.
   - 정렬은 RECENT(최신순). 과거로 거슬러 내려간다.
   - createdAt이 시작일보다 과거가 되면 종료.
   - 관찰 구간(start ~ end) 안의 글만 CSV로 저장.
4. commentId 기준 중복 제거 + 이어받기(resume) 지원.

주의: 토스 이용약관/robots.txt를 확인하고, --pause로 요청 간격을 지킬 것.
      활성 종목은 6개월치가 수십만 건일 수 있으니 시간이 오래 걸린다.
"""

import argparse
import csv
import os
import re
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from playwright.sync_api import sync_playwright

from anonymize import make_id

KST = timezone(timedelta(hours=9))

FIELDNAMES = ["commentId", "실제_닉네임", "닉네임_ID", "종목명",
              "주주_여부", "뱃지", "작성일", "내용"]

# ── 전처리 필터 ─────────────────────────────────────────────
_EMOJI_ONLY = re.compile(r"^[\s\W☀-➿\U0001F000-\U0001FAFF←-⇿⌀-⏿]+$")


def is_meaningful(text: str) -> bool:
    """텍스트 없는 이모티콘만 있는 글, 3자 이하 무의미한 글 제거."""
    if not text:
        return False
    stripped = text.strip()
    if len(re.sub(r"\s", "", stripped)) <= 3:   # 공백 제외 3자 이하 제거
        return False
    if _EMOJI_ONLY.match(stripped):             # 이모티콘/기호만 제거
        return False
    return True


def parse_created(s: str) -> datetime:
    """createdAt(나노초 9자리 + KST 오프셋) -> aware datetime."""
    s = s.replace("Z", "+00:00")
    s = re.sub(r"(\.\d{6})\d+", r"\1", s)   # 나노초 -> 마이크로초로 절삭
    return datetime.fromisoformat(s)


def extract_row(c: dict, fallback_stock: str) -> dict:
    """comments API의 댓글 객체 1건 -> CSV 행 dict."""
    author = c.get("author") or {}
    msg = c.get("message") or {}
    title = (msg.get("title") or "").strip()
    body = (msg.get("message") or "").strip()
    text = f"{title}\n{body}".strip() if (title and body) else (title or body)

    badge = author.get("badge")
    holding = (c.get("holding") or {}).get("shareHoldingStatus")
    board = c.get("board") or {}
    nickname = author.get("nickname") or ""

    return {
        "commentId": c.get("commentId"),
        "실제_닉네임": nickname,
        "닉네임_ID": make_id(nickname),
        "종목명": board.get("topic") or fallback_stock,
        "주주_여부": holding == "HOLDING",
        # 뱃지 없으면 빈 값(null), 있으면 뱃지 이름
        "뱃지": badge.get("title") if badge else "",
        "작성일": c.get("createdAt") or "",
        "내용": text,
    }


def build_url(template_url: str, last_id) -> str:
    """가로챈 API URL에서 정렬을 RECENT(최신순)로 강제하고 lastCommentId 커서를 갈아끼운다."""
    parts = urlparse(template_url)
    q = parse_qs(parts.query)
    q["commentSortType"] = ["RECENT"]   # 기본 탭은 POPULAR이므로 날짜순으로 강제
    if last_id is None:
        q.pop("lastCommentId", None)
    else:
        q["lastCommentId"] = [str(last_id)]
    new_query = urlencode({k: v[0] for k, v in q.items()})
    return urlunparse(parts._replace(query=new_query))


def fetch_page(context, template_url, headers, last_id):
    """comments API 한 페이지 호출 -> (results, key, hasNext, status)."""
    resp = context.request.get(build_url(template_url, last_id), headers=headers)
    if not resp.ok:
        return [], None, False, resp.status
    result = (resp.json() or {}).get("result") or {}
    return (result.get("results") or [], result.get("key"),
            bool(result.get("hasNext")), resp.status)


def seek_cursor(context, template_url, headers, newest_id, end_dt):
    """commentId가 시간순 증가하는 점을 이용해, 페이지 최상단 글이 end_dt 이하가 되는
    가장 큰 커서를 이진탐색으로 찾는다(종료일 근처로 점프해 불필요한 페이지를 건너뜀)."""
    lo, hi, best = 1, newest_id, None
    probes = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        results, _, _, _ = fetch_page(context, template_url, headers, mid)
        probes += 1
        if not results:
            hi = mid - 1
            continue
        top = parse_created(results[0]["createdAt"])
        if top > end_dt:        # 최상단이 아직 구간보다 최신 -> 더 과거(작은 커서)로
            hi = mid - 1
        else:                   # 구간 이하 도달 -> 더 큰 커서로 경계에 바짝
            best = mid
            lo = mid + 1
    print(f"[seek] 종료일 근처 커서 탐색 완료: {best} (probe {probes}회)")
    return best


def capture_api(page, community_url: str, login_wait: bool):
    """커뮤니티 페이지를 열고 comments API 요청 1건을 가로채 URL/헤더 반환."""
    captured = {}

    def on_req(r):
        if not captured and "/comments?" in r.url and "subjectId=" in r.url:
            captured["url"] = r.url
            captured["headers"] = r.headers

    page.on("request", on_req)
    page.goto(community_url, wait_until="domcontentloaded")

    if login_wait:
        input("브라우저에서 로그인한 뒤, 이 터미널에서 Enter를 누르세요... ")

    for _ in range(15):
        if captured:
            break
        page.mouse.wheel(0, 3000)
        time.sleep(1)

    if not captured:
        raise RuntimeError(
            "comments API 요청을 가로채지 못했습니다. URL이 종목 커뮤니티 페이지가 "
            "맞는지, 로그인이 필요한지(--login) 확인하세요."
        )
    # host/content-length 등 재전송에 부적절한 헤더 제거
    headers = {k: v for k, v in captured["headers"].items()
               if k.lower() not in ("host", "content-length", "accept-encoding")}
    return captured["url"], headers


def load_resume(out_path: str):
    """기존 CSV가 있으면 수집된 commentId 집합과 마지막 커서를 복원."""
    seen, last_id = set(), None
    if not os.path.exists(out_path):
        return seen, last_id
    with open(out_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cid = row.get("commentId")
            if cid:
                seen.add(int(cid))
    cur = out_path + ".cursor"
    if os.path.exists(cur):
        with open(cur, "r") as f:
            txt = f.read().strip()
            last_id = int(txt) if txt else None
    if seen:
        print(f"[resume] 기존 {len(seen)}건 발견, 커서 {last_id}부터 이어받기")
    return seen, last_id


def crawl(community_url: str, stock_name: str,
          start: str = "2025-10-01", end: str = "2026-03-31",
          out_path: str = "comments.csv", pause: float = 0.7,
          headless: bool = True, login_wait: bool = False,
          resume: bool = True):
    start_dt = datetime.fromisoformat(start).replace(tzinfo=KST)
    end_dt = (datetime.fromisoformat(end) + timedelta(days=1)).replace(tzinfo=KST)

    seen, last_id = (load_resume(out_path) if resume else (set(), None))
    new_count = 0

    file_exists = os.path.exists(out_path)
    f = open(out_path, "a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context()
        page = context.new_page()

        template_url, headers = capture_api(page, community_url, login_wait)
        print(f"[api] {template_url.split('?')[0]} 가로채기 완료")

        # 이어받기 커서가 없으면, 종료일 근처로 점프(seek)해서 시작
        if last_id is None:
            newest, _, _, st = fetch_page(context, template_url, headers, None)
            if not newest:
                print(f"[중단] 첫 페이지 응답 비어있음 (HTTP {st})")
                f.close(); browser.close(); return
            newest_id = newest[0]["commentId"]
            last_id = seek_cursor(context, template_url, headers, newest_id, end_dt)

        page_no = 0
        try:
            while True:
                results, key, has_next, status = fetch_page(
                    context, template_url, headers, last_id)
                if status != 200:
                    print(f"[중단] HTTP {status} — 잠시 후 재시도")
                    break
                if not results:
                    print("[완료] 더 이상 결과 없음")
                    break

                reached_old = False
                for c in results:
                    created = parse_created(c.get("createdAt", "")) if c.get("createdAt") else None
                    if created is None:
                        continue
                    if created < start_dt:
                        reached_old = True
                        continue
                    if created >= end_dt:        # 구간보다 최신 -> 건너뜀
                        continue
                    cid = c.get("commentId")
                    if cid in seen:
                        continue
                    row = extract_row(c, stock_name)
                    if not is_meaningful(row["내용"]):
                        continue
                    seen.add(cid)
                    writer.writerow(row)
                    new_count += 1

                last_id = key
                page_no += 1
                f.flush()
                with open(out_path + ".cursor", "w") as cf:
                    cf.write(str(last_id) if last_id is not None else "")

                if page_no % 20 == 0:
                    oldest = parse_created(results[-1]["createdAt"]).astimezone(KST)
                    print(f"  page {page_no} | 누적 {new_count}건 | 현재 {oldest:%Y-%m-%d}")

                if reached_old or not has_next or last_id is None:
                    print("[완료] 시작일 도달 또는 마지막 페이지")
                    break

                time.sleep(pause)   # rate limit
        finally:
            f.close()
            browser.close()

    print(f"수집 완료: 이번 실행 {new_count}건 추가 -> {out_path} (총 {len(seen)}건)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="토스증권 종목 피드 댓글 크롤러(API)")
    ap.add_argument("--url", required=True, help="종목 커뮤니티 페이지 URL")
    ap.add_argument("--stock", required=True, help="종목명 (예: TSLA)")
    ap.add_argument("--start", default="2025-10-01", help="수집 시작일 YYYY-MM-DD")
    ap.add_argument("--end", default="2026-03-31", help="수집 종료일 YYYY-MM-DD")
    ap.add_argument("--out", default="comments.csv", help="출력 CSV 경로")
    ap.add_argument("--pause", type=float, default=0.7, help="요청 간 대기(초)")
    ap.add_argument("--show", action="store_true", help="브라우저 창 표시")
    ap.add_argument("--login", action="store_true", help="시작 시 수동 로그인 대기")
    ap.add_argument("--no-resume", action="store_true", help="이어받기 비활성화")
    args = ap.parse_args()

    crawl(args.url, args.stock, start=args.start, end=args.end,
          out_path=args.out, pause=args.pause, headless=not args.show,
          login_wait=args.login, resume=not args.no_resume)
