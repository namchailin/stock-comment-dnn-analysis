"""닉네임 가명처리 모듈.

HMAC-SHA256(닉네임, 비밀 솔트)로 안정적인 가명 ID를 생성한다.
- 같은 닉네임 -> 항상 같은 ID (시계열/유저별 분석 가능)
- 비밀 솔트가 없으면 역추적(레인보우 테이블)이 막힘
- 요청에 따라 `실제_닉네임`과 `닉네임_ID`를 같은 데이터에 함께 보관한다.
  => 이는 '익명화'가 아니라 '가명처리'임에 유의.
"""

import hashlib
import hmac
import os
import secrets

# 비밀 솔트 우선순위:
#   1) 환경변수 PSEUDO_SALT (있으면 사용)
#   2) 없으면 .salt 파일에서 읽음
#   3) .salt 파일도 없으면 무작위로 생성해 저장 (최초 1회만)
# => 사용자가 따로 export 할 필요 없음.
_SALT_ENV = "PSEUDO_SALT"
_SALT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".salt")


def _get_salt() -> bytes:
    env_salt = os.environ.get(_SALT_ENV)
    if env_salt:
        return env_salt.encode("utf-8")

    if os.path.exists(_SALT_FILE):
        with open(_SALT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip().encode("utf-8")

    # 최초 실행: 무작위 솔트 생성 후 저장
    new_salt = secrets.token_hex(32)
    with open(_SALT_FILE, "w", encoding="utf-8") as f:
        f.write(new_salt)
    return new_salt.encode("utf-8")


def make_id(nickname: str, id_len: int = 16) -> str:
    """닉네임을 HMAC-SHA256 가명 ID(hex)로 변환한다.

    id_len: 반환할 hex 문자 수 (기본 16자, 최대 64자).
    """
    nickname = (nickname or "").strip()
    digest = hmac.new(_get_salt(), nickname.encode("utf-8"), hashlib.sha256).hexdigest()
    return digest[: max(8, min(id_len, 64))]


if __name__ == "__main__":
    # 데모: PSEUDO_SALT 세팅 후 실행
    for n in ["주식고수", "토스개미", "주식고수"]:
        print(f"{n:8s} -> {make_id(n)}")
