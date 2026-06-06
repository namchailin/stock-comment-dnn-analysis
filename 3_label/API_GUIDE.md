# 실험용 API 사용 가이드 (OpenAI 호환 엔드포인트)

이 API 키는 **OpenAI 호환** 엔드포인트다. OpenAI SDK 를 그대로 쓰되
`base_url` 만 바꾸면 된다. 새 프로젝트 어디서든 이 패턴으로 붙인다.

```
API_KEY=...
BASE_URL=https://api.ssunlp.co.kr/v1
MODEL=openrouter/xxx
```

---

## 1. 자격증명은 `.env` 에 (코드에 키 하드코딩 금지)

`.env`:
```dotenv
API_KEY=sk-...
BASE_URL=https://api.ssunlp.co.kr/v1
MODEL=openrouter/xxx
```

`.gitignore` 에 `.env` 한 줄 추가 → 키가 git 에 안 올라간다.

```bash
pip install openai python-dotenv      # 또는: uv add openai python-dotenv
```

---

## 2. 클라이언트 만들기 (이게 전부)

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # .env → 환경변수

client = OpenAI(
    api_key=os.environ["API_KEY"],
    base_url=os.environ["BASE_URL"],   # ← 이 한 줄이 핵심. 엔드포인트만 바꿔 끼움
)
MODEL = os.environ["MODEL"]
```

`base_url` 만 지정하면 나머지는 OpenAI 정식 API 와 코드가 100% 동일하다.

---

## 3. Chat (텍스트 생성)

```python
resp = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "한 문장으로 자기소개 해줘."},
    ],
    temperature=0.7,
    max_tokens=512,
)
print(resp.choices[0].message.content)
```

### ⚠️ 스트리밍은 끄고 시작 (이 엔드포인트 주의점)

이 프록시는 **스트림 chat 요청에 500** 을 내는 경우가 있다.
처음엔 `stream=False`(기본값) 로 쓰고, 꼭 필요할 때만 켜서 동작 확인.

```python
# 스트림이 필요하면 (지원될 때만):
stream = client.chat.completions.create(
    model=MODEL, messages=[...], stream=True,
    stream_options={"include_usage": True},
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### ⚠️ reasoning 모델이면 thinking 끄기

reasoning 계열 모델은 추론 토큰이 답을 다 먹어서 **빈 응답**이 나오거나,
지연·토큰비용이 폭증한다. 끄는 옵션은 **모델 계열마다 다르다** → 아래
[§8 모델 계열별 thinking/reasoning 끄기](#8-모델-계열별-thinkingreasoning-끄기) 표 참조.

가장 흔한 두 가지만 미리:

```python
# (A) Qwen3 등 vLLM/HF chat-template 계열
extra_body={"chat_template_kwargs": {"enable_thinking": False}}

# (B) OpenRouter 통합 (slug 가 openrouter/* 면 이게 먹을 가능성 높음)
extra_body={"reasoning": {"enabled": False}}
```

---

## 4. Embedding (벡터)

```python
out = client.embeddings.create(
    model="baai/bge-base-en-v1.5",   # 임베딩 전용 모델명 (chat 모델과 다름)
    input=["문장 1", "문장 2"],
)
vectors = [d.embedding for d in out.data]   # 각 dim 768
```

> 코사인 유사도로 쓸 거면 직접 L2 정규화하는 게 안전하다 (`v / ||v||`).
> 프로바이더가 정규화를 보장하지 않을 수 있음.

---

## 5. 모델명 형식 (OpenRouter-style slug)

이 엔드포인트는 **소문자 slug** 를 쓴다. HuggingFace 형식이 아니다.

| 맞음 (slug) | 틀림 (HF id) |
|---|---|
| `qwen/qwen3.5-9b` | `Qwen/Qwen3.5-9B-Instruct` |
| `openrouter/xxx` | `OpenRouter/XXX` |

쓸 수 있는 모델 목록은 `/v1/models` 로 확인:

```python
for m in client.models.list().data:
    print(m.id)
```

---

## 6. 견고하게 — 재시도 (선택)

프록시가 일시적으로 5xx(503 service_unavailable 등)를 낼 수 있다.
짧게 backoff 재시도하면 실험이 중간에 안 죽는다.

```python
import time, openai

def chat_with_retry(client, **kw):
    for delay in (0, 1, 3, 9):          # 즉시 → 1s → 3s → 9s
        if delay:
            time.sleep(delay)
        try:
            return client.chat.completions.create(**kw)
        except (openai.InternalServerError,
                openai.APITimeoutError,
                openai.APIConnectionError):
            continue
    raise RuntimeError("all retries failed")
```

---

## 8. 모델 계열별 thinking/reasoning 끄기

**중요 전제:** 실제로 듣는 파라미터는 모델 계열뿐 아니라 **프록시가 무엇을
upstream 으로 포워딩하느냐**에 달렸다. OpenAI-호환 프록시는 모르는 키를 그냥
무시(silently drop)하기도 한다 → **끈 다음 응답에 추론 흔적(`<think>`,
`reasoning_content`)이 없는지, 토큰 수가 줄었는지로 반드시 검증**한다.
모르는 모델이면 아래 후보를 하나씩 넣어보고 듣는 걸 채택.

OpenAI SDK 에서 비표준 파라미터는 전부 `extra_body={...}` 로 넘긴다
(`reasoning_effort` 만 정식 인자라 top-level 가능).

| 모델 계열 | 끄는 법 | 비고 |
|---|---|---|
| **OpenAI o1 / o3 / o4-mini** | **완전 비활성 불가.** `reasoning_effort="low"` 가 최소 | reasoning model 이라 0 으로 못 끔 |
| **OpenAI GPT-5 계열** | `reasoning_effort="minimal"` | "minimal" 은 GPT-5 부터. 사실상 거의 off |
| **OpenAI gpt-4o / 4.1 등** | 끌 것 없음 (애초에 thinking 없음) | — |
| **Qwen3 (vLLM/SGLang/HF)** | `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` | 또는 프롬프트 끝에 `/no_think` soft switch |
| **Qwen DashScope API** | `extra_body={"enable_thinking": False}` | DashScope 호환 endpoint |
| **DeepSeek-R1 (`deepseek-reasoner`)** | **불가.** 끄려면 `deepseek-chat`(V3) 로 모델 교체 | R1 은 항상 추론 |
| **Claude (Anthropic)** | 기본 OFF. 켤 때만 `thinking={"type":"enabled",...}` | 끄려면 그냥 안 보냄 / `{"type":"disabled"}` |
| **Gemini 2.5 Flash** | `extra_body={"reasoning_effort": "none"}` 또는 thinking_budget=0 | Pro 는 완전 비활성 불가 |
| **OpenRouter 통합 (`openrouter/*` slug)** | `extra_body={"reasoning": {"enabled": False}}` | 또는 `{"reasoning": {"exclude": True}}` (추론은 하되 응답서 숨김) |

### 통합 레버 — slug 가 `openrouter/*` 면 이걸 먼저

OpenRouter 의 `reasoning` 파라미터는 여러 계열을 한 형식으로 추상화한다.
이 프록시처럼 `openrouter/xxx` slug 를 쓰면 이게 가장 범용일 가능성이 높다:

```python
resp = client.chat.completions.create(
    model=MODEL,
    messages=[...],
    extra_body={
        "reasoning": {
            "enabled": False,        # 완전히 끔
            # "effort": "low",       # 끄는 대신 줄이기 (high|medium|low|minimal)
            # "max_tokens": 0,       # 추론 토큰 상한 (Anthropic/Gemini 계열)
            # "exclude": True,       # 추론은 하되 응답 본문엔 안 보이게
        }
    },
)
```

### 껐는지 검증 스니펫

```python
r = resp.choices[0].message
print("content:", (r.content or "")[:80])
print("reasoning_content:", getattr(r, "reasoning_content", None))   # None 이어야 꺼진 것
print("completion_tokens:", resp.usage.completion_tokens)            # 확 줄었는지
```

`reasoning_content` 가 비어 있고 `<think>` 가 안 보이고 토큰이 줄었으면 성공.

---

## 9. 체크리스트 (새 환경에서 처음 돌릴 때)

- [ ] `.env` 에 `API_KEY` / `BASE_URL` / `MODEL` 채움, `.gitignore` 에 `.env` 추가
- [ ] `pip install openai python-dotenv`
- [ ] `OpenAI(base_url=...)` 로 클라이언트 생성
- [ ] `stream=False` 로 먼저 chat 한 번 성공 확인
- [ ] reasoning 모델이면 §8 에서 계열 맞는 옵션으로 thinking 끄고 **검증**
- [ ] 모델명은 lowercase slug (`/v1/models` 로 확인)

---

### 한 줄 동작 확인

```bash
python -c "
import os; from dotenv import load_dotenv; from openai import OpenAI
load_dotenv()
c = OpenAI(api_key=os.environ['API_KEY'], base_url=os.environ['BASE_URL'])
print(c.chat.completions.create(model=os.environ['MODEL'],
      messages=[{'role':'user','content':'say ok'}], max_tokens=10).choices[0].message.content)
"
```

`ok` 가 나오면 끝.
