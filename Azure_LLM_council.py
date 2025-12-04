# Azure_LLM_council.py
# ---------------------------------------------------------
# Azure 기반 리소스에서 다음 모델들을 한 번에 council로 돌리는 예제:
#   - gpt-5
#   - gpt-5.1  (Chairman)
#   - gpt-4.1
#   - claude-opus-4-5
#   - claude-sonnet-4-5
# 임베딩:
#   - text-embedding-3-small
#
# 특징:
#   - 모든 모델이 같은 API 키(OPEN_AI_KEY_5)를 사용
#   - 공통 엔드포인트 OPEN_AI_ENDPOINT_5 를 사용
#   - OpenAI 계열은 /openai/v1/chat/completions / /openai/v1/embeddings
#   - Claude 계열은 /anthropic/v1/messages
#   - Stage1/2/3 + 임베딩 테스트 + 에러/디버그 출력
# ---------------------------------------------------------

# ---------------------------------------------------------
# Azure 리소스에서 다음 모델로 LLM Council을 구성하는 예제:
#   - gpt-5
#   - gpt-5.1  (Chairman)
#   - gpt-4.1
#   - claude-opus-4-5
#   - claude-sonnet-4-5
# 임베딩:
#   - text-embedding-3-small
#
# 특징:
#   - 모든 모델은 같은 API 키(AZURE_API_KEY)를 사용
#   - 공통 엔드포인트(OPEN_AI_ENDPOINT_5)를 사용
#   - OpenAI 계열: /openai/v1/chat/completions, /openai/v1/embeddings
#   - Claude 계열: /anthropic/v1/messages
#   - 모든 프롬프트와 출력 언어는 한국어 중심으로 설정
#   - Stage 1/2/3 + 임베딩 테스트 + 예외 출력
# ---------------------------------------------------------

import os
import json
import asyncio
from dataclasses import dataclass
from typing import List, Dict

import aiohttp
from dotenv import load_dotenv

# ---------------------------------------------------------
# 0. 환경 변수 로드 및 기본 설정
# ---------------------------------------------------------

load_dotenv()

# 공통 API 키
API_KEY = os.getenv("OPEN_AI_KEY_5")
if not API_KEY:
    raise RuntimeError("AZURE_API_KEY 환경변수를 설정하세요.")

# 공통 엔드포인트
BASE_ENDPOINT = os.getenv(
    "OPEN_AI_ENDPOINT_5").rstrip("/")

# OpenAI v1 스타일용 버전 (실제 v1에서는 URL에 안 쓸 수 있지만 보관용)
OPENAI_V1_API_VERSION = os.getenv("AZURE_OPENAI_V1_API_VERSION", "2024-02-15-preview")

# Claude(Anthropic) 버전
ANTHROPIC_VERSION = os.getenv("AZURE_ANTHROPIC_VERSION", "2023-06-01")

# ---------------------------------------------------------
# 1. 모델/임베딩 이름 (환경변수 없으면 디폴트 값 사용)
# ---------------------------------------------------------

DEP_GPT5   = os.getenv("AZURE_DEPLOY_GPT5",   "gpt-5")
DEP_GPT5_1 = os.getenv("AZURE_DEPLOY_GPT5_1", "gpt-5.1")
DEP_GPT4_1 = os.getenv("AZURE_DEPLOY_GPT4_1", "gpt-4.1")

ANTHROPIC_MODEL_CLAUDE_OPUS_4_5   = os.getenv("ANTHROPIC_MODEL_CLAUDE_OPUS_4_5",   "claude-opus-4-5")
ANTHROPIC_MODEL_CLAUDE_SONNET_4_5 = os.getenv("ANTHROPIC_MODEL_CLAUDE_SONNET_4_5", "claude-sonnet-4-5")

# council 멤버: (provider, model_name)
COUNCIL_MODELS = [
    ("openai",    DEP_GPT5),
    ("openai",    DEP_GPT5_1),
    ("openai",    DEP_GPT4_1),
    ("anthropic", ANTHROPIC_MODEL_CLAUDE_OPUS_4_5),
    ("anthropic", ANTHROPIC_MODEL_CLAUDE_SONNET_4_5),
]

# Chairman 은 gpt-5.1
CHAIRMAN_MODEL = ("openai", DEP_GPT5_1)

# ---------------------------------------------------------
# 2. 데이터 모델
# ---------------------------------------------------------

@dataclass
class ModelAnswer:
    provider: str   # "openai" / "anthropic"
    name: str       # gpt-5, claude-opus-4-5 등
    content: str    # 한국어 답변

@dataclass
class ReviewResult:
    reviewer: str          # "openai:gpt-5" 등
    ranking: List[str]     # "openai:gpt-5", "anthropic:claude-opus-4-5" 등
    comments: Dict[str, str]

# ---------------------------------------------------------
# 3. ID 헬퍼
# ---------------------------------------------------------

def model_id(provider: str, name: str) -> str:
    return f"{provider}:{name}"

def answer_id(ans: ModelAnswer) -> str:
    return model_id(ans.provider, ans.name)

# ---------------------------------------------------------
# 4. OpenAI v1 Chat / Embeddings 호출 (temperature 제거)
# ---------------------------------------------------------

async def call_openai_chat(
    session: aiohttp.ClientSession,
    model: str,
    messages: List[Dict[str, str]],
) -> str:
    """
    gpt-5/5.1/4.1 호출. temperature는 기본값만 허용되므로 지정하지 않습니다.[web:25][web:50]
    """
    url = f"{BASE_ENDPOINT}/openai/v1/chat/completions"

    payload = {
        "model": model,
        "messages": messages,
        # "temperature": 0.7  # 일부 환경에서 막혀 있으므로 제거
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    async with session.post(url, headers=headers, data=json.dumps(payload)) as resp:
        text = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"[OpenAI {model}] {resp.status}: {text}")
        data = json.loads(text)
        return data["choices"][0]["message"]["content"]

async def call_openai_embedding(
    session: aiohttp.ClientSession,
    model: str,
    text: str,
) -> List[float]:
    """
    text-embedding-3-small 임베딩 호출.[web:29][web:52]
    """
    url = f"{BASE_ENDPOINT}/openai/v1/embeddings"

    payload = {
        "model": model,
        "input": text,
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    async with session.post(url, headers=headers, data=json.dumps(payload)) as resp:
        text_resp = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"[Embedding {model}] {resp.status}: {text_resp}")
        data = json.loads(text_resp)
        return data["data"][0]["embedding"]

# ---------------------------------------------------------
# 5. Anthropic(Claude) 호출
# ---------------------------------------------------------

async def call_anthropic_chat(
    session: aiohttp.ClientSession,
    model: str,
    messages: List[Dict[str, str]],
) -> str:
    """
    Anthropic(Claude) messages API 호출.[web:41]
    """
    url = f"{BASE_ENDPOINT}/anthropic/v1/messages"

    system_prompt = None
    user_contents = []

    # OpenAI 스타일 messages를 Anthropic 포맷으로 변환
    for m in messages:
        if m["role"] == "system":
            system_prompt = m["content"]
        elif m["role"] == "user":
            user_contents.append({"type": "text", "text": m["content"]})

    anthropic_messages = []
    if user_contents:
        anthropic_messages.append({
            "role": "user",
            "content": user_contents,
        })

    payload = {
        "model": model,
        "max_tokens": 1000,
        "temperature": 0.7,
        "messages": anthropic_messages,
    }
    if system_prompt:
        payload["system"] = system_prompt

    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY,
        "anthropic-version": ANTHROPIC_VERSION,
    }

    async with session.post(url, headers=headers, data=json.dumps(payload)) as resp:
        text_resp = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"[Anthropic {model}] {resp.status}: {text_resp}")
        data = json.loads(text_resp)
        parts = data.get("content", [])
        texts = [p.get("text", "") for p in parts if p.get("type") == "text"]
        return "\n".join(texts)

# ---------------------------------------------------------
# 6. 공통 call_chat 래퍼
# ---------------------------------------------------------

async def call_chat(
    session: aiohttp.ClientSession,
    provider: str,
    name: str,
    messages: List[Dict[str, str]],
) -> str:
    if provider == "openai":
        return await call_openai_chat(session, name, messages)
    elif provider == "anthropic":
        return await call_anthropic_chat(session, name, messages)
    else:
        raise ValueError(f"Unknown provider: {provider}")

# ---------------------------------------------------------
# 7. Stage 1: 각 모델의 첫 답변 (한국어 지시)
# ---------------------------------------------------------

async def stage1_first_opinions(
    session: aiohttp.ClientSession,
    user_query: str,
) -> List[ModelAnswer]:
    print("[Stage 1] 각 모델의 첫 답변 생성 시작...")

    tasks = []
    for provider, name in COUNCIL_MODELS:
        messages = [
            {
                "role": "system",
                "content": (
                    "당신은 매우 유능한 한국어 어시스턴트입니다. "
                    "항상 한국어로만 답변하고, 사실에 근거해 정확하고 친절하게 설명하십시오."
                ),
            },
            {
                "role": "user",
                "content": (
                    "다음 질문에 대해 한국어로 자세하면서도 이해하기 쉽게 답변해 주세요.\n\n"
                    f"질문: {user_query}"
                ),
            },
        ]
        tasks.append(call_chat(session, provider, name, messages))

    results = await asyncio.gather(*tasks)

    answers: List[ModelAnswer] = []
    for (provider, name), content in zip(COUNCIL_MODELS, results):
        answers.append(ModelAnswer(provider=provider, name=name, content=content))

    for ans in answers:
        print(f"\n=== {answer_id(ans)} 의 답변 ===")
        print(ans.content)

    return answers

# ---------------------------------------------------------
# 8. Stage 2: 상호 리뷰 (리뷰도 한국어로)
# ---------------------------------------------------------

def build_review_prompt(user_query: str, all_answers: List[ModelAnswer]) -> str:
    """
    리뷰 프롬프트도 한국어로 작성하여, Claude 포함 모든 모델이 한국어로 평가하도록 유도.
    """
    lines = []
    lines.append("당신은 여러 LLM이 작성한 답변을 평가하는 심사위원입니다.")
    lines.append("각 답변은 서로 다른 모델이 작성했지만, 누구의 답변인지는 모른다고 가정합니다.")
    lines.append("")
    lines.append("해야 할 일:")
    lines.append("1. 주어진 답변들을 '정확성'과 '통찰력' 기준으로 가장 좋은 답변부터 가장 아쉬운 답변까지 순위를 매기십시오.")
    lines.append("2. 각 답변에 대해 한국어로 간단한 코멘트를 남기십시오.")
    lines.append("   - 장점: 무엇이 좋았는지")
    lines.append("   - 단점: 무엇이 부족했는지")
    lines.append("")
    lines.append("반드시 JSON 형식으로만 응답해야 합니다. 설명, 말머리, 마크다운, 코드 블록 등은 절대 넣지 마세요.")
    lines.append("")
    lines.append(f"사용자 질문:\n{user_query}\n")
    lines.append("답변 목록:")

    for idx, ans in enumerate(all_answers):
        label = f"Answer {idx+1}"
        lines.append(f"--- {label} ---")
        lines.append(ans.content)
        lines.append("")

    lines.append(
        "이제 아래 JSON 형식 그대로만 응답하세요:\n"
        "{\n"
        '  \"ranking\": [1, 2, 3, 4, 5],\n'
        '  \"comments\": {\n'
        '     \"1\": \"Answer 1에 대한 한국어 코멘트\",\n'
        '     \"2\": \"Answer 2에 대한 한국어 코멘트\",\n'
        '     \"3\": \"Answer 3에 대한 한국어 코멘트\",\n'
        '     \"4\": \"Answer 4에 대한 한국어 코멘트\",\n'
        '     \"5\": \"Answer 5에 대한 한국어 코멘트\"\n'
        "  }\n"
        "}\n"
        "JSON 앞뒤에 다른 문장을 절대 붙이지 마십시오."
    )
    return "\n".join(lines)

def extract_json_block(text: str) -> str | None:
    """
    Claude 등이 JSON 앞뒤에 말을 붙였을 때,
    가장 바깥쪽 { ... } 블록만 잘라내기 위한 간단한 헬퍼.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start:end+1]

async def stage2_review(
    session: aiohttp.ClientSession,
    user_query: str,
    all_answers: List[ModelAnswer],
) -> List[ReviewResult]:
    print("\n[Stage 2] 상호 리뷰 및 랭킹 생성 시작...")

    tasks = []
    for provider, name in COUNCIL_MODELS:
        prompt = build_review_prompt(user_query, all_answers)

        # Anthropic에는 JSON-only를 더 강하게 지시
        if provider == "anthropic":
            system_content = (
                "당신은 답변의 품질을 평가하는 심사위원입니다. "
                "반드시 유효한 JSON만 출력해야 하며, JSON 앞뒤에 다른 글자를 절대 넣지 마십시오. "
                "설명 문장, 마크다운, 코드 블록, 자연어는 모두 금지입니다. 오직 JSON만 출력하십시오."
            )
        else:
            system_content = (
                "당신은 답변의 품질을 평가하는 심사위원입니다. "
                "정확성과 통찰력을 기준으로 답변을 평가하고 한국어로 코멘트를 작성하세요."
            )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": prompt},
        ]
        tasks.append(call_chat(session, provider, name, messages))

    raw_reviews = await asyncio.gather(*tasks)

    results: List[ReviewResult] = []

    for (provider, name), review_text in zip(COUNCIL_MODELS, raw_reviews):
        reviewer_id = model_id(provider, name)

        try:
            raw_to_parse = review_text
            if provider == "anthropic":
                json_only = extract_json_block(review_text)
                if json_only is not None:
                    raw_to_parse = json_only
            data = json.loads(raw_to_parse)
        except json.JSONDecodeError:
            data = {
                "ranking": list(range(1, len(all_answers) + 1)),
                "comments": {
                    str(i+1): "JSON 파싱에 실패하여 구조화된 코멘트를 생성하지 못했습니다."
                    for i in range(len(all_answers))
                },
            }

        ranking_ids: List[str] = []
        for idx in data.get("ranking", []):
            if 1 <= idx <= len(all_answers):
                ranking_ids.append(answer_id(all_answers[idx - 1]))

        comments_by_id: Dict[str, str] = {}
        for idx_str, comment in data.get("comments", {}).items():
            try:
                idx = int(idx_str)
            except ValueError:
                continue
            if 1 <= idx <= len(all_answers):
                comments_by_id[answer_id(all_answers[idx - 1])] = comment

        results.append(
            ReviewResult(
                reviewer=reviewer_id,
                ranking=ranking_ids,
                comments=comments_by_id,
            )
        )

    # 콘솔 출력 (한국어 코멘트)
    for r in results:
        print(f"\n--- 리뷰어: {r.reviewer} ---")
        print("Ranking (best -> worst):", " > ".join(r.ranking))
        for target, comment in r.comments.items():
            print(f"- {target}: {comment}")

    return results

# ---------------------------------------------------------
# 9. Stage 3: Chairman 최종 답변 (한국어로 종합)
# ---------------------------------------------------------

def build_chairman_prompt(
    user_query: str,
    answers: List[ModelAnswer],
    reviews: List[ReviewResult],
) -> str:
    lines = []
    lines.append("당신은 여러 LLM이 토론한 결과를 종합하는 '의장(Chairman)' 모델입니다.")
    lines.append("아래 정보를 바탕으로, 사용자 질문에 대한 최선의 최종 답변을 한국어로 작성해야 합니다.")
    lines.append("")
    lines.append("당신이 받게 될 정보:")
    lines.append("1) 원래 사용자 질문")
    lines.append("2) 여러 모델이 개별적으로 생성한 답변")
    lines.append("3) 각 모델이 서로의 답변을 평가한 랭킹과 코멘트")
    lines.append("")
    lines.append("요구사항:")
    lines.append("- 여러 답변의 장점을 종합하고, 리뷰에서 지적된 약점을 보완하십시오.")
    lines.append("- 이론적 설명과 실제적인 조언을 균형 있게 포함하십시오.")
    lines.append("- 감정적으로 힘든 상태에 있는 사람에게도 도움이 될 수 있도록 공감적으로 표현하십시오.")
    lines.append("- 최종 답변은 오직 한국어로 작성하십시오.")
    lines.append("")
    lines.append(f"사용자 질문:\n{user_query}\n")

    lines.append("모델별 답변:")
    for ans in answers:
        lines.append(f"--- {answer_id(ans)} 의 답변 ---")
        lines.append(ans.content)
        lines.append("")

    lines.append("상호 리뷰 결과:")
    for r in reviews:
        lines.append(f"리뷰어: {r.reviewer}")
        lines.append(f"랭킹 (좋은 순): {', '.join(r.ranking)}")
        for target_id, comment in r.comments.items():
            lines.append(f"- {target_id} 에 대한 코멘트: {comment}")
        lines.append("")

    lines.append("위 정보를 모두 참고하여, 이제 사용자 질문에 대한 최종 답변을 한국어로 작성해 주세요.")
    return "\n".join(lines)

async def stage3_chairman(
    session: aiohttp.ClientSession,
    user_query: str,
    answers: List[ModelAnswer],
    reviews: List[ReviewResult],
) -> str:
    print("\n[Stage 3] Chairman 최종 답변 생성 시작...")

    prompt = build_chairman_prompt(user_query, answers, reviews)
    messages = [
        {
            "role": "system",
            "content": (
                "당신은 여러 LLM의 답변을 종합하는 한국어 전문가 의장입니다. "
                "항상 한국어로만 답변하십시오."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    provider, name = CHAIRMAN_MODEL
    final_answer = await call_chat(session, provider, name, messages)
    print("\n===== 최종 답변 (Chairman, 한국어) =====")
    print(final_answer)
    return final_answer

# ---------------------------------------------------------
# 10. main + 엔트리포인트
# ---------------------------------------------------------

async def main():
    print("=== LLM Council (한국어) 스크립트 진입 ===")
    user_query = input("질문을 입력하세요: ")

    async with aiohttp.ClientSession() as session:
        answers = await stage1_first_opinions(session, user_query)
        reviews = await stage2_review(session, user_query, answers)
        await stage3_chairman(session, user_query, answers, reviews)

if __name__ == "__main__":
    import traceback
    try:
        print("=== 스크립트 시작 ===")
        asyncio.run(main())
        print("=== 스크립트 정상 종료 ===")
    except Exception:
        print("=== 예외 발생 ===")
        traceback.print_exc()
        input("엔터를 누르면 창이 닫힙니다...")
