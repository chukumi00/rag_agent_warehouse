########################################################
# 창고 재고 및 매뉴얼 관리 웹 애플리케이션용 유틸리티 모듈
########################################################

import json
import os
import shutil

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


# 데이터 파일과 캐시 경로를 현재 스크립트 위치 기준으로 계산해 실행 위치에 덜 의존하게 한다.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
ENV_PATH = os.path.join(PROJECT_DIR, ".env")
ORIGIN_DIR = os.path.join(APP_DIR, "origin")
EMBEDDING_DIR = os.path.join(APP_DIR, "embedding")

INVENTORY_CSV = os.path.join(ORIGIN_DIR, "warehouse_inventory.csv")  # 원본 재고 CSV 파일
MANUAL_CSV = os.path.join(ORIGIN_DIR, "warehouse_manual.csv")           # 원본 매뉴얼 CSV 파일
QUESTIONS_FILE = os.path.join(ORIGIN_DIR, "warehouse_questions.txt")    # 원본 질문 예시 파일
INVENTORY_CACHE_CSV = os.path.join(EMBEDDING_DIR, "warehouse_inventory.csv")  # 재고 캐시 CSV 파일
MANUAL_CACHE_CSV = os.path.join(EMBEDDING_DIR, "warehouse_manual.csv")  # 매뉴얼 캐시 CSV 파일
MANUAL_CACHE_NPY = os.path.join(EMBEDDING_DIR, "warehouse_manual.npy")  # 매뉴얼 임베딩 캐시 파일
QUESTIONS_CACHE_FILE = os.path.join(EMBEDDING_DIR, "warehouse_questions.txt")  # 질문 예시 캐시 파일

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
# 이 값보다 낮은 유사도의 매뉴얼 검색 결과는 답변 근거에서 제외한다.
LOW_SIMILARITY_THRESHOLD = 0.25
# 에이전트가 반복적으로 도구를 호출하는 것을 막는 상한이다.
MAX_TOOL_ROUNDS = 5

# .env에 키가 없더라도 앱의 재고 표시는 가능하고, AI 호출 시 명시적으로 오류를 보여준다.
load_dotenv(ENV_PATH)
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY) if API_KEY else None

# OpenAI 클라이언트 초기화 및 API 키 확인 함수
def get_client() -> OpenAI:
    """AI 기능을 호출할 때 API 키 설정 여부를 검사하고 OpenAI 클라이언트를 반환한다."""
    if client is None:
        raise RuntimeError(
            f"OPENAI_API_KEY를 설정해야 AI 기능을 사용할 수 있습니다. 설정 파일: {ENV_PATH}"
        )
    return client

# 원본 및 임베딩 캐시 디렉터리 준비 함수
def ensure_directories() -> None:
    """원본 및 임베딩 캐시를 저장할 디렉터리를 준비한다."""
    os.makedirs(ORIGIN_DIR, exist_ok=True)
    os.makedirs(EMBEDDING_DIR, exist_ok=True)

# 캐시 파일 준비 함수
def ensure_file_cache(source_path: str, cache_path: str, refresh: bool = False) -> str:
    """캐시 파일이 없거나 갱신 요청이 있으면 원본을 복사해 캐시 경로를 반환한다."""
    ensure_directories()
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"원본 파일을 찾을 수 없습니다: {source_path}")
    if refresh or not os.path.exists(cache_path):
        shutil.copy2(source_path, cache_path)
    return cache_path


# 재고 조회 도구 (Inventory Tool)
def load_inventory() -> pd.DataFrame:
    """재고 CSV를 읽고 필수 열과 수량 데이터 형식을 확인한다."""
    cache_path = ensure_file_cache(INVENTORY_CSV, INVENTORY_CACHE_CSV)
    # 캐시 파일 수정 시각이 변경되면 Streamlit의 메모리 캐시도 새로 읽는다.
    modified_at = os.path.getmtime(cache_path)
    return _read_inventory_csv(cache_path, modified_at)


@st.cache_data(show_spinner=False)
def _read_inventory_csv(path: str, modified_at: float) -> pd.DataFrame:
    """재고 캐시 CSV를 읽어 검증하고, 동일 파일 버전은 메모리에 재사용한다."""
    del modified_at  # 파일 변경 시 캐시 키도 바뀌도록 사용하는 값이다.
    inventory = pd.read_csv(path)
    required = {"item_id", "item_name", "location", "quantity", "unit", "reorder_point"}
    missing = required.difference(inventory.columns)
    if missing:
        raise ValueError(f"재고 데이터에 필수 컬럼이 없습니다: {', '.join(sorted(missing))}")

    inventory["quantity"] = pd.to_numeric(inventory["quantity"], errors="raise")
    inventory["reorder_point"] = pd.to_numeric(inventory["reorder_point"], errors="raise")
    return inventory

# 추천 질문 도구 (Suggested Questions Tool)
def load_suggested_questions() -> list[str]:
    """추천 질문 캐시 파일에서 비어 있지 않은 줄만 읽어 UI 목록으로 반환한다."""
    cache_path = ensure_file_cache(QUESTIONS_FILE, QUESTIONS_CACHE_FILE)
    modified_at = os.path.getmtime(cache_path)
    return _read_suggested_questions(cache_path, modified_at)

# 추천 질문 캐시 읽기 내부 함수
@st.cache_data(show_spinner=False)
def _read_suggested_questions(path: str, modified_at: float) -> list[str]:
    """질문 파일을 읽고 같은 파일 버전은 Streamlit 메모리 캐시에서 재사용한다."""
    del modified_at  # 파일 변경 시 캐시 키도 바뀌도록 사용하는 값이다.
    with open(path, encoding="utf-8") as questions_file:
        questions = [line.strip() for line in questions_file if line.strip()]
    if not questions:
        raise ValueError("질문 예시 파일에 질문이 없습니다.")
    return questions

# 추천 질문 선택 함수
def select_suggested_question(question: str) -> None:
    """추천 질문 버튼의 선택값을 질문 입력 위젯에 반영한다."""
    st.session_state["warehouse_question"] = question

# 임베딩 생성 도구
def get_embedding(text: str) -> np.ndarray:
    """텍스트의 OpenAI 임베딩을 유사도 계산용 float32 배열로 변환한다."""
    response = get_client().embeddings.create(model=EMBEDDING_MODEL, input=text)
    return np.asarray(response.data[0].embedding, dtype=np.float32)

# 코사인 유사도 계산 도구
def cosine_similarity_matrix(embeddings: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
    """문서 임베딩 행렬과 질문 벡터 간 코사인 유사도를 계산한다."""
    query_norm = np.linalg.norm(query_vector) + 1e-12
    embedding_norms = np.linalg.norm(embeddings, axis=1) + 1e-12
    return (embeddings @ query_vector) / (embedding_norms * query_norm)

# 창고 매뉴얼 로드 및 임베딩 생성 도구 (Manual Embedding Tool)
def load_manual() -> pd.DataFrame:
    """창고 매뉴얼을 읽고 검색 불가한 빈 문서를 제거한다."""
    if not os.path.exists(MANUAL_CSV):
        raise FileNotFoundError(f"창고 운영 매뉴얼 파일을 찾을 수 없습니다: {MANUAL_CSV}")

    manual = pd.read_csv(MANUAL_CSV)
    if not {"category", "content"}.issubset(manual.columns):
        raise ValueError("운영 매뉴얼에는 'category'와 'content' 컬럼이 필요합니다.")
    manual = manual.dropna(subset=["content"]).copy()
    manual["content"] = manual["content"].astype(str).str.strip()
    manual = manual[manual["content"] != ""].reset_index(drop=True)
    if manual.empty:
        raise ValueError("운영 매뉴얼에 검색할 문서가 없습니다.")
    return manual

# 매뉴얼 임베딩 생성 함수
def build_manual_embeddings(manual: pd.DataFrame) -> np.ndarray:
    """매뉴얼 각 문서의 임베딩을 만들고 진행률을 Streamlit에 표시한다."""
    progress = st.progress(0, text="운영 매뉴얼 임베딩 준비 중...")
    vectors = []
    total = len(manual)

    for index, text in enumerate(manual["content"].tolist(), start=1):
        vectors.append(get_embedding(text))
        percent = int(index / total * 100)
        progress.progress(percent, text=f"임베딩 생성 중... {index}/{total} ({percent}%)")

    progress.empty()
    return np.vstack(vectors).astype(np.float32)

# 매뉴얼 인덱스 로드 및 재생성 도구
def load_manual_index() -> tuple[pd.DataFrame, np.ndarray]:
    """완전한 캐시가 있으면 재사용하고, 없으면 원본 매뉴얼로 인덱스를 생성한다."""
    ensure_directories()
    if os.path.exists(MANUAL_CACHE_CSV) and os.path.exists(MANUAL_CACHE_NPY):
        manual = pd.read_csv(MANUAL_CACHE_CSV)
        embeddings = np.load(MANUAL_CACHE_NPY)
        # 문서 행과 임베딩 행이 어긋나면 검색 결과를 잘못 연결할 수 있어 로드 단계에서 차단한다.
        if "content" not in manual.columns or embeddings.ndim != 2:
            raise ValueError("운영 매뉴얼 임베딩 캐시 형식이 올바르지 않습니다.")
        if embeddings.shape[0] != len(manual):
            raise ValueError("운영 매뉴얼 캐시의 문서 수와 임베딩 수가 일치하지 않습니다.")
        return manual, embeddings

    manual = load_manual()
    embeddings = build_manual_embeddings(manual)
    manual.to_csv(MANUAL_CACHE_CSV, index=False)
    np.save(MANUAL_CACHE_NPY, embeddings)
    return manual, embeddings

# 매뉴얼 인덱스 재생성 도구
def regenerate_manual_index() -> tuple[pd.DataFrame, np.ndarray]:
    """원본 매뉴얼을 다시 임베딩하고 CSV/NPY 캐시를 최신 데이터로 덮어쓴다."""
    manual = load_manual()
    embeddings = build_manual_embeddings(manual)
    manual.to_csv(MANUAL_CACHE_CSV, index=False)
    np.save(MANUAL_CACHE_NPY, embeddings)
    return manual, embeddings

# 창고 운영 매뉴얼 검색 도구 (Manual Search Tool)
def search_manual(
    manual: pd.DataFrame, embeddings: np.ndarray, query: str, top_k: int
) -> pd.DataFrame:
    """질문과 유사한 매뉴얼을 점수 내림차순으로 상위 K개 반환한다."""
    query_vector = get_embedding(query)
    similarities = cosine_similarity_matrix(embeddings, query_vector)
    results = manual.copy()
    results["similarity"] = similarities
    return results.sort_values("similarity", ascending=False).head(top_k)

# 재고 조회 도구
def lookup_inventory(query: str) -> dict:
    """품목 ID, 품목명 또는 로케이션으로 현재 재고를 검색한다(읽기 전용)."""
    inventory = load_inventory()
    normalized_query = query.strip().casefold()
    # 전체 조회 표현은 필터링하지 않고, 그 외 입력은 세 검색 열에서 부분 일치로 찾는다.
    if normalized_query not in {"", "*", "전체", "모두", "all"}:
        searchable = (
            inventory["item_id"].astype(str).str.casefold().str.contains(normalized_query, regex=False)
            | inventory["item_name"].astype(str).str.casefold().str.contains(normalized_query, regex=False)
            | inventory["location"].astype(str).str.casefold().str.contains(normalized_query, regex=False)
        )
        inventory = inventory[searchable]

    # 도구 응답을 JSON으로 직렬화할 수 있도록 NaN 값을 None으로 바꾸고 결과 수를 제한한다.
    records = inventory.astype(object).where(pd.notna(inventory), None).to_dict("records")
    return {"match_count": len(records), "items": records[:50]}


# 재고 조회 도구
def get_inventory_summary() -> dict:
    """전체 품목 수, 단위별 합계와 재주문 기준 이하 품목을 요약한다."""
    inventory = load_inventory()
    low_stock = inventory[inventory["quantity"] <= inventory["reorder_point"]]
    quantities_by_unit = (
        inventory.groupby("unit", dropna=False)["quantity"].sum().to_dict()
    )
    return {
        "sku_count": int(inventory["item_id"].nunique()),
        "low_stock_count": int(len(low_stock)),
        "quantities_by_unit": quantities_by_unit,
        "low_stock_items": low_stock.astype(object)
        .where(pd.notna(low_stock), None)
        .to_dict("records"),
    }


# 창고 운영 절차 검색 도구
def search_warehouse_procedures(
    query: str, manual: pd.DataFrame, embeddings: np.ndarray, top_k: int
) -> dict:
    """매뉴얼을 RAG 검색한 뒤 유사도가 기준 이상인 근거만 도구 결과로 반환한다."""
    hits = search_manual(manual, embeddings, query, top_k)
    hits = hits[hits["similarity"] >= LOW_SIMILARITY_THRESHOLD]
    results = hits[["category", "content", "similarity"]].copy()
    results["similarity"] = results["similarity"].astype(float).round(3)
    return {"match_count": len(results), "procedures": results.to_dict("records")}


# LLM에 제공하는 함수 정의이자 호출 허용 목록이다. 모든 함수는 재고나 원본 파일을 수정하지 않는다.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_inventory",
            "description": "현재 재고를 품목명, 품목 ID 또는 창고 위치로 조회합니다. 재고 변경은 하지 않습니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색할 품목명, 품목 ID 또는 위치. 전체 조회는 '전체'.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_inventory_summary",
            "description": "품목 수, 단위별 재고량, 재주문 기준 이하의 저재고 품목을 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_warehouse_procedures",
            "description": "입고, 적치, 피킹, 출고, 안전 및 재고 실사 관련 창고 운영 절차를 RAG로 검색합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "검색할 창고 운영 질문"}
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]

########################################################
# 질문에서 언급된 재고 찾기 도구 (Mentioned Inventory Finder)
def find_mentioned_inventory(question: str) -> dict | None:
    """질문에 품목 ID, 품목명 또는 품목명의 단어가 포함된 재고를 찾는다(없으면 None)."""
    inventory = load_inventory()
    normalized_question = question.casefold()

    def is_mentioned(row: pd.Series) -> bool:
        item_id = str(row["item_id"]).casefold()
        item_name = str(row["item_name"]).casefold()
        if item_id in normalized_question or item_name in normalized_question:
            return True
        # 품목명을 단어로 나눠 '라벨지'처럼 일부만 언급된 경우도 찾는다.
        return any(
            len(word) >= 2 and word in normalized_question for word in item_name.split()
        )

    matched = inventory[inventory.apply(is_mentioned, axis=1)]
    if matched.empty:
        return None
    records = matched.astype(object).where(pd.notna(matched), None).to_dict("records")
    return {"match_count": len(records), "items": records[:50]}


# agent runner
def run_agent(
    question: str,
    manual: pd.DataFrame,
    embeddings: np.ndarray,
    top_k: int,
) -> tuple[str, list[dict]]:
    """질문을 LLM에 전달하고, 요청된 조회/RAG 도구를 실행해 답변과 근거를 반환한다."""
    messages = [
        {
            "role": "system",
            "content": (
                "당신은 창고 운영관리 보조 에이전트입니다. 사용자의 요청에 맞춰 "
                "재고 조회 도구 및 운영 절차 RAG 검색 도구를 사용하세요. "
                "질문에 특정 품목명, 품목 ID, 위치, 재고 부족/수량이 언급되면 "
                "가정형 표현(예: '부족하면')이라도 먼저 재고 도구로 현재 상태를 확인하고, "
                "절차나 대응 방법을 묻는 내용이 있으면 운영 절차 검색도 함께 호출하세요. "
                "한 가지 도구만으로 답할 수 있는 질문에는 필요한 도구만 호출하세요. "
                "재고 수량은 조회 결과만 근거로 답하고, 절차는 검색된 매뉴얼 내용만 "
                "근거로 답하세요. 근거가 없거나 검색 결과가 없으면 확인할 수 없다고 "
                "분명히 말하세요. 입고/출고/재고 조정은 실행하거나 실행했다고 말하지 마세요. "
                "사용 가능한 도구는 읽기 전용입니다. 한국어로 간결하게 답하세요."
            ),
        },
        {"role": "user", "content": question},
    ]
    tool_results = []

    # 모델의 도구 선택에만 의존하지 않고, 질문 원문 기준의 재고/매뉴얼 조회를 코드에서 먼저 수행한다.
    # 모델이 만든 검색어는 매뉴얼 문장과 유사도가 낮을 수 있어 원문 질문 검색을 기본 근거로 둔다.
    prefetched = []
    mentioned = find_mentioned_inventory(question) # 질문에서 언급된 재고 항목을 찾는다.
    if mentioned: # 언급된 재고 항목이 있으면 prefetched에 추가한다.
        prefetched.append(("lookup_inventory", {"query": question}, mentioned)) # prefetched에 재고 조회 결과 추가
    procedures = search_warehouse_procedures(question, manual, embeddings, top_k) # 질문에서 관련된 운영 절차를 검색한다.
    if procedures["match_count"] > 0:
        prefetched.append(("search_warehouse_procedures", {"query": question}, procedures))

    if prefetched: # prefetched에 결과가 있으면 메시지에 추가한다.
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"prefetch_{index}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(arguments, ensure_ascii=False),
                        },
                    }
                    for index, (name, arguments, _) in enumerate(prefetched)
                ],
            }
        )
        for index, (name, arguments, result) in enumerate(prefetched): # prefetched 결과를 tool_results와 메시지에 추가한다.
            tool_results.append({"tool": name, "arguments": arguments, "result": result})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"prefetch_{index}",
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                }
            )

    # 도구 호출이 필요한 동안 결과를 대화에 추가하고, 최종 답변이 나올 때까지 반복한다.
    for _ in range(MAX_TOOL_ROUNDS):
        response = get_client().chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        message = response.choices[0].message
        if not message.tool_calls:
            if not message.content:
                raise RuntimeError("AI 에이전트가 빈 답변을 반환했습니다.")
            return message.content, tool_results

        # 도구 호출 요청도 대화 이력에 포함해야 모델이 도구 실행 결과를 이어서 해석할 수 있다.
        messages.append(message.model_dump(exclude_none=True))
        for tool_call in message.tool_calls:
            arguments = json.loads(tool_call.function.arguments)
            tool_name = tool_call.function.name
            # 모델이 요청한 도구명만 아래 허용 목록에서 실행하고, 알 수 없는 도구는 거부한다.
            if tool_name == "lookup_inventory": # 재고 조회 도구
                result = lookup_inventory(arguments["query"])
            elif tool_name == "get_inventory_summary": # 재고 요약 도구
                result = get_inventory_summary()
            elif tool_name == "search_warehouse_procedures": # 창고 운영 절차 검색 도구
                result = search_warehouse_procedures(
                    arguments["query"], manual, embeddings, top_k
                )
            else:
                raise ValueError(f"허용되지 않은 에이전트 도구입니다: {tool_name}")

            tool_results.append(
                {"tool": tool_name, "arguments": arguments, "result": result}
            )
            # tool_call_id로 해당 요청과 결과를 연결해 다음 모델 응답에 제공한다.
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                }
            )

    raise RuntimeError(f"에이전트가 최대 도구 호출 횟수({MAX_TOOL_ROUNDS})를 초과했습니다.")


# Streamlit 페이지 초기 설정 및 제목
st.set_page_config(page_title="AI 스마트 물류창고 운영관리", layout="wide")
st.title("AI 스마트 물류창고 운영관리")
st.caption("재고 CSV 조회 + 창고 운영 매뉴얼 RAG + 읽기 전용 Agent 데모")

with st.sidebar:
    st.subheader("에이전트 설정")
    top_k = st.slider("매뉴얼 검색 Top-K", min_value=1, max_value=5, value=3)
    st.caption("현재 MVP는 조회와 절차 안내만 지원하며 재고 데이터를 변경하지 않습니다.")
    st.divider()
    st.subheader("데이터 캐시")
    # 매뉴얼 임베딩(API 호출)을 먼저 처리해, 실패 시 재고/질문 캐시는 기존 상태로 유지한다.
    if st.button("캐시 전체 갱신", type="primary"):
        try:
            with st.spinner("매뉴얼 임베딩과 재고/질문 캐시를 갱신하는 중입니다..."):
                manual_df, manual_embeddings = regenerate_manual_index()
                ensure_file_cache(INVENTORY_CSV, INVENTORY_CACHE_CSV, refresh=True)
                ensure_file_cache(QUESTIONS_FILE, QUESTIONS_CACHE_FILE, refresh=True)
            st.session_state["manual_index"] = (manual_df, manual_embeddings)
            _read_inventory_csv.clear()
            _read_suggested_questions.clear()
            st.success("매뉴얼 임베딩, 재고, 추천 질문 캐시를 원본에서 갱신했습니다.")
        except Exception as exc:
            st.error(f"캐시 갱신 실패: {type(exc).__name__}: {exc}")
    st.caption("원본: rag/origin/ → 캐시: rag/embedding/")
    st.divider()
    st.subheader("추천 질문")
    # 버튼을 누르면 선택된 추천 문구가 아래 질문 입력란에 채워진다.
    try:
        suggested_questions = load_suggested_questions()
        for index, suggested_question in enumerate(suggested_questions):
            st.button(
                suggested_question,
                key=f"suggested_question_{index}",
                on_click=select_suggested_question,
                args=(suggested_question,),
                use_container_width=True,
            )
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"추천 질문을 불러오지 못했습니다: {exc}")

try:
    # Streamlit rerun마다 임베딩 API를 다시 호출하지 않도록 세션 상태에 인덱스를 보관한다.
    if "manual_index" not in st.session_state:
        with st.spinner("창고 운영 매뉴얼을 준비하는 중입니다..."):
            st.session_state["manual_index"] = load_manual_index()
    manual_df, manual_embeddings = st.session_state["manual_index"]
    inventory_df = load_inventory()
except Exception as exc:
    st.error(f"데이터 준비 실패: {type(exc).__name__}: {exc}")
    st.stop()

# 상단 지표와 상세 표는 CSV에서 읽은 현재 데모 재고 데이터를 그대로 보여준다.
low_stock_count = int((inventory_df["quantity"] <= inventory_df["reorder_point"]).sum())
metric_columns = st.columns(3)
metric_columns[0].metric("등록 품목(SKU)", int(inventory_df["item_id"].nunique()))
metric_columns[1].metric("재고 위치", int(inventory_df["location"].nunique()))
metric_columns[2].metric("재주문 기준 이하", low_stock_count)

with st.expander("현재 재고 데이터", expanded=False):
    st.dataframe(inventory_df, use_container_width=True, hide_index=True)

# 추천 질문 선택 및 직접 입력이 같은 세션 상태 키를 공유한다.
question = st.text_input(
    "창고 운영에 대해 질문하세요",
    placeholder="예: 안전재고 이하 품목과 현재 수량을 알려줘",
    key="warehouse_question",
)
ask = st.button("질문하기", type="primary", disabled=not question.strip())

if ask:
    try:
        # 에이전트가 질문 종류에 따라 재고 조회, 요약 또는 매뉴얼 검색 도구를 선택한다.
        with st.spinner("재고 및 운영 절차를 확인하고 있습니다..."):
            answer, tool_results = run_agent(
                question, manual_df, manual_embeddings, top_k
            )
        st.subheader("에이전트 답변")
        st.write(answer)

        # 답변이 어떤 도구 조회 결과를 바탕으로 작성됐는지 사용자가 확인할 수 있게 표시한다.
        if tool_results:
            with st.expander("조회 근거", expanded=True):
                for index, item in enumerate(tool_results, start=1):
                    st.markdown(f"**{index}. {item['tool']}**")
                    result = item["result"]
                    if "procedures" in result:
                        if result["procedures"]:
                            st.dataframe(
                                pd.DataFrame(result["procedures"]),
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            st.info("유사도 기준을 충족하는 매뉴얼 절차가 없습니다.")
                    elif "items" in result:
                        if result["items"]:
                            st.dataframe(
                                pd.DataFrame(result["items"]),
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            st.info("조건에 맞는 재고 품목이 없습니다.")
                    else:
                        st.json(result)
    except Exception as exc:
        st.error(f"질문 처리 실패: {type(exc).__name__}: {exc}")
