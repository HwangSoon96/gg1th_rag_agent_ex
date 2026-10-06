from langchain_openai import ChatOpenAI, embeddings

import os
import warnings
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore

load_dotenv(override=True)

api_key = os.getenv("LLM_API_KEY")

BASE_URL = "https://monogpt.kr/api/monorouter/v1"

# MonoRouter 분당 30회 제한(429) 대응: SDK 재시도는 회당 최대 8초 대기 → 10회면 약 40~55초 버팀
MAX_RETRIES = 10

# 레포 루트의 .cache/ 에 LLM 응답·임베딩 캐시 저장 (모든 노트북 공유, .gitignore 대상)
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
# SQLiteCache가 캐시 히트마다 내는 allowed_objects 경고 숨김 (직접 만든 로컬 캐시라 신뢰 가능, 경고를 끌 옵션이 없음)
warnings.filterwarnings("ignore", message="The default value of `allowed_objects`")
set_llm_cache(SQLiteCache(database_path=str(CACHE_DIR / "llm_cache.db")))

def llm_connect(
    # model="claude-haiku-4.5"
    # model="gemini-3.5-flash"
    model: str = "gpt-5.4-mini",
    api_key: str = api_key,
    temperature: float = 0,
    max_tokens: int = 2086,
    cache: bool = True,  # 매번 다른 답이 필요한 실습(temperature>0, 평가, 스트리밍 데모)은 False
):
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=BASE_URL,
        temperature=temperature,
        use_responses_api=False,  # base url로 할 때는 이부분 넣어야 함.(MonoRouter 사용)
        max_tokens=max_tokens,
        max_retries=MAX_RETRIES,
        cache=cache,
    )

from langchain_openai import OpenAIEmbeddings


def get_embeddings():
    embedding_model = "text-embedding-3-small"
    embeddings = OpenAIEmbeddings(
        api_key=api_key,
        model=embedding_model,
        base_url=BASE_URL,
        max_retries=MAX_RETRIES,
        )
    # 문서·질문 임베딩 캐시 (namespace=모델명이라 모델을 바꿔도 섞이지 않음)
    return CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        LocalFileStore(str(CACHE_DIR / "embeddings")),
        namespace=embedding_model,
        query_embedding_cache=True,
        key_encoder="sha256",
    )
