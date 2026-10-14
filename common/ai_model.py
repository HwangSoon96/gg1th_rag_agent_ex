from common.config import MODEL

# 기존 환경(1~3번)과 같은 MonoRouter 연결·캐시·재시도를 그대로 사용한다.
from common_config import LLM_CACHE, get_embeddings, llm_connect


def get_llm_model(
    model: str = MODEL,
    temperature: float = 0,
    max_tokens: int = 512,
    cache=LLM_CACHE,  # 노트북이 set_llm_cache(InMemoryCache())로 바꿔도 디스크 캐시를 써서 재실행 시 API를 다시 부르지 않는다.
):
    return llm_connect(model=model, temperature=temperature, max_tokens=max_tokens, cache=cache)


def get_embedding_model():
    return get_embeddings()
