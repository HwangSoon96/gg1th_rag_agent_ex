from langchain_openai import ChatOpenAI, embeddings

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

api_key = os.getenv("LLM_API_KEY")

BASE_URL = "https://monogpt.kr/api/monorouter/v1"

def llm_connect(
    # model="claude-haiku-4.5"
    # model="gemini-3.5-flash"
    model: str = "gpt-5.4-mini",
    api_key: str = api_key,
    temperature: float = 0,
    max_tokens: int = 2086
):
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=BASE_URL,
        temperature=temperature,
        use_responses_api=False,  # base url로 할 때는 이부분 넣어야 함.(MonoRouter 사용)
        max_tokens=max_tokens,
    )

from langchain_openai import OpenAIEmbeddings


def get_embeddings():
    embedding_model = "text-embedding-3-small"
    embeddings = OpenAIEmbeddings(
        api_key=api_key,
        model=embedding_model,
        base_url=BASE_URL,
        )
    return embeddings