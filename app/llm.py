"""DeepSeek 客户端封装。

DeepSeek 走 OpenAI 兼容协议,所以用 langchain_openai.ChatOpenAI 即可,
后续要换成通义千问/月之暗面/本地 Ollama 等只改 .env,无需改业务代码。
"""
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


def get_llm(temperature: float = 0.3, streaming: bool = False) -> ChatOpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or api_key == "your_deepseek_api_key_here":
        raise RuntimeError(
            "DEEPSEEK_API_KEY 未配置,请在 .env 文件中填入,"
            "申请地址:https://platform.deepseek.com/"
        )
    return ChatOpenAI(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        temperature=temperature,
        streaming=streaming,
        timeout=60,
    )
