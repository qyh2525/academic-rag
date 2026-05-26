"""app package init.

在这里统一加载 .env 并设置 HuggingFace 镜像,确保任何 `import app.xxx`
都会自动配置 HF_ENDPOINT,避免子模块各自处理时机不一致导致镜像没生效。
"""
import os
from dotenv import load_dotenv

load_dotenv()

# huggingface_hub / sentence_transformers / transformers 都认这个变量
# 必须在 huggingface_hub 被首次 import 之前设置,所以放在 package init 最早处
_hf_endpoint = os.getenv("HF_ENDPOINT")
if _hf_endpoint:
    os.environ["HF_ENDPOINT"] = _hf_endpoint
