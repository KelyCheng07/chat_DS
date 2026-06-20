"""
ChatDeepSeek CLI · 全局配置
"""

import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


# ---- API ----
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# ---- 窗口与上下文 ----
MAX_WINDOW = _env_int("MAX_WINDOW", 10)
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "deepseek-chat")
SIMILARITY_THRESHOLD = _env_float("SIMILARITY_THRESHOLD", 0.5)
SEMANTIC_TOP_K = _env_int("SEMANTIC_TOP_K", 4)

# ---- 摘要压缩 ----
COMPRESSION_THRESHOLD_TOKENS = _env_int("COMPRESSION_THRESHOLD_TOKENS", 2000)
COMPRESSION_MIN_RATIO = _env_float("COMPRESSION_MIN_RATIO", 5.0)

# ---- 重试 ----
MAX_RETRIES = _env_int("MAX_RETRIES", 3)
RETRY_DELAY = _env_int("RETRY_DELAY", 2)

# ---- 模型价格（每百万 token，美元） ----
MODEL_PRICES = {
    "deepseek-chat": {"prompt": 0.14, "completion": 0.28},
    "deepseek-reasoner": {"prompt": 0.55, "completion": 2.19},
}

# ---- 输出控制 ----
MAX_OUTPUT_TOKENS = _env_int("MAX_OUTPUT_TOKENS", 4096)

# ---- 路径 ----
CHATS_DIR = os.path.join(os.path.dirname(__file__), "chats")
META_DIR = os.path.join(CHATS_DIR, ".meta")

# ---- 确保目录存在 ----
os.makedirs(CHATS_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)
