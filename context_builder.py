"""
ChatDeepSeek CLI · 上下文组装与筛选管线

负责解析 @ 锚点、合并静态窗口与语义召回、
组装系统提示、执行 Token 溢出预防。
"""

import re
from typing import List, Tuple, Optional

from session_manager import Session, Message
from semantic_search import semantic_search, keyword_search
from config import (
    MAX_WINDOW,
    SIMILARITY_THRESHOLD,
    SEMANTIC_TOP_K,
    COMPRESSION_THRESHOLD_TOKENS,
    COMPRESSION_MIN_RATIO,
)

# tiktoken 是可选的，不可用时回退到字符估算
_tiktoken_enc = None


def _get_encoder():
    """获取 tiktoken 编码器（若可用）。"""
    global _tiktoken_enc
    if _tiktoken_enc is None:
        try:
            import tiktoken
            _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
        except (ImportError, Exception):
            _tiktoken_enc = False  # 标记为不可用
    return _tiktoken_enc if _tiktoken_enc is not False else None


# ---- Token 估算 ----
def estimate_tokens(text: str, model: str = "deepseek-chat") -> int:
    """估算文本的 token 数。优先使用 tiktoken，不可用时回退到字符估算。"""
    enc = _get_encoder()
    if enc:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    # 回退：粗略估算（中文 ~1 字/token，英文 ~3.5 字/token，取保守值）
    return max(1, len(text) // 2)


def estimate_message_tokens(msg: Message, model: str = "deepseek-chat") -> int:
    """估算单条消息的 token 数（含角色开销约 4 tokens）。"""
    return estimate_tokens(msg.content, model) + 4


def estimate_messages_tokens(messages: List[Message], model: str = "deepseek-chat") -> int:
    """估算消息列表的总 token 数。"""
    return sum(estimate_message_tokens(m, model) for m in messages)


# ---- @ 锚点解析 ----
def parse_at_anchors(text: str) -> Tuple[str, List[str], List[str]]:
    """
    解析用户输入中的 @ 锚点。

    Returns:
        (清理后的文本, 语义锚点列表, 精确关键词列表)
    """
    # 匹配 @"精确短语" 或 @关键词
    exact_pattern = re.compile(r'@"([^"]+)"')
    semantic_pattern = re.compile(r'(?<!")@([\w\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+)')

    exacts = exact_pattern.findall(text)
    cleaned = exact_pattern.sub("", text)

    semantics = semantic_pattern.findall(cleaned)
    cleaned = semantic_pattern.sub("", cleaned)

    cleaned = cleaned.strip()
    return cleaned, semantics, exacts


# ---- 上下文构建管线 ----
class ContextBuilder:
    """上下文构建器，封装完整的组装管线。"""

    def __init__(self, session: Session):
        self.session = session

    def build(
        self, user_input: str
    ) -> Tuple[List[dict], dict]:
        """
        构建发送给 API 的消息列表。

        Args:
            user_input: 用户原始输入（含 @ 锚点）。

        Returns:
            (messages 列表, 优化信息 dict)
        """
        optimization_info = {
            "window_count": 0,
            "semantic_count": 0,
            "keyword_count": 0,
            "summary_saved_tokens": 0,
            "truncated_count": 0,
            "at_not_found": [],
        }

        # 1. 解析 @ 锚点
        clean_input, semantics, exacts = parse_at_anchors(user_input)

        # 2. 收集消息索引
        all_messages = self.session.messages
        total = len(all_messages)

        # 3. 静态滑动窗口：最近 MAX_WINDOW 轮（每轮 user+assistant）
        window_start = max(0, total - MAX_WINDOW * 2)
        window_indices = set(range(window_start, total))

        # 4. 语义召回：对早于窗口的消息
        old_messages = [
            (i, all_messages[i].content)
            for i in range(0, window_start)
        ]

        semantic_indices: set = set()
        keyword_indices: set = set()
        not_found: List[str] = []

        # 语义召回
        if semantics and old_messages:
            query = " ".join(semantics)
            results = semantic_search(query, old_messages, threshold=SIMILARITY_THRESHOLD)
            for idx, _ in results:
                if idx not in window_indices:
                    semantic_indices.add(idx)
            optimization_info["semantic_count"] = len(semantic_indices)

        # 精确关键词召回
        if exacts and old_messages:
            results = keyword_search(exacts, old_messages, top_k=2)
            for idx, _ in results:
                if idx not in window_indices and idx not in semantic_indices:
                    keyword_indices.add(idx)
            optimization_info["keyword_count"] = len(keyword_indices)

        # @ 锚点未匹配提示
        # 规格：当 @xxx 未匹配到任何搜索结果时，提示用户
        all_found = semantic_indices | keyword_indices
        for anchor in semantics + exacts:
            matched = False
            for i in all_found:
                if anchor.lower() in all_messages[i].content.lower():
                    matched = True
                    break
            if not matched:
                not_found.append(anchor)

        # 去重：窗口消息优先
        keyword_indices -= window_indices
        keyword_indices -= semantic_indices
        optimization_info["at_not_found"] = not_found

        # 5. 合并消息索引（按原始顺序）
        all_indices = sorted(
            window_indices | semantic_indices | keyword_indices
        )

        optimization_info["window_count"] = len(window_indices)

        # 6. 构建消息列表
        selected_messages = [all_messages[i] for i in all_indices]

        # 7. 系统消息组装
        system_content = self._build_system_prompt()

        # 8. 构建 API 消息格式
        api_messages: List[dict] = []
        if system_content:
            api_messages.append({"role": "system", "content": system_content})

        for msg in selected_messages:
            api_messages.append({"role": msg.role, "content": msg.content})

        # 9. 添加当前用户输入
        api_messages.append({"role": "user", "content": clean_input})

        # 10. Token 溢出预防
        api_messages, truncated = self._prevent_overflow(api_messages)
        optimization_info["truncated_count"] = truncated

        return api_messages, optimization_info

    def _build_system_prompt(self) -> str:
        """按优先级组装系统提示。"""
        parts = []

        # 1. 核心角色（每轮必带）
        if self.session.core_role:
            parts.append(self.session.core_role)

        # 2. 知识摘要（优先摘要，无摘要则完整文档）
        # 注意：recall_doc_active 会强制注入完整文档
        if self.session.recall_doc_active and self.session.knowledge_doc:
            parts.append(f"[完整文档召回]\n{self.session.knowledge_doc}")
        elif self.session.knowledge_summary:
            parts.append(f"[知识要点]\n{self.session.knowledge_summary}")
        elif self.session.knowledge_doc:
            parts.append(self.session.knowledge_doc)

        # 3. 临时角色（仅当轮或持久）
        if self.session.temp_role:
            parts.append(f"[临时指令]\n{self.session.temp_role}")

        # 4. 远历史纪要（若开启摘要）
        if self.session.summary_enabled and self.session.history_summary:
            parts.append(f"[历史对话纪要]\n{self.session.history_summary}")

        return "\n\n".join(parts) if parts else ""

    def _prevent_overflow(
        self, messages: List[dict], max_tokens: int = 64000
    ) -> Tuple[List[dict], int]:
        """
        Token 溢出预防：若总 tokens 超出限制，从最旧的非系统消息开始裁切。

        保留：system 消息 + 最后一条 user 消息。
        """
        if not messages:
            return messages, 0

        total = sum(estimate_tokens(m["content"]) + 4 for m in messages)
        truncated = 0

        # 从最旧的非 system 消息开始裁切
        i = 0
        while total > max_tokens and i < len(messages):
            if messages[i]["role"] != "system" and messages[i]["role"] != "user":
                total -= estimate_tokens(messages[i]["content"]) + 4
                messages.pop(i)
                truncated += 1
                # 不递增 i，因为 pop 后下一个元素移到了当前位置
                continue
            i += 1

        # 如果还是超限，裁切非最后一条的 user 消息
        if total > max_tokens:
            i = 0
            while total > max_tokens and i < len(messages) - 1:
                if messages[i]["role"] == "user":
                    total -= estimate_tokens(messages[i]["content"]) + 4
                    messages.pop(i)
                    truncated += 1
                    continue
                i += 1

        return messages, truncated

    def estimate_total_tokens(self, messages: List[dict]) -> int:
        """估算消息列表的总 token 数。"""
        return sum(estimate_tokens(m["content"]) + 4 for m in messages)


# ---- 摘要生成 ----
def build_summary_prompt(messages: List[Message]) -> str:
    """构建用于生成历史摘要的 prompt。"""
    lines = ["请将以下对话历史压缩为一段简要的「对话纪要」，保留关键决策、重要结论和待办事项：\n"]
    for msg in messages:
        role_label = "用户" if msg.role == "user" else "助手"
        lines.append(f"[{role_label}] {msg.content[:500]}")
    lines.append("\n请用 2~3 句话总结以上对话的纪要。")
    return "\n".join(lines)


def should_compress(session: Session, window_start_idx: int) -> Tuple[bool, int]:
    """
    判断是否需要对远历史进行摘要压缩。

    Returns:
        (是否应压缩, 早于窗口的消息 token 总数)
    """
    if not session.summary_enabled:
        return False, 0

    old_messages = session.messages[:window_start_idx]
    if not old_messages:
        return False, 0

    old_tokens = estimate_messages_tokens(old_messages, session.model)
    if old_tokens < COMPRESSION_THRESHOLD_TOKENS:
        return False, old_tokens

    # 预估压缩后的长度（粗略：原长度 / COMPRESSION_MIN_RATIO）
    # 规格要求：压缩后 ≤ 原长的 20% 才执行（即压缩比 ≥ 5:1）
    estimated_compressed = old_tokens / COMPRESSION_MIN_RATIO
    if estimated_compressed > old_tokens * 0.2:
        # 压缩后长度 > 原长的 20%，不划算，跳过
        return False, old_tokens

    return True, old_tokens
