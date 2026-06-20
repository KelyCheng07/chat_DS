"""
ChatDeepSeek CLI · 语义召回引擎

优先使用轻量本地模型 sentence-transformers 计算余弦相似度。
若 sentence-transformers 不可用，回退到 TF-IDF 关键词匹配。
支持相似度阈值过滤，离线本地计算，无额外 API 调用。
"""

import re
import numpy as np
from typing import List, Tuple, Optional

from config import SIMILARITY_THRESHOLD, SEMANTIC_TOP_K

# 全局模型（延迟加载）
_model = None
_tfidf_vectorizer = None
_use_transformers = None  # None=未检测, True=可用, False=不可用


def _check_transformers() -> bool:
    """检测 sentence-transformers 是否可用。"""
    global _use_transformers
    if _use_transformers is None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
            _use_transformers = True
        except ImportError:
            _use_transformers = False
    return _use_transformers


def _get_model():
    """获取或初始化嵌入模型（单例）。"""
    global _model
    if _model is None and _check_transformers():
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _model


def _get_tfidf_vectorizer():
    """获取或初始化 TF-IDF 向量化器（回退方案）。"""
    global _tfidf_vectorizer
    if _tfidf_vectorizer is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        _tfidf_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), max_features=5000
        )
    return _tfidf_vectorizer


def encode_texts(texts: List[str]) -> np.ndarray:
    """将文本列表编码为向量。"""
    if not texts:
        return np.array([])
    if _check_transformers():
        model = _get_model()
        if model:
            return model.encode(texts, convert_to_numpy=True)
    # TF-IDF 回退
    vectorizer = _get_tfidf_vectorizer()
    try:
        return vectorizer.fit_transform(texts).toarray()
    except ValueError:
        return np.zeros((len(texts), 1))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度。"""
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _simple_similarity(query: str, text: str) -> float:
    """简单的关键词重叠相似度（纯 Python，无依赖）。

    对中文使用字符级 bigram 切分，英文使用单词切分。
    """
    def tokenize(s: str) -> set:
        tokens = set()
        s_lower = s.lower()
        # 按中文/非中文分段
        # 方案：对每段连续的 CJK 字符做 bigram，非 CJK 做单词分割
        segments = re.split(r"([\u4e00-\u9fff\uff00-\uffef]+)", s_lower)
        for seg in segments:
            if not seg:
                continue
            if re.match(r"[\u4e00-\u9fff\uff00-\uffef]", seg[0]):
                # CJK 段：字符级 bigram
                for i in range(len(seg)):
                    tokens.add(seg[i])
                    if i + 1 < len(seg):
                        tokens.add(seg[i:i + 2])
            else:
                # 非 CJK：单词切分
                for w in re.findall(r"[a-zA-Z0-9]+", seg):
                    tokens.add(w)
        return tokens

    query_tokens = tokenize(query)
    text_tokens = tokenize(text)
    if not query_tokens or not text_tokens:
        return 0.0
    intersection = query_tokens & text_tokens
    return len(intersection) / len(query_tokens)


def semantic_search(
    query: str,
    candidates: List[Tuple[int, str]],  # [(index, text), ...]
    top_k: int = None,
    threshold: float = None,
) -> List[Tuple[int, float]]:
    """
    对候选消息执行语义检索。

    优先使用 sentence-transformers 语义模型，
    若不可用则回退到 TF-IDF + 关键词匹配。

    Args:
        query: 当前提问文本。
        candidates: [(索引, 消息文本), ...]。
        top_k: 保留的最佳匹配数，默认使用配置的 SEMANTIC_TOP_K。
        threshold: 相似度阈值，默认使用配置的 SIMILARITY_THRESHOLD。

    Returns:
        [(索引, 相似度), ...]，按相似度降序排列。
    """
    if top_k is None:
        top_k = SEMANTIC_TOP_K
    if threshold is None:
        threshold = SIMILARITY_THRESHOLD

    if not candidates:
        return []

    texts = [t for _, t in candidates]

    scores: List[Tuple[int, float]] = []

    if _check_transformers():
        model = _get_model()
        if model:
            try:
                query_vec = model.encode([query], convert_to_numpy=True)[0]
                candidate_vecs = model.encode(texts, convert_to_numpy=True)

                query_norm = np.linalg.norm(query_vec)
                if query_norm > 0:
                    for i, (idx, _) in enumerate(candidates):
                        cand_norm = np.linalg.norm(candidate_vecs[i])
                        if cand_norm == 0:
                            continue
                        sim = float(
                            np.dot(query_vec, candidate_vecs[i])
                            / (query_norm * cand_norm)
                        )
                        if sim >= threshold:
                            scores.append((idx, sim))
            except Exception:
                pass  # 模型加载失败，回退到简单匹配

    # 回退方案：TF-IDF 关键词匹配
    if not scores:
        try:
            vectorizer = _get_tfidf_vectorizer()
            all_texts = [query] + texts
            tfidf_matrix = vectorizer.fit_transform(all_texts).toarray()
            query_vec = tfidf_matrix[0]
            for i, (idx, _) in enumerate(candidates):
                text_vec = tfidf_matrix[i + 1]
                sim = cosine_similarity(query_vec, text_vec)
                if sim >= threshold:
                    scores.append((idx, sim))
        except (ValueError, Exception):
            pass  # TF-IDF 失败，继续回退

    # 最终回退：纯关键词 Bigram 匹配（始终尝试）
    if not scores:
        for idx, text in candidates:
            sim = _simple_similarity(query, text)
            if sim >= threshold * 0.5:  # 简单匹配阈值减半
                scores.append((idx, sim))

    # 降序排序，取 top_k
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def keyword_search(
    keywords: List[str],
    candidates: List[Tuple[int, str]],
    top_k: int = 2,
) -> List[Tuple[int, float]]:
    """
    精确关键词匹配（不依赖嵌入）。

    对每个关键词，计算在每个候选文本中的匹配度（简单 TF 比例）。
    返回匹配度最高的 top_k 条。

    Args:
        keywords: 关键词列表。
        candidates: [(索引, 消息文本), ...]。
        top_k: 每关键词保留的最佳匹配数。

    Returns:
        [(索引, 匹配度), ...]，去重后按匹配度降序排列。
    """
    if not candidates:
        return []

    results: List[Tuple[int, float]] = []
    seen: set = set()

    for kw in keywords:
        kw_lower = kw.lower()
        for idx, text in candidates:
            if idx in seen:
                continue
            text_lower = text.lower()
            # 简单匹配：关键词出现次数 / 文本词数
            words = text_lower.split()
            if len(words) == 0:
                continue
            count = text_lower.count(kw_lower)
            if count > 0:
                score = count / len(words)
                results.append((idx, score))

    # 去重（保留最高分）
    best: dict = {}
    for idx, score in results:
        if idx not in best or score > best[idx]:
            best[idx] = score

    sorted_results = sorted(best.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]
