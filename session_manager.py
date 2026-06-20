"""
ChatDeepSeek CLI · 会话管理

提供会话的创建、切换、列表、重命名、删除、克隆，
以及 JSONL 持久化与原子写入。
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import CHATS_DIR, META_DIR, DEFAULT_MODEL


# ============================================================
# 消息格式
# ============================================================
class Message:
    """单条对话消息。"""
    def __init__(
        self,
        role: str,
        content: str,
        timestamp: Optional[str] = None,
        token_usage: Optional[dict] = None,
    ):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.token_usage = token_usage or {}

    def to_dict(self) -> dict:
        d = {"role": self.role, "content": self.content, "timestamp": self.timestamp}
        if self.role == "assistant" and self.token_usage:
            d["token_usage"] = self.token_usage
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        return cls(
            role=d["role"],
            content=d["content"],
            timestamp=d.get("timestamp", ""),
            token_usage=d.get("token_usage"),
        )


# ============================================================
# 会话类
# ============================================================
class Session:
    """单个会话，包含名称、模型、消息列表、元数据。"""
    def __init__(self, name: str, model: str = DEFAULT_MODEL):
        self.name = name
        self.model = model
        self.messages: List[Message] = []
        self.created_at = datetime.now(timezone.utc).isoformat()

        # 系统提示三要素
        self.core_role: str = ""          # 核心角色（≤3句）
        self.knowledge_doc: str = ""      # 一次性长文档（原始）
        self.knowledge_summary: str = ""  # 长文档摘要
        self.temp_role: str = ""          # 临时角色指令
        self.temp_role_persistent: bool = False
        self.recall_doc_active: bool = False  # 文档召回标记（仅当轮，不覆盖 temp_role）

        # 摘要压缩
        self.summary_enabled: bool = False
        self.history_summary: str = ""    # 远历史纪要
        self.summary_start_idx: int = 0   # 纪要覆盖的起始索引

    @property
    def last_message(self) -> Optional[Message]:
        if self.messages:
            return self.messages[-1]
        return None

    @property
    def last_activity(self) -> str:
        if self.last_message:
            return self.last_message.timestamp
        return self.created_at

    @property
    def message_count(self) -> int:
        return len(self.messages)

    def add_message(self, msg: Message):
        self.messages.append(msg)

    def to_meta(self) -> dict:
        return {
            "name": self.name,
            "model": self.model,
            "created_at": self.created_at,
            "core_role": self.core_role,
            "knowledge_summary": self.knowledge_summary,
            "knowledge_doc": self.knowledge_doc,
            "temp_role": self.temp_role,
            "temp_role_persistent": self.temp_role_persistent,
            "summary_enabled": self.summary_enabled,
            "history_summary": self.history_summary,
            "summary_start_idx": self.summary_start_idx,
        }

    @classmethod
    def from_meta(cls, meta: dict, messages: List[Message]) -> "Session":
        s = cls(name=meta["name"], model=meta.get("model", DEFAULT_MODEL))
        s.created_at = meta.get("created_at", s.created_at)
        s.core_role = meta.get("core_role", "")
        s.knowledge_summary = meta.get("knowledge_summary", "")
        s.knowledge_doc = meta.get("knowledge_doc", "")
        s.temp_role = meta.get("temp_role", "")
        s.temp_role_persistent = meta.get("temp_role_persistent", False)
        s.recall_doc_active = False  # 仅当轮生效，不持久化
        s.summary_enabled = meta.get("summary_enabled", False)
        s.history_summary = meta.get("history_summary", "")
        s.summary_start_idx = meta.get("summary_start_idx", 0)
        s.messages = messages
        return s


# ============================================================
# 会话管理器
# ============================================================
class SessionManager:
    """管理所有会话的创建、切换、持久化。"""

    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        self.current_name: Optional[str] = None
        self._load_all()

    # ---- 当前会话快捷访问 ----
    @property
    def current(self) -> Optional[Session]:
        if self.current_name:
            return self.sessions.get(self.current_name)
        return None

    # ---- 文件路径 ----
    @staticmethod
    def _jsonl_path(name: str) -> str:
        safe = _safe_filename(name)
        return os.path.join(CHATS_DIR, f"{safe}.jsonl")

    @staticmethod
    def _meta_path(name: str) -> str:
        safe = _safe_filename(name)
        return os.path.join(META_DIR, f"{safe}.json")

    # ---- 加载所有会话 ----
    def _load_all(self):
        """启动时扫描 chats/ 目录重建会话列表。"""
        restored = 0
        for fname in os.listdir(CHATS_DIR):
            if not fname.endswith(".jsonl"):
                continue
            safe_name = fname[:-6]  # 去掉 .jsonl
            jsonl_path = os.path.join(CHATS_DIR, fname)
            meta_path = os.path.join(META_DIR, f"{safe_name}.json")

            messages = self._load_jsonl(jsonl_path)
            meta = {}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except (json.JSONDecodeError, IOError):
                    pass

            name = meta.get("name", safe_name)
            session = Session.from_meta(meta, messages)
            self.sessions[name] = session
            restored += 1

        # 按最后活动时间排序，最近的在前面
        sorted_names = sorted(
            self.sessions.keys(),
            key=lambda n: self.sessions[n].last_activity,
            reverse=True,
        )
        # 自动进入最近使用的会话
        if sorted_names:
            self.current_name = sorted_names[0]
        else:
            # 不存在任何会话，自动创建默认会话
            self.create_session("默认")
            self.current_name = "默认"

        if restored > 0:
            from ui import console
            console.print(f"[dim]已恢复 {restored} 个历史会话。[/dim]")

    def _load_jsonl(self, path: str) -> List[Message]:
        """加载 JSONL 文件，容错处理损坏行。"""
        messages = []
        if not os.path.exists(path):
            return messages
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        messages.append(Message.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        from ui import console
                        console.print(
                            f"[yellow]⚠ 警告: {path} 第 {line_num} 行损坏，已跳过[/yellow]"
                        )
        except IOError:
            pass
        return messages

    # ---- 创建会话 ----
    def create_session(self, name: str, model: str = DEFAULT_MODEL) -> Session:
        if name in self.sessions:
            raise ValueError(f"会话 '{name}' 已存在")
        # 检查安全文件名冲突
        safe = _safe_filename(name)
        for existing_name in self.sessions:
            if _safe_filename(existing_name) == safe:
                raise ValueError(
                    f"会话名 '{name}' 与已有会话 '{existing_name}' 的文件名冲突，"
                    f"请使用不同的名称"
                )
        session = Session(name, model)
        self.sessions[name] = session
        self._save_session(session, save_messages=True)
        return session

    # ---- 切换会话 ----
    def switch_session(self, name: str) -> Session:
        """切换到指定会话，支持前缀模糊匹配。"""
        if not name:
            raise ValueError("会话名不能为空")
        if name in self.sessions:
            self.current_name = name
            return self.sessions[name]

        # 前缀模糊匹配
        matches = [n for n in self.sessions if n.startswith(name)]
        if len(matches) == 1:
            self.current_name = matches[0]
            return self.sessions[matches[0]]
        elif len(matches) > 1:
            from ui import console
            console.print(f"[yellow]多个匹配: {', '.join(matches)}[/yellow]")
            raise ValueError(f"多个会话匹配 '{name}'，请更精确指定")
        else:
            raise ValueError(f"会话 '{name}' 不存在")

    # ---- 列出会话 ----
    def list_sessions(self):
        """返回排序后的会话列表。"""
        return sorted(
            self.sessions.values(),
            key=lambda s: s.last_activity,
            reverse=True,
        )

    # ---- 重命名 ----
    def rename_session(self, new_name: str):
        session = self.current
        if session is None:
            raise ValueError("没有当前会话")
        if new_name in self.sessions:
            raise ValueError(f"会话 '{new_name}' 已存在")
        # 检查安全文件名冲突（排除自身）
        new_safe = _safe_filename(new_name)
        for existing_name in self.sessions:
            if existing_name != session.name and _safe_filename(existing_name) == new_safe:
                raise ValueError(
                    f"新名称 '{new_name}' 与已有会话 '{existing_name}' 的文件名冲突"
                )
        old_name = session.name
        old_safe = _safe_filename(old_name)
        new_safe = _safe_filename(new_name)

        # 删除旧文件
        old_jsonl = os.path.join(CHATS_DIR, f"{old_safe}.jsonl")
        old_meta = os.path.join(META_DIR, f"{old_safe}.json")
        for p in [old_jsonl, old_meta]:
            if os.path.exists(p):
                os.remove(p)

        session.name = new_name
        del self.sessions[old_name]
        self.sessions[new_name] = session
        self.current_name = new_name
        self._save_session(session, save_messages=True)

    # ---- 删除 ----
    def delete_session(self, name: str):
        if name not in self.sessions:
            raise ValueError(f"会话 '{name}' 不存在")
        safe = _safe_filename(name)
        jsonl_path = os.path.join(CHATS_DIR, f"{safe}.jsonl")
        meta_path = os.path.join(META_DIR, f"{safe}.json")
        for p in [jsonl_path, meta_path]:
            if os.path.exists(p):
                os.remove(p)
        del self.sessions[name]
        if self.current_name == name:
            remaining = list(self.sessions.keys())
            self.current_name = remaining[0] if remaining else None
            if not self.current_name:
                self.create_session("默认")
                self.current_name = "默认"

    # ---- 克隆 ----
    def clone_session(self, source_name: str, new_name: str) -> Session:
        if source_name not in self.sessions:
            raise ValueError(f"源会话 '{source_name}' 不存在")
        if new_name in self.sessions:
            raise ValueError(f"目标会话 '{new_name}' 已存在")
        # 检查安全文件名冲突
        new_safe = _safe_filename(new_name)
        for existing_name in self.sessions:
            if _safe_filename(existing_name) == new_safe:
                raise ValueError(
                    f"目标名 '{new_name}' 与已有会话 '{existing_name}' 的文件名冲突"
                )
        src = self.sessions[source_name]
        clone = Session(new_name, src.model)
        clone.core_role = src.core_role
        clone.knowledge_doc = src.knowledge_doc
        clone.knowledge_summary = src.knowledge_summary
        clone.summary_enabled = src.summary_enabled
        clone.history_summary = src.history_summary
        clone.summary_start_idx = src.summary_start_idx
        clone.messages = [
            Message(msg.role, msg.content, msg.timestamp, dict(msg.token_usage))
            for msg in src.messages
        ]
        self.sessions[new_name] = clone
        self._save_session(clone, save_messages=True)
        return clone

    # ---- 持久化 ----
    def save_current(self):
        """保存当前会话到磁盘（原子写入）。"""
        if self.current is not None:
            self._save_session(self.current, save_messages=True)

    def _save_session(self, session: Session, save_messages: bool = True):
        """原子写入：先写临时文件，关闭后再替换原文件。"""
        safe = _safe_filename(session.name)

        # 保存元数据
        meta_path = os.path.join(META_DIR, f"{safe}.json")
        tmp_meta_path = meta_path + ".tmp"
        try:
            with open(tmp_meta_path, "w", encoding="utf-8") as f:
                json.dump(session.to_meta(), f, ensure_ascii=False, indent=2)
            os.replace(tmp_meta_path, meta_path)
        except (IOError, OSError):
            # 回退：直接写入
            try:
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(session.to_meta(), f, ensure_ascii=False, indent=2)
            except (IOError, OSError):
                pass

        if not save_messages:
            return

        # 保存消息
        jsonl_path = os.path.join(CHATS_DIR, f"{safe}.jsonl")
        tmp_jsonl_path = jsonl_path + ".tmp"
        try:
            with open(tmp_jsonl_path, "w", encoding="utf-8") as f:
                for msg in session.messages:
                    f.write(json.dumps(msg.to_dict(), ensure_ascii=False) + "\n")
            os.replace(tmp_jsonl_path, jsonl_path)
        except (IOError, OSError):
            # 回退：直接写入
            try:
                with open(jsonl_path, "w", encoding="utf-8") as f:
                    for msg in session.messages:
                        f.write(json.dumps(msg.to_dict(), ensure_ascii=False) + "\n")
            except (IOError, OSError):
                pass


# ============================================================
# 工具函数
# ============================================================
def _safe_filename(name: str) -> str:
    """将会话名转为安全的文件名。"""
    safe = "".join(c if c.isalnum() or c in "._- " else "_" for c in name)
    return safe.strip() or "default"
