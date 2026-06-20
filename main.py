"""
ChatDeepSeek CLI · 入口 REPL 循环

基于 DeepSeek API 的高效终端对话应用。
"""

import sys
import time
from typing import Optional

from openai import OpenAI

# ---- 本地模块 ----
from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEFAULT_MODEL,
    MAX_RETRIES,
    RETRY_DELAY,
    MAX_OUTPUT_TOKENS,
)
from session_manager import SessionManager, Session, Message
from context_builder import ContextBuilder, build_summary_prompt, should_compress
from ui import (
    console,
    print_welcome,
    print_prompt,
    print_streaming,
    print_optimization_info,
    print_token_stats,
    print_session_list,
    print_switch_info,
    print_confirm_preview,
    print_help,
    print_error,
    print_success,
    print_warning,
    get_user_input,
    confirm_yes_no,
)


class ChatApp:
    """ChatDeepSeek CLI 主应用。"""

    def __init__(self):
        # 初始化 OpenAI 客户端
        if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your_api_key_here":
            print_error("请先在 .env 文件中设置 DEEPSEEK_API_KEY")
            sys.exit(1)

        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )

        # 初始化会话管理器
        self.manager = SessionManager()

        # 状态变量
        self.confirm_mode: bool = False
        self.running: bool = True
        self._multiline_lines: list = []  # 多行输入缓冲区

        # 累计 token 统计
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0

    # ================================================================
    # 主循环
    # ================================================================
    def run(self):
        """启动 REPL 主循环。"""
        print_welcome()

        while self.running:
            session = self.manager.current
            if session is None:
                print_error("无可用会话，正在创建默认会话...")
                self.manager.create_session("默认")
                self.manager.current_name = "默认"
                session = self.manager.current

            # ---- 多行输入模式 ----
            if self._multiline_lines:
                try:
                    user_input = get_user_input(">>> ")
                except (EOFError, KeyboardInterrupt):
                    self._multiline_lines = []
                    console.print("\n  [dim]已取消多行输入[/dim]")
                    continue

                stripped = user_input.strip()
                if stripped == "/end":
                    full_text = "\n".join(self._multiline_lines)
                    self._multiline_lines = []
                    if not full_text.strip():
                        print_warning("未输入任何内容，多行模式已退出")
                        continue
                    if self._handle_command(full_text):
                        continue
                    self._handle_chat(full_text)
                    continue
                elif stripped == "/cancel":
                    self._multiline_lines = []
                    print_warning("已取消多行输入")
                    continue
                else:
                    self._multiline_lines.append(user_input)
                    continue

            # ---- 普通模式 ----
            try:
                print_prompt(session.name, session.model)
                user_input = get_user_input("")
            except (EOFError, KeyboardInterrupt):
                console.print("\n  [dim]再见！[/dim]")
                break

            if not user_input.strip():
                continue

            # 处理命令
            if self._handle_command(user_input.strip()):
                continue

            # 处理对话
            self._handle_chat(user_input.strip())

        # 退出时保存
        self.manager.save_current()
        console.print("  [dim]会话已保存。再见！[/dim]")

    # ================================================================
    # 命令处理
    # ================================================================
    def _handle_command(self, text: str) -> bool:
        """处理以 / 或 ! 开头的命令。返回 True 表示已处理。"""
        # 快捷指令
        if text == "!exit":
            self.running = False
            return True
        if text == "!help":
            print_help()
            return True

        # 多行模式入口
        if text == "/multiline":
            self._multiline_lines = []
            console.print(
                "  [dim]已进入多行输入模式，逐行输入后 "
                "[bold]/end[/bold] 发送，[bold]/cancel[/bold] 取消[/dim]"
            )
            return True

        # / 命令
        if not text.startswith("/"):
            return False

        parts = text.split(maxsplit=2)
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        try:
            if cmd == "/new":
                self._cmd_new(args)
            elif cmd == "/switch":
                self._cmd_switch(args)
            elif cmd == "/list":
                self._cmd_list()
            elif cmd == "/rename":
                self._cmd_rename(args)
            elif cmd == "/delete":
                self._cmd_delete(args)
            elif cmd == "/clone":
                self._cmd_clone(args)
            elif cmd == "/model":
                self._cmd_model(args)
            elif cmd == "/role":
                self._cmd_role(args)
            elif cmd == "/core":
                self._cmd_core(args)
            elif cmd == "/doc":
                self._cmd_doc(args)
            elif cmd == "/recall_doc":
                self._cmd_recall_doc()
            elif cmd == "/summarize":
                self._cmd_summarize(args)
            elif cmd == "/tokens":
                self._cmd_tokens(args)
            elif cmd == "/confirm":
                self._cmd_confirm(args)
            else:
                print_error(f"未知命令: {cmd}（输入 !help 查看帮助）")
        except Exception as e:
            print_error(str(e))

        return True

    def _cmd_new(self, args: list):
        name = args[0] if args else "未命名"
        model = args[1] if len(args) > 1 else DEFAULT_MODEL
        session = self.manager.create_session(name, model)
        self.manager.current_name = name
        print_success(f"已创建并切换到会话 [{name}]（{model}）")

    def _cmd_switch(self, args: list):
        if not args:
            print_error("用法: /switch <会话名>")
            return
        name = args[0]
        try:
            self.manager.switch_session(name)
            session = self.manager.current
            print_success(f"已切换到 [{session.name}]")
            print_switch_info(session)
        except ValueError as e:
            print_error(str(e))

    def _cmd_list(self):
        sessions = self.manager.list_sessions()
        print_session_list(sessions, self.manager.current_name)

    def _cmd_rename(self, args: list):
        if not args:
            print_error("用法: /rename <新名称>")
            return
        new_name = args[0]
        old_name = self.manager.current.name if self.manager.current else "?"
        self.manager.rename_session(new_name)
        print_success(f"已将 [{old_name}] 重命名为 [{new_name}]")

    def _cmd_delete(self, args: list):
        if not args:
            print_error("用法: /delete <会话名>")
            return
        name = args[0]
        if not confirm_yes_no(f"确定要删除会话 [{name}] 吗？此操作不可撤销。"):
            print_warning("已取消删除")
            return
        try:
            self.manager.delete_session(name)
            print_success(f"已删除会话 [{name}]")
        except ValueError as e:
            print_error(str(e))

    def _cmd_clone(self, args: list):
        if len(args) < 2:
            print_error("用法: /clone <源会话> <新会话>")
            return
        src, new = args[0], args[1]
        try:
            self.manager.clone_session(src, new)
            print_success(f"已将会话 [{src}] 克隆到 [{new}]")
        except ValueError as e:
            print_error(str(e))

    def _cmd_model(self, args: list):
        if not args:
            print_error("用法: /model <模型ID>")
            return
        model = args[0]
        session = self.manager.current
        if session is None:
            return
        session.model = model
        self.manager._save_session(session, save_messages=False)
        print_success(f"当前会话模型已切换为 [{model}]")

    def _cmd_role(self, args: list):
        session = self.manager.current
        if session is None:
            return
        if not args:
            print_error("用法: /role <描述> 或 /role off 或 /role show")
            return
        arg = args[0]
        if arg.lower() == "off":
            session.temp_role = ""
            session.temp_role_persistent = False
            self.manager._save_session(session, save_messages=False)
            print_success("临时角色已关闭")
        elif arg.lower() == "show":
            if session.temp_role:
                console.print(f"  [dim]当前临时角色:[/dim] {session.temp_role}")
            else:
                console.print("  [dim]当前无临时角色[/dim]")
            if session.core_role:
                console.print(f"  [dim]核心角色:[/dim] {session.core_role}")
        else:
            full_text = " ".join(args)
            session.temp_role = full_text
            session.temp_role_persistent = True
            self.manager._save_session(session, save_messages=False)
            print_success(f"临时角色已设定并保存（持续生效，/role off 关闭）")

    def _cmd_core(self, args: list):
        session = self.manager.current
        if session is None:
            return
        if not args:
            print_error("用法: /core <核心角色描述>")
            return
        session.core_role = " ".join(args)
        self.manager._save_session(session, save_messages=False)
        print_success("核心角色已设定并保存")

    def _cmd_doc(self, args: list):
        session = self.manager.current
        if session is None:
            return
        if not args:
            print_error("用法: /doc <文档内容>")
            return
        doc_text = " ".join(args)
        session.knowledge_doc = doc_text
        print_success("文档已注入。是否立即生成摘要？")
        if confirm_yes_no("即将调用模型生成摘要（预计消耗约 200 tokens），是否继续？", "y"):
            self._generate_doc_summary(session)
        self.manager._save_session(session, save_messages=False)
        print_success("文档已保存")

    def _cmd_recall_doc(self):
        session = self.manager.current
        if session is None:
            return
        if not session.knowledge_doc:
            print_warning("当前会话无已注入的文档")
            return
        # 设置召回标记，不覆盖 temp_role（仅当轮生效）
        session.recall_doc_active = True
        print_success("完整文档已标记注入本轮上下文（不影响已有角色设定）")

    def _cmd_summarize(self, args: list):
        session = self.manager.current
        if session is None:
            return
        if not args:
            print_error("用法: /summarize on|off")
            return
        arg = args[0].lower()
        if arg == "on":
            session.summary_enabled = True
            self.manager._save_session(session, save_messages=False)
            print_success("自动摘要已开启")
        elif arg == "off":
            session.summary_enabled = False
            session.history_summary = ""
            self.manager._save_session(session, save_messages=False)
            print_success("自动摘要已关闭")
        else:
            print_error("用法: /summarize on|off")

    def _cmd_tokens(self, args: list):
        session = self.manager.current
        if not args:
            print_error("用法: /tokens last|session|total")
            return
        arg = args[0].lower()
        if arg == "last":
            if hasattr(self, "_last_prompt_tokens"):
                print_token_stats(
                    "上一轮消耗",
                    self._last_prompt_tokens,
                    self._last_completion_tokens,
                    session.model if session else DEFAULT_MODEL,
                )
            else:
                print_warning("尚无对话记录")
        elif arg == "session":
            # 计算当前会话累计
            prompt_sum = sum(
                m.token_usage.get("prompt", 0)
                for m in (session.messages if session else [])
                if m.role == "assistant"
            )
            completion_sum = sum(
                m.token_usage.get("completion", 0)
                for m in (session.messages if session else [])
                if m.role == "assistant"
            )
            print_token_stats(
                "本次会话累计",
                prompt_sum,
                completion_sum,
                session.model if session else DEFAULT_MODEL,
            )
        elif arg == "total":
            print_token_stats(
                "所有会话累计",
                self.total_prompt_tokens,
                self.total_completion_tokens,
                session.model if session else DEFAULT_MODEL,
            )
        else:
            print_error("用法: /tokens last|session|total")

    def _cmd_confirm(self, args: list):
        if not args:
            print_error("用法: /confirm on|off")
            return
        arg = args[0].lower()
        if arg == "on":
            self.confirm_mode = True
            print_success("发送前确认已开启")
        elif arg == "off":
            self.confirm_mode = False
            print_success("发送前确认已关闭")
        else:
            print_error("用法: /confirm on|off")

    # ================================================================
    # 对话处理
    # ================================================================
    def _handle_chat(self, text: str):
        """处理用户对话输入。"""
        session = self.manager.current
        if session is None:
            print_error("无当前会话")
            return

        # 构建上下文
        builder = ContextBuilder(session)
        api_messages, opt_info = self._build_context_with_compression(builder, text)

        # 发送前确认
        if self.confirm_mode:
            total_tokens = builder.estimate_total_tokens(api_messages)
            print_confirm_preview(api_messages, total_tokens)
            if not confirm_yes_no("是否发送？", "y"):
                print_warning("已取消发送")
                return

        # 发送请求并获取回复
        try:
            response_text, usage = self._send_with_retry(session, api_messages)
        except Exception as e:
            print_error(f"API 调用失败: {e}")
            return

        if response_text is None:
            return

        # 防止保存空回复
        if not response_text.strip():
            print_warning("模型返回空回复，未保存")
            return

        # 保存消息
        user_msg = Message(role="user", content=text)
        session.add_message(user_msg)

        assistant_msg = Message(
            role="assistant",
            content=response_text,
            token_usage=usage,
        )
        session.add_message(assistant_msg)

        # 更新统计
        self._last_prompt_tokens = usage.get("prompt", 0)
        self._last_completion_tokens = usage.get("completion", 0)
        self.total_prompt_tokens += usage.get("prompt", 0)
        self.total_completion_tokens += usage.get("completion", 0)

        # 打印优化信息
        print_optimization_info(opt_info)

        # 非持久的临时角色仅当轮生效
        if session.temp_role and not session.temp_role_persistent:
            session.temp_role = ""

        # 文档召回归位
        session.recall_doc_active = False

        # 持久化
        self.manager.save_current()

    def _build_context_with_compression(
        self, builder: ContextBuilder, text: str
    ) -> tuple:
        """构建上下文，并在需要时执行摘要压缩。"""
        session = self.manager.current
        total = len(session.messages)

        # 窗口起始索引
        from config import MAX_WINDOW as _MAX_WINDOW
        window_start = max(0, total - _MAX_WINDOW * 2)

        # 判断是否需要压缩
        should_compress_flag, old_tokens = should_compress(session, window_start)
        opt_info = {}

        if should_compress_flag:
            # 构建摘要请求
            old_messages = session.messages[:window_start]
            summary_prompt = build_summary_prompt(old_messages)

            # 调用模型生成摘要（静默模式）
            try:
                summary_text, usage = self._send_with_retry(
                    session,
                    [
                        {"role": "system", "content": "你是一个对话摘要助手。请简洁地总结对话。"},
                        {"role": "user", "content": summary_prompt},
                    ],
                    silent=True,  # 静默，不干扰用户
                )
                if summary_text:
                    session.history_summary = summary_text
                    session.summary_start_idx = window_start
                    opt_info["summary_saved_tokens"] = max(
                        0, old_tokens - 200
                    )  # 估算节省
                    # 统计摘要生成的 Token
                    self.total_prompt_tokens += usage.get("prompt", 0)
                    self.total_completion_tokens += usage.get("completion", 0)
            except Exception:
                pass  # 压缩失败不影响主流程

        # 构建主上下文
        api_messages, main_opt_info = builder.build(text)
        opt_info.update(main_opt_info)

        return api_messages, opt_info

    # ================================================================
    # API 调用
    # ================================================================
    def _send_with_retry(self, session: Session, messages: list, silent: bool = False) -> tuple:
        """
        发送请求并处理重试。

        Args:
            session: 当前会话。
            messages: API 消息列表。
            silent: 是否静默重试（用于摘要生成等后台任务）。

        Returns:
            (response_text, usage_dict)
        """
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                return self._call_api(session, messages, silent=silent)
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                if "401" in error_str or "403" in error_str:
                    raise Exception("认证失败，请检查 DEEPSEEK_API_KEY 是否正确")
                if "429" in error_str:
                    wait = RETRY_DELAY * (2 ** attempt)
                    if not silent:
                        print_warning(f"请求频率限制，{wait} 秒后重试...")
                    time.sleep(wait)
                elif attempt < MAX_RETRIES - 1:
                    wait = RETRY_DELAY * (2 ** attempt)
                    if not silent:
                        print_warning(f"网络异常，{wait} 秒后重试（{attempt + 2}/{MAX_RETRIES}）...")
                    time.sleep(wait)
                else:
                    raise Exception(f"重试 {MAX_RETRIES} 次后仍失败: {last_error}")

        raise Exception(f"重试 {MAX_RETRIES} 次后仍失败: {last_error}")

    def _call_api(self, session: Session, messages: list, silent: bool = False) -> tuple:
        """单次 API 调用，流式输出 + Markdown 渲染（代码高亮）。"""
        model = session.model

        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            max_tokens=MAX_OUTPUT_TOKENS,
            stream_options={"include_usage": True},
        )

        if not silent:
            console.print()  # 换行

        full_text = ""
        usage = {"prompt": 0, "completion": 0}

        # 使用 Live 实时渲染 Markdown（解决代码无高亮问题）
        from rich.live import Live
        from rich.markdown import Markdown

        live_ctx = Live(
            Markdown("", code_theme="one-dark"),
            console=console,
            refresh_per_second=8,
            transient=False,
        ) if not silent else None

        if live_ctx:
            live_ctx.__enter__()

        try:
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        content = delta.content
                        full_text += content
                        if live_ctx:
                            live_ctx.update(
                                Markdown(full_text, code_theme="one-dark")
                            )

                if hasattr(chunk, "usage") and chunk.usage:
                    usage["prompt"] = chunk.usage.prompt_tokens
                    usage["completion"] = chunk.usage.completion_tokens
        finally:
            if live_ctx:
                live_ctx.__exit__(None, None, None)

        if not silent:
            console.print()  # 最终换行
        return full_text, usage

    # ================================================================
    # 文档摘要生成
    # ================================================================
    def _generate_doc_summary(self, session: Session):
        """调用模型生成长文档摘要（静默模式）。"""
        prompt = (
            f"请用 2~3 句话总结以下文档的核心要点：\n\n{session.knowledge_doc}"
        )
        try:
            summary_text, usage = self._send_with_retry(
                session,
                [
                    {"role": "system", "content": "你是一个精准的文档摘要助手。"},
                    {"role": "user", "content": prompt},
                ],
                silent=True,
            )
            if summary_text:
                session.knowledge_summary = summary_text.strip()
                self.total_prompt_tokens += usage.get("prompt", 0)
                self.total_completion_tokens += usage.get("completion", 0)
                print_success("文档摘要已生成并缓存")
        except Exception as e:
            print_warning(f"摘要生成失败: {e}")


# ================================================================
# 入口
# ================================================================
def main():
    app = ChatApp()
    app.run()


if __name__ == "__main__":
    main()
