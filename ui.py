"""
ChatDeepSeek CLI · 终端 UI、流式打印、rich 美化
"""

import sys
from datetime import datetime, timezone
from typing import List, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console(highlight=False)

# 角色颜色映射
ROLE_COLORS = {
    "user": "bold cyan",
    "assistant": "bright_white",
    "system": "dim yellow",
}


def print_welcome(banner: bool = True):
    """打印欢迎信息。"""
    if banner:
        console.print()
        console.print(
            Panel.fit(
                "[bold bright_blue]ChatDeepSeek CLI[/bold bright_blue]  ·  "
                "[dim]高效 · 省 Token · 多会话[/dim]",
                border_style="bright_blue",
            )
        )
        console.print("  [dim]输入 [bold]!help[/bold] 查看帮助，[bold]!exit[/bold] 退出[/dim]")
        console.print()


def print_prompt(session_name: str, model: str):
    """打印提示符。"""
    text = Text()
    text.append(f"[{session_name}]", style="bold green")
    text.append(f" [{model}]", style="dim")
    text.append(" > ", style="bright_white")
    # 用 rich 输出，但不换行
    console.print(text, end="")


def print_streaming(content: str):
    """模拟流式输出效果，逐字打印。"""
    # 使用 Markdown 渲染整段
    md = Markdown(content, code_theme="one-dark")
    console.print(md)


def print_assistant_header():
    """打印助手回复前缀。"""
    console.print()  # 换行


def print_optimization_info(info: dict):
    """打印优化操作通报（灰色字体）。"""
    parts = []
    if info.get("window_count"):
        parts.append(f"滑动窗口 {info['window_count']//2} 轮")
    if info.get("semantic_count"):
        parts.append(f"语义召回 {info['semantic_count']} 条")
    if info.get("keyword_count"):
        parts.append(f"关键词召回 {info['keyword_count']} 条")
    if info.get("summary_saved_tokens"):
        parts.append(f"远历史已压缩（节省 {info['summary_saved_tokens']} tokens）")
    if info.get("truncated_count"):
        parts.append(f"裁切 {info['truncated_count']} 条超限消息")

    not_found = info.get("at_not_found", [])
    for anchor in not_found:
        console.print(f"  [yellow]⚠ 未找到与 \"{anchor}\" 相关的历史消息[/yellow]")

    if parts:
        console.print(f"  [dim][优化] {' | '.join(parts)}[/dim]")


def print_token_stats(label: str, prompt_tokens: int, completion_tokens: int, model: str):
    """打印 Token 统计信息。"""
    from config import MODEL_PRICES

    total = prompt_tokens + completion_tokens
    price = MODEL_PRICES.get(model, {"prompt": 0, "completion": 0})
    cost = (prompt_tokens / 1_000_000) * price["prompt"] + (
        completion_tokens / 1_000_000
    ) * price["completion"]

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(style="bright_white")
    table.add_row("Prompt tokens", str(prompt_tokens))
    table.add_row("Completion tokens", str(completion_tokens))
    table.add_row("Total tokens", str(total))
    table.add_row(
        "费用（≈ 参考价）",
        f"${cost:.6f}（以 API 账单为准）",
    )

    console.print(f"  [bold]{label}[/bold]")
    console.print(table)


def print_session_list(sessions: list, current_name: str):
    """打印会话列表。"""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_column(style="dim")
    table.add_column(style="dim")

    for s in sessions:
        marker = "[bold green]*[/bold green]" if s.name == current_name else " "
        name_display = (
            f"[bold green]{s.name}[/bold green]"
            if s.name == current_name
            else s.name
        )
        last_activity = _relative_time(s.last_activity)
        table.add_row(
            marker,
            f"{name_display} ({s.model})",
            f"{s.message_count} 条消息",
            f"最后活动: {last_activity}",
        )

    console.print(table)


def print_switch_info(session):
    """打印切换会话时的信息。"""
    last_msg = session.last_message
    if last_msg:
        preview = last_msg.content[:60].replace("\n", " ")
        if len(last_msg.content) > 60:
            preview += "…"
        console.print(
            f"  [dim]最后消息 ({_relative_time(last_msg.timestamp)}): "
            f'"{preview}"[/dim]'
        )


def print_confirm_preview(messages: List[dict], total_tokens: int):
    """打印发送前确认预览。"""
    console.print(f"\n  [bold yellow]═══ 发送前确认（约 {total_tokens} tokens）═══[/bold yellow]")
    for i, msg in enumerate(messages):
        role = msg["role"]
        color = ROLE_COLORS.get(role, "white")
        preview = msg["content"][:100].replace("\n", " ")
        if len(msg["content"]) > 100:
            preview += "…"
        console.print(f"  [{color}][{role}][/{color}] {preview}")
    console.print("  [bold yellow]══════════════════════════════════[/bold yellow]")


def print_help():
    """打印帮助信息。"""
    help_text = """
[bold bright_blue]ChatDeepSeek CLI · 帮助[/bold bright_blue]

[bold]会话管理[/bold]
  /new <名称> [模型]     创建新会话
  /switch <名称>          切换会话（支持前缀模糊匹配）
  /list                   列出所有会话
  /rename <新名称>         重命名当前会话
  /delete <名称>          删除会话（需确认）
  /clone <源> <目标>       复制会话上下文

[bold]模型与上下文[/bold]
  /model <模型ID>          切换当前会话模型
  /role <描述>             追加临时角色指令
  /role off               关闭临时角色
  /role show              查看当前角色设定
  /core <描述>             设定核心角色（≤3句）
  /doc <文本>              注入一次性长文档/知识库
  /recall_doc             重新注入完整文档
  /summarize on|off        开启/关闭自动历史摘要

[bold]Token 与统计[/bold]
  /tokens last            上轮消耗
  /tokens session         当前会话累计
  /tokens total           所有会话累计
  /confirm on|off          开启/关闭发送前确认

[bold]历史召回[/bold]
  @关键词                  语义召回历史消息
  @"精确短语"               关键词精确匹配召回

[bold]快捷指令[/bold]
  !exit                   退出程序
  !help                   显示此帮助
  /multiline              进入多行输入模式（/end 发送，/cancel 取消）
"""
    console.print(help_text)


def print_error(msg: str):
    """打印错误信息。"""
    console.print(f"  [red]✗ {msg}[/red]")


def print_success(msg: str):
    """打印成功信息。"""
    console.print(f"  [green]✓ {msg}[/green]")


def print_warning(msg: str):
    """打印警告信息。"""
    console.print(f"  [yellow]⚠ {msg}[/yellow]")


def get_user_input(prompt_text: str) -> str:
    """获取用户输入（支持 readline 历史）。"""
    try:
        return input(prompt_text)
    except (EOFError, KeyboardInterrupt):
        return "!exit"


def confirm_yes_no(prompt: str, default: str = "n") -> bool:
    """询问 Y/n 确认。"""
    answer = input(f"  {prompt} (Y/n): ").strip().lower()
    if not answer:
        answer = default
    return answer == "y" or answer == "yes"


# ============================================================
# 工具函数
# ============================================================
def _relative_time(iso_timestamp: str) -> str:
    """将 ISO 时间戳转为相对时间描述。"""
    try:
        dt = datetime.fromisoformat(iso_timestamp)
        now = datetime.now(timezone.utc)
        # 处理 naive datetime
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        diff = now - dt
        seconds = diff.total_seconds()
        if seconds < 60:
            return "刚刚"
        elif seconds < 3600:
            return f"{int(seconds // 60)} 分钟前"
        elif seconds < 86400:
            return f"{int(seconds // 3600)} 小时前"
        elif seconds < 604800:
            return f"{int(seconds // 86400)} 天前"
        else:
            return dt.strftime("%Y-%m-%d")
    except Exception:
        return iso_timestamp[:10]
