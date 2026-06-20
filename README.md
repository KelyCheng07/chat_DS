# ChatDeepSeek CLI

> **极致省 Token 的 DeepSeek 终端对话工具**  
> 多会话 · 语义召回 · 自动摘要 · 流式 Markdown

---

## 为什么用 ChatDeepSeek CLI

DeepSeek API 按 Token 计费，每轮对话把全部历史发过去很浪费。这个工具帮你自动筛选上下文：

- 只带最近 N 轮完整对话（滑动窗口）
- 旧消息按语义相似度智能召回（本地 AI 模型，不费 Token）
- 更早的历史自动压缩为纪要（省 80%+）
- `@关键词` 一键召回相关历史
- 多会话独立隔离，互不干扰

## 快速开始

```bash
# 1. 克隆
git clone https://github.com/KelyCheng07/chat_DS.git
cd chat_DS

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 DEEPSEEK_API_KEY

# 4. 启动
python main.py
```

第一次运行会自动创建「默认」会话。输入 `!help` 查看所有命令。

## 功能概览

| 功能 | 说明 |
|------|------|
| 🗂️ 多会话管理 | `/new` `/switch` `/list` `/rename` `/delete` `/clone` |
| 🎭 角色设定 | `/core` 核心角色 + `/role` 临时角色 + `/doc` 知识库 |
| 📡 @ 历史召回 | `@关键词` 语义搜索 + `@"精确短语"` 关键词匹配 |
| 📊 Token 统计 | `/tokens last/session/total` 含费用估算 |
| 🔄 自动摘要 | `/summarize on/off` 远历史智能压缩 |
| ✅ 发送前确认 | `/confirm on/off` 预览 Token 消耗 |
| 🎨 流式 Markdown | 代码高亮、标题、列表实时渲染 |
| 💾 本地存储 | JSONL 格式，透明可查，随时备份 |

## 配置参数

编辑 `.env` 文件调整行为：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MAX_WINDOW` | 10 | 滑动窗口保留轮数 |
| `SIMILARITY_THRESHOLD` | 0.5 | 语义召回相似度阈值 |
| `SEMANTIC_TOP_K` | 4 | 语义召回最多带回条数 |
| `MAX_OUTPUT_TOKENS` | 4096 | 每次回复最大 Token |
| `COMPRESSION_THRESHOLD_TOKENS` | 2000 | 压缩触发阈值 |
| `COMPRESSION_MIN_RATIO` | 5.0 | 最小压缩比 |
| `MAX_RETRIES` | 3 | API 重试次数 |

## 项目结构

```text
chat_deepseek/
├── main.py                 # REPL 入口
├── config.py               # 全局配置
├── session_manager.py      # 会话 CRUD + JSONL 持久化
├── context_builder.py      # 上下文组装管线
├── semantic_search.py      # 语义召回引擎（双回退）
├── ui.py                   # Rich 终端 UI
├── requirements.txt        # Python 依赖
├── .env.example            # 配置模板
├── USAGE.md                # 完整使用手册
└── chats/                  # 对话数据（不提交到 Git）
```

## 依赖

| 包 | 用途 | 必需 |
|----|------|:---:|
| `openai` | DeepSeek API 调用 | ✅ |
| `rich` | 终端 Markdown 渲染 | ✅ |
| `python-dotenv` | 环境变量管理 | ✅ |
| `numpy` | 向量计算 | ✅ |
| `tiktoken` | Token 精确估算 | ⚠️ |
| `sentence-transformers` | 语义相似度 | ⚠️ |

> ⚠️ 可选：不装也能跑，自动回退到 TF-IDF 匹配。

## 环境要求

- Python 3.10+
- DeepSeek API Key（[获取地址](https://platform.deepseek.com/api_keys)）

## 文档

- [完整用户手册 (USAGE.md)](USAGE.md) — 18 章，从入门到进阶
- [配置参数参考](USAGE.md#13-配置参数完全参考)
- [故障排除](USAGE.md#14-故障排除)

## 开源协议

[MIT License](LICENSE)

## 贡献

欢迎提 Issue 和 Pull Request。详见 [CONTRIBUTING.md](CONTRIBUTING.md)。
