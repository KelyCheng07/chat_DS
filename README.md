# ChatDeepSeek CLI

> **极致省 Token 的 DeepSeek 终端对话工具**  
> 多会话 · 语义召回 · 自动摘要 · 流式 Markdown

---

> 📝 **项目说明**  
> 本项目核心初始版本由 DeepSeek V4 依据梳理的规格书生成，本人负责需求梳理、实测校验、BUG 修复、后续迭代统筹。属于人机协作学习项目，用于个人学习与开源交流。

---

## 为什么用 ChatDeepSeek CLI

DeepSeek API 按 Token 计费，每轮对话把全部历史发过去很浪费。这个工具帮你自动筛选上下文：

- 只带最近 N 轮完整对话（滑动窗口）
- 旧消息按语义相似度智能召回（本地 AI 模型，不费 Token）
- 更早的历史自动压缩为纪要（省 80%+）
- `@关键词` 一键召回相关历史
- 多会话独立隔离，互不干扰

## 快速开始（从零开始）

### 第一步：安装 Python

本项目需要 **Python 3.10 或更高版本**。

<details>
<summary>点我展开 → 还不知道自己有没有 Python？怎么看版本？</summary>

**Windows：**
1. 按 `Win + R`，输入 `cmd` 回车
2. 输入 `python --version` 回车
3. 如果显示 `Python 3.10.x` 或更高 → 已安装，跳过这步
4. 如果显示"不是内部命令" → 去 [python.org](https://python.org) 下载安装
   - 下载时勾选 **"Add Python to PATH"**
   - 装完重启 CMD，再输入 `python --version` 确认

**macOS / Linux：** 一般自带了 Python 3，在终端输入 `python3 --version` 确认。
</details>

### 第二步：下载本项目

```bash
git clone https://github.com/KelyCheng07/chat_DS.git
cd chat_DS
```

> 如果提示 `git` 不是内部命令，先去 https://git-scm.com/download 下载安装 Git。
>
> 不想装 Git 也可以直接去 GitHub 页面点绿色「Code」→「Download ZIP」，解压后进 `chat_DS` 文件夹。

### 第三步：安装依赖包

```bash
pip install -r requirements.txt
```

<details>
<summary>点我展开 → 还不知道 pip 是什么？装完 Python 还是报错？</summary>

**`pip` 是 Python 的「应用商店」**，用来一键安装别人写好的代码库。

如果提示 `pip` 不是内部命令：

- Windows：`python -m pip install -r requirements.txt`
- macOS / Linux：`pip3 install -r requirements.txt`

如果你在中国大陆，下载可能很慢，可以加国内镜像加速：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**它会安装什么？**

| 包名 | 用途 |
|------|------|
| `openai` | 调用 DeepSeek 的 API（和 AI 对话的核心） |
| `rich` | 让终端显示彩色文字和代码高亮 |
| `python-dotenv` | 读取 `.env` 配置文件 |
| `numpy` | 数学计算，用于语义搜索 |
| `tiktoken` | 精确计算用了多少 Token（不装也行） |
| `sentence-transformers` | 让 @关键词 召回更准确（不装也行） |

> 全部装完大约 1~3 分钟，取决于网速。后面两个不装也能正常使用。
</details>

### 第四步：配置 API Key

```bash
# 把 .env.example 复制一份，改名为 .env
cp .env.example .env
```

用记事本打开 `.env`，找到这一行：

```ini
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxx
```

把 `sk-xxxxxxxx...` 替换成你在 [platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys) 申请的真正 Key。

### 第五步：启动

```bash
python main.py
```

第一次运行后会自动创建「默认」会话。输入 `!help` 查看所有命令。

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

- **Python 3.10+**（[下载](https://python.org)）
- **DeepSeek API Key**（[免费注册获取](https://platform.deepseek.com/api_keys)，充值后可用）
- 用到的所有 Python 库（`openai`、`rich` 等）—— `pip install -r requirements.txt` 一键安装

## 文档

- [完整用户手册 (USAGE.md)](USAGE.md) — 18 章，从入门到进阶
- [配置参数参考](USAGE.md#13-配置参数完全参考)
- [故障排除](USAGE.md#14-故障排除)

## 开源协议

[MIT License](LICENSE)

## 贡献

欢迎提 Issue 和 Pull Request。详见 [CONTRIBUTING.md](CONTRIBUTING.md)。
