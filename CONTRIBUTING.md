# 贡献指南

感谢你考虑为 ChatDeepSeek CLI 贡献代码！

## 如何贡献

### 报告 Bug

1. 使用 GitHub Issues 提交 Bug 报告
2. 描述复现步骤、期望行为和实际行为
3. 附上 Python 版本和操作系统信息

### 提交代码

1. **Fork** 本仓库
2. 创建特性分支：`git checkout -b feature/你的功能名`
3. 编写代码并添加测试
4. 运行测试：`python test_core.py`
5. 提交：`git commit -m "feat: 你的功能描述"`
6. 推送：`git push origin feature/你的功能名`
7. 发起 **Pull Request**

### 代码风格

- 类型注解（已有）
- 函数文档字符串（已有）
- 变量/函数用 `snake_case`，类用 `PascalCase`

### 测试

核心逻辑测试在 `test_core.py`，不需要 API Key 即可运行：

```bash
python test_core.py
```

## 开发环境搭建

```bash
git clone https://github.com/KelyCheng07/chat_DS.git
cd chat_DS
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env 填入测试用 API Key
```

## Commit 规范

建议使用语义化提交信息：

- `feat:` 新功能
- `fix:` Bug 修复
- `docs:` 文档更新
- `refactor:` 代码重构
- `test:` 测试相关
- `chore:` 构建/工具相关
