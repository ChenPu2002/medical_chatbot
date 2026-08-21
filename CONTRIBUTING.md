# 🤝 贡献指南

感谢你对 ChatDoctor 项目感兴趣！在提交贡献之前，请阅读以下指引。

## 开发环境搭建

请参阅 [docs/DEVELOPMENT.md](./docs/DEVELOPMENT.md)，其中包含 conda/pip 两种环境搭建方式、运行应用、重新训练模型、运行测试与 lint 的说明。

## 贡献流程

1. Fork 本仓库
2. 基于 `main` 创建功能分支（如 `feature/xxx` 或 `fix/xxx`）
3. 进行修改，并为新增的纯逻辑函数补充单元测试（放在 `tests/` 目录下）
4. 本地运行以下检查，确保全部通过：
   ```bash
   pytest tests/ -v
   pylint $(git ls-files '*.py')
   ```
5. 提交 Pull Request

## PR 会自动触发哪些检查

打开 PR 后，以下 GitHub Actions 工作流会自动运行（详见 [docs/CI_CD.md](./docs/CI_CD.md)）：

- **Pylint**：代码风格与静态检查（Python 3.8/3.9/3.10）
- **Tests**：`pytest` 单元测试 + 覆盖率
- **Dependency Vulnerability Scan（pip-audit）**：仅当依赖文件变更时触发
- **Secret Scanning（gitleaks）**：检测是否误提交密钥
- 合并后还会触发 **CodeQL** 静态安全分析

请确保这些检查全部通过后再请求 review。若 PR 修改了依赖版本，请同步更新 `environment.yml` 与 `requirements.txt` / `requirements-test.txt`。

## 提交规范

- 提交信息请清晰描述改动内容
- 避免在一次提交中混合无关的改动
- 不要提交 `.env` 或任何包含真实密钥的文件；如需示例配置，请更新 `.env.example`

## 报告问题

如发现 Bug 或有功能建议，请在 Issues 中提交，并尽量提供复现步骤、期望行为与实际行为。
