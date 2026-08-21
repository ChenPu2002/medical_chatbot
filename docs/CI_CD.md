# 🤖 CI/CD 与自动化 Bot 说明

仓库在 `.github/` 下配置了以下自动化工作流，均在 `push` / `pull_request`（或按计划）触发。提交 PR 后会看到对应的状态检查（checks）。

## GitHub Actions 工作流

| 工作流文件 | 触发条件 | 作用 |
|-----------|---------|------|
| `pylint.yml` | 变更 `*.py`/`.pylintrc` 时 push/PR | 在 Python 3.8/3.9/3.10 上运行 `pylint`，低于 `.pylintrc` 中 `fail-under=7.0` 阈值则失败 |
| `tests.yml` | 变更 `*.py`/`requirements*.txt` 时 push/PR | 运行 `pytest`（`tests/` 目录），并生成对 `tree_model_medicine` 模块的覆盖率报告作为 artifact |
| `pip-audit.yml` | 变更 `requirements*.txt`/`environment.yml` 时 push/PR，另外每周一定时执行 | 用 `pip-audit` 扫描 Python 依赖的已知漏洞 |
| `gitleaks.yml` | 每次 push/PR | 用 Gitleaks 扫描仓库历史/改动中的密钥泄露 |
| `codeql.yml` | push/PR 到 `main`，另外每周一定时执行 | GitHub CodeQL 对 Python 代码做静态安全分析 |
| `labeler.yml` | PR 打开/更新 | 根据改动路径（见 `.github/labeler.yml`）自动为 PR 打标签，如 `rag`、`data`、`ci/cd`、`tests`、`gui` |
| `stale.yml` | 每日定时 | 60 天无活动的 issue/PR 标记为 `stale`，再过 14 天无响应则自动关闭 |

## Dependabot

`.github/dependabot.yml` 每周检查两类生态并自动开 PR：

- `pip`：`requirements.txt` / `requirements-test.txt` 中的依赖版本更新
- `github-actions`：workflow 中使用的各个 Action 版本更新

这些 PR 需要人工 review 后合并，不会自动合并。

## 对开发流程的影响

- **提交 PR 时**：Pylint、Tests、pip-audit（若改了依赖文件）、Gitleaks 会自动运行，尽量在合并前拦截风格、测试、依赖漏洞和密钥泄露问题。
- **合并到 `main` 后**：CodeQL 会做一次静态安全分析；同时每周还有一次针对 `main` 的定时安全扫描和依赖漏洞扫描，不依赖是否有新提交。
- **日常维护**：Dependabot 和 Stale Bot 会持续产生 PR/标记，需要定期查看和处理，避免堆积。

## 本地复现 CI 检查

```bash
# Lint
pip install pylint
pylint $(git ls-files '*.py')

# 测试 + 覆盖率
pip install -r requirements-test.txt
pytest --cov=tree_model_medicine --cov-report=term tests/

# 依赖漏洞扫描
pip install pip-audit
pip-audit -r requirements-test.txt
```
