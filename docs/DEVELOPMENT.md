# 🛠️ 开发指南

## 环境搭建

项目以 `environment.yml`（conda）为权威依赖清单；仓库同时提供了 `requirements.txt` / `requirements-test.txt`（pip），供 CI 以及不使用 conda 的开发者使用。两者应保持依赖版本一致，修改依赖时请同步更新。

### 方式一：Conda（推荐，与 CI 保持环境一致时优先）

```bash
conda env create -f environment.yml
conda activate medical_chatbot
```

### 方式二：pip + venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-test.txt   # 包含运行依赖 + pytest/pytest-cov
```

### 配置 OpenAI API Key

复制 `.env.example` 为 `.env` 并填入你的 Key（`.env` 已被 `.gitignore` 忽略，不会被提交）：

```bash
cp .env.example .env
# 编辑 .env，将 OPENAI_API_KEY 替换为真实的 key
```

> 仅使用 Tree Model 时可以不配置 API Key。

## 运行应用

```bash
python main.py
```

首次加载模型可能需要约 1 分钟，之后加载少于 10 秒。

## 重新训练本地模型

```bash
python disease_prediction_model_generator.py
```

会读取 `data/training.csv` 重新训练 `RandomForestClassifier` 与 `SVC`，并覆盖写入 `model/rfc.model`、`model/svc.model`。

## 模型效果分析

用 Jupyter 打开 `inference_model_training.ipynb`，查看准确率对比、特征重要性和交叉验证结果。

## 运行测试

```bash
pytest tests/ -v
# 或带覆盖率
pytest --cov=tree_model_medicine --cov-report=term tests/
```

当前 `tests/test_tree_model_medicine.py` 覆盖了 `tree_model_medicine.py` 中不依赖随机性/GUI 的纯逻辑函数（症状模糊匹配、候选症状搜索等）。新增功能时，请优先为新的纯逻辑函数补充单元测试。

## 代码规范检查（Lint）

```bash
pip install pylint
pylint $(git ls-files '*.py')
```

规则配置见根目录 `.pylintrc`（`fail-under=7.0`，最大行宽 100）。提交 PR 前请确保本地 pylint 通过，CI 中的 `Pylint` workflow 会在 PR 阶段自动检查 Python 3.8/3.9/3.10 三个版本。

## 提交前检查清单

- [ ] `pytest tests/` 全部通过
- [ ] `pylint $(git ls-files '*.py')` 达到 `fail-under=7.0` 阈值
- [ ] 未在代码中硬编码任何密钥（`.env` 不应被提交）
- [ ] 若新增/修改依赖，`environment.yml` 与 `requirements*.txt` 已同步更新

更多关于 CI 检查项和自动化 Bot 的说明，请参阅 [CI_CD.md](./CI_CD.md)。
