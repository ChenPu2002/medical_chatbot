# 🏗️ 架构说明

## 整体结构

ChatDoctor 是一个基于 PyQt5 的桌面聊天机器人，支持两种问诊模式：

1. **Tree Model（本地机器学习模型）**：基于 `RandomForestClassifier` / `SVC`，通过症状匹配预测疾病，完全离线运行。
2. **API Model（OpenAI GPT）**：调用 OpenAI `gpt-3.5-turbo` 接口进行更自然的多轮对话。

`RAG_code/` 目录额外提供了一套基于检索增强生成（RAG）的实验性实现，使用 OpenAI Assistants API 的 `file_search` 工具从医学教材（`data/rag_data/`）中检索内容后再生成回答，与主程序解耦，按需使用。

## 模块与职责

```
main.py
  └── gui.py (ChatWindow)
        └── middleware.py
              ├── TreePredictor  ──> tree_model_medicine.py ──> model/rfc.model
              └── APIPredictor   ──> api_model.py            ──> OpenAI API

RAG_code/ (可选，独立于主流程)
  ├── middleware_rag.py  (RAG 版 TreePredictor)
  └── api_model_rag.py   (向量库创建 + Assistants API 调用)
```

| 模块 | 职责 |
|------|------|
| `main.py` | 应用入口，创建 `QApplication` 并启动 GUI |
| `gui.py` | PyQt5 界面：聊天窗口、模型切换下拉框、消息渲染（内嵌 HTML/JS 通过 `QWebEngineView`） |
| `middleware.py` | 两个核心类：`TreePredictor`（本地模型的多轮状态机）与 `APIPredictor`（OpenAI 对话封装），根据用户在 GUI 里选择的模式路由请求 |
| `tree_model_medicine.py` | 加载 `data/training.csv` 与 `model/rfc.model`，提供症状模糊匹配（`check_pattern`/`get_poss_symptom`）、关联症状推荐（`first_predict`）与最终诊断+建议生成（`get_advise`） |
| `api_model.py` | 使用 `openai` SDK 调用 `gpt-3.5-turbo`，维护对话历史 |
| `disease_prediction_model_generator.py` | 离线脚本：读取 `data/training.csv` 训练 `RandomForestClassifier`/`SVC` 并导出到 `model/` |
| `inference_model_training.ipynb` | 模型效果分析（准确率对比、特征重要性、交叉验证可视化） |
| `RAG_code/` | RAG 增强版的 middleware 与 API 封装，依赖 `data/rag_data/` 中的医学教材文本作为向量库来源 |

## Tree Model 的多轮对话状态机

`TreePredictor.response_maker` 通过 `self.count` 字段维护一个简单的状态机，大致流程：

1. `count == 0`：接收初始症状描述，做拼写模糊匹配（`fuzzy_searcher`）+ 候选症状匹配（`get_poss_symptom`）。
2. `count == 1`：用户从候选列表中选择具体症状后，`first_predict` 根据训练集中同类疾病的高频共现症状，推荐更多相关症状供用户勾选。
3. 后续轮次持续收集症状到 `user_report`，当收集足够信息后调用 `get_advise`，用训练好的 `rfc.model` 做预测，并从 `symptom_Description.csv` / `symptom_precaution.csv` 中查出对应的疾病描述与注意事项。

`response_maker_med` 是另一条独立的状态机，用于药品名称查询（模糊匹配 `data/medicine_use.csv` 中的药品名并返回用途说明）。

## 数据流概览

```
用户输入 (GUI)
   │
   ▼
middleware.TreePredictor / APIPredictor
   │
   ├─ Tree: tree_model_medicine.py ── data/training.csv, model/rfc.model,
   │                                   data/symptom_Description.csv,
   │                                   data/symptom_precaution.csv
   │
   └─ API:  api_model.py ── OpenAI Chat Completions API
   │
   ▼
gui.py 渲染回复到聊天窗口
```

更多关于每个数据文件的说明，请参阅 [DATA.md](./DATA.md)。
