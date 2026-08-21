# 🗂️ 数据与模型说明

## `data/` 目录

| 文件/目录 | 说明 |
|-----------|------|
| `training.csv` | 主训练集：132 个症状特征列 + `prognosis`（疾病标签）列，`tree_model_medicine.py` 与 `disease_prediction_model_generator.py` 都依赖它 |
| `dataset.csv` | 与 `training.csv` 结构相同的原始/备用数据集 |
| `symptom_Description.csv` | 疾病名称 -> 疾病描述文本，两列：`Disease, Description` |
| `symptom_precaution.csv` | 疾病名称 -> 最多 4 条注意事项，列：`Disease, Precaution_1..4` |
| `medicine_use.csv` | 药品名称 -> 最多多条用途说明，供 `middleware.response_maker_med` 模糊搜索使用 |
| `fuzzy_dictionary_unique.txt` | 症状/术语词典，供 `spellwise.Levenshtein` 做拼写纠错（`TreePredictor.fuzzy_searcher`） |
| `rag_data/*.txt` | 医学教材文本（解剖学、生理学、内科学等），供 `RAG_code/api_model_rag.py` 建立向量库使用，与主流程无关 |

> ⚠️ 修改 `training.csv` 的列结构（症状特征）后，必须重新运行 `disease_prediction_model_generator.py` 生成新模型，否则 `tree_model_medicine.py` 中 `symptoms_dict` 与模型输入维度会不匹配。

## `model/` 目录

| 文件 | 说明 |
|------|------|
| `rfc.model` | `RandomForestClassifier`，`tree_model_medicine.get_advise` 实际使用的预测模型 |
| `svc.model` | `SVC`（支持向量机），由训练脚本一并生成，主要用于 `inference_model_training.ipynb` 中的模型效果对比，当前主流程未直接调用 |

两者均通过 `joblib.dump` / `joblib.load` 序列化，重新训练时会被覆盖。

## 数据来源

- 训练数据集与药品/症状说明来自 Kaggle 公开的症状-疾病数据集（详见根目录 README 的 Acknowledgments）。
- `rag_data/` 中的教材文本仅用于 RAG 实验功能，属于可选扩展内容。
