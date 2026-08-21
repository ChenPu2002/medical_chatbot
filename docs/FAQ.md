# ❓ 常见问题（FAQ）

## OpenAI API 在香港无法访问怎么办？

项目最初于 2024 年 4 月在香港开发，彼时 OpenAI API 可直接访问。截至 2024 年底，香港地区访问 OpenAI API 已受限。如果你在受限地区：

- 使用 VPN 访问 OpenAI API；或
- 改用其他 API 提供商；或
- 只使用 **Tree Model（本地模型）** 模式，完全离线可用，无需任何外部网络访问。

## 为什么首次启动这么慢？

首次加载会导入 PyQt5 相关模块、读取 CSV 训练数据并加载序列化的 `rfc.model`，大约需要 1 分钟。之后由于文件系统缓存等原因，后续启动通常少于 10 秒。

## Tree Model 和 API Model 有什么区别？我该用哪个？

- **Tree Model**：基于本地训练的 Random Forest / SVM，完全离线，响应稳定但对话形式较固定（症状选择式交互）。
- **API Model**：基于 OpenAI GPT-3.5-turbo，对话更自然、更有上下文理解能力，但需要联网和有效的 API Key，会产生调用费用。

日常开发/演示建议优先用 Tree Model；需要更自然的问诊体验且能访问 OpenAI API 时再切换到 API Model。

## RAG 功能（`RAG_code/`）为什么单独放置，不集成到主程序？

RAG（Retrieval-Augmented Generation）版本使用 OpenAI Assistants API 的 `file_search`，需要额外的向量库创建和维护，调用成本明显高于普通 Chat Completions。为了保持主程序的成本和性能可控，RAG 相关代码被拆分到独立目录，作为可选的实验性功能按需使用，详见 [ARCHITECTURE.md](./ARCHITECTURE.md)。

## PyQt5 在 Linux/Windows 上启动有问题怎么办？

- **Linux**：确认已安装 Qt5 相关系统依赖（GUI 库、`libgl1` 等），并使用与 `environment.yml` 中一致的 `PyQt5==5.15.10` 版本。
- **Windows**：功能完全可用，个别控件尺寸可能与 macOS 略有差异，属已知的显示细节问题，不影响功能。
- **macOS**：官方推荐平台，体验最佳。

## 依赖版本不一致怎么办（conda vs pip）？

`environment.yml` 是 conda 环境的权威依赖清单；`requirements.txt` / `requirements-test.txt` 是给 CI 与 pip 用户使用的等价清单。修改任何依赖版本时，请同时更新这两处，避免本地环境和 CI 环境行为不一致。详见 [DEVELOPMENT.md](./DEVELOPMENT.md)。

## 提交 PR 后有很多自动检查/机器人评论，是怎么回事？

仓库配置了多个 GitHub Actions 工作流（Lint、测试、依赖漏洞扫描、密钥扫描、CodeQL）以及 Dependabot、Stale Bot、自动打标签等自动化。完整说明见 [CI_CD.md](./CI_CD.md)。
