# 项目背景 (Project Context)
本项目是一个为《龙与地下城 (D&D 5E)》设计的地下城主 (DM) 辅助 AI Agent。
核心技术栈：
- 语言：Python 3.12
- 项目管理: uv
- 服务框架：FastAPI (轻量级、性能优越、易于开发和部署)
- agent框架：LangSmith (提供强大的工具和接口，适合构建复杂的 agent 系统)
- 向量数据库：Pinecone (Serverless)
- 远程服务：embedding和reranker部署在局域网其他机器上，通过HTTP API调用，暂时不可用，本项目可以设计好接口并mock一个简单替代
- Embedding 模型：BAAI/bge-m3 (本地运行，双路输出 Dense & Sparse)
- Reranker 模型：BAAI/bge-reranker-v2-m3 (本地运行)
- 架构风格：Vibe Coding (偏好轻量级、可读性高、可用优先、容错其次、直觉驱动的代码，避免过度封装)

# 开发规范 (Coding Standards)
1. 类型提示 (Type Hints)：所有新编写的 Python 函数和方法必须包含明确的参数和返回值类型提示。
2. 文档管理：记录开发进度在PROGRESS.md中,大的架构改动再记录在CLAUDE.md中。
3. 依赖管理：尽量保持依赖精简。核心依赖为 `FlagEmbedding` 和 `pinecone-client`。
4. 项目管理：使用 `uv` 进行项目管理，并且保持python项目的标准结构。
5. 环境管理：相关环境变量（如 Pinecone API Key）应通过 `.env` 文件管理，并且不应直接硬编码在代码中。

# 终端操作偏好 (CLI Preferences)
- 在创建新文件或修改现有核心逻辑前，请先向我简述你的方案。
- 如果你需要运行测试脚本（如 `python test/retrieval.py`），你可以直接运行，无需询问。
- 需要添加apikey和选择模型时，给我推荐并征求我意见等我补充在`.env`