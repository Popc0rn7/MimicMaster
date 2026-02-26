# Mimic Master 架构设计文档

## 1. 系统架构概述

Mimic Master 是一个为《龙与地下城 (D&D 5E)》地下城主 (DM) 辅助的 AI Agent 系统。

### 1.1 核心组件

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI App                          │
├─────────────────────────────────────────────────────────────┤
│  Routes: /health, /retrieval, /embedding, /reranker, /agent │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  Memory       │    │   Services   │    │   Core        │
│  Module       │    │   Layer      │    │   Logic       │
├───────────────┤    ├───────────────┤    ├───────────────┤
│ - Knowledge   │    │ - Pinecone   │    │ - Retriever   │
│   Retriever   │    │ - Embedding  │    │ - Agent       │
│ - Episodic    │    │ - Reranker  │    │ - Assembler   │
│ - State       │    │ - MongoDB    │    │               │
└───────────────┘    └───────────────┘    └───────────────┘
                              │
                              ▼
                    ┌───────────────┐
                    │  External     │
                    │  Services     │
                    ├───────────────┤
                    │ - Pinecone    │
                    │ - BGE-M3      │
                    │ - BGE-Reranker│
                    │ - LLM API     │
                    └───────────────┘
```

### 1.2 数据流

1. **用户请求** → FastAPI Routes
2. **意图分类** → Intent Classifier
3. **知识检索** → Knowledge Retriever (Pinecone)
4. **上下文组装** → Assembler
5. **Agent 执行** → LangSmith Agent
6. **响应返回**

---

## 2. 知识库设计

### 2.1 Namespace 划分

按数据类型划分 namespace：

| Namespace | 含义 | 数据来源 |
|-----------|------|----------|
| `rules` | 核心规则 (职业、种族、技能、DM指南等) | rag_phb.txt, rag_dmg.txt |
| `monsters` | 怪物数据 | rag_mm.jsonl |
| `spells` | 法术数据 | (未来扩展) |
| `episodes` | 战役/剧集 | (用户自定义) |

### 2.2 Metadata 结构

```python
class KnowledgeMetadata:
    # 基础信息
    category: str          # monster, rule, dm_guide, spell
    source_book: str      # PHB, MM, DMG
    chapter: str           # 章节

    # 怪物特有
    name: str             # 名称
    monster_type: str      # aberration, dragon, undead...
    size: str             # tiny, small, medium, large, huge, gargantuan
    alignment: str         # 阵营
    cr: str               # 挑战等级

    # 规则特有
    rule_type: str         # class, spell, race, background, feat...
    section: str           # 章节名

    # DM 指南特有
    dm_type: str          # treasure, trap, encounter, variant...

    # 图片
    has_image: bool        # 是否有图片
    image_path: str        # 图片路径
```

### 2.3 图片处理流程

1. **检测**: 识别 `[IMG:xxx]` 占位符
2. **调用视觉模型**: 使用 BLIP 生成图片描述
3. **替换**: 将占位符替换为描述文本
4. **索引**: 将处理后的完整文本索引到 Pinecone

---

## 3. 服务层设计

### 3.1 Pinecone Service

负责向量数据库操作：
- `create_index()` - 创建索引
- `upsert()` - 插入/更新向量
- `query()` - 向量检索 (支持 hybrid search)

### 3.2 Embedding Service

负责文本向量化：
- 本地模式：FlagEmbedding (BGE-M3)
- 远程模式：HTTP API 调用
- Mock 模式：哈希生成伪向量

### 3.3 Reranker Service

负责结果重排序：
- 本地模式：FlagReranker (BGE-Reranker-v2-M3)
- 远程模式：HTTP API 调用
- Mock 模式：跳过重排序

---

## 4. 检索流程

### 4.1 Hybrid Knowledge Retrieval

```
Query Text
    │
    ▼
┌─────────────────┐
│  BGE-M3 Encode │
│  (Dense + Sparse)│
└─────────────────┘
    │
    ├── Dense Vec ──┐
    │                │
    ▼                ▼
┌──────────────────────┐
│  Pinecone Query      │
│  (Hybrid Search)     │
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│  BGE-Reranker       │
│  (Optional)         │
└──────────────────────┘
    │
    ▼
Results with Scores
```

### 4.2 检索参数

```python
await retriever.retrieve(
    query="强大的水下怪物",
    namespace="monsters",           # 指定 namespace
    top_k=5,                        # 返回数量
    filter={"source_book": "MM"},   # Metadata 过滤
    use_rerank=True                 # 是否重排序
)
```

---

## 5. API 路由设计

### 5.1 路由列表

| Method | Path | 描述 |
|--------|------|------|
| GET | /health | 健康检查 |
| POST | /retrieval/knowledge | 知识检索 |
| POST | /retrieval/episodic | 记忆检索 |
| POST | /embedding | 文本向量化 |
| POST | /reranker | 结果重排序 |
| POST | /agent/chat | Agent 对话 |
| GET | /frontend/games | 游戏列表 |
| POST | /frontend/games | 创建游戏 |

---

## 6. 配置管理

### 6.1 环境变量

```bash
# Pinecone
PINECONE_API_KEY=xxx
PINECONE_INDEX=mimic-rules-prod

# Embedding
EMBEDDING_PROVIDER_URL=http://192.168.1.x:8000/embeddings
EMBEDDING_DIMENSION=1024

# Reranker
RERANKER_PROVIDER_URL=http://192.168.1.x:8000/reranker

# Namespace (可选)
RULES_NAMESPACE=rules
MONSTERS_NAMESPACE=monsters
SPELLS_NAMESPACE=spells

# LangSmith
LANGSMITH_API_KEY=xxx
LANGSMITH_TRACING=true
```

### 6.2 配置类

所有配置通过 `Settings` 类管理，读取自环境变量和 `.env` 文件。

---

## 7. 依赖关系

### 7.1 核心依赖

```toml
# pyproject.toml
dependencies = [
    "fastapi>=0.109.0",
    "uvicorn>=0.27.0",
    "pinecone-client>=5.0.0",
    "pydantic>=2.5.0",
    "httpx>=0.26.0",
    "python-dotenv>=1.0.0",
]

# 可选依赖
optional-dependencies = {
    "embedding": ["FlagEmbedding>=1.2.0"],
    "reranker": ["FlagEmbedding>=1.2.0"],
    "mongodb": ["motor>=3.3.0"],
    "langsmith": ["langsmith>=0.1.0"],
}
```

---

## 8. 脚本工具

### 8.1 管理脚本

| 脚本 | 用途 |
|------|------|
| `scripts/setup_pinecone.py` | 创建 Pinecone 索引 |
| `scripts/index_knowledge.py` | 索引知识库 |
| `scripts/test_retrieval.py` | 测试检索功能 |

### 8.2 使用示例

```bash
# 创建索引
uv run python scripts/setup_pinecone.py

# 索引知识库
uv run python scripts/index_knowledge.py --source all
uv run python scripts/index_knowledge.py --source mm --limit 10

# 测试检索
uv run python -c "import asyncio; ..."
```
