# Mimic Master 命名规范

本文档定义了 Mimic Master 项目中 Pinecone 向量数据库和 LangSmith 追踪的命名规范。

## 项目标识

| 层级 | 标识符 | 说明 |
|------|--------|------|
| **项目名** | `mimic-master` | DM Agent 项目名称 |
| **服务前缀** | `mimic` | 所有资源的统一前缀 |

---

## Pinecone Index 命名

### Index 结构

```
mimic-{domain}-{environment}
```

### 命名规则

| 部分 | 格式 | 示例 |
|------|------|------|
| 项目前缀 | `mimic-` | `mimic-` |
| 领域 | `{domain}` | `rules`, `episodes` |
| 环境 | `-{environment}` | `-dev`, `-staging`, `-prod` |

### 建议的 Index 名称

| 环境 | Index 名称 |
|------|----------|
| 开发 | `mimic-rules-dev`, `mimic-episodes-dev` |
| 测试 | `mimic-rules-staging`, `mimic-episodes-staging` |
| 生产 | `mimic-rules-prod`, `mimic-episodes-prod` |

---

## Pinecone Namespace 命名

### Namespace 结构

```
{content-type}
```

### 标准命名空间

| Namespace | 用途 | 数据类型 |
|-----------|-------|----------|
| `rules` | D&D 5E 规则 | Dense + Sparse |
| `episodes` | 会话摘要和关键事件 | Dense only |
| `monsters` | 怪物图鉴和属性 | Dense + Sparse |
| `spells` | 法术描述和效果 | Dense + Sparse |
| `items` | 物品和装备 | Dense + Sparse |
| `npcs` | NPC 角色和背景 | Dense + Sparse |
| `locations` | 地点和场景描述 | Dense + Sparse |
| `campaign` | 战役和世界设定 | Dense only |

### Namespace 使用示例

```python
# 检索规则（混合搜索）
await retriever.retrieve(
    query="fireball damage",
    namespace="rules",
)

# 检索历史（密集搜索）
await episodic_retriever.retrieve(
    query="previous goblin fight",
    namespace="episodes",
)

# 索引文档
await pinecone_service.upsert_from_texts(
    ids=["rule-001", "rule-002"],
    texts=["Fireball deals 8d6...", "Magic Missile deals 3d4..."],
    namespace="rules",
)
```

---

## LangSmith 命名

### Project 命名

```
mimic-master
```

### 运行命名

| 运行环境 | Project 名称 |
|----------|----------|
| 开发 | `mimic-master-dev` |
| 测试 | `mimic-master-staging` |
| 生产 | `mimic-master` |

### LangSmith 配置示例

```bash
# .env
LANGSMITH_PROJECT=mimic-master
LANGSMITH_API_KEY=your_key_here
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

---

## 元数据规范

### 文档元数据

```json
{
  "source": "PHB",
  "page": 101,
  "chapter": "Combat",
  "version": "5E",
  "tags": ["evocation", "fire", "damage"],
  "content_type": "rule"
}
```

### Episode 元数据

```json
{
  "session_id": "game-001",
  "dm": "John Doe",
  "date": "2026-02-22",
  "location": "Tavern of the Broken Shield",
  "tags": ["combat", "goblins", "discovery"]
}
```

### Character 元数据

```json
{
  "player_name": "Alice",
  "race": "Human",
  "class": "Fighter",
  "level": 3,
  "created_at": "2026-02-01"
}
```

---

## 配置文件示例

### .env 配置

```bash
# Pinecone 配置
PINECONE_API_KEY=your_api_key_here
PINECONE_INDEX=mimic-rules-prod

# 命名空间配置
RULES_NAMESPACE=rules
EPISODES_NAMESPACE=episodes
MONSTERS_NAMESPACE=monsters
SPELLS_NAMESPACE=spells

# Embedding 配置
EMBEDDING_PROVIDER_URL=http://your-embedding-service:8000/embeddings
EMBEDDING_DIMENSION=1024

# Reranker 配置
RERANKER_PROVIDER_URL=http://your-reranker-service:8000/reranker

# LangSmith 配置
LANGSMITH_API_KEY=your_api_key_here
LANGSMITH_PROJECT=mimic-master
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_TRACING=true
```

---

## 命名检查清单

- [ ] Pinecone index 使用 `mimic-` 前缀
- [ ] Namespace 使用标准命名（rules, episodes 等）
- [ ] LangSmith project 名称为 `mimic-master`
- [ ] 所有元数据包含 `source` 字段
- [ ] Episode ID 使用 `{session_id}-{timestamp}` 格式
- [ ] 文档 ID 使用 `{namespace}-{hash}` 格式

---

**最后更新:** 2026-02-20
