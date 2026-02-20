# Mimic Master API Specification

**Base URL:** `http://<host>:8000/api/v1`

---

## 1. DM Agent API (暴露给用户)

### POST /agent

Process query with three-layer memory system.

**Request:**
```json
{
  "query": "What is Fireball damage?",
  "session_id": "session-123"
}
```

**Response:**
```json
{
  "response": "DM response text...",
  "retrieved_context": null,
  "session_id": "session-123"
}
```

---

## 2. 需要服务机提供的 API

服务机运行 embedding 和 reranker 模型，需提供以下接口：

### POST /embeddings

生成文本嵌入向量（同时返回 dense 和 sparse）。

**Request:**
```json
{
  "texts": ["text1", "text2"]
}
```

**Response:**
```json
{
  "embeddings": [
    {
      "dense": [0.1, 0.2, ..., 0.5],
      "sparse": {
        "indices": [10, 45, 16, ...],
        "values": [0.5, 0.2, 0.8, ...]
      }
    }
  ],
  "dense_dimension": 1024
}
```

**字段说明:**
- `dense`: Dense embedding 向量
- `sparse.indices`: 非零元素的索引
- `sparse.values`: 对应索引的值

### POST /reranker

对文档进行重排序。

**Request:**
```json
{
  "query": "query text",
  "documents": ["doc1", "doc2", "doc3"],
  "top_n": 3
}
```

**Response:**
```json
{
  "results": [0, 2],
  "scores": [0.95, 0.82]
}
```

---

## Pinecone Namespace 规范

| Namespace | 用途 |
|-----------|-------|
| `rules` | D&D 规则、法术、战斗 |
| `episodes` | 会话摘要 |
| `monsters` | 怪物图鉴 |
| `spells` | 法术描述 |

---

**Version:** 0.1.0
