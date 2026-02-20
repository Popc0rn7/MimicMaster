# Mimic Master API Specification

**Base URL:** `http://<host>:8000/api/v1`

---

## 1. Embedding Service

### POST /embeddings

Generate embeddings for texts.

**Request:**
```json
{
  "texts": ["text1", "text2"]
}
```

**Response:**
```json
{
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "dimension": 1024
}
```

---

## 2. Reranker Service

### POST /reranker

Rerank documents by query relevance.

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

## 3. Retrieval Service

### POST /retrieval

Query vector database.

**Request:**
```json
{
  "query": "search query",
  "top_k": 10,
  "filter": {"category": "combat"},
  "namespace": "rules"
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "doc-001",
      "content": "document content",
      "score": 0.89,
      "metadata": {"source": "PHB", "page": 190}
    }
  ],
  "total": 1,
  "query": "search query"
}
```

---

### POST /retrieval/upsert

Index documents.

**Request (Form Data):**
- `ids`: string[]
- `texts`: string[]
- `namespace`: string (default: "")
- `metadata`: object[] (optional)

**Response:**
```json
{
  "message": "Successfully upserted 5 documents"
}
```

---

## 4. DM Agent

### POST /agent

Process query with three-layer memory.

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

## Namespaces

| Namespace | Purpose |
|-----------|---------|
| `rules` | D&D rules, spells, combat |
| `episodes` | Session summaries |
| `monsters` | Monster stat blocks |
| `spells` | Spell descriptions |

---

**Version:** 0.1.0
