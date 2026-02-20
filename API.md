# Mimic Master API Documentation

## Overview

Mimic Master is a D&D 5E Dungeon Master AI Agent Assistant with three-layer memory architecture.

**Base URL:** `http://<host>:8000`
**API Version:** `v1`
**Version:** `0.1.0`

---

## Authentication

Currently no authentication is required. Add authentication before deploying to production.

---

## API Endpoints

### 1. Health Check

#### `GET /api/v1/health`

Check if the service is running and view service configurations.

**Response:**

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "services": {
    "pinecone": true,
    "langsmith": false,
    "embedding_mock": false,
    "reranker_mock": false
  }
}
```

---

### 2. Embedding Service

#### `POST /api/v1/embeddings`

Generate embeddings for the given texts.

**Request Body:**

```json
{
  "texts": ["Fireball spell description", "Longsword damage"]
}
```

**Response:**

```json
{
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "dimension": 1024
}
```

**Fields:**

| Field | Type | Required | Description |
|-------|------|-----------|-------------|
| texts | `string[]` | Yes | List of texts to embed (min 1) |

---

### 3. Reranker Service

#### `POST /api/v1/reranker`

Rerank documents based on query relevance.

**Request Body:**

```json
{
  "query": "What is the damage for Fireball?",
  "documents": [
    "Fireball is a 3rd-level evocation spell...",
    "Longsword deals 1d8 slashing damage...",
    "Shield spell grants +5 AC..."
  ],
  "top_n": 3
}
```

**Response:**

```json
{
  "results": [0, 1],
  "scores": [0.95, 0.82]
}
```

**Fields:**

| Field | Type | Required | Description |
|-------|------|-----------|-------------|
| query | `string` | Yes | Query text |
| documents | `string[]` | Yes | List of documents to rerank |
| top_n | `number` | No | Number of top results to return (default: all) |

---

### 4. Retrieval Service

#### `POST /api/v1/retrieval`

Retrieve relevant documents from the vector database.

**Request Body:**

```json
{
  "query": "How does cover work in combat?",
  "top_k": 5,
  "filter": {
    "category": "combat"
  },
  "namespace": "rules"
}
```

**Response:**

```json
{
  "results": [
    {
      "id": "rule-001",
      "content": "Cover provides +2 AC against ranged attacks...",
      "score": 0.89,
      "metadata": {
        "source": "Player's Handbook",
        "page": 190
      }
    }
  ],
  "total": 1,
  "query": "How does cover work in combat?"
}
```

**Fields:**

| Field | Type | Required | Description |
|-------|------|-----------|-------------|
| query | `string` | Yes | Query text |
| top_k | `number` | No | Number of results (1-100, default: 10) |
| filter | `object` | No | Metadata filter (e.g., `{"category": "combat"}`) |
| namespace | `string` | No | Pinecone namespace (default: "") |

---

#### `POST /api/v1/retrieval/upsert`

Upsert documents into the vector database.

**Request Body (Form Data):**

| Field | Type | Required | Description |
|-------|------|-----------|-------------|
| ids | `string[]` | Yes | List of document IDs |
| texts | `string[]` | Yes | List of document contents |
| namespace | `string` | No | Namespace for the documents (default: "") |
| metadata | `object[]` | No | Optional list of metadata dictionaries |

**Response:**

```json
{
  "message": "Successfully upserted 5 documents"
}
```

---

### 5. DM Agent

#### `POST /api/v1/agent`

Process a query through the DM agent with three-layer memory system.

**Request Body:**

```json
{
  "query": "What is the attack roll modifier for a Level 5 fighter?",
  "session_id": "session-123"
}
```

**Response:**

```json
{
  "response": "As a Level 5 fighter, you have a proficiency bonus of +3... [full response]",
  "retrieved_context": null,
  "session_id": "session-123"
}
```

**Fields:**

| Field | Type | Required | Description |
|-------|------|-----------|-------------|
| query | `string` | Yes | Player query or context |
| session_id | `string` | No | Session identifier for episodic retrieval |

**Agent Workflow:**

1. **Intent Classification** - Determines if user is asking about rules, combat, story, etc.
2. **State Retrieval** - Always includes current game state and dialogue history
3. **Conditional Retrieval**:
   - Rules query → Retrieves from knowledge base (rules namespace)
   - Recall history → Retrieves from episodic memory (episodes namespace)
   - Chat/Story → Skips retrieval for faster response
4. **Context Assembly** - Formats all information into a structured prompt
5. **Response Generation** - Generates a context-aware response

---

#### `POST /api/v1/agent/with-context`

Process a query and return the full assembled context (debug mode).

**Request Body:** Same as `/api/v1/agent`

**Response:** Same as `/api/v1/agent`, with `retrieved_context` containing the full assembled context as JSON.

---

## Data Models

### RetrievedDocument

```typescript
{
  id: string;           // Document ID
  content: string;       // Document content
  score: number;        // Similarity score (0-1)
  metadata: object;      // Additional metadata (source, page, etc.)
}
```

### AgentRequest

```typescript
{
  query: string;         // Player query or context
  context?: string[];     // Additional context (deprecated, now managed internally)
  session_id?: string;    // Session identifier
}
```

### AgentResponse

```typescript
{
  response: string;           // DM response
  retrieved_context?: string[]; // Context used for generating response
  session_id?: string;        // Session identifier
}
```

---

## Error Responses

All endpoints may return the following error responses:

### 400 Bad Request

```json
{
  "detail": "Validation error: field is required"
}
```

### 500 Internal Server Error

```json
{
  "detail": "Agent processing failed: Pinecone connection error"
}
```

---

## Configuration

The service behavior can be configured via environment variables:

| Variable | Description | Default |
|-----------|-------------|----------|
| `PINECONE_API_KEY` | Pinecone API key | - |
| `PINECONE_INDEX` | Pinecone index name | - |
| `EMBEDDING_PROVIDER_URL` | Embedding service URL | `"mock"` |
| `RERANKER_PROVIDER_URL` | Reranker service URL | `"mock"` |
| `LANGSMITH_API_KEY` | LangSmith API key | - |
| `LANGSMITH_PROJECT` | LangSmith project name | - |
| `EMBEDDING_DIMENSION` | Embedding vector dimension | `1024` |

---

## Memory Architecture

### Three-Layer Memory System

| Layer | Purpose | Storage | Retrieval |
|--------|----------|----------|-----------|
| **State & Working Memory** | Current game state, dialogue history | In-memory | Always included |
| **Static Knowledge** | Rules, spells, mechanics | Pinecone | Intent-driven (query_rules/combat) |
| **Episodic Memory** | Past session summaries | Pinecone | Intent-driven (recall_history) |

### Namespaces

| Namespace | Purpose |
|-----------|---------|
| `rules` | D&D 5E rules, spells, combat mechanics |
| `episodes` | Session summaries and key events |
| `monsters` | Monster stat blocks and descriptions |
| `spells` | Spell descriptions and effects |

---

## Quick Start Examples

### 1. Health Check

```bash
curl http://localhost:8000/api/v1/health
```

### 2. Query the DM Agent

```bash
curl -X POST http://localhost:8000/api/v1/agent \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the rules for opportunity attacks?",
    "session_id": "game-001"
  }'
```

### 3. Index Documents

```bash
curl -X POST "http://localhost:8000/api/v1/retrieval/upsert?namespace=rules" \
  -F 'ids=["rule-001","rule-002"]' \
  -F 'texts=["Opportunity attack rule...","Cover rule..."]'
```

### 4. Retrieve Documents

```bash
curl -X POST http://localhost:8000/api/v1/retrieval \
  -H "Content-Type: application/json" \
  -d '{
    "query": "opportunity attack",
    "top_k": 3,
    "namespace": "rules"
  }'
```

---

## Notes

- The agent uses a three-layer memory system to maintain context across conversations
- Embedding and reranker services can be mocked by setting the provider URL to `"mock"`
- Use `session_id` to maintain episodic memory across multiple queries
- The `/with-context` endpoint is useful for debugging and understanding what context was used

---

**Last Updated:** 2026-02-20
**API Version:** 0.1.0
