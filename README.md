# Mimic Master

D&D 5E 地下城主（DM）AI 助手，基于三层记忆架构的智能问答系统。

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                  DM Agent                           │
│  ┌──────────────────────────────────────────────┐   │
│  │       Context Assembler                  │   │
│  │  ┌────────────────────────────────────┐  │   │
│  │  │    Intent Classifier              │  │   │
│  │  └────────────────────────────────────┘  │   │
│  │  ┌──────────────┬───────────────────┐  │   │
│  │  │ State Memory │  Retrieval (Intent) │  │   │
│  │  │              │  ├─ Rules (Hybrid) │  │   │
│  │  │              │  └─ Episodes       │  │   │
│  │  └──────────────┴───────────────────┘  │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### 三层记忆系统

| 层级 | 用途 | 存储 | 检索触发 |
|------|------|------|-----------|
| **State & Working Memory** | 当前游戏状态、对话历史 | 内存 | 始终 |
| **Static Knowledge** | 规则、法术、战斗 | Pinecone | 规则查询 / 战斗 |
| **Episodic Memory** | 过往会话摘要 | Pinecone | 回忆往事 |

## 快速开始

### 安装依赖

```bash
uv sync
```

### 配置环境变量

创建 `.env` 文件：

```bash
# Pinecone
PINECONE_API_KEY=your_api_key
PINECONE_INDEX=your_index_name
EMBEDDING_DIMENSION=1024

# Embedding 服务（设为 "mock" 使用模拟模式）
EMBEDDING_PROVIDER_URL=http://your-embedding-server:port/embeddings

# Reranker 服务（设为 "mock" 使用模拟模式）
RERANKER_PROVIDER_URL=http://your-reranker-server:port/reranker

# LangSmith 追踪（可选）
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=mimic-master
```

### 启动服务

```bash
python main.py
```

或直接使用 uvicorn：

```bash
uv run uvicorn mimic_master.api.app:app --reload
```

服务将在 `http://localhost:8000` 启动。

## API 端点

详见 [API.md](API.md)

| 端点 | 方法 | 说明 |
|-------|------|------|
| `/api/v1/health` | GET | 健康检查 |
| `/api/v1/agent` | POST | DM Agent 对话 |
| `/api/v1/embeddings` | POST | 生成嵌入（服务机） |
| `/api/v1/reranker` | POST | 文档重排序（服务机） |
| `/api/v1/retrieval` | POST | 向量检索（混合搜索） |
| `/api/v1/retrieval/upsert` | POST | 索引文档 |

## 使用示例

### 1. DM Agent 对话

```bash
curl -X POST http://localhost:8000/api/v1/agent \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the damage for Fireball spell?",
    "session_id": "game-001"
  }'
```

### 2. 混合检索（Dense + Sparse）

```bash
curl -X POST http://localhost:8000/api/v1/retrieval \
  -H "Content-Type: application/json" \
  -d '{
    "query": "opportunity attack rules",
    "sparse_vector": {
      "indices": [10, 45, 16],
      "values": [0.5, 0.2, 0.8]
    },
    "top_k": 5,
    "namespace": "rules"
  }'
```

### 3. 索引文档

```bash
curl -X POST "http://localhost:8000/api/v1/retrieval/upsert?namespace=rules" \
  -F 'ids=["rule-001","rule-002"]' \
  -F 'texts=["Fireball is a 3rd-level evocation spell...","Cover grants +2 AC..."]'
```

## Pinecone Namespace

| Namespace | 用途 |
|-----------|-------|
| `rules` | D&D 5E 规则、法术、战斗机制 |
| `episodes` | 会话摘要和关键事件 |
| `monsters` | 怪物图鉴和数据 |
| `spells` | 法术描述和效果 |

## 项目结构

```
mimic-master/
├── mimic_master/
│   ├── api/              # FastAPI 路由
│   │   └── routers/
│   ├── core/             # DMAgent, DandDRetriever
│   ├── memory/           # 三层记忆架构
│   │   ├── state_memory.py        # 状态与工作记忆
│   │   ├── knowledge_retriever.py # 静态知识检索（混合搜索）
│   │   ├── episodic_retriever.py  # 情节记忆
│   │   ├── intent_classifier.py    # 意图分类
│   │   └── assembler.py          # 上下文组装器
│   ├── models/           # Pydantic 数据模型
│   ├── services/         # 服务层（单例模式）
│   └── config.py         # 配置管理
├── main.py             # 应用入口
├── pyproject.toml      # 项目配置
└── .env.example        # 环境变量模板
```

## 开发规范

- **类型提示**: 所有函数必须有参数和返回值类型提示
- **单例模式**: Embedding 和 Reranker 模型单例加载，避免重复加载
- **容错处理**: 所有外部调用必须有异常捕获
- **环境管理**: 使用 `.env` 管理配置，不硬编码

## CI/CD

### 自动化测试

推送代码到 `main` 分支时自动运行测试：

```bash
# 运行本地测试
uv run pytest

# 检查代码风格
uv run ruff check .
uv run black --check .
```

### 部署

#### 方式一：使用部署脚本

```bash
# 配置服务器信息
export SERVER_HOST=your-server.com
export SERVER_USER=your-username
export SERVER_PATH=/opt/mimic-master

# 执行部署
./deploy.sh
```

#### 方式二：使用 GitHub Actions

在 GitHub 仓库中配置以下 Secrets：

| Secret | 说明 |
|--------|------|
| `SSH_PRIVATE_KEY` | SSH 私钥（用于连接服务器） |
| `SERVER_HOST` | 服务器地址 |
| `SERVER_USER` | 服务器用户名 |
| `SERVER_PATH` | 服务器部署路径 |

推送到 `main` 分支后自动部署。

#### 方式三：使用 Docker

```bash
# 构建镜像
docker build -t mimic-master:latest .

# 运行容器
docker run -d \
  --name mimic-master \
  -p 8000:8000 \
  --env-file .env \
  mimic-master:latest
```

或使用 docker-compose：

```bash
docker-compose up -d --build
```

## 依赖

| 包 | 用途 |
|-----|------|
| fastapi | Web 框架 |
| pinecone | 向量数据库 |
| langsmith | 追踪和监控 |
| pydantic | 数据验证 |
| httpx | HTTP 客户端 |

## 版本

0.1.0
