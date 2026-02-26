# API 文档

## 基础信息

- **Base URL**: `http://localhost:8000` (可通过 `VITE_API_BASE_URL` 环境变量配置)
- **当前状态**: Mock 模式，所有端点返回模拟数据

---

## 对话接口

### POST /api/v1/agent

发送查询到 DM Agent

**请求体**:
```typescript
{
  query: string        // 用户输入
  session_id: string   // 会话 ID
}
```

**响应**:
```typescript
{
  response: string                    // AI 响应内容
  retrieved_context: {                // 检索到的上下文
    rules: RetrievedDocument[]       // 规则文档
    episodes: RetrievedDocument[]    // 历史记录
    total_tokens: number            // 总 token 数
  } | null
  session_id: string                 // 会话 ID
}
```

**数据库操作**:
- 读取对话历史 (dialogue_history 表)
- 读取规则库 (rules 表)
- 读取情节记忆 (episodes 表)
- 写入新对话记录 (dialogue_history 表)

---

### POST /api/v1/agent/with-context

调试模式，返回完整上下文信息

**请求体**: 同 `/api/v1/agent`

**响应**:
```typescript
{
  response: string
  context: {
    system_prompt: string        // 系统提示词
    state_context: string        // 游戏状态上下文
    knowledge_context: string     // 知识库上下文
    episodic_context: string     // 情节上下文
    dialogue_history: string     // 对话历史
  }
}
```

**数据库操作**:
- 读取会话状态 (sessions 表)
- 读取场景信息 (scenes 表)
- 读取玩家状态 (players 表)
- 读取 NPC 信息 (npcs 表)

---

## 会话管理接口

### GET /api/v1/session/{session_id}/state

获取指定会话的完整游戏状态

**响应**:
```typescript
{
  session_id: string
  current_scene: {
    location: string
    time: string
    weather: string
  }
  players: Player[]       // 玩家列表
  npcs: NPC[]            // NPC 列表
}
```

**数据库操作**:
- SELECT FROM sessions WHERE id = {session_id}
- SELECT FROM scenes WHERE session_id = {session_id}
- SELECT FROM players WHERE session_id = {session_id}
- SELECT FROM npcs WHERE session_id = {session_id}

---

## 场景管理接口

### POST /api/v1/session/{session_id}/scene

创建新场景

**请求体**:
```typescript
{
  location: string
  time: string
  weather: string
}
```

**数据库操作**:
- INSERT INTO scenes (session_id, location, time, weather, ...)

### PUT /api/v1/session/{session_id}/scene

更新当前场景

**请求体**: 同 POST

**数据库操作**:
- UPDATE scenes SET ... WHERE session_id = {session_id}

---

## 玩家管理接口

### POST /api/v1/session/{session_id}/players

添加玩家到会话

**请求体**:
```typescript
{
  name: string
  hp_max: number
  hp_current: number
  status_effects: string[]
  spell_slots?: SpellSlot[]
}
```

**数据库操作**:
- INSERT INTO players (session_id, name, ...)

### PUT /api/v1/session/{session_id}/players/{player_name}

更新玩家状态

**请求体**: Player 的部分字段

**数据库操作**:
- UPDATE players SET ... WHERE session_id = {session_id} AND name = {player_name}

### DELETE /api/v1/session/{session_id}/players/{player_name}

从会话中移除玩家

**数据库操作**:
- DELETE FROM players WHERE session_id = {session_id} AND name = {player_name}

---

## NPC 管理接口

### POST /api/v1/session/{session_id}/npcs

添加 NPC 到会话

**请求体**:
```typescript
{
  name: string
  description: string
  is_active: boolean
}
```

**数据库操作**:
- INSERT INTO npcs (session_id, name, description, is_active, ...)

### PUT /api/v1/session/{session_id}/npcs/{npc_name}

更新 NPC 信息

**请求体**: NPC 的部分字段

**数据库操作**:
- UPDATE npcs SET ... WHERE session_id = {session_id} AND name = {npc_name}

### DELETE /api/v1/session/{session_id}/npcs/{npc_name}

从会话中移除 NPC

**数据库操作**:
- DELETE FROM npcs WHERE session_id = {session_id} AND name = {npc_name}

---

## 图鉴管理接口

### GET /api/v1/gallery

获取当前用户的图鉴收藏列表

**查询参数**:
- `type`: 可选，按类型筛选（map/item/npc/location/other）
- `limit`: 可选，返回数量限制
- `offset`: 可选，分页偏移量

**响应**:
```typescript
{
  items: GalleryItem[]
  total: number
}
```

**数据库操作**:
- SELECT FROM gallery WHERE user_id = {current_user_id} ORDER BY pinned_at DESC

### POST /api/v1/gallery

添加图片到图鉴

**请求体**:
```typescript
{
  url: string              // 图片 URL 或 base64
  name: string            // 图片名称
  type: ImageType         // 类型：map/item/npc/location/other
  description?: string     // 可选描述
  source?: 'upload' | 'chat'  // 来源：用户上传 或 聊天中收藏
  message_id?: string     // 如果来自聊天，关联的消息 ID
}
```

**响应**:
```typescript
{
  id: string              // 图鉴项 ID
  url: string
  name: string
  type: ImageType
  pinned_at: string
}
```

**数据库操作**:
- INSERT INTO gallery (user_id, url, name, type, description, source, message_id, pinned_at, ...)

### PUT /api/v1/gallery/{item_id}

更新图鉴项信息

**请求体**:
```typescript
{
  name?: string
  type?: ImageType
  description?: string
}
```

**数据库操作**:
- UPDATE gallery SET ... WHERE id = {item_id}

### DELETE /api/v1/gallery/{item_id}

删除图鉴项

**数据库操作**:
- DELETE FROM gallery WHERE id = {item_id}

### POST /api/v1/gallery/upload

上传图片文件

**请求**:
- Content-Type: `multipart/form-data`
- `file`: 图片文件
- `type`: 图片类型（可选）

**响应**:
```typescript
{
  id: string
  url: string           // 上传后的图片 URL
  name: string
}
```

---

## 数据表结构

### sessions
| 字段 | 类型 | 说明 |
|------|------|------|
| id | string | 会话 ID |
| created_at | timestamp | 创建时间 |
| updated_at | timestamp | 更新时间 |

### scenes
| 字段 | 类型 | 说明 |
|------|------|------|
| session_id | string | 会话 ID (外键) |
| location | string | 地点 |
| time | string | 时间 |
| weather | string | 天气 |

### players
| 字段 | 类型 | 说明 |
|------|------|------|
| session_id | string | 会话 ID (外键) |
| name | string | 玩家名 |
| hp_max | int | 最大生命值 |
| hp_current | int | 当前生命值 |
| status_effects | json | 状态效果数组 |
| spell_slots | json | 法术位信息 |

### npcs
| 字段 | 类型 | 说明 |
|------|------|------|
| session_id | string | 会话 ID (外键) |
| name | string | NPC 名 |
| description | string | 描述 |
| is_active | boolean | 是否活跃 |

### dialogue_history
| 字段 | 类型 | 说明 |
|------|------|------|
| session_id | string | 会话 ID (外键) |
| role | string | 角色 (user/assistant) |
| content | text | 内容 |
| timestamp | timestamp | 时间戳 |

### rules
| 字段 | 类型 | 说明 |
|------|------|------|
| id | string | 规则 ID |
| category | string | 分类 |
| content | text | 内容 |

### episodes
| 字段 | 类型 | 说明 |
|------|------|------|
| id | string | 记录 ID |
| session_id | string | 会话 ID |
| content | text | 内容 |
| timestamp | timestamp | 时间戳 |

### gallery
| 字段 | 类型 | 说明 |
|------|------|------|
| id | string | 图鉴 ID |
| user_id | string | 用户 ID (外键) |
| url | string | 图片 URL 或 base64 |
| name | string | 图片名称 |
| type | string | 类型：map/item/npc/location/other |
| description | text | 可选描述 |
| source | string | 来源：upload/chat |
| message_id | string | 关联的消息 ID (可选) |
| pinned_at | timestamp | 收藏时间 |
| created_at | timestamp | 创建时间 |

---

## 类型定义

### ImageType
```typescript
type ImageType = 'map' | 'item' | 'npc' | 'location' | 'other'
```

### GalleryItem
```typescript
interface GalleryItem {
  id: string
  url: string
  name: string
  description?: string
  type: ImageType
  pinnedAt: string
  source?: 'chat' | 'upload'
  messageId?: string
}
```

### SpellSlot
```typescript
interface SpellSlot {
  level: number
  available: number
  max: number
}
```

### RetrievedDocument
```typescript
interface RetrievedDocument {
  id: string
  content: string
  score: number    // 相似度分数 0-1
}
```

---

## 战役管理接口

### GET /api/v1/campaigns

获取当前用户的战役列表

**查询参数**:
- `status`: 可选，按状态筛选（new/in_progress/completed/archived）
- `limit`: 可选，返回数量限制
- `offset`: 可选，分页偏移量

**响应**:
```typescript
{
  campaigns: Campaign[]
  total: number
}
```

**数据库操作**:
- SELECT FROM campaigns WHERE user_id = {current_user_id} ORDER BY updated_at DESC

### POST /api/v1/campaigns

创建新战役

**请求体**:
```typescript
{
  name: string
  description: string
  file_data?: string  // base64 encoded JSON (用于导入)
}
```

**响应**:
```typescript
{
  campaign: Campaign
  campaign_id: string
}
```

**数据库操作**:
- INSERT INTO campaigns (user_id, name, description, status, created_at, ...)
- 如果包含 file_data，解析 JSON 并导入相关数据

### POST /api/v1/campaigns/{campaign_id}/initialize

初始化战役会话（添加玩家并创建会话）

**请求体**:
```typescript
{
  players: AddPlayerRequest[]
}
```

**响应**:
```typescript
{
  session_id: string
  campaign: Campaign
  players: Player[]
}
```

**数据库操作**:
- INSERT INTO sessions (campaign_id, user_id, created_at, ...)
- INSERT INTO players (session_id, ...) for each player
- UPDATE campaigns SET status = 'in_progress'

### GET /api/v1/campaigns/{campaign_id}

获取战役详情

**响应**:
```typescript
{
  campaign: Campaign
  players?: Player[]
  scenes?: Scene[]
  dialogue_count?: number
}
```

**数据库操作**:
- SELECT FROM campaigns WHERE id = {campaign_id}
- SELECT FROM sessions WHERE campaign_id = {campaign_id}

---

## 数据表结构

### campaigns
| 字段 | 类型 | 说明 |
|------|------|------|
| id | string | 战役 ID |
| user_id | string | 用户 ID (外键) |
| name | string | 战役名称 |
| description | text | 战役描述 |
| status | string | 状态：new/in_progress/completed/archived |
| created_at | timestamp | 创建时间 |
| updated_at | timestamp | 更新时间 |
| last_played_at | timestamp | 最后游戏时间 |

---

## 后端 API 需求总结

### 优先级 P0 - 必须实现

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | /api/v1/campaigns | 获取战役列表 |
| POST | /api/v1/campaigns | 创建战役（支持 file_data 导入） |
| POST | /api/v1/campaigns/{id}/initialize | 初始化战役会话 |
| GET | /api/v1/session/{id}/state | 获取游戏状态 |
| POST | /api/v1/agent | 对话接口（已有） |

### 优先级 P1 - 建议实现

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | /api/v1/campaigns/{id} | 获取战役详情 |
| PUT | /api/v1/session/{id}/scene | 更新场景 |
| POST | /api/v1/session/{id}/players | 添加玩家 |
| PUT | /api/v1/session/{id}/players/{name} | 更新玩家状态 |
| DELETE | /api/v1/session/{id}/players/{name} | 移除玩家 |
| POST | /api/v1/session/{id}/npcs | 添加 NPC |
| PUT | /api/v1/session/{id}/npcs/{name} | 更新 NPC |
| DELETE | /api/v1/session/{id}/npcs/{name} | 移除 NPC |
