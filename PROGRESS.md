# 开发进度

## 2026-03-03

### 变更

- 迁移 Pydantic v2 配置写法：用 `ConfigDict` 替代 `class Config`，消除 `PydanticDeprecatedSince20` 警告（`populate_by_name` 保持不变）
- 规范 Pydantic 模型可变默认值：把 `=[]` 改为 `Field(default_factory=list)`，避免潜在共享状态问题

## 2026-03-02

### 变更

- 改进 `scripts/enhance_metadata.py` 处理非 mm 数据：
  - 新增 `process_non_mm_file()` 函数：按 chapter 聚合文本 → 分块 → 路径注入
  - 支持自定义 `chunk_size` (默认 500) 和 `chunk_overlap` (默认 50)
  - 优先按段落 (\n\n) 切分，其次按句子，避免打断有序列表
  - 输出格式：`id`, `text` (含路径前缀), `metadata.hierarchy_path`, `metadata.chunk_index`
  - 命令行支持 `--source phb|dmg|mm|all` 和 `--chunk-size`, `--chunk-overlap` 参数

## 2026-02-27

### 变更

- 清理索引脚本的无用/低价值参数：
   - `scripts/process_images.py`：移除 `--dry-run`、`--batch-size` 及相关分支逻辑（旧 `scripts/index_images.py` 作为兼容 shim）
   - `scripts/index_knowledge.py`：移除未实际生效的 `--batch-size` 参数

- 重构图片处理与索引流程：
   - `scripts/process_images.py`：改为纯“图片清洗/描述生成”，不再负责 Pinecone upsert
   - `scripts/index_knowledge.py`：索引前自动确保 `knowledge/use/rag_{source}.jsonl` 存在（缺失则先运行图片清洗；兼容旧的 `*_described.jsonl`）

## 2026-02-28

### 变更

- `scripts/process_images.py` 通用化：不再依赖固定 `--source` 选项列表；默认自动扫描 `knowledge/raw/rag_*.jsonl` 并输出到 `knowledge/use/` 同名文件

## 2024-02-26

### 完成功能

1. **知识库索引系统**
   - 设计 namespace 方案：按数据类型划分 (rules, monsters, spells, episodes)
   - 设计 metadata 结构：category, source_book, chapter, monster_type 等字段
   - 实现 `KnowledgeMetadata` 模型

2. **索引脚本**
   - 创建 `scripts/index_knowledge.py`
   - 支持单独索引 mm/phb/dmg 或全部索引
   - 支持 `--limit` 参数限制数量

3. **Bug 修复**
   - 修复 Pinecone Pydantic 对象序列化问题
   - 修复 mock retrieval 向量提取问题

4. **图片描述 JSONL 处理优化**
   - 更新 `scripts/index_images.py`：输出 JSONL 保留原始字段（如 chapter），仅替换/更新 text 中的图片描述内容

### 测试验证

- Monsters namespace 检索正常 ✓
- Rules namespace 检索正常 ✓
- Metadata 正确包含 category, source_book 等字段 ✓

---

## 历史进度

### 2024-02-xx (前期)

- FastAPI 应用框架搭建
- Pinecone 服务集成
- 知识检索模块 (Hybrid Search)
- Agent 模块 (LangSmith)
