# simpleRAG

一个面向通用知识与记忆检索的轻量 RAG 项目。

- 数据准备：支持普通 Markdown 结构分块 + `HISTORY.md` 事件分块
- 检索：向量检索（FAISS）+ BM25 + RRF 融合重排
- 生成：查询重写、查询路由、回答生成（支持流式接口）
- 启动方式：`simple-rag` CLI（`ingest / ask / chat / mcp-server`）

## Architecture

```text
DataModule
  -> load_documents()
  -> chunk_documents()
      - HISTORY.md: event chunking
      - other .md: markdown header chunking
  -> parent-child mapping

IndexModule
  -> HuggingFaceEmbeddings
  -> FAISS build/load/save

RetrieveModule
  -> vector retriever (similarity)
  -> BM25 retriever
  -> RRF rerank

GenerationIntegrationModule
  -> query_rewrite()
  -> query_router()
  -> generate_xxx_answer()

SimpleRAG
  -> ingest()
  -> load_index()
  -> retrieve()
  -> answer()/answer_stream()

app.py
  -> legacy CLI entry

simple_rag/cli.py
  -> CLI: ingest / ask / chat / mcp-server

simple_rag/mcp_server.py
  -> MCP tools: rag_ingest / rag_retrieve / rag_answer
```

## Project Structure

```text
simpleRAG/                     
  app.py             
  .env
  data/                       # Workspace
    history_sample.md
    sample_docs.txt
  index_store/                # 存储vector索引
  src/
    simple_rag/               # core
      data_module.py
      index_module.py
      retrieve_module.py
      rag.py
```

## Setup

```bash
uv sync
```

在 `.env` 中至少配置：

```dotenv
DEEPSEEK_API_KEY=your_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B
RAG_DATA_PATH=./data
RAG_INDEX_PATH=./index_store
LOG_LEVEL=INFO
```

## Data Preparation

`DataModule` 规则：

1. 读取 `--data-path` 指定的文件或目录（目录会递归读取 `*.md`）
2. 若文件名是 `HISTORY.md`，按顶层时间戳事件切分：
   - 匹配格式：`[YYYY-MM-DD HH:MM] ...`
   - 生成事件级 chunk，并写入时间范围等 metadata
3. 其他 Markdown 使用标题分块（`# / ## / ###`）
4. 维护父子映射：
   - 子块用于检索
   - 父文档用于生成阶段补全上下文

## Retrieval

`RetrieveModule` 检索流程：

1. 向量召回（FAISS）
2. BM25 召回
3. 使用 RRF（Reciprocal Rank Fusion）融合重排
4. 返回 top-k 文档

## Generation

`GenerationIntegrationModule` 支持：

- `query_rewrite`：将模糊问题改写成更利于检索的问题
- `query_router`：路由到 `list/detail/general`
- `generate_basic_answer`：普通回答
- `generate_step_by_step_answer`：分步骤分析
- `generate_list_answer`：列表类回答

## CLI

### 1) 建索引

```bash
uv run simple-rag ingest
```

可选参数：

- `--data-path`
- `--index-path`
- `--embedding-model`
- `--llm-model`

### 2) 单次问答

```bash
uv run simple-rag ask "用户问题"
```

常用参数：

- `--top-k 5`
- `--reindex`（答前强制重建索引）
- `--step`（强制分步骤回答）

### 3) 交互式多轮

```bash
uv run simple-rag chat
```

- 退出词默认：`exit`, `quit`, `q`
- 可通过 `--exit-words` 自定义

## Quick Example

```bash
# 1) 用 sample 数据建索引
uv run simple-rag ingest --data-path ./data

# 2) 提问
uv run simple-rag ask "昨天我和助手聊过什么？" --top-k 5

# 3) 启动 MCP Server
uv run simple-rag --data-path ./data --index-path ./index_store mcp-server --transport stdio --name simple-rag
```

## Notes

- `index_store/` 存储向量索引，不是原始文档目录。
- 原始文档建议放在 `data/`（或通过 `.env` 设置 `RAG_DATA_PATH`）。
- 如果你用 `seju-lite/workspace/memory/HISTORY.md` 测试，建议先复制截取样本到 `data/`，便于快速迭代。

> Next: 0.2.0 基于Unstructured的文档解析
