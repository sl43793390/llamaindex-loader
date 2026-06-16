# llamaindex-loader

> A complete **LlamaIndex** pipeline: multi-format document parsing → splitting → vectorization (Milvus) → RAG chat.

[中文文档](./README.md)

---

## 1. Overview

This project demonstrates how to use **LlamaIndex** to parse documents of various formats
(**Word / Excel / PDF / Txt / Markdown / HTML / JSON / Web**), split them, store the chunks
into **Milvus**, and run memory-enabled **Retrieval-Augmented Generation (RAG)** chats against
any **OpenAI-compatible LLM**.

It works as a scaffold for:
- Enterprise knowledge bases
- Document Q&A bots
- Customer-support assistants

Highlights:
- **Multi-format parsing** — Word / Excel / PDF / MD / HTML / JSON / Web in one line
- **Rich splitters** — sentence, token, Markdown, HTML, JSON, code, semantic
- **Milvus persistence** — both embedded (zero-deps) and server modes
- **OpenAI-compatible LLM** — DeepSeek / Qwen / Ollama / vLLM / dmxapi all work
- **Two chat modes** — multi-turn rewriting (Condense) or raw-query (custom engine)
- **Full-pipeline debug log** — system prompt / retrieved nodes / LLM prompt / response
- **Milvus inspector** — `inspect_milvus()` to confirm the `text` field is actually written
- **3-tier config priority** — call-site args > `config.py` > env vars, change anywhere

---

## 2. Architecture

```
┌────────────────────────┐
│  dataLoader (Parsing)  │  Word / Excel / PDF / Txt / MD / HTML / JSON / Web
└──────────┬─────────────┘
           │  List[Document]
           ▼
┌────────────────────────┐
│  spliter  (Chunking)   │  Sentence / Token / Markdown / HTML / JSON / Code / Semantic
└──────────┬─────────────┘
           │  List[BaseNode]
           ▼
┌────────────────────────────────────────────────┐
│  vectorStore (Milvus + Chat)                   │
│   ├── milvus_store.py  : write / load / inspect│
│   ├── callbacks.py     : RAGDebugHandler       │
│   ├── custom_engine.py : CustomRAGChatEngine   │
│   └── chat.py          : build_chat_engine()   │
└──────────┬─────────────────────────────────────┘
           │  VectorStoreIndex
           ▼
┌────────────────────────┐
│  chat  (RAG)           │  Retriever → Context → LLM → Answer
└────────────────────────┘
```

End-to-end pipeline:
```
Parse (dataLoader) -> Split (spliter) -> Write to Milvus (vectorStore) -> RAG chat
```

---

## 3. Project Layout

```
llamaindex-loader/
├── config.py                 # Unified config (Embedding / LLM / Milvus / Chat)
├── main.py                   # Entry point that wires the whole pipeline
├── requirements.txt          # Dependencies
├── README.md                 # Chinese documentation
├── README_EN.md              # English documentation (this file)
│
├── dataLoader/               # 1. Document parsing
│   ├── __init__.py           #   Public API re-exports
│   └── loaders.py            #   Reader wrappers + auto_load / load_directory
│
├── spliter/                  # 2. Document splitting
│   ├── __init__.py           #   Public API re-exports
│   └── splitters.py          #   9 splitters + auto_split
│
└── vectorStore/              # 3. Vector store + chat
    ├── __init__.py           #   Public API re-exports
    ├── milvus_store.py       #   Milvus client, index, inspect_milvus
    ├── callbacks.py          #   RAGDebugHandler (full-pipeline log)
    ├── custom_engine.py      #   CustomRAGChatEngine (self-implemented)
    └── chat.py               #   build_chat_engine / build_simple_query_engine / chat_loop
```

---

## 4. Core Modules

### 4.1 `dataLoader` — Document Parsing
| Function | Backend / Use case |
|---|---|
| `load_word(path)` | `.docx` (DocxReader) |
| `load_excel(path, sheet_name=None, pandas_config=None)` | `.xlsx` / `.xls` (PandasExcelReader) |
| `load_pdf(path, return_full_document=False)` | `.pdf` (pypdf, lightweight) |
| `load_pdf_pymupdf(path)` | `.pdf` (PyMuPDF, **recommended**) |
| `load_pdf_plumber(path)` | `.pdf` (pdfplumber, great for tables) |
| `load_pdf_auto(path, backend="pymupdf")` | Unified PDF entry, swappable backend (`pymupdf` / `pdfplumber` / `pypdf`) |
| `load_txt(path, encoding="utf-8")` | `.txt` |
| `load_markdown(path)` | `.md` / `.markdown` |
| `load_html(path, tag="body", ignore_no_id=True)` | `.html` / `.htm` |
| `load_json(path, levels_back=None, collapse_length=None, is_jsonl=False)` | `.json` / `.jsonl` |
| `load_web(urls, use_bs4=True)` | Web pages (BeautifulSoup / SimpleWebPage) |
| `auto_load(path)` | Pick a loader by extension; falls back to `SimpleDirectoryReader` |
| `load_directory(dir, recursive=True, required_exts=None)` | Bulk-load a directory (recursive; avoid for huge directories) |
| `list_supported_files(dir, recursive=True, required_exts=None)` | **List files only, do not read**, pair with `index.insert_nodes` for streaming ingest |

### 4.2 `spliter` — Document Splitting
| Function | Description |
|---|---|
| `split_by_sentence(docs, chunk_size=1024, chunk_overlap=200)` | Sentence-based splitting (**default**, recommended) |
| `split_by_token(docs, chunk_size=512, chunk_overlap=50, separator=" ")` | Token-based splitting (fits LLM token limits) |
| `split_simple(docs, chunk_size=1024, chunk_overlap=200, separator=" ")` | Simple fixed-size splitting |
| `split_sentence_window(docs, window_size=3, ...)` | Sentence window (good for context-augmented QA) |
| `split_markdown(docs)` | Markdown heading hierarchy |
| `split_html(docs)` | HTML tag structure |
| `split_json(docs)` | JSON nested structure |
| `split_code(docs, language="python", ...)` | Code-aware splitting (class/function, tree-sitter) |
| `split_semantic(docs, embed_model, ...)` | Semantic similarity (requires an embedding model) |
| `auto_split(docs, doc_type="text")` | Pick a splitter by `doc_type` |

### 4.3 `vectorStore` — Milvus + Chat
| Function / Class | Description |
|---|---|
| `get_embed_model()` | Build an OpenAI-compatible Embedding instance |
| `get_llm()` | Build an OpenAI-compatible LLM instance |
| `configure_settings(embed_model=None, llm=None)` | Register global Embedding / LLM / chunk_size / chunk_overlap |
| `build_milvus_store(uri=None, ..., overwrite=None)` | Create a Milvus client (always passes `output_fields=["text"]`) |
| `create_index(nodes, milvus_store=None)` | Persist nodes to Milvus and build an index |
| `load_existing_index(milvus_store=None)` | Load an existing collection (`overwrite=False`) |
| `inspect_milvus(collection_name=None, uri=None, limit=3)` | **Inspector** — print schema, row count, sample rows |
| `build_chat_engine(index, ..., enable_question_rewriting=None, debug=None)` | Build a memory-enabled chat engine |
| `build_simple_query_engine(index, ...)` | One-shot query engine (no memory) |
| `chat_loop(engine)` | CLI chat loop |
| `CustomRAGChatEngine` | **Self-implemented chat engine** with full control over message assembly |
| `RAGDebugHandler` | **Callback handler** that prints the full RAG pipeline |

---

## 5. Quick Start

### 5.1 Install
```bash
python -m venv .venv
source .venv/bin/activate              # macOS / Linux
# .\.venv\Scripts\activate            # Windows
pip install -r requirements.txt

# Local embedded Milvus needs the milvus-lite extra
pip install pymilvus[milvus_lite]
```

### 5.2 Configure
Set environment variables (or edit [config.py](./config.py) directly):

```bash
# OpenAI-compatible LLM / Embedding (DeepSeek / Qwen / Ollama / vLLM / dmxapi are all OK)
export OPENAI_API_KEY=sk-xxxxxx
export OPENAI_BASE_URL=https://api.openai.com/v1
export LLM_MODEL=gpt-4o-mini
export EMBED_MODEL=text-embedding-3-small
export EMBED_DIM=1536

# Milvus (local embedded mode)
export MILVUS_URI=./milvus_llamaindex.db
export MILVUS_COLLECTION=llamaindex_rag
export MILVUS_OVERWRITE=true

# Chat engine
export ENABLE_QUESTION_REWRITING=false
export SIMILARITY_TOP_K=5
export MEMORY_TOKEN_LIMIT=3000
export RAG_DEBUG=false
export RAG_DEBUG_TEXT_LIMIT=800
```

> See [config.py](./config.py) for the full list of fields. **Every config supports the 3-tier priority "call-site > `config.py` > env vars"** (see §6.3).

### 5.3 Run
Open [main.py](./main.py) and uncomment the source you want:

```python
from main import ingest_file, ingest_directory, ingest_web, start_chat
from vectorStore import load_existing_index

# Option A: parse a single file -> split -> ingest
index = ingest_file("Oracle-19c-安装及常用命令.md")

# Option B: parse every supported file in a directory -> split -> ingest
index = ingest_directory("./data")  # streaming, OK for 10GB+ directories

# Option C: scrape web pages -> split -> ingest
index = ingest_web(["https://docs.llamaindex.ai/en/stable/"])  # streaming, per-URL

# Option D: load an existing Milvus collection (the default — fastest startup)
index = load_existing_index()

# Start the chat
# enable_question_rewriting=True  -> rewrite multi-turn question (chatty use)
# enable_question_rewriting=False -> keep the original question (default)
# debug=True  -> print system prompt / retrieved nodes / LLM prompt / response
start_chat(index, enable_question_rewriting=False, debug=True)
```

```bash
python main.py
```

The CLI chat will look like:
```
=== RAG 对话已就绪,输入 'exit' / 'quit' 退出 ===

请输入: What is the main conclusion of this PDF?
【用户输入】
What is the main conclusion of this PDF?
...
助手: ...
```

---

## 6. Key Switches

### 6.1 `enable_question_rewriting` — Chat Mode
| Value | Behavior | Engine |
|---|---|---|
| `True` | Rewrite multi-turn question into a standalone query, then retrieve | LlamaIndex built-in `CondenseQuestionChatEngine` |
| `False` (default) | Keep the original question; chat history + retrieved context go straight to the LLM | **Self-implemented** `CustomRAGChatEngine` (full control over message assembly) |

**When to use which?**
- Conversational / context-dependent questions ("does it support Chinese?", "tell me more") → `True`
- Single-shot, keyword-sensitive lookups → `False`

**Why a self-implemented engine?**
LlamaIndex's built-in `ContextChatEngine` behaves inconsistently across versions — the
`{context_str}` placeholder in `context_template` is sometimes left un-replaced, so the LLM
receives a blank context. `CustomRAGChatEngine` uses the simplest possible `str.format()`
to assemble messages, eliminating that risk.

### 6.2 `debug` — Full-Pipeline Log
When enabled, the following are printed (in order):

1. The current **system prompt**
2. **QA User template / Context User template** (to confirm which template is actually used)
3. Every **node retrieved from Milvus** (text + similarity score + source)
4. The **messages** that get sent to the LLM
5. The **LLM's response**
6. Token / call parameters

Example output (excerpt):
```
======================================================================
【系统提示词(System Prompt)】
======================================================================
你是一个专业的知识库助手,请根据下方提供的参考资料用中文回答用户问题......

======================================================================
【检索 #1 - 查询: 这份文档的核心结论】
======================================================================
命中节点数: 5

--- Node 1  score=0.9123  source=Oracle-19c-安装及常用命令.md ---
...node content...

======================================================================
【LLM 调用 #1 - 发送的 messages】
======================================================================
[Message 0] role=system
你是一个专业的知识库助手...

[Message 1] role=user
参考资料:
...node content...
问题: 这份文档的核心结论
请回答:

======================================================================
【LLM 调用 #1 - 模型返回】
======================================================================
本文档的核心结论是......
```

### 6.3 Config Priority (call-site > `config.py` > env vars)

Every switch supports three ways to set it. **Higher = higher priority**:

```python
# 1. Call-site argument (highest — good for one-off overrides)
start_chat(index, enable_question_rewriting=False, debug=True)

# 2. config.py field (project-level default)
#    In config.py:
#    CHAT.enable_question_rewriting = False
#    CHAT.debug = True

# 3. Environment variable (deployment / CI)
#    export ENABLE_QUESTION_REWRITING=false
#    export RAG_DEBUG=true
```

When `start_chat(index)` is called with `enable_question_rewriting=None` / `debug=None`, the
runtime falls back through 2 → 3.

### 6.4 Streaming Ingest (No OOM on Huge Directories / Many URLs)

`main.ingest_directory` / `main.ingest_web` use a **streaming** path by default, so a 10GB+
corpus or a long URL list will not exhaust memory:

| Step | Action | Resident memory |
|---|---|---|
| 1. List files | `list_supported_files` only walks the directory | ≈ 0 (only file metadata) |
| 2. Empty index | `create_index([], milvus_store=store)` creates the Milvus collection | ≈ 0 |
| 3. for loop (× N) | `auto_load` → `auto_split` → `index.insert_nodes(nodes)` | ≈ **one file's text + its nodes** |
| 4. Every 10 files | `gc.collect()` | — |

The legacy path (`load_directory` + one-shot `create_index`) keeps the **whole corpus** in
memory and OOMs on multi-GB directories. The streaming path keeps only the per-file peak.

> See [main.py](./main.py) for the implementation. For small directories (< 100MB) the
> legacy one-shot path is shorter; use whichever you prefer.

---

## 7. Maintenance & Extension

### 7.1 Add a new document type
1. Implement a `load_xxx` function in [dataLoader/loaders.py](./dataLoader/loaders.py)
2. Re-export it in [dataLoader/__init__.py](./dataLoader/__init__.py)
3. Add the extension to the `mapping` dict inside `auto_load`

### 7.2 Switch Embedding / LLM
Just change environment variables:
- **DeepSeek** — `OPENAI_BASE_URL=https://api.deepseek.com/v1`, `LLM_MODEL=deepseek-chat`
- **Qwen / DashScope (OpenAI-compatible)** — set `OPENAI_BASE_URL` to the OpenAI-compatible endpoint
- **Ollama** — `OPENAI_BASE_URL=http://localhost:11434/v1`, `LLM_MODEL=qwen2.5`
- **Zhipu / vLLM / dmxapi** — set the corresponding `OPENAI_BASE_URL` and `LLM_MODEL`

> `EMBED_DIM` must match the chosen embedding model, otherwise Milvus schema creation will fail.

### 7.3 Switch Milvus mode
- **Embedded** (default): `MILVUS_URI=./milvus_llamaindex.db` + `pip install pymilvus[milvus_lite]`
- **Standalone / cluster**: `MILVUS_URI=http://localhost:19530` (start a Milvus server first; optionally set `MILVUS_TOKEN`)

### 7.4 Tune chunking
Adjust `Settings.chunk_size` / `Settings.chunk_overlap` in [config.py](./config.py), or pass
arguments to `split_by_sentence` / `split_by_token` directly.

### 7.5 Rebuild the index
- Embedded mode: delete the `milvus_llamaindex.db/` directory and re-run
- Server mode: set `MILVUS_OVERWRITE=true`, or call `milvus_store.drop_collection()`

### 7.6 Verify Milvus was actually populated
```bash
python -c "from vectorStore import inspect_milvus; inspect_milvus()"
```
This prints the schema, row count, and a few sample rows so you can confirm the `text` field
contains real content.

### 7.7 Calling the chat engine directly (SDK style)
```python
from vectorStore import (
    configure_settings, build_milvus_store, create_index,
    build_chat_engine, CustomRAGChatEngine, RAGDebugHandler,
)
from llama_index.core import Settings
from llama_index.core.callbacks.base import CallbackManager

configure_settings()
store = build_milvus_store()
index = create_index([], milvus_store=store)   # existing collection, no new nodes
engine: CustomRAGChatEngine = build_chat_engine(index, debug=True)

resp = engine.chat("Your question")
print(resp.response)
print("source nodes:", len(resp.source_nodes))
```

---

## 8. FAQ

**Q1. `ModuleNotFoundError: No module named 'milvus_lite'`?**
Local embedded Milvus needs it:
```bash
pip install pymilvus[milvus_lite]
```
Or use a Milvus server URI instead.

**Q2. `ModuleNotFoundError: No module named 'llama_index.readers.file'`?**
```bash
pip install llama-index-readers-file
```
Or reinstall everything: `pip install -r requirements.txt`.

**Q3. Embedding dimension mismatch?**
`EMBED_DIM` must equal the dimension of the chosen embedding model
(OpenAI `text-embedding-3-small` = 1536, `bge-large-zh` = 1024, etc.).

**Q4. Chinese PDF returns garbage / empty?**
Try a different backend:
```python
from dataLoader import load_pdf_auto
load_pdf_auto("x.pdf", backend="pdfplumber")
# or
load_pdf_auto("x.pdf", backend="pymupdf")
```
Scanned PDFs need OCR first.

**Q5. Use a local embedding (BGE / M3E)?**
```bash
pip install sentence-transformers
```
Then replace `get_embed_model` in [vectorStore/milvus_store.py](./vectorStore/milvus_store.py) with
`HuggingFaceEmbedding`.

**Q6. Milvus returns N nodes but their content is empty?**
Make sure `MilvusVectorStore` is constructed with `output_fields=["text"]` — Milvus by default
returns only `id` + `score`. `build_milvus_store()` already passes this; if you construct the
client yourself, remember to add it.

**Q7. `ImportError: cannot import name 'BaseCallbackHandler' from 'llama_index.core.callbacks'`?**
The import path moved in newer LlamaIndex. Use:
```python
from llama_index.core.callbacks.base import BaseCallbackHandler
```
Already fixed in this project — see [vectorStore/callbacks.py](./vectorStore/callbacks.py).

**Q8. `TypeError: RAGDebugHandler.end_trace() got an unexpected keyword argument 'trace_map'`?**
LlamaIndex passes `trace_map` to `end_trace`. The signature must be:
```python
def end_trace(self, trace_id=None, trace_map=None):
    ...
```
Already fixed in this project.

**Q9. I don't see the debug log?**
Make sure debug is enabled in at least one place: ① `start_chat(index, debug=True)` or
② `CHAT.debug = True` or ③ `RAG_DEBUG=true`. Also, `RAGDebugHandler` is registered on
`Settings.callback_manager` — don't replace `Settings.callback_manager` *after* calling
`start_chat`.

**Q10. The chat-mode switch has no effect?**
Make sure the call-site argument matches what you expect. With debug enabled, both the
`CustomRAGChatEngine` (rewriting-off) and the `CondenseQuestionChatEngine` (rewriting-on)
system templates are printed, so you can verify which one is actually in use.

**Q11. Parsing a huge directory OOMs?**
Use `main.ingest_directory` / `main.ingest_web`. They use a streaming path: `list_supported_files` first (no content read), then `index.insert_nodes` per file. Do not call `dataLoader.load_directory` for huge corpora — that reads everything into RAM at once.

**Q12. `HTMLTagReader` returns an empty list?**
By default `tag="section"` and `ignore_no_id=False` will drop nodes without an id. `load_html` already defaults to `tag="body"` + `ignore_no_id=True`, so just use the wrapper.

**Q13. `JSONReader.__init__() got an unexpected keyword argument 'levels'`?**
The new `llama-index-readers-json` renamed the parameter to `levels_back`. `load_json` is aligned; do not pass the old name manually.

---

## 9. License

For learning and internal use only.
