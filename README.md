# llamaindex-loader

> 基于 **LlamaIndex** 的多格式文档 → 切分 → 向量化(Milvus)→ RAG 对话 完整流水线

---

## 1. 项目简介

本项目演示了如何用 LlamaIndex 解析 **Word / Excel / PDF / Txt / Markdown / HTML / JSON / Web** 等多种数据源,
经过切分后写入 **Milvus** 向量数据库,并通过 **OpenAI 兼容的 LLM** 实现带记忆的检索增强对话(RAG Chat)。

适合作为企业内部知识库、文档问答机器人、客服助手的脚手架。

核心特性:
- **多格式解析** —— 一行代码切换 Word/Excel/PDF/MD/HTML/JSON/Web
- **多种切分器** —— 句子、Token、Markdown、HTML、JSON、代码、语义切分
- **Milvus 持久化** —— 嵌入式(零依赖)与集群模式皆可
- **OpenAI 兼容 LLM** —— 支持 DeepSeek / 通义千问 / Ollama / vLLM / dmxapi 等
- **两种对话模式** —— 多轮问题改写(Condense)与原文直接检索(自实现引擎)
- **全链路 Debug 日志** —— 系统提示词 / 检索节点 / LLM prompt / 模型返回 一键打印
- **Milvus 自检** —— `inspect_milvus()` 验证 text 字段是否真的写进去了
- **三档配置优先级** —— 调用参数 > `config.py` > 环境变量,改一处即可

---

## 2. 架构

```
┌────────────────────────┐
│  dataLoader (解析)     │  Word / Excel / PDF / Txt / MD / HTML / JSON / Web
└──────────┬─────────────┘
           │  List[Document]
           ▼
┌────────────────────────┐
│  spliter   (切分)      │  Sentence / Token / Markdown / HTML / JSON / Code / Semantic
└──────────┬─────────────┘
           │  List[BaseNode]
           ▼
┌────────────────────────────────────────────────┐
│  vectorStore (Milvus + 对话)                    │
│   ├── milvus_store.py  : 写入/加载/自检        │
│   ├── callbacks.py     : RAGDebugHandler       │
│   ├── custom_engine.py : CustomRAGChatEngine   │
│   └── chat.py          : build_chat_engine()   │
└──────────┬─────────────────────────────────────┘
           │  VectorStoreIndex
           ▼
┌────────────────────────┐
│  chat(检索对话)        │  Retriever → Context → LLM → 回答
└────────────────────────┘
```

完整流水线:
```
解析 (dataLoader) -> 切分 (spliter) -> 写入 Milvus (vectorStore) -> 检索对话
```

---

## 3. 目录结构

```
llamaindex-loader/
├── config.py                 # 统一配置(Embedding / LLM / Milvus / Chat)
├── main.py                   # 入口,串联整条流水线
├── requirements.txt          # 依赖
├── README.md                 # 中文文档(本文件)
├── README_EN.md              # 英文文档
│
├── dataLoader/               # 1. 文档解析
│   ├── __init__.py           #   公共 API 导出
│   └── loaders.py            #   各类 Reader 封装 + auto_load / load_directory
│
├── spliter/                  # 2. 文档切分
│   ├── __init__.py           #   公共 API 导出
│   └── splitters.py          #   9 种切分器 + auto_split
│
└── vectorStore/              # 3. 向量存储 + 对话
    ├── __init__.py           #   公共 API 导出
    ├── milvus_store.py       #   Milvus 客户端、索引、inspect_milvus 自检
    ├── callbacks.py          #   RAGDebugHandler(全链路日志)
    ├── custom_engine.py      #   CustomRAGChatEngine(自实现,完全控制消息拼装)
    └── chat.py               #   build_chat_engine / build_simple_query_engine / chat_loop
```

---

## 4. 核心模块

### 4.1 `dataLoader` — 文档解析
| 函数 | 后端 / 用途 |
|---|---|
| `load_word(path)` | `.docx`(DocxReader) |
| `load_excel(path, sheet_name=None, pandas_config=None)` | `.xlsx` / `.xls`(PandasExcelReader) |
| `load_pdf(path, return_full_document=False)` | `.pdf`(pypdf,依赖最轻) |
| `load_pdf_pymupdf(path)` | `.pdf`(PyMuPDF,综合最优,**推荐**) |
| `load_pdf_plumber(path)` | `.pdf`(pdfplumber,擅长表格) |
| `load_pdf_auto(path, backend="pymupdf")` | 统一 PDF 入口,可切后端(`pymupdf` / `pdfplumber` / `pypdf`) |
| `load_txt(path, encoding="utf-8")` | `.txt` |
| `load_markdown(path)` | `.md` / `.markdown` |
| `load_html(path, tag="body", ignore_no_id=True)` | `.html` / `.htm` |
| `load_json(path, levels_back=None, collapse_length=None, is_jsonl=False)` | `.json` / `.jsonl` |
| `load_web(urls, use_bs4=True)` | 网页抓取(BeautifulSoup / SimpleWebPage) |
| `auto_load(path)` | 按后缀自动选 loader,未识别后缀回退 `SimpleDirectoryReader` |
| `load_directory(dir, recursive=True, required_exts=None)` | 批量加载目录(默认递归,大目录慎用) |
| `list_supported_files(dir, recursive=True, required_exts=None)` | **只列文件不读内容**,配合 `index.insert_nodes` 做流式入库 |

### 4.2 `spliter` — 文档切分
| 函数 | 说明 |
|---|---|
| `split_by_sentence(docs, chunk_size=1024, chunk_overlap=200)` | 句子切分(**默认**,推荐) |
| `split_by_token(docs, chunk_size=512, chunk_overlap=50, separator=" ")` | 按 token 切分(贴合 LLM 限制) |
| `split_simple(docs, chunk_size=1024, chunk_overlap=200, separator=" ")` | 简单定长切分 |
| `split_sentence_window(docs, window_size=3, ...)` | 句子窗口切分(检索时还原上下文) |
| `split_markdown(docs)` | 按 Markdown 标题层级 |
| `split_html(docs)` | 按 HTML 标签 |
| `split_json(docs)` | 按 JSON 嵌套结构 |
| `split_code(docs, language="python", ...)` | 按代码语义(类/函数,基于 tree-sitter) |
| `split_semantic(docs, embed_model, ...)` | 按语义相似度(需 Embedding) |
| `auto_split(docs, doc_type="text")` | 按 `doc_type` 自动选 splitter |

### 4.3 `vectorStore` — Milvus + 对话
| 函数 / 类 | 说明 |
|---|---|
| `get_embed_model()` | 获取 OpenAI 兼容的 Embedding 实例 |
| `get_llm()` | 获取 OpenAI 兼容的 LLM 实例 |
| `configure_settings(embed_model=None, llm=None)` | 全局注册 Embedding / LLM / chunk_size / chunk_overlap |
| `build_milvus_store(uri=None, ..., overwrite=None)` | 构造 Milvus 客户端(**强制带** `output_fields=["text"]`) |
| `create_index(nodes, milvus_store=None)` | 写入 Milvus 并构建索引 |
| `load_existing_index(milvus_store=None)` | 加载已存在的 collection(`overwrite=False`) |
| `inspect_milvus(collection_name=None, uri=None, limit=3)` | **自检** —— 打印 schema、行数、样本数据 |
| `build_chat_engine(index, ..., enable_question_rewriting=None, debug=None)` | 构造带记忆的对话引擎 |
| `build_simple_query_engine(index, ...)` | 一次性查询引擎(无记忆) |
| `chat_loop(engine)` | 命令行交互循环 |
| `CustomRAGChatEngine` | **自实现的 Chat Engine**,完全控制消息拼装 |
| `RAGDebugHandler` | **回调处理器**,打印 RAG 全链路日志 |

---

## 5. 快速开始

### 5.1 安装依赖
```bash
python -m venv .venv
.\.venv\Scripts\activate           # Windows
pip install -r requirements.txt

# 本地嵌入式 Milvus 需额外安装 milvus-lite
pip install pymilvus[milvus_lite]
```

### 5.2 配置环境变量(或直接改 [config.py](./config.py))
```bash
# OpenAI 兼容的 LLM / Embedding(可指向 DeepSeek、通义千问、Ollama、vLLM、dmxapi 等)
set OPENAI_API_KEY=sk-xxxxxx
set OPENAI_BASE_URL=https://api.openai.com/v1
set LLM_MODEL=gpt-4o-mini
set EMBED_MODEL=text-embedding-3-small
set EMBED_DIM=1536

# Milvus(嵌入式本地模式)
set MILVUS_URI=./milvus_llamaindex.db
set MILVUS_COLLECTION=llamaindex_rag
set MILVUS_OVERWRITE=true

# 对话引擎
set ENABLE_QUESTION_REWRITING=false
set SIMILARITY_TOP_K=5
set MEMORY_TOKEN_LIMIT=3000
set RAG_DEBUG=false
set RAG_DEBUG_TEXT_LIMIT=800
```

> 完整配置字段见 [config.py](./config.py)。**所有配置项都支持"调用参数 > `config.py` > 环境变量"三档优先级**(详见 §6.3)。

### 5.3 运行
打开 [main.py](./main.py),按需取消注释:

```python
from main import ingest_file, ingest_directory, ingest_web, start_chat
from vectorStore import load_existing_index

# 方式 A:解析单文件 -> 切分 -> 入库
index = ingest_file("Oracle-19c-安装及常用命令.md")

# 方式 B:解析目录下所有文档 -> 切分 -> 入库(流式,适合 10GB+ 大目录)
index = ingest_directory("./data")

# 方式 C:抓取网页 -> 切分 -> 入库(流式,逐 URL)
index = ingest_web(["https://docs.llamaindex.ai/en/stable/"])

# 方式 D:加载已有 Milvus collection(默认走这一条,启动最快)
index = load_existing_index()

# 启动对话
# enable_question_rewriting=True  -> 改写多轮问题(口语化场景推荐)
# enable_question_rewriting=False -> 保留原问题,直接送入检索(默认)
# debug=True  -> 打印系统提示词 / Milvus 检索节点 / LLM prompt / 模型返回
start_chat(index, enable_question_rewriting=False, debug=True)
```

```bash
python main.py
```

启动后命令行交互:
```
=== RAG 对话已就绪,输入 'exit' / 'quit' 退出 ===

请输入: 这份文档的核心结论是什么?
【用户输入】
这份文档的核心结论是什么?
...
助手: ...
```

---

## 6. 关键开关

### 6.1 `enable_question_rewriting` —— 对话模式
| 值 | 行为 | 引擎 |
|---|---|---|
| `True` | 把多轮问题改写为独立查询再检索 | LlamaIndex 内置 `CondenseQuestionChatEngine` |
| `False` (默认) | 保留原问题,直接把历史 + 检索结果送入 LLM | **自实现** `CustomRAGChatEngine`(完全控制消息拼装) |

**何时选哪种?**
- 多轮口语化提问("那它支持中文吗?"、"再详细说说") → `True`(改写)
- 单轮精确查询 / 检索关键词不能丢 → `False`(原文)

**为什么自实现引擎?**
LlamaIndex 内置 `ContextChatEngine` 在不同版本下行为不一致 —— `context_template` 的 `{context_str}` 占位符偶尔不被替换,导致 LLM 看到空白参考资料。`CustomRAGChatEngine` 用最朴素的 `str.format()` 显式拼装,可彻底避免。

### 6.2 `debug` —— RAG 全链路日志
开启后会依次打印:

1. **系统提示词**(System Prompt)
2. **QA User 模板 / Context User 模板**(用于确认模板是否真的生效)
3. 从 **Milvus 检索出来的所有节点**(文本 + 相似度分数 + 来源)
4. 拼装后 **送入 LLM 的 messages**
5. **LLM 返回的回答**
6. Token / 调用参数

示例输出(节选):
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
...节点内容...

======================================================================
【LLM 调用 #1 - 发送的 messages】
======================================================================
[Message 0] role=system
你是一个专业的知识库助手...

[Message 1] role=user
参考资料:
...节点内容...
问题: 这份文档的核心结论
请回答:

======================================================================
【LLM 调用 #1 - 模型返回】
======================================================================
本文档的核心结论是......
```

### 6.3 配置优先级(调用参数 > `config.py` > 环境变量)

所有开关都支持三种写法,**越靠上优先级越高**:

```python
# 1. 调用时显式传参(优先级最高,适合临时改)
start_chat(index, enable_question_rewriting=False, debug=True)

# 2. 直接改 config.py 实例字段(适合项目级默认)
#    config.py 里:
#    CHAT.enable_question_rewriting = False
#    CHAT.debug = True

# 3. 环境变量(适合部署/CI)
#    set ENABLE_QUESTION_REWRITING=false
#    set RAG_DEBUG=true
```

`start_chat(index)` 留空 `enable_question_rewriting` / `debug` 时,会按 2 → 3 顺序回退。

### 6.4 流式入库(大目录 / 多 URL 不爆内存)

`main.ingest_directory` / `main.ingest_web` 默认走**流式**路径,避免一次性把整个目录 / 全部网页内容载入内存:

| 阶段 | 动作 | 内存占用 |
|---|---|---|
| 1. 列文件 | `list_supported_files` 扫目录只拿路径 | ≈ 0(只看文件元信息) |
| 2. 建空索引 | `create_index([], milvus_store=store)` 建好 Milvus 集合 | ≈ 0 |
| 3. for 循环 (× N 文件) | `auto_load` → `auto_split` → `index.insert_nodes(nodes)` | ≈ **单文件文本 + 该文件切出的节点** |
| 4. 每 10 个文件 | `gc.collect()` | — |

旧实现(一次性 `load_directory` + 一次性 `create_index`)的常驻内存 ≈ **全目录文本**,10GB 目录会 OOM。新实现 ≈ **单文件峰值**,10GB 也能稳跑。

> 实现细节见 [main.py](./main.py)。如果你的目录小到可以一次载入(比如 < 100MB),可以直接用 `dataLoader.load_directory` + `create_index(nodes)` 的旧路径,代码更短。

---

## 7. 维护与扩展

### 7.1 添加新的文档类型
1. 在 [dataLoader/loaders.py](./dataLoader/loaders.py) 实现 `load_xxx` 函数
2. 在 [dataLoader/__init__.py](./dataLoader/__init__.py) 导出
3. 在 `auto_load` 的 `mapping` 字典补一项

### 7.2 切换 Embedding / LLM
只改环境变量即可:
- **DeepSeek**:`OPENAI_BASE_URL=https://api.deepseek.com/v1`、`LLM_MODEL=deepseek-chat`
- **通义千问(Qwen / DashScope)**:走 OpenAI 兼容端点,设置 `OPENAI_BASE_URL` + 对应 `LLM_MODEL`
- **Ollama**:`OPENAI_BASE_URL=http://localhost:11434/v1`、`LLM_MODEL=qwen2.5`
- **智谱 / vLLM / dmxapi** —— 设置对应的 `OPENAI_BASE_URL` 与 `LLM_MODEL`

> `EMBED_DIM` 必须与所选 Embedding 模型匹配,否则 Milvus 建表失败。

### 7.3 切换 Milvus 模式
- **嵌入式**(默认):`MILVUS_URI=./milvus_llamaindex.db` + `pip install pymilvus[milvus_lite]`
- **单机 / 集群**:`MILVUS_URI=http://localhost:19530`,需先启动 Milvus 服务,可选填 `MILVUS_TOKEN`

### 7.4 调整切分粒度
[config.py](./config.py) 中的 `Settings.chunk_size` / `Settings.chunk_overlap`,或在 `split_by_sentence` 等函数直接传参。

### 7.5 重建索引
- 嵌入式模式:删除 `milvus_llamaindex.db/` 目录后重新运行
- 服务端模式:`set MILVUS_OVERWRITE=true`,或调用 `milvus_store.drop_collection()`

### 7.6 验证 Milvus 写入是否成功
```bash
python -c "from vectorStore import inspect_milvus; inspect_milvus()"
```
会打印 schema、行数、样本数据,确认 `text` 字段有内容。

### 7.7 直接调 Chat Engine(SDK 方式)
```python
from vectorStore import (
    configure_settings, build_milvus_store, create_index,
    build_chat_engine, CustomRAGChatEngine, RAGDebugHandler,
)
from llama_index.core import Settings
from llama_index.core.callbacks.base import CallbackManager

configure_settings()
store = build_milvus_store()
index = create_index([], milvus_store=store)   # 已有 collection,空 nodes
engine: CustomRAGChatEngine = build_chat_engine(index, debug=True)

resp = engine.chat("你的问题")
print(resp.response)
print("引用节点数:", len(resp.source_nodes))
```

---

## 8. 常见问题(FAQ)

**Q1. `ModuleNotFoundError: No module named 'milvus_lite'`?**
本地嵌入式 Milvus 模式需要它:
```bash
pip install pymilvus[milvus_lite]
```
或改用 Milvus 服务端 URI。

**Q2. `ModuleNotFoundError: No module named 'llama_index.readers.file'`?**
```bash
pip install llama-index-readers-file
```
或重装全部依赖:`pip install -r requirements.txt`。

**Q3. Embedding dimension mismatch?**
`EMBED_DIM` 必须与所选 Embedding 模型输出的维度一致
(OpenAI `text-embedding-3-small` = 1536,`bge-large-zh` = 1024)。

**Q4. 中文 PDF 乱码 / 抽取不出来?**
切换后端:
```python
from dataLoader import load_pdf_auto
load_pdf_auto("x.pdf", backend="pdfplumber")
# 或
load_pdf_auto("x.pdf", backend="pymupdf")
```
扫描件需先 调用pdf_to_image 类,提取图片,再进行OCR。

**Q5. 想用本地 Embedding(BGE / M3E)?**
```bash
pip install sentence-transformers
```
在 [vectorStore/milvus_store.py](./vectorStore/milvus_store.py) 的 `get_embed_model` 中改为 `HuggingFaceEmbedding`。

**Q6. Milvus 检索出 N 个节点但内容是空的?**
检查 `MilvusVectorStore` 是否传了 `output_fields=["text"]` —— Milvus 默认只返回 `id` + `score`。
`build_milvus_store()` 已自动加上;如你自行构造 MilvusVectorStore 请记得带上。

**Q7. `ImportError: cannot import name 'BaseCallbackHandler' from 'llama_index.core.callbacks'`?**
在新版 LlamaIndex 中路径变了,改为:
```python
from llama_index.core.callbacks.base import BaseCallbackHandler
```
本项目已修正,见 [vectorStore/callbacks.py](./vectorStore/callbacks.py)。

**Q8. `TypeError: RAGDebugHandler.end_trace() got an unexpected keyword argument 'trace_map'`?**
LlamaIndex 内部会传 `trace_map` 给 `end_trace`。回调签名需写成:
```python
def end_trace(self, trace_id=None, trace_map=None):
    ...
```
本项目已修正。

**Q9. 看不到 debug 日志?**
确认三处都开:① `start_chat(index, debug=True)` 或 ② `CHAT.debug = True` 或 ③ `RAG_DEBUG=true`。
另外 `RAGDebugHandler` 走 `Settings.callback_manager` —— 不要在外部覆盖 `Settings.callback_manager` 后再调用 `start_chat`。

**Q10. 检索改写开关不生效?**
确认 `start_chat` 调用时传的 `enable_question_rewriting` 与 `config.CHAT.enable_question_rewriting` 一致;同时
`CustomRAGChatEngine`(关闭改写)和 `CondenseQuestionChatEngine`(开启改写)的 system 模板不同,debug 开启时会一并打印出来。

**Q11. 解析大目录内存爆了(OOM)?**
直接用 `main.ingest_directory` / `main.ingest_web`,它内部走流式(先 `list_supported_files` 只列文件再 `index.insert_nodes` 逐文件入库)。不要用 `dataLoader.load_directory` 一次性载入,那会读全目录到内存。

**Q12. `HTMLTagReader` 返回空列表?**
LlamaIndex 默认 `tag="section"` + `ignore_no_id=False`,而一般 HTML 没有带 id 的 `<section>`,所以返回 0。`load_html` 已默认改为 `tag="body"`(取整页正文),直接用即可。

**Q13. `JSONReader.__init__() got an unexpected keyword argument 'levels'`?**
新版 `llama-index-readers-json` 把参数名改成 `levels_back`。`load_json` 已对齐,不要自己用旧名传参。

---
