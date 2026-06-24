# ragasEvaluator — RAGAS 端到端 RAG 评测模块

基于 [Ragas](https://docs.ragas.io/) 框架的 RAG 系统评估工具，提供从文档切分到指标计算的完整评测流水线。

## 评测流程

```
原始文档 (Document[])
    │
    ▼  dataLoader.auto_load
切分后节点 (BaseNode[])
    │
    ├──── A: 喂给 RAG 索引 ──────► VectorStoreIndex
    │                                    │
    └──── B: 喂给 LLM 生成 Q&A ► (question, ground_truth) pairs
                                         │
                                RAGRunner.run(questions)
                                         │
                                (answer, contexts) pairs
                                         │
                         按 question 对齐合并
                                         │
              {question, answer, contexts, ground_truth}
                                         │
                   RAGASEvaluator.evaluate_batch(...)
                                         │
                         各项 RAGAS 指标分数
```

## 评估指标

| 指标 | 含义 | 取值范围 |
|------|------|----------|
| **faithfulness** | 忠实度：回答是否忠实于检索到的上下文 | 0 ~ 1 |
| **answer_relevancy** | 答案相关性：回答是否与问题相关 | 0 ~ 1 |
| **context_precision** | 上下文精确率：检索到的上下文中有用信息的比例 | 0 ~ 1 |
| **context_recall** | 上下文召回率：标准答案所需信息在上下文中的覆盖度 | 0 ~ 1 |

## 文件结构

```
ragasEvaluator/
├── __init__.py              # 模块入口，统一导出 API
├── RAGASevaluatorUtil.py    # RAGAS 评估器（封装 ragas.evaluate）
├── testset_generator.py     # 用 LLM 从 chunk 生成 (question, ground_truth) 问答对
├── rag_runner.py            # 封装 RAG 引擎调用，统一输出 (answer, contexts)
├── pipeline.py              # 端到端评测流水线 + 一站式工厂函数
├── run_eval_demo.py         # 命令行 Demo 脚本
├── _smoke.py                # 冒烟测试（不依赖 LLM/Embedding/Milvus）
└── README.md
```

## 快速开始

### 方式一：命令行 Demo

```bash
# 使用内置样本文档（零外部依赖）
python -m ragasEvaluator.run_eval_demo

# 指定自己的文档
python -m ragasEvaluator.run_eval_demo --file data/sample.pdf

# 选择切分方法
python -m ragasEvaluator.run_eval_demo --splitter agentic

# 控制参数
python -m ragasEvaluator.run_eval_demo --n-qa 2 --top-k 5

# 保存结果到 JSON
python -m ragasEvaluator.run_eval_demo --output result.json
```

### 方式二：Python API

```python
from dataLoader import auto_load
from ragasEvaluator import run_ragas_eval, RAGASEvaluator

# 一站式评测（推荐）
docs = auto_load("data/sample.pdf")
result = run_ragas_eval(docs=docs, n_questions_per_chunk=2)

print("样本数:", result["n"])
print("指标分数:", result["scores"])
```

### 方式三：分步控制

```python
from dataLoader import auto_load
from spliter import split_by_sentence
from vectorStore.milvus_store import configure_settings, create_index
from ragasEvaluator import (
    TestsetGenerator,
    RAGRunner,
    RAGASEvalPipeline,
    RAGASEvaluator,
)

# 1. 加载 & 切分
docs = auto_load("data/sample.pdf")
nodes = split_by_sentence(docs)

# 2. 构建索引
configure_settings()
index = create_index(nodes)

# 3. 创建流水线
pipeline = RAGASEvalPipeline(
    index=index,
    n_questions_per_chunk=2,
    similarity_top_k=3,
    use_chat_engine=False,  # 评测推荐用 query engine（无 memory）
)

# 4. 执行评测
result = pipeline.run(nodes=nodes, evaluator=RAGASEvaluator())
print(result["scores"])
```

### 方式四：单独使用各组件

```python
from ragasEvaluator import TestsetGenerator, RAGRunner, RAGASEvaluator

# 只生成测试集
generator = TestsetGenerator(n_questions_per_chunk=2)
qa_pairs = generator.generate_from_nodes(nodes)
# qa_pairs -> [{"question": ..., "ground_truth": ..., "source_chunk": ...}, ...]

# 只跑 RAG
runner = RAGRunner(index=index, similarity_top_k=3)
rag_results = runner.run(["什么是 RAG？", "Milvus 的优势是什么？"])
# rag_results -> [{"question": ..., "answer": ..., "contexts": [...]}, ...]

# 只评估
evaluator = RAGASEvaluator()
scores = evaluator.evaluate_batch({
    "question": ["什么是 RAG？"],
    "answer": ["RAG 是检索增强生成技术..."],
    "contexts": [["RAG 全称 Retrieval-Augmented Generation..."]],
    "ground_truth": ["RAG 通过检索外部知识库增强大模型回答质量。"],
})
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--file` | None | 文档路径（.pdf/.docx/.md/.txt 等），不指定则用内置样本 |
| `--splitter` | sentence | 切分方法: sentence / agentic / semantic |
| `--n-qa` | 1 | 每 chunk 生成几个问答对（推荐 1~3） |
| `--top-k` | 3 | RAG 检索 top_k |
| `--use-chat-engine` | False | 使用 chat engine（有 memory）；默认 query engine（无 memory） |
| `--output` | None | 将评测结果保存为 JSON 文件路径 |

## 依赖

```bash
pip install ragas datasets
```

> 项目本身已有 llama-index、milvus 等依赖，此处仅列出 ragasEvaluator 额外需要的包。

## 冒烟测试

不依赖 LLM / Embedding / Milvus，验证模块导入、JSON 解析、数据对齐等基础逻辑：

```bash
python ragasEvaluator/_smoke.py
```

## 注意事项

- **评测推荐使用 query engine**（`use_chat_engine=False`），避免多轮对话的 memory 污染评测结果。
- **每 chunk 生成 1~3 个问答对**即可，过多会增加 LLM 调用成本且可能降低质量。
- **LLM 输出 JSON 解析**内置多层容错（Markdown 包裹、前后废话、单引号等），但极端情况仍可能解析失败，此时该 chunk 会被跳过。
- **单条失败不阻塞**：无论是测试集生成还是 RAG 查询，单条数据失败只会打印警告并跳过，不影响整体流程。
