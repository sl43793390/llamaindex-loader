import sys
import os

# 确保项目根目录在 sys.path 中，否则找不到 dataLoader / vectorStore 等兄弟包
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ragasEvaluator import run_ragas_eval

# ============================================================
# 选择评测模式:
#   rebuild_index = True  → 重新加载文档、切分、构建索引（首次评测用）
#   rebuild_index = False → 使用已有 Milvus 索引，跳过文档加载和索引构建
# ============================================================
rebuild_index = False

if rebuild_index:
    # 模式1: 重新构建索引（需要提供文档）
    from dataLoader.loaders import auto_load
    docs = auto_load("demo.docx")
    result = run_ragas_eval(docs=docs, n_questions_per_chunk=2)
else:
    # 模式2: 使用已有索引（无需提供文档，自动从 Milvus 加载）
    result = run_ragas_eval(rebuild_index=False, n_questions_per_chunk=2)


print("样本数:", result["n"])
print("指标分数:", result["scores"])
#  'scores': {'faithfulness': 0.9048, 'answer_relevancy': 0.8782, 'context_precision': 0.8722, 'context_recall': 0.8889}, 
# 'n': 18}