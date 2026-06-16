"""
main
~~~~~~
LlamaIndex 文档加载 + Milvus RAG 对话 示例入口。

完整流水线:
    解析 (dataLoader) -> 切分 (spliter) -> 存入 Milvus (vectorStore) -> 检索对话

运行方式::

    python main.py

切换数据源、调试开关见 ``if __name__ == "__main__"`` 段落。

内存说明:
    ``ingest_directory`` / ``ingest_web`` 走"流式"路径:先枚举路径,
    再**逐个文件 / URL** 地 ``load -> split -> embed -> insert``,
    处理完即释放,避免一次性把整个目录 / 全部网页内容载入内存。
    这对数十 GB 的文档目录尤其重要。
"""
import gc
from pathlib import Path
from typing import List, Optional, Union

from llama_index.core import VectorStoreIndex

from dataLoader import auto_load, list_supported_files, load_web
from spliter import auto_split
from vectorStore import (
    build_chat_engine,
    build_milvus_store,
    chat_loop,
    configure_settings,
    create_index,
    load_existing_index,
)


# ============================================================
# 数据入库
# ============================================================
def ingest_file(file_path: Union[str, Path]) -> VectorStoreIndex:
    """
    单文件入库流水线:解析 -> 切分 -> 写入 Milvus。

    Args:
        file_path: 任意受支持格式的文件路径。

    Returns:
        已写入 Milvus 的 ``VectorStoreIndex``。
    """
    print(f"\n=== 加载: {file_path} ===")
    docs = auto_load(file_path)

    # 根据后缀选切分器
    ext = Path(file_path).suffix.lower()
    type_map = {
        ".md": "markdown", ".markdown": "markdown",
        ".html": "html", ".htm": "html",
        ".json": "json",
    }
    doc_type = type_map.get(ext, "text")
    nodes = auto_split(docs, doc_type=doc_type)
    print(f"切分得到 {len(nodes)} 个节点")

    # 写入 Milvus
    milvus_store = build_milvus_store()
    return create_index(nodes, milvus_store=milvus_store)


def ingest_directory(
    directory: Union[str, Path],
    doc_type: str = "text",
    recursive: bool = True,
) -> VectorStoreIndex:
    """
    目录下所有支持文档**逐个**流式入库,避免大目录 OOM。

    流程:
        1) :func:`list_supported_files` 扫描目录拿到所有文件路径(**不读内容**);
        2) 构造一个**空**的 ``VectorStoreIndex``(Milvus 集合已建好);
        3) for 循环里:
              load(file) -> split(docs) -> index.insert(nodes)
           每轮结束后 ``del`` 局部变量 + ``gc.collect()`` 释放内存;
        4) 任意单文件失败被捕获并打印,该文件跳过,不影响后续文件。

    内存模型:
        - 旧实现(``load_directory`` + 一次性 ``create_index``):常驻 ≈ 全目录文本。
        - 本实现:常驻 ≈ 单文件文本 + 该文件切出的节点。10GB 目录也能跑。

    Args:
        directory: 目录路径。
        doc_type: 切分器类型,见 :func:`spliter.auto_split`。
        recursive: 是否递归子目录。

    Returns:
        已写入 Milvus 的 ``VectorStoreIndex``。
    """
    print(f"\n=== 加载目录(流式): {directory} ===")

    # 1) 列举文件,但不读内容。
    files = list_supported_files(directory, recursive=recursive)
    if not files:
        print("未发现受支持的文件,直接返回空集合的 index。")
        return create_index([], milvus_store=build_milvus_store())

    total_size = sum((p.stat().st_size for p in files), 0)
    print(f"发现 {len(files)} 个文件,合计 {total_size / 1024 / 1024:.2f} MB,开始逐个处理...")

    # 2) 一次性建好 Milvus 集合(空),后续 insert 都用这个。
    store = build_milvus_store()
    index = create_index([], milvus_store=store)

    # 3) 逐文件处理。
    success_count = 0
    fail_count = 0
    total_nodes = 0
    type_map = {
        ".md": "markdown", ".markdown": "markdown",
        ".html": "html", ".htm": "html",
        ".json": "json",
    }

    for i, file_path in enumerate(files, 1):
        rel = file_path.relative_to(Path(directory)) if file_path.is_relative_to(Path(directory)) else file_path
        try:
            print(f"\n[{i}/{len(files)}] {rel}")

            # 3.1 加载
            docs = auto_load(file_path)
            if not docs:
                print("  · 跳过(空内容)")
                continue

            # 3.2 切分(按文件后缀决定切分器类型)
            file_doc_type = type_map.get(file_path.suffix.lower(), doc_type)
            nodes = auto_split(docs, doc_type=file_doc_type)
            if not nodes:
                print("  · 跳过(切分后无节点)")
                continue

            # 3.3 入库(Embedding 在这一步发生)
            index.insert_nodes(nodes)
            total_nodes += len(nodes)
            success_count += 1
            print(f"  · 切分得到 {len(nodes)} 个节点 (累计 {total_nodes})")

        except Exception as e:  # noqa: BLE001 - 单文件失败要吞掉,不能拖垮整批
            fail_count += 1
            print(f"  · 失败,跳过: {e}")

        finally:
            # 3.4 显式释放引用,让大对象能尽快被回收。
            docs = None  # type: ignore[assignment]
            nodes = None  # type: ignore[assignment]
            # 每 10 个文件主动触发一次 full GC,缓解内存碎片。
            if i % 10 == 0:
                gc.collect()

    print(
        f"\n=== 目录入库完成: 成功 {success_count}, 失败 {fail_count}, "
        f"累计 {total_nodes} 个节点 ==="
    )
    # 最后再跑一次 GC,把残留临时对象清掉。
    gc.collect()
    return index


def ingest_web(
    urls: List[str],
    doc_type: str = "text",
) -> VectorStoreIndex:
    """
    抓取网页内容并**逐 URL**流式入库。

    流程与 :func:`ingest_directory` 相同,只是数据源是 URL 列表,
    同样按 URL 逐个拉取、解析、切分、入库、释放,避免一次抓取整个
    列表的内容导致内存爆掉。

    Args:
        urls: 网页 URL 列表。
        doc_type: 切分器类型,见 :func:`spliter.auto_split`。

    Returns:
        已写入 Milvus 的 ``VectorStoreIndex``。
    """
    print(f"\n=== 加载网页(流式): {len(urls)} 个 URL ===")

    if not urls:
        print("URL 列表为空,直接返回空集合的 index。")
        return create_index([], milvus_store=build_milvus_store())

    store = build_milvus_store()
    index = create_index([], milvus_store=store)

    success_count = 0
    fail_count = 0
    total_nodes = 0

    for i, url in enumerate(urls, 1):
        try:
            print(f"\n[{i}/{len(urls)}] {url}")
            # load_web 接受列表,这里传单元素列表让它只解析当前 URL
            docs = load_web([url])
            if not docs:
                print("  · 跳过(空内容)")
                continue
            nodes = auto_split(docs, doc_type=doc_type)
            if not nodes:
                print("  · 跳过(切分后无节点)")
                continue
            index.insert_nodes(nodes)
            total_nodes += len(nodes)
            success_count += 1
            print(f"  · 切分得到 {len(nodes)} 个节点 (累计 {total_nodes})")
        except Exception as e:  # noqa: BLE001
            fail_count += 1
            print(f"  · 失败,跳过: {e}")
        finally:
            docs = None  # type: ignore[assignment]
            nodes = None  # type: ignore[assignment]
            if i % 10 == 0:
                gc.collect()

    print(
        f"\n=== 网页入库完成: 成功 {success_count}, 失败 {fail_count}, "
        f"累计 {total_nodes} 个节点 ==="
    )
    gc.collect()
    return index


# ============================================================
# 对话启动
# ============================================================
def start_chat(
    index: VectorStoreIndex,
    enable_question_rewriting: Optional[bool] = None,
    debug: Optional[bool] = None,
) -> None:
    """
    启动 RAG 命令行对话循环。

    Args:
        index: 已构建好的 ``VectorStoreIndex``。
        enable_question_rewriting:
            - True :使用 ``CondenseQuestionChatEngine``,会把多轮问题改写为独立查询
            - False:使用 :class:`CustomRAGChatEngine`,保留原问题直接检索
            - None :从 ``config.CHAT.enable_question_rewriting`` 读取
        debug:
            - True :打印系统提示词、Milvus 检索节点、送入 LLM 的 prompt、模型返回
            - False/None:从 ``config.CHAT.debug`` 读取
    """
    chat_engine = build_chat_engine(
        index,
        enable_question_rewriting=enable_question_rewriting,
        debug=debug,
    )
    print(f"[配置] 问题改写: {'开启' if enable_question_rewriting else '关闭'}")
    print(f"[配置] 调试日志: {'开启' if debug else '关闭'}")
    chat_loop(chat_engine)


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    # 1. 注册全局 Settings(Embedding + LLM)
    configure_settings()

    # 2. 选择数据源(三选一,注释切换)

    # 方式 A:解析单文件 -> 切分 -> 入库(单文件,内存友好)
    # index = ingest_file("Oracle-19c-安装及常用命令.md")

    # 方式 B:解析目录下所有文档 -> 切分 -> 入库
    #   流式:逐个文件 load -> split -> insert,适合 10GB+ 大目录。
    # index = ingest_directory("./data")

    # 方式 C:抓取网页 -> 切分 -> 入库
    #   流式:逐 URL 抓取 + 入库,避免一次性载入全部网页。
    # index = ingest_web(["https://docs.llamaindex.ai/en/stable/"])

    # =============已存在 Milvus collection,直接加载(不重新写入)=============
    index = load_existing_index()

    # 3. 启动对话
    #    enable_question_rewriting=True  -> 改写多轮问题(默认,口语化场景推荐)
    #    enable_question_rewriting=False -> 保留原问题,直接送入检索
    #    留 None 则从环境变量 / config.py 读取
    #
    #    debug=True  -> 打印 系统提示词 / Milvus 检索节点 / LLM prompt / 模型返回
    #    debug=False -> 静默运行
    #    留 None 则从环境变量 RAG_DEBUG / config.CHAT.debug 读取
    start_chat(index, enable_question_rewriting=False, debug=True)
