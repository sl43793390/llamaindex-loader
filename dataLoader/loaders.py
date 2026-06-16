"""
dataLoader.loaders
~~~~~~~~~~~~~~~~~~~~
使用 LlamaIndex 实现的多种文档解析器,统一返回 ``List[Document]``。

支持格式
--------
- Word  (.docx)        -> DocxReader
- Excel (.xlsx/.xls)   -> PandasExcelReader
- PDF   (.pdf)         -> PDFReader / PyMuPDFReader / PDFPlumberReader
- Txt   (.txt)         -> SimpleDirectoryReader
- Markdown (.md)       -> MarkdownReader
- HTML  (.html)        -> HTMLTagReader
- JSON  (.json)        -> JSONReader
- Web   (URL 列表)     -> BeautifulSoupWebReader / SimpleWebPageReader

对未识别后缀,可使用 ``auto_load`` 按后缀自动路由;
批量解析目录使用 ``load_directory``;
需要"只列文件不读内容"的流式场景使用 :func:`list_supported_files`。
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document
from llama_index.readers.file import (
    DocxReader,
    PandasExcelReader,
    PDFReader,
    PyMuPDFReader,
    MarkdownReader,
    HTMLTagReader,
)
from llama_index.readers.json import JSONReader
from llama_index.readers.web import (
    BeautifulSoupWebReader,
    SimpleWebPageReader,
)

# 项目支持的全部文件后缀(用于 auto_load / load_directory / list_supported_files)。
SUPPORTED_EXTS: tuple = (
    ".docx", ".xlsx", ".xls", ".pdf",
    ".txt", ".md", ".markdown",
    ".html", ".htm", ".json",
)

# `PDFPlumberReader` 在新版 llama-index-readers-file 中被移除,
# 这里做可选导入,失败时 ``load_pdf_plumber`` / ``load_pdf_auto(backend="pdfplumber")``
# 会抛 ``ImportError`` 提示用户安装旧版或换用其他后端。
try:
    from llama_index.readers.file import PDFPlumberReader
    _HAS_PDFPLUMBER = True
except ImportError:  # pragma: no cover - 依赖缺失分支
    PDFPlumberReader = None  # type: ignore[assignment]
    _HAS_PDFPLUMBER = False


# ============================================================
# 1. Word (.docx)
# ============================================================
def load_word(file_path: Union[str, Path]) -> List[Document]:
    """
    解析 Word 文档 (.docx)。

    Args:
        file_path: .docx 文件路径。

    Returns:
        Document 列表(每个段落可能为一个 Document)。
    """
    reader = DocxReader()
    return reader.load_data(file=file_path)


# ============================================================
# 2. Excel (.xlsx / .xls)
# ============================================================
def load_excel(
    file_path: Union[str, Path],
    sheet_name: Optional[str] = None,
    pandas_config: Optional[Dict] = None,
) -> List[Document]:
    """
    解析 Excel 文档。

    Args:
        file_path: .xlsx / .xls 文件路径。
        sheet_name: 指定 sheet 名;为 None 时读取全部 sheet。
        pandas_config: 透传给 pandas.read_excel 的额外参数。

    Returns:
        Document 列表(每个 sheet 通常对应一个 Document)。
    """
    reader = PandasExcelReader(pandas_config=pandas_config or {})
    return reader.load_data(file=file_path, sheet_name=sheet_name)


# ============================================================
# 3. Txt
# ============================================================
def load_txt(
    file_path: Union[str, Path],
    encoding: str = "utf-8",
) -> List[Document]:
    """
    解析纯文本文件。

    Args:
        file_path: .txt 文件路径。
        encoding: 文件编码,默认 utf-8。

    Returns:
        Document 列表。
    """
    reader = SimpleDirectoryReader(
        input_files=[str(file_path)],
        encoding=encoding,
    )
    return reader.load_data()


# ============================================================
# 4. Markdown
# ============================================================
def load_markdown(file_path: Union[str, Path]) -> List[Document]:
    """
    解析 Markdown 文档,按章节切分 Document。

    Args:
        file_path: .md 文件路径。

    Returns:
        Document 列表。
    """
    reader = MarkdownReader()
    return reader.load_data(file=file_path)


# ============================================================
# 5. HTML (本地文件)
# ============================================================
def load_html(
    file_path: Union[str, Path],
    tag: str = "body",
    ignore_no_id: bool = False,
) -> List[Document]:
    """
    解析本地 HTML 文档(按标签提取正文内容)。

    Args:
        file_path: .html / .htm 文件路径。
        tag: 提取的 HTML 标签名,默认 ``"body"``。
            - ``"section"``:LlamaIndex 默认,只取带 id 的 <section>(一般 HTML
              没有 id,会返回空列表)。
            - ``"body"``:取整页正文,适合一般 HTML 文件。
            - ``"p"``/``"div"``/其它:按需选,多个同名标签会分别成 Document。
        ignore_no_id: 是否只保留带 ``id`` 的标签块,默认 ``False``(全保留)。
            注意 LlamaIndex 的语义:**``True`` 会丢掉所有没 id 的标签**,
            对 ``tag="body"`` 这种场景,通常应保持 ``False``。

    Returns:
        Document 列表。
    """
    reader = HTMLTagReader(tag=tag, ignore_no_id=ignore_no_id)
    return reader.load_data(file=file_path)


# ============================================================
# 6. JSON
# ============================================================
def load_json(
    file_path: Union[str, Path],
    levels_back: Optional[int] = None,
    collapse_length: Optional[int] = None,
    is_jsonl: bool = False,
) -> List[Document]:
    """
    解析 JSON / JSONL 文档。

    Args:
        file_path: .json / .jsonl 文件路径。
        levels_back: 回溯嵌套层数,0 表示全展开;None 表示把整段 JSON 展平
            为一个 Document。
        collapse_length: 当 ``levels_back`` 不为 None 时,超过该长度的 JSON
            片段会被折叠成单行。
        is_jsonl: 是否按 JSONL 解析。

    Returns:
        Document 列表。

    Note:
        新版 ``llama-index-readers-json`` 的参数名是 ``levels_back``(旧版叫
        ``levels``);用关键字传参最稳。
    """
    reader = JSONReader(
        levels_back=levels_back,
        collapse_length=collapse_length,
        is_jsonl=is_jsonl,
    )
    return reader.load_data(input_file=str(file_path))


# ============================================================
# 7. PDF —— 三种后端可选
# ============================================================
def load_pdf(
    file_path: Union[str, Path],
    return_full_document: bool = False,
) -> List[Document]:
    """
    解析 PDF(基于 pypdf,依赖最轻)。

    Args:
        file_path: .pdf 文件路径。
        return_full_document:
            - False:每页一个 Document
            - True :整本 PDF 合并为一个 Document

    Returns:
        Document 列表。
    """
    reader = PDFReader(return_full_document=return_full_document)
    return reader.load_data(file=file_path)


def load_pdf_pymupdf(file_path: Union[str, Path]) -> List[Document]:
    """
    使用 PyMuPDF 解析 PDF,对复杂排版/扫描件更友好(推荐)。

    Args:
        file_path: .pdf 文件路径。

    Returns:
        Document 列表。
    """
    reader = PyMuPDFReader()
    return reader.load_data(file_path=str(file_path))


def load_pdf_plumber(file_path: Union[str, Path]) -> List[Document]:
    """
    使用 pdfplumber 解析 PDF,擅长抽取表格(财务报表类 PDF 推荐)。

    Args:
        file_path: .pdf 文件路径。

    Returns:
        Document 列表。

    Raises:
        ImportError: 当前 ``llama-index-readers-file`` 版本未提供 ``PDFPlumberReader``。
    """
    if not _HAS_PDFPLUMBER:
        raise ImportError(
            "当前 llama-index-readers-file 未提供 PDFPlumberReader,"
            "请改用 backend='pymupdf' 或安装支持 pdfplumber 的旧版。"
        )
    reader = PDFPlumberReader()
    return reader.load_data(file=file_path)


def load_pdf_auto(
    file_path: Union[str, Path],
    backend: str = "pymupdf",
) -> List[Document]:
    """
    统一 PDF 解析入口。

    Args:
        file_path: .pdf 文件路径。
        backend: 后端类型,可选 ``"pymupdf"`` / ``"pdfplumber"`` / ``"pypdf"``。

    Returns:
        Document 列表。

    Raises:
        ValueError: backend 不是以上三种之一时抛出。
    """
    backend = backend.lower()
    if backend == "pymupdf":
        return load_pdf_pymupdf(file_path)
    if backend == "pdfplumber":
        return load_pdf_plumber(file_path)
    if backend == "pypdf":
        return load_pdf(file_path)
    raise ValueError(f"不支持的 PDF 后端: {backend}")


# ============================================================
# 8. Web
# ============================================================
def load_web(
    urls: List[str],
    use_bs4: bool = True,
) -> List[Document]:
    """
    抓取并解析网页内容。

    Args:
        urls: 网页 URL 列表。
        use_bs4:
            - True :使用 BeautifulSoupWebReader(更灵活,推荐)
            - False:使用 SimpleWebPageReader(简单抓取)

    Returns:
        Document 列表(每个 URL 通常对应一个 Document)。
    """
    if use_bs4:
        reader = BeautifulSoupWebReader()
    else:
        reader = SimpleWebPageReader(html_to_text=True)
    return reader.load_data(urls=urls)


# ============================================================
# 9. 通用入口(按后缀自动选择 Loader)
# ============================================================
def auto_load(file_path: Union[str, Path]) -> List[Document]:
    """
    根据文件后缀自动选择合适的 Loader。

    Args:
        file_path: 任意受支持格式的文件路径。

    Returns:
        Document 列表。

    Raises:
        ValueError: 后缀无法识别时退化为 SimpleDirectoryReader。
    """
    ext = Path(file_path).suffix.lower()
    mapping = {
        ".docx": load_word,
        ".xlsx": load_excel,
        ".xls": load_excel,
        ".pdf": load_pdf_auto,
        ".txt": load_txt,
        ".md": load_markdown,
        ".markdown": load_markdown,
        ".html": load_html,
        ".htm": load_html,
        ".json": load_json,
    }
    if ext not in mapping:
        # 未识别后缀退化为通用加载器
        return SimpleDirectoryReader(input_files=[str(file_path)]).load_data()
    return mapping[ext](file_path)


# ============================================================
# 10. 批量加载目录
# ============================================================
def load_directory(
    directory: Union[str, Path],
    recursive: bool = True,
    required_exts: Optional[List[str]] = None,
) -> List[Document]:
    """
    加载目录下所有支持的文档(默认递归)。

    警告:
        本函数会一次性把目录里所有文件读入内存。
        对超大型目录(数 GB 以上)请改用 :func:`list_supported_files` +
        ``index.insert(nodes)`` 的流式模式(见 ``main.ingest_directory``)。

    Args:
        directory: 目录路径。
        recursive: 是否递归子目录。
        required_exts: 限定加载的文件后缀;为 None 时加载全部支持格式(见 :data:`SUPPORTED_EXTS`)。

    Returns:
        Document 列表。
    """
    if required_exts is None:
        required_exts = list(SUPPORTED_EXTS)
    reader = SimpleDirectoryReader(
        input_dir=str(directory),
        recursive=recursive,
        required_exts=required_exts,
    )
    return reader.load_data()


def list_supported_files(
    directory: Union[str, Path],
    recursive: bool = True,
    required_exts: Optional[List[str]] = None,
) -> List[Path]:
    """
    只列出目录下受支持的文件路径,**不读取文件内容**。

    与 :func:`load_directory` 的区别:
        - ``load_directory`` 一次性把全目录载入,大目录(数 GB)易触发 OOM。
        - ``list_supported_files`` 只做文件系统扫描,内存常驻只与"文件数"相关,
          适合配合 ``index.insert(nodes)`` 做"逐文件加载-切分-入库"的流式入库。

    Args:
        directory: 目录路径。
        recursive: 是否递归子目录。
        required_exts: 限定后缀;为 None 时使用 :data:`SUPPORTED_EXTS`。

    Returns:
        排序后的 :class:`pathlib.Path` 列表。
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"不是有效目录: {directory}")

    ext_set = (
        tuple(e.lower() for e in required_exts)
        if required_exts is not None
        else SUPPORTED_EXTS
    )
    ext_set = tuple(e if e.startswith(".") else f".{e}" for e in ext_set)

    if recursive:
        # pathlib 的 rglob("*"):跨平台,符号链接按 os.walk 默认行为。
        # 加一层 is_file() 过滤,顺便跳过断链 / 隐藏文件目录项。
        matches = (
            p for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in ext_set
        )
    else:
        matches = (
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in ext_set
        )
    return sorted(matches)
