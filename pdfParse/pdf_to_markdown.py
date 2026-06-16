"""
PDF → Markdown 解析器。

使用 PyMuPDF (pymupdf) 直接从 PDF 抽取带格式信息的文本块,
然后启发式地识别:
    - 一级 / 二级 / 三级 / ... 标题 (按字号相对正文的大小)
    - 有序 / 无序列表 (按行首符号)
    - 代码块 (按等宽字体)
    - 表格 (用 PyMuPDF 的 ``find_tables``)
    - 加粗 / 斜体 (用 ``text-mark`` 包裹)
    - 链接 (PDF 里的 URI 注解)
    - 图片 (可选,把图保存到磁盘并以 ``![alt](path)`` 形式插入)

依赖:
    - pymupdf  (1.24+)
    - Pillow   (可选,只在 ``extract_images=True`` 时使用)
"""
from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pymupdf


# ============================================================
# 内部数据结构
# ============================================================
@dataclass
class _SpanInfo:
    """从 ``page.get_text("dict")`` 抽取出的"行 + 跨字"信息。"""

    text: str
    font: str
    size: float
    flags: int
    color: int
    bbox: Tuple[float, float, float, float]
    is_monospace: bool = False
    is_bold: bool = False
    is_italic: bool = False


@dataclass
class _LineInfo:
    """同一行的所有跨字聚合。"""

    y0: float
    y1: float
    spans: List[_SpanInfo] = field(default_factory=list)
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.spans)

    @property
    def avg_size(self) -> float:
        if not self.spans:
            return 0.0
        return sum(s.size for s in self.spans) / len(self.spans)

    @property
    def max_size(self) -> float:
        if not self.spans:
            return 0.0
        return max(s.size for s in self.spans)

    @property
    def all_monospace(self) -> bool:
        return bool(self.spans) and all(s.is_monospace for s in self.spans)


# ============================================================
# 主类
# ============================================================
class PDFToMarkdown:
    """
    PDF → Markdown 转换器。

    用法::

        parser = PDFToMarkdown("report.pdf")
        md = parser.convert()                 # 整篇转成一段 Markdown
        pages = parser.convert_pages()        # 每页一个 Markdown 段

        # 或流式遍历:
        for page_no, md_chunk in parser.iter_pages():
            ...

    Args:
        file_path: 待解析的 PDF 文件路径。
        body_size_hint:
            正文字号,用于反推标题级别。``None`` 时按页自动估算
            (取该页字号出现次数最多的值作为 body)。
        code_font_pattern:
            视作"等宽字体 / 代码"的字体名正则(不区分大小写)。
        heading_size_ratios:
            字号 / body 的比值 → 标题级别。默认 ``{1.6: 1, 1.35: 2, 1.18: 3, 1.08: 4}``。
        extract_images:
            是否把 PDF 里的内嵌图片导出到 ``image_dir`` 并在 markdown 中
            用 ``![](path)`` 引用。
        image_dir:
            图片保存目录,``None`` 时用 ``<pdf_stem>_images/``。

    Note:
        本类是 **纯文本解析** —— 不调用任何外部 LLM,适合批量预处理。
        对扫描件 / 图片 PDF,先用 :class:`PDFToImage` 转图,再走 OCR
        (如 PaddleOCR / Tesseract) 拿到文字。
    """

    _DEFAULT_HEADING_RATIOS: Dict[float, int] = {
        1.6: 1,   # H1
        1.35: 2,  # H2
        1.18: 3,  # H3
        1.08: 4,  # H4
    }

    def __init__(
        self,
        file_path: Union[str, Path],
        body_size_hint: Optional[float] = None,
        # 常见正宽 / 代码字体。覆盖 Windows / macOS / Linux / Office / LaTeX 各家。
        # 注意:|?<= 用 \b 单词边界,避免 `monaco` 误匹配 `monaco-narrow` 之类的也算;
        # `mono` / `code` 用正向断言模糊匹配(必须独立成段才视为 mono)。
        code_font_pattern: str = (
            r"(\bcourier|\bconsolas|\bmenlo|\bmonaco|"
            r"\blucida[\s_]*console|"
            r"\bdejavu[\s_]*sans[\s_]*mono|"
            r"\bliberation[\s_]*mono|"
            r"\binconsolata|\bubuntu[\s_]*mono|"
            r"\bandale[\s_]*mono|\bfira[\s_]*code|"
            r"\bsource[\s_]*code[\s_]*pro|"
            r"(?<![a-z])mono(?![a-z])|"
            r"(?<![a-z])code(?![a-z]))"
        ),
        heading_size_ratios: Optional[Dict[float, int]] = None,
        extract_images: bool = False,
        image_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.file_path = Path(file_path)
        if not self.file_path.is_file():
            raise FileNotFoundError(f"PDF 文件不存在: {self.file_path}")

        self.body_size_hint = body_size_hint
        self.code_font_pattern = re.compile(code_font_pattern, re.IGNORECASE)
        self.heading_size_ratios: Dict[float, int] = (
            dict(heading_size_ratios)
            if heading_size_ratios is not None
            else dict(self._DEFAULT_HEADING_RATIOS)
        )
        self.extract_images = extract_images
        self.image_dir: Optional[Path] = (
            Path(image_dir) if image_dir is not None
            else self.file_path.with_name(f"{self.file_path.stem}_images")
        ) if extract_images else None

        # 延迟到 convert() 打开,避免构造时 IO 出错搞坏 import。
        self._doc: Optional[pymupdf.Document] = None

    # ---------- 上下文协议 ----------
    def __enter__(self) -> "PDFToMarkdown":
        self._open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """关闭底层 PDF 句柄。"""
        if self._doc is not None:
            self._doc.close()
            self._doc = None

    def _open(self) -> pymupdf.Document:
        if self._doc is None:
            self._doc = pymupdf.open(str(self.file_path))
        return self._doc

    @property
    def page_count(self) -> int:
        """PDF 页数(会触发首次打开)。"""
        return self._open().page_count

    # ---------- 公开 API ----------
    def convert(self) -> str:
        """
        整篇 PDF 转成一个 Markdown 字符串,页与页之间用 ``\\n\\n---\\n\\n`` 分隔。

        Returns:
            Markdown 文本。
        """
        parts: List[str] = []
        for _page_no, page_md in self.iter_pages():
            if page_md.strip():
                parts.append(page_md.strip())
        return "\n\n---\n\n".join(parts)

    def convert_pages(self) -> List[str]:
        """
        每页一个 Markdown 段,返回 ``List[str]``(下标 = 页号 - 1)。

        Returns:
            每页 Markdown 文本的列表。
        """
        return [md for _, md in self.iter_pages()]

    def iter_pages(self) -> Iterator[Tuple[int, str]]:
        """
        流式遍历每一页,产出 ``(page_no_1based, markdown_chunk)``。

        Yields:
            ``(page_no, markdown)``。
        """
        doc = self._open()
        for i, page in enumerate(doc, 1):
            yield i, self._convert_page(page, page_no=i)

    def save(self, output_path: Union[str, Path]) -> Path:
        """
        把整篇 Markdown 写到文件。

        Args:
            output_path: 输出 .md 路径。

        Returns:
            写入后的 :class:`Path`。
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.convert(), encoding="utf-8")
        return output_path

    # ---------- 单页解析 ----------
    def _convert_page(self, page: pymupdf.Page, page_no: int) -> str:
        # 1) 解析文本行
        lines = self._extract_lines(page)

        # 2) 反推正文字号(用于标题判定)
        body_size = self.body_size_hint or self._estimate_body_size(lines)

        # 3) 提取表格
        tables = self._extract_tables(page)

        # 4) 提取链接
        link_map = self._extract_links(page)

        # 5) 可选:导出图片
        image_map: Dict[float, List[Path]] = {}
        if self.extract_images and self.image_dir is not None:
            image_map = self._extract_images(page, page_no)

        # 6) 把表格按 bbox 插回到行流(避免重复)
        return self._assemble_markdown(
            page,
            lines=lines,
            tables=tables,
            link_map=link_map,
            image_map=image_map,
            body_size=body_size,
            page_no=page_no,
        )

    # ---------- 文本行抽取 ----------
    def _extract_lines(self, page: pymupdf.Page) -> List[_LineInfo]:
        """从 page.get_text("dict") 聚合出 _LineInfo 列表,按 y 坐标降序。"""
        raw = page.get_text("dict")
        lines: List[_LineInfo] = []

        for block in raw.get("blocks", []):
            if block.get("type", 0) != 0:  # 0 = 文本块
                continue
            for line in block.get("lines", []):
                spans: List[_SpanInfo] = []
                y0 = line.get("bbox", [0, 0, 0, 0])[1]
                y1 = line.get("bbox", [0, 0, 0, 0])[3]
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if not text:
                        continue
                    font = span.get("font", "")
                    size = float(span.get("size", 0))
                    flags = int(span.get("flags", 0))
                    color = int(span.get("color", 0))
                    spans.append(
                        _SpanInfo(
                            text=text,
                            font=font,
                            size=size,
                            flags=flags,
                            color=color,
                            bbox=tuple(span.get("bbox", (0, 0, 0, 0))),
                            is_monospace=bool(self.code_font_pattern.search(font)),
                            # pymupdf: bit 0 = superscripted, bit 1 = italic, bit 2 = serifed
                            # bit 3 = monospaced, bit 4 = bold
                            is_bold=bool(flags & 16) or "bold" in font.lower(),
                            is_italic=bool(flags & 2) or "italic" in font.lower(),
                        )
                    )
                if not spans:
                    continue
                lines.append(
                    _LineInfo(
                        y0=y0,
                        y1=y1,
                        spans=spans,
                        bbox=tuple(line.get("bbox", (0, 0, 0, 0))),
                    )
                )

        # y0 越大 = 越靠上,排序时翻转 → 越上越前
        lines.sort(key=lambda l: (-l.y0, l.bbox[0]))
        return lines

    @staticmethod
    def _estimate_body_size(lines: List[_LineInfo]) -> float:
        """
        估算正文字号(用"字符数加权众数")。

        比直接取频次最高的字号更稳:
            表格 / 页脚短行不会因为"行数多"被误判为正文。
        """
        if not lines:
            return 10.0
        # 用字符数加权:每个 span 的 size 按 text 长度计数
        weight: Dict[float, int] = {}
        for ln in lines:
            for sp in ln.spans:
                if not sp.text.strip():
                    continue
                k = round(sp.size, 1)
                weight[k] = weight.get(k, 0) + max(len(sp.text), 1)
        if not weight:
            # 退化:用绝对频次
            counter: Dict[float, int] = {}
            for ln in lines:
                for sp in ln.spans:
                    k = round(sp.size, 1)
                    counter[k] = counter.get(k, 0) + 1
            if not counter:
                return 10.0
            return max(counter.items(), key=lambda kv: kv[1])[0]
        return max(weight.items(), key=lambda kv: kv[1])[0]

    # ---------- 表格抽取 ----------
    @staticmethod
    def _extract_tables(page: pymupdf.Page) -> List[Dict[str, Any]]:
        """
        用 PyMuPDF 的 ``find_tables()`` 抽表格。

        返回 ``[{bbox, headers, rows}, ...]``;
        没有表格时返空列表。
        """
        try:
            found = page.find_tables()
        except Exception:
            return []
        out: List[Dict[str, Any]] = []
        for t in getattr(found, "tables", []):
            try:
                df = t.to_pandas()
            except Exception:
                continue
            out.append(
                {
                    "bbox": tuple(t.bbox),
                    "headers": list(df.columns.astype(str)),
                    "rows": df.astype(str).values.tolist(),
                }
            )
        return out

    # ---------- 链接抽取 ----------
    @staticmethod
    def _extract_links(page: pymupdf.Page) -> Dict[Tuple[int, int, int, int], str]:
        """
        抽 PDF 里的链接 / URI 注解,key = bbox 像素矩形,value = 目标 URI / 名字。
        """
        result: Dict[Tuple[int, int, int, int], str] = {}
        try:
            for link in page.get_links():
                uri = link.get("uri") or link.get("name") or ""
                if not uri:
                    continue
                # 转成 int key 以便 hash
                bbox = tuple(int(v) for v in link.get("from", (0, 0, 0, 0)))
                result[bbox] = uri
        except Exception:
            pass
        return result

    # ---------- 图片导出 ----------
    def _extract_images(
        self,
        page: pymupdf.Page,
        page_no: int,
    ) -> Dict[float, List[Path]]:
        """
        把当前页的所有图片导出到 ``image_dir``,按页 + 序号命名。

        Returns:
            ``{y0: [Path, ...]}``,y0 用来在 markdown 中按垂直位置插入。
        """
        assert self.image_dir is not None
        self.image_dir.mkdir(parents=True, exist_ok=True)
        result: Dict[float, List[Path]] = {}
        for img_index, img in enumerate(page.get_images(full=True), 1):
            xref = img[0]
            try:
                pix = pymupdf.Pixmap(self._open(), xref)
                # 处理 CMYK / 灰度 → RGB
                if pix.n - pix.alpha >= 4:
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                out_path = self.image_dir / f"page{page_no:03d}_img{img_index:02d}.png"
                pix.save(str(out_path))
                pix = None
            except Exception:
                continue
            # 用页面坐标近似;若需要更精细可调 ``page.get_image_rects(xref)``
            result.setdefault(0.0, []).append(out_path)
        return result

    # ---------- 组装 Markdown ----------
    # 把"看起来像代码"的单行内容按 ; { } 自动换行,避免代码被挤在一行。
    # 故意不切 '=' / ',' / '()',以免破坏自然语言的句子。
    _CODE_BREAK_RE = re.compile(
        r"(?<![=!<>])"        # 不在 ==/!=/<=/>= 后
        r"(?:;|\}\s*\{|"
        r"^\s*\}|\{\s*$)"
    )

    @classmethod
    def _split_code_line(cls, text: str) -> List[str]:
        """把单行内的多段代码按 ; 和 }{ 拆开;非代码行不动。"""
        # 只在密度够高时才拆:每 50 字符至少 1 个 ; 或 { 或 }
        if len(text) < 60:
            return [text]
        score = sum(text.count(c) for c in "{};") / max(len(text) / 50, 1)
        if score < 0.6:  # 阈值:密度不够就保持原样
            return [text]
        # 在 ; 后(后跟空白 + 标识符)切;在 }{ 之间切;在 } 开头切
        parts: List[str] = []
        buf = ""
        i = 0
        while i < len(text):
            ch = text[i]
            buf += ch
            # ; 后面紧跟空白 + 标识符 → 切
            if ch == ";" and i + 1 < len(text) and text[i + 1] in " \t\xa0":
                # 跳过后续空白
                j = i + 1
                while j < len(text) and text[j] in " \t\xa0":
                    buf += text[j]
                    j += 1
                # 检查下一个非空白是否是标识符起头(字母 / _ / 中文)
                if j < len(text) and (text[j].isalnum() or text[j] == "_" or "\u4e00" <= text[j] <= "\u9fff"):
                    parts.append(buf.rstrip())
                    buf = ""
                    i = j
                    continue
            # } 后跟 { → 切
            if ch == "}" and i + 1 < len(text) and text[i + 1] == "{":
                buf += "{"
                parts.append(buf)
                buf = ""
                i += 2
                continue
            i += 1
        if buf.strip():
            parts.append(buf.rstrip())
        return parts if len(parts) > 1 else [text]

    @classmethod
    def _looks_like_code_paragraph(cls, lines: List[str]) -> bool:
        """
        判断一组连续非空行是否整体像代码块。

        规则(满足其一即可):
            - 行均很短 (< 80 字符) 且 ≥60% 行以 ; { } 结尾,或以空白 + 标识符开头
            - 行均很短且包含 ≥2 个典型代码符号({} ; -> :: ())
        """
        if len(lines) < 3:
            return False
        nonempty = [l for l in lines if l.strip()]
        if len(nonempty) < 3:
            return False
        avg_len = sum(len(l) for l in nonempty) / len(nonempty)
        if avg_len > 90:
            return False
        tail_code = sum(
            1 for l in nonempty
            if l.rstrip().endswith((";", "{", "}", ":", ","))
            or l.lstrip().startswith(("}", ")", "]"))
        )
        sym_count = sum(l.count(";") + l.count("{") + l.count("}") + l.count("->") + l.count("::") for l in nonempty)
        return (tail_code / len(nonempty) >= 0.5) or sym_count >= len(nonempty) * 1.5

    @classmethod
    def _is_dense_code_line(cls, line: str) -> bool:
        """单行是否像代码:长度 + 符号密度 + 模式。"""
        if len(line) < 60:
            return False
        # 符号密度:每 50 字符至少 1 个 ; { } -> ::
        sym = sum(line.count(c) for c in "{};") + line.count("->") + line.count("::")
        # 必须有 ≥2 个 ; 或 1 个 { 或 1 个 },且总符号 / 长度 ≥ 0.015(每 50 字符 ≥ 0.75 个)
        if ";" in line and line.count(";") + line.count("{") + line.count("}") < 2:
            return False
        if sym < 2:
            return False
        density = sym / max(len(line), 1)
        if density < 0.015:
            return False
        # 模式:必须看起来像代码 —— 包含典型代码关键字 / 调用 / 字符串
        code_hint = (
            "->" in line
            or "::" in line
            or "{" in line
            or "}" in line
            or "()" in line
            or re.search(r"\b\w+\(", line) is not None  # 标识符后跟 (
            or re.search(r'"[^"]+"', line) is not None  # 含字符串字面量
        )
        return code_hint and density >= 0.012

    def _assemble_markdown(
        self,
        page: pymupdf.Page,
        lines: List[_LineInfo],
        tables: List[Dict[str, Any]],
        link_map: Dict[Tuple[int, int, int, int], str],
        image_map: Dict[float, List[Path]],
        body_size: float,
        page_no: int,
    ) -> str:
        """
        把 lines + tables + images 拼成 markdown。

        策略:
            1) 把 lines 先按垂直位置切成"段"(连续非空行),逐段判断:
                - code   → 整段包在 ``` 里,内部不合并行
                - single-line code → 按 ; }{ 自动换行,再包在 ``` 里
                - 其它  → 标题 / 列表 / 段落
            2) 把表格 / 图片 bbox 与段垂直位置比较,就近插入。
        """
        # 1) 把表格 / 图片作为插入点准备好
        insert_points: List[Tuple[float, str, Any]] = []
        for tbl in tables:
            insert_points.append((float(tbl["bbox"][3]), "table", tbl))
        for y, paths in image_map.items():
            for p in paths:
                insert_points.append((float(y), "image", p))
        insert_points.sort(key=lambda x: x[0])

        # 2) 把 lines 按 y 切成段(连续非空行 = 一段)
        paragraphs: List[Tuple[float, List[_LineInfo]]] = []
        buf: List[_LineInfo] = []
        buf_y = 0.0
        for line in lines:
            txt = line.text.rstrip()
            if not txt.strip():
                if buf:
                    paragraphs.append((buf_y, buf))
                    buf = []
            else:
                if not buf:
                    buf_y = line.y0
                buf.append(line)
        if buf:
            paragraphs.append((buf_y, buf))

        out: List[str] = []

        def flush_inserts_up_to(y: float) -> None:
            while insert_points and insert_points[0][0] <= y:
                _y, kind, payload = insert_points.pop(0)
                if kind == "table":
                    out.append(self._format_table(payload))
                else:
                    out.append(self._format_image(payload))
                out.append("")

        for p_y, p_lines in paragraphs:
            flush_inserts_up_to(p_y)

            # 取段内所有 span 的文字 + 行内格式
            line_texts = [ln.text.rstrip() for ln in p_lines]
            first_line = p_lines[0]

            # ---- 段级判断:代码块? ----
            # (a) 整段都是正宽字体 → 必为代码
            all_mono = all(ln.all_monospace for ln in p_lines) and len(p_lines) >= 1
            # (b) 整段不长 + 看起来像代码(比例字体的代码)
            looks_code = self._looks_like_code_paragraph(line_texts)

            if all_mono or looks_code:
                out.append("```")
                for t in line_texts:
                    # 如果一行过长,先尝试按 ; }{ 拆
                    split = self._split_code_line(t) if looks_code and not all_mono else [t]
                    for s in split:
                        out.append(s)
                out.append("```")
                out.append("")
                continue

            # ---- 标题 / 列表 / 普通段落:沿用行级判断 ----
            # 注意:即便段被判断为"非代码",行内 mono / 代码行也要单独处理,
            # 否则中文注释会把 4~5 行短代码挤成 1 行。
            in_code_block = False
            for line in p_lines:
                line_text = line.text.rstrip()
                if not line_text.strip():
                    continue

                # 单行代码:等宽字体 或 高密度代码 → 单独成块
                is_mono_line = line.all_monospace and len(line_text) >= 2
                is_dense = self._is_dense_code_line(line_text)
                if is_mono_line or is_dense:
                    if not in_code_block:
                        out.append("```")
                        in_code_block = True
                    # 单行内还能再拆的,按 ; }{ 拆
                    if is_dense and not is_mono_line:
                        for s in self._split_code_line(line_text):
                            out.append(s)
                    else:
                        out.append(line_text)
                    continue
                # 普通行收尾代码块
                if in_code_block:
                    out.append("```")
                    out.append("")
                    in_code_block = False

                # 标题
                heading_level = self._detect_heading_level(line, body_size)
                if heading_level:
                    rendered = self._format_inline(line, link_map=link_map)
                    out.append(f"{'#' * heading_level} {rendered}")
                    out.append("")
                    continue

                # 列表 / 引用
                bullet = self._detect_bullet(line_text)
                if bullet:
                    marker, body_text = bullet
                    new_line = _LineInfo(
                        y0=line.y0, y1=line.y1,
                        spans=[_SpanInfo(
                            text=body_text,
                            font=line.spans[0].font,
                            size=line.spans[0].size,
                            flags=line.spans[0].flags,
                            color=line.spans[0].color,
                            bbox=line.spans[0].bbox,
                        )],
                    )
                    rendered = self._format_inline(new_line, link_map=link_map)
                    out.append(f"{marker.strip()}{rendered}")
                    continue

                # 普通行
                rendered = self._format_inline(line, link_map=link_map)
                # 同段内的行:合并(用空格,而不是换行)
                if out and out[-1] != "" and not out[-1].startswith(
                    ("#", "- ", "* ", "+ ", "> ", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "0.", "```")
                ):
                    out[-1] = out[-1] + " " + rendered
                else:
                    out.append(rendered)

            if in_code_block:
                out.append("```")
                out.append("")

            out.append("")

        # 收尾:把页内剩下的表格 / 图片刷出来
        for _y, kind, payload in insert_points:
            if kind == "table":
                out.append(self._format_table(payload))
            else:
                out.append(self._format_image(payload))
            out.append("")

        # 合并连续空行
        cleaned: List[str] = []
        prev_blank = True
        for ln in out:
            is_blank = not ln.strip()
            if is_blank and prev_blank:
                continue
            cleaned.append(ln)
            prev_blank = is_blank

        return "\n".join(cleaned).strip()

    # ---------- 标题 / 列表识别 ----------
    def _detect_heading_level(self, line: _LineInfo, body_size: float) -> Optional[int]:
        """根据行内字号 / body 的比值反推标题级别。"""
        if not line.spans or body_size <= 0:
            return None
        max_size = line.max_size
        ratio = max_size / body_size
        # 找到第一个 >= ratio 的阈值
        best_level: Optional[int] = None
        best_diff = 0.05
        for thr, level in self.heading_size_ratios.items():
            if ratio + 1e-9 >= thr and thr - ratio > -best_diff:
                # 选最接近且不超过 ratio 的阈值
                pass
            # 简化:取阈值 <= ratio 的最大阈值
        sorted_thr = sorted(self.heading_size_ratios.items(), key=lambda kv: -kv[0])
        for thr, level in sorted_thr:
            if ratio + 0.02 >= thr:
                return level
        return None

    _BULLET_RE = re.compile(
        r"^\s*("
        r"[-*+·•●○] "          # 无序符号
        r"|\d+[\.\)]\s"        # 有序 1. / 1)
        r"|[a-zA-Z][\.\)]\s"   # a. / a)
        r"|>\s"                # 引用
        r")"
    )

    @classmethod
    def _detect_bullet(cls, text: str) -> Optional[Tuple[str, str]]:
        """识别列表 / 引用,返 (marker, body_text)。"""
        m = cls._BULLET_RE.match(text)
        if not m:
            return None
        marker = m.group(1)
        body = text[m.end():]
        return marker, body

    # ---------- 行内格式 ----------
    def _format_inline(
        self,
        line: _LineInfo,
        link_map: Dict[Tuple[int, int, int, int], str],
    ) -> str:
        """把单行的所有 span 拼成 markdown 行内字符串。"""
        out_parts: List[str] = []
        for sp in line.spans:
            t = sp.text
            if not t:
                continue
            # 行内代码(等宽)
            if sp.is_monospace:
                t = f"`{t}`"
            if sp.is_bold and not sp.is_monospace:
                t = f"**{t}**"
            elif sp.is_italic and not sp.is_monospace:
                t = f"*{t}*"
            # 链接
            uri = link_map.get(tuple(int(v) for v in sp.bbox))
            if uri:
                t = f"[{t}]({uri})"
            out_parts.append(t)
        return "".join(out_parts).strip()

    # ---------- 表格 / 图片 ----------
    @staticmethod
    def _format_table(tbl: Dict[str, Any]) -> str:
        headers = tbl.get("headers") or []
        rows = tbl.get("rows") or []
        if not headers and not rows:
            return ""

        def _esc(s: str) -> str:
            return s.replace("|", "\\|").replace("\n", " ")

        lines: List[str] = []
        if headers:
            lines.append("| " + " | ".join(_esc(h) for h in headers) + " |")
            lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        else:
            cols = max(len(r) for r in rows) if rows else 1
            lines.append("|" + "|".join(["---"] * cols) + "|")
        for r in rows:
            lines.append("| " + " | ".join(_esc(c) for c in r) + " |")
        return "\n".join(lines)

    @staticmethod
    def _format_image(path: Path) -> str:
        return f"![{path.name}]({path.as_posix()})"
