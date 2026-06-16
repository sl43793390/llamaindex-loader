"""
PDF → 图片 转换器。

使用 PyMuPDF 的 ``Page.get_pixmap(dpi=...)`` 把每一页栅格化成
PNG / JPG / JPEG / BMP / TIFF / WebP 等 Pillow 支持的格式。

特点:
    - 支持单页 / 连续范围 / 全部页
    - 支持自定义 DPI、缩放矩阵、裁剪矩形、alpha 通道
    - 单页 / 全量都返 :class:`Path` 列表,方便后续 OCR / 上传
    - 自动建目录 / 编号 / 处理 CMYK 颜色空间

依赖:
    - pymupdf
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Union

import pymupdf


_SUPPORTED_FORMATS = {"png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp", "pam", "pnm"}


class PDFToImage:
    """
    PDF → 图片 转换器。

    用法::

        c = PDFToImage("doc.pdf", dpi=200, image_format="png")
        paths = c.convert_all("./out")           # 输出 doc_page_001.png ...
        one = c.convert_page(1, "./out/p1.png")  # 单独导出第 1 页
        some = c.convert_range(2, 5, "./out")    # 2~5 页

    Args:
        file_path: 待转换的 PDF。
        dpi: 渲染分辨率,默认 200(印刷质量)。一般 150~300。
        image_format: 输出图片格式,支持 ``png`` / ``jpg`` / ``jpeg`` /
            ``bmp`` / ``tiff`` / ``webp``。默认 ``"png"``。
        alpha: 是否保留透明通道(默认 False,白底)。
        clip: 裁剪框 ``(x0, y0, x1, y1)``,``None`` = 整页。
        jpg_quality: JPG 质量 0-100,只在 ``image_format in {"jpg","jpeg"}`` 时生效。

    Note:
        - 单文件内存峰值 ≈ 单页栅格化后的 Pixmap,大文档也不会爆内存。
        - 扫描件 / 图 PDF 也能正常转出图片。
        - 想做 OCR 时,直接拿返回的 ``Path`` 列表喂给 PaddleOCR / Tesseract。
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        dpi: int = 200,
        image_format: str = "png",
        alpha: bool = False,
        clip: Optional[Iterable[float]] = None,
        jpg_quality: int = 90,
    ) -> None:
        self.file_path = Path(file_path)
        if not self.file_path.is_file():
            raise FileNotFoundError(f"PDF 文件不存在: {self.file_path}")

        fmt = image_format.lower().lstrip(".")
        if fmt not in _SUPPORTED_FORMATS:
            raise ValueError(
                f"不支持的 image_format={image_format!r},"
                f"支持: {sorted(_SUPPORTED_FORMATS)}"
            )
        self.dpi = int(dpi)
        self.image_format = fmt
        self.alpha = bool(alpha)
        self.clip = tuple(clip) if clip is not None else None
        self.jpg_quality = int(jpg_quality)

        # 延迟打开
        self._doc: Optional[pymupdf.Document] = None

    # ---------- 上下文协议 ----------
    def __enter__(self) -> "PDFToImage":
        self._open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._doc is not None:
            self._doc.close()
            self._doc = None

    def _open(self) -> pymupdf.Document:
        if self._doc is None:
            self._doc = pymupdf.open(str(self.file_path))
        return self._doc

    @property
    def page_count(self) -> int:
        return self._open().page_count

    # ---------- 公开 API ----------
    def convert_all(
        self,
        output_dir: Union[str, Path],
        name_template: Optional[str] = None,
    ) -> List[Path]:
        """
        把整篇 PDF 转成图片,写入 ``output_dir``。

        Args:
            output_dir: 输出目录,不存在会自动创建。
            name_template: 命名模板,支持 ``{stem}``(PDF 文件名)、
                ``{page}``(页号 0 补齐) 和 ``{fmt}``(格式)。
                默认 ``"{stem}_page_{page:03d}.{fmt}"``。

        Returns:
            生成的 :class:`Path` 列表(按页号升序)。
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        template = name_template or "{stem}_page_{page:03d}.{fmt}"
        doc = self._open()
        out_paths: List[Path] = []
        for i, page in enumerate(doc, 1):
            out_path = output_dir / template.format(
                stem=self.file_path.stem,
                page=i,
                fmt=self.image_format,
            )
            self._render_page(page, out_path)
            out_paths.append(out_path)
        return out_paths

    def convert_page(
        self,
        page_no: int,
        output_path: Union[str, Path],
    ) -> Path:
        """
        导出第 ``page_no`` 页(1-based)。

        Args:
            page_no: 页号,从 1 开始。
            output_path: 输出图片路径,父目录不存在会自动创建。

        Returns:
            写入后的 :class:`Path`。
        """
        doc = self._open()
        if not (1 <= page_no <= doc.page_count):
            raise IndexError(
                f"页号 {page_no} 越界,有效范围 1..{doc.page_count}"
            )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._render_page(doc[page_no - 1], output_path)
        return output_path

    def convert_range(
        self,
        start: int,
        end: int,
        output_dir: Union[str, Path],
        name_template: Optional[str] = None,
    ) -> List[Path]:
        """
        导出 ``start..end`` 闭区间的页(均 1-based)。

        Args:
            start: 起始页号(1-based,含)。
            end: 结束页号(1-based,含)。
            output_dir: 输出目录。
            name_template: 同 :meth:`convert_all`。

        Returns:
            生成的 :class:`Path` 列表(按页号升序)。
        """
        doc = self._open()
        if start < 1 or end > doc.page_count or start > end:
            raise IndexError(
                f"页范围 [{start}, {end}] 越界,有效范围 1..{doc.page_count}"
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        template = name_template or "{stem}_page_{page:03d}.{fmt}"

        out_paths: List[Path] = []
        for i in range(start, end + 1):
            out_path = output_dir / template.format(
                stem=self.file_path.stem,
                page=i,
                fmt=self.image_format,
            )
            self._render_page(doc[i - 1], out_path)
            out_paths.append(out_path)
        return out_paths

    # ---------- 内部 ----------
    def _render_page(self, page: pymupdf.Page, out_path: Path) -> None:
        """栅格化单页并保存为指定格式。"""
        # DPI → 缩放矩阵(72 是 PDF 的默认单位:1 inch = 72 points)
        zoom = max(self.dpi, 1) / 72.0
        mat = pymupdf.Matrix(zoom, zoom)

        pix = page.get_pixmap(matrix=mat, alpha=self.alpha, clip=self.clip)

        # Pillow 走 JPG 质量;其它格式 PyMuPDF 自己 save() 即可
        if self.image_format in {"jpg", "jpeg"}:
            # PyMuPDF Pixmap.save() 对 jpg 不直接接受 quality,统一过 Pillow
            self._save_via_pillow(pix, out_path, fmt="JPEG", quality=self.jpg_quality)
        else:
            pix.save(str(out_path))

        pix = None  # 显式释放

    @staticmethod
    def _save_via_pillow(
        pix: pymupdf.Pixmap,
        out_path: Path,
        fmt: str,
        quality: int,
    ) -> None:
        """把 Pixmap 转成 Pillow Image 后保存(JPG 用)。"""
        try:
            from PIL import Image
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "保存 JPG 需要 Pillow,请先安装: pip install Pillow"
            ) from e

        # pymupdf 1.24+ 提供 tobytes("png") / 直接取 samples
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        if mode == "RGBA":
            # JPG 不支持 alpha,合成到白底
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        img.save(str(out_path), format=fmt, quality=quality)
