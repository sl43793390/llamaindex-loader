"""
pdfParse —— PDF 解析 / 转换工具集。

对外暴露两个高层类:
    - :class:`PDFToMarkdown`  PDF → Markdown (支持标题 / 列表 / 表格 / 代码块 / 图片 / 链接)
    - :class:`PDFToImage`     PDF → 图片    (单页 / 区间 / 全部)

依赖: pymupdf (>= 1.24),可选 Pillow (仅 JPG 输出用到)。

典型用法::

    from pdfParse import PDFToMarkdown, PDFToImage

    # 1) PDF → Markdown
    parser = PDFToMarkdown("report.pdf", extract_images=True)
    md = parser.convert()                       # 整篇 markdown
    parser.save("report.md")                    # 落盘
    parser.close()

    # 2) PDF → 图片
    with PDFToImage("report.pdf", dpi=200, image_format="png") as conv:
        conv.convert_all("./images")            # 全部页
        conv.convert_page(1, "./images/p1.png") # 单页
        conv.convert_range(2, 5, "./images")    # 2..5 页
"""
from .pdf_to_markdown import PDFToMarkdown
from .pdf_to_image import PDFToImage

__all__ = ["PDFToMarkdown", "PDFToImage"]
