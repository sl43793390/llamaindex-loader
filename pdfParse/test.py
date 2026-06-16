from pathlib import Path
from pdf_to_markdown import PDFToMarkdown
from pdf_to_image import PDFToImage

# 用 __file__ 锚定到脚本所在目录,
# 这样从任何工作目录跑 python pdfParse/test.py 都能找到文件。
HERE = Path(__file__).resolve().parent
PDF = HERE / "UFX开发指南新.pdf"

if not PDF.is_file():
    raise FileNotFoundError(f"测试 PDF 不存在: {PDF}")

# PDF → Markdown
# with PDFToMarkdown(str(PDF), extract_images=False) as p:
#     md = p.convert()
#     p.save(HERE / "report.md")

# PDF → 图片
with PDFToImage(str(PDF), dpi=300, image_format="png") as conv:
    conv.convert_all(str(HERE / "images"))            # 全部
    # conv.convert_page(1, str(HERE / "images" / "p1.png"))  # 第 1 页
    # conv.convert_range(2, 5, str(HERE / "images"))         # 第 2~5 页