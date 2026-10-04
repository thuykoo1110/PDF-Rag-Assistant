import os
import re
import pymupdf 
import pdfplumber

from collections import Counter
from typing import List, Dict, Any, Tuple

ChunkDict = Dict[str, Any]
BBox = Tuple[float, float, float, float]  # (x0, y0, x1, y1), gốc tọa độ ở góc trên trái

MAX_HEADING_LEN = 120 

# =============================================================================
# TASK 1.1 + 1.2 + 1.3 — HÀM CHÍNH (Pipeline hoàn chỉnh)
# =============================================================================

def process_pdf(file_path: str, workspace_name: str) -> List[ChunkDict]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'không thấy file: {file_path}')
    if not workspace_name:
        raise ValueError("workspace_name empty")
    table_chunk, table_bboxes = _extract_table_chunks(file_path, workspace_name)
    text_chunk = _extract_text_chunks(file_path, workspace_name, table_bboxes)

    sum_chunk = text_chunk + table_chunk
    sum_chunk.sort(key= lambda x: x["metadata"]["page"])  # xếp theo trang

    return sum_chunk


def _body_font_size(pages_blocks: List[List[dict]]) -> float:

    counter = Counter()
    for blocks in pages_blocks:
        for block in blocks:
            if block.get("type") != 0:  # chỉ lấy text block
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    n_chars = len(span.get("text", "").strip())
                    if n_chars:
                        counter[round(span.get("size", 0), 1)] += n_chars
    return counter.most_common(1)[0][0] if counter else 12.0


def _join_lines(lines: List[str]) -> str:
    out = ""
    for raw in lines:
        line = raw.replace("­", "").strip()  # bỏ soft hyphen, khoảng trắng đầu/cuối
        if not line:
            continue
        if not out:
            out = line
        elif out.endswith("-") and len(out) > 1 and out[-2].isalpha() and line[0].islower():
            out = out[:-1] + line
        else:
            out += " " + line
    return out


def _overlaps(bbox: BBox, regions: List[BBox], thr: float = 0.5) -> bool:
    area = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1e-6)
    for r in regions:
        ix = max(0.0, min(bbox[2], r[2]) - max(bbox[0], r[0]))
        iy = max(0.0, min(bbox[3], r[3]) - max(bbox[1], r[1]))
        if ix * iy / area > thr:
            return True
    return False


def _extract_text_chunks(
    file_path: str,
    workspace_name: str,
    table_bboxes: Dict[int, List[BBox]] = None,
) -> List[ChunkDict]:
    """
    Dùng PyMuPDF đọc text, nhận diện block Heading/Paragraph,
    rồi cắt thành chunk có nghĩa và gắn metadata.

    Args:
        file_path      (str): Đường dẫn file PDF
        workspace_name (str): Tên workspace
        table_bboxes   (dict): {số trang (từ 1): [bbox bảng, ...]} từ _extract_table_chunks;
                               các block nằm trong bảng sẽ bị bỏ qua vì đã có chunk "table".

    Returns:
        List[ChunkDict]: Các chunk loại "text"
    """
    chunks = []
    filename = os.path.basename(file_path)
    table_bboxes = table_bboxes or {}

    with pymupdf.open(file_path) as document:
        cur_heading = ''
        cur_chunk_text = ''
        page_start = 1

        # Đọc block của mọi trang một lần, rồi tính cỡ chữ thân bài của cả tài liệu
        pages_blocks = [page.get_text('dict').get('blocks', []) for page in document]
        body_font_size = _body_font_size(pages_blocks)

        for page_index, blocks in enumerate(pages_blocks):
            page_tables = table_bboxes.get(page_index + 1, [])

            # phân loại block
            for block in blocks:
                # bỏ qua block nằm trong bảng 
                if page_tables and _overlaps(block.get('bbox', (0, 0, 0, 0)), page_tables):
                    continue

                line_texts = []
                max_font_block = 0.0

                for line in block.get('lines', []):
                    line_text = ""
                    for span in line.get('spans', []):
                        line_text += span.get('text', "")
                        if span.get('size', 0) > max_font_block:
                            max_font_block = span.get('size', 0)
                    line_texts.append(line_text)

                # nối các dòng bằng khoảng trắng (không để chữ dính nhau ở chỗ xuống dòng)
                block_text = _join_lines(line_texts).strip()

                if not block_text: continue

                # heading
                if _is_heading(block_text, max_font_block, body_font_size):
                    # đẩy chunk cũ list chunk
                    if cur_chunk_text:
                        chunks.append({
                            'content': cur_chunk_text.strip(),
                            'metadata':{
                                "file_name": filename,
                                "page": page_start,
                                "workspace_name": workspace_name,
                                "chunk_type": "text",
                                "heading": cur_heading
                            }
                        })

                        # update current chunk text
                        cur_chunk_text = ''
                    cur_heading = block_text
                    page_start = page_index + 1
                # paragraph
                else:
                    if not cur_chunk_text:  page_start = page_index + 1
                    cur_chunk_text += block_text + '\n'

                    # limit chunk size 1000
                    if len(cur_chunk_text) > 5000:
                        chunks.append({
                            'content': cur_chunk_text.strip(),
                            'metadata':{
                                "file_name": filename,
                                "page": page_start,
                                "workspace_name": workspace_name,
                                "chunk_type": "text",
                                "heading": cur_heading
                            }
                        })
                        cur_chunk_text = ""
        # last chunk
        if cur_chunk_text.strip():
            chunks.append({
                'content': cur_chunk_text.strip(),
                'metadata':{
                    "file_name": filename,
                    "page": page_start,
                    "workspace_name": workspace_name,
                    "chunk_type": "text",
                    "heading": cur_heading
                }
            })
    return chunks



# =============================================================================
# TASK 1.1 — HELPER: Extract bảng biểu với pdfplumber
# =============================================================================

def _extract_table_chunks(
    file_path: str, workspace_name: str
) -> Tuple[List[ChunkDict], Dict[int, List[BBox]]]:
    """
    Trích bảng bằng pdfplumber.

    Returns:
        (chunks, table_bboxes): chunk loại "table" và bbox của từng bảng theo trang
        ({số trang từ 1: [bbox, ...]}) để phần text bỏ qua vùng bảng.
    """
    chunks = []
    table_bboxes: Dict[int, List[BBox]] = {}
    filename = os.path.basename(file_path)

    with pdfplumber.open(file_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            # find_tables() cho cả bbox lẫn nội dung (cùng cách phát hiện với extract_tables)
            for found in page.find_tables():
                table = found.extract()
                if not table: continue
                table_bboxes.setdefault(page_idx, []).append(tuple(found.bbox))

                markdow_table = ""
                for i, row in enumerate(table):
                    row_cleaning = [str(cell).replace("\n", "") if cell is not None else "" for cell in row] 
                    markdow_table += "| " + " | ".join(row_cleaning) + " |\n"

                    if i ==0:
                        markdow_table += "| " + " | ".join(["---"] * len(row_cleaning)) + " |\n"
                        
                if markdow_table.strip():
                    chunks.append({
                        "content": markdow_table.strip(),
                        "metadata": {
                            "file_name": filename,
                            "page": page_idx,
                            "workspace_name": workspace_name,
                            "chunk_type": "table",
                            "heading": "bảng"
                        }
                    })
    return chunks, table_bboxes

# =============================================================================
# TASK 1.2 — HELPER: Nhận diện Heading
# =============================================================================

def _is_heading(text: str, font_size: float, body_font_size: float) -> bool:
    text = text.strip()
    if not text:  return False

    if len(text) > MAX_HEADING_LEN:  return False

    if font_size >= body_font_size * 1.25: return True
    
    pattern = r"(?i)^(" \
              r"chương\s+\d+|phần\s+[IVXLCDM]+|bài\s+\d+|mục\s+\d+|" \
              r"\d+(\.\d+)+|" \
              r"\d+[\.\)]|" \
              r"[IVX]+[\.\)]|" \
              r"[A-Z][\.\)]|" \
              r"(mục lục|lời mở đầu|mở đầu|kết luận|tài liệu tham khảo|phụ lục)" \
              r")\b"
    if re.match(pattern, text): return True

    if text.isupper():  return True # tiêu đề thừn in hoa

    return False


# =============================================================================
# BACKWARD COMPATIBILITY — Giữ lại hàm cũ để không break code hiện tại
# =============================================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    [DEPRECATED] Hàm cũ từ pdf_processor.py — giữ lại để không break code.
    Nên dùng process_pdf() thay thế.
    """
    import pypdf
    reader = pypdf.PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> list:
    """
    [DEPRECATED] Hàm cắt chunk cũ theo ký tự — giữ lại để không break code.
    Nên dùng process_pdf() thay thế.
    """
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 1 <= size:
            cur += p + "\n"
        else:
            if cur:
                chunks.append(cur.strip())
            cur = (cur[-overlap:] + p + "\n") if overlap else (p + "\n")
    if cur.strip():
        chunks.append(cur.strip())
    return chunks
