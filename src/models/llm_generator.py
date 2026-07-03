"""
llm_generator.py  —  v3.0
Bổ sung:  generate_web_search_answer()  dùng Google Custom Search + Gemini tổng hợp
"""

import os
import time
import re
import asyncio
import aiohttp
import google.generativeai as genai
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# Kiểu alias
RetrievedChunk = Dict[str, Any]
AnswerDict     = Dict[str, Any]
CitationDict   = Dict[str, Any]

# ── API KEY ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("⚠️ CHƯA CÓ API KEY: Vui lòng kiểm tra lại file .env")

genai.configure(api_key=GEMINI_API_KEY)

# ── Cấu hình mô hình ─────────────────────────────────────────────────────────
DEFAULT_LLM_MODEL   = "gemini-2.5-flash-lite"
DEFAULT_EMBED_MODEL = "gemini-embedding-001"
MAX_CONTEXT_CHARS   = 20000

# ── Google Custom Search (dùng cho Web Search) ───────────────────────────────
# Cần set trong .env:
#   GOOGLE_SEARCH_API_KEY=...
#   GOOGLE_SEARCH_CX=...
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY", "")
GOOGLE_SEARCH_CX      = os.getenv("GOOGLE_SEARCH_CX", "")


# ════════════════════════════════════════════════════════════════════════════
# TASK 3.1 — Sinh câu trả lời có trích dẫn (giữ nguyên)
# ════════════════════════════════════════════════════════════════════════════

def generate_cited_answer(
    question: str,
    retrieved_chunks: List[RetrievedChunk],
    model_name: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.0
) -> AnswerDict:
    context = _build_context(retrieved_chunks)
    prompt  = _build_cited_prompt(question, context)

    try:
        model    = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(temperature=temperature)
        )
        raw_answer = response.text
    except Exception as e:
        raw_answer = f"⚠️ Lỗi từ máy chủ AI: {str(e)}"

    citations    = _extract_citations(retrieved_chunks, raw_answer)
    clean_answer = re.sub(r'\s*\[Nguồn \d+\]|\s*\[\d+\]', '', raw_answer)

    # Fallback citation
    if not citations and retrieved_chunks and "⚠️ Lỗi" not in raw_answer:
        best = retrieved_chunks[0]
        meta = best.get('metadata', {})
        content = best.get('content', '')
        citations.append({
            "file_name": meta.get('file_name', 'Tài liệu không tên'),
            "page":      meta.get('page', 1),
            "heading":   meta.get('heading', ""),
            "snippet":   content[:120] + "..." if len(content) > 120 else content
        })

    return {"answer": clean_answer, "citations": citations, "model": model_name}


# ════════════════════════════════════════════════════════════════════════════
# TASK 3.3 — Tóm tắt (giữ nguyên)
# ════════════════════════════════════════════════════════════════════════════

def summarize_doc(
    doc_text: str,
    model_name: str = DEFAULT_LLM_MODEL
) -> AnswerDict:
    """Tóm tắt ngắn gọn nội dung cốt lõi dựa trên toàn bộ văn bản của tài liệu."""
    if not doc_text.strip():
        return {"summary": "⚠️ Tài liệu không có nội dung văn bản để tóm tắt.", "model": model_name}

    prompt = f"""Bạn là chuyên gia phân tích tài liệu cao cấp. Hãy đọc toàn bộ văn bản được cung cấp dưới đây và viết một bản tóm tắt ngắn gọn, súc tích về nội dung cốt lõi của tài liệu này.

---
VĂN BẢN TOÀN VẸN:
{doc_text}
---

YÊU CẦU:
- Bố cục rõ ràng, mạch lạc (sử dụng dấu gạch đầu dòng).
- Tập trung vào mục đích chính, các luận điểm cốt lõi hoặc định nghĩa quan trọng nhất.
- Độ dài khoảng 3-5 câu chất lượng cao.

TÓM TẮT:"""

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(temperature=0.2)
        )
        ans = response.text
    except Exception as e:
        ans = f"⚠️ Lỗi tóm tắt: {str(e)}"

    return {"summary": ans, "citations": [], "model": model_name}


# ════════════════════════════════════════════════════════════════════════════
# TASK 3.3 — Giải thích thuật ngữ (giữ nguyên)
# ════════════════════════════════════════════════════════════════════════════

def explain_term(
    term: str,
    retrieved_chunks: List[RetrievedChunk],
    model_name: str = DEFAULT_LLM_MODEL
) -> AnswerDict:
    context = _build_context(retrieved_chunks)
    prompt  = f"""Hãy giải thích thuật ngữ: "{term}" dựa trên ngữ cảnh dưới đây.

---
CONTEXT:
{context}
---
YÊU CẦU:
- Định nghĩa dễ hiểu. Nêu ví dụ nếu có trong tài liệu.
- Trích dẫn nguồn [Nguồn N].

GIẢI THÍCH:"""

    try:
        model    = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(temperature=0.0)
        )
        raw_answer = response.text
    except Exception as e:
        raw_answer = f"⚠️ Lỗi tra cứu: {str(e)}"

    citations    = _extract_citations(retrieved_chunks, raw_answer)
    clean_answer = re.sub(r'\s*\[Nguồn \d+\]|\s*\[\d+\]', '', raw_answer)

    return {"answer": clean_answer, "citations": citations, "model": model_name}


# ════════════════════════════════════════════════════════════════════════════
# MỚI — Agentic Web Search Answer
# Gọi Google Custom Search API → lấy snippet → Gemini tổng hợp với tài liệu
# ════════════════════════════════════════════════════════════════════════════

async def generate_web_search_answer(
    question: str,
    doc_chunks: List[RetrievedChunk],
    model_name: str = DEFAULT_LLM_MODEL,
    num_web_results: int = 5
) -> dict:
    """
    1. Gọi Google Custom Search lấy top-N kết quả web.
    2. Kết hợp snippet web + context tài liệu thành prompt.
    3. Gemini sinh câu trả lời tổng hợp.

    Returns dict gồm:
        answer       : str
        citations    : list  (từ tài liệu)
        web_results  : list  (url, title, snippet từ web)
        model        : str
    """
    # ── Bước 1: Tìm kiếm web ─────────────────────────────────────────────
    web_results = await _google_custom_search(question, num_web_results)

    # ── Bước 2: Xây dựng context kết hợp ────────────────────────────────
    web_context = _build_web_context(web_results)
    doc_context = _build_context(doc_chunks)

    prompt = f"""Bạn là trợ lý thông minh tổng hợp thông tin từ hai nguồn: tài liệu nội bộ và web.

═══ NGỮ CẢNH TỪ TÀI LIỆU ═══
{doc_context if doc_context else "(Không có tài liệu liên quan)"}

═══ KẾT QUẢ TỪ WEB ═══
{web_context if web_context else "(Không có kết quả web)"}

═══ CÂU HỎI ═══
{question}

═══ YÊU CẦU ═══
- Ưu tiên tài liệu nội bộ nếu có thông tin liên quan.
- Bổ sung từ web nếu tài liệu không đủ.
- Nêu rõ thông tin nào từ tài liệu, thông tin nào từ web.
- Trích dẫn nguồn tài liệu bằng [Nguồn N] nếu dùng.
- Trả lời bằng tiếng Việt, súc tích và chính xác.

TRẢ LỜI:"""

    # ── Bước 3: Gọi Gemini ───────────────────────────────────────────────
    try:
        model    = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(temperature=0.3)
        )
        raw_answer = response.text
    except Exception as e:
        raw_answer = f"⚠️ Lỗi AI: {str(e)}"

    citations    = _extract_citations(doc_chunks, raw_answer)
    clean_answer = re.sub(r'\s*\[Nguồn \d+\]|\s*\[\d+\]', '', raw_answer)

    return {
        "answer":      clean_answer,
        "citations":   citations,
        "web_results": web_results,
        "model":       model_name
    }


async def _google_custom_search(query: str, num: int = 5) -> List[dict]:
    """Gọi Google Custom Search JSON API và trả về list {title, url, snippet}."""
    if not GOOGLE_SEARCH_API_KEY or not GOOGLE_SEARCH_CX:
        # Nếu chưa cấu hình key → trả danh sách rỗng (không crash)
        return []

    url    = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_SEARCH_API_KEY,
        "cx":  GOOGLE_SEARCH_CX,
        "q":   query,
        "num": min(num, 10),
        "hl":  "vi",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.status != 200:
                    return []
                data  = await resp.json()
                items = data.get("items", [])
                return [
                    {
                        "title":   item.get("title", ""),
                        "url":     item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                    }
                    for item in items
                ]
    except Exception:
        return []


def _build_web_context(web_results: List[dict]) -> str:
    if not web_results:
        return ""
    parts = []
    for i, r in enumerate(web_results, 1):
        parts.append(
            f"[Web {i}] {r.get('title','')}\n"
            f"URL: {r.get('url','')}\n"
            f"Tóm tắt: {r.get('snippet','')}\n"
        )
    return "\n".join(parts)


# ════════════════════════════════════════════════════════════════════════════
# GENERATE EMBEDDINGS (giữ nguyên)
# ════════════════════════════════════════════════════════════════════════════

def get_embedding(texts: list, model_name: str = DEFAULT_EMBED_MODEL) -> list:
    all_embeddings = []
    batch_size     = 90

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            result = genai.embed_content(
                model=model_name,
                content=batch,
                task_type="retrieval_document"
            )
            all_embeddings.extend(result['embedding'])
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                print(f"⏳ Quá tải API, đang đợi 60s...")
                time.sleep(60)
                result = genai.embed_content(
                    model=model_name,
                    content=batch,
                    task_type="retrieval_document"
                )
                all_embeddings.extend(result['embedding'])
            else:
                raise e
    return all_embeddings


# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS (giữ nguyên)
# ════════════════════════════════════════════════════════════════════════════

def _build_context(
    retrieved_chunks: List[RetrievedChunk],
    max_chars: int = MAX_CONTEXT_CHARS
) -> str:
    context_parts  = []
    current_length = 0

    for idx, chunk in enumerate(retrieved_chunks):
        source_label = f"[Nguồn {idx + 1}]"
        meta         = chunk.get('metadata', {})
        file_name    = meta.get('file_name', 'Unknown')
        page_num     = meta.get('page', 1)
        content      = chunk.get('content', '')

        chunk_text  = f"--- Khối Ngữ Cảnh {source_label} ---\n"
        chunk_text += f"Nội dung: {content}\n"
        chunk_text += f"Nguồn file: {file_name}, Trang: {page_num}\n\n"

        if current_length + len(chunk_text) > max_chars:
            break

        context_parts.append(chunk_text)
        current_length += len(chunk_text)

    return "".join(context_parts)


def _build_cited_prompt(question: str, context: str) -> str:
    return f"""Bạn là một trợ lý hỏi đáp tài liệu học tập trung thực.
Nhiệm vụ: Hãy trả lời câu hỏi dựa TRÊN ĐÚNG THÔNG TIN TRONG PHẦN NGỮ CẢNH.

---
HƯỚNG DẪN QUAN TRỌNG:
1. TUYỆT ĐỐI KHÔNG dùng kiến thức bên ngoài. Nếu không có thông tin, hãy nói: "Tôi không biết câu trả lời dựa trên tài liệu đã cung cấp."
2. Chèn nhãn trích dẫn [Nguồn X] ở cuối mỗi ý.

NGỮ CẢNH:
{context}

CÂU HỎI:
{question}

TRẢ LỜI:"""


def _extract_citations(retrieved_chunks: List[RetrievedChunk], raw_answer: str) -> List[CitationDict]:
    citations    = []
    seen_indices = []

    for idx in range(len(retrieved_chunks)):
        source_number = idx + 1
        if f"[Nguồn {source_number}]" in raw_answer or f"[{source_number}]" in raw_answer:
            if idx not in seen_indices:
                seen_indices.append(idx)

    for i in seen_indices:
        chunk   = retrieved_chunks[i]
        content = chunk.get('content', '')
        snippet = content[:100] + "..." if len(content) > 100 else content
        meta    = chunk.get('metadata', {})
        citations.append({
            "file_name": meta.get('file_name', 'Unknown'),
            "page":      meta.get('page', 1),
            "heading":   meta.get('heading', ""),
            "snippet":   snippet
        })

    return citations
