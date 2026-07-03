"""
=============================================================================
MODULE: retriever.py
TASKS : 2.1 Nâng cấp Vector DB sang PersistentClient
        2.2 Xây dựng Workspaces (lọc theo metadata)
        2.3 Thuật toán Cross Analysis (nhiều file cùng workspace)
        2.4 [Optional] Hybrid Search (BM25 + Vector)
=============================================================================

API CONTRACT - Các hàm được định nghĩa ở đây là "giao kèo" với các module khác.
Tên hàm, tham số, và kiểu trả về KHÔNG ĐƯỢC THAY ĐỔI.
Chỉ viết code bên trong phần thân hàm.

Kiểu dữ liệu đầu vào (ChunkDict) — nhận từ parser.py của Thùy:
{
    "content" : str,
    "metadata": {
        "file_name"      : str,
        "page"           : int,
        "workspace_name" : str,
        "chunk_type"     : str,   # "text" | "table" | "heading"
        "heading"        : str,
    }
}

Kiểu dữ liệu đầu ra (RetrievedChunk) — trả cho llm_generator.py của Tiến:
{
    "content" : str,
    "metadata": {
        "file_name"      : str,
        "page"           : int,
        "workspace_name" : str,
        "chunk_type"     : str,
        "heading"        : str,
    },
    "score"   : float    # Điểm similarity (0.0 - 1.0), cao hơn = liên quan hơn
}
"""

from typing import List, Dict, Any, Optional

# Kiểu alias
ChunkDict     = Dict[str, Any]
RetrievedChunk = Dict[str, Any]

# Đường dẫn lưu ChromaDB xuống ổ cứng (thay đổi nếu cần)
CHROMA_PERSIST_DIR = "./chroma_db"

# --- Singleton ChromaDB client ---
_chroma_client = None


def _get_client():
    """Singleton ChromaDB client —  tạo 1 lần, dùng mãi."""
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _chroma_client


def _get_collection(workspace_name: str):
    """
    Lấy collection đã tồn tại (read-only, KHÔNG tạo mới).
    Dùng cho các hàm đọc: search, list_files, hybrid_search.

    Raises:
        ValueError: Nếu workspace không tồn tại
    """
    client = _get_client()
    try:
        return client.get_collection(name=workspace_name)
    except Exception:
        raise ValueError(f"Workspace '{workspace_name}' không tồn tại.")

# =============================================================================
# TASK 2.2 + 2.3 — HÀM CHÍNH (được gọi bởi FastAPI endpoint /api/chat)
# =============================================================================

def search_workspace(
    query: str,
    workspace_name: str,
    top_k: int = 5
) -> List[RetrievedChunk]:
    """
    Tìm kiếm top-k chunks liên quan nhất trong một workspace cụ thể.
    Tự động quét nhiều file khác nhau trong cùng workspace (Cross Analysis).

    Args:
        query          (str): Câu hỏi của người dùng
        workspace_name (str): Tên workspace cần tìm kiếm trong đó
        top_k          (int): Số lượng chunk trả về (default: 5)

    Returns:
        List[RetrievedChunk]: Danh sách chunks sắp xếp theo độ liên quan giảm dần.
                              Mỗi phần tử chứa "content", "metadata", "score".

    Raises:
        ValueError: Nếu workspace_name không tồn tại trong DB

    Example:
        >>> results = search_workspace("Định nghĩa giới hạn là gì?", "Giải Tích", top_k=4)
        >>> for r in results:
        ...     print(r["metadata"]["file_name"], r["metadata"]["page"], r["score"])
        giai_tich_chuong1.pdf  5  0.92
        giai_tich_chuong2.pdf  12 0.87
    """
    if top_k <= 0:
        return []

    from src.models.llm_generator import get_embedding

    collection = _get_collection(workspace_name)

    count = collection.count()
    if count == 0:
        raise ValueError(
            f"Workspace '{workspace_name}' không có dữ liệu. "
            f"Hãy upload PDF trước."
        )

    # Clamp n_results để không vượt quá số documents trong collection
    k = min(top_k, count)

    result = collection.query(
        query_embeddings=get_embedding([query]),
        n_results=k,
        where={"workspace_name": workspace_name},
        include=["documents", "metadatas", "distances"],
    )

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    chunks = [
        {
            "content": document,
            "metadata": metadata or {},
            "score": 1.0 / (1.0 + max(float(distances[index]), 0.0))
            if index < len(distances) and distances[index] is not None
            else 0.0,
        }
        for index, (document, metadata) in enumerate(zip(documents, metadatas))
    ]

    # Sắp xếp theo score giảm dần (cao nhất = liên quan nhất)
    chunks.sort(key=lambda c: c["score"], reverse=True)
    return chunks


# =============================================================================
# TASK 2.1 — Quản lý kết nối ChromaDB Persistent
# =============================================================================

def get_or_create_collection(workspace_name: str):
    """
    Lấy (hoặc tạo mới) một ChromaDB collection tương ứng với workspace.
    Sử dụng PersistentClient để dữ liệu được lưu xuống ổ cứng.

    Lưu ý: workspace_name phải là ASCII hợp lệ cho ChromaDB
           (3-63 ký tự, chỉ chứa [a-zA-Z0-9_-]).

    Args:
        workspace_name (str): Tên workspace (dùng làm tên collection)

    Returns:
        chromadb.Collection: Collection object của ChromaDB
    """
    client = _get_client()
    return client.get_or_create_collection(name=workspace_name)


def add_chunks_to_workspace(workspace_name: str, chunks: List[ChunkDict]) -> None:
    """
    Vector hóa và lưu danh sách chunks vào workspace (ChromaDB collection).
    Được gọi bởi FastAPI endpoint POST /api/upload.

    Args:
        workspace_name (str)         : Tên workspace đích
        chunks         (List[ChunkDict]): Danh sách chunks từ parser.process_pdf()

    Returns:
        None

    Raises:
        ValueError: Nếu chunks rỗng
    """
    if not chunks:
        raise ValueError("chunks must not be empty")

    from uuid import uuid4
    from src.models.llm_generator import get_embedding

    collection = get_or_create_collection(workspace_name)
    texts = [chunk["content"] for chunk in chunks]
    # Đảm bảo metadata.workspace_name khớp với tham số workspace_name
    metadatas = []
    for chunk in chunks:
        meta = dict(chunk["metadata"])
        meta["workspace_name"] = workspace_name
        metadatas.append(meta)
    embeddings = get_embedding(texts)
    batch_id = uuid4().hex

    collection.add(
        ids=[f"{workspace_name}:{batch_id}:{i}" for i in range(len(chunks))],
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def delete_workspace(workspace_name: str) -> None:
    """
    Xóa toàn bộ dữ liệu của một workspace khỏi ChromaDB.

    Args:
        workspace_name (str): Tên workspace cần xóa

    Returns:
        None
    """
    client = _get_client()
    try:
        client.delete_collection(name=workspace_name)
    except Exception:
        pass  # Workspace không tồn tại, bỏ qua


def list_workspaces() -> List[str]:
    """
    Trả về danh sách tên tất cả workspace hiện có trong ChromaDB.

    Returns:
        List[str]: Ví dụ ["Giải Tích", "CTDL", "Toán Rời Rạc"]
    """
    client = _get_client()

    collections = client.list_collections()
    result = []
    for col in collections:
        col_name = col.name if hasattr(col, "name") else str(col)
        result.append(col_name)
    return result


def list_files_in_workspace(workspace_name: str) -> List[str]:
    """
    Trả về danh sách tên file PDF trong một workspace.

    Args:
        workspace_name (str): Tên workspace

    Returns:
        List[str]: Ví dụ ["giai_tich_c1.pdf", "giai_tich_c2.pdf"]
    """
    try:
        collection = _get_collection(workspace_name)
    except ValueError:
        return []  # Workspace không tồn tại → không có file
    result = collection.get(include=["metadatas"])

    file_names = {
        metadata["file_name"]
        for metadata in result.get("metadatas", [])
        if metadata and metadata.get("file_name")
    }
    return sorted(file_names)


# =============================================================================
# TASK 2.4 [OPTIONAL] — Hybrid Search
# =============================================================================

def hybrid_search(
    query: str,
    workspace_name: str,
    top_k: int = 5,
    alpha: float = 0.7
) -> List[RetrievedChunk]:
    """
    [Tùy chọn nâng cao] Kết hợp Vector Search và BM25 keyword search,
    dùng Reciprocal Rank Fusion (RRF) để rerank kết quả.

    Args:
        query          (str)  : Câu hỏi của người dùng
        workspace_name (str)  : Tên workspace
        top_k          (int)  : Số chunk trả về
        alpha          (float): Trọng số của vector search (0.0-1.0).
                                1-alpha là trọng số của BM25.

    Returns:
        List[RetrievedChunk]: Chunks đã được rerank
    """
    if top_k <= 0:
        return []
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0")

    import math
    import re
    from collections import Counter, defaultdict
    from src.models.llm_generator import get_embedding

    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", (text or "").lower())

    def _result_key(chunk: RetrievedChunk) -> tuple:
        metadata = chunk.get("metadata") or {}
        return (
            chunk.get("content", ""),
            metadata.get("file_name"),
            metadata.get("page"),
            metadata.get("chunk_type"),
            metadata.get("heading"),
        )

    def _to_score(distance: float) -> float:
        if distance is None:
            return 0.0
        return 1.0 / (1.0 + max(float(distance), 0.0))

    collection = _get_collection(workspace_name)
    stored = collection.get(include=["documents", "metadatas"])
    documents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or [{} for _ in documents]

    if not documents:
        return []

    # Lấy dư ứng viên để có đủ dữ liệu cho bước rerank.
    vector_limit = min(len(documents), max(top_k * 3, top_k))
    vector_chunks: List[RetrievedChunk] = []

    # Vector search: tìm các chunk gần câu hỏi nhất theo ngữ nghĩa.
    if alpha > 0.0:
        query_embedding = get_embedding([query])
        vector_result = collection.query(
            query_embeddings=query_embedding,
            n_results=vector_limit,
            include=["documents", "metadatas", "distances"],
        )
        result_docs = (vector_result.get("documents") or [[]])[0]
        result_metadatas = (vector_result.get("metadatas") or [[]])[0]
        result_distances = (vector_result.get("distances") or [[]])[0]
        vector_chunks = [
            {
                "content": doc,
                "metadata": metadata or {},
                "score": _to_score(
                    result_distances[index] if index < len(result_distances) else None
                ),
            }
            for index, (doc, metadata) in enumerate(zip(result_docs, result_metadatas))
        ]

    bm25_chunks: List[RetrievedChunk] = []
    query_terms = _tokenize(query)

    # BM25: ưu tiên các chunk chứa đúng từ khóa trong câu hỏi.
    if alpha < 1.0 and query_terms:
        tokenized_docs = [_tokenize(doc) for doc in documents]
        doc_freq = Counter()
        for tokens in tokenized_docs:
            doc_freq.update(set(tokens))

        doc_count = len(tokenized_docs)
        avg_doc_len = sum(len(tokens) for tokens in tokenized_docs) / doc_count
        avg_doc_len = avg_doc_len or 1.0
        query_counts = Counter(query_terms)
        k1 = 1.5
        b = 0.75
        scored_docs = []

        for index, tokens in enumerate(tokenized_docs):
            if not tokens:
                continue
            term_freq = Counter(tokens)
            doc_len = len(tokens)
            score = 0.0
            for term, query_count in query_counts.items():
                frequency = term_freq.get(term, 0)
                if frequency == 0:
                    continue
                idf = math.log(1.0 + (doc_count - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
                denominator = frequency + k1 * (1.0 - b + b * doc_len / avg_doc_len)
                score += query_count * idf * (frequency * (k1 + 1.0) / denominator)
            if score > 0.0:
                scored_docs.append((score, index))

        max_bm25 = max((score for score, _ in scored_docs), default=1.0)
        for score, index in sorted(scored_docs, reverse=True)[:vector_limit]:
            bm25_chunks.append({
                "content": documents[index],
                "metadata": metadatas[index] or {},
                "score": score / max_bm25,
            })

    if alpha == 1.0:
        return vector_chunks[:top_k]
    if alpha == 0.0:
        return bm25_chunks[:top_k]

    chunks_by_key: Dict[tuple, RetrievedChunk] = {}
    fused_scores = defaultdict(float)
    rrf_k = 60.0

    # RRF gộp thứ hạng từ vector search và BM25, tránh trùng chunk.
    for rank, chunk in enumerate(vector_chunks, start=1):
        key = _result_key(chunk)
        chunks_by_key[key] = chunk
        fused_scores[key] += alpha / (rrf_k + rank)

    for rank, chunk in enumerate(bm25_chunks, start=1):
        key = _result_key(chunk)
        chunks_by_key.setdefault(key, chunk)
        fused_scores[key] += (1.0 - alpha) / (rrf_k + rank)

    reranked = []
    for key, score in fused_scores.items():
        chunk = dict(chunks_by_key[key])
        chunk["score"] = min(score * (rrf_k + 1.0), 1.0)
        reranked.append(chunk)

    return sorted(reranked, key=lambda chunk: chunk["score"], reverse=True)[:top_k]

# =============================================================================
# Helper Function to summarize_doc
# =============================================================================
def get_full_text_by_file(file_name: str, workspace_name: str) -> str:
    """
    [MỚI] Lấy toàn bộ văn bản thuần của một file trong workspace,
    phục vụ cho tính năng tóm tắt tài liệu bằng LLM Context lớn (Gemini).
    """
    try:
        collection = _get_collection(workspace_name)
    except ValueError:
        return ""

    # Dùng .get() lọc chính xác file_name trong metadata
    result = collection.get(
        where={
            "$and": [
                {"workspace_name": workspace_name},
                {"file_name": file_name}
            ]
        },
        include=["documents", "metadatas"]
    )

    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []

    if not documents:
        return ""

    # Gộp document và page thành tuple rồi sort theo số trang tăng dần
    sorted_pages = sorted(
        zip(documents, metadatas),
        key=lambda pair: pair[1].get("page", 0) if pair[1] else 0
    )

    # Nối tất cả các chuỗi văn bản lại thành một văn bản toàn vẹn
    full_text = "\n".join([page[0] for page in sorted_pages])
    return full_text

# =============================================================================
# BACKWARD COMPATIBILITY — Giữ lại class cũ để không break code hiện tại
# =============================================================================

class RAGPipeline:
    """
    [DEPRECATED] Class cũ từ rag_pipeline.py — giữ lại để không break code.
    Nên dùng search_workspace() và add_chunks_to_workspace() thay thế.
    """
    def __init__(self, collection_name: str = "rag_collection"):
        import chromadb
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(collection_name)

    def add_documents(self, chunks: list):
        from src.models.llm_generator import get_embedding
        embeddings = get_embedding(chunks)
        self.collection.add(
            ids=[str(i) for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings
        )

    def retrieve(self, query: str, k: int = 4) -> list:
        from src.models.llm_generator import get_embedding
        query_vector = get_embedding([query])
        res = self.collection.query(query_embeddings=query_vector, n_results=k)
        return res["documents"][0]
