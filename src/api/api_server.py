"""
api_server.py  —  PDF RAG Chatbot  v3.0
Tính năng mới:
  1. Google OAuth2 (đăng nhập / đăng xuất)
  2. Chat History lưu theo user (JSON file, giống ChatGPT sidebar)
  3. Agentic Web Search  POST /api/chat/web
"""

from fastapi import FastAPI, UploadFile, Form, File, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import tempfile, os, json, uuid
from datetime import datetime
from pathlib import Path

# ── OAuth2 ──────────────────────────────────────────────────────────────────
from authlib.integrations.starlette_client import OAuth, OAuthError
from starlette.middleware.sessions import SessionMiddleware

# ── Pipeline ────────────────────────────────────────────────────────────────
from src.data.parser        import process_pdf
from src.pipeline.retriever import (
    search_workspace, add_chunks_to_workspace,
    delete_workspace, list_workspaces,
    list_files_in_workspace, get_or_create_collection,
    get_full_text_by_file # Helper Func to sumarize_doc
)
from src.models.llm_generator import (
    generate_cited_answer, summarize_doc, explain_term,
    generate_web_search_answer           # ← hàm mới (xem llm_generator.py)
)

import logging, traceback
logging.basicConfig(level=logging.DEBUG)

# ════════════════════════════════════════════════════════════════════════════
# APP + MIDDLEWARE
# ════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="PDF RAG Chatbot API",
    description="Backend API cho chatbot hỏi đáp tài liệu học tập – v3.0",
    version="3.0.0"
)

# ⚠️ Thay bằng SECRET_KEY ngẫu nhiên thật sự khi deploy
SECRET_KEY = os.getenv("SESSION_SECRET", "change-me-to-a-random-secret-32chars")

app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ════════════════════════════════════════════════════════════════════════════
# GOOGLE OAUTH2
# Cần set trong .env:
#   GOOGLE_CLIENT_ID=...
#   GOOGLE_CLIENT_SECRET=...
#   FRONTEND_URL=http://localhost:8000   (redirect về sau login)
# ════════════════════════════════════════════════════════════════════════════
oauth = OAuth()
oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID", ""),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET", ""),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:8000")


def _current_user(request: Request) -> Optional[dict]:
    """Trả về thông tin user từ session, hoặc None nếu chưa đăng nhập."""
    return request.session.get("user")


# ── Auth routes ──────────────────────────────────────────────────────────────

@app.get("/auth/login")
async def login(request: Request):
    """Redirect tới trang đăng nhập Google."""
    redirect_uri = request.url_for("auth_callback")
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/auth/callback", name="auth_callback")
async def auth_callback(request: Request):
    """Google redirect về đây sau khi user đăng nhập thành công."""
    try:
        token = await oauth.google.authorize_access_token(request)
    except OAuthError as e:
        raise HTTPException(status_code=400, detail=f"OAuth error: {str(e)}")

    user_info = token.get("userinfo")
    if not user_info:
        raise HTTPException(status_code=400, detail="Không lấy được thông tin user")

    request.session["user"] = {
        "sub":     user_info["sub"],           # Google unique ID
        "email":   user_info["email"],
        "name":    user_info.get("name", ""),
        "picture": user_info.get("picture", ""),
    }
    return RedirectResponse(url=FRONTEND_URL)


@app.get("/auth/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url=FRONTEND_URL)


@app.get("/auth/me")
async def me(request: Request):
    """Trả về thông tin user hiện tại (hoặc null nếu chưa đăng nhập)."""
    user = _current_user(request)
    return {"user": user}


# ════════════════════════════════════════════════════════════════════════════
# CHAT HISTORY  (lưu file JSON theo user_id)
# Cấu trúc thư mục:  ./chat_history/<user_sub>/<session_id>.json
# ════════════════════════════════════════════════════════════════════════════
HISTORY_ROOT = Path(os.getenv("CHAT_HISTORY_DIR", "./chat_history"))
HISTORY_ROOT.mkdir(parents=True, exist_ok=True)


def _user_dir(user_sub: str) -> Path:
    d = HISTORY_ROOT / user_sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def _session_path(user_sub: str, session_id: str) -> Path:
    return _user_dir(user_sub) / f"{session_id}.json"


# ── Pydantic schemas cho Chat History ───────────────────────────────────────

class HistoryMessage(BaseModel):
    role: str        # "user" | "assistant"
    content: str
    citations: Optional[List[dict]] = []
    timestamp: Optional[str] = None


class ChatSession(BaseModel):
    session_id: str
    title: str
    workspace_name: str
    created_at: str
    updated_at: str
    messages: List[HistoryMessage] = []


class SessionCreate(BaseModel):
    workspace_name: str
    title: Optional[str] = "Cuộc hội thoại mới"


class SaveMessagesRequest(BaseModel):
    session_id: str
    messages: List[HistoryMessage]
    title: Optional[str] = None   # cập nhật tiêu đề nếu truyền


# ── Endpoints Chat History ───────────────────────────────────────────────────

@app.get("/api/history/sessions", response_model=List[dict])
async def list_sessions(request: Request):
    """Liệt kê tất cả chat session của user hiện tại."""
    user = _current_user(request)
    if not user:
        return []   # chưa đăng nhập → trả list rỗng (không lỗi)

    d = _user_dir(user["sub"])
    sessions = []
    for f in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            sessions.append({
                "session_id":     data["session_id"],
                "title":          data.get("title", "Cuộc hội thoại"),
                "workspace_name": data.get("workspace_name", ""),
                "created_at":     data.get("created_at", ""),
                "updated_at":     data.get("updated_at", ""),
                "message_count":  len(data.get("messages", [])),
            })
        except Exception:
            pass
    return sessions


@app.post("/api/history/sessions", response_model=ChatSession, status_code=201)
async def create_session(body: SessionCreate, request: Request):
    """Tạo một chat session mới."""
    user = _current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Chưa đăng nhập")

    now = datetime.utcnow().isoformat()
    session = ChatSession(
        session_id=uuid.uuid4().hex,
        title=body.title or "Cuộc hội thoại mới",
        workspace_name=body.workspace_name,
        created_at=now,
        updated_at=now,
        messages=[],
    )
    _session_path(user["sub"], session.session_id).write_text(
        session.model_dump_json(indent=2), encoding="utf-8"
    )
    return session


@app.get("/api/history/sessions/{session_id}", response_model=ChatSession)
async def get_session(session_id: str, request: Request):
    """Lấy toàn bộ messages của một session."""
    user = _current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Chưa đăng nhập")

    p = _session_path(user["sub"], session_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Session không tồn tại")
    return json.loads(p.read_text(encoding="utf-8"))


@app.put("/api/history/sessions/{session_id}")
async def save_session(session_id: str, body: SaveMessagesRequest, request: Request):
    """Cập nhật messages (và tuỳ chọn title) cho một session."""
    user = _current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Chưa đăng nhập")

    p = _session_path(user["sub"], session_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Session không tồn tại")

    data = json.loads(p.read_text(encoding="utf-8"))
    data["messages"] = [m.model_dump() for m in body.messages]
    data["updated_at"] = datetime.utcnow().isoformat()
    if body.title:
        data["title"] = body.title
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "ok"}


@app.delete("/api/history/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str, request: Request):
    """Xóa một chat session."""
    user = _current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Chưa đăng nhập")

    p = _session_path(user["sub"], session_id)
    if p.exists():
        p.unlink()


# ════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS — giữ lại từ v2
# ════════════════════════════════════════════════════════════════════════════

class WorkspaceCreate(BaseModel):
    name: str

class WorkspaceInfo(BaseModel):
    name: str
    file_count: int

class ChatRequest(BaseModel):
    question: str
    workspace_name: str
    top_k: int = 5

class WebChatRequest(BaseModel):
    question: str
    workspace_name: str
    top_k: int = 5

class Citation(BaseModel):
    file_name: str
    page: int
    heading: str
    snippet: str

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    model: str

class WebChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    web_results: List[dict]   # kết quả web thô để frontend hiển thị
    model: str

class SummarizeRequest(BaseModel):
    workspace_name: str
    file_name: Optional[str] = None

class SummarizeResponse(BaseModel):
    summary: str
    model: str

class ExplainRequest(BaseModel):
    term: str
    workspace_name: str

class ExplainResponse(BaseModel):
    explanation: str
    citations: List[Citation]
    model: str

class UploadResponse(BaseModel):
    file_name: str
    workspace_name: str
    chunk_count: int
    page_count: int
    status: str


# ════════════════════════════════════════════════════════════════════════════
# WORKSPACE ENDPOINTS (giữ nguyên từ v2)
# ════════════════════════════════════════════════════════════════════════════

@app.post("/api/workspaces", response_model=WorkspaceInfo, status_code=201)
async def create_workspace(body: WorkspaceCreate):
    try:
        get_or_create_collection(body.name)
        return WorkspaceInfo(name=body.name, file_count=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi tạo workspace: {str(e)}")


@app.get("/api/workspaces", response_model=List[WorkspaceInfo])
async def get_workspaces():
    try:
        names = list_workspaces()
        return [WorkspaceInfo(name=n, file_count=len(list_files_in_workspace(n))) for n in names]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/workspaces/{workspace_name}", status_code=204)
async def remove_workspace(workspace_name: str):
    try:
        delete_workspace(workspace_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/workspaces/{workspace_name}/files", response_model=List[str])
async def get_workspace_files(workspace_name: str):
    try:
        return list_files_in_workspace(workspace_name)
    except Exception as e:
        raise HTTPException(status_code=404, detail="Không tìm thấy workspace")


# ════════════════════════════════════════════════════════════════════════════
# UPLOAD PDF (giữ nguyên từ v2)
# ════════════════════════════════════════════════════════════════════════════

@app.post("/api/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    workspace_name: str = Form(...)
):
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        chunks = process_pdf(tmp_path, workspace_name)
        if not chunks:
            raise ValueError("Không thể trích xuất nội dung từ PDF.")

        real_filename = file.filename
        for c in chunks:
            if "metadata" in c:
                c["metadata"]["file_name"] = real_filename

        add_chunks_to_workspace(workspace_name, chunks)
        page_count = max([c["metadata"].get("page", 1) for c in chunks])

        return UploadResponse(
            file_name=real_filename,
            workspace_name=workspace_name,
            chunk_count=len(chunks),
            page_count=page_count,
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload lỗi: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ════════════════════════════════════════════════════════════════════════════
# CHAT — RAG thuần tài liệu (giữ nguyên)
# ════════════════════════════════════════════════════════════════════════════

@app.post("/api/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    try:
        chunks = search_workspace(body.question, body.workspace_name, body.top_k)
        result = generate_cited_answer(body.question, chunks)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════════════════════════════════════════
# CHAT — Agentic Web Search  (MỚI)
# Kết hợp RAG tài liệu + tìm kiếm web bằng Google Custom Search API
# ════════════════════════════════════════════════════════════════════════════

@app.post("/api/chat/web", response_model=WebChatResponse)
async def chat_web(body: WebChatRequest):
    """
    Agentic RAG: tìm trong tài liệu + search web, tổng hợp thành câu trả lời duy nhất.
    Cần set trong .env:
        GOOGLE_SEARCH_API_KEY=...
        GOOGLE_SEARCH_CX=...         (Custom Search Engine ID)
    """
    try:
        # 1. Lấy chunks từ tài liệu (vẫn dùng RAG)
        doc_chunks = []
        try:
            doc_chunks = search_workspace(body.question, body.workspace_name, body.top_k)
        except Exception:
            pass  # workspace có thể rỗng, không sao

        # 2. Gọi hàm tổng hợp web + tài liệu
        result = await generate_web_search_answer(body.question, doc_chunks)
        return WebChatResponse(**result)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════════════════════════════════════════
# SUMMARIZE & EXPLAIN (giữ nguyên từ v2)
# ════════════════════════════════════════════════════════════════════════════

@app.post("/api/summarize", response_model=SummarizeResponse)
async def summarize(body: SummarizeRequest):
    try:
        # Trường hợp 1: Frontend truyền cụ thể tên file cần tóm tắt
        if body.file_name and body.file_name.strip():
            full_text = get_full_text_by_file(body.file_name, body.workspace_name)
            if not full_text.strip():
                raise HTTPException(status_code=44, detail=f"Không tìm thấy dữ liệu hoặc file '{body.file_name}' rỗng.")
        
        # Trường hợp 2: Frontend KHÔNG truyền file_name (bấm nút chung ở topbar)
        else:
            # Lấy danh sách toàn bộ file trong workspace bằng hàm có sẵn của Phi
            all_files = list_files_in_workspace(body.workspace_name)
            if not all_files:
                raise HTTPException(status_code=404, detail=f"Workspace '{body.workspace_name}' chưa có file nào để tóm tắt.")
            
            # Quét qua từng file để lấy text và gộp lại
            text_segments = []
            for fname in all_files:
                f_text = get_full_text_by_file(fname, body.workspace_name)
                if f_text.strip():
                    text_segments.append(f"--- NỘI DUNG FILE: {fname} ---\n{f_text}")
            
            full_text = "\n\n".join(text_segments)

        # Kiểm tra cuối cùng xem tổng lượng text thu được có bị rỗng không
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="Không thu thập được nội dung văn bản nào để tóm tắt.")

        # Gọi hàm xử lý truyền chuỗi văn bản thuần sang cho Tiến xử lý với Gemini
        result = summarize_doc(full_text)
        return SummarizeResponse(**result)
        
    except HTTPException as he:
        raise he
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/explain", response_model=ExplainResponse)
async def explain(body: ExplainRequest):
    try:
        chunks = search_workspace(body.term, body.workspace_name, top_k=3)
        result = explain_term(body.term, chunks)
        return ExplainResponse(
            explanation=result.get("answer", "Không có nội dung"),
            citations=result.get("citations", []),
            model=result.get("model", "unknown")
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════════════════════════════════════════
# HEALTH + STATIC
# ════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "3.0.0"}


@app.get("/")
async def serve_frontend():
    return FileResponse("frontend/index.html")
