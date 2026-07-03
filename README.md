# PDF RAG Assistant 

Chatbot hỏi đáp tài liệu học tập, hỗ trợ nhiều workspace, trích dẫn nguồn, phân tích chéo nhiều file PDF, đăng nhập Google, lưu lịch sử chat và tìm kiếm web bổ sung.

---

## Kiến trúc tổng quan

```
Browser (frontend/index.html)  ──(HTTP/REST + Session Cookie)──►  FastAPI Backend
     │  Vanilla HTML/CSS/JS (SPA)                                        │
     ├─ Đăng nhập Google (OAuth2)                                        ├──► ChromaDB (Persistent, ./chroma_db)
     ├─ Sidebar: Workspaces + Lịch sử chat                                ├──► Gemini API (LLM + Embedding)
     ├─ Upload PDF (FormData)                                             ├──► Google Custom Search API (Web Search)
     ├─ Chat (RAG thuần / kèm Web Search)                                 └──► chat_history/<user_sub>/*.json
     ├─ Tóm tắt tài liệu, Giải thích thuật ngữ
     └─ Voice input (Web Speech API) + Text-to-Speech
```

Backend gồm 4 module chính, mỗi module do một thành viên phụ trách:

| Module | File | Phụ trách |
|---|---|---|
| Parsing & Chunking | `src/data/parser.py` | Thùy |
| Vector Store & Retrieval | `src/pipeline/retriever.py` | Phi |
| LLM Generation & Web Search | `src/models/llm_generator.py` | Tiến |
| API & Frontend | `src/api/api_server.py`, `frontend/index.html` | Mạnh |

---

## Tính năng chính

- **Workspace**: nhóm nhiều PDF theo chủ đề, hỏi đáp chéo (cross-analysis) trên toàn bộ file trong workspace.
- **RAG có trích dẫn**: câu trả lời kèm nguồn (tên file, số trang, tiêu đề mục).
- **Web Search Agentic** (`/api/chat/web`): kết hợp kết quả từ tài liệu (RAG) và tìm kiếm web (Google Custom Search) để tổng hợp câu trả lời.
- **Tóm tắt tài liệu** và **Tra cứu / giải thích thuật ngữ** theo ngữ cảnh workspace.
- **Đăng nhập Google (OAuth2)**: lưu phiên làm việc qua session cookie.
- **Lịch sử chat theo user**: lưu dưới dạng file JSON, hiển thị dạng sidebar giống ChatGPT.
- **Trích xuất PDF nâng cao**: nhận diện heading theo font-size/pattern, trích bảng biểu sang Markdown table (PyMuPDF + pdfplumber).
- **Hybrid Search (tùy chọn)**: kết hợp Vector Search + BM25 bằng Reciprocal Rank Fusion.
- **Giao diện**: dark theme, hỗ trợ nhập bằng giọng nói (Web Speech API) và đọc to câu trả lời (TTS).

---

## Cấu trúc thư mục

```
pdf-rag-chatbot/
├── frontend/
│   └── index.html          ← MẠNH: Giao diện SPA thuần HTML/CSS/JS (không dùng Streamlit)
├── src/
│   ├── data/
│   │   └── parser.py        ← THÙY: PDF parsing + chunking + metadata
│   ├── pipeline/
│   │   └── retriever.py     ← PHI: ChromaDB persistent + workspace search + hybrid search
│   ├── models/
│   │   └── llm_generator.py ← TIẾN: Gemini prompting + citations + web search + embedding
│   └── api/
│       └── api_server.py    ← MẠNH: FastAPI endpoints (OAuth2, chat, history, workspace)
├── chroma_db/                ← Dữ liệu vector lưu persistent (tự sinh khi chạy)
├── chat_history/              ← Lịch sử chat theo user (tự sinh khi chạy)
├── configs/
│   └── config.yaml
├── notebooks/
├── requirements.txt
├── git_setup.sh
├── .env                       ← Biến môi trường (không commit)
└── README.md
```

> Lưu ý: hệ thống **không còn dùng Streamlit** (`src/app/`) — toàn bộ giao diện hiện là một trang tĩnh `frontend/index.html` được FastAPI phục vụ trực tiếp tại `/`.

---

## Cài đặt & Chạy

### 1. Cài dependencies

```bash
pip install -r requirements.txt
```

### 2. Cấu hình biến môi trường (`.env`)

```env
# Gemini API — bắt buộc (dùng cho cả LLM sinh câu trả lời và embedding)
GEMINI_API_KEY=your_gemini_api_key

# Google OAuth2 — đăng nhập
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
FRONTEND_URL=http://localhost:8000

# Session — bắt buộc đổi khi deploy thật
SESSION_SECRET=change-me-to-a-random-secret-32chars

# Google Custom Search — dùng cho tính năng Web Search (tùy chọn)
GOOGLE_SEARCH_API_KEY=your_search_api_key
GOOGLE_SEARCH_CX=your_custom_search_engine_id

# Thư mục lưu lịch sử chat (tùy chọn, mặc định ./chat_history)
CHAT_HISTORY_DIR=./chat_history
```

### 3. Chạy server

```bash
uvicorn src.api.api_server:app --reload --port 8000
```

Mở trình duyệt tại **http://localhost:8000** — FastAPI phục vụ luôn cả frontend (`frontend/index.html`) và API, không cần chạy thêm tiến trình nào khác.

---

## Danh sách API Endpoints

### Auth
| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/auth/login` | Chuyển hướng đăng nhập Google |
| GET | `/auth/callback` | Google redirect về sau khi đăng nhập |
| GET | `/auth/logout` | Đăng xuất, xóa session |
| GET | `/auth/me` | Lấy thông tin user hiện tại |

### Chat History
| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/api/history/sessions` | Liệt kê các session chat của user |
| POST | `/api/history/sessions` | Tạo session chat mới |
| GET | `/api/history/sessions/{session_id}` | Lấy toàn bộ tin nhắn của 1 session |
| PUT | `/api/history/sessions/{session_id}` | Lưu/cập nhật tin nhắn của session |
| DELETE | `/api/history/sessions/{session_id}` | Xóa session |

### Workspace & Upload
| Method | Endpoint | Mô tả |
|---|---|---|
| POST | `/api/workspaces` | Tạo workspace mới |
| GET | `/api/workspaces` | Liệt kê workspace + số file |
| DELETE | `/api/workspaces/{workspace_name}` | Xóa workspace |
| GET | `/api/workspaces/{workspace_name}/files` | Liệt kê file trong workspace |
| POST | `/api/upload` | Upload PDF, tự parse + chunk + lưu vector |

### Chat / RAG
| Method | Endpoint | Mô tả |
|---|---|---|
| POST | `/api/chat` | Hỏi đáp RAG thuần tài liệu, trả lời kèm trích dẫn |
| POST | `/api/chat/web` | RAG + Web Search, tổng hợp câu trả lời từ cả hai nguồn |
| POST | `/api/summarize` | Tóm tắt toàn bộ hoặc 1 file trong workspace |
| POST | `/api/explain` | Giải thích thuật ngữ theo ngữ cảnh workspace |

### Khác
| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/health` | Kiểm tra trạng thái server |
| GET | `/` | Trả về `frontend/index.html` |

---

## Công nghệ sử dụng

- **Backend**: FastAPI, Starlette (Session Middleware), Authlib (Google OAuth2)
- **Vector DB**: ChromaDB (`PersistentClient`, lưu tại `./chroma_db`)
- **LLM & Embedding**: Google Gemini API (`gemini-2.5-flash-lite` cho sinh câu trả lời, `gemini-embedding-001` cho embedding)
- **Web Search**: Google Custom Search JSON API
- **PDF Parsing**: PyMuPDF (text + heading detection theo font-size), pdfplumber (trích bảng biểu → Markdown table)
- **Frontend**: HTML/CSS/JS thuần (dark theme, không framework), Web Speech API (voice input + TTS)

---
