"""ASTRA INTEL — Flask app (API + static frontend)."""
from __future__ import annotations

import os

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

from astra_intel import db, llm, qa
from astra_intel.extract import ExtractionError
from astra_intel.ingest import ingest

try:  # optional: load .env for local development
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(BASE, "sample_docs")
MAX_MB = int(os.environ.get("MAX_UPLOAD_MB", "25"))
MAX_QUESTION = 1000

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024


def err(msg, code=400):
    return jsonify({"error": msg}), code


@app.errorhandler(RequestEntityTooLarge)
def too_big(_):
    return err(f"File too large (limit {MAX_MB} MB).", 413)


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/status")
def status():
    cfg = llm.config()  # best available default
    return jsonify({"llm_mode": cfg["mode"], "model": cfg["model"], "default_mode": cfg["mode"],
                    "providers": llm.providers(), "documents": len(db.list_documents()),
                    "samples": sorted(f for f in os.listdir(SAMPLES) if f.endswith(".pdf")) if os.path.isdir(SAMPLES) else []})


@app.get("/api/system")
def system():
    """Static technical description of the running pipeline (shown in the UI 'System' panel)."""
    from astra_intel import extract
    from astra_intel.nlp import BM25
    docs = db.list_documents()
    return jsonify({
        "pipeline": [
            {"stage": "Extraction", "tech": "pypdf, per-page text; running headers/footers, URLs, citation marks removed"},
            {"stage": "Chunking", "tech": f"~{extract.CHUNK_CHARS} chars, {extract.OVERLAP_CHARS} overlap, never crossing a page; bibliography chunks dropped"},
            {"stage": "Retrieval", "tech": f"Okapi BM25 (k1=1.4, b=0.75), light stemmer, top-{qa.TOP_K} (+ best chunk per document when searching all)"},
            {"stage": "Grounding gate", "tech": "IDF-weighted query-term coverage, off-topic term check, statistic check"},
            {"stage": "Answer", "tech": "LLM (cite-or-refuse prompt, temperature 0) or extractive sentences"},
            {"stage": "Storage", "tech": "SQLite: documents, page-bound chunks, chat history"},
        ],
        "library": {"documents": len(docs), "chunks": sum(d["n_chunks"] for d in docs), "pages": sum(d["pages"] for d in docs)},
        "limits": {"max_upload_mb": MAX_MB, "max_question_chars": MAX_QUESTION},
        "providers": llm.providers(),
        "server": {"framework": "Flask", "storage": "SQLite"},
    })


@app.get("/api/docs")
def docs():
    return jsonify(db.list_documents())


@app.get("/api/docs/<int:doc_id>")
def doc_detail(doc_id):
    d = db.get_document(doc_id)
    return jsonify(d) if d else err("Document not found.", 404)


@app.get("/api/docs/<int:doc_id>/page/<int:page>")
def doc_page(doc_id, page):
    text = db.page_text(doc_id, page)
    return jsonify({"doc_id": doc_id, "page": page, "text": text}) if text else err("Page not found.", 404)


@app.post("/api/docs")
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return err("No file received. Choose a PDF to upload.")
    try:
        res = ingest(f.read(), os.path.basename(f.filename), request.form.get("mode", "auto"))
    except llm.NotConfigured as e:
        return err(str(e), 409)
    except ExtractionError as e:
        return err(str(e), 422)
    except Exception:  # never leak internals to the client
        app.logger.exception("ingest failed")
        return err("Something went wrong while processing this file.", 500)
    return jsonify({**res, "document": db.get_document(res["id"])}), (200 if res["duplicate"] else 201)


@app.post("/api/samples/<name>")
def load_sample(name):
    safe = os.path.basename(name)
    path = os.path.join(SAMPLES, safe)
    if safe != name or not safe.endswith(".pdf") or not os.path.isfile(path):
        return err("Unknown sample document.", 404)
    with open(path, "rb") as fh:
        try:
            res = ingest(fh.read(), safe, request.args.get("mode", "auto"))
        except llm.NotConfigured as e:
            return err(str(e), 409)
        except ExtractionError as e:
            return err(str(e), 422)
    return jsonify({**res, "document": db.get_document(res["id"])})


@app.delete("/api/docs/<int:doc_id>")
def delete_doc(doc_id):
    ok = db.delete_document(doc_id)
    qa.invalidate()
    return jsonify({"deleted": ok}) if ok else err("Document not found.", 404)


def _scope_ok(scope):
    if scope == "all":
        return True
    return scope.isdigit() and db.get_document(int(scope)) is not None


@app.post("/api/chat")
def chat():
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    scope = str(body.get("scope", "all"))
    if not db.list_documents():
        return err("Upload a document first, then ask a question.", 409)
    if not question:
        return err("Type a question first.")
    if len(question) > MAX_QUESTION:
        return err(f"Question too long (max {MAX_QUESTION} characters).")
    if not _scope_ok(scope):
        return err("That document no longer exists.", 404)
    try:
        return jsonify(qa.answer(scope, question, body.get("mode", "auto")))
    except llm.NotConfigured as e:
        return err(str(e), 409)
    except llm.LLMError as e:
        return err(str(e), 400)


@app.get("/api/history")
def history():
    scope = request.args.get("scope", "all")
    return jsonify(db.get_history(scope)) if _scope_ok(scope) else err("Unknown scope.", 404)


@app.delete("/api/history")
def clear_history():
    scope = request.args.get("scope", "all")
    db.clear_history(scope)
    return jsonify({"cleared": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=os.environ.get("FLASK_DEBUG") == "1")
