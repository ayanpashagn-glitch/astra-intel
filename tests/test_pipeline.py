"""Run with:  python -m pytest -q   (uses the bundled sample PDFs; no API key needed)."""
import io
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
SAMPLES = os.path.join(os.path.dirname(__file__), "..", "sample_docs")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRA_DB_PATH", str(tmp_path / "t.db"))
    for k in ("ANTHROPIC_API_KEY", "LLM_API_KEY", "LLM_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    import importlib
    from astra_intel import db, qa
    importlib.reload(db)
    importlib.reload(qa)
    import app as appmod
    importlib.reload(appmod)
    qa.invalidate()
    return appmod.app.test_client()


def up(client, name="defence-electronics-electronic-warfare.pdf", mode="auto"):
    with open(os.path.join(SAMPLES, name), "rb") as f:
        return client.post("/api/docs", data={"file": (io.BytesIO(f.read()), name), "mode": mode}, content_type="multipart/form-data")


def ask(client, q, scope="all"):
    return client.post("/api/chat", json={"question": q, "scope": scope}).get_json()


# ---- upload / error states -------------------------------------------------
def test_upload_ok_and_duplicate(client):
    r = up(client)
    assert r.status_code == 201 and r.get_json()["document"]["pages"] == 12
    assert up(client).get_json()["duplicate"] is True
    assert len(client.get("/api/docs").get_json()) == 1


@pytest.mark.parametrize("name,data,msg", [
    ("a.pdf", b"", "empty"),
    ("a.pdf", b"%PDF-1.4 not really", "corrupt"),
    ("a.exe", b"MZ", "Unsupported"),
])
def test_bad_uploads_give_clear_errors(client, name, data, msg):
    r = client.post("/api/docs", data={"file": (io.BytesIO(data), name)}, content_type="multipart/form-data")
    assert r.status_code == 422 and msg.lower() in r.get_json()["error"].lower()


def test_blank_pdf_is_reported_as_scanned(client):
    from pypdf import PdfWriter
    buf = io.BytesIO(); w = PdfWriter(); w.add_blank_page(200, 200); w.write(buf)
    r = client.post("/api/docs", data={"file": (io.BytesIO(buf.getvalue()), "b.pdf")}, content_type="multipart/form-data")
    assert r.status_code == 422 and "scanned" in r.get_json()["error"].lower()


def test_chat_before_upload_is_rejected(client):
    r = client.post("/api/chat", json={"question": "hi", "scope": "all"})
    assert r.status_code == 409


def test_empty_and_overlong_question(client):
    up(client)
    assert client.post("/api/chat", json={"question": "  ", "scope": "all"}).status_code == 400
    assert client.post("/api/chat", json={"question": "x" * 1001, "scope": "all"}).status_code == 400


def test_sample_endpoint_blocks_path_traversal(client):
    assert client.post("/api/samples/..%2Fapp.py").status_code == 404


# ---- grounding --------------------------------------------------------------
def test_answer_has_page_citation_and_highlights(client):
    up(client)
    r = ask(client, "What are the three main subdivisions of electronic warfare?")
    assert r["status"] == "grounded"
    assert "electronic attack" in r["answer"].lower()
    cited = [s for s in r["sources"] if s["cited"]]
    assert cited and all(s["page"] >= 1 for s in cited) and cited[0]["highlights"]


def test_unanswerable_statistic_is_refused(client):   # the "honesty test" from the starter questions
    for n in ("unmanned-aerial-vehicle-overview.pdf", "autonomous-systems-unmanned-ground-vehicle.pdf"):
        up(client, n)
    r = ask(client, "What percentage of UAV missions are fully autonomous, according to the documents?")
    assert r["status"] == "not_found"


@pytest.mark.parametrize("q", ["Who won the 2018 FIFA World Cup?", "What is the capital of France?", "Explain how to bake sourdough bread"])
def test_off_topic_is_refused(client, q):
    up(client)
    assert ask(client, q)["status"] == "not_found"


def test_repeat_question_is_consistent(client):
    up(client)
    q = "How does electronic warfare relate to radar jamming?"
    a, b = ask(client, q), ask(client, q)
    assert a["answer"] == b["answer"] and [s["page"] for s in a["sources"]] == [s["page"] for s in b["sources"]]


# ---- multi-turn / persistence ----------------------------------------------
def test_history_persists_and_clears(client):
    up(client)
    ask(client, "What is electronic warfare?", "1")
    h = client.get("/api/history?scope=1").get_json()
    assert [m["role"] for m in h] == ["user", "assistant"] and h[1]["meta"]["sources"]
    client.delete("/api/history?scope=1")
    assert client.get("/api/history?scope=1").get_json() == []


def test_followup_uses_previous_question(client):
    up(client)
    ask(client, "What is jamming in electronic warfare?", "1")
    r = ask(client, "Why is it used?", "1")          # "it" only makes sense with the previous turn
    assert r["status"] in ("grounded", "low_confidence")


def test_delete_document_removes_chunks_and_history(client):
    up(client); ask(client, "What is electronic warfare?", "1")
    assert client.delete("/api/docs/1").status_code == 200
    assert client.get("/api/docs").get_json() == []
    assert client.get("/api/history?scope=1").status_code == 404


# ---- LLM layer (mocked HTTP: verifies request/response handling, not a live API) ----
class FakeResp:
    def __init__(self, status, body): self.status_code, self._b = status, body
    def json(self): return self._b


def test_anthropic_answer_parsing_and_citations(client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    seen = {}
    def fake_post(url, headers=None, json=None, timeout=None):
        seen.update(url=url, headers=headers, body=json)
        return FakeResp(200, {"content": [{"type": "text", "text": "EW has three subdivisions [P1]."}]})
    import astra_intel.llm as llm
    monkeypatch.setattr(llm.requests, "post", fake_post)
    up(client)
    r = ask(client, "What are the three main subdivisions of electronic warfare?")
    assert r["answer"].startswith("EW has three") and r["mode"] == "anthropic"
    assert seen["url"].endswith("/v1/messages") and seen["headers"]["x-api-key"] == "test"
    assert seen["body"]["temperature"] == 0 and "ONLY the numbered passages" in seen["body"]["system"]
    assert any(s["cited"] and s["id"] == "P1" for s in r["sources"])


def test_llm_not_in_document_sentinel(client, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    import astra_intel.llm as llm
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: FakeResp(200, {"choices": [{"message": {"content": "NOT_IN_DOCUMENT"}}]}))
    up(client)
    assert ask(client, "What are the three main subdivisions of electronic warfare?")["status"] == "not_found"


def test_llm_failure_falls_back_to_extractive(client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "bad")
    import astra_intel.llm as llm
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: FakeResp(401, {}))
    up(client)
    r = ask(client, "What are the three main subdivisions of electronic warfare?")
    assert r["status"] == "grounded" and "rejected" in r["note"] and r["mode"] == "extractive"


# ---- selectable mode / "not configured" -------------------------------------
def test_status_reports_provider_configuration(client):
    p = client.get("/api/status").get_json()["providers"]
    assert p["local"]["configured"] and not p["anthropic"]["configured"] and not p["openai"]["configured"]


def test_choosing_unconfigured_provider_returns_not_configured(client):
    up(client)
    for m in ("anthropic", "openai"):
        r = client.post("/api/chat", json={"question": "What is electronic warfare?", "scope": "all", "mode": m})
        assert r.status_code == 409 and "not configured" in r.get_json()["error"].lower()
    # nothing was saved to history by the rejected request
    assert client.get("/api/history?scope=all").get_json() == []


def test_upload_with_unconfigured_provider_rejected(client):
    with open(os.path.join(SAMPLES, "defence-electronics-electronic-warfare.pdf"), "rb") as f:
        r = client.post("/api/docs", data={"file": (io.BytesIO(f.read()), "a.pdf"), "mode": "anthropic"},
                        content_type="multipart/form-data")
    assert r.status_code == 409


def test_explicit_local_mode_ignores_configured_key(client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    import astra_intel.llm as llm
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM must not be called")))
    up(client, mode="local")
    r = ask_mode(client, "What are the three main subdivisions of electronic warfare?", "local")
    assert r["tech"]["llm_called"] is False and r["mode"] == "extractive"


def ask_mode(client, q, mode, scope="all"):
    return client.post("/api/chat", json={"question": q, "scope": scope, "mode": mode}).get_json()


def test_technical_details_present_and_persisted(client):
    up(client)
    r = ask_mode(client, "What are the three main subdivisions of electronic warfare?", "local", "1")
    t = r["tech"]
    assert t["gate"]["tier"] == "high" and t["query_terms"] and t["top_scores"] and t["timing_ms"]["total"] >= 0
    assert client.get("/api/history?scope=1").get_json()[-1]["meta"]["tech"]["answer_engine"] == "extractive (local)"


def test_system_endpoint(client):
    up(client)
    s = client.get("/api/system").get_json()
    assert s["library"]["documents"] == 1 and len(s["pipeline"]) >= 5 and "providers" in s


# ---- summary requests (regression: "Summarise the key points" used to return "not found") ----
@pytest.mark.parametrize("q", ["Summarise the key points of this document", "Give me an overview", "Summarize this document",
                               "What are the main points?", "tl;dr"])
def test_summary_requests_return_stored_summary(client, q):
    up(client)
    r = ask(client, q, "1")
    assert r["status"] == "grounded" and "(p." in r["answer"] and r["tech"]["answer_engine"] == "stored document summary"


def test_summary_across_all_documents_names_each_one(client):
    up(client); up(client, "unmanned-aerial-vehicle-overview.pdf")
    r = ask(client, "Summarise the key points of this document", "all")
    assert r["status"] == "grounded" and "defence-electronics" in r["answer"] and "unmanned-aerial" in r["answer"]


def test_summary_of_a_topic_still_uses_retrieval(client):
    up(client)
    r = ask(client, "Summarise what the document says about jamming", "1")
    assert r["tech"]["answer_engine"] != "stored document summary" and r["status"] in ("grounded", "low_confidence")


def test_summaries_exclude_reference_junk(client):
    for n in ("defence-electronics-electronic-warfare.pdf", "unmanned-aerial-vehicle-overview.pdf", "autonomous-systems-unmanned-ground-vehicle.pdf"):
        up(client, n)
    for d in client.get("/api/docs").get_json():
        text = client.get(f"/api/docs/{d['id']}").get_json()["summary"]
        assert "://" not in text and "See also" not in text and "Retrieved" not in text
