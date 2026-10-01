"use strict";
/* ASTRA INTEL frontend. All model/user text is inserted with textContent (never innerHTML). */

const $ = (id) => document.getElementById(id);
const state = { docs: [], scope: "all", status: null, busy: false, sources: [], mode: "local" };
const MODE_LABEL = { local: "Local", anthropic: "Claude API", openai: "OpenAI-compatible" };

const SUGGESTIONS = [
  "Summarise the key points of this document",
  "What are the major applications discussed?",
  "What limitations or risks are mentioned?",
];

function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) n.setAttribute(k, v);
  }
  for (const c of kids.flat(Infinity)) if (c != null) n.append(c.nodeType ? c : document.createTextNode(c));
  return n;
}

async function api(path, opts = {}) {
  let res;
  try {
    res = await fetch(path, opts);
  } catch {
    throw new Error("Cannot reach the ASTRA INTEL server. Is it running?");
  }
  let body = null;
  try { body = await res.json(); } catch { /* non-JSON error page */ }
  if (!res.ok) throw new Error((body && body.error) || `Request failed (${res.status}).`);
  return body;
}

function banner(msg, kind = "error") {
  const b = $("banner");
  if (!msg) { b.hidden = true; return; }
  b.textContent = msg;
  b.className = "banner" + (kind === "info" ? " info" : "");
  b.hidden = false;
}

/* ---------- answer-engine selector ---------- */
function pickInitialMode() {
  const p = state.status.providers;
  let saved = null;
  try { saved = localStorage.getItem("astra.mode"); } catch { /* storage blocked */ }
  state.mode = saved && p[saved] && p[saved].configured ? saved : state.status.default_mode;
}

function renderMode() {
  const p = state.status.providers, box = $("modeSwitch");
  box.replaceChildren(...["local", "anthropic", "openai"].map((m) => {
    const info = p[m], on = state.mode === m;
    const btn = el("button", { type: "button", role: "radio", "aria-checked": String(on), class: info.configured ? "" : "off",
      title: info.configured ? `${MODE_LABEL[m]}: ${info.model}` : `${MODE_LABEL[m]} is not configured. ${info.hint}`,
      onclick: () => chooseMode(m) },
      el("span", { class: "t" }, MODE_LABEL[m]),
      el("span", { class: "m" }, info.configured ? info.model : "Not configured"));
    return btn;
  }));
}

function chooseMode(m) {
  const info = state.status.providers[m];
  if (!info.configured) {
    banner(`${MODE_LABEL[m]} is not configured. ${info.hint}, then restart the server. Staying on ${MODE_LABEL[state.mode]}.`, "info");
    return;
  }
  banner("");
  state.mode = m;
  try { localStorage.setItem("astra.mode", m); } catch { /* ignore */ }
  renderMode();
}

/* ---------- system dialog ---------- */
async function openSystem() {
  const body = $("sysBody");
  body.replaceChildren(el("p", { class: "muted" }, "Loading…"));
  $("sysDialog").showModal();
  try {
    const s = await api("/api/system");
    const stat = (v, l) => el("div", {}, el("b", {}, String(v)), el("span", {}, l));
    body.replaceChildren(
      el("h3", {}, "Library"),
      el("div", { class: "stat" }, stat(s.library.documents, "documents"), stat(s.library.pages, "pages"), stat(s.library.chunks, "searchable passages")),
      el("h3", {}, "Pipeline"),
      el("table", {}, el("tbody", {}, s.pipeline.map((r) => el("tr", {}, el("th", {}, r.stage), el("td", {}, r.tech))))),
      el("h3", {}, "Answer engines"),
      el("table", {}, el("tbody", {}, ["local", "anthropic", "openai"].map((m) => {
        const i = s.providers[m];
        return el("tr", {}, el("th", {}, MODE_LABEL[m]),
          el("td", {}, el("span", { class: i.configured ? "ok" : "no" }, i.configured ? "Configured" : "Not configured"),
            ` · ${i.model}`, i.endpoint && i.configured ? ` · ${i.endpoint}` : "", i.configured ? "" : ` — ${i.hint}`));
      }))),
      el("h3", {}, "Limits"),
      el("p", {}, `Upload max ${s.limits.max_upload_mb} MB · question max ${s.limits.max_question_chars} characters · server: ${s.server.framework} + ${s.server.storage}`));
  } catch (e) { body.replaceChildren(el("p", { class: "muted" }, e.message)); }
}

/* ---------- library ---------- */
function renderDocs() {
  const ul = $("docList");
  ul.replaceChildren();
  if (state.docs.length) {
    ul.append(docRow("all", "All documents", `${state.docs.length} loaded, searched together`));
    for (const d of state.docs) ul.append(docRow(String(d.id), d.name, `${d.pages} pages, ${d.n_chunks} passages`, d));
  }
  const loaded = new Set(state.docs.map((d) => d.name));
  const missing = ((state.status && state.status.samples) || []).filter((s) => !loaded.has(s));
  renderScope();
  $("samples").hidden = missing.length === 0;
  $("sampleButtons").replaceChildren(...missing.map((s) =>
    el("button", { type: "button", onclick: () => loadSample(s) }, s.replace(".pdf", ""))));
}

function renderScope() {
  const t = $("scopeTitle"), m = $("scopeMeta");
  if (!state.docs.length) { t.textContent = "ASTRA INTEL"; m.textContent = "No document loaded yet"; return; }
  if (state.scope === "all") { t.textContent = "All documents"; m.textContent = `${state.docs.length} document(s) searched together`; return; }
  const d = state.docs.find((x) => String(x.id) === state.scope);
  t.textContent = d ? d.name : "Document"; m.textContent = d ? `${d.pages} pages · ${d.n_chunks} searchable passages` : "";
}

function docRow(scope, name, meta, doc) {
  const li = el("li", { class: "doc" + (state.scope === scope ? " active" : "") });
  li.append(el("button", { class: "pick", type: "button", "aria-pressed": String(state.scope === scope), onclick: () => setScope(scope) },
    el("span", { class: "name" }, name), el("span", { class: "meta" }, meta)));
  if (doc) li.append(el("button", { class: "del", type: "button", title: "Remove document", "aria-label": `Remove ${name}`, onclick: () => removeDoc(doc) }, "×"));
  return li;
}

async function refreshDocs() {
  state.docs = await api("/api/docs");
  if (state.scope !== "all" && !state.docs.some((d) => String(d.id) === state.scope)) state.scope = "all";
  renderDocs();
  renderBrief();
  updateComposer();
}

async function setScope(scope) {
  state.scope = scope;
  state.sources = [];
  renderDocs(); renderBrief(); renderEvidence();
  await loadHistory();
}

async function removeDoc(d) {
  if (!confirm(`Remove "${d.name}" and its conversation?`)) return;
  try { await api(`/api/docs/${d.id}`, { method: "DELETE" }); await refreshDocs(); await loadHistory(); }
  catch (e) { banner(e.message); }
}

/* ---------- upload ---------- */
function uploadState(msg, kind) {
  const u = $("uploadState");
  if (!msg) { u.hidden = true; return; }
  u.hidden = false;
  u.className = "upload-state" + (kind === "error" ? " error" : "");
  u.replaceChildren(...(kind === "busy" ? [el("span", { class: "spin" })] : []), msg);
}

async function upload(file) {
  if (!file) return;
  banner("");
  uploadState(`Reading and summarising ${file.name}…`, "busy");
  const fd = new FormData();
  fd.append("file", file);
  fd.append("mode", state.mode);
  try {
    const r = await api("/api/docs", { method: "POST", body: fd });
    await refreshDocs();
    await setScope(String(r.id));
    uploadState(r.duplicate ? "That document was already uploaded. Selected it." : "");
  } catch (e) {
    uploadState(e.message, "error");
  }
  $("file").value = "";
}

async function loadSample(name) {
  uploadState(`Loading ${name}…`, "busy");
  try {
    const r = await api(`/api/samples/${encodeURIComponent(name)}?mode=${state.mode}`, { method: "POST" });
    await refreshDocs();
    await setScope(String(r.id));
    uploadState("");
  } catch (e) { uploadState(e.message, "error"); }
}

/* ---------- brief ---------- */
function renderSummary(text) {
  const box = el("div", { class: "summary" });
  for (const part of text.split(/(\(p\.\d+\))/)) {
    box.append(/^\(p\.\d+\)$/.test(part) ? el("span", { class: "pg" }, part) : part);
  }
  return box;
}

function keywordChips(d) {
  return el("div", { class: "chips" }, (d.keywords || []).map((k) =>
    el("button", { class: "chip", type: "button", title: "Ask about this topic", onclick: () => ask(`What does the document say about ${k}?`) }, k)));
}

async function renderBrief() {
  const b = $("brief");
  if (!state.docs.length) { b.hidden = true; b.replaceChildren(); return; }
  b.hidden = false;
  if (state.scope === "all") {
    const box = el("div", {}, el("p", { class: "kicker" }, "Library brief"), el("h2", {}, "All documents"),
      el("div", { class: "sub" }, "Expand a document for its summary. Pick one in the sidebar to focus on it."));
    b.replaceChildren(box);
    for (const d of state.docs) {
      const full = await api(`/api/docs/${d.id}`).catch(() => null);
      if (state.scope !== "all") return; // user switched while loading
      if (full) box.append(el("details", {}, el("summary", {}, full.name), renderSummary(full.summary), keywordChips(full)));
    }
    return;
  }
  const scope = state.scope;
  const d = await api(`/api/docs/${scope}`).catch(() => null);
  if (!d || state.scope !== scope) return;
  const how = d.summary_mode === "extractive" ? "extractive summary" : `summary by ${MODE_LABEL[d.summary_mode] || d.summary_mode}`;
  const sum = renderSummary(d.summary);
  sum.classList.add("clamp");
  const tog = el("button", { class: "link", type: "button", onclick: () => {
    if (b.classList.contains("compact")) { b.classList.remove("compact"); sum.classList.add("clamp"); tog.textContent = "Show full summary"; return; }
    tog.textContent = sum.classList.toggle("clamp") ? "Show full summary" : "Show less"; } }, "Show full summary");
  if (b.classList.contains("compact")) tog.textContent = "Show summary";
  b.replaceChildren(el("p", { class: "kicker" }, "Document brief"), el("div", { class: "sub" }, `${d.pages} pages · ${d.n_chunks} passages · ${how}`),
    sum, tog, keywordChips(d));
}

/* ---------- thread ---------- */
const STATUS_LABEL = {
  grounded: "Answered from the document",
  low_confidence: "Low confidence: closest passages, not a confirmed answer",
  not_found: "Not found in the document",
};

function richText(text, sources) {
  const frag = document.createDocumentFragment();
  const byId = Object.fromEntries((sources || []).map((s) => [s.id, s]));
  // tokens: [P3] citations and **bold**
  for (const part of text.split(/(\[P\d+\]|\*\*[^*]+\*\*)/)) {
    const cite = part.match(/^\[(P\d+)\]$/);
    if (cite) {
      const s = byId[cite[1]];
      frag.append(el("button", { class: "cite", type: "button", title: s ? `${s.doc_name}, page ${s.page}` : "source",
        onclick: () => focusSource(cite[1]) }, s ? `p.${s.page}` : cite[1]));
    } else if (/^\*\*[^*]+\*\*$/.test(part)) frag.append(el("strong", {}, part.slice(2, -2)));
    else frag.append(part);
  }
  return frag;
}

function botMessage(m) {
  const meta = m.meta || {};
  const status = meta.status || "grounded";
  const node = el("div", { class: `msg bot ${status}` },
    el("div", { class: "status" }, STATUS_LABEL[status] || "", meta.tech ? el("span", { class: "engine-tag" }, `· ${meta.tech.answer_engine}`) : null),
    el("div", { class: "answer" }, richText(m.content, meta.sources)));
  if (meta.note) node.append(el("div", { class: "note" }, meta.note));
  const seen = new Set();
  const cited = (meta.sources || []).filter((s) => {
    const k = `${s.doc_id}:${s.page}`;
    if (!s.cited || seen.has(k)) return false;
    seen.add(k);
    return true;
  });
  const shown = status === "grounded" && cited.length ? cited : [];
  if (shown.length) {
    node.append(el("div", { class: "srcline" }, shown.map((s) =>
      el("button", { type: "button", onclick: () => focusSource(s.id, meta.sources) }, `${shortName(s.doc_name)}, p.${s.page}`))));
  }
  if (meta.tech) node.append(techDetails(meta.tech));
  if ((meta.sources || []).length) {
    node.append(el("div", { class: "srcline" }, el("button", { type: "button", onclick: () => showSources(meta.sources) },
      status === "grounded" ? "Show all evidence" : "Show closest passages")));
  }
  return node;
}

function techDetails(t) {
  const g = t.gate, max = Math.max(...t.top_scores.map((x) => x.bm25), 1);
  const tierCls = { high: "tag-ok", low: "tag-warn", none: "tag-bad" }[g.tier];
  const kv = (k, ...v) => [el("dt", {}, k), el("dd", {}, ...v)];
  const body = el("div", { class: "tech-body" },
    el("dl", { class: "kv" },
      ...kv("Answer engine", t.answer_engine),
      ...kv("Model", t.model),
      ...kv("LLM called", t.llm_called ? `yes (temperature ${t.temperature})` : "no"),
      ...kv("Query terms (stems)", t.query_terms.length ? t.query_terms.map((x, i) => [i ? " " : "", el("code", {}, x)]) : "none"),
      ...kv("Searched", `${t.chunks_searched} passages → top ${t.passages_retrieved} by BM25`),
      ...kv("Grounding gate", el("span", { class: tierCls }, g.tier.toUpperCase()), ` · term coverage ${(g.coverage * 100).toFixed(0)}% · statistic check ${g.statistic_check ? "pass" : "fail"}`),
      ...kv("Timing", `retrieval ${t.timing_ms.retrieval} ms` + (t.timing_ms.llm != null ? ` · LLM ${t.timing_ms.llm} ms` : "") + ` · total ${t.timing_ms.total} ms`)));
  if (t.top_scores.length) {
    body.append(el("table", { class: "mini" }, el("thead", {}, el("tr", {}, ["Passage", "Page", "BM25 score", ""].map((h) => el("th", {}, h)))),
      el("tbody", {}, t.top_scores.map((x) => el("tr", {}, el("td", {}, x.id), el("td", {}, String(x.page)), el("td", {}, x.bm25.toFixed(2)),
        el("td", {}, el("span", { class: "bar2", style: `width:${Math.round((x.bm25 / max) * 90)}px` })))))));
  }
  return el("details", { class: "tech" }, el("summary", {}, "Technical details"), body);
}

function shortName(n) { return n.replace(/\.(pdf|txt|md)$/i, "").slice(0, 28); }

function hero() {
  if (!state.docs.length) {
    return el("div", { class: "hero" }, el("p", { class: "kicker" }, "Defence document intelligence"),
      el("h2", {}, "Upload a document. Ask it anything."),
      el("p", { class: "lede" }, "Answers come only from your file, and every one shows the page it was found on."),
      el("ol", { class: "steps" },
        el("li", {}, el("b", {}, "Upload"), " a PDF in the sidebar, or load an ASTRA starter document."),
        el("li", {}, el("b", {}, "Ask"), " a question in plain English."),
        el("li", {}, el("b", {}, "Verify"), " the cited page in the Evidence panel.")));
  }
  return el("div", { class: "hero" }, el("p", { class: "kicker" }, "Ready"), el("h2", {}, "What do you want to know?"),
    el("p", { class: "lede" }, "Try one of these, or type your own question below."),
    el("div", { class: "cards" }, SUGGESTIONS.map((q) => el("button", { class: "scard", type: "button", onclick: () => ask(q) }, q))));
}

function renderThread(messages) {
  const t = $("thread");
  t.replaceChildren();
  for (const m of messages) t.append(m.role === "user" ? el("div", { class: "msg user" }, m.content) : botMessage(m));
  $("brief").classList.toggle("compact", messages.length > 0);
  const lk = $("brief").querySelector(".link"); if (lk) lk.textContent = messages.length ? "Show summary" : "Show full summary";
  if (!messages.length) t.append(hero());
  t.scrollTop = t.scrollHeight;
}

async function loadHistory() {
  if (!state.docs.length) { renderThread([]); return; }
  try {
    const h = await api(`/api/history?scope=${encodeURIComponent(state.scope)}`);
    renderThread(h);
    const last = [...h].reverse().find((m) => m.role === "assistant");
    state.sources = last && last.meta ? last.meta.sources || [] : [];
    renderEvidence();
  } catch (e) { banner(e.message); }
}

/* ---------- evidence ---------- */
function highlighted(text, ranges) {
  const out = document.createDocumentFragment();
  let pos = 0;
  for (const [s, e] of ranges || []) {
    if (s < pos) continue;
    out.append(text.slice(pos, s), el("mark", {}, text.slice(s, e)));
    pos = e;
  }
  out.append(text.slice(pos));
  return out;
}

function showSources(sources) { state.sources = sources || []; renderEvidence(); }

function renderEvidence() {
  const box = $("evidence");
  if (!state.sources.length) {
    box.replaceChildren(el("div", { class: "ev-empty" }, "Source passages appear here with their page number and the matched words highlighted."));
    return;
  }
  box.replaceChildren(...state.sources.map((s) => {
    const card = el("div", { class: "ev" + (s.weak ? " weak" : ""), id: `ev-${s.id}` },
      el("header", {}, el("span", { class: "tag" }, s.id),
        el("div", { class: "where" }, el("b", {}, `Page ${s.page}`), el("span", {}, s.doc_name)),
        el("span", { class: "badge" }, s.weak ? "closest match" : s.cited ? "cited" : "related")),
      el("div", { class: "txt" }, highlighted(s.text, s.highlights)));
    const more = el("button", { class: "more", type: "button" }, "Show full page");
    more.addEventListener("click", async () => {
      const open = card.querySelector(".page");
      if (open) { open.remove(); more.textContent = "Show full page"; return; }
      try {
        const p = await api(`/api/docs/${s.doc_id}/page/${s.page}`);
        card.append(el("div", { class: "page" }, p.text));
        more.textContent = "Hide full page";
      } catch (e) { banner(e.message); }
    });
    card.append(more);
    return card;
  }));
}

function focusSource(id, sources) {
  if (sources) showSources(sources);
  const card = $(`ev-${id}`);
  if (!card) return;
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  card.classList.add("flash");
  setTimeout(() => card.classList.remove("flash"), 1600);
}

/* ---------- asking ---------- */
function updateComposer() {
  const none = state.docs.length === 0;
  $("q").disabled = none || state.busy;
  $("send").disabled = none || state.busy;
  $("clear").disabled = none;
  $("q").placeholder = none ? "Upload a document to start asking questions" : "Ask a question about the selected document(s)";
}

async function ask(question) {
  question = (question || $("q").value).trim();
  if (!question || state.busy) return;
  if (!state.docs.length) { banner("Upload a document first, then ask a question."); return; }
  banner("");
  state.busy = true; updateComposer();
  const t = $("thread");
  t.querySelector(".hero")?.remove();
  $("brief").classList.add("compact");
  { const lk = $("brief").querySelector(".link"); if (lk) lk.textContent = "Show summary"; }
  t.append(el("div", { class: "msg user" }, question), el("div", { class: "msg bot typing", id: "typing" }, el("span", { class: "spin" }), "Searching the document…"));
  t.scrollTop = t.scrollHeight;
  $("q").value = ""; autosize();
  try {
    const r = await api("/api/chat", { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ scope: state.scope, question, mode: state.mode }) });
    $("typing").remove();
    t.append(botMessage({ content: r.answer, meta: r }));
    t.scrollTop = t.scrollHeight;
    state.sources = r.sources || [];
    renderEvidence();
  } catch (e) {
    $("typing")?.remove();
    banner(e.message);
  }
  state.busy = false; updateComposer(); $("q").focus();
}

function autosize() { const q = $("q"); q.style.height = "auto"; q.style.height = Math.min(q.scrollHeight, 140) + "px"; }

/* ---------- boot ---------- */
async function boot() {
  $("send").addEventListener("click", () => ask());
  $("q").addEventListener("input", autosize);
  $("q").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); } });
  $("clear").addEventListener("click", async () => {
    await api(`/api/history?scope=${encodeURIComponent(state.scope)}`, { method: "DELETE" }).catch((e) => banner(e.message));
    state.sources = []; renderEvidence(); renderThread([]);
  });
  const drop = $("drop");
  $("file").addEventListener("change", (e) => upload(e.target.files[0]));
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); } });
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => upload(e.dataTransfer.files[0]));
  try {
    state.status = await api("/api/status");
    pickInitialMode();
    renderMode();
    $("sysBtn").addEventListener("click", openSystem);
    $("sysClose").addEventListener("click", () => $("sysDialog").close());
    $("sysDialog").addEventListener("click", (e) => { if (e.target === $("sysDialog")) $("sysDialog").close(); });
    const p = state.status.providers;
    if (!p.anthropic.configured && !p.openai.configured) {
      banner("No API configured, so ASTRA INTEL is using Local mode: answers are sentences quoted from the document. Claude API and OpenAI-compatible show 'Not configured' until you add a key to .env.", "info");
    }
    await refreshDocs();
    await loadHistory();
  } catch (e) { banner(e.message); }
}
boot();
