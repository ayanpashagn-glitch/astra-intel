"use strict";
const $ = id => document.getElementById(id);
let docs=[], scope="all", sources=[], mode="local";

async function api(path, opts={}) {
  const r=await fetch(path,opts); let b={}; try{b=await r.json()}catch{}
  if(!r.ok) throw new Error(b.error||("Request failed: "+r.status)); return b;
}
function msg(text, cls=""){ const d=document.createElement("div"); d.className="msg "+cls; d.textContent=text; return d; }
function renderEvidence(){
  const box=$("evidence"); box.innerHTML="";
  if(!sources.length){box.textContent="Source passages appear here.";return}
  for(const s of sources){const c=document.createElement("article");c.className="ev";const h=document.createElement("h3");h.textContent=`${s.id} · ${s.doc_name} · page ${s.page}`;const p=document.createElement("p");p.textContent=s.text;c.append(h,p);box.append(c)}
}
async function refresh(){
  docs=await api("/api/docs"); const l=$("docList"); l.innerHTML="";
  if(docs.length){const a=document.createElement("button");a.textContent="All documents";a.onclick=()=>setScope("all");l.append(a)}
  for(const d of docs){const b=document.createElement("button");b.textContent=`${d.name} · ${d.pages} pages`;b.onclick=()=>setScope(String(d.id));l.append(b)}
  $("q").disabled=!docs.length; $("send").disabled=!docs.length;
}
async function setScope(s){scope=s;sources=[];renderEvidence();await history()}
async function history(){const t=$("thread");t.innerHTML="";if(!docs.length){t.append(msg("Upload a PDF to begin.","bot"));return}const h=await api("/api/history?scope="+encodeURIComponent(scope));for(const m of h)t.append(msg(m.content,m.role==="user"?"user":"bot"))}
async function upload(f){if(!f)return;const fd=new FormData();fd.append("file",f);fd.append("mode",mode);$("uploadState").hidden=false;$("uploadState").textContent="Processing "+f.name+"…";try{const r=await api("/api/docs",{method:"POST",body:fd});await refresh();await setScope(String(r.id));$("uploadState").hidden=true}catch(e){$("uploadState").textContent=e.message}}
async function ask(){const q=$("q").value.trim();if(!q)return;const t=$("thread");t.append(msg(q,"user"));$("q").value="";try{const r=await api("/api/chat",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({question:q,scope,mode})});t.append(msg(r.answer,"bot"));sources=r.sources||[];renderEvidence()}catch(e){t.append(msg(e.message,"bot error"))}}
async function boot(){
  $("file").onchange=e=>upload(e.target.files[0]);$("send").onclick=ask;$("q").onkeydown=e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();ask()}};
  $("clear").onclick=async()=>{await api("/api/history?scope="+encodeURIComponent(scope),{method:"DELETE"});await history()};
  try{const st=await api("/api/status");mode=st.default_mode||"local";await refresh();await history()}catch(e){$("banner").hidden=false;$("banner").textContent=e.message}
}
boot();