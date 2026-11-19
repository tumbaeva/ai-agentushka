"use strict";

// ---------------------------------------------------------------- helpers
async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  const ct = res.headers.get("content-type") || "";
  const data = ct.includes("application/json") ? await res.json() : await res.text();
  if (!res.ok) throw new Error((data && data.error) || res.statusText);
  return data;
}
const $ = (id) => document.getElementById(id);
function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}
function esc(s) {
  return (s == null ? "" : String(s)).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
function jsonPost(body) {
  return { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
}
function highlight(snippet, ranges) {
  if (!ranges || !ranges.length) return esc(snippet);
  let out = "", last = 0;
  for (const [a, b] of ranges) {
    if (a < last) continue;
    out += esc(snippet.slice(last, a)) + "<mark>" + esc(snippet.slice(a, b)) + "</mark>";
    last = b;
  }
  return out + esc(snippet.slice(last));
}
function shortName(name, n = 28) {
  return name.length > n ? name.substring(0, n - 3) + "..." : name;
}
function normName(n) {
  return (n || "").replace(/\.[^.]+$/, "").trim().toLowerCase();
}
function normScores(items) {
  const vals = items.map((it) => Number(it.score) || 0);
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  return vals.map((v) => (max > min ? (v - min) / (max - min) : 1));
}
function scoreRange(items) {
  const vals = items.map((it) => Number(it.score) || 0);
  if (!vals.length) return "—";
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  return min.toFixed(4) + " … " + max.toFixed(4);
}
function modeRange(mode) {
  switch (mode) {
    case "vector": return "0 … 1";
    case "bm25": return "0 … ∞";
    case "hybrid":
    case "multi_query": return "≈ 0 … 0.1 (RRF)";
    case "multi_query_rerank": return "≈ −10 … +10 (логит)";
    default: return "—";
  }
}

// ---------------------------------------------------------------- tabs
document.querySelectorAll(".tabs .tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    if (!tab.dataset.tab) return;  // link-tab (e.g. Рекомендации) navigates away
    document.querySelectorAll(".tabs .tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".tabpane").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    $("pane-" + tab.dataset.tab).classList.add("active");
    if (tab.dataset.tab === "chat") loadSessions();
    if (tab.dataset.tab === "docs") loadDocs();
    if (tab.dataset.tab === "analytics") { maybeShowHeatmap(); loadQuestions(); }
  });
});

function maybeShowHeatmap() {
  const probe = new Image();
  probe.onload = () => {
    $("heatmapImg").src = "/heatmap?ts=" + Date.now();
    $("heatmapWrap").classList.remove("hidden");
  };
  probe.src = "/heatmap?ts=" + Date.now();
}

// ---------------------------------------------------------------- config
async function loadModels() {
  try {
    const m = await api("/api/models");
    const emb = $("embModel");
    emb.innerHTML = "";
    m.embedding.forEach((o) => emb.appendChild(new Option(o.name, o.id)));
    emb.value = m.current.embedding_id || "local";

    const llm = $("llmModel");
    llm.innerHTML = "";
    m.llm.forEach((o) => llm.appendChild(new Option(o.name, o.model)));
    llm.value = m.current.llm_model || m.llm[0].model;

    const rr = $("rerankModel");
    rr.innerHTML = "";
    m.rerank.forEach((o) => rr.appendChild(new Option(o.name, o.id)));
    rr.value = m.current.rerank_id || "mmarco";

    $("llmBaseUrl").value = m.current.llm_base_url || "";
    $("chunkSize").value = m.current.chunk_size;
    $("chunkOverlap").value = m.current.chunk_overlap;
    $("source").value = m.current.source || "kb_md";
    $("rrfK").value = m.current.rrf_k;
    $("vectorWeight").value = m.current.vector_weight;
    $("bm25Weight").value = m.current.bm25_weight;
    $("enableRerank").checked = !!m.current.enable_rerank;
    $("enableRewrite").checked = !!m.current.enable_query_rewrite;
    $("enableHyde").checked = !!m.current.enable_hyde;
    $("temp").value = m.current.default_temp;
    $("tempMq").value = m.current.default_temp_mq;
    $("tempValue").textContent = m.current.default_temp;
    $("tempMqValue").textContent = m.current.default_temp_mq;

    setPill(m);
  } catch (e) {
    $("statusPill").textContent = "ошибка конфигурации";
  }
}

function setPill(m) {
  const pill = $("statusPill");
  const chunks = m.index_chunks || 0;
  if (m.index_status === "building") {
    pill.textContent = "индекс: строится…";
    pill.style.color = "var(--warn)";
  } else if (m.index_status === "ready") {
    pill.textContent = `индекс готов · чанков: ${chunks}` + (m.rerank_available ? " · rerank ✓" : "");
    pill.style.color = "var(--ok)";
  } else if (m.index_status === "error") {
    pill.textContent = "индекс: ошибка";
    pill.style.color = "var(--err)";
  } else if (m.index_status === "loading") {
    pill.textContent = "индекс: загрузка…";
    pill.style.color = "var(--muted)";
  } else {
    pill.textContent = "индекс: пусто";
    pill.style.color = "var(--muted)";
  }
  const info = $("indexInfo");
  if (info) {
    if (m.index_status === "building") info.textContent = "Индекс строится — создаю чанки…";
    else if (m.index_status === "ready") info.textContent = `Чанков в индексе: ${chunks}`;
    else if (m.index_status === "error") info.textContent = "Индекс не построен";
    else info.textContent = "";
  }
}

async function pollIndexStatus() {
  for (let i = 0; i < 120; i++) {
    try {
      const m = await api("/api/models");
      setPill(m);
      if (m.index_status === "ready" || m.index_status === "error" || m.index_status === "empty") return;
    } catch (e) { /* server not ready yet */ }
    await new Promise((r) => setTimeout(r, 1500));
  }
}

async function saveConfig() {
  const body = {
    chunk_size: +$("chunkSize").value,
    chunk_overlap: +$("chunkOverlap").value,
    embedding_id: $("embModel").value,
    source: $("source").value,
    rrf_k: +$("rrfK").value,
    vector_weight: +$("vectorWeight").value,
    bm25_weight: +$("bm25Weight").value,
    rerank_id: $("rerankModel").value,
    enable_rerank: $("enableRerank").checked,
    enable_query_rewrite: $("enableRewrite").checked,
    enable_hyde: $("enableHyde").checked,
    llm_model: $("llmModel").value,
    llm_base_url: $("llmBaseUrl").value,
    default_temp: +$("temp").value,
    default_temp_mq: +$("tempMq").value,
  };
  await api("/api/config", jsonPost(body));
  await loadModels();
  alert("Настройки сохранены. Нажмите «Пересоздать чанки», если меняли embeddings или чанкинг.");
}

async function rebuild() {
  const btn = $("btnRebuild");
  btn.disabled = true;
  $("rebuildStatus").textContent = "Пересоздание чанков…";
  try {
    const d = await api("/api/rebuild", jsonPost({
      chunk_size: +$("chunkSize").value,
      chunk_overlap: +$("chunkOverlap").value,
      embedding_id: $("embModel").value,
      source: $("source").value,
    }));
    $("rebuildStatus").textContent = `Готово: чанков ${d.chunks}, документов ${d.documents}`;
    await loadModels();
    loadSnapshots();
  } catch (e) {
    $("rebuildStatus").textContent = "Ошибка: " + e.message;
  } finally {
    btn.disabled = false;
  }
}

async function saveModel() {
  try {
    const d = await api("/api/save_index", { method: "POST" });
    alert("Модель сохранена: " + d.path + " (чанков: " + d.meta.chunks_total + ")");
    loadSnapshots();
  } catch (e) { alert("Ошибка: " + e.message); }
}

async function loadSnapshots() {
  const box = $("snapshotsList");
  if (!box) return;
  let data;
  try { data = await api("/api/snapshots"); } catch (e) { return; }
  box.innerHTML = "";
  if (!data.snapshots.length) { box.appendChild(el("p", "muted", "Сохранённых моделей нет.")); return; }
  data.snapshots.forEach((s) => {
    const card = el("div", "snap" + (s.active ? " active" : ""));
    const head = el("div", "snap-head");
    head.appendChild(el("span", null, s.name));
    head.appendChild(el("span", "dim", `${s.chunks_total || 0} чанк.`));
    card.appendChild(head);
    card.appendChild(el("div", "snap-meta", `${s.embedding_model || "?"} · chunk ${s.chunk_size}/${s.chunk_overlap}`));
    const actions = el("div", "snap-actions");
    const apply = el("button", "btn-green", "Применить");
    apply.onclick = async () => {
      apply.disabled = true;
      try {
        await api("/api/snapshots/load", jsonPost({ name: s.name }));
        await loadModels(); await loadSnapshots();
      } catch (e) { alert("Ошибка: " + e.message); }
      finally { apply.disabled = false; }
    };
    const del = el("button", "danger", "Удалить");
    del.onclick = async () => {
      if (!confirm("Удалить снэпшот " + s.name + "?")) return;
      await api("/api/snapshots/" + encodeURIComponent(s.name), { method: "DELETE" });
      loadSnapshots();
    };
    actions.appendChild(apply); actions.appendChild(del);
    card.appendChild(actions);
    box.appendChild(card);
  });
}

async function saveSecrets() {
  const secrets = {};
  if ($("gigachatId").value) secrets.GIGACHAT_API_KEY_ID = $("gigachatId").value;
  if ($("gigachatSecret").value) secrets.GIGACHAT_API_KEY_SECRET = $("gigachatSecret").value;
  if ($("hfToken").value) secrets.HF_TOKEN = $("hfToken").value;
  await api("/api/secrets", jsonPost({ secrets }));
  ["gigachatId", "gigachatSecret", "hfToken"].forEach((i) => ($(i).value = ""));
  alert("Секреты сохранены.");
}

// ---------------------------------------------------------------- search
function resultsTable(results, withScore = true) {
  const norm = normScores(results);
  const range = scoreRange(results);
  const mr = modeRange(results.length ? results[0].mode : "");
  const table = el("table", "results-table");
  table.innerHTML = "<thead><tr><th>#</th><th>Документ (в knowledge_base.md)</th><th>Раздел</th><th>Score</th><th>Score (норм.)</th><th>min…max</th><th>Диапазон режима</th><th>Фрагмент</th></tr></thead>";
  const tbody = el("tbody");
  results.forEach((r, i) => {
    const tr = el("tr");
    tr.innerHTML =
      `<td>${i + 1}</td>` +
      `<td class="file"><a href="/api/docs/${encodeURIComponent(r.file_name)}" target="_blank" title="${esc(r.file_name)}">${esc(shortName(r.file_name))}</a></td>` +
      `<td>${esc(r.section || "")}</td>` +
      `<td class="score">${r.score}</td>` +
      `<td class="score">${norm[i].toFixed(3)}</td>` +
      `<td class="score">${range}</td>` +
      `<td class="score">${mr}</td>` +
      `<td class="preview">${esc((r.text || "").substring(0, 160))}</td>`;
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  return table;
}

async function doSearch() {
  const q = $("query").value.trim();
  if (!q) return alert("Введите вопрос");
  const panel = $("searchPanel");
  panel.classList.remove("hidden");
  const ap = $("answerPanel"); if (ap) ap.classList.add("hidden");
  const cp = $("comparePanel"); if (cp) cp.classList.add("hidden");
  $("searchContent").innerHTML = '<div class="loading">Поиск…</div>';
  try {
    const d = await api("/api/search", jsonPost({ query: q, mode: $("mode").value, k: +$("k").value }));
    const multi = d.mode === "multi_query" || d.mode === "multi_query_rerank";
    $("searchInfo").innerHTML = `Модель: <strong>${esc(d.embedding_model)}</strong> · режим: <strong>${esc(d.mode)}</strong> · k=${esc($("k").value)}` +
      (multi ? ` · температура переформулировки: <span class="warn">${esc(d.temperature_mq)}</span>` : "") +
      ` · <span class="dim">Score (норм.) — min-max внутри выдачи</span>`;
    $("searchContent").innerHTML = "";
    if (!d.results.length) { $("searchContent").appendChild(el("p", "muted", "Ничего не найдено.")); return; }
    $("searchContent").appendChild(resultsTable(d.results));
  } catch (e) { $("searchContent").innerHTML = `<p style="color:var(--err)">${esc(e.message)}</p>`; }
}

async function doAnswer() {
  const q = $("query").value.trim();
  if (!q) return alert("Введите вопрос");
  const panel = $("answerPanel");
  panel.classList.remove("hidden");
  const sp = $("searchPanel"); if (sp) sp.classList.add("hidden");
  const cp2 = $("comparePanel"); if (cp2) cp2.classList.add("hidden");
  $("answerContent").innerHTML = '<div class="loading">Генерация…</div>';
  try {
    const d = await api("/api/answer", jsonPost({
      query: q, mode: $("mode").value, k: +$("k").value,
      temperature: +$("temp").value, temperature_mq: +$("tempMq").value,
    }));
    const box = $("answerContent");
    const mode = d.mode || $("mode").value;
    const multi = mode === "multi_query" || mode === "multi_query_rerank";
    let strip = `<div class="info-strip">Модель: <strong>${esc(d.llm_model)}</strong> · режим: <strong>${esc(mode)}</strong> · температура ответа: <strong>${esc(d.temperature)}</strong>`;
    if (multi) strip += ` · температура переформулировки: <span class="warn">${esc(d.temperature_mq)}</span>`;
    strip += "</div>";
    box.innerHTML = strip + `<div class="answer-box">${esc(d.answer)}</div>`;
    if (d.citations && d.citations.length) {
      const h = el("h3", null, "Источники"); h.style.marginTop = "16px"; box.appendChild(h);
      const table = el("table", "results-table");
      table.innerHTML = "<thead><tr><th>#</th><th>Документ (в knowledge_base.md)</th><th>Раздел</th><th>Score</th><th>Score (норм.)</th><th>min…max</th><th>Диапазон режима</th><th>Фрагмент</th></tr></thead>";
      const tbody = el("tbody");
      const cNorm = normScores(d.citations);
      const cRange = scoreRange(d.citations);
      const cMode = modeRange(mode);
      d.citations.forEach((c, ci) => {
        const tr = el("tr");
        tr.innerHTML =
          `<td>${c.rank}</td>` +
          `<td class="file"><a href="/api/docs/${encodeURIComponent(c.file_name)}" target="_blank">${esc(shortName(c.file_name))}</a></td>` +
          `<td>${esc(c.section || "—")}</td>` +
          `<td class="score">${c.score}</td>` +
          `<td class="score">${cNorm[ci].toFixed(3)}</td>` +
          `<td class="score">${cRange}</td>` +
          `<td class="score">${cMode}</td>` +
          `<td class="preview">${highlight(c.snippet, c.highlights)}</td>`;
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);
      box.appendChild(table);
    }
  } catch (e) { $("answerContent").innerHTML = `<p style="color:var(--err)">${esc(e.message)}</p>`; }
}

async function doCompare() {
  const q = $("query").value.trim();
  if (!q) return alert("Введите вопрос");
  const panel = $("comparePanel");
  panel.classList.remove("hidden");
  const sp2 = $("searchPanel"); if (sp2) sp2.classList.add("hidden");
  const ap2 = $("answerPanel"); if (ap2) ap2.classList.add("hidden");
  const box = $("compareContent");
  box.innerHTML = '<div class="loading">Сравнение…</div>';
  try {
    const d = await api("/api/compare", jsonPost({ query: q, k: +$("k").value }));
    renderCompare(box, d);
  } catch (e) { box.innerHTML = `<p style="color:var(--err)">${esc(e.message)}</p>`; }
}

function renderCompare(box, d) {
  box.innerHTML = "";
  const tabs = el("div", "compare-tabs");
  const container = el("div");
  d.modes.forEach((m, idx) => {
    const tab = el("div", "tab" + (idx === 0 ? " active" : ""), m.mode);
    const block = el("div", "mode-block" + (idx === 0 ? " active" : ""));
    tab.onclick = () => {
      tabs.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      container.querySelectorAll(".mode-block").forEach((b) => b.classList.remove("active"));
      block.classList.add("active");
    };
    tabs.appendChild(tab);

    const tempLine = `температура ответа: <strong>${esc(m.temperature)}</strong>` +
      (m.temperature_mq != null
        ? ` · температура переформулировки: <span class="warn">${esc(m.temperature_mq)}</span>`
        : "");
    if (m.error) {
      block.innerHTML = `<h3>${esc(m.mode)}</h3><div class="info-strip">${tempLine}</div><p style="color:var(--err)">${esc(m.error)}</p>`;
    } else if (!m.results.length) {
      block.innerHTML = `<h3>${esc(m.mode)}</h3><div class="info-strip">${tempLine}</div><p class="muted">Ничего не найдено.</p>`;
    } else {
      const table = el("table", "results-table");
      table.innerHTML = "<thead><tr><th>#</th><th>Документ (в knowledge_base.md)</th><th>Раздел</th><th>Score</th><th>Score (норм.)</th><th>min…max</th><th>Диапазон режима</th><th>Фрагмент</th></tr></thead>";
      const tbody = el("tbody");
      const mNorm = normScores(m.results);
      const mRange = scoreRange(m.results);
      const mMode = modeRange(m.mode);
      m.results.forEach((r, ri) => {
        const tr = el("tr");
        tr.innerHTML =
          `<td>${r.rank}</td>` +
          `<td class="file"><a href="/api/docs/${encodeURIComponent(r.file_name)}" target="_blank">${esc(shortName(r.file_name))}</a></td>` +
          `<td>${esc(r.section || "")}</td><td class="score">${r.score}</td>` +
          `<td class="score">${mNorm[ri].toFixed(3)}</td>` +
          `<td class="score">${mRange}</td>` +
          `<td class="score">${mMode}</td>` +
          `<td class="preview">${esc(r.preview)}</td>`;
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);
      block.appendChild(el("h3", null, m.mode));
      const info = el("div", "info-strip");
      info.innerHTML = tempLine;
      block.appendChild(info);
      block.appendChild(table);
    }
    container.appendChild(block);
  });
  box.appendChild(tabs);
  box.appendChild(container);
}

async function analyticsCompare() {
  const q = $("cmpQuery").value.trim();
  if (!q) return alert("Введите вопрос");
  const box = $("cmpResult");
  box.innerHTML = '<div class="loading">Сравнение…</div>';
  try {
    const d = await api("/api/compare", jsonPost({ query: q, k: +$("k").value }));
    renderCompare(box, d);
  } catch (e) { box.innerHTML = `<p style="color:var(--err)">${esc(e.message)}</p>`; }
}

// ---------------------------------------------------------------- chat
let currentSession = null;

async function loadSessions() {
  const data = await api("/api/chat/sessions");
  const ul = $("sessionList");
  ul.innerHTML = "";
  data.sessions.forEach((s) => {
    const li = el("li", s.id === currentSession ? "active" : "");
    li.appendChild(el("span", null, s.title || "Диалог"));
    const del = el("span", "del", "✕");
    del.onclick = async (e) => {
      e.stopPropagation();
      await api("/api/chat/sessions/" + s.id, { method: "DELETE" });
      if (s.id === currentSession) { currentSession = null; $("chatMessages").innerHTML = ""; }
      loadSessions();
    };
    li.appendChild(del);
    li.onclick = () => openSession(s.id);
    ul.appendChild(li);
  });
}

async function openSession(id) {
  currentSession = id;
  const data = await api("/api/chat/sessions/" + id);
  const box = $("chatMessages");
  box.innerHTML = "";
  data.messages.forEach((m) => box.appendChild(renderMessage(m)));
  box.scrollTop = box.scrollHeight;
  loadSessions();
}

function renderMessage(m) {
  const node = el("div", "msg " + m.role);
  node.appendChild(el("div", null, m.text));
  if (m.citations && m.citations.length) {
    const details = el("details", "cites");
    const summary = el("summary", null, `Источники (${m.citations.length})`);
    details.appendChild(summary);
    m.citations.forEach((c) => {
      const div = el("div", "cite");
      div.innerHTML = `[${c.rank}] ${esc(c.file_name)} — <em>${esc(c.section || "—")}</em>`;
      details.appendChild(div);
    });
    node.appendChild(details);
  }
  return node;
}

async function sendChat() {
  const text = $("chatText").value.trim();
  if (!text) return;
  const box = $("chatMessages");
  box.appendChild(renderMessage({ role: "user", text }));
  $("chatText").value = "";
  box.scrollTop = box.scrollHeight;
  const btn = $("btnSend"); btn.disabled = true;
  try {
    const data = await api("/api/chat/message", jsonPost({
      query: text, session_id: currentSession, mode: $("mode").value, k: +$("k").value,
      temperature: +$("temp").value,
    }));
    currentSession = data.session_id;
    box.appendChild(renderMessage({ role: "assistant", text: data.answer, citations: data.citations }));
    box.scrollTop = box.scrollHeight;
    loadSessions();
  } catch (e) {
    box.appendChild(renderMessage({ role: "assistant", text: "Ошибка: " + e.message }));
  } finally { btn.disabled = false; }
}

// ---------------------------------------------------------------- analytics
function renderEvalSummary(summary) {
  const box = $("evalSummary");
  box.innerHTML = "";
  const cols = ["mode", "hit@k", "doc_acc@k", "recall@k", "precision@k", "doc_precision@k", "mrr@k"];
  const table = el("table", "results-table");
  table.innerHTML = "<thead><tr>" + cols.map((c) => `<th>${c}</th>`).join("") + "</tr></thead>";
  const tbody = el("tbody");
  Object.entries(summary).forEach(([mode, m]) => {
    const tr = el("tr");
    tr.innerHTML = `<td>${esc(mode)}</td>` + cols.slice(1).map((c) => `<td class="score">${m[c]}</td>`).join("");
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  box.appendChild(table);
}

function revealHeatmap() {
  $("heatmapImg").src = "/heatmap?ts=" + Date.now();
  $("heatmapWrap").classList.remove("hidden");
}

async function runEval() {
  const btn = $("btnEval"); btn.disabled = true; $("evalStatus").textContent = "Считаю…";
  try {
    const d = await api("/api/eval/run", jsonPost({ k: +$("k").value }));
    renderEvalSummary(d.summary);
    revealHeatmap();
    $("evalStatus").textContent = "готово";
  } catch (e) { $("evalStatus").textContent = "ошибка: " + e.message; }
  finally { btn.disabled = false; }
}

async function saveEval() {
  const btn = $("btnSaveEval"); btn.disabled = true; $("evalStatus").textContent = "Сохранение…";
  try {
    const q = $("query").value.trim();
    const d = await api("/api/eval/save", jsonPost({ k: +$("k").value, query: q }));
    renderEvalSummary(d.summary);
    revealHeatmap();
    $("evalStatus").textContent = "сохранено в data/" + d.path + " (" + d.files.length + " файлов)";
  } catch (e) { $("evalStatus").textContent = "ошибка: " + e.message; }
  finally { btn.disabled = false; }
}

// ---------------------------------------------------------------- eval questions
let evalQuestions = [];
let evalDocs = [];

async function loadQuestions() {
  const box = $("questionsList");
  if (!box) return;
  let data;
  try { data = await api("/api/eval/questions"); } catch (e) { return; }
  evalQuestions = data.questions || [];
  evalDocs = data.documents || [];
  renderQuestions();
}

function renderQuestions() {
  const box = $("questionsList");
  box.innerHTML = "";
  if (!evalQuestions.length) { box.appendChild(el("p", "muted", "Вопросов нет.")); populateDetailSelects(); return; }
  const table = el("table", "results-table q-table");
  table.innerHTML = "<thead><tr><th>Вопрос</th><th>Релевантные документы</th><th>Ключевые слова</th><th></th></tr></thead>";
  const tbody = el("tbody");
  evalQuestions.forEach((q, idx) => {
    const tr = el("tr");

    const tdQ = el("td");
    const ta = el("textarea"); ta.value = q.query || "";
    ta.oninput = () => { q.query = ta.value; };
    tdQ.appendChild(ta); tr.appendChild(tdQ);

    const tdD = el("td");
    const sel = el("select"); sel.multiple = true;
    sel.size = Math.min(5, Math.max(1, evalDocs.length));
    evalDocs.forEach((d) => {
      const o = new Option(d, d);
      o.selected = (q.relevant_docs || []).some((rd) => normName(rd) === normName(d));
      sel.appendChild(o);
    });
    sel.onchange = () => { q.relevant_docs = Array.from(sel.selectedOptions).map((o) => o.value); };
    tdD.appendChild(sel); tr.appendChild(tdD);

    const tdK = el("td");
    const ki = el("textarea"); ki.rows = 3;
    ki.value = (q.keywords || []).join(", ");
    ki.oninput = () => { q.keywords = ki.value.split(",").map((s) => s.trim()).filter(Boolean); };
    tdK.appendChild(ki); tr.appendChild(tdK);

    const tdA = el("td");
    const del = el("span", "q-del", "удалить");
    del.onclick = () => { evalQuestions.splice(idx, 1); renderQuestions(); };
    tdA.appendChild(del); tr.appendChild(tdA);

    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  box.appendChild(table);
  populateDetailSelects();
}

function addQuestion() {
  evalQuestions.push({ id: "Q" + (evalQuestions.length + 1), query: "", relevant_docs: [], keywords: [] });
  renderQuestions();
}

async function saveQuestions() {
  const status = $("questionsStatus");
  status.textContent = "Сохранение…";
  try {
    const d = await api("/api/eval/questions", jsonPost({ questions: evalQuestions }));
    status.textContent = `Сохранено вопросов: ${d.count}`;
    await loadQuestions();
  } catch (e) { status.textContent = "Ошибка: " + e.message; }
}

async function generateQuestions() {
  const status = $("questionsStatus");
  status.textContent = "Модель придумывает вопросы…";
  try {
    const d = await api("/api/eval/generate", jsonPost({ count: 5 }));
    const got = d.questions || [];
    got.forEach((q) => evalQuestions.push(q));
    renderQuestions();
    status.textContent = got.length ? `Добавлено вопросов: ${got.length}. Проверьте и сохраните.` : "Модель не вернула вопросы";
  } catch (e) { status.textContent = "Ошибка: " + e.message; }
}

function populateDetailSelects() {
  const qSel = $("detailQuestion");
  const mSel = $("detailMode");
  if (!qSel || !mSel) return;
  const qPrev = qSel.value;
  qSel.innerHTML = "";
  evalQuestions.forEach((q, i) => qSel.appendChild(new Option((q.query || "(без запроса)").slice(0, 60), String(i))));
  if (qPrev && +qPrev < qSel.options.length) qSel.value = qPrev;
  const mPrev = mSel.value;
  mSel.innerHTML = "";
  ["vector", "bm25", "hybrid", "multi_query", "multi_query_rerank"].forEach((m) => mSel.appendChild(new Option(m, m)));
  mSel.value = mPrev || "hybrid";
}

async function showDetail() {
  const box = $("detailResult");
  if (!evalQuestions.length) { box.innerHTML = ""; box.appendChild(el("p", "muted", "Сначала добавьте тестовые вопросы.")); return; }
  box.innerHTML = '<div class="loading">Поиск…</div>';
  try {
    const d = await api("/api/eval/detail", jsonPost({
      index: +$("detailQuestion").value, mode: $("detailMode").value, k: +$("k").value,
    }));
    box.innerHTML = "";
    const info = el("div", "info-strip");
    info.innerHTML = `Вопрос: <strong>${esc(d.question.query)}</strong> · режим: <strong>${esc(d.mode)}</strong> · k=${d.k}`;
    box.appendChild(info);
    if (!d.chunks.length) { box.appendChild(el("p", "muted", "Ничего не найдено.")); return; }
    const table = el("table", "results-table");
    table.innerHTML = "<thead><tr><th>#</th><th>Релевантен</th><th>Документ</th><th>Раздел</th><th>Score</th><th>Score (норм.)</th><th>min…max</th><th>Диапазон режима</th><th>Ключевые слова</th><th>Фрагмент</th></tr></thead>";
    const tbody = el("tbody");
    const dNorm = normScores(d.chunks);
    const dRange = scoreRange(d.chunks);
    const dMode = modeRange(d.mode);
    d.chunks.forEach((c, ci) => {
      const tr = el("tr");
      tr.innerHTML =
        `<td>${c.rank}</td>` +
        `<td style="text-align:center">${c.relevant ? "✅" : "❌"}</td>` +
        `<td class="file">${esc(shortName(c.file_name))}</td>` +
        `<td>${esc(c.section || "—")}</td>` +
        `<td class="score">${c.score}</td>` +
        `<td class="score">${dNorm[ci].toFixed(3)}</td>` +
        `<td class="score">${dRange}</td>` +
        `<td class="score">${dMode}</td>` +
        `<td>${esc((c.matched_keywords || []).join(", ") || "—")}</td>` +
        `<td class="preview">${highlight(c.snippet, c.highlights)}</td>`;
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    box.appendChild(table);
  } catch (e) { box.innerHTML = `<p style="color:var(--err)">${esc(e.message)}</p>`; }
}

// ---------------------------------------------------------------- docs
async function loadDocs() {
  const data = await api("/api/docs");
  const box = $("docsList");
  box.innerHTML = "";
  const files = data.files || [];

  const kbRow = el("div", "doc-row");
  kbRow.innerHTML = `<span class="file"><strong>${esc(data.kb_file || "knowledge_base.md")}</strong></span> ` +
    `<span class="badge">единый файл базы знаний</span> <span class="dim">${files.length} докум. внутри</span>`;
  box.appendChild(kbRow);

  if (!files.length) { box.appendChild(el("p", "muted", "Документы не найдены.")); return; }
  files.forEach((f) => {
    const row = el("div", "doc-row");
    const info = el("div");
    info.innerHTML = `<span class="file"><a href="/api/docs/${encodeURIComponent(f.name)}" target="_blank">${esc(f.name)}</a></span> ` +
      `<span class="badge">${esc(f.category)}</span> <span class="dim">${f.sections || 0} разд. · ${Math.round((f.chars || 0) / 1024)} КБ</span>`;
    const del = el("span", "del", "удалить");
    del.onclick = async () => {
      if (!confirm("Удалить документ «" + f.name + "» из базы знаний?")) return;
      await api("/api/documents/" + encodeURIComponent(f.name), { method: "DELETE" });
      loadDocs(); loadModels();
    };
    row.appendChild(info);
    // защита от случайного удаления единственного документа
    if (files.length > 1) row.appendChild(del);
    box.appendChild(row);
  });
  if (files.length === 1) {
    box.appendChild(el("p", "muted", "Единственный документ базы знаний удалить нельзя — залейте новый knowledge_base.md, чтобы заменить базу."));
  }
}

async function uploadFiles(files) {
  for (const file of files) {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("/api/documents", { method: "POST", body: fd });
    if (!res.ok) {
      const d = await res.json().catch(() => ({}));
      alert("Ошибка загрузки " + file.name + ": " + (d.error || res.statusText));
    }
  }
  loadDocs(); loadModels();
}

// ---------------------------------------------------------------- init
function on(id, fn) { const e = $(id); if (e) e.onclick = fn; return e; }

function bind() {
  on("btnSearch", doSearch);
  on("btnAnswer", doAnswer);
  on("btnCompare", doCompare);
  on("btnRebuild", rebuild);
  on("btnSaveModel", saveModel);
  on("btnSaveConfig", saveConfig);
  on("btnSaveSecrets", saveSecrets);
  on("btnEval", runEval);
  on("btnSaveEval", saveEval);
  on("btnAddQuestion", addQuestion);
  on("btnGenerate", generateQuestions);
  on("btnSaveQuestions", saveQuestions);
  on("btnDetail", showDetail);
  on("btnAnalyticsCompare", analyticsCompare);
  on("btnNewChat", async () => {
    const s = await api("/api/chat/sessions", jsonPost({}));
    openSession(s.id);
  });
  on("btnSend", sendChat);

  const t = $("temp");
  if (t) t.oninput = () => ($("tempValue").textContent = t.value);
  const tm = $("tempMq");
  if (tm) tm.oninput = () => ($("tempMqValue").textContent = tm.value);

  const dz = $("dropzone");
  if (dz) {
    dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
    dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
    dz.addEventListener("drop", (e) => {
      e.preventDefault(); dz.classList.remove("drag");
      uploadFiles(e.dataTransfer.files);
    });
  }
  const fi = $("fileInput");
  if (fi) fi.addEventListener("change", (e) => uploadFiles(e.target.files));

  document.querySelectorAll(".panel-toggle").forEach((node) => {
    node.onclick = () => { const tgt = $(node.dataset.target); if (tgt) tgt.classList.toggle("collapsed"); };
  });
}

async function init() {
  bind();
  try { await loadModels(); } catch (e) { /* noop */ }
  pollIndexStatus();
  try {
    const modes = await api("/api/modes");
    const sel = $("mode");
    if (sel) {
      sel.innerHTML = "";
      modes.modes.forEach((m) => sel.appendChild(new Option(m, m)));
      sel.value = "hybrid";
    }
  } catch (e) { /* noop */ }
  await Promise.allSettled([loadDocs(), loadSnapshots(), loadQuestions()]);
}

init();
