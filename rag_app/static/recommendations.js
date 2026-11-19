"use strict";

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

const METRIC_COLS = ["hit@k", "doc_acc@k", "recall@k", "precision@k", "doc_precision@k", "mrr@k"];
let OPTIONS = null;

async function init() {
  OPTIONS = await api("/api/recommend/options");
  $("qCount").textContent = OPTIONS.questions;

  const emb = $("embOptions");
  emb.innerHTML = "";
  OPTIONS.embeddings.forEach((m) => {
    const lab = el("label", "opt");
    const cb = el("input"); cb.type = "checkbox"; cb.value = m.id;
    cb.checked = m.id === OPTIONS.current.embedding_id;
    lab.appendChild(cb);
    lab.appendChild(el("span", null, m.name));
    emb.appendChild(lab);
  });

  const chunks = $("chunkOptions");
  chunks.innerHTML = "";
  [400, 500, 700, 1000].forEach((cs) => {
    const lab = el("label", "opt");
    const cb = el("input"); cb.type = "checkbox"; cb.value = String(cs);
    cb.checked = cs === OPTIONS.current.chunk_size;
    lab.appendChild(cb);
    lab.appendChild(el("span", null, String(cs)));
    chunks.appendChild(lab);
  });

  const modes = $("modeOptions");
  modes.innerHTML = "";
  OPTIONS.modes.forEach((m) => {
    const lab = el("label", "opt");
    const cb = el("input"); cb.type = "checkbox"; cb.value = m; cb.checked = true;
    lab.appendChild(cb);
    lab.appendChild(el("span", null, m));
    modes.appendChild(lab);
  });

  const met = $("metric");
  met.innerHTML = "";
  OPTIONS.metrics.forEach((m) => met.appendChild(new Option(m, m)));
  met.value = "score_avg";

  $("btnRun").onclick = run;
  $("btnApply").onclick = apply;
  $("btnSaveSweep").onclick = saveSweep;

  try {
    const st = await api("/api/recommend/status");
    if (st.status === "running") poll();
    else if (st.status === "done" && st.result) render(st.result);
  } catch (e) { /* noop */ }
}

function checkedValues(containerId) {
  return Array.from(document.querySelectorAll("#" + containerId + " input:checked")).map((c) => c.value);
}

function collectOptions() {
  const chunkSizes = checkedValues("chunkOptions").map(Number);
  const custom = $("chunkCustom").value.split(",").map((s) => parseInt(s.trim(), 10)).filter((n) => n > 0);
  const temps = $("mqTemps").value.split(",").map((s) => parseFloat(s.trim())).filter((n) => !isNaN(n));
  return {
    embedding_ids: checkedValues("embOptions"),
    chunk_sizes: Array.from(new Set(chunkSizes.concat(custom))),
    modes: checkedValues("modeOptions"),
    metric: $("metric").value,
    temperature_mq: temps.length ? temps : null,
  };
}

async function run() {
  const opts = collectOptions();
  if (!opts.embedding_ids.length) return alert("Выберите хотя бы одну модель");
  if (!opts.chunk_sizes.length) return alert("Выберите хотя бы один размер чанка");
  if (!opts.modes.length) return alert("Выберите хотя бы один режим");
  $("btnRun").disabled = true;
  try {
    await api("/api/recommend/run", jsonPost(opts));
    poll();
  } catch (e) {
    $("sweepProgress").textContent = "Ошибка: " + e.message;
    $("btnRun").disabled = false;
  }
}

async function poll() {
  $("bestPanel").classList.add("hidden");
  $("tablePanel").classList.add("hidden");
  for (let i = 0; i < 3000; i++) {
    await new Promise((r) => setTimeout(r, 1500));
    let st;
    try { st = await api("/api/recommend/status"); } catch (e) { continue; }
    $("sweepProgress").textContent =
      (st.status === "running" ? "Идёт подбор… " : "") + (st.message || "") +
      (st.total ? ` (${st.done}/${st.total})` : "");
    if (st.status === "done") { $("btnRun").disabled = false; render(st.result); return; }
    if (st.status === "error") {
      $("sweepProgress").textContent = "Ошибка: " + (st.error || "?");
      $("btnRun").disabled = false;
      return;
    }
  }
}

function render(result) {
  $("sweepProgress").textContent = "Готово. Метрика ранжирования: " + result.metric;
  renderBest(result);
  renderTable(result);
}

function renderBest(result) {
  const panel = $("bestPanel");
  const box = $("bestContent");
  box.innerHTML = "";
  panel.classList.remove("hidden");
  const rec = result.recommended;
  if (!rec) { box.appendChild(el("p", "muted", "Не удалось получить рекомендацию.")); return; }

  const rows = [
    ["Embedding-модель", rec.embedding_model],
    ["Размер чанка / перекрытие", `${rec.chunk_size} / ${rec.chunk_overlap}`],
    ["Лучший режим поиска", rec.default_mode],
    ["Reranker", rec.enable_rerank ? "включён" : "выключен"],
    ["RRF k", rec.rrf_k],
    ["Вес вектора / BM25", `${rec.vector_weight} / ${rec.bm25_weight}`],
    ["Температура ответа", rec.default_temp],
    ["Температура multi-query", rec.default_temp_mq],
  ];
  const table = el("table", "results-table");
  table.innerHTML = "<thead><tr><th>Настройка</th><th>Значение</th></tr></thead>";
  const tbody = el("tbody");
  rows.forEach(([k, v]) => {
    const tr = el("tr");
    tr.innerHTML = `<td>${esc(k)}</td><td class="score">${esc(v)}</td>`;
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  box.appendChild(table);

  const m = result.best;
  const mTable = el("table", "results-table");
  mTable.innerHTML = "<thead><tr><th>Метрика</th><th>Значение</th></tr></thead>";
  const mt = el("tbody");
  ["score_avg"].concat(METRIC_COLS).forEach((c) => {
    const tr = el("tr");
    tr.innerHTML = `<td>${esc(c)}</td><td class="score">${esc(m[c])}</td>`;
    mt.appendChild(tr);
  });
  mTable.appendChild(mt);
  const h = el("h3", null, "Метрики лучшей конфигурации"); h.style.marginTop = "16px"; box.appendChild(h);
  box.appendChild(mTable);
}

function renderTable(result) {
  const panel = $("tablePanel");
  panel.classList.remove("hidden");
  const box = $("tableContent");
  box.innerHTML = "";
  const cols = ["embedding_id", "chunk_size", "mode", "temperature_mq"].concat(METRIC_COLS).concat(["score_avg"]);
  const table = el("table", "results-table");
  table.innerHTML = "<thead><tr>" + cols.map((c) => `<th>${esc(c)}</th>`).join("") + "</tr></thead>";
  const tbody = el("tbody");
  const best = result.best;
  result.configs.forEach((c) => {
    const tr = el("tr");
    if (best && c === best) tr.style.outline = "2px solid var(--ok)";
    tr.innerHTML = cols.map((k) => {
      let v = c[k];
      if (v == null) v = "—";
      const cls = METRIC_COLS.includes(k) || k === "score_avg" ? "score" : "";
      return `<td class="${cls}">${esc(v)}</td>`;
    }).join("");
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  box.appendChild(table);
}

async function apply() {
  const st = await api("/api/recommend/status");
  const rec = st.result && st.result.recommended;
  if (!rec) return alert("Сначала выполните подбор");
  if (!confirm("Применить рекомендуемые настройки и пересобрать индекс?")) return;
  $("btnApply").disabled = true;
  $("applyStatus").textContent = "Применяю и пересобираю индекс…";
  try {
    const d = await api("/api/recommend/apply", jsonPost({ recommended: rec }));
    $("applyStatus").textContent = `Готово: чанков ${d.chunks}, документов ${d.documents}.`;
  } catch (e) {
    $("applyStatus").textContent = "Ошибка: " + e.message;
  } finally {
    $("btnApply").disabled = false;
  }
}

async function saveSweep() {
  const st = await api("/api/recommend/status");
  if (!st.result) return alert("Сначала выполните подбор");
  $("btnSaveSweep").disabled = true;
  $("applyStatus").textContent = "Сохраняю…";
  try {
    const d = await api("/api/recommend/save", { method: "POST" });
    $("applyStatus").textContent = `Сохранено в data/${d.path} (${d.files.join(", ")})`;
  } catch (e) {
    $("applyStatus").textContent = "Ошибка: " + e.message;
  } finally {
    $("btnSaveSweep").disabled = false;
  }
}

init();
