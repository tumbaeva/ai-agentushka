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
function kb(size) {
  if (size < 1024) return size + " Б";
  return Math.round(size / 1024) + " КБ";
}

let filesState = [];
let draftDirty = false;

function markDirty(dirty) {
  draftDirty = dirty;
  const st = $("editStatus");
  if (!st) return;
  st.textContent = dirty ? "есть несохранённые правки" : "";
  st.style.color = dirty ? "var(--warn)" : "";
}

// ---------------------------------------------------------------- files
async function loadFiles() {
  const box = $("filesList");
  let data;
  try { data = await api("/api/kb/files"); } catch (e) {
    box.innerHTML = `<p style="color:var(--err)">${esc(e.message)}</p>`; return;
  }
  filesState = data.files || [];
  if (data.llm_available === false) {
    $("useLlm").disabled = true;
    $("useLlm").parentElement.title = "LLM недоступен: укажите ключ GigaChat в настройках.";
  }
  if (data.ocr_available === false) {
    $("useOcr").disabled = true;
    $("useOcr").parentElement.title = "OCR недоступен: установите rapidocr-onnxruntime и pypdfium2.";
  }
  renderFiles();
}

function renderFiles() {
  const box = $("filesList");
  box.innerHTML = "";
  if (!filesState.length) {
    box.appendChild(el("p", "muted", "Пока нет документов — загрузите .docx / .pdf / .txt / .md в папку source_docs/."));
    return;
  }
  filesState.forEach((f, idx) => {
    const row = el("div", "doc-row");

    const head = el("div");
    head.style.display = "flex";
    head.style.alignItems = "center";
    head.style.gap = "8px";

    const cb = el("input");
    cb.type = "checkbox"; cb.checked = !!f.selected;
    cb.onchange = () => updateSelection(cb.checked, f.name);
    head.appendChild(cb);

    const info = el("span");
    const textInfo = f.empty
      ? `<span class="badge" style="background:var(--warn);color:#1e293b">текст не извлечён</span>`
      : `<span class="dim">текст: ${f.chars} симв.</span>`;
    info.innerHTML = `<span class="file">${esc(f.name)}</span> ` +
      `<span class="badge">${esc(f.ext)}</span> <span class="dim">${kb(f.size)}</span> ${textInfo}`;
    head.appendChild(info);
    row.appendChild(head);

    const actions = el("div");
    actions.style.display = "flex";
    actions.style.gap = "6px";

    const up = el("button", null, "↑");
    up.title = "Выше";
    up.disabled = idx === 0;
    up.onclick = () => move(idx, -1);
    actions.appendChild(up);

    const down = el("button", null, "↓");
    down.title = "Ниже";
    down.disabled = idx === filesState.length - 1;
    down.onclick = () => move(idx, 1);
    actions.appendChild(down);

    const del = el("span", "del", "удалить");
    del.onclick = () => removeFile(f.name);
    actions.appendChild(del);

    row.appendChild(actions);
    box.appendChild(row);
  });
  $("buildStatus").textContent = `Выбрано: ${filesState.filter((f) => f.selected).length} из ${filesState.length}`;
}

async function updateSelection(checked, name) {
  filesState = filesState.map((f) => f.name === name ? { ...f, selected: checked } : f);
  renderFiles();
  await api("/api/kb/selection", jsonPost({
    selected: filesState.filter((f) => f.selected).map((f) => f.name),
  }));
}

async function move(idx, delta) {
  const target = idx + delta;
  if (target < 0 || target >= filesState.length) return;
  const arr = filesState.slice();
  [arr[idx], arr[target]] = [arr[target], arr[idx]];
  filesState = arr;
  renderFiles();
  await api("/api/kb/order", jsonPost({ order: filesState.map((f) => f.name) }));
}

async function removeFile(name) {
  if (!confirm(`Удалить файл «${name}» из source_docs/?`)) return;
  try {
    const d = await api("/api/kb/files/" + encodeURIComponent(name), { method: "DELETE" });
    filesState = d.files || [];
    renderFiles();
  } catch (e) { alert("Ошибка: " + e.message); }
}

async function uploadFiles(files) {
  const status = $("uploadStatus");
  for (const file of files) {
    const fd = new FormData();
    fd.append("file", file);
    status.textContent = "Загрузка: " + file.name + "…";
    try {
      const res = await fetch("/api/kb/upload", { method: "POST", body: fd });
      const d = await res.json().catch(() => ({}));
      if (!res.ok) alert("Ошибка загрузки " + file.name + ": " + (d.error || res.statusText));
    } catch (e) {
      alert("Ошибка загрузки " + file.name + ": " + e.message);
    }
  }
  status.textContent = "";
  loadFiles();
}

// ---------------------------------------------------------------- build
async function preview() {
  const btn = $("btnPreview"); btn.disabled = true;
  $("buildStatus").textContent = "Сборка…";
  try {
    const d = await api("/api/kb/preview", jsonPost({ use_llm: $("useLlm").checked, use_ocr: $("useOcr").checked, use_headings: $("useHeadings").checked }));
    $("previewText").value = d.text || "";
    markDirty(false);
    const llmNote = d.llm_requested
      ? (d.llm_used ? " · LLM: применён" : " · LLM: недоступен, текст как есть")
      : "";
    const ocrNote = d.ocr_requested
      ? (d.ocr_used ? " · OCR: применён" : " · OCR: не потребовался/недоступен")
      : "";
    const headNote = d.headings_requested
      ? (d.headings_used ? " · разделы: расставлены" : " · разделы: не найдены")
      : "";
    let html = `Собрано документов: <strong>${d.selected}</strong> · ` +
      `символов: <strong>${d.chars}</strong>${llmNote}${ocrNote}${headNote}`;
    if (d.warnings && d.warnings.length) {
      html += `<br><span style="color:var(--warn)">⚠ ${d.warnings.map(esc).join("<br>⚠ ")}</span>`;
    }
    $("previewInfo").innerHTML = html;
    $("buildStatus").textContent = "Черновик собран. Можно переносить.";
  } catch (e) {
    $("buildStatus").textContent = "Ошибка: " + e.message;
  } finally { btn.disabled = false; }
}

async function saveDraft() {
  const btn = $("btnSaveDraft"); btn.disabled = true;
  try {
    const d = await api("/api/kb/save_draft", jsonPost({ text: $("previewText").value }));
    markDirty(false);
    $("editStatus").textContent = `Сохранено · символов: ${d.chars}`;
    $("editStatus").style.color = "var(--ok)";
    $("buildStatus").textContent = "Правки сохранены в черновик.";
  } catch (e) {
    $("editStatus").textContent = "Ошибка: " + e.message;
  } finally { btn.disabled = false; }
}

async function promote() {
  const text = $("previewText").value;
  if (!text.trim()) return alert("Сначала нажмите «Собрать и предпросмотреть».");
  if (!confirm("Перенести собранный knowledge_base.md в docs/?")) return;
  const btn = $("btnPromote"); btn.disabled = true;
  $("buildStatus").textContent = "Перенос…";
  try {
    await api("/api/kb/promote", jsonPost({ text }));
    markDirty(false);
    $("buildStatus").textContent =
      "Файл перенесён в docs/knowledge_base.md. Пересоберите индекс на главной — «Пересоздать чанки».";
  } catch (e) {
    $("buildStatus").textContent = "Ошибка: " + e.message;
  } finally { btn.disabled = false; }
}

// ---------------------------------------------------------------- init
function bind() {
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

  $("btnPreview").onclick = preview;
  $("btnPromote").onclick = promote;
  $("btnSaveDraft").onclick = saveDraft;
  $("btnDownload").onclick = () => { window.location = "/api/kb/download"; };

  const pt = $("previewText");
  if (pt) pt.addEventListener("input", () => markDirty(true));
}

bind();
loadFiles();
