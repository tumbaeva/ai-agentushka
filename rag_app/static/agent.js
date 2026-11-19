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
function esc(s) {
  return (s == null ? "" : String(s)).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
function jsonPost(body) {
  return { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
}

const FONT_PX = { S: "13px", M: "14px", L: "16px" };
const BTN_PAD = { small: "8px 16px", normal: "12px 24px", large: "15px 30px" };
let profile = window.__AGENT__ || {};
let promptTimer = null;

// ---------------------------------------------------------------- collect
function collect() {
  return {
    agent_name: $("agentName").value,
    subtitle: $("subtitle").value,
    logo_type: $("logoType").value,
    logo_emoji: $("logoEmoji").value,
    logo_initials: $("logoInitials").value,
    favicon_type: $("logoType").value,
    palette: {
      user_bg: $("palUserBg").value,
      user_text: $("palUserText").value,
      agent_bg: $("palAgentBg").value,
      agent_text: $("palAgentText").value,
      ui_bg: $("palUiBg").value,
      ui_field: $("palUiField").value,
      ui_text: $("palUiText").value,
      ui_subtitle: $("palUiSubtitle").value,
    },
    font_size: $("fontSize").value,
    button_size: $("buttonSize").value,
    radius: parseInt($("radius").value, 10),
    compact: $("compact").checked,
    sidebar_width: parseInt($("sidebarWidth").value, 10),
    welcome_message: $("welcomeMessage").value,
    starter_questions: $("starterQuestions").value.split("\n").map((s) => s.trim()).filter(Boolean),
    show_sources: $("showSources").checked,
    ui_language: $("uiLanguage").value,
    contact_phone: $("contactPhone").value,
    style_id: $("styleId").value,
    system_prompt: $("systemPrompt").value,
    temperature: parseFloat($("temperature").value),
    length: $("length").value,
  };
}

function logoGlyph(p) {
  if (p.logo_type === "file" && p.logo_file) return null; // rendered as <img>
  if (p.logo_type === "initials") return p.logo_initials || "AI";
  return p.logo_emoji || "🤖";
}

// ---------------------------------------------------------------- render
function applyTheme(p) {
  const root = document.documentElement;
  root.dataset.font = p.font_size;

  const pal = p.palette || {};
  const uiBg = pal.ui_bg, uiField = pal.ui_field, uiText = pal.ui_text;
  // the 7 agent colours
  root.style.setProperty("--user-bg", pal.user_bg);
  root.style.setProperty("--user-text", pal.user_text);
  root.style.setProperty("--agent-bg", pal.agent_bg);
  root.style.setProperty("--agent-text", pal.agent_text);
  root.style.setProperty("--ui-bg", uiBg);
  root.style.setProperty("--ui-field", uiField);
  root.style.setProperty("--ui-text", uiText);
  root.style.setProperty("--subtitle", pal.ui_subtitle || "#94a3b8");
  // map onto the legacy variables so shared selectors keep working
  root.style.setProperty("--bg", uiBg);
  root.style.setProperty("--panel", uiBg);
  root.style.setProperty("--input", uiField);
  root.style.setProperty("--text", uiText);
  root.style.setProperty("--accent", uiField);
  root.style.setProperty("--accent-2", uiText);
  root.style.setProperty("--heading", uiText);
  root.style.setProperty("--user-color", pal.user_bg);
  root.style.setProperty("--agent-color", pal.agent_bg);
  root.style.setProperty("--radius", p.radius + "px");
  root.style.setProperty("--font-size", FONT_PX[p.font_size] || "14px");
  root.style.setProperty("--sidebar-w", p.sidebar_width + "px");
  root.style.setProperty("--btn-pad", BTN_PAD[p.button_size] || BTN_PAD.normal);
  document.body.classList.toggle("compact", !!p.compact);
}

function renderBranding(p) {
  $("topName").textContent = p.agent_name || "AI-агент";
  $("prevName").textContent = p.agent_name || "AI-агент";
  $("topSubtitle").textContent = p.subtitle || "";
  $("prevSubtitle").textContent = p.subtitle || "";
  const glyph = logoGlyph(p);
  for (const id of ["topLogo", "prevLogo"]) {
    const node = $(id);
    if (glyph === null) {
      node.innerHTML = "";
      const img = document.createElement("img");
      img.src = "/api/agent/asset/" + encodeURIComponent(p.logo_file);
      img.alt = "logo";
      img.className = "logo-img";
      node.appendChild(img);
    } else {
      node.textContent = glyph;
    }
  }
}

function renderStarter(p) {
  const box = $("starterChips");
  box.innerHTML = "";
  (p.starter_questions || []).forEach((q) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = q;
    chip.onclick = () => { $("chatText").value = q; sendChat(); };
    box.appendChild(chip);
  });
}

function render() {
  const p = collect();
  profile = Object.assign({}, profile, p);
  applyTheme(p);
  renderBranding(p);
  renderStarter(p);
  $("radiusValue").textContent = p.radius;
  $("sidebarValue").textContent = p.sidebar_width;
  $("tempValue").textContent = p.temperature;
  refreshPrompt();
}

let promptEdited = false;

function refreshPrompt() {
  if (promptEdited) return;
  if (promptTimer) clearTimeout(promptTimer);
  promptTimer = setTimeout(async () => {
    if (promptEdited) return;
    const p = collect();
    try {
      const d = await api("/api/agent/prompt/preview", jsonPost({ style_id: p.style_id, length: p.length }));
      $("systemPrompt").value = d.system_prompt || "";
    } catch (e) { /* ignore */ }
  }, 250);
}

// ---------------------------------------------------------------- actions
async function saveProfile() {
  const btn = $("btnSave"); btn.disabled = true;
  try {
    const d = await api("/api/agent/profile", jsonPost(collect()));
    profile = d.profile;
    $("saveStatus").style.color = "var(--ok)";
    $("saveStatus").textContent = "Настройки сохранены.";
  } catch (e) {
    $("saveStatus").style.color = "var(--err)";
    $("saveStatus").textContent = "Ошибка: " + e.message;
  } finally { btn.disabled = false; }
}

async function exportApp() {
  const btn = $("btnExport"); btn.disabled = true;
  $("exportStatus").textContent = "Сборка архива…";
  try {
    const res = await fetch("/api/agent/export", jsonPost(collect()));
    if (!res.ok) throw new Error(res.statusText);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const cd = res.headers.get("content-disposition") || "";
    const m = /filename="?([^"]+)"?/.exec(cd);
    a.href = url;
    a.download = m ? m[1] : "agent.zip";
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
    $("exportStatus").style.color = "var(--ok)";
    $("exportStatus").textContent = "Архив собран и скачан.";
  } catch (e) {
    $("exportStatus").style.color = "var(--err)";
    $("exportStatus").textContent = "Ошибка: " + e.message;
  } finally { btn.disabled = false; }
}

async function uploadDoc(files) {
  const status = $("docStatus");
  for (const file of files) {
    const fd = new FormData();
    fd.append("file", file);
    status.textContent = "Загрузка: " + file.name + "…";
    try {
      const res = await fetch("/api/agent/document", { method: "POST", body: fd });
      const d = await res.json().catch(() => ({}));
      if (!res.ok) { alert("Ошибка: " + (d.error || res.statusText)); continue; }
      const rb = d.rebuilt || {};
      let msg = "Загружено: " + file.name;
      if (rb.chunks != null) msg += ` · чанков: ${rb.chunks}, документов: ${rb.documents}`;
      if (d.warning) msg += " · " + d.warning;
      status.textContent = msg;
      status.style.color = "var(--ok)";
    } catch (e) {
      status.textContent = "Ошибка: " + e.message;
      status.style.color = "var(--err)";
    }
  }
}

async function uploadAsset(file) {
  const fd = new FormData();
  fd.append("file", file);
  $("assetStatus").textContent = "Загрузка логотипа…";
  try {
    const res = await fetch("/api/agent/asset", { method: "POST", body: fd });
    const d = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(d.error || res.statusText);
    profile.logo_file = d.name;
    profile.favicon_file = d.name;
    $("logoType").value = "file";
    $("assetStatus").textContent = "Логотип: " + d.name;
    render();
  } catch (e) {
    $("assetStatus").style.color = "var(--err)";
    $("assetStatus").textContent = "Ошибка: " + e.message;
  }
}

function bubble(role, text, citations) {
  const wrap = document.createElement("div");
  wrap.className = "chat-bubble " + (role === "user" ? "user" : "bot");
  const body = document.createElement("div");
  body.innerHTML = esc(text).replace(/\n/g, "<br>");
  wrap.appendChild(body);
  if (citations && citations.length) {
    const det = document.createElement("details");
    const sum = document.createElement("summary");
    sum.textContent = "Источники (" + citations.length + ")";
    det.appendChild(sum);
    citations.forEach((c) => {
      const p = document.createElement("div");
      p.className = "cite";
      p.innerHTML = `<span class="file">${esc(c.label || c.file_name)}</span>`;
      det.appendChild(p);
    });
    wrap.appendChild(det);
  }
  $("chatMessages").appendChild(wrap);
  $("chatMessages").scrollTop = $("chatMessages").scrollHeight;
}

async function sendChat() {
  const text = $("chatText").value.trim();
  if (!text) return;
  $("chatText").value = "";
  bubble("user", text);
  const btn = $("btnSend"); btn.disabled = true;
  try {
    const d = await api("/api/agent/chat/message", jsonPost({ query: text }));
    bubble("assistant", d.answer || "(пусто)", d.show_sources ? d.citations : null);
  } catch (e) {
    bubble("assistant", "Ошибка: " + e.message);
  } finally { btn.disabled = false; }
}

// ---------------------------------------------------------------- palettes
let palettesState = [];
const PALETTE_INPUTS = {
  user_bg: "palUserBg", user_text: "palUserText",
  agent_bg: "palAgentBg", agent_text: "palAgentText",
  ui_bg: "palUiBg", ui_field: "palUiField", ui_text: "palUiText",
  ui_subtitle: "palUiSubtitle",
};

function fillPaletteSelect(items) {
  const sel = $("paletteList");
  if (!sel) return;
  sel.innerHTML = "";
  sel.appendChild(new Option("— сохранённые палитры —", ""));
  items.forEach((it) => sel.appendChild(new Option(it.name, it.name)));
}

async function loadPalettes() {
  try {
    const d = await api("/api/agent/palettes");
    palettesState = d.palettes || [];
    fillPaletteSelect(palettesState);
  } catch (e) { /* ignore */ }
}

function applyPaletteColors(colors) {
  Object.entries(PALETTE_INPUTS).forEach(([key, id]) => {
    if (colors && colors[key]) $(id).value = colors[key];
  });
  render();
}

async function savePalette() {
  const name = ($("paletteName").value || "").trim();
  if (!name) return alert("Введите название палитры");
  try {
    const d = await api("/api/agent/palettes", jsonPost({ name, palette: collect().palette }));
    palettesState = d.palettes || [];
    fillPaletteSelect(palettesState);
    $("paletteStatus").textContent = "Палитра сохранена: " + name;
  } catch (e) {
    $("paletteStatus").textContent = "Ошибка: " + e.message;
  }
}

function applySelectedPalette() {
  const name = $("paletteList").value;
  const item = palettesState.find((p) => p.name === name);
  if (!item) return;
  applyPaletteColors(item.colors);
  $("paletteName").value = name;
  $("paletteStatus").textContent = "Загружена палитра: " + name;
}

async function deleteSelectedPalette() {
  const name = $("paletteList").value;
  if (!name) return;
  if (!confirm('Удалить палитру «' + name + '»?')) return;
  try {
    const d = await api("/api/agent/palettes/" + encodeURIComponent(name), { method: "DELETE" });
    palettesState = d.palettes || [];
    fillPaletteSelect(palettesState);
    $("paletteStatus").textContent = "Палитра удалена: " + name;
  } catch (e) {
    $("paletteStatus").textContent = "Ошибка: " + e.message;
  }
}

// ---------------------------------------------------------------- init
function bind() {
  const inputs = document.querySelectorAll("input, select, textarea");
  inputs.forEach((n) => { if (n.id !== "systemPrompt") n.addEventListener("input", render); });
  $("logoType").addEventListener("change", () => { render(); });

  const sp = $("systemPrompt");
  sp.addEventListener("input", () => { promptEdited = true; });

  const bps = $("btnPaletteSave");
  if (bps) bps.onclick = savePalette;
  const pl = $("paletteList");
  if (pl) pl.onchange = applySelectedPalette;
  const bpd = $("btnPaletteDelete");
  if (bpd) bpd.onclick = deleteSelectedPalette;

  $("btnSave").onclick = saveProfile;
  $("btnExport").onclick = exportApp;
  $("btnSend").onclick = sendChat;
  $("chatText").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); }
  });

  const fi = $("docInput");
  if (fi) fi.addEventListener("change", (e) => uploadDoc(e.target.files));
  const dz = $("dropzone");
  if (dz) {
    dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
    dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
    dz.addEventListener("drop", (e) => { e.preventDefault(); dz.classList.remove("drag"); uploadDoc(e.dataTransfer.files); });
  }
  const lf = $("logoFile");
  if (lf) lf.addEventListener("change", (e) => { if (e.target.files[0]) uploadAsset(e.target.files[0]); });

  // Hide "back to dashboard" link in the standalone (chat-only) app.
  if (location.pathname === "/") { const bl = $("backLink"); if (bl) bl.style.display = "none"; }
}

bind();
if ((profile.system_prompt || "").trim()) {
  $("systemPrompt").value = profile.system_prompt;
  promptEdited = true;
}
render();
loadPalettes();
