"use strict";

// Standalone chat-only agent: applies the saved branding/theme and talks to
// the agent chat endpoint. No settings / prompt UI.

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

const P = window.__AGENT__ || {};
const FONT_PX = { S: "13px", M: "14px", L: "16px" };
const BTN_PAD = { small: "8px 16px", normal: "12px 24px", large: "15px 30px" };

function applyTheme() {
  const root = document.documentElement;
  const pal = P.palette || {};
  const uiBg = pal.ui_bg || "#0f172a", uiField = pal.ui_field || "#0ea5e9", uiText = pal.ui_text || "#e2e8f0";
  // the 7 agent colours
  root.style.setProperty("--user-bg", pal.user_bg || "#0ea5e9");
  root.style.setProperty("--user-text", pal.user_text || "#ffffff");
  root.style.setProperty("--agent-bg", pal.agent_bg || "#7c3aed");
  root.style.setProperty("--agent-text", pal.agent_text || "#ffffff");
  root.style.setProperty("--ui-bg", uiBg);
  root.style.setProperty("--ui-field", uiField);
  root.style.setProperty("--ui-text", uiText);
  root.style.setProperty("--subtitle", pal.ui_subtitle || "#94a3b8");
  // legacy mapping so shared selectors keep working
  root.style.setProperty("--bg", uiBg);
  root.style.setProperty("--panel", uiBg);
  root.style.setProperty("--input", uiField);
  root.style.setProperty("--text", uiText);
  root.style.setProperty("--accent", uiField);
  root.style.setProperty("--accent-2", uiText);
  root.style.setProperty("--heading", uiText);
  root.style.setProperty("--user-color", pal.user_bg || "#0ea5e9");
  root.style.setProperty("--agent-color", pal.agent_bg || "#7c3aed");
  root.style.setProperty("--font-size", FONT_PX[P.font_size] || "14px");
  root.style.setProperty("--radius", (P.radius || 12) + "px");
  root.style.setProperty("--btn-pad", BTN_PAD[P.button_size] || BTN_PAD.normal);
  document.body.classList.toggle("compact", !!P.compact);
}

function renderLogo() {
  const node = $("agentLogo");
  if (P.logo_type === "file" && P.logo_file) {
    node.innerHTML = "";
    const img = document.createElement("img");
    img.src = "/api/agent/asset/" + encodeURIComponent(P.logo_file);
    img.alt = "logo";
    img.className = "logo-img";
    node.appendChild(img);
  } else if (P.logo_type === "initials") {
    node.textContent = P.logo_initials || "AI";
  } else {
    node.textContent = P.logo_emoji || "🤖";
  }
}

function renderStarter() {
  const box = $("starterChips");
  box.innerHTML = "";
  (P.starter_questions || []).forEach((q) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = q;
    chip.onclick = () => { $("chatText").value = q; send(); };
    box.appendChild(chip);
  });
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

async function send() {
  const text = $("chatText").value.trim();
  if (!text) return;
  $("chatText").value = "";
  bubble("user", text);
  const btn = $("btnSend");
  btn.disabled = true;
  try {
    const d = await api("/api/agent/chat/message", jsonPost({ query: text }));
    bubble("assistant", d.answer || "(пусто)", d.show_sources ? d.citations : null);
  } catch (e) {
    bubble("assistant", "Ошибка: " + e.message);
  } finally {
    btn.disabled = false;
  }
}

function bind() {
  $("btnSend").onclick = send;
  $("chatText").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });
}

let ready = false;

async function pollStatus() {
  const box = $("indexStatus");
  try {
    const s = await api("/api/agent/status");
    if (s.status === "ready" && s.chunks > 0) {
      ready = true;
      box.style.display = "none";     // no service info in the UI
      $("btnSend").disabled = false;
      $("chatText").disabled = false;
      return;
    }
    if (s.status === "empty" || s.status === "error") {
      // index won't be ready: allow chatting so the fallback (with contacts) works
      box.style.display = "none";
      $("btnSend").disabled = false;
      $("chatText").disabled = false;
      return;
    }
    box.className = "index-status";
    box.textContent = "Загрузка…";
  } catch (e) {
    box.textContent = "Загрузка…";
  }
  if (!ready) setTimeout(pollStatus, 1500);
}

applyTheme();
renderLogo();
renderStarter();
if (P.welcome_message) bubble("assistant", P.welcome_message);
bind();
$("btnSend").disabled = true;
$("chatText").disabled = true;
pollStatus();
