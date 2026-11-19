"use strict";

// Floating tooltip for any element with [data-tip].
// Rendered on <body> with position:fixed so it is never clipped by
// scrollable panels (sidebar / tab panes).
(function () {
  const tip = document.createElement("div");
  tip.className = "tooltip hidden";
  document.body.appendChild(tip);
  let current = null;

  function place(el) {
    const r = el.getBoundingClientRect();
    const w = tip.offsetWidth;
    const h = tip.offsetHeight;
    let left = r.left + r.width / 2 - w / 2;
    left = Math.max(8, Math.min(left, window.innerWidth - w - 8));
    let top = r.top - h - 8;
    if (top < 8) top = r.bottom + 8;
    tip.style.left = left + "px";
    tip.style.top = top + "px";
  }

  function show(el) {
    const text = el.getAttribute("data-tip");
    if (!text) return;
    current = el;
    tip.textContent = text;
    tip.classList.remove("hidden");
    place(el);
  }

  function hide() {
    current = null;
    tip.classList.add("hidden");
  }

  document.addEventListener("mouseover", (e) => {
    const el = e.target.closest && e.target.closest("[data-tip]");
    if (el && el !== current) show(el);
  });
  document.addEventListener("mouseout", (e) => {
    const el = e.target.closest && e.target.closest("[data-tip]");
    if (el) hide();
  });
  document.addEventListener("focusin", (e) => {
    const el = e.target.closest && e.target.closest("[data-tip]");
    if (el) show(el);
  });
  document.addEventListener("focusout", hide);
  document.addEventListener("scroll", hide, true);
  window.addEventListener("resize", hide);
})();
