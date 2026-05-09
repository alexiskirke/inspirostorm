const MAX_SELECT = 3;

const state = {
  results: [],
  selected: new Map(), // id -> source object
  generations: new Map(), // gen_id -> generation record
  pollTimer: null,
  avatarPollTimer: null,
  kbPollTimer: null,
};

const els = {
  results: document.getElementById("results"),
  emptyHint: document.getElementById("empty-hint"),
  resultsCount: document.getElementById("results-count"),
  selectedCount: document.getElementById("selected-count"),
  generateBtn: document.getElementById("generate-btn"),
  gallery: document.getElementById("gallery"),
  galleryEmpty: document.getElementById("gallery-empty"),
  defaultModel: document.getElementById("default-model"),
  defaultRatio: document.getElementById("default-ratio"),
};

// Where the React avatar app is served from. In dev it's :5173 (vite),
// in production it's the same origin as Scout via reverse proxy.
const CHAT_BASE_URL =
  (window.SCOUT_DEFAULTS && window.SCOUT_DEFAULTS.chat_base_url) ||
  "http://localhost:5173";

els.defaultModel.textContent = window.SCOUT_DEFAULTS.model;
els.defaultRatio.textContent = window.SCOUT_DEFAULTS.ratio;

// ----- tabs -----
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document
      .querySelectorAll(".tab")
      .forEach((t) => t.classList.toggle("tab--active", t === tab));
    document.querySelectorAll(".scan-form").forEach((f) => {
      f.classList.toggle("hidden", f.dataset.source !== tab.dataset.tab);
    });
  });
});

// ----- scan forms -----
function formToObj(form) {
  const fd = new FormData(form);
  const obj = {};
  for (const [k, v] of fd.entries()) {
    const trimmed = String(v).trim();
    if (!trimmed) continue;
    obj[k] = trimmed;
  }
  return obj;
}

async function handleScan(form) {
  const src = form.dataset.source;
  const obj = formToObj(form);
  if (obj.days) obj.days = Number(obj.days);
  if (obj.limit) obj.limit = Number(obj.limit);
  if (src === "arxiv" && obj.categories) {
    obj.categories = obj.categories
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
  }
  setScanning(form, true);
  try {
    const res = await fetch(`/api/scan/${src}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(obj),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      alert(`Scan failed: ${err.detail || res.statusText}`);
      return;
    }
    const data = await res.json();
    state.results = data.items;
    state.selected.clear();
    renderResults();
    renderSelectionBar();
  } finally {
    setScanning(form, false);
  }
}

function setScanning(form, on) {
  const btn = form.querySelector("button[type=submit]");
  if (!btn.dataset.label) btn.dataset.label = btn.textContent;
  btn.disabled = on;
  btn.textContent = on ? "Scanning…" : btn.dataset.label;
}

document.getElementById("form-github").addEventListener("submit", (e) => {
  e.preventDefault();
  handleScan(e.currentTarget);
});
document.getElementById("form-arxiv").addEventListener("submit", (e) => {
  e.preventDefault();
  handleScan(e.currentTarget);
});

// ----- result rendering -----
function renderResults() {
  els.resultsCount.textContent = state.results.length
    ? `(${state.results.length})`
    : "";
  els.emptyHint.style.display = state.results.length ? "none" : "block";
  els.results.innerHTML = "";

  for (const item of state.results) {
    const card = document.createElement("div");
    card.className = "result";
    card.dataset.id = item.id;
    card.setAttribute("role", "button");
    card.setAttribute("tabindex", "0");
    card.setAttribute("aria-label", `Select ${item.title}`);

    const title = document.createElement("div");
    title.className = "result__title";
    title.textContent = item.title;

    const sub = document.createElement("div");
    sub.className = "result__sub";
    sub.textContent = item.subtitle || "";

    const desc = document.createElement("div");
    desc.className = "result__desc";
    desc.textContent = truncate(item.description, 220);

    const meta = document.createElement("div");
    meta.className = "result__meta";
    if (item.source === "github") {
      const m = item.meta || {};
      if (m.language) meta.appendChild(tag(m.language, "tag--orange"));
      if (typeof m.stars === "number")
        meta.appendChild(tag(`★ ${m.stars}`, "tag--gray"));
      if (m.updated_at)
        meta.appendChild(tag(`upd ${m.updated_at.slice(0, 10)}`, "tag--gray"));
    } else if (item.source === "arxiv") {
      const m = item.meta || {};
      if (m.primary_category) meta.appendChild(tag(m.primary_category));
      if (m.published)
        meta.appendChild(tag(`pub ${m.published.slice(0, 10)}`, "tag--gray"));
    }

    if (item.url) {
      const link = document.createElement("a");
      link.href = item.url;
      link.target = "_blank";
      link.rel = "noreferrer";
      link.textContent = item.source === "github" ? "view repo →" : "arXiv →";
      link.addEventListener("click", (e) => e.stopPropagation());
      meta.appendChild(link);
    }

    const badge = document.createElement("div");
    badge.className = "checkbox-badge";
    badge.textContent = "";

    card.append(badge, title, sub, desc, meta);
    card.addEventListener("click", () => toggleSelect(item, card, badge));
    els.results.appendChild(card);
  }
  applySelectionStyling();
}

function tag(text, extraClass) {
  const el = document.createElement("span");
  el.className = `tag${extraClass ? " " + extraClass : ""}`;
  el.textContent = text;
  return el;
}
function truncate(s, n) {
  if (!s) return "";
  return s.length > n ? s.slice(0, n).trim() + "…" : s;
}

function toggleSelect(item, card, badge) {
  if (state.selected.has(item.id)) {
    state.selected.delete(item.id);
  } else {
    if (state.selected.size >= MAX_SELECT) return;
    state.selected.set(item.id, item);
  }
  applySelectionStyling();
  renderSelectionBar();
}

function applySelectionStyling() {
  document.querySelectorAll(".result").forEach((card, i) => {
    const id = card.dataset.id;
    const selectedIndex = [...state.selected.keys()].indexOf(id);
    const selected = selectedIndex !== -1;
    card.classList.toggle("selected", selected);
    const badge = card.querySelector(".checkbox-badge");
    badge.textContent = selected ? String(selectedIndex + 1) : "";
    const limitReached = state.selected.size >= MAX_SELECT && !selected;
    card.classList.toggle("disabled", limitReached);
  });
}

function renderSelectionBar() {
  els.selectedCount.textContent = `${state.selected.size} / ${MAX_SELECT} selected`;
  els.generateBtn.disabled = state.selected.size === 0;
}

// ----- generation -----
els.generateBtn.addEventListener("click", async () => {
  if (state.selected.size === 0) return;
  els.generateBtn.disabled = true;
  els.generateBtn.textContent = "Starting…";
  try {
    const items = [...state.selected.values()];
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ items }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      alert(`Generate failed: ${err.detail || res.statusText}`);
      return;
    }
    const data = await res.json();
    for (const g of data.generations) {
      state.generations.set(g.id, {
        id: g.id,
        source_id: g.source_id,
        prompt: g.prompt,
        status: "pending",
      });
    }
    renderGallery();
    schedulePoll();
  } finally {
    els.generateBtn.textContent = "Generate avatars";
    renderSelectionBar();
  }
});

function schedulePoll() {
  if (state.pollTimer) return;
  state.pollTimer = setInterval(pollGenerations, 2500);
  pollGenerations();
}

async function pollGenerations() {
  const pending = [...state.generations.values()].filter(
    (g) => g.status === "pending" || g.status === "running",
  );
  if (pending.length === 0) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
    return;
  }
  const ids = pending.map((g) => g.id).join(",");
  try {
    const res = await fetch(`/api/generations?ids=${encodeURIComponent(ids)}`);
    const data = await res.json();
    for (const g of data.items) {
      state.generations.set(g.id, g);
    }
    renderGallery();
  } catch (e) {
    console.warn("poll failed", e);
  }
}

async function loadGalleryHistory() {
  try {
    const res = await fetch("/api/generations?limit=24");
    const data = await res.json();
    for (const g of data.items) {
      if (!state.generations.has(g.id)) state.generations.set(g.id, g);
    }
    renderGallery();
    if (data.items.some((g) => g.status === "pending" || g.status === "running")) {
      schedulePoll();
    }
    if (data.items.some((g) => g.avatar_status === "creating")) {
      scheduleAvatarPoll();
    }
  } catch (e) {
    console.warn("history load failed", e);
  }
}

function renderGallery() {
  const items = [...state.generations.values()].sort((a, b) =>
    (b.created_at || "").localeCompare(a.created_at || ""),
  );
  els.galleryEmpty.style.display = items.length ? "none" : "block";
  els.gallery.innerHTML = "";
  for (const g of items) {
    const card = document.createElement("div");
    card.className = "gen";

    const wrap = document.createElement("div");
    wrap.className = "gen__image-wrap";
    if (g.image_url) {
      const img = document.createElement("img");
      img.src = g.image_url;
      img.alt = g.source_title || g.source_id;
      wrap.appendChild(img);
    } else {
      const status = document.createElement("div");
      status.className = "gen__status";
      const spin = document.createElement("div");
      spin.className = "spinner";
      status.append(spin, document.createTextNode(statusLabel(g.status)));
      wrap.appendChild(status);
    }

    const body = document.createElement("div");
    body.className = "gen__body";

    const src = document.createElement("div");
    src.className = "gen__source";
    src.textContent = g.source_title || g.source_id;

    const sub = document.createElement("div");
    sub.className = "gen__sub";
    const subParts = [g.source_type];
    if (g.source_meta?.subtitle) subParts.push(g.source_meta.subtitle);
    sub.textContent = subParts.filter(Boolean).join(" · ");

    const status = document.createElement("div");
    status.innerHTML = `<span class="badge badge--${g.status}">${statusLabel(g.status)}</span>`;

    const prompt = document.createElement("div");
    prompt.className = "gen__prompt";
    prompt.textContent = g.prompt || "";

    body.append(src, sub, status, prompt);

    if (g.failure_reason) {
      const err = document.createElement("div");
      err.className = "gen__error";
      err.textContent = g.failure_reason;
      body.appendChild(err);
    }

    if (g.character_name) {
      const persona = document.createElement("div");
      persona.className = "gen__persona";
      const voice = g.voice_preset ? ` · voice: ${g.voice_preset}` : "";
      persona.textContent = `${g.character_name}${voice}`;
      body.appendChild(persona);
    }

    if (g.status === "succeeded") {
      const actions = document.createElement("div");
      actions.className = "gen__actions";

      if (g.runway_avatar_id) {
        const chat = document.createElement("a");
        chat.className = "btn btn--chat";
        const params = new URLSearchParams({
          customAvatarId: g.runway_avatar_id,
          name: g.character_name || g.source_title || "Avatar",
        });
        chat.href = `${CHAT_BASE_URL}/?${params.toString()}`;
        chat.target = "_blank";
        chat.rel = "noreferrer";
        chat.textContent = "Chat live →";
        actions.appendChild(chat);

        actions.appendChild(renderKbBadge(g));
      } else if (g.avatar_status === "creating") {
        const status = document.createElement("span");
        status.className = "btn btn--ghost";
        status.innerHTML = `<span class="spinner spinner--sm"></span> Creating avatar…`;
        actions.appendChild(status);
      } else {
        const make = document.createElement("button");
        make.className = "btn btn--make";
        make.textContent = "Make this an avatar";
        make.addEventListener("click", () => createAvatar(g.id, make));
        actions.appendChild(make);
      }

      if (g.avatar_error) {
        const err = document.createElement("div");
        err.className = "gen__error";
        err.textContent = `Avatar: ${g.avatar_error}`;
        actions.appendChild(err);
      }

      body.appendChild(actions);
    }

    if (g.source_url) {
      const link = document.createElement("a");
      link.href = g.source_url;
      link.target = "_blank";
      link.rel = "noreferrer";
      link.textContent = "open source →";
      link.className = "gen__source-link";
      body.appendChild(link);
    }

    card.append(wrap, body);
    els.gallery.appendChild(card);
  }
}

function renderKbBadge(g) {
  const wrap = document.createElement("div");
  wrap.className = "kb";
  if (g.kb_status === "ready") {
    const docs = g.kb_doc_count || 0;
    const k = Math.round((g.kb_size_chars || 0) / 1000);
    wrap.innerHTML = `<span class="kb__badge kb__badge--ready">📚 KB · ${docs} doc${docs === 1 ? "" : "s"} · ${k}k chars</span>`;
    const refresh = document.createElement("button");
    refresh.className = "btn btn--ghost btn--sm";
    refresh.textContent = "rebuild";
    refresh.title = "Re-fetch the source and replace the avatar's attached docs";
    refresh.addEventListener("click", () => attachKnowledge(g.id, refresh));
    wrap.appendChild(refresh);
  } else if (g.kb_status === "building") {
    wrap.innerHTML = `<span class="kb__badge kb__badge--building"><span class="spinner spinner--sm"></span> Ingesting source…</span>`;
  } else if (g.kb_status === "failed") {
    wrap.innerHTML = `<span class="kb__badge kb__badge--failed">KB failed</span>`;
    const retry = document.createElement("button");
    retry.className = "btn btn--ghost btn--sm";
    retry.textContent = "retry";
    retry.addEventListener("click", () => attachKnowledge(g.id, retry));
    wrap.appendChild(retry);
    if (g.kb_error) {
      const err = document.createElement("div");
      err.className = "gen__error";
      err.textContent = g.kb_error;
      wrap.appendChild(err);
    }
  } else {
    const add = document.createElement("button");
    add.className = "btn btn--ghost btn--sm";
    add.textContent = "Add knowledge";
    add.title = "Ingest README + repo files (or PDF) and attach as Runway documents";
    add.addEventListener("click", () => attachKnowledge(g.id, add));
    wrap.appendChild(add);
  }
  return wrap;
}

async function attachKnowledge(genId, btn) {
  if (btn) {
    btn.disabled = true;
    btn.textContent = "Submitting…";
  }
  try {
    const res = await fetch(`/api/generations/${genId}/knowledge`, { method: "POST" });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      alert(`Knowledge build failed: ${err.detail || res.statusText}`);
      return;
    }
    const g = state.generations.get(genId);
    if (g) g.kb_status = "building";
    renderGallery();
    scheduleKbPoll();
  } catch (e) {
    alert(`Knowledge build failed: ${e}`);
  }
}

async function createAvatar(genId, btn) {
  btn.disabled = true;
  btn.textContent = "Submitting…";
  try {
    const res = await fetch(`/api/generations/${genId}/avatar`, { method: "POST" });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      alert(`Avatar creation failed: ${err.detail || res.statusText}`);
      btn.disabled = false;
      btn.textContent = "Make this an avatar";
      return;
    }
    // Optimistically reflect the new state, then poll until ready.
    const g = state.generations.get(genId);
    if (g) g.avatar_status = "creating";
    renderGallery();
    scheduleAvatarPoll();
  } catch (e) {
    alert(`Avatar creation failed: ${e}`);
    btn.disabled = false;
    btn.textContent = "Make this an avatar";
  }
}

function scheduleAvatarPoll() {
  if (state.avatarPollTimer) return;
  state.avatarPollTimer = setInterval(async () => {
    const pending = [...state.generations.values()].filter(
      (g) => g.avatar_status === "creating" || g.kb_status === "building",
    );
    if (pending.length === 0) {
      clearInterval(state.avatarPollTimer);
      state.avatarPollTimer = null;
      return;
    }
    const ids = pending.map((g) => g.id).join(",");
    try {
      const res = await fetch(`/api/generations?ids=${encodeURIComponent(ids)}`);
      const data = await res.json();
      for (const g of data.items) state.generations.set(g.id, g);
      renderGallery();
    } catch (e) {
      console.warn("avatar poll failed", e);
    }
  }, 3000);
}

function scheduleKbPoll() {
  // The avatar poll already covers kb_status === 'building' rows, so
  // make this an alias rather than a duplicate timer.
  scheduleAvatarPoll();
}

function statusLabel(s) {
  switch (s) {
    case "succeeded":
      return "ready";
    case "failed":
      return "failed";
    case "running":
      return "rendering…";
    default:
      return "queued…";
  }
}

loadGalleryHistory();
