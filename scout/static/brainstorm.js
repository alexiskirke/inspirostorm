/**
 * Scout Brainstorms page — pair two custom avatars, dispatch into a Zoom,
 * and watch their conversation get summarised + synthesised.
 *
 * Backend endpoints used:
 *   GET  /api/generations?limit=200          — populate the avatar dropdowns
 *   GET  /api/brainstorm/threads             — list all pair threads
 *   POST /api/brainstorm/start               — create-or-resume thread + start session
 *   GET  /api/brainstorm/threads/:id         — full thread state (sessions + memory)
 *   POST /api/brainstorm/sessions/:id/end    — end a live session (auto-summarises)
 *   POST /api/brainstorm/threads/:id/synthesise — gpt-5.5 deep synthesis
 *   GET  /api/brainstorm/threads/:id/synthesis  — list past syntheses
 *   GET  /api/sessions/:id/transcript            — meet server live transcript
 */

const els = {
  formNew:        document.getElementById("form-new"),
  newError:       document.getElementById("new-error"),
  threads:        document.getElementById("threads"),
  threadsEmpty:   document.getElementById("threads-empty"),
  panel:          document.getElementById("thread-panel"),
  title:          document.getElementById("thread-title"),
  meta:           document.getElementById("thread-meta"),
  memory:         document.getElementById("thread-memory"),
  sessions:       document.getElementById("thread-sessions"),
  synth:          document.getElementById("thread-synth"),
  refreshBtn:     document.getElementById("thread-refresh"),
  synthesiseBtn:  document.getElementById("thread-synthesise"),
};

const MEET_BASE_URL = "http://localhost:3000";

const state = {
  avatars:    [],          // generations with runway_avatar_id
  threads:    [],
  selectedThreadId: null,
  livePollers: new Map(),  // sessionId -> intervalHandle (transcript polling)
  moviePollTimer: null,
};

// ----- bootstrap -----
async function init() {
  await Promise.all([loadAvatars(), loadThreads()]);
  // If URL has ?thread=xxx, open that one immediately.
  const qs = new URLSearchParams(location.search);
  const t = qs.get("thread");
  if (t) selectThread(t);
}

// ----- avatars (the dropdowns) -----
async function loadAvatars() {
  const res = await fetch("/api/generations?limit=200");
  const data = await res.json();
  state.avatars = (data.items || []).filter(
    (g) => g.runway_avatar_id && g.character_name
  );
  renderAvatarOptions();
}

function renderAvatarOptions() {
  for (const name of ["avatar_a_gen_id", "avatar_b_gen_id"]) {
    const sel = els.formNew.querySelector(`select[name=${name}]`);
    sel.innerHTML = '<option value="">— pick —</option>';
    for (const a of state.avatars) {
      const o = document.createElement("option");
      o.value = a.id;
      o.textContent = `${a.character_name} (${a.source_title || a.source_type})`;
      sel.appendChild(o);
    }
  }
}

// ----- threads list -----
async function loadThreads() {
  const res = await fetch("/api/brainstorm/threads");
  const data = await res.json();
  state.threads = data.items || [];
  renderThreads();
}

function renderThreads() {
  els.threads.innerHTML = "";
  els.threadsEmpty.style.display = state.threads.length ? "none" : "block";
  for (const t of state.threads) {
    const card = document.createElement("div");
    card.className = "thread-card";
    if (t.id === state.selectedThreadId) card.classList.add("active");
    card.innerHTML = `
      <div class="thread-card__pair">
        ${avatarThumb(t.avatar_a_image, t.avatar_a_name)}
        <div class="thread-card__vs">×</div>
        ${avatarThumb(t.avatar_b_image, t.avatar_b_name)}
      </div>
      <div class="thread-card__name">
        ${escapeHtml(t.avatar_a_name)} <span class="muted">×</span> ${escapeHtml(t.avatar_b_name)}
      </div>
      <div class="thread-card__meta">
        ${t.session_count || 0} session${t.session_count === 1 ? "" : "s"}
        ${t.last_session_at ? ` · last ${t.last_session_at.slice(0, 16).replace("T", " ")}` : ""}
      </div>
      ${t.topic_seed ? `<div class="thread-card__meta">Topic: ${escapeHtml(t.topic_seed)}</div>` : ""}
    `;
    card.addEventListener("click", () => selectThread(t.id));
    els.threads.appendChild(card);
  }
}

function avatarThumb(image_path, name) {
  if (image_path) {
    return `<img class="thread-card__avatar" src="/data/images/${image_path}" alt="${escapeHtml(name)}" />`;
  }
  // Initial fallback if no image
  const initial = (name || "?").trim().charAt(0).toUpperCase();
  return `<div class="thread-card__avatar" style="display:flex;align-items:center;justify-content:center;font-weight:700;color:var(--muted);">${initial}</div>`;
}

// ----- thread page -----
async function selectThread(threadId) {
  state.selectedThreadId = threadId;
  // Reflect in URL so refresh keeps you here.
  const url = new URL(location.href);
  url.searchParams.set("thread", threadId);
  history.replaceState(null, "", url);
  renderThreads();              // re-render to show .active highlight
  await loadThreadDetails();
}

async function loadThreadDetails() {
  if (!state.selectedThreadId) return;
  const [tRes, sRes] = await Promise.all([
    fetch(`/api/brainstorm/threads/${state.selectedThreadId}`),
    fetch(`/api/brainstorm/threads/${state.selectedThreadId}/synthesis`),
  ]);
  const t = await tRes.json();
  const s = await sRes.json();
  els.panel.classList.remove("hidden");
  els.title.textContent =
    `${nameOf(t.thread.avatar_a_gen_id)} × ${nameOf(t.thread.avatar_b_gen_id)}`;
  els.meta.innerHTML = [
    `pair_key: <code>${t.thread.pair_key}</code>`,
    t.thread.topic_seed ? `topic: ${escapeHtml(t.thread.topic_seed)}` : null,
    `created: ${t.thread.created_at?.slice(0, 19).replace("T", " ")}`,
  ].filter(Boolean).join(" · ");

  els.memory.textContent = t.state?.rolling_summary?.trim() ||
    "(no sessions summarised yet — start a session, end it, and gpt-5.1 will write the memory)";

  renderSessions(t.sessions || []);
  renderSyntheses(s.items || []);
}

function nameOf(genId) {
  const a = state.avatars.find((x) => x.id === genId);
  return a ? a.character_name : genId.slice(0, 8);
}

function renderSessions(sessions) {
  els.sessions.innerHTML = "";
  if (sessions.length === 0) {
    els.sessions.innerHTML = '<em class="muted">No sessions yet. Start one with the form at the top.</em>';
    return;
  }
  // Add a "start new session" button at the top of the list (if no live one).
  const live = sessions.find((s) => s.status === "live");
  const startBar = document.createElement("div");
  startBar.className = "controls";
  startBar.style.marginBottom = "10px";
  if (!live) {
    const startBtn = document.createElement("button");
    startBtn.className = "btn btn--make";
    startBtn.textContent = "Start new session in this thread";
    startBtn.addEventListener("click", () => continueThread());
    startBar.appendChild(startBtn);
  } else {
    const endBtn = document.createElement("button");
    endBtn.className = "btn btn--ghost";
    endBtn.textContent = "End live session (and trigger summary)";
    endBtn.addEventListener("click", () => endSession(live.id));
    startBar.appendChild(endBtn);
  }
  els.sessions.appendChild(startBar);

  for (const s of sessions) {
    const card = document.createElement("div");
    card.className = "session-card";
    const head = document.createElement("div");
    head.className = "session-card__head";
    head.innerHTML = `
      <div>
        <div class="session-card__title">
          ${s.id.slice(0, 8)} · <span class="tag tag--${s.status}">${s.status}</span>
        </div>
        <div class="session-card__meta">
          started ${s.started_at?.slice(0, 19).replace("T", " ")}
          ${s.ended_at ? ` · ended ${s.ended_at.slice(0, 19).replace("T", " ")}` : ""}
          ${s.topic ? ` · topic: ${escapeHtml(s.topic)}` : ""}
        </div>
      </div>
    `;
    card.appendChild(head);

    if (s.status === "live") {
      const tBox = document.createElement("div");
      tBox.className = "session-card__transcript";
      tBox.innerHTML = "<em>connecting transcript stream…</em>";
      card.appendChild(tBox);
      // Poll transcripts from the meet server for both slots.
      pollLiveTranscript(s, tBox);
    } else {
      // Show the recorded transcripts side by side (collapsed-ish).
      const both = renderRecordedTranscript(s);
      if (both) card.appendChild(both);
    }
    els.sessions.appendChild(card);
  }
}

function renderRecordedTranscript(sess) {
  // Backend returns transcript_a / transcript_b (decoded) where the JSON
  // suffix has been stripped by brainstorm._row_to_dict, OR it can leave
  // them as transcript_a_json strings — handle both.
  const a = decodeMaybe(sess.transcript_a ?? sess.transcript_a_json);
  const b = decodeMaybe(sess.transcript_b ?? sess.transcript_b_json);
  if ((!a || !a.length) && (!b || !b.length)) return null;
  const merged = mergeAndFormat(a, b);
  if (!merged) return null;
  const tBox = document.createElement("div");
  tBox.className = "session-card__transcript";
  tBox.textContent = merged;
  return tBox;
}

function decodeMaybe(v) {
  if (!v) return [];
  if (Array.isArray(v)) return v;
  if (typeof v === "string") {
    try { return JSON.parse(v) || []; } catch { return []; }
  }
  return [];
}

function mergeAndFormat(a, b) {
  // Just concatenate per-slot since each is already chronological for
  // its own avatar; we tag them so user can tell who said what.
  const lines = [];
  for (const seg of a) {
    if (seg && seg.final !== false && seg.text) {
      lines.push(`[A:${seg.participantIdentity || "?"}] ${seg.text}`);
    }
  }
  for (const seg of b) {
    if (seg && seg.final !== false && seg.text) {
      lines.push(`[B:${seg.participantIdentity || "?"}] ${seg.text}`);
    }
  }
  return lines.join("\n");
}

function pollLiveTranscript(sess, container) {
  const sids = [sess.meet_session_id_a, sess.meet_session_id_b].filter(Boolean);
  if (sids.length === 0) {
    container.innerHTML = "<em class='muted'>no meet sessionId on this row</em>";
    return;
  }
  const fetchOnce = async () => {
    const lines = [];
    for (const [i, sid] of sids.entries()) {
      try {
        const r = await fetch(`${MEET_BASE_URL}/api/sessions/${sid}/transcript`);
        if (!r.ok) continue;
        const d = await r.json();
        for (const e of (d.entries || [])) {
          if (!e || e.final === false || !e.text) continue;
          lines.push(`[${i === 0 ? "A" : "B"}:${e.participantIdentity || "?"}] ${e.text}`);
        }
      } catch (err) {
        // meet server not reachable — show one warning line, keep polling
      }
    }
    container.textContent = lines.length
      ? lines.join("\n")
      : "(no transcript yet — say something)";
  };
  // Stop any previous interval for this session
  if (state.livePollers.has(sess.id)) clearInterval(state.livePollers.get(sess.id));
  fetchOnce();
  const handle = setInterval(fetchOnce, 2500);
  state.livePollers.set(sess.id, handle);
}

function renderSyntheses(items) {
  els.synth.innerHTML = "";
  if (items.length === 0) {
    els.synth.innerHTML = '<em class="muted">No syntheses yet. Click <b>Synthesise (gpt-5.5)</b> after a few sessions to see what they came up with.</em>';
    return;
  }
  for (const s of items) {
    const card = document.createElement("div");
    card.className = "synth-card";
    card.innerHTML = `
      <div class="synth-card__pitch">🎬 ${escapeHtml(s.movie_pitch || "")}</div>
      <div class="synth-card__text">${escapeHtml(s.text_md || "")}</div>
      <div class="synth-card__meta">model: ${s.model_used || "?"} · ${s.created_at?.slice(0, 19).replace("T", " ")}</div>
    `;
    if (s.ideas && s.ideas.length) {
      const grid = document.createElement("div");
      grid.className = "synth-card__ideas";
      for (const idea of s.ideas) {
        const i = document.createElement("div");
        i.className = "synth-card__idea";
        i.innerHTML = `<b>${escapeHtml(idea.headline || "")}</b><br>${escapeHtml(idea.detail || "")}<br><em>${escapeHtml(idea.next_move || idea.raised_by || "")}</em>`;
        grid.appendChild(i);
      }
      card.appendChild(grid);
    }
    card.appendChild(renderMovieBlock(s));
    els.synth.appendChild(card);
  }
}

function renderMovieBlock(s) {
  const wrap = document.createElement("div");
  wrap.className = "synth-card__movie";
  const status = (s.movie_status || "idle").toLowerCase();

  if (status === "ready" && s.movie_path) {
    const v = document.createElement("video");
    v.className = "synth-card__video";
    v.controls = true;
    v.preload = "metadata";
    v.src = `/data/movies/${encodeURIComponent(s.movie_path)}`;
    wrap.appendChild(v);

    const meta = document.createElement("div");
    meta.className = "synth-card__movie-meta";
    meta.innerHTML = `🎞 movie ready · model: ${escapeHtml(s.movie_model || "?")} · `;
    const regen = document.createElement("button");
    regen.className = "btn btn--ghost btn--sm";
    regen.textContent = "regenerate";
    regen.title = "Re-render the movie (~400 credits, ~4 minutes)";
    regen.addEventListener("click", () => triggerMovie(s.id, regen));
    meta.appendChild(regen);
    wrap.appendChild(meta);
  } else if (status === "building") {
    const pill = document.createElement("div");
    pill.className = "synth-card__movie-pill synth-card__movie-pill--building";
    pill.innerHTML = `<span class="spinner spinner--sm"></span> building 24-sec movie · ~3-4 minutes (composites + 3 Veo clips + music + ffmpeg)`;
    wrap.appendChild(pill);
    // Auto-refresh while building.
    scheduleMoviePoll();
  } else if (status === "failed") {
    const pill = document.createElement("div");
    pill.className = "synth-card__movie-pill synth-card__movie-pill--failed";
    pill.textContent = `movie failed: ${s.movie_error || "unknown"}`;
    wrap.appendChild(pill);
    const retry = document.createElement("button");
    retry.className = "btn btn--ghost btn--sm";
    retry.textContent = "retry";
    retry.addEventListener("click", () => triggerMovie(s.id, retry));
    wrap.appendChild(retry);
  } else {
    // idle — offer button
    const btn = document.createElement("button");
    btn.className = "btn btn--make";
    btn.textContent = "🎬 Generate 24-sec movie (~400 credits)";
    btn.title = "3 Veo clips × 8s + composites + sound_effect underscore + ffmpeg mix.";
    btn.addEventListener("click", () => triggerMovie(s.id, btn));
    wrap.appendChild(btn);
  }
  return wrap;
}

async function triggerMovie(synthId, btn) {
  if (btn) {
    btn.disabled = true;
    btn.textContent = "Submitting…";
  }
  try {
    const res = await fetch(`/api/brainstorm/synthesis/${synthId}/movie`, {
      method: "POST",
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      alert(`movie request failed: ${err.detail || res.statusText}`);
      return;
    }
    // Optimistically refresh the thread page so the building pill shows up.
    setTimeout(loadThreadDetails, 1000);
  } catch (e) {
    alert(`movie request failed: ${e}`);
  }
}

function scheduleMoviePoll() {
  if (state.moviePollTimer) return;
  state.moviePollTimer = setInterval(async () => {
    if (!state.selectedThreadId) {
      clearInterval(state.moviePollTimer);
      state.moviePollTimer = null;
      return;
    }
    const r = await fetch(`/api/brainstorm/threads/${state.selectedThreadId}/synthesis`).catch(() => null);
    if (!r || !r.ok) return;
    const data = await r.json();
    const stillBuilding = (data.items || []).some((s) => s.movie_status === "building");
    renderSyntheses(data.items || []);
    if (!stillBuilding) {
      clearInterval(state.moviePollTimer);
      state.moviePollTimer = null;
    }
  }, 6000);
}

// ----- actions -----
els.formNew.addEventListener("submit", async (e) => {
  e.preventDefault();
  els.newError.classList.add("hidden");
  const fd = new FormData(els.formNew);
  const body = {
    avatar_a_gen_id: fd.get("avatar_a_gen_id"),
    avatar_b_gen_id: fd.get("avatar_b_gen_id"),
    topic: (fd.get("topic") || "").toString().trim() || null,
    meeting_url: (fd.get("meeting_url") || "").toString().trim() || null,
  };
  if (!body.avatar_a_gen_id || !body.avatar_b_gen_id) {
    els.newError.textContent = "Pick two avatars.";
    els.newError.classList.remove("hidden");
    return;
  }
  if (body.avatar_a_gen_id === body.avatar_b_gen_id) {
    els.newError.textContent = "Avatars must be different.";
    els.newError.classList.remove("hidden");
    return;
  }
  const btn = els.formNew.querySelector("button[type=submit]");
  btn.disabled = true; btn.textContent = "Starting…";
  try {
    const res = await fetch("/api/brainstorm/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      els.newError.textContent = err.detail || res.statusText;
      els.newError.classList.remove("hidden");
      return;
    }
    const data = await res.json();
    await loadThreads();
    selectThread(data.thread.id);
  } finally {
    btn.disabled = false; btn.textContent = "Start brainstorm";
  }
});

async function continueThread() {
  if (!state.selectedThreadId) return;
  const t = state.threads.find((x) => x.id === state.selectedThreadId);
  if (!t) return;
  const meeting = prompt(
    "Zoom URL for this session (leave blank to use the server default):",
    "",
  );
  const topic = prompt(
    "Topic for THIS session (optional — leave blank to use the thread's seed):",
    "",
  );
  const body = {
    avatar_a_gen_id: t.avatar_a_gen_id,
    avatar_b_gen_id: t.avatar_b_gen_id,
    topic: topic ? topic.trim() : null,
    meeting_url: meeting ? meeting.trim() : null,
  };
  const res = await fetch("/api/brainstorm/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    alert(`start failed: ${err.detail || res.statusText}`);
    return;
  }
  await loadThreadDetails();
}

async function endSession(sessionId) {
  if (state.livePollers.has(sessionId)) {
    clearInterval(state.livePollers.get(sessionId));
    state.livePollers.delete(sessionId);
  }
  const res = await fetch(
    `/api/brainstorm/sessions/${sessionId}/end?reason=manual`,
    { method: "POST" },
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    alert(`end failed: ${err.detail || res.statusText}`);
    return;
  }
  // Wait a moment for the bg summariser to write state, then refresh.
  setTimeout(loadThreadDetails, 4000);
  loadThreadDetails();
}

els.refreshBtn.addEventListener("click", () => loadThreadDetails());

els.synthesiseBtn.addEventListener("click", async () => {
  if (!state.selectedThreadId) return;
  els.synthesiseBtn.disabled = true;
  els.synthesiseBtn.textContent = "Synthesising…";
  try {
    const res = await fetch(
      `/api/brainstorm/threads/${state.selectedThreadId}/synthesise`,
      { method: "POST" },
    );
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      alert(`synthesis failed: ${err.detail || res.statusText}`);
      return;
    }
    await loadThreadDetails();
  } finally {
    els.synthesiseBtn.disabled = false;
    els.synthesiseBtn.textContent = "Synthesise (gpt-5.5)";
  }
});

// ----- helpers -----
function escapeHtml(s) {
  return String(s || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

init();
