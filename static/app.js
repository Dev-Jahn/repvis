"use strict";

const $ = (id) => document.getElementById(id);

const els = {
  dropzone: $("dropzone"), fileInput: $("file-input"),
  chips: $("source-chips"), chipsHint: $("chips-hint"),
  modelSelect: $("model-select"), modelNote: $("model-note"), gpuBadge: $("gpu-badge"),
  removebg: $("opt-removebg"), l2: $("opt-l2"),
  maxside: $("opt-maxside"), maxframes: $("opt-maxframes"), fps: $("opt-fps"),
  btnRun: $("btn-run"),
  runStatus: $("run-status"), progFill: $("prog-fill"), progMsg: $("prog-msg"),
  matrix: $("matrix"), matrixEmpty: $("matrix-empty"),
  transport: $("transport"),
  btnPlay: $("btn-play"), seek: $("seek"), timeCur: $("time-cur"), timeDur: $("time-dur"),
  speed: $("speed"), btnLoop: $("btn-loop"), btnClear: $("btn-clear"),
  filterDefs: $("filters").querySelector("defs"),
};

// ------------------------------------------------------------ workspace state
let models = [];
const sources = new Map();        // sourceId -> {id, name, size, selected}
const cols = [];                  // ordered model keys (matrix columns)
const rows = [];                  // ordered sourceIds (matrix rows)
const origCells = new Map();      // sourceId -> {cell, video}
const pcaCells = new Map();       // `${sid}|${model}` -> {cell, video, filter, runId, perm, invert}
const runMeta = new Map();        // runId -> {sid, model, pcaUrl}
let masterWired = false;
let currentEv = null;             // active run's EventSource
let running = false;              // a run group is in flight

// ------------------------------------------------------------- model list
async function loadModels() {
  try {
    const r = await fetch("/api/models");
    const data = await r.json();
    models = data.models;
    els.gpuBadge.textContent = `${data.gpus} GPU${data.gpus === 1 ? "" : "s"} ready`;
    els.modelSelect.innerHTML = "";
    for (const m of models) {
      const opt = document.createElement("option");
      opt.value = m.key;
      opt.textContent = m.available ? m.label : `${m.label} (unavailable)`;
      opt.disabled = !m.available;
      els.modelSelect.appendChild(opt);
    }
    const firstAvail = models.find((m) => m.available);
    if (firstAvail) els.modelSelect.value = firstAvail.key;
    updateModelNote();
  } catch {
    els.gpuBadge.textContent = "server offline";
  }
}
function updateModelNote() {
  const m = models.find((x) => x.key === els.modelSelect.value);
  if (!m) { els.modelNote.textContent = ""; return; }
  els.modelNote.textContent = `${m.note} · patch ${m.patch} · ≤ ${m.max_side}px`;
  els.maxside.value = m.max_side;
}
els.modelSelect.addEventListener("change", updateModelNote);
const modelLabel = (key) => (models.find((m) => m.key === key) || {}).label || key;

// ------------------------------------------------------------- sources
async function loadSources() {
  try {
    const data = await (await fetch("/api/sources")).json();
    for (const s of data.sources) addSourceChip(s);
  } catch { /* ignore */ }
}

function addSourceChip(rec) {
  if (sources.has(rec.id)) return;
  sources.set(rec.id, { ...rec, selected: false });
  els.chipsHint.classList.add("hidden");
  const chip = document.createElement("button");
  chip.className = "chip";
  chip.dataset.sid = rec.id;
  chip.innerHTML = `<span class="chk-box"></span><span class="chip-name"></span>` +
                   `<span class="chip-size"></span>`;
  chip.querySelector(".chip-name").textContent = rec.name;
  chip.querySelector(".chip-size").textContent =
    rec.size ? `${(rec.size / 1048576).toFixed(0)}MB` : "";
  chip.addEventListener("click", () => toggleSource(rec.id, chip));
  els.chips.appendChild(chip);
}
function toggleSource(sid, chip) {
  const s = sources.get(sid);
  s.selected = !s.selected;
  chip.classList.toggle("selected", s.selected);
  updateRunButton();
}
function selectedSources() {
  return [...sources.values()].filter((s) => s.selected).map((s) => s.id);
}
function updateRunButton() {
  const n = selectedSources().length;
  els.btnRun.disabled = running || n === 0;
  els.btnRun.textContent = running ? "Running…"
    : n === 0 ? "Select source(s) to run"
    : n === 1 ? "Run ▸" : `Run · joint PCA over ${n} videos ▸`;
}

// ---- upload ----
els.dropzone.addEventListener("click", () => els.fileInput.click());
els.fileInput.addEventListener("change", (e) => uploadFiles(e.target.files));
["dragenter", "dragover"].forEach((ev) =>
  els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) =>
  els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.remove("drag"); }));
els.dropzone.addEventListener("drop", (e) => uploadFiles(e.dataTransfer.files));

async function uploadFiles(fileList) {
  const files = [...fileList].filter((f) => f.type.startsWith("video/") || /\.(mp4|mov|webm|mkv|avi|m4v)$/i.test(f.name));
  for (const file of files) {
    const tmp = addUploadingChip(file.name);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const rec = await (await fetch("/api/sources", { method: "POST", body: fd })).json();
      tmp.remove();
      if (sources.has(rec.id)) {                 // dedup: already in tray
        const ex = els.chips.querySelector(`.chip[data-sid="${rec.id}"]`);
        if (ex) { ex.classList.add("flash"); setTimeout(() => ex.classList.remove("flash"), 600); }
      } else {
        addSourceChip(rec);
      }
    } catch (e) {
      tmp.querySelector(".chip-name").textContent = `${file.name} — upload failed`;
      tmp.classList.add("error");
    }
  }
}
function addUploadingChip(name) {
  els.chipsHint.classList.add("hidden");
  const chip = document.createElement("div");
  chip.className = "chip uploading";
  chip.innerHTML = `<span class="spin"></span><span class="chip-name"></span>`;
  chip.querySelector(".chip-name").textContent = name;
  els.chips.appendChild(chip);
  return chip;
}

// ------------------------------------------------------------- run
els.btnRun.addEventListener("click", startRun);

async function startRun() {
  const sel = selectedSources();
  if (!sel.length || running) return;
  const model = els.modelSelect.value;
  const opts = {
    remove_bg: els.removebg.checked, l2norm: els.l2.checked,
    max_frames: Math.max(8, parseInt(els.maxframes.value, 10) || 600),
    fps: Math.max(1, parseFloat(els.fps.value) || 24),
    max_side: parseInt(els.maxside.value || "0", 10),
  };

  running = true; updateRunButton();
  els.runStatus.classList.remove("hidden");
  setProgress(0, "Submitting…", false);

  let res;
  try {
    res = await (await fetch("/api/runs", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_ids: sel, model, opts }),
    })).json();
  } catch (e) { finishRun(); return setProgress(0, "Submit failed: " + e, true); }
  if (!res.group_id) { finishRun(); return setProgress(0, res.detail || "Failed to start", true); }

  // lay out the cells immediately (running state)
  ensureColumn(model);
  for (const run of res.runs) {
    ensureRow(run.source_id, run.original_url);
    runMeta.set(run.run_id, { sid: run.source_id, model, pcaUrl: run.pca_url });
    setPcaCellRunning(run.source_id, model, run.run_id);
  }
  showMatrix();

  const groupRuns = res.runs.map((r) => r.run_id);
  const ev = new EventSource(`/api/runs/${res.group_id}/events`);
  currentEv = ev;
  ev.onmessage = (msg) => {
    const d = JSON.parse(msg.data);
    setProgress(d.progress, d.message, d.status === "error");
    for (const [rid, r] of Object.entries(d.runs || {})) {
      if (r.status === "done") fillPcaCell(rid, r.result);
      else if (r.status === "error") setRunFailed(rid);
    }
    if (d.status === "error") {
      ev.close(); finishRun();
      groupRuns.forEach(setRunFailed);
      return;
    }
    if (d.status === "done") {
      ev.close(); finishRun();
      setTimeout(() => els.runStatus.classList.add("hidden"), 1200);
    }
  };
  ev.onerror = () => {
    if (currentEv !== ev) return;   // closed by us (Clear/done) — ignore
    ev.close(); finishRun();
    groupRuns.forEach(setRunFailed);
    setProgress(0, "Lost connection to server", true);
  };
}

function finishRun() {
  running = false;
  if (currentEv) currentEv = null;
  updateRunButton();
}

function setProgress(frac, msg, isError) {
  els.progFill.style.width = `${Math.round((frac || 0) * 100)}%`;
  els.progFill.style.background = isError ? "var(--danger)" : "";
  els.progMsg.textContent = msg || "";
  els.progMsg.classList.toggle("error", !!isError);
}

// ------------------------------------------------------------- matrix layout
function showMatrix() {
  els.matrixEmpty.classList.add("hidden");
  els.matrix.classList.remove("hidden");
  els.transport.classList.remove("hidden");
}
function gridTemplate() {
  els.matrix.style.gridTemplateColumns =
    `minmax(70px,150px) repeat(${1 + cols.length}, minmax(180px, 1fr))`;
}
function makeCell(r, c, cls) {
  const d = document.createElement("div");
  d.className = "cell " + (cls || "");
  d.style.gridRow = String(r);
  d.style.gridColumn = String(c);
  els.matrix.appendChild(d);
  return d;
}
function headerCell(c, text, cls) {
  const d = makeCell(1, c, "hcell " + (cls || ""));
  d.textContent = text;
  return d;
}

function ensureColumn(model) {
  if (cols.includes(model)) return;
  if (cols.length === 0) {            // first time: also build corner + Original header
    gridTemplate();
    makeCell(1, 1, "hcell corner");
    headerCell(2, "Original", "orig");
  }
  cols.push(model);
  gridTemplate();
  headerCell(2 + cols.length, modelLabel(model), "model");
}

function ensureRow(sid, originalUrl) {
  if (rows.includes(sid)) return;
  rows.push(sid);
  const r = 1 + rows.length;   // header is row 1
  const label = makeCell(r, 1, "rlabel");
  label.textContent = (sources.get(sid) || {}).name || sid;
  label.title = label.textContent;
  // Original video cell
  const cell = makeCell(r, 2, "vcell");
  const video = document.createElement("video");
  video.playsInline = true; video.muted = true; video.preload = "auto";
  video.src = originalUrl;
  cell.appendChild(video);
  origCells.set(sid, { cell, video });
  if (!masterWired) {
    wireMaster(video);
  } else {
    const m = master();   // a row added mid-playback must join the running master
    if (m && m !== video && !m.paused) { video.play().catch(() => {}); syncOne(video); }
  }
}

function cellPos(sid, model) {
  return { r: 2 + rows.indexOf(sid), c: 3 + cols.indexOf(model) };
}

function setPcaCellRunning(sid, model, runId) {
  const key = `${sid}|${model}`;
  let entry = pcaCells.get(key);
  const { r, c } = cellPos(sid, model);
  if (!entry) {
    const cell = makeCell(r, c, "vcell pca");
    entry = { cell, video: null, filter: null, runId, perm: [0, 1, 2], invert: [false, false, false] };
    pcaCells.set(key, entry);
  } else {
    if (entry.video) { entry.video.pause(); entry.video.removeAttribute("src"); entry.video.load(); }
    if (entry.filter && entry.filter.parentNode) entry.filter.parentNode.remove();  // drop old swizzle filter
    entry.filter = null;
    entry.runId = runId;
    entry.cell.innerHTML = "";
    entry.video = null;
  }
  entry.cell.classList.add("running");
  entry.cell.innerHTML = `<div class="cell-busy"><span class="spin big"></span><span>processing…</span></div>`;
}

function fillPcaCell(runId, result) {
  const meta = runMeta.get(runId);
  if (!meta) return;
  const key = `${meta.sid}|${meta.model}`;
  const entry = pcaCells.get(key);
  if (!entry || entry.runId !== runId || entry.video) return;  // already filled
  entry.cell.classList.remove("running");
  entry.cell.innerHTML = "";

  const fid = `swz-${runId}`;
  const filter = makeFilter(fid);
  applyFilter(filter, entry.perm, entry.invert);

  const video = document.createElement("video");
  video.playsInline = true; video.muted = true; video.preload = "auto";
  video.src = meta.pcaUrl;
  video.style.filter = `url(#${fid})`;
  entry.cell.appendChild(video);
  entry.cell.appendChild(buildCtl(entry, filter, result));
  entry.video = video;
  entry.filter = filter;

  // join playback already in progress
  const m = master();
  if (m && !m.paused) { video.play().catch(() => {}); syncOne(video); }
}

function setRunFailed(rid) {
  const meta = runMeta.get(rid);
  if (!meta) return;
  const entry = pcaCells.get(`${meta.sid}|${meta.model}`);
  if (!entry || entry.video || entry.runId !== rid) return;  // already filled or superseded
  entry.cell.classList.remove("running");
  entry.cell.innerHTML = `<div class="cell-busy err">✕ failed</div>`;
}

// ------------------------------------------------------------- PC→RGB swizzle
const PERMS = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
const permLabel = (p) => `R=${p[0] + 1} G=${p[1] + 1} B=${p[2] + 1}`;

function makeFilter(id) {
  const NS = "http://www.w3.org/2000/svg";
  const f = document.createElementNS(NS, "filter");
  f.setAttribute("id", id);
  f.setAttribute("color-interpolation-filters", "sRGB");
  const m = document.createElementNS(NS, "feColorMatrix");
  m.setAttribute("type", "matrix");
  f.appendChild(m);
  els.filterDefs.appendChild(f);
  return m;  // the feColorMatrix node
}
// output channel o reads PC perm[o]; invert[o] => 1 - x (sign flip, normalized in [0,1])
function applyFilter(fe, perm, invert) {
  const rows4 = [];
  for (let o = 0; o < 3; o++) {
    const row = [0, 0, 0, 0, 0];
    const s = invert[o] ? -1 : 1;
    row[perm[o]] = s;
    row[4] = invert[o] ? 1 : 0;
    rows4.push(row);
  }
  rows4.push([0, 0, 0, 1, 0]);
  fe.setAttribute("values", rows4.flat().join(" "));
}

function buildCtl(entry, filter, result) {
  const bar = document.createElement("div");
  bar.className = "ctl";

  const sel = document.createElement("select");
  sel.className = "perm";
  PERMS.forEach((p, i) => {
    const o = document.createElement("option");
    o.value = String(i);
    o.textContent = permLabel(p);
    sel.appendChild(o);
  });
  sel.value = String(PERMS.findIndex((p) => p.join() === entry.perm.join()));
  sel.title = "Which principal component drives each color channel";
  sel.addEventListener("change", () => {
    entry.perm = PERMS[parseInt(sel.value, 10)].slice();
    applyFilter(filter, entry.perm, entry.invert);
  });
  bar.appendChild(sel);

  const inv = document.createElement("div");
  inv.className = "inv";
  ["R", "G", "B"].forEach((c, i) => {
    const chip = document.createElement("button");
    chip.className = `invchip i${c}`;
    chip.textContent = c;
    chip.title = `Invert ${c} channel`;
    chip.addEventListener("click", () => {
      entry.invert[i] = !entry.invert[i];
      chip.classList.toggle("on", entry.invert[i]);
      applyFilter(filter, entry.perm, entry.invert);
    });
    inv.appendChild(chip);
  });
  bar.appendChild(inv);

  if (result) {
    const info = document.createElement("span");
    info.className = "cellinfo";
    info.textContent = `${result.grid} · ${result.width}×${result.height}`;
    info.title = `${result.frames} frames · proc ${result.proc} · ${result.out_fps} fps · ${result.gpus} GPU(s)`;
    bar.appendChild(info);
  }
  return bar;
}

// ------------------------------------------------------------- synced player
function allVideos() {
  const vs = [];
  for (const { video } of origCells.values()) if (video) vs.push(video);
  for (const { video } of pcaCells.values()) if (video) vs.push(video);
  return vs;
}
function master() {
  const sid = rows[0];
  return sid ? (origCells.get(sid) || {}).video : null;
}
function followers() {
  const m = master();
  return allVideos().filter((v) => v !== m);
}
const fmt = (t) => {
  if (!isFinite(t)) t = 0;
  const mm = Math.floor(t / 60), s = Math.floor(t % 60);
  return `${mm}:${s.toString().padStart(2, "0")}`;
};
function ratio(f) {
  const m = master();
  if (m && isFinite(m.duration) && m.duration > 0 && isFinite(f.duration) && f.duration > 0)
    return f.duration / m.duration;
  return 1;
}
function syncOne(f) {
  const m = master();
  if (!m || !isFinite(m.currentTime)) return;
  const target = Math.min(f.duration || 0, m.currentTime * ratio(f));
  if (Math.abs((f.currentTime || 0) - target) > 0.25) f.currentTime = target;
  f.playbackRate = parseFloat(els.speed.value) * ratio(f);
}
function syncAll() { for (const f of followers()) syncOne(f); }

function wireMaster(m) {
  masterWired = true;
  m.addEventListener("loadedmetadata", () => { els.timeDur.textContent = fmt(m.duration); }, { once: true });
  m.addEventListener("timeupdate", () => {
    if (!m.seeking) {
      els.seek.value = String(Math.round((m.currentTime / (m.duration || 1)) * 1000));
      els.timeCur.textContent = fmt(m.currentTime);
    }
    syncAll();
  });
  m.addEventListener("play", () => {
    els.btnPlay.textContent = "❚❚";
    for (const f of followers()) { f.playbackRate = parseFloat(els.speed.value) * ratio(f); f.play().catch(() => {}); }
  });
  m.addEventListener("pause", () => { els.btnPlay.textContent = "▶"; for (const f of followers()) f.pause(); });
  m.addEventListener("ended", () => {
    if (els.btnLoop.classList.contains("active")) {
      m.currentTime = 0; for (const f of followers()) f.currentTime = 0;
      m.play(); for (const f of followers()) f.play().catch(() => {});
    } else { els.btnPlay.textContent = "▶"; }
  });
}

els.btnPlay.addEventListener("click", () => {
  const m = master();
  if (!m) return;
  if (m.paused) { m.play(); for (const f of followers()) f.play().catch(() => {}); }
  else { m.pause(); for (const f of followers()) f.pause(); }
});
els.seek.addEventListener("input", () => {
  const m = master();
  if (!m) return;
  const t = (els.seek.value / 1000) * (m.duration || 0);
  m.currentTime = t;
  els.timeCur.textContent = fmt(t);
  syncAll();
});
els.speed.addEventListener("change", () => {
  const m = master();
  if (m) m.playbackRate = parseFloat(els.speed.value);
  syncAll();
});
els.btnLoop.addEventListener("click", () => els.btnLoop.classList.toggle("active"));

els.btnClear.addEventListener("click", () => {
  if (currentEv) { currentEv.close(); currentEv = null; }   // stop a stale SSE stream
  finishRun();
  const m = master();
  if (m) m.pause();
  for (const v of allVideos()) { v.pause(); v.removeAttribute("src"); v.load(); }
  els.matrix.innerHTML = "";
  els.filterDefs.innerHTML = "";
  cols.length = 0; rows.length = 0;
  origCells.clear(); pcaCells.clear(); runMeta.clear();
  masterWired = false;
  els.matrix.classList.add("hidden");
  els.transport.classList.add("hidden");
  els.matrixEmpty.classList.remove("hidden");
  els.runStatus.classList.add("hidden");
  setProgress(0, "", false);
  fetch("/api/flush", { method: "POST" }).catch(() => {});
});

document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !els.transport.classList.contains("hidden") &&
      !(e.target.closest && e.target.closest("input,select,button,textarea"))) {
    e.preventDefault(); els.btnPlay.click();
  }
});

loadModels();
loadSources();
updateRunButton();
