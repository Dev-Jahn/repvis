"use strict";

const $ = (id) => document.getElementById(id);

const els = {
  dropzone: $("dropzone"), fileInput: $("file-input"),
  chips: $("source-chips"), chipsHint: $("chips-hint"),
  modelSelect: $("model-select"), modelNote: $("model-note"), gpuBadge: $("gpu-badge"),
  l2: $("opt-l2"),
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
const colHeads = new Map();       // model -> header cell
const origCells = new Map();      // sourceId -> {label, cell, video}
const pcaCells = new Map();       // `${sid}|${model}` -> {cell, video, filter, runId, perm, invert}
const runMeta = new Map();        // runId -> {sid, model, pcaUrl}
let masterEl = null;              // transport master: first row's Original video
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

// ------------------------------------------------------------- workspace load
async function loadWorkspace() {
  try {
    const data = await (await fetch("/api/workspace")).json();
    for (const s of data.sources) addSourceChip(s);
    for (const run of data.runs) restoreRun(run);
    if (data.runs.length) showMatrix();
    for (const g of data.active || []) attachGroup(g.group_id, g.model, g.runs);
  } catch { /* server offline */ }
}
function restoreRun(run) {
  ensureColumn(run.model);
  ensureRow(run.source_id, run.original_url);
  runMeta.set(run.run_id, { sid: run.source_id, model: run.model, pcaUrl: run.pca_url, seg: run.seg || null });
  setPcaCellRunning(run.source_id, run.model, run.run_id);
  fillPcaCell(run.run_id, run.result);
}

// ------------------------------------------------------------- sources
function addSourceChip(rec) {
  if (sources.has(rec.id)) return;
  sources.set(rec.id, { ...rec, selected: false });
  els.chipsHint.classList.add("hidden");
  const chip = document.createElement("button");
  chip.className = "chip";
  chip.dataset.sid = rec.id;
  chip.innerHTML = `<span class="chk-box"></span><span class="chip-name"></span>` +
                   `<span class="chip-size"></span><span class="chip-x" title="Delete source and its results">×</span>`;
  chip.querySelector(".chip-name").textContent = rec.name;
  chip.querySelector(".chip-size").textContent =
    rec.size ? `${(rec.size / 1048576).toFixed(0)}MB` : "";
  chip.querySelector(".chip-x").addEventListener("click", (e) => {
    e.stopPropagation();
    deleteSource(rec.id, chip);
  });
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

async function deleteSource(sid, chip) {
  const name = (sources.get(sid) || {}).name || sid;
  if (!confirm(`Delete "${name}" and all of its results?`)) return;
  let r;
  try { r = await fetch(`/api/sources/${sid}`, { method: "DELETE" }); }
  catch { r = null; }
  if (!r || !r.ok) {
    chip.classList.add("error");
    setTimeout(() => chip.classList.remove("error"), 1200);
    return;
  }
  sources.delete(sid);
  chip.remove();
  removeRow(sid);
  if (!sources.size) els.chipsHint.classList.remove("hidden");
  updateRunButton();
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
    l2norm: els.l2.checked,
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

  attachGroup(res.group_id, model, res.runs);
}

// Lay out the group's cells (running state) and follow its SSE stream. Also
// used to re-attach to an in-flight group after a page reload.
function attachGroup(groupId, model, runsList) {
  running = true; updateRunButton();
  els.runStatus.classList.remove("hidden");
  ensureColumn(model);
  for (const run of runsList) {
    ensureRow(run.source_id, run.original_url);
    runMeta.set(run.run_id, { sid: run.source_id, model, pcaUrl: run.pca_url, seg: run.seg || null });
    setPcaCellRunning(run.source_id, model, run.run_id);
  }
  showMatrix();

  const groupRuns = runsList.map((r) => r.run_id);
  const ev = new EventSource(`/api/runs/${groupId}/events`);
  currentEv = ev;
  ev.onmessage = (msg) => {
    const d = JSON.parse(msg.data);
    setProgress(d.progress, d.message, d.status === "error");
    for (const [rid, r] of Object.entries(d.runs || {})) {
      if (r.status === "done") {
        const mm = runMeta.get(rid);       // pick up seg if the SSE done event carries it
        if (mm && r.seg) mm.seg = r.seg;
        fillPcaCell(rid, r.result);
      } else if (r.status === "error") setRunFailed(rid);
    }
    updateBusyCells(groupRuns, d);
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

function updateBusyCells(groupRuns, d) {
  const pct = Math.round((d.progress || 0) * 100);
  for (const rid of groupRuns) {
    const meta = runMeta.get(rid);
    if (!meta) continue;
    const entry = pcaCells.get(`${meta.sid}|${meta.model}`);
    if (!entry || entry.runId !== rid || entry.video) continue;
    const lbl = entry.cell.querySelector(".cell-busy .busy-msg");
    if (lbl) lbl.textContent = `${d.stage || "…"} · ${pct}%`;
  }
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
  colHeads.set(model, headerCell(2 + cols.length, modelLabel(model), "model"));
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
  origCells.set(sid, { label, cell, video });
  if (!masterEl) {
    wireMaster(video);
  } else {
    const m = master();   // a row added mid-playback must join the running master
    if (m && m !== video && !m.paused) { video.play().catch(() => {}); syncOne(video); }
  }
}

function unloadVideo(v) {
  if (!v) return;
  v.pause(); v.removeAttribute("src"); v.load();
}

function removeRow(sid) {
  const i = rows.indexOf(sid);
  if (i === -1) return;
  rows.splice(i, 1);
  const o = origCells.get(sid);
  origCells.delete(sid);
  if (o) {
    if (o.video === masterEl) masterEl = null;
    unloadVideo(o.video);
    o.cell.remove(); o.label.remove();
  }
  for (const model of cols) {
    const key = `${sid}|${model}`;
    const entry = pcaCells.get(key);
    if (!entry) continue;
    pcaCells.delete(key);
    if (entry.runId) runMeta.delete(entry.runId);
    stopSeg(entry);
    unloadVideo(entry.video);
    if (entry.filter && entry.filter.parentNode) entry.filter.parentNode.remove();
    entry.cell.remove();
  }
  if (!rows.length) { resetMatrix(); return; }
  pruneColumns();
  relayout();
  if (!masterEl) {   // the removed row owned the transport — hand it to the new first row
    const v = (origCells.get(rows[0]) || {}).video;
    if (v) wireMaster(v);
  }
}

function pruneColumns() {
  for (let ci = cols.length - 1; ci >= 0; ci--) {
    const model = cols[ci];
    if (rows.some((sid) => pcaCells.has(`${sid}|${model}`))) continue;
    cols.splice(ci, 1);
    const head = colHeads.get(model);
    if (head) head.remove();
    colHeads.delete(model);
  }
}

function relayout() {
  gridTemplate();
  cols.forEach((model, ci) => {
    const head = colHeads.get(model);
    if (head) head.style.gridColumn = String(3 + ci);
  });
  rows.forEach((sid, ri) => {
    const r = String(2 + ri);
    const o = origCells.get(sid);
    if (o) { o.label.style.gridRow = r; o.cell.style.gridRow = r; }
    cols.forEach((model, ci) => {
      const e = pcaCells.get(`${sid}|${model}`);
      if (e) { e.cell.style.gridRow = r; e.cell.style.gridColumn = String(3 + ci); }
    });
  });
}

function resetMatrix() {
  for (const e of pcaCells.values()) stopSeg(e);
  for (const v of allVideos()) unloadVideo(v);
  els.matrix.innerHTML = "";
  els.filterDefs.innerHTML = "";
  cols.length = 0; rows.length = 0;
  colHeads.clear(); origCells.clear(); pcaCells.clear(); runMeta.clear();
  masterEl = null;
  els.matrix.classList.add("hidden");
  els.transport.classList.add("hidden");
  els.matrixEmpty.classList.remove("hidden");
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
    stopSeg(entry);
    unloadVideo(entry.video);
    if (entry.filter && entry.filter.parentNode) entry.filter.parentNode.remove();  // drop old swizzle filter
    entry.filter = null;
    entry.runId = runId;
    entry.cell.innerHTML = "";
    entry.video = null;
  }
  entry.cell.classList.add("running");
  entry.cell.innerHTML = `<div class="cell-busy"><span class="spin big"></span><span class="busy-msg">queued…</span></div>`;
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

  // SAM foreground segmentation: click layer + point markers over the (baked) video.
  // seg source: fresh run -> result.seg, reload/re-attach -> run entry's seg in runMeta.
  // Show the click layer + controls whenever a run rendered (seg exists), not only when
  // the auto-seg succeeded — a failed/empty auto-seg is recovered by clicking a point.
  const seg = (result && result.seg) || meta.seg || null;
  entry.seg = seg;
  if (seg) {
    entry.points = (seg.points || []).map((p) => p.slice());
    startSeg(entry);                            // click layer + markers over the video
    entry.cell.appendChild(buildSegCtl(entry)); // ↺ reset + Refit
  }

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

  const meta = runMeta.get(entry.runId);
  if (meta) {
    const a = document.createElement("a");
    a.className = "dl";
    a.href = meta.pcaUrl;
    const src = ((sources.get(meta.sid) || {}).name || meta.sid).replace(/\.[^.]+$/, "");
    a.download = `${src}_${meta.model}_pca.mp4`;
    a.title = "Download PCA video (encoded as PC1→R PC2→G PC3→B — the swizzle above is display-only)";
    a.textContent = "⬇";
    bar.appendChild(a);
  }

  if (result) {
    const info = document.createElement("span");
    info.className = "cellinfo";
    info.textContent = `${result.grid} · ${result.width}×${result.height}`;
    info.title = `${result.frames} frames · proc ${result.proc} · ${result.out_fps} fps · ${result.gpus} GPU(s)`;
    bar.appendChild(info);
  }
  return bar;
}

// -------------------------------------------------- SAM foreground segmentation
// The mask is pixel-accurate (SAM2), temporally propagated, and BAKED into pca.mp4
// server-side — no client overlay. Here we let the user refine it by clicking the
// video: left click adds a positive (+) point, Alt/Option+click a negative (−) one.
// Clicks map clientX/Y -> SOURCE pixel coords through the video content rect
// (object-fit: contain). Each edit POSTs the whole point set, then reloads the
// re-baked video. Point markers are drawn over the video and kept positioned.

// clientX/Y -> source pixel (x,y) via the video's contain-fit content rect.
// Returns null if the video has no intrinsic size yet or the click is outside it.
function clickToSource(v, clientX, clientY) {
  const vw = v.videoWidth, vh = v.videoHeight;
  if (!vw || !vh) return null;
  const rect = v.getBoundingClientRect();
  const scale = Math.min(rect.width / vw, rect.height / vh);
  const dw = vw * scale, dh = vh * scale;
  const dx = rect.left + (rect.width - dw) / 2, dy = rect.top + (rect.height - dh) / 2;
  const x = (clientX - dx) / scale, y = (clientY - dy) / scale;
  if (x < 0 || y < 0 || x > vw || y > vh) return null;   // gutter / letterbox
  return [x, y];
}

// (Re)draw the point markers at their source coords mapped into the display box.
function drawMarkers(entry) {
  const cont = entry.markers, v = entry.video;
  if (!cont || !v) return;
  cont.innerHTML = "";
  const vw = v.videoWidth, vh = v.videoHeight;
  const bw = v.clientWidth, bh = v.clientHeight;
  if (!vw || !vh || !bw || !bh) return;
  const scale = Math.min(bw / vw, bh / vh);
  const dw = vw * scale, dh = vh * scale, offx = (bw - dw) / 2, offy = (bh - dh) / 2;
  for (const [x, y, label] of entry.points) {
    const dot = document.createElement("div");
    dot.className = "segdot " + (label ? "pos" : "neg");
    dot.style.left = (offx + x * scale) + "px";
    dot.style.top = (offy + y * scale) + "px";
    cont.appendChild(dot);
  }
}

// Which seg frame a click lands on: DURATION ratio × seg.frames (NOT fps — an NVENC
// sub-1fps encode can round, so seg.fps may disagree with the real frame count).
function frameOfClick(entry) {
  const v = entry.video, n = (entry.seg && entry.seg.frames) || 1;
  const dur = v.duration;
  const f = (isFinite(dur) && dur > 0) ? Math.floor((v.currentTime / dur) * n) : 0;
  return Math.max(0, Math.min(n - 1, f));
}

function onSegClick(entry, e) {
  if (entry.segBusy) return;
  const pt = clickToSource(entry.video, e.clientX, e.clientY);
  if (!pt) return;
  e.preventDefault();
  entry.points.push([pt[0], pt[1], e.altKey ? 0 : 1, frameOfClick(entry)]);   // Alt/Option -> negative
  drawMarkers(entry);
  submitSeg(entry);
}

function startSeg(entry) {
  const v = entry.video;
  const layer = document.createElement("div");
  layer.className = "seglayer";
  layer.title = "Click to add a foreground point (+), Alt/Option-click for background (−)";
  const markers = document.createElement("div");
  markers.className = "segmarkers";
  layer.appendChild(markers);
  entry.seglayer = layer; entry.markers = markers;
  v.insertAdjacentElement("afterend", layer);          // over the video, below .ctl

  layer.addEventListener("pointerdown", (e) => onSegClick(entry, e));
  entry.segRo = new ResizeObserver(() => drawMarkers(entry));
  entry.segRo.observe(v);
  entry._segload = () => drawMarkers(entry);
  v.addEventListener("loadeddata", entry._segload);     // videoWidth known -> place dots
  drawMarkers(entry);
}

function stopSeg(entry) {
  if (!entry) return;
  if (entry.segRo) entry.segRo.disconnect();
  if (entry._segload && entry.video) entry.video.removeEventListener("loadeddata", entry._segload);
  if (entry.seglayer && entry.seglayer.parentNode) entry.seglayer.remove();
  entry.segRo = null; entry._segload = null;
  entry.seglayer = null; entry.markers = null; entry.busyEl = null;
}

// Adopt the point set the server settled on (auto-seed after a reset, or echoed back).
function adoptSeg(entry, seg) {
  if (!seg) return;
  entry.seg = seg;
  entry.points = (seg.points || []).map((p) => p.slice());
  const meta = runMeta.get(entry.runId);
  if (meta) meta.seg = seg;
  drawMarkers(entry);
}

function setSegBusy(entry, on) {
  entry.segBusy = on;
  entry.cell.classList.toggle("segbusy", on);
  if (on && !entry.busyEl && entry.seglayer) {
    const b = document.createElement("div");
    b.className = "seg-spin";
    b.innerHTML = `<span class="spin big"></span>`;
    entry.seglayer.appendChild(b);
    entry.busyEl = b;
  } else if (!on && entry.busyEl) {
    entry.busyEl.remove(); entry.busyEl = null;
  }
}

// Swap in the freshly re-baked pca.mp4 (cache-busted) without dropping playback.
function reloadSegVideo(entry) {
  const meta = runMeta.get(entry.runId);
  if (!meta || !entry.video) return;
  const wasPlaying = master() && !master().paused;
  entry.video.addEventListener("loadeddata", () => {
    drawMarkers(entry);
    if (wasPlaying) { entry.video.play().catch(() => {}); syncOne(entry.video); }
  }, { once: true });
  entry.video.src = meta.pcaUrl + "?v=" + Date.now();
  entry.video.load();
}

// POST the current point set (empty => server re-auto-seeds), then reload the video.
async function submitSeg(entry) {
  if (entry.segBusy) return;
  setSegBusy(entry, true);
  try {
    const r = await fetch(`/api/runs/${entry.runId}/segment`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ points: entry.points }),
    });
    const d = await r.json().catch(() => ({}));
    if (!r.ok || !d.ok) throw new Error(d.detail || "segment failed");
    adoptSeg(entry, d.seg);
    reloadSegVideo(entry);
  } catch { /* keep the current points; the user can retry */ }
  finally { setSegBusy(entry, false); }
}

async function refitSeg(entry) {
  if (entry.segBusy) return;
  setSegBusy(entry, true);
  try {
    const r = await fetch(`/api/runs/${entry.runId}/refit`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: "{}",
    });
    const d = await r.json().catch(() => ({}));
    if (!r.ok || !d.ok) throw new Error(d.detail || "refit failed");
    if (d.seg) adoptSeg(entry, d.seg);
    reloadSegVideo(entry);
  } catch { /* leave colors as-is */ }
  finally { setSegBusy(entry, false); }
}

// controls row: ↺ reset (re-auto-seed) + Refit (re-fit PCA colors over the foreground)
function buildSegCtl(entry) {
  const bar = document.createElement("div");
  bar.className = "segctl";

  const hint = document.createElement("span");
  hint.className = "seghint";
  const seg = entry.seg || {};
  if (seg.error) {
    hint.textContent = "segmentation failed — click to add a point";
    hint.style.color = "var(--danger)";
  } else if (seg.empty) {
    hint.textContent = "no foreground — click the subject";
    hint.style.color = "var(--danger)";
  } else {
    hint.textContent = "click +  ·  alt −";
  }
  bar.appendChild(hint);

  const reset = document.createElement("button");
  reset.className = "segbtn reset";
  reset.textContent = "↺";
  reset.title = "Reset to the automatic seed";
  reset.addEventListener("click", () => { entry.points = []; submitSeg(entry); });
  bar.appendChild(reset);

  const refit = document.createElement("button");
  refit.className = "segbtn refit";
  refit.textContent = "Refit";
  refit.title = "Re-fit the PCA colors over the current foreground";
  refit.addEventListener("click", () => refitSeg(entry));
  bar.appendChild(refit);

  return bar;
}

// ------------------------------------------------------------- synced player
function allVideos() {
  const vs = [];
  for (const { video } of origCells.values()) if (video) vs.push(video);
  for (const { video } of pcaCells.values()) if (video) vs.push(video);
  return vs;
}
function master() { return masterEl; }
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

// Handlers guard on `masterEl === m` so listeners on a replaced master no-op.
function wireMaster(m) {
  masterEl = m;
  const setDur = () => { els.timeDur.textContent = fmt(m.duration); };
  if (isFinite(m.duration) && m.duration > 0) setDur();
  else m.addEventListener("loadedmetadata", setDur, { once: true });
  m.addEventListener("timeupdate", () => {
    if (masterEl !== m) return;
    if (!m.seeking) {
      els.seek.value = String(Math.round((m.currentTime / (m.duration || 1)) * 1000));
      els.timeCur.textContent = fmt(m.currentTime);
    }
    syncAll();
  });
  m.addEventListener("play", () => {
    if (masterEl !== m) return;
    els.btnPlay.textContent = "❚❚";
    for (const f of followers()) { f.playbackRate = parseFloat(els.speed.value) * ratio(f); f.play().catch(() => {}); }
  });
  m.addEventListener("pause", () => {
    if (masterEl !== m) return;
    els.btnPlay.textContent = "▶"; for (const f of followers()) f.pause();
  });
  m.addEventListener("ended", () => {
    if (masterEl !== m) return;
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
  if (!confirm("Clear all results? Sources are kept; PCA videos are deleted from the server.")) return;
  if (currentEv) { currentEv.close(); currentEv = null; }   // stop a stale SSE stream
  finishRun();
  resetMatrix();
  els.runStatus.classList.add("hidden");
  setProgress(0, "", false);
  fetch("/api/runs", { method: "DELETE" }).catch(() => {});
  fetch("/api/flush", { method: "POST" }).catch(() => {});
});

document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !els.transport.classList.contains("hidden") &&
      !(e.target.closest && e.target.closest("input,select,button,textarea"))) {
    e.preventDefault(); els.btnPlay.click();
  }
});

(async () => {
  await loadModels();      // model labels must exist before the matrix is rebuilt
  await loadWorkspace();
  updateRunButton();
})();
