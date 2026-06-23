"use strict";

const $ = (id) => document.getElementById(id);

const els = {
  dropzone: $("dropzone"), fileInput: $("file-input"),
  modelSelect: $("model-select"), modelNote: $("model-note"), gpuBadge: $("gpu-badge"),
  removebg: $("opt-removebg"), l2: $("opt-l2"),
  maxside: $("opt-maxside"), maxframes: $("opt-maxframes"), fps: $("opt-fps"),
  btnProcess: $("btn-process"),
  panelUpload: $("panel-upload"), panelProgress: $("panel-progress"), panelResult: $("panel-result"),
  progStage: $("prog-stage"), progFill: $("prog-fill"), progMsg: $("prog-msg"),
  resultMeta: $("result-meta"),
  vidOrig: $("vid-orig"), vidPca: $("vid-pca"),
  btnPlay: $("btn-play"), seek: $("seek"), timeCur: $("time-cur"), timeDur: $("time-dur"),
  speed: $("speed"), btnLoop: $("btn-loop"), btnReset: $("btn-reset"),
};

let selectedFile = null;
let models = [];

// ---------------- model list ----------------
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
  } catch (e) {
    els.gpuBadge.textContent = "server offline";
  }
}
function updateModelNote() {
  const m = models.find((x) => x.key === els.modelSelect.value);
  if (!m) { els.modelNote.textContent = ""; return; }
  els.modelNote.textContent = `${m.note} · patch ${m.patch} · ≤ ${m.max_side}px long side`;
  els.maxside.value = m.max_side;  // default to this model's cap
}
els.modelSelect.addEventListener("change", updateModelNote);

// ---------------- file selection ----------------
function setFile(file) {
  if (!file || !file.type.startsWith("video/")) return;
  selectedFile = file;
  els.dropzone.classList.add("has-file");
  els.dropzone.querySelector(".dz-title").textContent = file.name;
  els.dropzone.querySelector(".dz-hint").textContent =
    `${(file.size / 1048576).toFixed(1)} MB · ready to process`;
  els.btnProcess.disabled = false;
  els.btnProcess.textContent = "Visualize features ▸";
}
els.dropzone.addEventListener("click", () => els.fileInput.click());
els.fileInput.addEventListener("change", (e) => setFile(e.target.files[0]));
["dragenter", "dragover"].forEach((ev) =>
  els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) =>
  els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.remove("drag"); }));
els.dropzone.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); });

// ---------------- processing ----------------
els.btnProcess.addEventListener("click", startJob);

async function startJob() {
  if (!selectedFile) return;
  const fd = new FormData();
  fd.append("file", selectedFile);
  fd.append("model", els.modelSelect.value);
  fd.append("remove_bg", els.removebg.checked);
  fd.append("l2norm", els.l2.checked);
  fd.append("max_frames", els.maxframes.value || "600");
  fd.append("fps", els.fps.value || "24");
  fd.append("max_side", els.maxside.value || "0");

  els.panelUpload.classList.add("hidden");
  els.panelResult.classList.add("hidden");
  els.panelProgress.classList.remove("hidden");
  setProgress("uploading", 0, "Uploading video…", false);

  let res;
  try {
    res = await (await fetch("/api/jobs", { method: "POST", body: fd })).json();
  } catch (e) {
    return setProgress("error", 0, "Upload failed: " + e, true);
  }
  if (!res.job_id) return setProgress("error", 0, res.detail || "Failed to start job", true);

  const ev = new EventSource(`/api/jobs/${res.job_id}/events`);
  ev.onmessage = (msg) => {
    const d = JSON.parse(msg.data);
    if (d.status === "error") {
      ev.close();
      return setProgress("error", 0, d.message || d.error || "Processing error", true);
    }
    setProgress(d.stage, d.progress, d.message, false);
    if (d.status === "done") {
      ev.close();
      showResult(res.input_url, res.pca_url, d.result);
    }
  };
  ev.onerror = () => { ev.close(); setProgress("error", 0, "Lost connection to server", true); };
}

function setProgress(stage, frac, msg, isError) {
  els.progStage.textContent = stage;
  els.progFill.style.width = `${Math.round((frac || 0) * 100)}%`;
  els.progMsg.textContent = msg || "";
  els.progStage.classList.toggle("error", !!isError);
  els.progMsg.classList.toggle("error", !!isError);
  if (isError) els.progFill.style.background = "var(--danger)";
}

// ---------------- result + synced player ----------------
function showResult(origUrl, pcaUrl, meta) {
  els.panelProgress.classList.add("hidden");
  els.panelResult.classList.remove("hidden");
  if (meta) {
    els.resultMeta.textContent =
      `${meta.frames} frames · proc ${meta.proc} → PCA grid ${meta.grid} · output ${meta.width}×${meta.height} @ ${meta.out_fps} fps · ${meta.gpus} GPU(s)`;
  }
  els.vidOrig.src = origUrl;
  els.vidPca.src = pcaUrl;
  els.vidOrig.load();
  els.vidPca.load();
  initPlayer();
}

const fmt = (t) => {
  if (!isFinite(t)) t = 0;
  const m = Math.floor(t / 60), s = Math.floor(t % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
};

let playerReady = false;
function initPlayer() {
  playerReady = false;
  const o = els.vidOrig;
  o.addEventListener("loadedmetadata", () => {
    els.timeDur.textContent = fmt(o.duration);
    playerReady = true;
  }, { once: true });
}

function master() { return els.vidOrig; }
function follower() { return els.vidPca; }

// The PCA video can have a different duration than the original (long videos get
// subsampled to N frames), so sync proportionally by time-fraction.
function ratio() {
  const m = master(), f = follower();
  if (isFinite(m.duration) && m.duration > 0 && isFinite(f.duration) && f.duration > 0)
    return f.duration / m.duration;
  return 1;
}
function syncFollower() {
  const m = master(), f = follower();
  if (!isFinite(m.currentTime)) return;
  const target = Math.min(f.duration || 0, m.currentTime * ratio());
  if (Math.abs(f.currentTime - target) > 0.25) f.currentTime = target;
}

els.btnPlay.addEventListener("click", () => {
  const m = master();
  if (m.paused) { m.play(); follower().play(); }
  else { m.pause(); follower().pause(); }
});

els.vidOrig.addEventListener("play", () => {
  els.btnPlay.textContent = "❚❚";
  follower().playbackRate = parseFloat(els.speed.value) * ratio();
  follower().play();
});
els.vidOrig.addEventListener("pause", () => { els.btnPlay.textContent = "▶"; follower().pause(); });

els.vidOrig.addEventListener("timeupdate", () => {
  const m = master();
  if (!m.seeking) {
    els.seek.value = String(Math.round((m.currentTime / (m.duration || 1)) * 1000));
    els.timeCur.textContent = fmt(m.currentTime);
  }
  syncFollower();
});

els.vidOrig.addEventListener("ended", () => {
  if (els.btnLoop.classList.contains("active")) {
    master().currentTime = 0; follower().currentTime = 0;
    master().play(); follower().play();
  } else { els.btnPlay.textContent = "▶"; }
});

els.seek.addEventListener("input", () => {
  const m = master();
  const t = (els.seek.value / 1000) * (m.duration || 0);
  m.currentTime = t;
  follower().currentTime = Math.min(follower().duration || 0, t * ratio());
  els.timeCur.textContent = fmt(t);
});

els.speed.addEventListener("change", () => {
  const r = parseFloat(els.speed.value);
  master().playbackRate = r;
  follower().playbackRate = r * ratio();
});

els.btnLoop.addEventListener("click", () => els.btnLoop.classList.toggle("active"));

els.btnReset.addEventListener("click", () => {
  master().pause(); follower().pause();
  // release the decoded video buffers, then flush server-side GPU cache
  els.vidOrig.removeAttribute("src"); els.vidPca.removeAttribute("src");
  els.vidOrig.load(); els.vidPca.load();
  els.panelResult.classList.add("hidden");
  els.panelUpload.classList.remove("hidden");
  els.progFill.style.background = "";
  fetch("/api/flush", { method: "POST" })
    .then((r) => r.json())
    .then((d) => { if (d && d.freed_mb) console.log(`VRAM flushed: ${d.freed_mb} MB freed`); })
    .catch(() => {});
});

// keyboard: space toggles play when result visible
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !els.panelResult.classList.contains("hidden") &&
      e.target.tagName !== "INPUT" && e.target.tagName !== "SELECT") {
    e.preventDefault(); els.btnPlay.click();
  }
});

loadModels();
