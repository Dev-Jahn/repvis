# repvis — Development Progress

repvis is a web tool that visualizes the **dense patch-feature geometry of a video**. For a chosen vision backbone (DINOv2, DINOv3, or V-JEPA 2.1) it extracts per-patch D-dimensional features for every frame, fits a PCA over the whole clip, maps the top-3 principal components to RGB, and renders the result as a video shown side-by-side with the original. It runs as a FastAPI + vanilla-JS single-page app organized as a persistent **matrix workspace** (rows = source videos, columns = Original + each model), supports **joint PCA across multiple videos** (shared basis → same color = same feature direction), and is tuned for a multi-GPU Blackwell box with a fully-GPU decode→extract→encode pipeline. Code is public at github.com/Dev-Jahn/repvis; the V-JEPA 2.1 weights are public at hf.co/Dev-Jahn/vjepa2.1-vitl-fpc64-384 and download on first use.

---

## Rounds

### 2026-07-08-backlog-zero
- **목표**: parallel-sweep 라운드의 **fable-5 리뷰 사이클 완주**(발견→검증→수정) + **남은 백로그 4건 전부 소진** — 태스크 잔량 0 달성.
- **Shipped**:
  - `fix/same-rid-mutation-marker` **(major, 리뷰 발견)** — same-rid 동시 mutation에서 첫 완료자의 discard가 두 번째의 보호를 벗기던 구멍: LOCK 안 same-rid **409 가드**(segment/refit 양쪽) + RED/GREEN 증명된 회귀 테스트. (`628becd`)
  - `fix/segcache-device-sig` **(major, 리뷰 발견)** — `_SegCache` sig에 device 포함 `(source_id, T, dev)`: device drift가 cross-device 500 대신 cold miss. (`628becd`)
  - `fix/create-runs-registration-window` (minor, 리뷰 발견) — 소스 검증+mkdir+GROUPS 등록을 단일 LOCK 임계구역으로. (`628becd`)
  - `spike/fp8-attention` — **fidelity 게이트 NO-GO 확정**: torchao full-FP8(속도 1.39–1.63x 후보)이 실사 클립에서 PCA subspace 4.7–7.5°(게이트 2°) / shared-basis ΔE p95 5.4–9.4(게이트 3) — cosine 0.99가 숨기는 basis-rotation 함정 실증. 게이트 러너 `scripts/fp8_gate_dinov2.py` 커밋. (`6dde2f1`)
  - `perf/phase2-sam-decode-tail` — feats.f16 덤프를 SAM과 오버랩(비동기 writer + atomic replace): 덤프 잔여 5–6.5s→~1s, phase-2 wall −4–5s. NVDEC 프레임 재사용은 **정확성 근거로 기각**(NVDEC↔CPU 비트 불일치). 남은 레버 = SAM2 자체(~40s) → `perf/multi-gpu-sam`. (`c78ea48`)
  - `feat/saliency-artifact-tokens` — median/MAD |z|>3.5 norm-outlier 필터로 flat-scene saliency argmax hijack 수정; 합성 fill_frame 회귀는 **본질적 트레이드오프로 수용**(무객체 gradient 클립). CPU 단위테스트 승격. (`f30136e`, `9c8184d`)
  - `fix/auth-hardening` — /api/login per-IP 지수 백오프(5회→429), #t= fragment 무조건 제거, README Security 섹션. (`e0f3089`)
  - `fix/autoseed-negative-gate` — **ruling**: negative 게이트 추가 대신 주장 강등("never worse"는 5-clip 경험적 결과) — eval에서 hole-free 실증 + 실세계 피해 증거 없음. docstring에 기록. (`9c8184d`)
- **Gates**: CPU pytest **45/5skip**; ruff clean; same-rid 409 RED/GREEN; fp8 게이트 재현 2회 일치; tail 수정 masks byte-identical(비결정성 기저와 동일 크기); 리뷰 claim 5건 생존(분위수는 확장 fuzz까지).
- **SSOT**: unchanged.
- **Decisions pending**: none.
- **Review**: requested (`docs/reviews/2026-07-08-backlog-zero-request.md`).
- **Next**: 잔여 minor 4건 등록됨(`perf/multi-gpu-sam`, `chore/decode-codec-coverage`, `fix/stale-vfr-run-migration`, `perf/segcache-byte-budget`) — GPU 가용성/우선순위에 따라 착수. 라이브 서버 재시작(신규 auth/캐시/디코드 반영) 대기.

### 2026-07-08-parallel-sweep
- **목표**: remediation 리뷰 ingest(신규 Major 1건) 처리 후 **남은 태스크 전량을 병렬로 소진** — 모든 산출물을 GPU 실측으로 검증해 병합/재작업/폐기 판정.
- **Shipped**:
  - `fix/delete-mutation-stale-snapshot` **(major)** — delete/supersede가 락 밖 rmtree로 mutation 중 run dir을 지우던 race를 **단일-LOCK 규율**로 봉합 + atomic meta.json; barrier 회귀 테스트 RED/GREEN 증명. (`d39ee3b`)
  - `feat/endpoint-access-control` **(major)** — 공유 토큰(REPVIS_TOKEN) 미들웨어 + 브라우저 쿠키 플로우(미디어/SSE). 보안 3-lens 리뷰가 찾은 **이벤트루프 기아 Major**(async upload/SSE가 루프에서 blocking LOCK 획득)를 `asyncio.to_thread`로 수정. (`afefcb6`, `0dccb91`)
  - `fix/vfr-decode-alignment` **(major)** — spike가 CPU로 입증(`9c266d5`)한 VFR 정렬 붕괴의 근인이 **phase-1 자체의 timestamp 기반 seek**임을 GPU로 확정(NVDEC는 sparse open-GOP에서 크래시까지); 양 경로가 공유하는 **positional no-seek primitive**(`video_io.iter_frames_at`)로 통일. E2E VFR centroid 오차 <4px, 단일-unit 디코드 2배 가속. (`34ce9dd`)
  - `perf/sam-session-cache` — run별 SAM2 세션/비전피처 캐시: warm 클릭 **2.7x**(841ms), cold==warm masks byte-match, idle VRAM ~0. 단일-LOCK delete 경로에 `drop_seg_cache` 통합. (`e09c7d5`)
  - `feat/sam-autoseed-quality` — 멀티포인트 시드에 **피처 유사도 게이트**(peak k≥2는 peak 1과 cosine 근접 시만) + border negative. 3-arm 검증: 회귀 0, fill_frame 0.407→0.9995. (`9ed7a52`)
  - `perf/bench-giant-huge-compile` — 추정(~+15%)을 실측으로 대체: giant compile **1.09x**/fp8 **1.39x**, vith16plus compile **1.29x**/fp8 **1.63x** (sm_120, torchao). (`49ba297`)
  - `fix/weighted-quantile-plotting-position` — weighted type-7 분위수(equal weights에서 `torch.quantile`과 동치) + `tests/test_pca.py`. (`04991fd`)
  - `fix/upload-delete-source-phantom` — dup upload↔delete_source phantom race를 LOCK 임계구역으로 봉합. (`dc585d2`)
  - `spike/frame-alignment-check` **(major)** — "정렬은 정확" 주장 검증: CFR(closed/open-GOP)은 성립, **VFR에서 붕괴** — `fix/vfr-decode-alignment` 파생. (`9c266d5`)
  - `chore/push-remove-bg` — be04e13 push 확인 후 종료.
- **Dropped**: `perf/parallel-joint-encode` — **측정 기반 폐기**: 1080p/1200f에서 인코드는 phase-2 87.7s의 ~3%(진짜 tail은 SAM2 64% + SAM용 CPU 디코드 17% + feats 덤프 ~14s). 실험은 `wip/parallel-joint-encode`에 보존, 후속 `perf/phase2-sam-decode-tail` 등록.
- **Gates**: CPU pytest 41/5skip; GPU frame-alignment 24 pass + api 14 pass; delete-race fix RED/GREEN 증명; 적대 리뷰 3-lens×2회(동시성/보안 — 보안 리뷰가 Major 1건 적발·수정).
- **SSOT**: unchanged.
- **Decisions pending**: none.
- **Review**: requested (`docs/reviews/2026-07-08-parallel-sweep-request.md`).
- **Next**: `spike/fp8-attention` fidelity Phase 0/1, `perf/phase2-sam-decode-tail`, `fix/auth-hardening`, `feat/saliency-artifact-tokens` — 4건 병렬 진행 중(worktree).

### 2026-07-07-review-fixes
- **목표**: gpt-5.5-pro 도메인 리뷰의 **REAL finding 5건 + refit 설계(ruling) + 공유 모델 로드 lock** 수정. 파일 단위 병렬 구현.
- **Shipped** (`1262e15`):
  - `fix/segment-click-frame-idx` **(blocker)** — 클릭에 프레임 관통(`[x,y,label,frame]`, duration 비율 계산, sam 프레임별 조건화 + reverse 전파). 예전엔 모든 클릭이 frame 0으로 해석돼 마스크 오염.
  - `fix/sam-failure-silent-fallback` — `seg{available,error,empty}`, 실패해도 컨트롤 유지(복구), refine 실패는 비클로버.
  - `fix/shared-model-load-lock` — 신규 `repvis/modelload.py` 공유 `LOAD_LOCK`(extractor↔SAM dtype race 차단).
  - `fix/segment-point-validation` — finite/범위 검증(400/422).
  - `fix/run-mutation-mutex` — segment/refit 중 DELETE/supersede 거부.
  - `fix/refit-soft-weight-mask` — 하드 `avgpool>0.5` 폐기 → **fg-fraction 가중 PCA**(얇은 구조 보존). `decision/refit-mask-grid-threshold` ruling 반영.
- **Gates**: CPU pytest 5/1, **GPU pytest 6**(4-tuple·validation·SAM-failure-no-clobber·mask-ratio·재디코드 결정성), 라이브 E2E(50% 재생 클릭 → `frame 24` 저장, 실패 시 UI 유지). ruff clean.
- **Touched (not done)**: `spike/frame-alignment-check` — 재디코드 결정성 테스트 추가, 전체 VFR/open-GOP fixture + GPU↔CPU 정확성은 후속.
- **SSOT**: unchanged.
- **Review**: requested (`docs/reviews/2026-07-07-review-fixes-request.md`).
- **Next**: `spike/frame-alignment-check`, `feat/sam-autoseed-quality`, `perf/sam-session-cache`, `feat/endpoint-access-control`(!major).

### 2026-07-07-sam2-foreground
- **목표**: 패치-클러스터링 remove_bg(64×36 그리드라 경계가 뭉개지고 테두리부터 깎임)를 **SAM2 경량 세그멘테이션**으로 교체 — 자동 DINO-saliency 시드 + `/`−클릭 보정, 픽셀 정확 마스크를 PCA 영상에 베이크.
- **Shipped**:
  - `feat/sam2-foreground-segmentation` — SAM2(`sam2.1-hiera-tiny`, Apache-2.0) 세그: 자동 시드 + click(+)/alt-click(−) 보정, 시간축 전파, 마스크 베이크. `repvis/sam.py` 신규. (done · `071502a`)
  - `feat/per-cell-bg-threshold-refit` — (선행) 셀별 threshold 슬라이더 + Refit 버튼; 마스크 부분은 SAM2로 대체, **Refit(색 재fit)은 유지**. (done · `3aff231`)
  - `chore/adopt-jahns-workflow` — 하네스 채택(config·tasks·ADR-0000·CLAUDE.md·roadmap). (done · `858dfd8`)
- **Gates**: 전용 gate 태스크 없음. 검증 — CPU pytest 5 pass/1 skip, **GPU pytest 6 pass**(joint run + segment + refit), 실브라우저 E2E(자동 세그 베이크 배경 ~82% 검정, +클릭 1회로 인물 픽셀 분리, marker/reset/Refit 동작).
- **SSOT**: unchanged (SSOT 비활성 프로젝트).
- **Dropped**: `fix/remove-bg-horizontal-planes` — remove_bg 자체가 폐기돼 obsolete.
- **Decisions pending**: none.
- **Review**: requested (`docs/reviews/2026-07-07-sam2-foreground-request.md`).
- **Next**: `feat/sam-autoseed-quality`(자동 시드 개선), `perf/sam-session-cache`(클릭 지연 단축), `feat/endpoint-access-control`(!major). 미푸시 6커밋 push 대기.

---

## Timeline

### 2026-06-23 — Initial build & first public release — `a56e51c`
- **Research & environment probe.** Confirmed the concept (patch-token features → top-3 PCA → RGB, *not* attention/segmentation). Verified the box, CUDA/uv/ffmpeg+NVENC availability, and HF gating status before writing any code. Two real forks surfaced to the user instead of being silently resolved: DINOv3 is gated (license+token, but the account is authenticated so it works), and "V-JEPA 2.1" has no official `transformers`/HF release.
- **V-JEPA 2.1 path decision.** Chose the user's self-contained HF port (vendored to `src/vjepa21_hf`), which works with stable `transformers`, over a community checkpoint that would have required a PR-fork of `transformers` and broken the shared DINOv3 path. Converted the official ViT-L `.pt` to a 1.3 GB safetensors (clean load, no missing/unexpected keys); deleted the 4.8 GB training checkpoint intermediate.
- **Core modules.** Wrote `config.py`, `video_io.py`, `pca.py`, `extract.py` (multi-GPU sharding), `pipeline.py`, `server.py` (FastAPI + SSE), and the dark drag-&-drop frontend (`index.html`, `app.js`, `style.css`, no CDN deps).
- **Key correctness fix — V-JEPA RoPE dtype.** The port's custom RoPE upcasts q,k to fp32 while v stays bf16, so a pure-bf16 forward crashes in SDPA. Root-cause fix: **load V-JEPA in fp32 and run its forward under bf16 autocast** (casts SDPA inputs consistently, keeps bf16 speed). DINOv2/v3 keep their pure-bf16 SDPA path.
- **Own GPU resize/normalize.** The HF processor silently ignored the `size=` override (stayed at 224 → 16×16 grid), so preprocessing (per-model mean/std, exact target grids) is done on-GPU instead.
- **PCA policy.** Fit PCA **once over the whole clip** so colors stay temporally consistent across frames.
- **Verified end-to-end.** Decode→extract→PCA→NVENC on a test clip: DINOv2 ~2.5 s, V-JEPA 2.1 ViT-L ~3.3 s (auto-sharded across 3 GPUs); DINOv3 (gated) downloaded on first use and ran in ~7.4 s. Full browser flow validated with Playwright; synced side-by-side playback drift 0.000 s.
- **Aspect-preserving resolution policy.** Replaced the square S×S downscale with `proc_hw()`: preserve aspect ratio, snap each side to a **patch multiple**, cap only the long side, compute the grid dynamically per video, restore output to the source resolution. `interpolate_pos_encoding=True` for DINOv2 (learned pos-emb); DINOv3 (RoPE) and V-JEPA (interpolate_rope) handle variable resolution natively.
- **Memory-bounded streaming redesign.** Rewrote the pipeline to stream in bounded chunks so peak GPU memory is O(one chunk), independent of video length. Fit the PCA basis once on a capped subsample; project chunk-by-chunk with feature offload to host RAM; select GPUs by free VRAM (no cuda:0 hardcode); auto-flush VRAM in the job `finally` block (preserving loaded model weights). Validated a real **2.16 hr / 1080p / ~195k-frame** file at 6000 sampled frames: **no OOM, ~12.4 min**, 24-chunk streaming across 8 GPUs. Added **proportional (time-fraction) player sync** so side-by-side stays locked even when PCA-video duration ≠ source duration (sub-1 fps clips get rounded up by NVENC), plus a per-job `/api/flush` endpoint wired to the UI "+ New" button.
- **First public release.** Moved weights to HF (public model card), pointed config at the HF repo with a local-dir fallback, ran an adversarial multi-agent secret/PII scan over exactly the committed files, fixed leaked absolute paths and machine-specific defaults (bind default → 127.0.0.1, `0.0.0.0` now opt-in), and pushed. All prior build work squashed into `a56e51c`.

### 2026-06-23 — V-JEPA port: strict channels-first contract — `1d4c454`
- Removed the port's layout-sniffing heuristic (which treated any dim of size 1 or 3 as the channel — fragile for T=3 or grayscale) at all three sites, adopting Meta's original strict **(B,C,T,H,W)** contract dictated by the `nn.Conv3d` patch-embed API. Output byte-identical because repvis already feeds `(1,3,T,384,384)` and never hit the permute path. Breaking change for external channels-last callers is documented in the model card.

### 2026-06-24 — uv project & cu130 migration — `4e81477`
- The project had **no dependency manifest** (cu128 lived only in a README install line). Added a uv-managed project: `pyproject.toml` (`package=false` — run from repo, not built) with an explicit `[[tool.uv.index]]` for **PyTorch cu130** routing torch/torchvision/torchcodec there while everything else comes from PyPI latest; committed `uv.lock`; replaced the README install steps with a single `uv sync`.
- **Resolved:** torch 2.12.1+cu130, torchvision 0.27.1, torchcodec 0.14.0, transformers 5.12.1, cu13 runtime libs. Blackwell sm_120 verified (bf16 matmul OK; full pipeline DINOv2 4.2 s / V-JEPA 3.5 s). Portability = user edits the one index URL (…/cu128, …/cpu) and re-runs `uv sync`. Version discovery switched from string-sort (mis-ranked torchcodec 0.14 as 0.9) to a version-sorted probe.

### 2026-06-24 — remove-bg Otsu fix (v1) — `703dfa5`
- Fixed the DINOv2-style 2-pass remove-background PCA. The 2-stage structure (PC1 → fg/bg mask → 2nd PCA top-3 on fg only → black bg) was already correct; only the threshold was wrong: `_quantile(pc1, 0.5)` (median) splits ~50% of tokens regardless of content. Replaced with an **Otsu** data-driven threshold, foreground = minority side, plus a degeneracy guard (fixed so `c1/thr` are set only on a non-degenerate split, keeping fit and projection consistent). Also documented that **L2-normalize is correct-but-subtle** (runs PCA on angular geometry) with near-zero effect on these backbones since token norms are fairly uniform and post-projection percentile normalization re-scales each axis anyway.

### 2026-06-25 — Matrix / joint-PCA workspace — `3b49fff`
- Rebuilt repvis around a **matrix workspace** delivering five features: (1) multi-video **joint PCA**, (2) content-addressed input **dedup** (`sources/<sha>`), (3) source/model-decoupled UX, (4) cross-model comparison as added columns, (5) **PC→RGB permutation switching**.
- **Data model** moved from `job=(upload,model,opts)` with per-job copied input to content-addressed `sources/<sha>` + `runs` groups.
- **Key insight (#5):** PC→RGB is a pure linear channel reshuffle, so the server always encodes canonical PC1→R,PC2→G,PC3→B **once** and the client swaps a `feColorMatrix` CSS filter for all 6 permutations (+ sign inversion) — zero re-encode, GPU-accelerated, native `<video>` playback preserved (no canvas draw loop).
- **Joint PCA:** pool selected sources' tokens into one fit buffer → shared basis + shared lo/hi + shared remove-bg threshold, fit once, project each source with the same basis (requires identical model — feature spaces aren't comparable across models).
- **Adversarial review + fixes.** A background multi-dimension review found 26 raw / 23 confirmed / 0 blocker issues, grouped into root causes and fixed: **joint equal-contribution** (subsample each source to the minimum token count before pooling, so a large video can't dominate the basis), stuck failed-run state, new-row playback join, concurrency/Clear teardown (EventSource + progress), per-cell re-run teardown (no filter accumulation), plus robustness (reject 0-byte uploads, sid/rid path-param validation, server-side `max_frames` clamp, `runs/` startup cleanup). The flagged "CPU holds all sources' dense features → RAM O(source count)" was consciously **deferred** (high-RAM host); only the comment was made honest — this became the disk-spill work later.

### 2026-07-02 → 07-03 — Persistent workspace, disk-spilled features, source deletion — `a8d4370`
- Open-ended "comprehensively improve" pass (run under the escalated model). Landed: **workspace persistence** — completed runs stored under `runs/` with meta; `GET /api/workspace` restores the matrix after refresh/server restart; refreshing mid-run re-attaches to the in-progress group's SSE from current progress; re-running the same (source,model) cell auto-supersedes the old result; failed runs leave no directory.
- **Joint-PCA host-RAM cap removed** (resolves the deferred review finding): feature chunks spill to the run dir as **fp16** (half the disk/IO of fp32), read back sequentially in projection and deleted on consume, so host RAM stays ~1 chunk even for many long videos jointly.
- **Source deletion** via chip-`x` (deletes source + derived results; 409 while running; matrix reflows, prunes empty model columns, rewires the sync-playback master). **Clear** changed to delete server-side completed results behind a confirm dialog (a client-only clear would resurrect on refresh due to persistence). Cell UI gained a per-stage progress label and a download button. `REPVIS_DATA_DIR` relocates `sources/`+`runs/`; a `tests/` suite was added (default = GPU-free API tests; `REPVIS_TEST_GPU=1` exercises the full joint pipeline). Encoding stays the canonical PC1→R mapping — the per-cell PC→RGB swizzle is **display-only**, so the downloaded video is canonical. Verified with pytest 6/6 (incl. GPU) and a Playwright E2E (upload → joint run → refresh-restore → delete-and-rewire → Clear → mid-run refresh re-attach).

### 2026-07-03 — Perf overhaul: fully-pipelined all-GPU path (~8×) — `e072b19`
- **Mandate:** GPU mostly idle, CPU-bound — profile, move CPU work to GPU with *no fallback / no option* (lock to the single fastest path), and make single-GPU multi-video joint flawless. Profiling (600 frames 1080p, single GPU, **GPU util avg 6%**) attributed the wall to: torchcodec **CPU decode** 21.9 s/47%, feature spill-to-disk (`torch.save`) 8.5 s/18%, encode pipe-write 6.0 s/13%, pageable GPU→CPU copy 3.1 s/7% — model forward only 3.4 s/7%.
- **Redesign:** decode via **torchcodec NVDEC** (frames land on GPU, no H2D copy); encode via **NVENC** consuming GPU tensors directly (no D2H copy); overlap by splitting the video into units across GPUs, each unit running decode / forward / feature-D2H on **separate CUDA streams** handed off by events. **Stream discipline (critical):** torchcodec synchronizes to the **default** stream, so the default is reserved for decode-only and *all compute runs on side streams* — putting compute on the default stream was measured to slow decode **10×**. Preprocessing resize moved to fp16; `REPVIS_STREAM_CHUNK` default lowered 256 → **64** (256 gives too-coarse overlap granularity).
- **Multi-GPU deadlock, root-caused.** A 7-GPU run hung; `py-spy` + faulthandler traced it to a `finally: dt.join()` waiting on a decode thread blocked in `dq.put()` on a full queue whose consumer had already stopped — and the real first exception was being swallowed. An **adversarial concurrency review** confirmed this was one bug class across 4 sibling queue handoffs. Fixed by making **every queue handoff and blocking get cancel-aware with guaranteed thread teardown**.
- **Thread-safety of model loading.** `expected BFloat16 but found Float` reproduced on a clean 6-GPU run: six threads calling `from_pretrained(dtype=bfloat16)` race torch's **global default dtype**. Fixed by **serializing model construction** behind a lock.
- **Correctness — V-JEPA clip boundary.** A parity check vs the CPU baseline caught V-JEPA feature cosine at only 0.945: a 16-frame first chunk crossed V-JEPA's 32-frame clip boundary, shifting the temporal window. Fixed with **clip-aligned chunking** (an `align` param through `iter_chunks`).
- **Robustness — seek_mode.** A real joint run hit "no more frames left to decode": a clip's container **over-reports frame count** (202 reported vs ~128 exact-decodable) and torchcodec's batched `get_frames_at` dies under `seek_mode="exact"` on that GOP. Fixed by `seek_mode="approximate"` everywhere (decodes all frames at every batch size).
- **Results.** DINOv2-B 600 frames **46.8 s → 5.5 s (~8×)**; V-JEPA **40 s → 6.3 s (~6×)**. Fidelity vs old CPU path: feature cosine ≥ **0.995**, PCA-RGB PSNR ≥ **39 dB** (parity: DINOv2 feature cos 0.9968 / RGB 0.9999; V-JEPA after align-fix 0.9974 / 0.9998). Multi-GPU 1500-frame scaling ~**1.5×** — phase-1 (decode+extract) scales across GPUs but phase-2 (single-stream NVENC encode) is 1 GPU/source by design. Stress test (error injected at every stage): all failures raise in 1–5 s with no hang, and a normal run after all injections still succeeds. ruff clean; pytest 6/6; leak scan clean. Pushed to public GitHub on explicit request (fast-forward, linear history).

### 2026-07-03 — Per-model torch.compile (RoPE models only) — `847e28f`
- Answered the user's challenge ("is this really the fastest code in the world?" / "compile should give 2×") with a 4-axis benchmark instead of a verbal rebuttal. Findings drove **selective compile**: enable `torch.compile` (default mode) **only for the RoPE models** (DINOv3 +38–41%, V-JEPA +28% forward-only; +15–16% in-pipeline wall), leave the **DINOv2 family eager** (GEMM-bound, ~1.06–1.07× forward and a *net loss* on wall because decode is already hidden behind forward). Added a **persistent on-disk inductor cache** rather than preset-resolution precompile (`proc_hw` realistically produces 79–89 distinct grids; presets cover <10%, and dynamic-shape single-graph compile cuts throughput). Fixed a real bug: the old `REPVIS_COMPILE=1` max-autotune-compiled *all* models, slowing DINOv2-base ~6%. **Dead ends:** TensorRT 11.1 builds/runs on sm_120 but only 1.02× (cuBLAS GEMM unchanged); FP8 GEMM is 1.6–1.9× but diluted to 1.18–1.25× overall by unquantized bf16 attention, and FP8 attention carries too much fidelity risk to pursue.

### 2026-07-03 — Add two large backbones — `7d75a15`
- With disk freed (see below), downloaded and registered **DINOv2 ViT-g/14 (giant, 1.1 B, 1536-dim)** and **DINOv3 ViT-H+/16 (huge+, 840 M, 1280-dim, 4 register tokens — handled by the existing generic prefix-strip, prefix=5)**. Registry-driven UI auto-exposes them on restart. `compile` EXCLUDED for giant (GEMM-bound, like the rest of the DINOv2 family), `compile=True` for huge+ (RoPE). Batch sizes from measured VRAM: giant bs=32 (7.3 GB), huge+ bs=64 (8.2 GB). Throughput: giant ~32 img/s (E2E 300 frames/1 GPU 17.8 s); huge+ ~47 img/s (12.2 s). ruff clean; pytest 6/6.
- *(Infra aside, not repvis code:* the benchmark workflow filled the shared root disk to 100%. Only self-generated regenerable caches were reclaimed; a separate, verified NFS migration of another user's data — checksum-compared, open-fd-scanned, atomic symlink cutover — freed the box from 100% back to 77% so the giant/huge downloads could proceed.)*

### 2026-07-05 — remove_bg robust masking — `be04e13` *(committed, not yet pushed)*
- Diagnosed and fixed a torn/split "remove background" output. Three root causes: (1) **RoPE positional-gradient leakage** — PC1 encoded left/right instead of foreground/background, splitting the frame vertically; (2) the **"foreground = minority side" inversion** — when the subject fills >half the frame the background is the minority, so the person got erased (measured in 5 of 8 cases); (3) no spatial cleanup (salt-and-pepper noise on V-JEPA).
- **Method (after a 12-strategy offline comparison over 2 clips × 4 models):**
  - **Gray-field debias** — push a uniform gray frame through the model to capture its pure positional response (no content) and subtract it from features. Real objects are never in the gray response so they can't be erased; for models without positional leak (DINOv2) it degrades to a no-op. One forward per model+resolution, cached.
  - **k-means (k=4) on top-8 PCs + border prior** — background = the cluster over-represented on the **top/upper-side** border (the bottom border is excluded because subjects are usually cut off there). Any failure errs toward *keeping more object*.
  - **Two 3×3 majority-filter passes** to denoise; joint runs share the clustering across sources for consistent fg/bg judgments.
- **Result:** 8/8 (model × clip) cases preserve the person (up from 5/8 inverted); the DINOv3 vertical-cut artifact is eliminated. E2E joint remove_bg 600 frames in **12.7 s** (gray-field fit ~2 s, negligible after cache). pytest CPU 6 + GPU pipeline pass; ruff clean; leak scan clean. **Accepted limitation:** large horizontal planes (floor, desk) often stay classified as foreground — a deliberate safe-direction over-inclusion.

---

## Model registry

| id | backbone | params / feat-dim | patch / prefix | compile | notes |
|---|---|---|---|---|---|
| dinov2-base | DINOv2 ViT-B/14 | — / 768 | 14 / — | off | pure bf16 SDPA; learned pos-emb (`interpolate_pos_encoding=True`) |
| dinov2-large | DINOv2 ViT-L/14 | — / 1024 | 14 | off | GEMM-bound |
| dinov2-giant | DINOv2 ViT-g/14 | 1.1 B / 1536 | 14 | off | bs=32; ~32 img/s |
| dinov3-vitb16 | DINOv3 ViT-B/16 | — / — | 16 / 5 (CLS+4 reg) | **on** | RoPE; gated license (authenticated) |
| dinov3-vith16plus | DINOv3 ViT-H+/16 | 840 M / 1280 | 16 / 5 | **on** | bs=64; ~47 img/s |
| vjepa21-vitl | V-JEPA 2.1 ViT-L | — / 1024 | — / 0 (T,H,W order) | **on** | vendored port; **fp32 weights under bf16 autocast** |

Compile is opt-in via `REPVIS_COMPILE=1` and applies only to the `on` (RoPE) models.

---

## Key architecture & performance decisions

- **Full-GPU pipeline + stream discipline.** NVDEC decode → side-stream compute → NVENC encode, unit-based multi-GPU bin-packing, cancel-safe queues. torchcodec syncs against the **default** CUDA stream, so the default stream is decode-only and compute runs on side streams — violating this starves decode >10×.
- **V-JEPA bf16-autocast over fp32 RoPE.** The port's custom RoPE upcasts q,k to fp32; loading the weights fp32 and running under bf16 autocast casts SDPA inputs consistently, keeping bf16 speed without the pure-bf16 SDPA crash. DINO models keep pure-bf16 SDPA.
- **torch.compile only helps RoPE models.** DINOv2 (incl. giant) is cuBLAS-GEMM-bound and near peak; compile/TRT/FP8 are all ≤1.25× forward and a net loss on wall. Selective per-model gating + a persistent inductor cache (once-per-lifetime kernel builds) beat preset-resolution precompile.
- **remove_bg = gray-field debias + k-means(k=4) + top-border prior + majority filter.** Positional-only debias that cannot erase real objects; border prior instead of minority-vote; joint runs share the clustering.
- **Resolution policy.** Aspect-preserving downscale, each side snapped to a patch multiple, only the long side capped, grid computed per video, output restored to source resolution. Peak GPU mem O(one chunk); `REPVIS_STREAM_CHUNK` default 64.
- **Device selection.** `select_devices()` picks GPUs by free VRAM (skips <16 GB free); never hardcodes cuda:0. Per-job VRAM flush uses `empty_cache()` semantics (returns only unused reserved blocks — model weights stay resident).
- **seek_mode="approximate"** everywhere — `"exact"` crashes on real-world `ffmpeg -ss -c copy` clips whose containers over-report frame count.
- **from_pretrained is not thread-safe** — it flips torch's global default dtype, so model construction is serialized behind a lock for concurrent multi-GPU loads.
- **PC→RGB swizzle is display-only** — server encodes canonical PC1→R,PC2→G,PC3→B once; the browser recolors via a `feColorMatrix` filter (no re-encode, native playback), so downloads stay canonical.
- **Content-addressed sources + persistent runs** — re-processing identical bytes reuses the id (disk never grows); completed runs survive refresh/restart and mid-run refresh re-attaches to the live SSE.

---

## Dead ends (do not retry)

- **Compute on torchcodec's default CUDA stream** — measured 10× decode slowdown.
- **torch.compile as the default / global path** — ~58 s warmup and per-resolution recompile stalls vs ~1 s steady-state gain; regresses DINOv2. Kept opt-in and RoPE-only.
- **TensorRT (11.1, sm_120)** — builds and runs but only 1.02×; cuBLAS GEMM path unchanged.
- **FP8** — GEMM 1.6–1.9× but diluted to 1.18–1.25× overall by bf16 attention; FP8 attention not pursued (fidelity risk). Left as a known forward-speed ceiling.
- **Preset-resolution precompile** — 79–89 distinct grids in practice; presets cover <10%; dynamic-shape single-graph compile cuts throughput. Replaced by the persistent inductor cache.
- **remove_bg alternatives** — *note: the whole feature-clustering remove_bg was replaced by SAM2 pixel segmentation (see §Rounds `2026-07-07-sam2-foreground`); these are historical cautions, not current guidance:* CLS-token cosine saliency (fails on DINOv3 — CLS lives in a different subspace); temporal-mean / DCT-field debias (deletes static subjects); "foreground = minority side of PC1" (inverts on people-dominant frames); PCA R² positional-vs-semantic test (not discriminative — temporal-mean maps are always smooth); k=2 side-picking (flips with k-means init). Also the earlier **median/50% PC1 split** (fixed half-black frame) superseded by Otsu, then by the k=4 method.
- **HF processor for resize** — silently ignores `size=`. **String-sort version discovery** — mis-ranked torchcodec. **davevanveen V-JEPA checkpoint** — not cross-loadable into the port. **Layout-sniffing in the V-JEPA port** — fragile; replaced by the strict (B,C,T,H,W) contract.

---

## Status snapshot (2026-07-05 / 06 — superseded; current state is in §Rounds above)

> This is the point-in-time snapshot from the reconstruction. Since then: `be04e13` and everything
> through the SAM2 rework are **pushed**; feature-clustering remove_bg was **replaced by SAM2**
> segmentation. See §Rounds (top) for the live state.


- **Working tree clean.** `main` is **1 commit ahead of origin** — `be04e13` (remove_bg robust masking) is **committed but NOT yet pushed** (push happens only on explicit user instruction, after a leak scan of `git diff origin/main..HEAD`).
- **Tests:** CPU pytest **5 passed / 1 skipped** (the GPU pipeline test is skipped without `REPVIS_TEST_GPU=1`); with GPU enabled the full suite passes. ruff clean.
- **Models:** 6 registered (DINOv2 B/L/g, DINOv3 B/16 + H+/16, V-JEPA 2.1 ViT-L). Registry-driven UI.
- **Serving:** bind host/port and GPU selection are configurable via env (`HOST`/`PORT`/`REPVIS_GPUS`, `REPVIS_STREAM_CHUNK`, `REPVIS_DATA_DIR`, `REPVIS_COMPILE`); default bind is localhost because the server has **no authentication** — wider binds are a conscious, warned choice. The environment probed was a shared multi-GPU Blackwell box (per-GPU ~97 GB, NVENC h264/hevc/av1). The remove_bg round left a server instance running on a dedicated GPU with a public bind at the user's request.

### Open loose ends / natural next steps
- **Push `be04e13`** to the public remote when approved (leak-scan first).
- **remove_bg horizontal-plane limitation** — floors/desks often stay classified as foreground (accepted safe-direction tradeoff); would need a stronger geometric/semantic prior to resolve.
- **FP8 attention** — the only remaining path to a true ~2× forward, deliberately unshipped due to fidelity risk; FP8/compile numbers for the giant and huge+ variants are not benchmarked (huge+ compile gain is only *expected* ~+15%).
- **Multi-GPU joint speedup capped ~1.5×** — phase-2 NVENC encode is single-stream, 1 GPU/source by design; parallelizing encode is the lever if joint throughput becomes a priority.
- **No access control on job/source endpoints** — anyone who can reach the server (and knows an id) can fetch content; fine for tailnet/localhost, a gap for any wider bind.
- **`samples/test.mp4` and `sources/` content are gitignored** and must never be committed.
