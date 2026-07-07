# Spike: FP8 attention for a ~2× DINO forward speedup

**Task:** `spike/fp8-attention` · **Status:** unshipped proposal, needs GPU validation
**Target box:** 8× Blackwell **sm_120** (RTX PRO 6000-class), torch `2.12.1+cu130`, transformers `5.12.1`

## TL;DR (the honest headline)

- **"FP8 attention" alone will not give ~2×.** It only touches the two batched
  matmuls inside SDPA (`QKᵀ` and `P·V`). At repvis's *highest* resolution those
  are ~33–50 % of forward FLOPs; at typical resolutions much less. Amdahl caps
  the realistic **attention-only** win at **~1.15–1.35×** forward, not 2×.
- **~2× requires FP8 on the linear layers too** (QKV/out projections + the MLP,
  which are 8–12 ND² of the 12 ND²+2 N²D per layer). That is a bigger, riskier
  change (calibration on every GEMM) and is where the real speedup lives.
- The fidelity risk is **not** per-pixel noise — it is a *structured* bias that
  **rotates the top-3 PCA basis** and silently remaps colors while per-token
  cosine still reads 0.999. The gate must measure the **subspace rotation** and
  the **rendered color ΔE**, not just cosine. (`scripts/fp8_fidelity.py`
  self-test demonstrates exactly this: a 5° feature rotation keeps cosine at
  0.99988 but the subspace metric catches it.)
- **Recommendation: conditional GO on a spike**, but scope it as *full-FP8
  transformer* on **DINOv2** (the numerically-clean, GEMM-peak family), gated by
  the harness below. Treat attention-only FP8 as the fallback if full FP8 fails
  fidelity. **NO-GO for DINOv3/V-JEPA in this spike** (RoPE upcasts q/k to fp32 —
  fighting that is a separate, larger effort).

---

## 1. Where FP8 attention would apply in the forward

`repvis/extract.py::_Extractor` loads DINO via
`AutoModel.from_pretrained(..., dtype=torch.bfloat16, attn_implementation="sdpa")`
and calls `self.model(pixel_values=x).last_hidden_state`. Each ViT block does:

```
x → LN → [Wq,Wk,Wv] → q,k,v           # QKV projection      3·N·D²   GEMM
        SDPA(q,k,v):
            scores = q @ kᵀ / √d       # QKᵀ                 N²·D     BMM  ← FP8 target
            p      = softmax(scores)   # softmax             (stays high precision)
            out    = p @ v             # P·V                 N²·D     BMM  ← FP8 target
        out → Wo                       # output projection     N·D²   GEMM
x → LN → fc1 → GELU → fc2              # MLP (D→4D→D)         8·N·D²   GEMM
```

**"FP8 attention" = casting q,k,v and p to fp8 and running the two BMMs on fp8
tensor cores**, keeping the softmax accumulation in fp32/bf16. The softmax
*must* stay higher-precision: fp8's ~3-bit mantissa (E4M3) cannot represent the
exp/normalize range without catastrophic error. This is the standard
FlashAttention-3-fp8 / cuDNN-fused-attention recipe.

### FLOP budget — why attention-only can't reach 2×

Per layer: linear work = 12 ND² (3 QKV + 1 out + 8 MLP), attention BMM = 2 N²D.
Attention fraction `f = N / (6D + N)`. repvis token counts `N = grid_h·grid_w`
(patch 14/16, `max_side=1024`):

| model | D | N @ 1024px (~) | attn FLOP frac `f` | attn-only 2× BMM → fwd speedup |
|---|---|---|---|---|
| dinov2-base | 768 | ~4600 | 0.50 | 1.33× (ideal) / ~1.25× (real) |
| dinov2-large | 1024 | ~4600 | 0.43 | 1.30× / ~1.22× |
| dinov2-giant | 1536 | ~4600 | 0.33 | 1.25× / ~1.18× |

"real" discounts the fp8 BMM to ~1.6–1.7× (softmax stays bf16, plus q/k/v/p
cast + amax overhead). **At lower resolution N shrinks, `f` drops, and the win
evaporates** — a 512px clip on base is `f≈0.33`. So attention-only FP8 is a
resolution-dependent 1.15–1.35×. The advertised ~2× is only reachable by also
running the 12 ND² of GEMMs in FP8 (`torch._scaled_mm` / TE `Linear`), which is
a *superset* change, not this task's literal scope. **Be explicit with the
reviewer that "2× forward" = full FP8, not FP8 attention.**

---

## 2. Concrete prototyping paths on Blackwell sm_120

Ordered by effort. All are inference-only (no training, so no fp8 gradients).

**(A) torch-native, dynamic-scaled — lowest friction, best first probe.**
No fused fp8-attention kernel is guaranteed on sm_120, so prototype the *math*
first by monkeypatching the block's attention: cast q,k,v to `float8_e4m3fn`
with per-tensor amax scales, run the two matmuls via `torch._scaled_mm`
(available, confirmed on this box), keep softmax in bf16. This is a fidelity
probe, **not** a speed probe — hand-rolled `_scaled_mm` + explicit softmax will
likely be *slower* than fused bf16 SDPA. Use it to answer "does fp8 attention
survive the PCA?" before investing in a fused kernel.
- HF hook point: replace the module's `attn_implementation` with a registered
  custom attention (`transformers` 5.x `ALL_ATTENTION_FUNCTIONS` / eager-callable
  interface), or subclass and swap `Dinov2SelfAttention.forward`. No model
  re-download; weights are untouched (fp8 is applied to *activations*).

**(B) FlashAttention-3 fp8 — the real speed path, but sm_120 is the risk.**
FA3's fp8 kernels are Hopper (sm_90a)-tuned; datacenter Blackwell is sm_100,
**consumer/workstation Blackwell is sm_120**. FA3 fp8 support/tuning for sm_120
is not something to assume — **verify the installed FA build actually has an
sm_120 fp8 kernel** (`import flash_attn; flash_attn.__version__`; check for a
`fp8`/`e4m3` API and that it doesn't fall back to a slow path or error on
sm_120) before counting on it. If present, this is the path that can deliver the
attention-side speedup for real.

**(C) Transformer Engine (`transformer_engine.pytorch`) — if going full-FP8.**
TE gives `DotProductAttention` (fp8 via cuDNN fused attention) *and* fp8
`Linear`, so it's the natural vehicle for the 2× full-FP8 variant, with
delayed-scaling `fp8_autocast` recipes built in. Cost: you don't get to reuse
HF's `Dinov2Model` as-is — you'd wrap/patch the attention+MLP submodules or run
TE modules with the HF weights loaded in. Highest effort; only worth it once (A)
shows fp8 attention passes fidelity. Confirm the installed TE cuDNN-attention
backend advertises an sm_120 fp8 kernel (same caveat as B).

**Recommended order:** (A) fidelity probe → if pass, (B) for attention-only
speed *or* (C) for full-FP8 2× → re-run the same fidelity gate on the fused
kernel's real output (fused kernels round differently than the (A) reference).

---

## 3. What breaks: calibration, scaling, and the numerics feeding PCA

**Scaling / calibration.**
- E4M3 (mantissa 3 bits, max ~448) is the inference format for q/k/v/p; E5M2 is
  for gradients (irrelevant here). Every fp8 matmul needs a **scale** so the
  tensor's amax maps into range without overflow/underflow.
- **Inference-only lets us skip a calibration dataset**: use **dynamic (current)
  scaling** — compute `amax` of q,k,v per forward and scale live. Costs an amax
  reduction but needs no calibration history and adapts per input. **Delayed
  scaling** (TE default, amax history) is faster but needs a calibration pass
  over representative frames and can lag on a distribution shift (new video →
  stale scale → clipping). For a spike, **prefer dynamic scaling** — it removes
  "did we calibrate on the right data?" as a confound.
- Per-tensor scale may be too coarse for attention (a few outlier channels in
  q/k dominate amax and crush everyone else to 0 in fp8). **Per-head / per-token
  (row-wise) scaling** is the likely-necessary refinement; budget for it.

**Numerics feeding PCA — the part that actually matters for repvis.**
The render is: features → `_fit` (SVD top-3, sign-fixed) → project → **2/98
percentile** normalize → RGB. Two distinct error channels:
1. **Unstructured error** (random fp8 rounding): per-token noise. Mostly washed
   out by SVD (fits dominant variance) and percentile clipping. Low risk.
2. **Structured/biased error** (the dangerous one): if fp8 attention
   systematically shrinks or tilts a feature direction, it **rotates the top-3
   PCA basis**. Because color = projection onto that basis, a few-degree
   rotation **globally remaps hues** — a visible, coherent color shift, not
   noise. Per-token cosine can stay 0.999 while this happens (demonstrated in
   the harness self-test). The 2/98 percentile step *amplifies* sensitivity:
   fp8 outliers can move the lo/hi range and rescale all colors.
3. **Refit path** (`refit_display`, per-cell SAM foreground): re-fits the basis
   on a *small* foreground token set — fewer tokens ⇒ the basis is *more*
   sensitive to fp8 perturbation than the global fit. Gate the refit path too,
   not just the global render.
4. **Temporal flicker:** dynamic per-frame scaling means the fp8 scale changes
   frame-to-frame; if it nudges the basis differently each frame, colors
   **shimmer** even on static content. Must be measured on video, not stills.

---

## 4. Fidelity metrics to gate on

Implemented CPU-side in `scripts/fp8_fidelity.py` (numpy-only, consumes dumps —
no GPU, no model). Three metrics + a pass/fail, run on baseline-vs-candidate
dumps produced on the GPU box:

1. **Per-token feature cosine** (`feature_cosine`): mean, p50, **p1 (worst 1%)**,
   min. Necessary but *not sufficient* — it misses basis rotation.
2. **Top-3 PCA subspace principal angles** (`subspace_principal_angles`): fit the
   top-3 basis on each dump, report principal angles (deg). **This is the metric
   that catches the color-remap failure.** 0° = identical basis.
3. **Rendered color ΔE** (`color_delta_e`): sRGB→Lab, per-pixel CIE76 ΔE between
   the two rendered frames (mean/p50/**p95**/max). Run it two ways:
   (a) both frames rendered with the *fp16* basis → isolates feature error;
   (b) each with its *own* basis → the true end-to-end user-visible change.
   Feed **masked foreground pixels only** (match what a viewer judges). CIE76 is
   the stub; swap CIEDE2000 for the final number (same harness shape).

**Proposed gates** (in `GATES`, tune on baseline↔baseline noise floor first):

| metric | gate | rationale |
|---|---|---|
| cosine mean | ≥ 0.9995 | dominant-signal preservation |
| cosine p1 | ≥ 0.995 | no pocket of wrecked tokens |
| subspace max angle | ≤ 2.0° | color basis barely moves |
| ΔE p95 | ≤ 3.0 | 95% of pixels below "obvious" |

Plus a **temporal** check: frame-to-frame ΔE of the *fp8 stream* must not exceed
the fp16 stream's frame-to-frame ΔE by more than a small margin (no new flicker).
And a **speed** gate: net forward wall-time speedup measured at the *actual*
serving resolutions must clear a threshold (e.g. ≥ 1.3× attention-only, ≥ 1.8×
full-FP8) or the fidelity cost isn't worth it.

---

## 5. Go/No-Go experiment plan

**Phase 0 — noise floor (½ day, GPU).** Run the existing bf16 forward twice
(cudnn.benchmark makes it slightly non-deterministic) and on the same frames at
fp16 vs bf16. Feed both into the harness. This calibrates the gates: fp8 must not
exceed the bf16↔fp16 baseline delta by much. If bf16↔fp16 already moves the
subspace >2°, loosen the gate honestly rather than fail fp8 for free variance.

**Phase 1 — fidelity probe, path (A) (1 day, GPU).** Monkeypatch DINOv2-base
SDPA with the `_scaled_mm` dynamic-scaled fp8 attention (softmax in bf16). Dump
features + rendered frames for a handful of representative videos (varied
content/resolution). Run the harness on the global-render path *and* the
`refit_display` path. **Decision A:** if it fails the gate even at per-token
scaling → **NO-GO on fp8 attention** (numerics can't feed PCA); stop. If it
passes → continue, and record whether per-tensor sufficed or per-head was needed.

**Phase 2 — real kernel + speed, path (B) or (C) (1–2 days, GPU).** Only if
Phase 1 passes. Wire the fused fp8 kernel (verify sm_120 support first, §2),
re-dump, **re-run the harness on the fused output** (different rounding than the
(A) reference). Measure end-to-end forward wall-time at serving resolutions.
**Decision B:** GO iff (fidelity gate passes on fused output) AND (speedup ≥
threshold) AND (no new temporal flicker). Otherwise NO-GO / fall back to
attention-only or shelve.

**Phase 3 — scope decision.** If attention-only passes fidelity but speed is
only ~1.2×, decide whether to extend to full-FP8 (fp8 Linear on QKV/out/MLP via
TE or `_scaled_mm`) for the 2× — re-running the *entire* gate, since fp8 GEMMs
add their own basis-rotation risk on top of attention. Keep DINOv3/V-JEPA out of
scope (RoPE fp32 upcast; separate spike).

**Minimal prototype sketch (Phase 1, path A):**

```python
# monkeypatch inside _Extractor after load (DINOv2 only), inference-only.
import torch, torch.nn.functional as F
F8 = torch.float8_e4m3fn
FMAX = 448.0

def _to_fp8(t):                       # dynamic per-tensor scale
    scale = (t.abs().amax().clamp_min(1e-12)) / FMAX
    return (t / scale).clamp(-FMAX, FMAX).to(F8), scale

def fp8_attention(q, k, v, scale_qk):        # q,k,v: (B,H,N,d) bf16
    qf, sq = _to_fp8(q); kf, sk = _to_fp8(k)
    # scores via fp8 tensor cores, dequant back to bf16 for softmax
    scores = torch._scaled_mm(qf.reshape(-1, qf.shape[-1]),
                              kf.reshape(-1, kf.shape[-1]).T,
                              scale_a=sq, scale_b=sk,
                              out_dtype=torch.bfloat16).reshape(*q.shape[:3], -1)
    p = F.softmax(scores * scale_qk, dim=-1)          # softmax stays bf16
    pf, sp = _to_fp8(p); vf, sv = _to_fp8(v)
    return torch._scaled_mm(pf..., vf..., scale_a=sp, scale_b=sv,
                            out_dtype=torch.bfloat16)  # reshape elided
# NOTE: this is a *fidelity* reference, not fast (per-BMM reshape + explicit
# softmax). Real speed comes from a fused FA3/TE/cuDNN kernel in Phase 2. The
# _scaled_mm reshape/transpose plumbing above is sketch-level, not literal.
```

---

## 6. Risk summary (honest)

| risk | likelihood | severity | mitigation |
|---|---|---|---|
| attention-only ≠ 2× (Amdahl) | **certain** | high (expectation) | reframe as full-FP8; measure `f` per resolution |
| fp8 rotates PCA basis → color remap | medium | high (visible) | subspace-angle gate; per-head scaling |
| sm_120 has no tuned fp8 kernel | **medium-high** | high (no speedup) | verify FA3/TE sm_120 fp8 *before* Phase 2 |
| temporal flicker from dynamic scale | medium | medium | temporal ΔE gate; consider fixed/clamped scale |
| refit path (few tokens) more fragile | medium | medium | gate refit path separately |
| per-tensor scale clips q/k outliers | medium | medium | per-head / row-wise scaling |
| DINOv3/V-JEPA RoPE fp32 upcast fights fp8 | high | — | out of scope this spike |

**Bottom line:** GO on a *bounded* Phase-0→1 fidelity spike on DINOv2 (cheap, 1.5
days, kills the idea fast if numerics don't survive PCA). Only commit to the
Phase-2 kernel/speed work — and the "2×" framing — after Phase 1 passes and sm_120
fp8-kernel support is confirmed. Do not ship on cosine similarity alone.
