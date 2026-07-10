<!-- jahns-workflow feedback: the body below is the reviewer reply VERBATIM (byte-exact copy via `jw review ingest`) — do not edit it; a triage skeleton is appended beneath it. -->
round: 2026-07-08-backlog-zero
reviewer: gpt-5.5-pro
ingested: 2026-07-10
source: /tmp/review.md

---

## 검토 범위

* 패킷: `docs/reviews/2026-07-08-backlog-zero-request.md`
* 패킷상 Reviewed HEAD: `9c8184dfb2528bafed93cf4ea6cefc03c32c5ab8`, base: `5776596a0cd993fc1af6ef70f92d2434a87ec520`
* 현재 `main`은 Reviewed HEAD보다 1커밋 앞서지만, 추가 변경은 `PROGRESS.md`, `ROADMAP.md`, `tasks.yaml`, review request 등 문서/하네스 계열입니다. 코드 리뷰는 `9c8184d` 기준으로 진행했습니다.
* 직접 `git clone`은 이 환경에서 사용할 수 없어 GitHub Connector로 대상 커밋의 실제 파일을 읽었습니다. GPU/E2E/실영상 평가는 재실행하지 못했습니다.

---

## 확정 finding

### Major — 순차 디코드 방식이 `max_frames` 샘플링 계약을 성능상 무력화합니다

현재 `compute_indices()`는 전체 원본 프레임 위치에서 target fps와 `max_frames`에 맞춰 **샘플 위치 목록**을 만듭니다. 즉 반환값은 “샘플 ordinal”이 아니라 원본 stream상의 frame position입니다.

그런데 새 공통 디코드 정의인 `iter_frames_at()`은 seek를 전혀 쓰지 않고, 각 `want` 위치까지 `_core.get_next_frame()`로 처음부터 순차 decode하며 `pos <= want`가 될 때까지 모든 중간 프레임을 버립니다.   이 함수는 phase-1 GPU decode 경로인 `GpuVideoSource.iter_chunks()`에도 그대로 쓰입니다.

**실패 메커니즘:**

* 예를 들어 30fps, 10분짜리 영상은 원본 약 18,000프레임입니다.
* 사용자가 `max_frames=900`으로 샘플링해도 `indices`는 0..18,000 범위에 퍼진 900개 위치가 됩니다.
* `iter_frames_at()`은 900개만 decode하는 것이 아니라 마지막 샘플 위치까지 거의 전체 원본 프레임을 순차 decode합니다.
* 더 나쁘게, phase-1은 source를 여러 unit으로 나누고 각 unit마다 새 `GpuVideoSource`를 만듭니다. `_start_decode()`는 unit slice의 `indices`로 fresh decoder를 생성합니다.  각 unit의 decoder가 자기 첫 샘플 위치까지 다시 stream start부터 walk하므로, 뒤쪽 unit일수록 prefix decode가 반복됩니다.
* 결과적으로 “최대 900프레임만 추출한다”는 사용자의 기대와 달리, 긴 영상에서는 원본 전체 또는 그 이상을 decode하는 경로가 됩니다. GPU extraction보다 CPU/NVDEC sequential scan이 지배해 `max_frames`, multi-GPU sharding, phase-2 tail 최적화의 체감 효과를 크게 잠식할 수 있습니다.

이 변경의 correctness 동기는 이해됩니다. 코드 주석도 approximate/exact seek가 VFR/open-GOP/trimmed clip에서 backend별 frame mapping divergence와 crash를 만든다고 설명합니다.  다만 현재 구현은 robust alignment를 위해 sparse sampling의 성능 계약을 포기한 형태입니다. 이건 minor perf가 아니라 긴 영상에서 처리량과 latency class를 바꾸는 systems issue입니다.

**개선 방향:**

1. **한 source당 하나의 sequential decoder pass만 수행**하십시오. unit별 fresh decoder가 stream start부터 prefix를 반복 decode하지 않도록, source-level decode dispatcher가 sampled frames를 순차 생산하고 unit/worker 쪽으로 fan-out하는 구조가 필요합니다.
2. 또는 **source-level sampled frame cache**를 두십시오. phase-1과 SAM re-decode가 같은 sampled RGB를 공유하면 alignment와 성능을 동시에 얻습니다. 단, host RAM/disk budget을 명시해야 합니다.
3. seek가 안전한 container와 unsafe container를 나누는 **hybrid policy**도 가능합니다. 정상 CFR mp4는 indexed seek, VFR/open-GOP/trimmed 의심 clip만 sequential fallback으로 보내십시오.
4. 테스트에는 “10분 synthetic CFR에서 `max_frames=900`일 때 실제 decoder step count가 O(900)에 가까운지”를 보는 instrumentation이 필요합니다. 지금의 결정성/정합성 테스트만으로는 이 성능 회귀를 잡지 못합니다.

---

## Open domain questions / 결정 필요

### 1. `_auto_seed`의 norm-outlier filter는 “real subject를 배제하지 않는다”는 claim이 아직 과합니다

`_auto_seed()`는 saliency argmax 전에 patch feature norm의 median/MAD modified z-score가 `|z| > 3.5`인 token을 positive 후보에서 제외합니다. 이 필터는 high-norm뿐 아니라 low-norm도 양방향으로 제거합니다.  이후 positive prompt는 필터링된 `work`에서 greedy top-k로 선택되고, 후보가 모두 `-inf`이면 positive 없이 중단됩니다.  negative prompt는 별도로 항상 border 최소 saliency patch에 추가됩니다.

따라서 small foreground가 norm distribution상 outlier인 경우, 실제 subject patch가 positive 후보에서 제거될 수 있습니다. 현재 unit test는 object blob의 norm을 의도적으로 in-band로 두고 artifact만 extreme norm으로 만든 synthetic 한 케이스를 검증합니다.  이 테스트는 “artifact를 피한다”는 회귀는 막지만, “real foreground norm-outlier를 제거하지 않는다”는 claim은 막지 못합니다.

**권장 보강:**

* positive 후보가 0개가 되는 경우에는 unfiltered saliency top-1로 fallback하십시오.
* positive와 border-negative가 같은 cell 또는 매우 가까운 cell에 심기는 경우는 negative를 생략하거나 다음 border 후보로 이동하십시오.
* 실제/합성 fixture에 “작고 고대비인 foreground가 norm-outlier인 경우”와 “textureless close-up subject”를 추가하십시오.
* 문서 claim은 “현재 4~5 clip empirical eval에서는 object token |z|≈1이었다” 수준으로 낮추는 편이 맞습니다.

이 항목은 기본 segmentation 품질 리스크이지만, 클릭 복구가 가능하고 패킷상 auto-seed quality가 별도 축으로 관리되고 있어 confirmed major로 올리지는 않았습니다.

---

### 2. FP8 NO-GO는 의사결정으로는 타당하지만, 적용 범위는 “DINOv2-base measured”로 유지해야 합니다

FP8 spike 문서는 real footage에서 torchao full-FP8이 PCA subspace를 4.7–7.5° 회전시키고 ΔE p95 5.4–9.4를 만든다고 기록합니다.  이 정도면 “repvis의 PCA-color render에는 full-FP8을 shipping하지 않는다”는 결론은 타당합니다.

다만 문서의 결론은 DINOv2 giant/large도 family numerics상 더 나을 이유가 없다고 추론합니다.  이 추론은 실무적으로 합리적이지만, 측정은 `dinov2-base` 한 모델입니다. 이후 누군가 이 결과를 성능/품질 tradeoff 문서에서 재사용할 때는 “base measured, large/giant inferred”를 유지해야 합니다. 패킷도 이 약점을 인지하고 있습니다.

---

## Residual risks

* GPU-only claims은 재실행하지 못했습니다. 특히 async writer의 mask/video identity, saliency 3-arm GPU eval, FP8 reproducibility는 패킷의 자체 증거에 의존했습니다.
* Same-rid mutation guard와 delete/supersede race는 정적 검토상 이전 finding의 핵심 stale-snapshot 문제를 해소한 것으로 보입니다. `run_segment()`와 `run_refit()`는 `LOCK` 안에서 existence check, active-run check, same-rid mutation check, marker add를 수행하고, `delete_runs()`도 같은 `LOCK` 안에서 skip 계산과 rmtree를 수행합니다.
* Async `feats.f16` writer는 temp file 후 `os.replace()`를 쓰고 caller가 반드시 join/error check를 하도록 구현되어 있어 torn final file 방지는 구조상 맞습니다.  다만 writer failure branch는 패킷에서도 least-exercised branch로 인정되어 있어 fault-injection 테스트를 더 넣는 것이 좋습니다.


---

## Findings (triage skeleton — verify each before registering)

_No `JW-GPT-NNN` finding blocks parsed — triage the verbatim reply directly._

---

## Triage (2026-07-10, verified against 0b3af3f)

| # | Finding (요약) | Verdict | Evidence | Task |
|---|---|---|---|---|
| 1 | **Major** — 순차 decode가 `max_frames` sampling 계약을 무력화 (마지막 샘플까지 전량 walk + unit별 prefix 재decode + SAM 2차 full CPU walk) | **REAL (major)** | `video_io.py:63-75` — `want`까지 전 프레임을 순차 decode; `pipeline.py:166-169` — 소스당 최대 `4×ndev` unit 분할; `pipeline.py:235` — unit마다 fresh `GpuVideoSource` → 뒤쪽 unit일수록 stream start부터 prefix 재walk; `pipeline.py:350-352` — SAM re-decode가 CPU에서 두 번째 full walk. 보정: stride가 GOP보다 조밀한 짧은 클립에서는 seek 기반도 사실상 전 프레임을 decode하므로 체감 격차는 "긴 영상 × sparse 샘플"에서 발생 — 그래도 latency class 변화라는 판정은 성립. 참고: `tests/test_frame_alignment.py:157-164`가 exact seek ≡ positional (crash 안 하는 클래스 한정)을 이미 byte-exact로 증명 → 원칙적 hybrid의 근거 | `perf/sparse-decode-full-walk` |
| 2 | auto-seed norm filter가 real subject를 배제할 수 있고 커버리지가 artifact 쪽만 검증 | **부분 REAL (minor)** | 제시된 메커니즘 중 "positive 후보가 모두 -inf이면 positive 없이 중단"은 부정확 — MAD 정의상 토큰의 ≥50%가 \|z\| ≤ 0.6745이므로 전 grid가 flag될 수 없고 peak 1은 무게이트로 항상 식재됨 (`pipeline.py:424-435`). 그러나 (a) norm-outlier인 소형 real subject가 후보에서 제외되는 리스크는 미검증 사실, (b) 완전 uniform 프레임에서 positive와 border negative가 같은 cell에 심기는 충돌은 실제 가능 (`argmax`=cell 0=border, `argmin`도 동일 cell) | `fix/autoseed-outlier-subject-coverage` |
| 3 | FP8 NO-GO 적용 범위를 "base measured, large/giant inferred"로 유지 | **동의 — 조치 불요** | `docs/spikes/fp8-attention.md:359-360`이 이미 정확히 "(base measured; giant/large share the family numerics …)"로 한정 서술 | — |
| 4 | async feats writer failure branch에 fault-injection 테스트 권장 | **REAL (minor)** | 패킷 스스로 least-exercised branch로 인정; 해당 분기 테스트 부재 확인 | `chore/feats-writer-fault-injection` |

Residual-risks 단락의 same-rid guard / delete race / atomic replace 정적 확인은 이전 라운드 판정과
일치 (추가 조치 없음).
