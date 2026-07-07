<!-- jahns-workflow feedback: the body below is the reviewer reply VERBATIM (byte-exact copy via `jw review ingest`) — do not edit it; a triage skeleton is appended beneath it. -->
round: 2026-07-07-review-fixes
reviewer: gpt-5.5-pro
ingested: 2026-07-07
source: /tmp/review.md

---

## 리뷰 범위

* 패킷: `docs/reviews/2026-07-07-review-fixes-request.md`
* 패킷상 Reviewed commit: `1262e1572bd1daf56b4d5cf34497ffce47891cfe`, diff base: `071502ac85de130d551521f3e29d7735c5796c08`
* 현재 `main`은 `db60f7e66a3eb4f48d5121a309381082077cbe50`로, 코드 변경이 아니라 review-fixes 라운드 문서 마감 커밋입니다.
* 컨테이너에서 `git clone`은 DNS 실패로 불가했습니다. 대신 GitHub Connector로 대상 커밋의 실제 파일 내용을 직접 조회했습니다. GPU/브라우저 E2E는 재실행하지 못했습니다.

---

## 확정 finding

### 1. Major — `ACTIVE_RUN_MUTATIONS`는 삭제와 segment/refit 사이의 check-then-delete race를 완전히 막지 못합니다

패킷의 claim 3은 “`DELETE /api/runs` 또는 `DELETE /api/sources` 중 segment/refit 중인 run dir이 rmtree되지 않는다”는 것입니다. 실제 수정은 `ACTIVE_RUN_MUTATIONS` set을 두고, segment/refit endpoint에서 run id를 추가한 뒤 작업을 `EXEC`에 submit하는 방식입니다.

문제는 destructive path가 **현재 active set의 snapshot만 읽고 lock을 놓은 뒤 실제 삭제를 수행**한다는 점입니다. `DELETE /api/runs`는 lock 안에서 `skip = _active_run_ids() | ACTIVE_RUN_MUTATIONS`를 한 번 계산한 뒤, lock 밖에서 `RUNS_DIR`을 순회하며 `shutil.rmtree()`를 실행합니다.

구체적인 실패 interleaving은 다음과 같습니다.

1. 요청 A: `DELETE /api/runs`가 lock을 잡고 `skip`을 계산합니다. 이 시점에는 `rid`가 아직 `ACTIVE_RUN_MUTATIONS`에 없습니다.
2. 요청 A가 lock을 놓습니다.
3. 요청 B: `/api/runs/{rid}/segment`가 lock을 잡고 `ACTIVE_RUN_MUTATIONS.add(rid)`를 수행합니다.
4. 요청 A가 stale `skip`으로 디렉터리 순회를 계속해서 `rid` run dir을 삭제합니다.
5. 요청 B의 `segment_and_render()`는 `_load_run()`에서 `meta.json`, `feats.f16`, `state.pt`를 읽거나 이후 source frame을 재디코드해야 하므로, 삭제 타이밍에 따라 404/500 또는 부분 상태가 됩니다.

`DELETE /api/sources/{sid}`도 같은 계열의 race가 있습니다. `derived` run list를 lock 밖에서 만들고, lock 안에서 active mutation 여부를 확인한 뒤, lock 밖에서 source dir과 derived run dirs를 삭제합니다.  이 사이에 segment/refit이 mutation을 등록하면 source video 또는 run dir이 mutation 중 삭제될 수 있습니다.

`_persist_run()`의 supersede 삭제도 stale snapshot pattern입니다. `ACTIVE_RUN_MUTATIONS`를 lock 안에서 복사한 뒤, lock 밖에서 기존 `(source, model)` run dir을 삭제합니다.  `EXEC(max_workers=1)` 때문에 새 run과 segment/refit GPU 작업 자체는 직렬화되지만, endpoint가 mutation을 이미 accepted/registered한 뒤 supersede snapshot이 오래되면 “등록된 mutation 대상이 삭제되지 않는다”는 의도는 여전히 보장되지 않습니다.

**개선 방향:**

* destructive operation을 “snapshot 후 lock 밖 삭제”로 두지 마십시오.
* 최소 수정은 `DELETING_RUNS` / `DELETING_SOURCES`를 추가하는 것입니다.

  * lock 안에서 victim을 계산하고, `ACTIVE_RUN_MUTATIONS`와 충돌하지 않는지 확인한 뒤, victim을 `DELETING_RUNS`에 등록합니다.
  * segment/refit은 lock 안에서 `rid not in DELETING_RUNS`를 확인한 뒤 `ACTIVE_RUN_MUTATIONS.add(rid)`까지 원자적으로 수행합니다.
  * 실제 `rmtree`는 lock 밖에서 하되, finally에서 `DELETING_RUNS`를 정리합니다.
* 더 강한 방식은 lock 안에서 victim run dir을 atomic rename으로 `.trash/<uuid>`로 옮긴 뒤 lock을 놓고 trash를 삭제하는 것입니다. 그러면 segment/refit의 존재 확인과 mutation 등록이 같은 lock discipline을 따를 때 stale path를 잡을 수 없습니다.
* `/segment`와 `/refit`의 `feats.f16` 존재 확인도 lock 밖 precheck가 아니라, 삭제 상태 확인과 mutation 등록이 끝나는 lock block 안쪽으로 이동하는 편이 맞습니다.
* 회귀 테스트는 barrier를 둔 concurrency test가 필요합니다. 예: monkeypatch `segment_and_render()`가 시작 직후 block되게 한 뒤, 동시에 `DELETE /api/runs`와 `DELETE /api/sources/{sid}`를 호출하고, 409 또는 skip 및 run dir 생존을 assert해야 합니다.

---

## 확인된 수정 사항 중 추가 finding은 없음

다음 항목들은 정적 검토 기준으로는 이전 finding의 핵심 실패 메커니즘이 해소된 것으로 보입니다.

* point schema는 `[x, y, label, frame]`으로 확장됐고, 클라이언트는 `currentTime / duration * seg.frames`로 frame을 계산해 전송합니다.
* SAM2 wrapper는 point를 frame별로 그룹화하고 각 conditioning frame마다 `add_inputs_to_inference_session()` 후 `model(... frame_idx=f)`를 호출합니다.  HuggingFace 쪽 `propagate_in_video_iterator()`도 start frame 기본값을 “input point가 있는 가장 이른 frame”으로 잡고, reverse일 때 그 frame부터 0 방향으로 처리합니다.
* refine 경로는 initial-run fallback wrapper `_segment()`를 쓰지 않고 `_run_sam()`을 직접 호출하므로, SAM failure가 기존 `pca.mp4`/`masks.u1`/`meta["seg"]`를 all-foreground로 clobber하지 않는 구조입니다.
* server point validation은 4-tuple, finite coordinate, label, non-negative integer frame을 400으로 거르고, run-specific bounds는 pipeline에서 422로 매핑합니다.
* shared model load lock은 `repvis/modelload.py`로 분리되었고, extractor와 SAM2 양쪽이 같은 `LOAD_LOCK`을 사용합니다.
* refit은 hard `>0.5` gate 대신 foreground fraction weight를 사용하고, `refit_display(..., weights=...)`에서 weighted mean 및 `sqrt(w)`-scaled SVD를 수행합니다.

---

## Open questions / 결정 필요

### A. Weighted quantile의 “unweighted와 동일하게 reduce” 주장은 엄밀히는 재검토 필요

`_weighted_quantile()`은 midpoint plotting position CDF를 쓰며, 주석은 equal weights일 때 `torch.quantile`과 맞는다고 설명합니다.  그러나 작은 foreground token 수에서는 PyTorch linear quantile과 정확히 같지 않습니다. 예를 들어 token 수가 16이면 2% quantile은 midpoint CDF에서 최솟값으로 clamp되는 반면, `torch.quantile` linear interpolation은 최솟값과 두 번째 값 사이가 됩니다.

이것은 segmentation correctness blocker는 아닙니다. 다만 refit 결과의 display range가 작은 foreground mask에서 outlier에 더 민감해질 수 있습니다. “weighted PCA의 basis”는 equal weights에서 unweighted와 같은 방향으로 reduce되지만, “lo/hi quantile까지 동일”이라고 주장하려면 별도 정의를 명확히 하거나 weighted quantile 방식을 바꿔야 합니다.

---

## Residual risks

* GPU test, live browser E2E, SAM2 실제 모델 behavior는 직접 재실행하지 못했습니다.
* 패킷도 인정하듯 GPU phase-1 decode와 CPU re-decode의 exact frame alignment는 아직 open입니다. 추가된 테스트는 같은 CPU re-decode의 결정성 확인에 가깝고, GPU decode path와 CPU SAM re-decode path의 open-GOP/VFR exactness까지 닫지는 못합니다.
* 현재 남은 major risk는 concurrency입니다. `ACTIVE_RUN_MUTATIONS`의 개념은 맞지만, 삭제/슈퍼시드 쪽이 stale snapshot으로 동작해서 claim 3의 “mutex actually excludes”는 아직 완전히 성립하지 않습니다.


---

## Findings (triage skeleton — verify each before registering)

_No `JW-GPT-NNN` finding blocks parsed — triage the verbatim reply directly._
