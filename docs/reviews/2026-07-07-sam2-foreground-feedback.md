<!-- jahns-workflow feedback: the body below is the reviewer reply VERBATIM (byte-exact copy via `jw review ingest`) — do not edit it; a triage skeleton is appended beneath it. -->
round: 2026-07-07-sam2-foreground
reviewer: gpt-5.5-pro
ingested: 2026-07-07
source: /tmp/review.md

---

## 검토 범위

* 리뷰 패킷은 `docs/reviews/2026-07-07-sam2-foreground-request.md`이고, 패킷의 대상 커밋은 `071502ac85de130d551521f3e29d7735c5796c08`입니다.
* 현재 `main` 최신은 `e26a140c4e00e8392beae89e83accc0906fd3fcb`이며, 이는 `docs(round): close 2026-07-07-sam2-foreground` 문서 마감 커밋입니다. 코드 대상 커밋 `071502ac...`에서 1커밋 앞선 상태라서 리뷰를 중단할 mismatch는 아니라고 판단했습니다.
* 커밋 범위는 초기 커밋 `a56e51c053e6ceca750e5608d5f75da4ac0d73d7`부터 현재 `HEAD`까지로 보았습니다.
* 컨테이너에서 직접 `git clone`은 DNS 문제로 실패했기 때문에, GitHub Connector로 현재 `HEAD`의 파일 원문과 커밋 메타데이터를 조회해 정적 검토했습니다. GPU/브라우저 E2E는 직접 재실행하지 못했습니다.

---

## 확정 finding

### 1. Critical — 세그먼트 보정 클릭이 항상 `frame 0` 프롬프트로 적용됩니다

현재 클라이언트는 클릭 위치를 `[x, y, label]`만으로 저장하고 서버에 전송합니다. 프레임 번호나 클릭 시점이 payload에 없습니다.

서버/파이프라인 쪽도 `_segment(..., seed_frame=0)` 기본값을 그대로 사용하고, `segment_and_render()`는 프레임 인덱스를 전달하지 않습니다.   `sam.segment()` 자체도 모든 point prompt를 `seed_frame`에 배치한다고 명시되어 있으며 기본값은 `0`입니다.

**실패 메커니즘:** 사용자가 재생 중인 후반 프레임에서 움직인 객체를 클릭하면, 그 좌표가 프레임 0의 좌표로 해석됩니다. 프레임 0에서 해당 좌표가 배경이거나 다른 객체라면, SAM2는 잘못된 positive/negative prompt를 받고 전체 propagation이 오염됩니다. 특히 카메라/피사체 움직임이 있는 클립에서는 “클릭해서 보정 가능”이라는 UX/알고리즘 계약이 깨집니다.

**개선 방향:**

* 클라이언트 point schema를 `[{x, y, label, frame}]` 또는 `[x, y, label, frame]`로 바꾸십시오.
* `frame = round(video.currentTime * seg.fps)`를 기준으로 계산하되, 서버에서 `0 <= frame < meta.frames`로 재검증해야 합니다.
* 서버는 point들을 frame별로 그룹화하고, SAM2 session에 해당 `frame_idx`마다 prompt를 추가한 뒤 propagate해야 합니다.
* 단기 우회라면 UI를 프레임 0 보정 전용으로 제한해야 합니다. 즉, 보정 모드 진입 시 모든 비디오를 `currentTime=0`으로 seek/lock하고 “frame 0 prompt only”를 명확히 해야 합니다.

---

### 2. Major — SAM2 실패가 all-foreground fallback으로 숨겨지고, 사용자는 복구할 수 없습니다

`_segment()`는 SAM2 실행 중 어떤 예외가 나도 전부 catch한 뒤 `(T, oh, ow)` all-foreground mask를 만들고 `seg.available=False`로 돌려줍니다. 빈 mask도 `available=False`가 됩니다.

그런데 클라이언트는 `seg.available`이 true일 때만 click layer와 `↺ reset` / `Refit` 컨트롤을 붙입니다.  즉 SAM2가 실패하거나 auto-seed가 empty mask를 만들면, 결과는 “마스크 없는 PCA 영상”으로 정상 완료된 것처럼 보이고, 사용자는 클릭 보정도 할 수 없습니다.

**실패 메커니즘:**

* HF weight download 실패, SAM2 API shape mismatch, CUDA OOM, processor 예외, empty mask 등이 모두 정상 렌더로 위장됩니다.
* 초기 렌더는 all foreground라 배경 제거가 전혀 되지 않습니다.
* `seg.available=False` 때문에 UI 보정 도구가 사라져서 사용자가 직접 positive point를 찍어 복구할 수 없습니다.
* 현재 테스트도 이 상황을 잡지 못합니다. GPU 테스트는 `seg` 메타 존재와 `/segment`, `/refit` 200 응답만 확인하고, `seg.available`, mask foreground ratio, 비디오의 black-background 비율, 또는 클릭 전후 mask 변화를 assert하지 않습니다.

**개선 방향:**

* SAM2 예외와 empty mask를 구분하십시오.

  * 모델/API/메모리 예외: `seg.error`를 기록하고, initial run에서는 실패를 명시하거나 degraded result로 표시.
  * empty mask: `available=True`, `mask_empty=True`처럼 보정 UI는 유지.
* initial auto-seed가 실패해도 사용자가 새 positive point를 줄 수 있도록 click layer를 표시해야 합니다.
* refinement 요청에서 SAM2가 실패하면 기존 `pca.mp4`, `masks.u1`, `meta.json`을 유지하고 500/422를 반환하십시오. 지금처럼 all-foreground로 덮어쓰면 좋은 상태가 파괴될 수 있습니다.
* 테스트는 최소한 다음을 추가해야 합니다:

  * `meta["seg"]["available"] is True`
  * `masks.u1` unpack 후 foreground ratio가 `0 < ratio < 1`
  * `/segment` 후 `masks.u1` 또는 decoded frame black ratio가 실제로 변함
  * SAM2를 monkeypatch로 강제 실패시켰을 때 endpoint가 조용히 200/all-foreground로 성공하지 않음

---

### 3. Major — `from_pretrained()` 전역 dtype race를 막는 lock이 모델군 간에는 공유되지 않습니다

`extract.py`의 manager는 `from_pretrained(dtype=...)`가 torch global default dtype을 건드려 thread-safe하지 않다고 설명하고, `_load_lock`으로 extractor 모델 생성을 직렬화합니다.  `sam.py`도 동일한 이유로 SAM2 construction을 `_load_lock` 뒤에 둡니다.

하지만 `run_group()`은 extractor warm thread들과 SAM warm thread들을 동시에 시작합니다. 두 lock은 서로 다른 객체이므로, extractor `from_pretrained()`와 SAM2 `from_pretrained()`가 동시에 실행될 수 있습니다.

**실패 메커니즘:** 코드 주석의 가정이 맞다면, 현재 구조는 정확히 그 race를 다른 manager 사이에서 재도입합니다. 결과적으로 한 모델이 의도치 않게 fp32/bf16 dtype으로 로드되거나, forward 중 dtype mismatch가 재발할 수 있습니다.

**개선 방향:**

* `repvis/model_load_lock.py` 같은 단일 전역 lock을 만들고, extractor와 SAM2 모두 동일 lock을 사용하십시오.
* 또는 warm-up 순서를 명시적으로 두 단계로 나누십시오: 모든 extractor warm 완료 후 SAM warm.
* CI에 cold-start 병렬 로딩 테스트를 추가하십시오. 예: 2 GPU 또는 fake multi-device 설정에서 extractor/SAM warm을 반복 호출하고 model parameter dtype 및 첫 forward dtype을 assert.

---

### 4. Major — server-side point validation이 좌표 범위와 finite 여부를 검증하지 않습니다

`_parse_points()`는 payload가 list인지, 각 point가 `[number, number, 0|1]` 형태인지 정도만 확인합니다. 좌표가 finite인지, source frame 안에 있는지, 현재 run의 실제 width/height와 맞는지는 검증하지 않습니다.  이후 `segment_and_render()`는 이 좌표를 그대로 float/int 변환해 SAM2로 넘깁니다.

**실패 메커니즘:** 악의적이거나 stale한 클라이언트가 `[-1e9, 1e9, 1]`, `NaN`, `Infinity`, 또는 source 크기를 벗어난 좌표를 보내면, SAM2 processor가 예외를 내거나 비정상 mask를 만들 수 있습니다. 현재는 finding 2의 broad catch 때문에 이런 입력도 all-foreground overwrite로 이어질 수 있습니다.

**개선 방향:**

* `_parse_points()`는 shape/type만 검사하고, run-specific validation은 `_load_run()` 이후 source frame 크기를 안 뒤 수행하십시오.
* `math.isfinite(x)`, `math.isfinite(y)`를 확인하십시오.
* `0 <= x < W`, `0 <= y < H`, `label in {0,1}`, `0 <= frame < T`를 서버에서 강제하십시오.
* invalid point는 SAM2까지 보내지 말고 400/422로 거절해야 합니다.
* 테스트에 out-of-bounds, NaN/Infinity, negative, excessive point count를 추가하십시오.

---

### 5. Major — completed run 삭제와 segment/refit 재렌더가 상호 배제되지 않습니다

`/segment`와 `/refit`은 `_active_run_ids()`로 queued/running group만 확인한 뒤 `EXEC.submit(...).result()`로 GPU 작업을 실행합니다.   그러나 `/api/runs` DELETE는 active group id만 제외하고 completed run directory를 삭제합니다. segment/refit 중인 run은 active group에 들어가지 않습니다.  source 삭제도 동일하게 active group만 막고, segment/refit 작업 상태는 보지 않습니다.

**실패 메커니즘:** 사용자가 `Refit` 또는 `/segment` 요청을 실행 중인 동안 다른 탭/클라이언트에서 “clear completed results” 또는 source delete를 호출하면, 재렌더 작업이 읽고 있는 `run_dir`, `feats.f16`, `masks.u1`, source video가 삭제될 수 있습니다. 결과는 500, temp file 실패, 깨진 workspace 상태, 또는 부분적으로 교체된 `pca.mp4`가 될 수 있습니다.

**개선 방향:**

* `ACTIVE_GPU_TASKS` 또는 `ACTIVE_RUN_MUTATIONS: set[run_id]`를 두고 segment/refit 시작부터 finally까지 등록하십시오.
* `/api/runs` DELETE, `/api/sources/{sid}` DELETE, 같은 `(source, model)` 재실행 supersede 로직은 이 set을 확인해야 합니다.
* 더 단순하게는 destructive endpoint도 `EXEC`를 통해 serialize하고, 실행 전후에 run/source 존재를 재검증하십시오.
* segment/refit이 실행 중인 run은 workspace에서 busy 상태로 노출해 UI가 clear/delete를 막도록 하십시오.

---

## Open domain questions / 결정 필요 사항

### A. “frame alignment is exact” 주장은 현재 코드와 검증만으로는 너무 강합니다

phase 1은 `GpuVideoSource`에서 CUDA `VideoDecoder(..., seek_mode="approximate")`로 명시 index들을 디코드합니다.   반면 SAM2 재세그먼트 경로는 CPU `VideoDecoder(..., seek_mode="approximate")`로 같은 index들을 다시 디코드합니다.  메타에는 `frame_indices`가 저장되지만, 테스트는 길이 일치만 확인합니다.

같은 index list를 쓰는 것은 좋은 invariant이지만, `approximate` seek + CPU/GPU decoder backend 차이까지 포함해서 “pixel/mask/frame exact”라고 단정하려면 추가 검증이 필요합니다.

**권장 검증:**

* phase 1에서 실제 디코드된 frame의 cheap checksum 또는 perceptual hash를 선택적으로 기록하십시오.
* `_decode_source_frames()` 재디코드 결과의 checksum과 비교하는 debug/test mode를 두십시오.
* open-GOP, VFR, `ffmpeg -ss -c copy` 계열 clip을 fixture로 넣고 frame count/order/hash를 검증하십시오.
* 만약 exact alignment가 제품 핵심이면, SAM도 phase 1에서 디코드한 RGB frame을 저장하거나, 최소한 동일 decoder/device path를 재사용하는 편이 안전합니다.

### B. Thin structure refit 기준은 의도인지 확인이 필요합니다

`refit_and_render()`는 pixel mask를 feature grid로 내릴 때 `adaptive_avg_pool2d(...) > 0.5`를 씁니다. 즉 patch 면적의 과반이 foreground인 token만 refit 대상입니다.  얇은 팔, 다리, 도구, 자전거 바퀴 같은 구조물은 SAM mask에는 남아도 PCA refit에서는 제외될 수 있습니다.

이것이 의도라면 괜찮습니다. “foreground 전체의 색 대비를 최대화”가 목표라면 threshold를 낮추거나, soft weighting으로 PCA/refit을 하는 편이 domain적으로 더 맞습니다.

---

## 권장 우선순위

1. **먼저 point schema에 `frame`을 추가하십시오.** 현재 보정 UX의 가장 큰 correctness hole입니다.
2. **SAM2 실패를 all-foreground 성공으로 숨기지 마십시오.** 실패는 실패로 노출하고, empty/low-quality mask는 사용자가 복구할 수 있게 해야 합니다.
3. **검증을 강화하십시오.** 지금 GPU 테스트는 SAM2가 완전히 죽어도 통과할 수 있는 구조입니다.
4. **전역 model load lock을 공유하십시오.** 현재 warm-up 병렬화는 코드 주석이 경고한 dtype race를 manager 간에 다시 열고 있습니다.
5. **run mutation lifecycle을 명시하십시오.** segment/refit/delete/supersede가 같은 run_dir를 동시에 건드리지 못하게 해야 합니다.


---

## Findings (triage skeleton — verify each before registering)

_No `JW-GPT-NNN` finding blocks parsed — triage the verbatim reply directly._
