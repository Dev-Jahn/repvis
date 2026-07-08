<!-- jahns-workflow feedback: the body below is the reviewer reply VERBATIM (byte-exact copy via `jw review ingest`) — do not edit it; a triage skeleton is appended beneath it. -->
round: 2026-07-08-parallel-sweep
reviewer: fable-5
ingested: 2026-07-08
source: /tmp/review.md

---

# 2026-07-08-parallel-sweep 외부 리뷰 회신

## 검토 범위

* 패킷: `docs/reviews/2026-07-08-parallel-sweep-request.md`
* Reviewed commit: `5776596a0cd993fc1af6ef70f92d2434a87ec520` (diff base `1262e1572bd1daf56b4d5cf34497ffce47891cfe`)
* 방법: 레포 직접 접근. `git diff`/`git log`로 라운드 변경 전체를 본 뒤 `repvis/server.py`, `repvis/video_io.py`, `repvis/pipeline.py`, `repvis/pca.py`, `repvis/sam.py`, `static/index.html`, `static/app.js`, `tests/test_api.py`, `tests/test_frame_alignment.py`, `tests/test_pca.py` 현재본을 정독했습니다.
* CPU 검증 실행: `uv run pytest -q` → **41 passed, 5 skipped** (barrier race 3종 + phantom-upload + auth + frame-alignment CPU 전부 green). 추가로 (a) claim 1을 깨는 **동시 same-rid mutation 재현 스크립트**, (b) `_weighted_quantile` 적대적 numerics probe(n=2..64 × 101-point q grid, zero-weight fuzz 200회, tie/degenerate 케이스)를 직접 작성·실행했습니다.
* GPU 작업은 전부 점유 중이라 재실행하지 않았습니다(NVDEC cross-backend 테스트, cache sha256 byte-identity, 3-arm seed eval, backbone bench, fp8 spike). 해당 항목은 코드 로직 + 제출된 증거 검토로 대체하고 Residual risks에 명시합니다. transformers 5.12.1의 `Sam2VideoInferenceSession` 소스는 로컬 설치본으로 직접 확인했습니다.
* Out of scope 준수: jahns-workflow 하네스, in-flight 브랜치(fp8 gate, phase-2 tail, auth-hardening, saliency artifact-token)는 다루지 않았습니다.

---

## 확정 finding

### 1. Major — `ACTIVE_RUN_MUTATIONS`가 set이라서, 동일 rid에 대한 두 번째 in-flight mutation의 보호가 첫 번째의 `discard`로 벗겨집니다 (claim 1 반증, 실행 재현 확인)

single-LOCK discipline 자체(victim 선정 + rmtree를 한 `with LOCK:` 안으로)는 올바르게 들어갔고, 단일 mutation vs delete 조합은 barrier 테스트대로 배제됩니다. 그러나 **mutation 마커의 수명 관리**에 구멍이 있습니다.

`run_segment`의 LOCK 구간(`repvis/server.py:541-546`)은 `rid in ACTIVE_RUN_MUTATIONS`를 검사하지 않습니다 — 409 조건은 `_active_run_ids()`(그룹)뿐입니다. 따라서 같은 rid로 `/segment`(또는 `/refit`) 요청 두 개가 동시에 들어오면 둘 다 통과해 `add(rid)`(set이라 두 번째는 no-op) 후 각각 EXEC에 submit합니다. 첫 요청이 끝나면 finally(`server.py:558-560`)의 `ACTIVE_RUN_MUTATIONS.discard(rid)`가 **두 번째 요청이 아직 큐/실행 중인데도** 마커를 제거합니다. 그 순간부터:

* `delete_runs`(`server.py:606`)의 `skip`에 rid가 없음 → **실행 중인 segment 밑에서 run dir을 rmtree**
* `delete_source`(`server.py:381`)의 derived-mutation 검사 통과 → source video + run dir 삭제 (두 번째 segment는 이후 `_source_path` 재디코드/파일 쓰기 대상 상실)
* `_persist_run`(`server.py:156`)의 supersede skip도 무효

실행 재현(스크립트로 강제 interleaving: A `/segment` 실행 중 B `/segment` 등록 → A 종료·discard → B 실행 중 `delete_runs`):

```
rid in ACTIVE_RUN_MUTATIONS while segment #2 in flight: False
delete_runs removed=1 (expected 0 if protected)
feats.f16 present during in-flight segment #2: False
RACE CONFIRMED: run dir rmtree'd while a /segment mutation was executing
```

결과 범위: 두 번째 mutation의 스퓨리어스 500이 최선이고, 최악은 rmtree가 dir 트리를 걷는 동안 `segment_and_render`가 `masks.u1`/`meta.json`(os.replace)/`pca.mp4`를 다시 써서 **부분 부활한 좀비 run dir**(meta.json+pca.mp4는 있는데 feats.f16은 없는, workspace에는 잡히는 셀)이 남는 경우입니다. UI에서 더블클릭 두 번이면 same-rid 동시 요청이 나오므로 도달 가능성은 현실적입니다.

**수정 제안**: EXEC이 어차피 실행을 직렬화하므로 의미상 가장 깨끗한 수정은 LOCK 구간에서 `if rid in ACTIVE_RUN_MUTATIONS: raise HTTPException(409, "this run is being mutated")`로 동일 rid 동시 mutation을 거부하는 것입니다(refcount dict도 가능하지만 과합니다). 회귀 테스트는 기존 barrier 테스트의 변형으로: same-rid segment 두 개를 겹치게 한 뒤 첫 번째 완료 시점에 `delete_runs`를 쏘고 dir 생존을 assert.

### 2. Major — `_SegCache` warm hit이 세션 생성 시점의 GPU에 고정되어 있어, device 선택이 바뀌면 warm click이 cross-device 오류로 500이 됩니다 (mask 오염은 아님)

`segment_and_render`는 매 호출마다 `_load_run`에서 `dev = select_devices()[0]`를 새로 뽑고(`repvis/pipeline.py:776`), cache hit이면 그 **새 dev의 모델**로 `sam.segment_session(sess, pts, dev)`를 호출합니다(`pipeline.py:840`, `repvis/sam.py:154`의 `MANAGER.get(device)`). 그런데 세션은 build 시점 device에 고정입니다: `sam.build_session`이 `inference_device=dev`로 생성하고(`sam.py:136-139`), transformers 5.12.1의 `Sam2VideoInferenceSession`은 point 입력·캐시 feature·프레임을 전부 `self.inference_device`로 옮깁니다(`modeling_sam2_video.py:145, 215, 295, 317` 확인). cache sig는 `(source_id, T)`뿐이라(`pipeline.py:809`) device 불일치를 걸러내지 못합니다.

`select_devices`는 per-call로 emptiest-first 재평가되는 것이 설계 의도이고(공유 박스에서 VRAM 부하는 클릭 사이에 흔히 변합니다), device가 바뀐 순간의 warm click은 dev_new 모델 × dev_old 입력의 cross-device op으로 RuntimeError → `except BaseException` 경로가 cache를 drop하고 raise → HTTP 500. 잘못된 mask가 나오기 전에 fail-fast하므로 claim 5의 correctness는 유지되지만, 이 라운드의 warm-click deliverable 자체가 멀티 GPU 구성에서 간헐적으로 깨지고(재시도 시 cold rebuild로 자가 복구), 사용자에겐 원인 불명의 500입니다. 현재 라이브 서버처럼 `REPVIS_GPUS`로 단일 GPU에 핀하면 발현하지 않습니다 — 코드 레벨 결함이지 배포 레벨 사고는 아닙니다.

**수정 제안**: sig에 device를 포함(`(source_id, T, dev)` — drift 시 cold miss로 새 device에서 재빌드)하거나, hit 시 `dev`를 `sess.inference_device`로 덮어쓰는 것. 전자가 select_devices의 부하 회피 의미를 보존하므로 낫습니다. GPU를 확보하면 "세션을 cuda:A에서 build → cuda:B 모델로 segment_session" 한 줄 재현으로 확정 가능합니다.

### 3. Minor — `create_runs`의 등록 전 창: run dir 생성/소스 검증이 `GROUPS` 등록보다 앞서 있어, 그 사이 delete가 끼어들 수 있습니다 (claim 1의 절대 문장에 대한 두 번째 반례; 결과는 무해한 실패)

`create_runs`는 소스 검증(`server.py:441`)과 `rd.mkdir`(`server.py:446`)를 끝낸 뒤에야 LOCK을 잡고 `GROUPS[gid]`를 등록합니다(`server.py:454-455`). 그 창에서 `delete_runs`는 아직 어떤 skip-set에도 없는 새 run dir들을 rmtree할 수 있고, `delete_source`는 방금 검증된 소스를 지울 수 있습니다(활성 그룹이 없으므로 409에 안 걸림). 이후 그룹은 phase-2 feats 쓰기 또는 probe에서 실패해 error로 표면화되고, task finally가 잔여 dir을 청소하므로 **데이터 오염은 없습니다**. "no thread interleaving" 문장의 자구는 깨지지만 결과는 우아한 실패입니다. mkdir+검증을 LOCK 안(GROUPS 등록과 동일 임계구역)으로 옮기면 닫힙니다.

---

## 공격했으나 생존한 claim (처분)

* **Claim 2 (event-loop 무차단)** — 생존. async 핸들러는 `upload_source`(LOCK 구간을 `_materialize_source`로 묶어 `asyncio.to_thread`, `server.py:337`)와 `events`(`_events_tick`을 to_thread, `server.py:508`)뿐이고, 나머지 LOCK 사용처는 전부 sync def(Starlette threadpool)입니다. 루프 스레드에서 LOCK을 잡는 경로는 없습니다. 참고: upload의 스트리밍 `out.write`(`server.py:326`)는 여전히 루프 스레드의 blocking 디스크 쓰기지만 이번 라운드 이전부터 있던 것이고 로컬 fs에선 무시 가능 수준입니다.
* **Claim 3 (auth 완전성)** — 생존. 게이트가 default-deny(exempt 3개 외 전부 401, `server.py:207,235`)라 우회는 "보호 자원이 exempt 경로 문자열로 서빙"되어야 하는데 세 exempt 핸들러 모두 고정 응답입니다. 미들웨어와 라우터가 같은 디코드된 scope path를 쓰므로 인코딩 불일치 우회도 없습니다. `/static`·`/docs`·SSE·미디어 모두 미들웨어를 통과하고, 비교는 `secrets.compare_digest`, unset 시 기존 open 동작 보존을 테스트로 확인. Secure flag/rate-limit/URL-fragment 토큰의 히스토리 잔존은 이미 `fix/auth-hardening`으로 트래킹됨을 확인했습니다.
* **Claim 4 (positional decode identity)** — 픽스처된 클립 클래스 범위에서 생존. `iter_frames_at`(`video_io.py:38-75`)의 pre-roll skip/end-clamp/EOF 로직에서 반례를 찾지 못했고(중복 인덱스 OK, 오름차순 전제는 두 호출자 모두 충족, `compute_indices`의 linspace는 step≥1이라 strictly increasing), CPU 정렬 테스트 12종을 재실행해 green 확인. 클라이언트 클릭→frame 매핑도 CFR로 재인코딩된 pca.mp4 위에서 일어나므로(`app.js:452,613-618`) VFR 원본이어도 일관됩니다. NVDEC cross-backend는 재실행 불가 — Residual 참조.
* **Claim 5 (cache correctness)** — correctness(잘못된 mask 불가)는 생존: 입력(feats/frame_indices/소스)이 run 단위로 불변, 실패 시 drop, sig guard, 삭제/supersede/refit 전 경로에서 drop이 LOCK 안에서 호출됨을 확인. 문서화된 drop→re-put race도 correctness 무해 확인 — 단 re-put이 LRU=2에서 **살아있는 세션을 밀어낼 수 있어** perf blip은 가능(문서의 "simply never hit"보다 반 발 나쁨). device 고정 문제는 finding 2로 분리.
* **Claim 7 (weighted quantile)** — 생존. 제출 테스트보다 넓게 공격: equal weights에서 n=2..64 전부, q 101점(0/1 포함), atol 1e-9로 `torch.quantile`과 일치; zero weight 섞은 200회 fuzz에서 CDF·출력 단조성/bracketing 유지; 단조성은 대수적으로도 확인(cdf 증분의 분자 = `w_i·above_i + below_i·w_{i+1} ≥ 0`). 유일한 정의역 경계 — 양수 weight가 하나뿐이면 zero-weight 이웃 쪽으로 보간됨(이상값 10.0 대신 12.5/15/17.5) — 은 refit이 `keep = w > 0`으로 사전 필터하므로(`pipeline.py:891-893`) 프로덕션 도달 불가.
* **Claim 8 (encode drop)** — 생존. 측정치(encode 2.5–3.6s / 87.7s)는 한 구성이지만, SAM 비용이 입력 해상도에 거의 불변(내부 1024 리사이즈)인 반면 encode는 해상도에 선형이어도 4K에서조차 ~10–14s 수준이라 병목 역전이 없다는 추론은 타당합니다. 실험을 `wip/parallel-joint-encode`에 보존한 처분도 적절.

---

## Open questions / 결정 필요

### A. Claim 6의 전칭 문장("어떤 클립 클래스에서도 single-point보다 나쁘지 않다")은 n=5 eval로는 지지되지 않습니다

gate 로직 자체(cosine to peak-1 vs cosine to mean, `pipeline.py:405`)는 상대 기준이라 튜닝 상수가 없다는 장점이 있지만, **negative prompt는 gate 없이 항상 심어집니다**(`pipeline.py:411-415`). 피사체가 프레임 경계까지 가득 차면 patch mean ≈ 피사체 prototype이 되어 "최저 saliency border patch"가 피사체 위에 놓이는 것이 구성상 강제됩니다 — 이때 OLD seed에는 없던 negative가 피사체에 박히므로 "never worse than single-point"의 반례 후보입니다. 제출된 fill_frame 0.407→0.9995는 특정 클립 하나의 결과입니다. 제안: (i) negative에도 대칭 gate(feature가 f1보다 mu에 코사인-가까울 때만 심기)를 걸거나, (ii) claim을 "5-clip eval에서 무회귀"라는 경험적 진술로 강등하고 invariant로 취급하지 않기. GPU 확보 시 border까지 채우는 클로즈업 클립 1개로 판별 가능합니다.

---

## Residual risks

* **GPU 증거 일체 미재현** (GPU 전 대수 점유): NVDEC cross-backend identity 24건, cache cold/warm sha256 byte-identity ×3, 3-arm seed eval, backbone bench(1.09x/1.39x/1.29x/1.63x), fp8 spike. 코드 로직과 제출 아티팩트 검토로만 판정했습니다. 특히 claim 5의 byte-identity는 `reset_tracking_data`가 memory-bank state를 완전히 비운다는 transformers 내부 가정에 얹혀 있어, transformers 버전 업 시 재검증이 필요합니다.
* **컨테이너/코덱 커버리지**: 정렬 픽스처 4종이 전부 x264+MP4입니다. `_VIDEO_EXTS`는 webm/mkv/avi도 받는데(`server.py:45`), 해당 demuxer들의 pts/`begin_stream_seconds` 의미론(pre-roll skip의 `1e-6` epsilon 포함)과 HEVC open-GOP은 미검증입니다. positional 정의 자체는 demuxer-중립이지만 pre-roll 판정은 아닙니다.
* **torchcodec private API** (`_core.get_next_frame`, 0.14.0 핀): CPU 테스트가 API 제거는 잡지만 NVDEC 측 동작 변화는 CPU CI에서 조용히 지나갈 수 있음(패킷 자인 사항과 일치).
* **pre-fix VFR run 미마이그레이션**: 구 timestamp-index 정의로 만들어진 run의 mask/feats는 그대로임(자인 사항). 사용자가 구 run에 클릭을 이어가면 새 positional 재디코드와 구 feats가 어긋납니다 — 최소한 구 run 감지·재실행 유도라도 고려할 것.
* **`_SegCache` host RAM 상한**: 세션은 processor가 1024²로 리사이즈한 float32 프레임을 통째로 보관합니다(`processing_sam2_video.py:549-552` 확인). max_frames=900이면 프레임만 ~11.3GB/세션 + vision feature cache, LRU=2로 최악 20GB+. 대용량 RAM 박스에선 무해하나 `_MAX_CACHED_RUNS`는 entry 수가 아니라 byte 예산으로 묶는 편이 안전합니다.
* (기존 사항) `GROUPS`가 완료 후에도 제거되지 않아 서버 수명 동안 단조 증가 — 이번 라운드 밖, 소량이라 cosmetic.


---

## Findings (triage skeleton — verify each before registering)

_No `JW-GPT-NNN` finding blocks parsed — triage the verbatim reply directly._
