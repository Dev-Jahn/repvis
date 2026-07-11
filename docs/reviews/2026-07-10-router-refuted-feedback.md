<!-- jahns-workflow feedback: the body below is the reviewer reply VERBATIM (byte-exact copy via `jw review ingest`) — do not edit it; a triage skeleton is appended beneath it. -->
round: 2026-07-10-router-refuted
reviewer: gpt-5.5-pro
ingested: 2026-07-11
source: /tmp/review.md

---

## 검토 범위

* 패킷상 Reviewed HEAD는 `c5c0ef66cfd993268318e05f18a3ab2120fa2e3f`, base는 `9c8184dfb2528bafed93cf4ea6cefc03c32c5ab8`입니다.
* 현재 `main`은 `b590c7100385146d743953bb8154ceb982a75215`이며, Reviewed HEAD보다 `docs(round): close 2026-07-10-router-refuted` 문서 마감 커밋 하나 앞섭니다. 코드 mismatch는 아닙니다.
* `git clone`은 현재 환경의 DNS 제약으로 실패했습니다. GitHub Connector를 통해 Reviewed HEAD의 실제 파일을 직접 검토했습니다.
* Reviewed HEAD에는 별도 CI status가 등록되어 있지 않아, 패킷에 기재된 CPU/GPU 테스트 수치는 독립 재확인하지 못했습니다.

## 판정

**이번 diff에서 새로 확인된 Major/Critical 구현 결함은 없습니다.**

다만 이전 라운드의 Major인 `perf/sparse-decode-full-walk`는 해결된 것이 아니라, 잘못된 exact-seek 후보를 철회하고 다시 열린 상태입니다. 따라서 이 라운드는 “Major 수정 완료”가 아니라 **correctness를 보존한 안전한 rollback 및 반례 기록**으로 보는 것이 정확합니다.

---

## 확정된 outstanding Major

### Major — 긴 영상의 positional full walk와 unit별 prefix 재디코드는 그대로 남아 있습니다

`iter_frames_at()`은 각 sampled position에 도달할 때까지 `_core.get_next_frame()`을 반복하므로, 마지막 샘플이 영상 후반에 있으면 사실상 원본 전체를 디코드합니다. EOF 이후에만 마지막 프레임을 재사용할 뿐, sparse sampling 자체는 decode work를 줄이지 않습니다.

phase 1은 한 source를 최대 여러 unit으로 분리하고, 각 unit마다 별도의 `GpuVideoSource`를 생성합니다. 따라서 후반 unit의 decoder도 다시 stream 시작점부터 해당 unit의 첫 sampled index까지 walk합니다.

그 후 `_render_source()`는 SAM 입력을 위해 CPU에서 같은 source를 다시 positional decode합니다. 즉 현재 구조에는 다음 비용이 모두 남습니다.

1. 마지막 sampled frame까지의 full walk
2. unit마다 반복되는 prefix walk
3. phase 1 종료 후 SAM용 두 번째 CPU full walk

패킷도 long-video latency가 unchanged라고 명시하므로, 이 문제를 숨기거나 완료 처리한 것은 아닙니다.

**권장 상태:** `perf/sparse-decode-full-walk`는 계속 Major로 유지하는 것이 맞습니다. 현 router refutation을 근거로 severity를 낮추거나 “alignment suite 강화로 해결됨”으로 닫아서는 안 됩니다.

---

## Open domain question

### `(file, indices, chunking, torchcodec-version)`만으로 exact replay를 인증하는 것은 충분하지 않습니다

패킷의 방향 (a)는 실제 production index set을 positional 결과와 전수 비교한 뒤, repeat SAM re-decode에서 exact fetch를 재사용하는 방안입니다. 패킷이 제안한 certification key는 `(file, indices, chunking, torchcodec-version)`입니다.

현재 lock은 `torchcodec==0.14.0+cu130`을 고정합니다.  그러나 persistent certificate가 후속 프로세스에서도 exact output을 무검증 신뢰한다면, 다음 요소도 결과에 영향을 줄 수 있으므로 key가 부족합니다.

* source의 **전체 content hash**. 경로나 축약 ID가 아니라 full SHA-256이어야 합니다.
* ordered `frame_indices` 전체와 각 exact API 호출의 **정확한 batch partition 및 호출 순서**
* decoder lifecycle: 매 batch마다 fresh decoder인지, 한 decoder에 여러 호출을 누적하는지
* backend/device: 적어도 `cpu`/`cuda`, 향후 GPU exact를 사용한다면 GPU architecture와 driver
* torchcodec wheel hash뿐 아니라 실제 FFmpeg/libavcodec build/version
* `seek_mode`, decoder options, thread 설정
* certification schema 및 router implementation version

특히 이번 반례는 “같은 파일에서 요청한 index set 자체가 output을 바꾼다”는 것이므로, 단순한 `chunk_size=64` 같은 값으로는 부족합니다. 예를 들어 아래 두 호출은 같은 indices와 nominal chunk size를 갖더라도 동일한 키로 취급하면 안 됩니다.

```text
get_frames_at([0, 2, 4, ..., 62])
get_frames_at([64, 66, ..., 126])
```

```text
get_frames_at([0, 2, 4, ..., 126])
```

호출 경계와 decoder 재사용 여부가 다르기 때문입니다.

### 안전한 인증 프로토콜

boolean sidecar인 “이 파일은 exact-safe”를 저장하는 방식은 권장하지 않습니다. 다음 구조가 더 안전합니다.

1. **초기 certification**

   * 실제 production ordered batch sequence로 exact decode
   * 같은 requested indices를 positional walk로 decode
   * requested frame별 positional output hash를 저장
   * exact output이 각 hash와 모두 같을 때만 certificate 생성

2. **매 replay**

   * fresh decoder를 생성
   * 인증된 동일 batch sequence로 exact decode
   * 결과를 저장된 positional frame hash와 다시 비교
   * 하나라도 다르면 결과를 폐기하고 positional fallback
   * certificate를 invalid 처리

이렇게 하면 decoder threading이나 malformed-stream error concealment가 비결정적인 경우도 결과 검증에서 잡힙니다. 반대로 replay 때 output hash를 확인하지 않고 sidecar만 신뢰하려면, 환경 fingerprint를 훨씬 더 넓게 잡아도 완전한 보장은 어렵습니다.

또한 현재 `_SegCache`가 최근 2개 run의 decoded frames와 SAM vision features를 이미 유지하므로, exact certification의 효과는 모든 클릭이 아니라 **cache eviction, refit, process restart 후의 cache miss**에 한정됩니다.  따라서 certification 구축 및 유지 복잡도 대비 실제 hit rate를 먼저 계측해야 합니다.

---

## 남은 Major 해결 방향 평가

### (b) source당 한 번의 positional decode 후 unit fan-out

정합성 측면에서는 가장 강한 방향입니다. 다만 현재 각 `GpuVideoSource`는 해당 unit이 배정된 GPU에 직접 NVDEC frame을 생성합니다.  한 source에 decoder 하나만 두면, 여러 compute GPU로 frame을 보내기 위해 다음 중 하나가 필요합니다.

* decode GPU에서 compute GPU로 CUDA P2P copy
* peer access가 없는 경우 host bounce
* 해당 source의 모든 extraction을 decode GPU 하나에 고정

따라서 구현 시에는 source-level decode coordinator와 bounded per-unit queue만으로 끝나지 않습니다. CUDA event, `record_stream()`, peer-access topology, fallback policy까지 설계해야 합니다.

권장 형태는 다음입니다.

```text
one source decoder on GPU D
  ├─ sampled chunk for unit A → local GPU extraction
  ├─ sampled chunk for unit B → P2P copy → GPU E extraction
  └─ sampled chunk for unit C → P2P copy → GPU F extraction
```

P2P가 불가능하거나 PCIe copy가 extraction보다 느린 환경에서는 source 단위 single-GPU 처리로 fallback해야 합니다. 이 방향은 prefix 재디코드를 근본적으로 없애지만, 기존 multi-NVDEC 병렬성의 일부를 포기하므로 반드시 실제 1/2/4/8 GPU topology에서 측정해야 합니다.

### (c) SAM CPU walk를 phase 1과 overlap

구현 난도가 더 낮고 correctness 위험도 작지만, decode work 자체는 줄이지 않고 latency만 숨깁니다.

주의할 점은 현재 phase 1의 pinned feature D2H와 CPU SAM decode가 동시에 host memory bandwidth를 사용한다는 것입니다. 또한 decoded RGB frames를 더 오래 유지하므로 peak RAM은 같더라도 high-RAM plateau가 길어집니다. source가 여러 개인 joint run에서는 source별 full RGB frame list를 동시에 축적하지 않도록 concurrency와 byte budget이 필요합니다.

### 권장 순서

1. **(c)를 독립적인 measured optimization으로 먼저 검증**

   * CPU decode와 D2H contention 포함
   * source별 동시 frame accumulation 제한
2. **(b)를 근본 해결로 설계**

   * P2P topology-aware fan-out
   * bounded queues
   * no-peer fallback
3. **(a)는 좁은 보조 최적화로만 사용**

   * 실제 replay hash 검증 필수
   * `_SegCache` miss 빈도와 비용 대비 효과를 먼저 계측

---

## 테스트 및 증거의 residual risk

### Exact-seek characterization은 “byte divergence”는 고정하지만 `20/22/24 → 18` 매핑 자체는 고정하지 않습니다

현재 테스트는 dense exact output이 atlas와 모두 동일하다는 것을 확인한 뒤, sparse exact output 중 atlas의 동일 index와 byte-identical하지 않은 frame이 하나 이상 있음을 assert합니다.

따라서 테스트가 기계적으로 증명하는 것은 다음입니다.

```text
dense exact == positional atlas
sparse exact != positional atlas for at least one requested index
```

패킷과 docstring에 적힌 구체적인 collapse인 `20, 22, 24 → 18`은 테스트 assertion에 포함되어 있지 않습니다. 이것은 현 rollback을 뒤집을 사유는 아닙니다. sparse exact가 positional output과 달라진다는 사실만으로도 현재 router의 correctness certification은 실패합니다. 다만 반례 mechanism을 장기적으로 보존하려면 다음 중 하나를 추가하는 편이 낫습니다.

```python
assert np.array_equal(sparse[k20], atl[18])
assert np.array_equal(sparse[k22], atl[18])
assert np.array_equal(sparse[k24], atl[18])
```

또는 각 sparse output을 atlas 전체에 nearest/exact match시켜 requested index와 actual resolved index의 mapping을 assert하십시오. 그러면 단순한 reconstruction-byte 차이와 실제 frame collapse를 구분할 수 있습니다.

---

## 최종 결론

* **신규 blocking finding:** 없음
* **현재 rollback/containment:** 정적 검토 기준으로 문제 없음
* **기존 Major:** `perf/sparse-decode-full-walk`는 여전히 유효하며 미해결
* **exact certification 방향:** 제안된 key만으로는 불충분; exact output을 저장된 positional hash와 replay마다 검증해야 함
* **우선 구현 방향:** overlap으로 단기 latency를 줄이고, topology-aware one-pass fan-out으로 근본적인 반복 decode를 제거하는 것이 적절합니다.


---

## Findings (triage skeleton — verify each before registering)

_No `JW-GPT-NNN` finding blocks parsed — triage the verbatim reply directly._

---

## Triage (2026-07-11, verified against c5c0ef6/b590c71)

**신규 Major/Critical finding: 0. Rollback/containment 정적 검토 통과.**

| # | Finding (요약) | Verdict | Evidence / 조치 | Task |
|---|---|---|---|---|
| 1 | outstanding Major(`perf/sparse-decode-full-walk`)는 미해결 — Major로 유지하고 "alignment suite 강화로 해결"로 닫지 말 것 | **동의 — 현 상태와 일치** | 태스크는 이미 open·major로 유지 중(라운드에서 --touched 처리, done 아님); PROGRESS 엔트리도 "라운드는 안전한 rollback + 반례 기록"으로 서술 | (기존 태스크 유지) |
| 2 | exact-replay 인증 key `(file, indices, chunking, torchcodec-version)`는 불충분 — batch partition/호출 경계·decoder lifecycle·content hash·ffmpeg build까지 결과에 영향; boolean sidecar 대신 **per-frame positional hash를 저장하고 매 replay마다 exact output을 재검증**(불일치 시 폐기+인증 무효화)하는 프로토콜 권장; `_SegCache`가 반복 클릭을 이미 흡수하므로 hit-rate 계측이 선행 조건 | **REAL (설계 지침)** — 반례 자체가 index-set 의존성을 증명했으므로 호출 경계까지 key에 포함해야 한다는 지적은 타당; 별도 태스크 대신 열린 major의 방향 (a)에 병합 | `perf/sparse-decode-full-walk` title 갱신 (우선순위 (c)→(b)→(a), replay-hash 검증 프로토콜, hit-rate 선행 계측 반영) | — |
| 3 | 특성화 테스트가 byte-divergence만 assert — `20/22/24→18` collapse mapping은 미고정: 단순 reconstruction-byte 차이와 실제 frame collapse를 구분 못함 | **REAL (minor)** | `tests/test_frame_alignment.py:293-294` 확인 — `misaligned` 존재만 assert. 각 misaligned output을 atlas 전체와 대조해 "이른 frame으로의 collapse"를 assert하도록 보강 (본 ingest 직후 수정) | `chore/splice-test-pin-collapse-mapping` |

방향 (b)/(c)에 대한 권고(P2P topology fan-out 설계 요건, D2H bandwidth contention·RGB 누적 상한)는
모두 방향 평가로서 태스크 title에 병합함. 신규 등록 1건(minor), blocker 0건.
