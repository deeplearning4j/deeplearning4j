# ADR 0107: INT8 Quantized KV Decode Path (V2)

## Status

Proposed

Proposed by: Adam Gibson (9 Jul 2026)

## Context

### V1 state (landed, working)

`KvCacheStrategy.QUANTIZED` (`GenerationPipeline.java:1205–1306`) keeps the full
`float32` or `float16` `staticKvBuffers` as the **live decode copy** and additionally
materialises a post-prefill INT8 archive (`quantizedKvBuffers` + `kvScaleBuffers` in
`InGraphKvState`). The quantisation is performed once after the full prefill via
`KVCacheQuantize` applied to the entire `[B, maxKvLen, H_kv, D]` buffer.

V1 limitations that prevent it from reducing live memory:

1. **Float KV buffers are never freed.** Both `staticKvBuffers` (float) and
   `quantizedKvBuffers` (INT8) live simultaneously in `InGraphKvState`. The INT8 archive
   is an extra copy, not a replacement.
2. **Decode writes remain float.** `kvInPlaceWriteBSHD` in `dot_product_attention_v2.cpp`
   (`helpers::kvInPlaceWriteBSHD(keyCache, keys, cachePosPtr, ...)`) writes a new float
   K/V vector at `cachePosition` in the live float buffer. There is no per-step quantise
   step on the write path.
3. **No per-step decode read path.** There is no IN-GRAPH dequantise node that reads INT8
   and feeds the GQA attention kernel. `fusedGQADecodeCuda` consumes float K/V arrays.
4. **V1 is incompatible with rotating KV** (explicit guard at `GenerationPipeline.java:1207–1211`).

### Dormant native infrastructure

- `kv_cache_quantize.h` / `cpu/kv_cache_quantize.cpp` / `cuda/kv_cache_quantize.cu`:
  symmetric per-row absmax INT8 quantisation (and INT4, FP8_E4M3, FP8_E5M2).
  **Scale granularity: one float scale per logical row**, where "row" is the last
  dimension (headDim). Concretely, for layout `[B, maxKvLen, H_kv, D]` flattened to
  `numRows = B * maxKvLen * H_kv` rows of length `D`, there is one float32 scale per
  `(batch, token, kv_head)` triple.
- `turbo_quant_attention` op + helpers: asymmetric QJL-correction attention operating on
  compressed key representations. **Its contract does NOT fit the GQA single-token decode
  path** — it requires pre-computed MSE-reconstructed keys (`kMse`), QJL sign bits
  (`qjlSigns`), residual norms, and a random QJL projection matrix `[D, D]` as separate
  fixed tensors. These do not exist in the GGUF BSHD decode loop; wiring them would
  require permanent offline preprocessing of the key cache and a `[D, D]` matrix kept in
  device memory per head per layer. The complexity and memory overhead outweigh the
  accuracy benefit at INT8 widths. **`turbo_quant_attention` is NOT selected as the read
  path for V2.**
- `fusedGQADecodeCuda` (`FlashAttentionHelper.h:416–419`): fused GQA decode kernel with
  signature `(query, key, value, output, scale, context, attentionBias)` taking 4D BSHD
  tensors. This is the kernel that must be extended to accept INT8 K/V.

### ADR 0106 substrate dependency

ADR 0106 (consolidated decode substrate) will generalise the native decode step to a
frozen `[B_max × W_max]` multi-position forward. V2 quantised-KV **must be a first-class
requirement of that substrate**: the KV read/write surgery must happen once in the C++
kernel contract so that speculative-verify windows (W > 1) also read INT8 KV without a
second redesign.

### Bandwidth and memory arithmetic (Qwen-0.8B-class model)

Reference model: Qwen3.5-0.8B, 28 attention layers, H=14 Q-heads, H_kv=2 KV-heads,
D=64, maxKvLen=2048, B=1.

| Metric | fp16 K/V | INT8 K/V | INT4 K/V |
|---|---|---|---|
| Per-position KV vector | 2 × H_kv × D × 2B = 512B | 2 × H_kv × D × 1B = 256B | 2 × H_kv × D × 0.5B = 128B |
| Full cache (28L, 2048T) | 28 × 512 × 2048 = 28MB | 28 × 256 × 2048 = 14MB | 28 × 128 × 2048 = 7MB |
| Scale overhead (INT8, fp32) | — | 28 × 2 × 2 × 2048 × 4B ≈ 2.3MB | same |
| Net live reduction | 0% | **-42% vs fp16** | -63% |

For a larger model (32L, H_kv=8, D=128, maxKvLen=8192, fp32 baseline):

| Precision | Live KV bytes |
|---|---|
| fp32 | 32 × 8 × 128 × 8192 × 4B = 1.07GB |
| fp16 | 537MB |
| INT8 | **268MB** |
| INT4 | 134MB |

Per-decode-step DRAM traffic reduction (single-token GQA, reading all 2048 KV positions):

```
fp16: 28 layers × 2 KV-heads × 2048 tokens × 64 dims × 2B × 2 (K+V) = 28MB/step
INT8: 28 × 2 × 2048 × 64 × 1B × 2                                    = 14MB/step
Scale: 28 × 2 × 2048 × 4B × 2                                         = 0.9MB/step
INT8 net: 14.9MB/step  (-47% DRAM vs fp16)
```

Scale read overhead is ~6% of INT8 payload — negligible. Bandwidth-bound decode (RTX
4090: ~1TB/s at decode utilisation) translates this directly to ~2× KV-read throughput.

## Decision

### 1. Write Path: Quantised-on-write via a new `kvInPlaceWriteQuantisedBSHD` helper

**Decision: quantise at write time inside the attention op, not via an in-graph node.**

Rationale: The in-graph-kv path has `cachePosition` as a device-resident scalar read by
`kvInPlaceWriteBSHD` without a host round-trip (required for CUDA-graph compatibility —
see `dot_product_attention_v2.cpp:202–212`). Inserting an in-graph `kv_cache_quantize`
op would require a separate plan output (the scale), a scatter write of that scale into
the scale buffer at `cachePosition`, and two execution-time nodes per attention layer per
step. This is mechanically feasible but doubles the attention node count and adds
scheduler pressure.

The write-time alternative adds a new helper `kvInPlaceWriteQuantisedBSHD` that:
1. Reads the float K (or V) vector `keys[0, 0, :, :]` at step `t` (shape `[1,1,H_kv,D]`).
2. Computes per-row absmax (one warp reduction per KV-head row of length D).
3. Writes INT8 quantised values into `quantisedKeyCache[0, cachePos, :, :]`.
4. Writes float32 scale into `keyScales[0, cachePos, :]` (shape `[1, maxKvLen, H_kv]`).

The float K/V input tensor (`keys` from the current decode step) is **not** stored
permanently — it is a transient view from the Q/K/V projection outputs. The permanent
storage is the quantised buffer + scale buffer. This means no float KV buffer persists
after prefill is complete.

**Scale granularity: per-token-per-head (one scale per `(token, kv_head)` pair).**

This is the only append-friendly granularity for one-token-at-a-time writes:

- Per-channel scales (one scale per dimension D, computed over all T tokens) require
  seeing the full column before the scale is valid — impossible for streaming append.
- Per-block-of-N scales (e.g., block-16 along the token dimension) require buffering
  N tokens before emitting one scale. Adds latency and complexity for marginal accuracy
  gain over per-token.
- Per-token-per-head requires exactly one warp reduction per head row per step. For
  D=64 this fits in a single 32-thread warp. For D=128 it takes two warp reductions.
  Both complete within the same CUDA stream without any host interaction.

**Scale buffer layout:** `float32[B, maxKvLen, H_kv]` — one float32 per position per KV
head, stored contiguous with the same spatial indexing as the quantised data buffer
`int8[B, maxKvLen, H_kv, D]`. Scale reads during attention are a simple stride-1 gather
along the token dimension, one scale per `seqK` position per KV head.

**Size math for Qwen-0.8B (H_kv=2, D=64, maxKvLen=2048, B=1, 28 layers):**

```
INT8 K buffer per layer: 1 × 2048 × 2 × 64 × 1B = 262KB
Scale K buffer per layer: 1 × 2048 × 2 × 4B     =  16KB
INT8 V buffer per layer: identical
Scale V buffer per layer: identical
Total per layer:  (262+16) × 2 = 556KB
Total 28 layers:              = 15.6MB  (vs 56MB float32)
```

For Qwen-class large (H_kv=8, D=128, maxKvLen=8192, B=1, 32 layers):

```
INT8 K per layer: 1 × 8192 × 8 × 128 × 1B = 8MB
Scale K per layer: 1 × 8192 × 8 × 4B      = 256KB
Total per layer:  (8+0.25) × 2 = 16.5MB
Total 32 layers:              = 528MB  (vs 2.14GB float32)
```

### 2. Read Path: Inline-dequant in the GQA decode kernel

**Decision: inline INT8 load + scale multiply into `fusedGQADecodeCuda`.**

`turbo_quant_attention` is eliminated from consideration (see Context section). The two
remaining candidates are:

**(a) Inline dequant in `fusedGQADecodeCuda`:** Extend the kernel signature to accept
`(int8 K, float scale_K, int8 V, float scale_V)` alongside the existing float query. The
inner loop changes from `float kval = K[...]` to `float kval = (float)K[...] * scaleK[head_pos]`.
This is a two-instruction change per load — zero additional kernel launches, zero
additional DRAM round-trips vs the float path (INT8 loads are narrower, bandwidth is
already reduced).

**(b) Dequantise-then-read:** materialise float K/V scratch buffers per step, call
`kvCacheDequantize`, then call the existing float kernel. Requires `2 × H_kv × seqK × D`
float scratch per layer per step — identical byte count to the original float cache,
negating all memory savings.

Option (a) is unambiguously correct. Option (b) is a workaround and is rejected.

For the ADR-0106 multi-position substrate (`W_max > 1`), the same kernel extension
applies: the masking and multi-row query loop in the generalised substrate kernel reads
INT8 K/V with the same per-head scale. The substrate kernel contract must accept
`(int8* K, float* scaleK, int8* V, float* scaleV)` pointers from the start so no
second kernel redesign is needed when speculative/contrastive modes land.

**K-versus-V asymmetric precision mode (optional, recommended for quality):**

Post-RoPE K quantisation concentrates error in high-frequency rotary dimensions.
Empirically (from oMLX/turboquant practice), quantisation error in K produces larger
attention-score perturbations than equal error in V (because V error averages out in the
softmax-weighted sum). A `KV_QUANT_MODE` config flag with values:

| Mode | K precision | V precision |
|---|---|---|
| `INT8_KV` (default) | INT8 | INT8 |
| `FP16K_INT8V` | fp16 (2B) | INT8 | 
| `INT8K_INT4V` | INT8 | INT4 |

`FP16K_INT8V` sacrifices the K memory savings (~25% overall instead of ~50%) for
accuracy. `INT8K_INT4V` doubles V compression for memory-limited deployments. The default
`INT8_KV` is the primary target; the others are config flags, not default paths.

### 3. Prefill: reuse V1 whole-buffer quantise; free float buffers immediately after

After the prefill K/V outputs have been written into the static KV buffer and quantised
(exactly as in V1 STEP 2, `GenerationPipeline.java:1283–1306`), V2 adds one action: the
`staticKvBuffers` float arrays are **freed immediately** (before decode begins) after the
INT8 buffers are populated.

The lifecycle becomes:
1. Prefill runs → K/V outputs collected per layer (transient).
2. `kvCacheQuantize` applied to each layer's K/V → `quantisedKvBuffers` + `kvScaleBuffers`
   allocated and populated (exact V1 code path).
3. **NEW in V2:** `staticKvBuffers` entries closed and map nulled. The float tensors are
   freed. This is the single action that makes live memory smaller.
4. Decode plan is frozen with `quantisedKvBuffers` and `kvScaleBuffers` as the only
   persistent KV storage. The frozen plan's ext-input pointers point to the INT8/scale
   buffers (not float).

This is a purely Java-side change in `prefillWarmupAndFreeze` (or its inlined equivalent
in `setupInGraphKvState`). No C++ change is required for the free step — `INDArray.close()`
releases device memory in the normal way.

**Exactly one float buffer must survive past step 3:** the transient `[1,1,H_kv,D]` float
K/V view used inside each attention op for the current decode token. This is a stack-local
view from Q/K/V projection — it is NOT in `staticKvBuffers` and is already ephemeral.

### 4. CUDA-graph capture compatibility

The captured decode step requires that all pointers and shapes are fixed at capture time
and do not change between replays. Analysis of each new element:

| Element | Fixed? | Rationale |
|---|---|---|
| `int8* quantisedKeyCache[L]` | Yes | same allocation per layer, stable device pointer |
| `float* keyScaleBuffer[L]` | Yes | same allocation per layer, stable device pointer |
| `int8* quantisedValueCache[L]` | Yes | same |
| `float* valueScaleBuffer[L]` | Yes | same |
| `cachePosition` device scalar | Yes | existing pattern — pointer stable, value updated D2D |
| Scale write at `cachePosition` | Yes | scatter write via stable INT8/scale buffers |
| Per-step scale computation | **Hazard** | see below |

**Capture hazard: per-row warp reduction in `kvInPlaceWriteQuantisedBSHD`.**

The per-token per-head scale computation (warp reduction for absmax) is a reduction over
D values per head row. On the write path this runs BEFORE the scale is written to
`keyScaleBuffer`. This is:
1. A pure compute kernel (no atomics across rows — each row is independent).
2. Launched into the DSP execution stream.
3. Outputs to a fixed device address (`keyScaleBuffer[0, cachePos, :]`).

A warp-level `__shfl_down_sync` reduction with no inter-row atomics is capture-safe: the
result address is determined by `cachePosition` at capture time (it is a compile-time
constant from the captured graph's perspective because `cachePosition` is a device scalar
whose address is baked). The VALUES change per replay but that is exactly what CUDA-graph
replay is designed for.

No capture hazards remain. The write kernel and the read (attention) kernel are both
capture-safe with the INT8 path.

### 5. ADR-0106 substrate integration

The multi-position substrate (W_max, B_max) built in ADR 0106 must accept quantised KV
from the start. Kernel contract requirements:

```cpp
// Extended fusedGQADecodeCuda signature for the substrate:
void fusedGQADecodeQuantisedCuda(
    NDArray* query,           // [B, W, H_q, D]  — W positions, B hypotheses
    const int8_t* keyQ,       // [B, seqK, H_kv, D]  INT8 quantised
    const float*  keyScale,   // [B, seqK, H_kv]     per-token-per-head scale
    const int8_t* valQ,       // [B, seqK, H_kv, D]  INT8 quantised
    const float*  valScale,   // [B, seqK, H_kv]
    NDArray* output,          // [B, W, H_q, D]
    double scale,
    NDArray* attentionBias,   // [B, 1, W, seqK] or null — the substrate mask
    LaunchContext* context);
```

The `[B, 1, W, seqK]` mask from the substrate already encodes causal + hypothesis
isolation. The attention kernel must apply it before softmax. This contract must be
specified here so that ADR 0106 Phase 1 implementation writes a quantisation-ready kernel
from the start.

The `float32` variant (`fusedGQADecodeCuda`, existing) remains for CPU execution and for
the non-quantised case. It is not removed.

### 6. Accuracy: sensitivity analysis, mitigations, and acceptance gates

**K quantisation error analysis:**

Post-RoPE K vectors have a characteristic frequency structure: high-frequency rotary
dimensions rotate rapidly between positions and tend to produce high-magnitude, outlier
values. Per-token-per-head INT8 quantisation with absmax scaling handles single-position
outliers correctly (the scale adapts per token). Accumulated error in Q@K^T grows as
O(sqrt(D) / 127) per attention score under uniform assumptions — for D=128 that is
approximately 0.9% RMS error per score before softmax. After softmax the low-probability
tails absorb most error; the high-probability heads are relatively stable.

**V quantisation error analysis:**

V vectors are projection outputs of residual-stream values — generally smoother than K.
Per-token-per-head INT8 gives similar theoretical error but lower empirical impact because
the softmax-weighted sum averages the error across all attending positions.

**Mitigation: skip-last-layer (recommended, optional):**

oMLX's `turboquant_kv.py` and prior literature (QuaRot, KIVI) observe that the final few
transformer attention layers produce the largest quality impact when quantised. The
`skipLastLayersQuantised` config (default: 1) keeps the last N layer pairs in float16
and quantises all others. This adds `2 × N × 2 × maxKvLen × H_kv × D × 2B` float16 bytes
— for N=1, D=128, H_kv=8, maxKvLen=8192 that is 33MB additional, a rounding error at
the scale of model-weight memory.

**Acceptance gates (mandatory for V2 to be declared "done"):**

| Gate | Criterion |
|---|---|
| DSP regression batch | 1590 tests, 0 failures, 0 errors (existing mandatory batch) |
| Token match-rate (Qwen 0.8B, 250 tokens) | `>=90%` greedy tokens identical to fp16 baseline |
| Token match-rate with `FP16K_INT8V` mode | `>=95%` |
| `lateSteady tok/s` (OPTIMAL config) | neutral ± 3% vs pre-V2 (memory savings, not regression) |
| Live KV memory assertion | Java assertion in `InGraphKvState`: `staticKvBuffers == null` after freeze (V2 path) |
| DSP gate after freeze | `DspPlanAssertions.assertPointerStable(executor)` passes on INT8 buffers |
| CPU correctness | `TestQuantisedKvDecodeV2` on CPU backend: INT8 decode output matches dequant-then-decode within 1e-2 |
| CUDA correctness | Same test class on CUDA backend |

### 7. Rotating-KV composability

V1 explicitly rejects `QUANTIZED + rotatingKvEnabled` (guard at `GenerationPipeline.java:1207`).

Analysis: rotating KV (`RotatingKvSlotMap`) maps logical global positions to physical KV
slots via a ring of non-sink tokens (attention-sink positions pinned at the head, the rest
overwritten in a rotating window). For V2, the write path must quantise at the **physical
slot** assigned by `RotatingKvSlotMap`, not the raw `cachePosition`. The scale buffer
must also be written at the physical slot. The decode read path is unchanged: it reads
the quantised buffer at the physical KV layout (attention already sees physical slot
order).

The mutual exclusion can be lifted when `RotatingKvSlotMap.physicalSlotFor(globalPos)` is
passed instead of `cachePosition` to `kvInPlaceWriteQuantisedBSHD`. This is a Java-side
change (the scalar fed to the write kernel becomes the mapped physical slot). No C++
change is required beyond accepting the physical slot as the write position.

**V2 does NOT implement this.** It keeps the V1 guard. Rotating + quantised is a follow-up
ADR.

### 8. Migration from V1

| V1 component | V2 fate |
|---|---|
| `isQuantizedKv` flag + `KvCacheStrategy.QUANTIZED` enum | Survives unchanged |
| `config.getKvQuantFormat()` | Survives; extended with `FP16K_INT8V` / `INT8K_INT4V` modes |
| `quantizedKvBuffers` / `kvScaleBuffers` in `InGraphKvState` | Survive as the primary live storage |
| `staticKvBuffers` (float) | V2: freed immediately after STEP 2 quantise. V1: kept alive. |
| `replaceQuantizedBuffer()` helper | Survives for the initial-fill path |
| V1 guard "QUANTIZED incompatible with rotating" | Survives in V2 |
| V1 code path in `STEP 2` (quantise the prefill region) | Becomes the prefill initial-fill step; **no code change needed** |
| V1 decode loop (reads float staticKvBuffers) | **Superseded.** Decode loop reads INT8 quantised buffers directly. |
| `fusedGQADecodeCuda` float signature | Kept for non-quantised path; extended, not replaced |
| `kvInPlaceWriteBSHD` | Kept for non-quantised path; new `kvInPlaceWriteQuantisedBSHD` added |

### 9. Implementation plan

#### Phase 1: Write path (native, header-safe)

Files touched:
- `libnd4j/include/ops/declarable/helpers/kv_cache_quantize.h` — add
  `kvInPlaceWriteQuantisedBSHD` declaration
- `libnd4j/include/ops/declarable/helpers/cuda/kv_cache_quantize.cu` — implement
  `kvInPlaceWriteQuantisedBSHD`: per-token-per-head absmax reduction + INT8 scatter +
  scale scatter, all into pre-allocated fixed-address buffers; CUDA-graph-safe
- `libnd4j/include/ops/declarable/helpers/cpu/kv_cache_quantize.cpp` — CPU equivalent

Test: unit test in `platform-tests/` — `TestQuantisedKvWriteV2`: write N float K/V
vectors into an INT8 buffer via the new helper, dequantise, compare to original within
1e-2 tolerance. Both backends.

#### Phase 2: Read path (kernel extension)

Files touched:
- `libnd4j/include/helpers/cuda/FlashAttentionHelper.cu` — add
  `fusedGQADecodeQuantisedCuda` alongside the existing `fusedGQADecodeCuda`. Accept
  `(int8* keyQ, float* keyScale, int8* valQ, float* valScale)`. Inner loop:
  `float kval = (float)keyQ[idx] * keyScale[head_pos_idx]`.
- `libnd4j/include/helpers/FlashAttentionHelper.h` — declare `fusedGQADecodeQuantisedCuda`

Test: unit test `TestQuantisedKvReadV2`: fill INT8 KV buffer from known fp16 data, run
`fusedGQADecodeQuantisedCuda`, compare to `fusedGQADecodeCuda` (float). Token match-rate
≥ 90% on synthetic attention inputs. CUDA only (CPU has no fused kernel; use reference
dequant + standard attention for CPU path).

#### Phase 3: Java wiring + float buffer release

Files touched:
- `nd4j/samediff-llm/.../generation/GenerationPipeline.java` — after V1 STEP 2 quantise
  block, add: close and null `staticKvBuffers` when `isQuantizedKv`. Set
  `state.staticKvBuffers = null` before freeze.
- `nd4j/samediff-llm/.../generation/InGraphKvState.java` — add assertion in `close()`:
  `assert !isQuantizedV2 || staticKvBuffers == null : "V2: float KV leaked past freeze"`.
- Wire `quantisedKvBuffers` and `kvScaleBuffers` as the ext inputs for the decode plan
  (replacing `staticKvBuffers` on the QUANTIZED path). The ext-input index slots for KV
  now point to INT8 buffers.
- Wire `kvInPlaceWriteQuantisedBSHD` as the attention write call inside
  `dot_product_attention_v2.cpp` when the `kvQuantisedExt` inputs are non-null (signal
  via a new iArg or input slot).

Test: `TestQuantisedKvDecodeV2` — full decode pipeline on Qwen 0.8B GGUF, QUANTIZED
strategy, 250 tokens. Assertions: (a) no float KV buffer after freeze, (b) `lateSteady
tok/s` ≥ pre-V2 baseline, (c) token match-rate vs fp16 ≥ 90%.

#### Phase 4: ADR-0106 substrate contract

When ADR 0106 Phase 1 lands (generalised `[B_max × W_max]` substrate), extend
`fusedGQADecodeQuantisedCuda` to accept the `[B, 1, W, seqK]` grid mask from the
substrate. Add `B_max / W_max` grid dims to the kernel. This phase depends on ADR 0106
Phase 1.

#### Phase 5: K-vs-V asymmetric modes

Add `KV_QUANT_MODE` config to `GenerationPipelineConfig`. Implement `FP16K_INT8V`
by keeping K as fp16 in `quantisedKvBuffers` (INT16 dtype store, or a separate fp16
buffer map). Implement `INT8K_INT4V` by routing V through the existing
`kvCacheQuantizeInt4*` path.

Files: `GenerationPipelineConfig.java`, `KvCacheStrategy.java` (new `KV_QUANT_MODE`
enum), `kv_cache_quantize.h/cpp/cu` (INT4 path already exists).

#### Phase 6: skip-last-layer accuracy guard

Add `skipLastLayersQuantised` (default: 1) to `GenerationPipelineConfig`. In STEP 2,
skip quantise for `layerIdx >= numLayers - skipLastLayersQuantised` and keep those
layers in float16. In the decode plan ext-input wiring, those layers use the float path;
the remaining layers use the INT8 path.

Files: `GenerationPipeline.java` (STEP 2 guard), ext-input wiring code.

## Consequences

- **Live memory:** Float KV buffers freed post-prefill; live KV footprint drops by ~50% on
  INT8 or ~75% on INT4. For a 32-layer, maxKvLen=8192 model this reclaims ~500MB–1GB.
- **DRAM bandwidth:** Per-step KV reads fall by ~47% (INT8) or ~68% (INT4). On bandwidth-
  bound decode this maps to proportional tok/s improvement beyond the existing baseline.
- **Accuracy cost:** ~10% token mismatch at INT8 (within the accepted gate); mitigated to
  ~5% with `FP16K_INT8V`; further mitigated with `skipLastLayersQuantised=1`.
- **CUDA-graph compatibility preserved:** All new buffers are fixed-allocation with
  stable device addresses. The write kernel uses device-scalar `cachePosition`; the read
  kernel extends the existing captured kernel. No new dynamic allocations enter the
  captured graph.
- **V1 users:** `KvCacheStrategy.QUANTIZED` with default config upgrades transparently
  to V2. The V1 behaviour (float buffers kept alive) is no longer supported; the
  migration is one-directional.
- **ADR-0106 compatibility:** The kernel contract specified in Decision 5 ensures the
  substrate lands with INT8 KV already supported.
- **Rotating KV:** still incompatible; the path to removing the guard is described in
  Decision 7 for a follow-up.

## Alternatives considered

- **`turbo_quant_attention` as the read path** — rejected. Its contract requires permanent
  offline per-key MSE reconstruction, QJL sign bits, residual norms, and a `[D, D]`
  projection matrix, none of which exist in the GGUF BSHD decode loop. The extra permanent
  buffers exceed INT8 memory savings for small D.
- **In-graph `kv_cache_quantize` node per step** — rejected. Doubles attention node count,
  adds a plan-output (the per-step scale), requires an in-graph scatter op. The
  quantise-at-write approach embeds this in a single kernel inside the attention op.
- **Block-of-N scale granularity** — rejected for the append path. Requires buffering N
  tokens before emitting one scale, adding latency and state. Per-token-per-head is
  append-friendly with only marginal accuracy difference.
- **Dequantise-to-scratch then use existing float kernel** — rejected. Eliminates all
  memory savings; equivalent to keeping float buffers.

## Validation

```bash
# Phase 1–3 correctness (run from platform-tests/)
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
  /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-cuda-12.9 \
  -Dtest=TestQuantisedKvWriteV2,TestQuantisedKvReadV2,TestQuantisedKvDecodeV2 \
  -Dnd4j.dsp.diagnostics=ALL -Dnd4j.dsp.diagnostics.level=full \
  2>&1 | tee /tmp/quantised-kv-v2-tests.log

# DSP regression gate (mandatory)
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
  /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-cuda-12.9 \
  -Dtest=DspHandleDataModelTest,DspBufferAliasAccuracyTest,DspHandleTest,DspLifecycleExhaustiveTest,DspLifecycleValidationTest,DspFrozenConstantInvariantTest,DspExtInputStalenessTest,DspSlotLifecycleAuditTest,DspConcurrentPlanSharingTest,DspCompositeReplayTest,TestDspShapePrePass \
  2>&1 | tee /tmp/dsp-core-batch-post-quantised-kv.log

# Benchmark (Phase 3 gate)
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
  ./run-benchmark.sh --backend cuda --tokens 250 --config OPTIMAL \
  --op-timing --diag-replay --diag-json /tmp/quantised-kv-bench.json \
  2>&1 | tee /tmp/quantised-kv-bench.log
```

Token match-rate measurement: compute greedy token sequence with `KvCacheStrategy.STATIC`
(fp16 baseline) and with `KvCacheStrategy.QUANTIZED` (V2 INT8), compare token-by-token
over 250 positions. Acceptance: ≥ 90%.

## Related ADRs

- [0096](0096%20-%20LLM%20Generation%20Pipeline.md) — generation pipeline; `KvCacheStrategy` hierarchy
- [0097](0097%20-%20Decode%20Path%20Performance%20Optimizations.md) — `fusedGQADecodeCuda` origin
- [0105](0105%20-%20Generation%20Session%20Continuation%20(Resume%20Decode).md) — frozen-plan pointer-stability
  contract that the INT8 buffers must satisfy
- [0106](0106%20-%20Consolidated%20Masked%20Multi-Position%20Decode%20Substrate.md) — substrate; V2 kernel
  contract must be co-designed with Phase 1 of that ADR
