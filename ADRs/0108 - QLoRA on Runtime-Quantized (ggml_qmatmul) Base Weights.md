# ADR 0108: QLoRA on Runtime-Quantized (ggml_qmatmul) Base Weights

## Status

Accepted (implementation in progress)

Proposed by: Adam Gibson (9 Jul 2026)

## Context

LoRA / PEFT fine-tuning of imported GGUF models was only possible after
**dequantizing the whole model to dense FP32**. `ConversionOptions.forTraining()`
(`nd4j-ggml/.../convert/ConversionOptions.java:172`) sets
`quantizationMode = DEQUANTIZE_TO_FLOAT32` + `targetDataType = FLOAT`, so a 13 GB
Q4_K_M GGUF expands into ~50 GB of dense SameDiff shards before a single LoRA
parameter is added. This defeats the entire point of a quantized workflow: the
adapter is a few MB, but the base balloons 4×.

The runtime-quantized path already exists and is closer to correct.
`ConversionOptions.runtimeQuantizedMatmul()` (`:206`) keeps supported types
(Q8_0, Q4_K, Q6_K) as **packed INT8 byte buffers** and emits `ggml_qmatmul` ops
(`GGMLToSameDiffConverter.java:407`, `LLaMAArchitecture.java:381`) that dequantize
on the fly with fp32 accumulation. But this path was inference-only, and the PEFT
machinery could not attach to it:

- **`PeftModel.applyLora`** (`PeftModel.java:279`) locates targets by scanning for
  `VariableType.VARIABLE` weights and **skips anything whose shape is not rank-2**
  (`:283`). A packed GGUF weight is a rank-1 `BYTE` buffer plus a `LONG[3]`
  `.__q__` metadata companion, so LoRA injection never fired for it.
- **`injectLoraIntoGraph`** (`:310`) builds `W_eff = W + scaling·(B@A)` and rewires
  consumers — impossible against packed bytes; there is no dense `W` to add to.
- **`GgmlQMatMul`** (`ops/impl/transforms/custom/GgmlQMatMul.java`) had **no
  `doDiff`** and there was no `ggml_qmatmul_bp`, so autograd could not backprop
  through a quantized layer at all — blocking the activation gradient that earlier
  layers (and their adapters) depend on.
- **`LoraMatMul`** and its native `lora_matmul`/`lora_matmul_bp` are rank-2 only,
  and assume a dense base weight.

A read-only audit (see Consequences) found this is one instance of a broader set of
LoRA/PEFT limitations, all addressed here.

## Decision

Add first-class **QLoRA on a quantized (`ggml_qmatmul`) base**: the packed base
stays quantized and frozen; only low-rank adapters train. Correctness is delivered
by a **graph-level residual branch** (the minimum viable, low-risk path); a **fused
op** is added for performance.

### Op contract (native, libnd4j)

Forward `ggml_qmatmul` (unchanged): inputs `(A, W_packed)`, iArgs
`(quantType{4=Q8_0,8=Q4_K,10=Q6_K}, N, K, outputDtype{0=FP32,1=HALF})`, semantics
`out[m,n] = Σ_k A[m,k]·dequant(W[n,k]) == A @ Wᵀ` where `W` is logical `[N,K]`.

New ops (mirroring the `lora_matmul.cpp` pattern — fwd+bp co-located, composing
`sd::ops::matmul` and `sd::ops::ggml_dequantize` internally):

| Op | Inputs | Args | Output(s) | Semantics |
|---|---|---|---|---|
| `ggml_qmatmul_bp` | A, W_packed, gradOut | iArgs: quantType,N,K | dA | `dA = gradOut @ dequant(W)` (no transpose). Frozen weight → no weight grad. |
| `ggml_qmatmul_lora` | A, W_packed, loraA[rank,K], loraB[N,rank] | tArg: scaling; iArgs: quantType,N,K,outputDtype | out[.,N] | `ggml_qmatmul(A,W) + scaling·((A@loraAᵀ)@loraBᵀ)` |
| `ggml_qmatmul_lora_bp` | A, W_packed, loraA, loraB, gradOut | tArg: scaling; iArgs: quantType,N,K | dA, dLoraA, dLoraB | base+lora activation grad; adapter grads; **no** weight grad |

All support rank-2 `[M,K]` and rank-3 `[B,S,K]` activations (rank-3 handled as
`[B·S,K]`). Registered in `headers/llm.h` + `OpTraitTable.cpp`
(`MATMUL`, and `MATMUL|BP` for the backward ops).

**LoRA dims for a quantized base:** `loraA = [rank, K]` (random/Kaiming init),
`loraB = [N, rank]` (**zero** init so the adapter starts as identity),
`delta = scaling·(A @ loraAᵀ) @ loraBᵀ`.

### Injection mechanism (Java, nd4j-api)

`PeftModel` gains a quantized-aware pass in `applyLora` that runs **alongside** the
existing dense pass (hybrid graphs with both quantized and dense layers are
supported):

1. Scan `model.getOps()` for ops with `opName() == "ggml_qmatmul"`.
2. For each, read `packedWeight = input(1)` and `(quantType,N,K,outputDtype)` from
   the op's iArgs; the activation is `input(0)`.
3. Match the **packedWeight variable name** (the GGUF tensor name, e.g.
   `blk.0.attn_q.weight`) against `LoraConfig.targetModules` — same matching
   semantics as dense weights.
4. On a match: create `loraA[rank,K]` / `loraB[N,rank]`, compute the residual
   `newOut = qmatmulOut + delta` with plain SameDiff ops (the graph-level path,
   which only needs `ggml_qmatmul_bp` for backprop), and rewire consumers of
   `qmatmulOut` via the existing `replaceVariableUsages` helper.
5. Set `packedWeight` to `CONSTANT` (frozen); only `loraA`/`loraB` train.

`GgmlQMatMul.doDiff` returns the activation gradient (via `ggml_qmatmul_bp`) and a
no-op gradient for the frozen packed weight, so autograd flows through quantized
layers to everything below them.

The fused `ggml_qmatmul_lora` op is validated independently and is available as an
opt-in performance path; the default correctness path is the graph-level residual,
per the principle that the residual is sufficient for correctness and the fused
kernel is a throughput optimization.

### Conversion path (nd4j-ggml)

New preset `ConversionOptions.forTrainingQuantized()` =
`RUNTIME_QUANTIZED_MATMUL` + `forTraining(true)` + `HALF`. This is the correct
QLoRA entry point: the base stays packed (~13 GB stays ~13 GB), `ggml_qmatmul` ops
are emitted, and adapters are the only trainable params. `forTraining()`'s javadoc
is updated to warn it dequantizes to FP32 (the ~50 GB blowup) and is only for
full-parameter fine-tuning.

### Adapter persistence

`PeftModel.saveAdapter` / `loadAdapterWeights` / `loadConfig` (previously TODO
stubs) are implemented as a real, name-keyed round-trip. **Adapter-only**:
`adapter_config.json` (peftType, r, alpha, scaling, targetModules, adapter dtype,
and per quantized target `{opName, quantType, N, K, outputDtype}`) plus one `.npy`
per adapter variable, keyed by variable name. The base GGUF is never re-saved and
no `merged.sd` is written. `mergeAndUnload` on a quantized base throws a clear
error (a float delta cannot be folded into packed bytes without requantization)
rather than silently corrupting the buffer.

## Consequences

Also fixes the broader LoRA/PEFT limitation surface uncovered by the audit:

- **DoRA** no longer silently routes to plain LoRA; a dedicated `applyDora`
  creates the magnitude vector and uses `DoraMatMul`.
- **`multi_lora_matmul`** gains its missing backward (`multi_lora_matmul_bp` +
  Java `doDiff`), enabling multi-adapter training.
- **`lora_matmul`/`LoraMatMul`** accept rank-3 `[B,S,·]` activations.
- **`LoraLayer`** honors the configured adapter dtype instead of hardcoding FP32.
- **LoftQ** dequantizes an INT8 packed weight before SVD instead of feeding raw
  bytes to the initializer.

Testing (in `platform-tests/`): op-validation gradient checks for
`ggml_qmatmul_bp`, `ggml_qmatmul_lora`(+bp), and `multi_lora_matmul_bp`; a
`ggml_qmatmul_lora` forward == graph-residual equivalence check; a PeftModel QLoRA
end-to-end test on a synthetic quantized graph (loss/grad flow, packed base stays
CONSTANT, zero-init adapter is a no-op at step 0); and a `saveAdapter`/load
round-trip.

Costs: `headers/llm.h` gains new declarations (a one-time large native recompile);
the fused op adds CPU+CUDA kernels to maintain. The fused-op SameDiff fluent-API
exposure is added through op-codegen (`NeuralNetwork.kt` + regenerate) as a
separately gated step to avoid clobbering unrelated uncommitted generated sources.

## Related

- ADR 0052 GGML/GGUF Model Import, ADR 0053 GGML Quantization Handling
- `ggml_qmatmul` forward op and helper (`libnd4j/.../nn/ggml_qmatmul.cpp`,
  `helpers/ggml_qmatmul.h`)
- Canonical fwd+bp op pattern: `libnd4j/.../nn/lora_matmul.cpp`
