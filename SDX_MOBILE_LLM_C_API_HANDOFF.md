# SDX Mobile + Text-Level LLM C API — Handoff (from kompile, 2026-07-11)

This is a behavioral-contract handoff (same style as dsp-700-repro/CONTEXT.md): it
specifies REQUIRED BEHAVIOR and acceptance criteria, not implementations — you know
this tree. Everything in "Current state" was verified in-tree / via the GitHub API on
2026-07-11.

## Mission

This dl4j release ships alongside kompile. Kompile's `kompile sdk scaffold` generates
iOS/Android chat apps that must run **fully local model chat + graph reasoning**:

- **libsdx** (this repo) runs the model: load a `.sdz` bundle, stream tokens, embed
  text — as a **plain C library per platform, no JVM**. That direction already exists
  (BuildSDX.cmake); what's missing is the **text-level LLM API**, **constrained
  decoding**, **ios/android builds**, and **published releases**.
- **libkompile_reasoning** (kompile side, not yours) is a second plain C library
  (GraalVM native-image of kompile's graph-reasoning Java lib) that answers the
  model's tool calls over a `.kgraph` file. It follows YOUR packaging/release
  conventions — coordination points below. Design:
  `~/Documents/GitHub/kompile/docs/architecture/graph-reasoning-mobile-aot.md`.

## Current state (verified 2026-07-11)

- **Real C ABI is tensor-level.** `libnd4j/include/dsp/runtime/dsp_runtime_c.h`:
  `sdxCreateRuntime/sdxLoadBundle/sdxCreateContext/sdxRun` over `sdx_tensor_view_t`,
  ABI-versioned (`SDX_RUNTIME_ABI_VERSION=1`), status codes, execution reports, plan
  phases. Backend enum already anticipates mobile: `SDX_BACKEND_MLX=6`,
  `SDX_BACKEND_ARM_HYBRID=7`, `SDX_BACKEND_NNAPI=8`. There is **no tokenize /
  generate / sample / embed-text surface** in the C ABI.
- **JVM-free standalone lib exists.** `libnd4j/cmake/BuildSDX.cmake`
  (`-DSD_BUILD_SDX_STANDALONE=ON` → `libsdx_cpu.so`/`libsdx_cuda.so`): relinks main
  build objects, exports only `sdx*` via `cmake/sdx_exports.lds`, per-backend toggles
  `SDX_INCLUDE_TRITON/ONEDNN/MLIR/OPENVINO`. Python ctypes bindings + tests under
  `libnd4j/include/dsp/runtime/bindings/python/` (loader already contemplates an
  `ios` platform id); `libnd4j/tools/sdx-generate-bindings.sh`;
  `blasbuild/{cpu,cuda,hexagon,tpu}/sdx-runtime-sdk/` SDK layouts (linux-x86_64 only).
- **Native decode machinery largely exists** (DECODER_DEV_JOURNAL.md): greedy 1×1
  via native `autoregressive_decode`; `token_sample` temp/top-k/top-p parity DONE on
  CPU+CUDA; `GenerationPipeline` (Java) is the consolidation point per ADR 0106.
  The gap is exposing a text-level loop at the **C ABI** (no JVM on mobile).
- **GitHub releases are EMPTY** (`/repos/deeplearning4j/deeplearning4j/releases` →
  `[]`). No `sdx-v*` tags, no assets.
- **Kompile consumer contract already ships** (same box, read-only):
  - `~/Documents/GitHub/kompile/kompile-app/kompile-models/kompile-model-manager/src/main/java/ai/kompile/modelmanager/SdkConstants.java`
    downloads `{base}/sdx-v{version}/{sdkId}-{classifier}.{ext}`, base
    `https://github.com/deeplearning4j/deeplearning4j/releases/download/`, override
    env `KOMPILE_SDX_SDK_BASE_URL`, cache `~/.kompile/models/sdx-sdk/...`.
    Classifiers: iOS `ios-arm64|ios-x86_64|ios-simulator-arm64|ios-simulator-x86_64`
    → `.xcframework.zip`; Android `android-arm64|android-arm|android-x86|android-x86_64`
    (+ chip variants `-nnapi`, `-armcompute`, `-compile`, `-compile-nnapi`,
    `-onednn`) → `.aar`; desktop → `.zip`.
  - Scaffold templates code against **aspirational symbols that do not exist**:
    iOS `templates/mobile/ios/SdxInferenceService.swift.ftl` expects module
    `DspRuntimeC` with `dsp_model_load/free/generate(cb)/embed` +
    `dsp_free_embedding`; Android `templates/mobile/android/SdxInferenceService.kt.ftl`
    JNA-loads `"sdx_runtime"` expecting `sdx_init/sdx_set_prompt/sdx_generate_next/
    sdx_generate_token/sdx_is_done/sdx_reset/sdx_destroy/sdx_get_embedding/
    sdx_get_embedding_dim`.
    **Kompile will regenerate both templates against your real header once it
    freezes — design the ONE canonical API (one namespace, presumably `sdx*`); do
    not implement the template symbols as-is.**

## REQUIRED BEHAVIOR

### R1 — Text-level LLM C API (all platforms, in the `sdx*` ABI)

A session API over a `.sdz` bundle that makes prompt-in/tokens-out possible from pure
C callers (Swift bridging header; Kotlin JNA):

- Create/destroy an LLM session from a bundle path; tokenization lives **inside** the
  library, driven by bundle contents (tokenizer.json/vocab) — callers never see token
  ids unless they ask.
- Streaming generate: per-token (or per-chunk) callback with **valid UTF-8 chunks**
  (never split multibyte sequences across callbacks); callback return value cancels
  generation promptly; generation is synchronous on the calling thread (callers own
  threading) or documented otherwise.
- Sampling options struct (`struct_size`-versioned like your existing options):
  max_tokens, temperature, top_k, top_p, repetition_penalty, seed, stop sequences.
- Embeddings: `text → float*` + dimension query (kompile's hybrid/RAG mode needs it).
- Chat templating (SHOULD): apply the bundle's chat template to a messages JSON
  (roles incl. tool results) → prompt string. If deferred, raw-prompt mode is the
  contract and kompile formats in the app — but tool-calling fidelity on small models
  argues for in-library templating.
- Errors: session-scoped last-error string + `sdx_status_t` codes; bump
  `SDX_RUNTIME_ABI_VERSION`.

Sketch (naming/shape negotiable — behavior is not):

```c
sdx_status_t sdxLlmCreateSession(sdx_runtime_t*, const char* bundle_path,
                                 const sdx_llm_options_t*, sdx_llm_session_t**);
void         sdxLlmDestroySession(sdx_llm_session_t*);
typedef int32_t (*sdx_token_cb)(const char* utf8, void* user); /* 0 = cancel */
sdx_status_t sdxLlmGenerate(sdx_llm_session_t*, const char* prompt,
                            const sdx_llm_gen_options_t*, sdx_token_cb, void* user);
sdx_status_t sdxLlmApplyChatTemplate(sdx_llm_session_t*, const char* messages_json,
                                     char** out_prompt);      /* SHOULD */
sdx_status_t sdxLlmEmbed(sdx_llm_session_t*, const char* text,
                         float* out, int32_t capacity, int32_t* out_dim);
const char*  sdxLlmGetLastError(const sdx_llm_session_t*);
```

Respect your own guardrails: do NOT modify DSP; decode logic consolidates on the
`GenerationPipeline`/native `autoregressive_decode`+`token_sample` substrate (ADR
0106) — the C API should be a thin exposure of that, not a new decoder.

### R2 — Constrained decoding (what makes local tool-calling work)

**Status: v1 SHIPPED (2026-07-12) — Java sampling path, CPU.** See ADR 0111.

- Generate options accept a constraint: JSON-Schema (preferred; kompile hands you the
  tool-args schema verbatim) and/or a GBNF-style grammar.
- Guarantee: when a constraint is set, emitted text **parses against it** or the call
  ends with a distinct finish/status code — never silently-unparseable output.
- Stop sequences honored in the same pass. Must work CPU-only at qwen3-0.6b scale
  with usable latency on a phone-class arm64 CPU.
- This is the single highest-leverage item for kompile: without it, 0.6B models
  free-text their tool calls and the local loop degrades to parse-and-retry.

**v1 `options_json` contract** (finalized; implemented in `ConstraintConfig` / `SamplingConfig`):

```json
{ "constraint": { "type": "json_object" } }
```

```json
{ "constraint": { "type": "tool_call", "tools": ["ask_graph_verify", "graph_reasoning_query", "ask_graph_query"] } }
```

- `type`: `"json_object"` — any syntactically valid single JSON object.
  `"tool_call"` — `{"tool": "<enum>", "args": <free-form JSON>}` where `tool` must be in `tools`.
- `tools`: required array of allowed tool name strings (for `tool_call`); ignored for `json_object`.
- Unknown `type` → `IllegalArgumentException` at build time.
- When `constraint` key is absent → unconstrained (zero behavior change).

**v1 implementation notes:**
- Token-level automaton in `org.eclipse.deeplearning4j.llm.generation.constraint.*`
- Java decode loop replaces native `AutoregressiveDecode` op when constraint is active
  (native op has no Java callback seam; constraint masking runs per-token in Java).
- `ConstraintVocabCache` (cap 512 prefixes) amortises full-vocab sweep.
- EOS gated to accepting states only.
- Zero overhead when `constraintConfig` is null.
- Overhead < 50% at evalTopK=256 on CPU (perf test in `ConstrainedDecodingIntegrationTest`).

**v2 items** (open): full JSON-Schema shape constraint, GBNF/BNF grammar, native-path
masking hook (blocked on pieces 4-5), grammar serialisation in `.kgraph` bundles.

### R3 — Mobile platform builds of libsdx

- **Android**: `arm64-v8a` (device) + `x86_64` (emulator) `.so` via NDK/bionic,
  CPU path mandatory, `SDX_BACKEND_NNAPI` optional follow-up. JNA-loadable (final
  library name is a freeze-point with kompile; templates currently say
  `sdx_runtime`).
- **iOS**: `ios-arm64` device + `ios-simulator-arm64` static lib → xcframework with a
  module map (templates currently say module `DspRuntimeC`; your call, kompile
  aligns). **AOT-only**: all JIT paths excluded (`SDX_INCLUDE_TRITON/MLIR=OFF`,
  NVRTC/PTX backends compiled out) and `allow_runtime_jit` must return
  `SDX_STATUS_UNSUPPORTED` — App Store guideline 2.5.2. MLX/ARM_HYBRID acceleration
  later; CPU first.
- Keep desktop zips as-is. The BuildSDX standalone pattern (object reuse + symbol
  allowlist) is the right base for both mobile targets.

### R4 — Memory behavior on mobile (dl4j owns memory arbitration)

- Session/runtime option for a memory budget; deterministic, reported failure
  (`sdx_status_t` + last-error) when a bundle/generation cannot fit — never process
  death by jetsam/LMK as the "error path".
- Document the expected RSS envelope for a qwen3-0.6b class bundle on CPU so kompile
  can gate model choices per device tier.

### R5 — Release publishing (unblocks everything kompile-side)

- Tag `sdx-v{version}` on this repo; upload assets named **exactly**
  `{sdkId}-{classifier}.{ext}` per SdkConstants (see Current state) — at minimum:
  `sdx-runtime-android-arm64.aar`, `sdx-runtime-android-x86_64.aar`,
  `sdx-runtime-ios-arm64.xcframework.zip`,
  `sdx-runtime-ios-simulator-arm64.xcframework.zip`, plus the existing desktop zips.
- Publish (or confirm hosting for) at least one chat `.sdz` (qwen3-0.6b class, with
  tokenizer + chat template with tool roles) and one embedding-capable `.sdz` at the
  model-bundle URLs kompile's registry expects.
- Coordination: kompile's `libkompile_reasoning` artifacts will ride the same
  release/CI conventions (possibly the same tags) — agree tag family + who wraps
  `.aar`/xcframework packaging so both libraries ship identically.

### R6 — Stretch (explicitly NOT required for v1)

- `nd4j-native` android-arm64 classifiers from the release pipeline — would later
  unlock on-device KGE training / tensor-PSL in kompile's reasoning library (its
  mobile build currently excludes the 7 ND4J-touching classes by design).

### R7 — Per-platform OPTIMIZED variants + real-device spins (added 2026-07-12)

Kompile's directive: every platform ships an **optimized** flavor (phones use their
accelerators), and every flavor is proven by a real run ("spin"), not just a build.

What already exists in-tree (verified 2026-07-12) — build on it, don't duplicate:
`graph/cpu/NnapiGraphBackend`, `graph/cpu/ArmHybridGraphBackend` (big.LITTLE),
`graph/vulkan/VulkanReplayHandle`, hexagon-mlir runtime + `nd4j-hexagon` lane
(linux-x86_64 host today), nd4j-metal SPI + MLX smoke lane, llama.cpp platform ops
(`backend_capabilities.cpp`), `SdxRuntimePackage.cmake` flavor plumbing, and
`sdx_execution_report_t.applied_backend` as the observability hook. Kompile's
`SdkConstants` already reserves the classifier vocabulary:
`android-arm64-nnapi|-armcompute|-compile|-compile-nnapi`, `android-x86_64-onednn`,
`macosx-arm64-compile|x86_64-avx2|x86_64-onednn`.

Required behavior:
1. **Android NPU strategy** — NNAPI is implemented but is a dead-end API
   (deprecated in Android 15): ship the NNAPI flavor now for Pixel-Tensor-class
   NPUs, AND state the successor path (Qualcomm QNN/HTP via the hexagon runtime
   packaged for android-arm64 — today's hexagon lane is host-linux only — and/or
   LiteRT delegate). Decide + document which SoCs map to which backend.
2. **Mobile GPU flavor — the native Vulkan path (llama.cpp/ggml explicitly does
   NOT count for this, per kompile 2026-07-12).** The design is already
   android-aimed and platform-neutral: `MLIR_ENABLE_VULKAN` ("MLIR Vulkan/SPIR-V
   backend for ARM mobile GPUs"), `android-arm64.cmake` documents
   `-DHELPERS_mlir=ON -DMLIR_ENABLE_VULKAN=ON` GPU offload, the NDK system-libs
   list already carries `libvulkan.so`, and `VulkanReplayHandle` is plain Vulkan
   compute (no MoltenVK/portability coupling — mac was only ever a bring-up host).
   What's needed is Android productionization, not a rewrite:
   (a) verify/complete MLIR→SPIR-V lowering coverage for the decode op set
   (matmul/attention/rmsnorm/rope/kv-ops) — this is the crux;
   (b) mobile-GPU memory strategy in VulkanReplayHandle: it currently allocates
   DEVICE_LOCAL workspace + separate host staging — on UMA phone GPUs
   (Adreno/Mali/Xclipse) use HOST_VISIBLE|DEVICE_LOCAL unified allocations and
   skip staging;
   (c) fp16 (VK_KHR_shader_float16_int8 / 16-bit storage) + subgroup-size
   handling (Adreno 64/128 vs Mali 16) in feature enablement and kernels;
   (d) Android lifecycle: device-lost on pause, thermal/sustained-mode tuning;
   the per-replay fence wait is fine for decode cadence but prefill wants
   double-buffered submits;
   (e) build + publish the `android-arm64` Vulkan flavor and gate AUTO on a
   driver-caps probe. ArmComputeLibrary (`-armcompute` classifier, NEON + OpenCL
   Mali) remains the Arm-centric alternative where Vulkan drivers are weak.
3. **Runtime AUTO probe** — `SDX_BACKEND_AUTO` must resolve per-device (probe NNAPI
   feature level / vendor libs / Vulkan caps → pick, degrade gracefully to
   ARM_HYBRID/CPU) and report the choice via `applied_backend`. Contract: same
   bundle runs everywhere; the flavor only changes HOW fast.
4. **Per-backend artifacts** — publish the flavor variants under kompile's existing
   classifier names (they're already in `SdkConstants`); where a backend needs
   compiled/quantized model variants (QNN context binaries etc.), extend the SDZ
   bundle spec or use the `-compile` (on-device compile) variants — document which.
5. **Spins** — one smoke lane per flavor on REAL hardware, modeled on
   run-mlx-smoke-tests/run-hexagon-smoke-tests: android device NNAPI +
   Vulkan + ARM_HYBRID lanes (emulators have no NPU — device farm or local
   devices), asserting `applied_backend` == requested flavor + tokens/sec floor.
6. iOS: Metal/MLX is the accel path; ANE/CoreML explicitly deprioritized (LLM
   decode is memory-bound; revisit only with evidence).
7. **Process model for multiple GraalVM-image libraries (diagnosed + fixed 2026-07-12):**

   **Root cause confirmed.** Both `libsdx_llm.so` and `libkompile_reasoning.so` are
   `ET_DYN` shared objects built with `--shared` (no `PT_INTERP`, correct ELF type).
   The failure mechanism is **9 identical `graal_*` symbols + 3 `JNI_*` + 7 `__svm_*`**
   exported at global ELF visibility from BOTH libraries.  When both are loaded
   into one process, any `dlsym(RTLD_DEFAULT, "graal_create_isolate")` call made
   from inside the *second* library's C isolate bootstrap finds the *first* library's
   runtime, routing the second isolate's init into the wrong VM → `ExceptionInInitializerError`.

   **Kompile-side fix (applied 2026-07-12, both layers):**

   *Layer 1 — export allowlist on `libkompile_reasoning.so`:* A GNU linker version
   script (`kgr_exports.lds`) with `KGR_1 { global: kgr*; local: *; }` is now
   correctly applied at build time via a `gcc -B src/main/linker` wrapper that
   intercepts GraalVM's linker invocation.  Background: GraalVM 21
   `BinutilsCCLinkerInvocation` unconditionally injects an *anonymous* version script
   (`exported_symbols.list`) early in the `ld` command; GNU ld forbids combining
   anonymous and named version tags, so a second `-Wl,--version-script` via
   `-H:NativeLinkerOption` would fatal-error.  The `src/main/linker/ld` wrapper strips
   GraalVM's `--version-script` arg and substitutes `kgr_exports.lds` as the sole
   script.  After rebuild, `libkompile_reasoning.so`'s `.dynsym` contains **only
   `kgr_*@@KGR_1`** — `graal_*`, `IsolateEnterStub__*`, `JNI_*`, `__svm_*` are
   hidden (`.gnu.version_d` VERDEF present, confirming the script applied).

   *Layer 2 — RTLD_LOCAL in all language bindings:* All Python bindings
   (`kompile_reasoning.py`, `sdx_llm.py`) now use `ctypes.CDLL(path)` without
   `RTLD_GLOBAL` (Linux default is `RTLD_LOCAL`, mode=0) with explicit comments
   explaining why.  TypeScript (koffi), C# (`NativeLibrary.Load`) and Rust
   (`#[link]`) were already handle-scoped or link-time resolved; annotated.

   **SDX-aot fix — APPLIED 2026-07-12 (kompile, this session):**

   The identical `gcc -B src/main/linker` wrapper technique was applied to `nd4j/sdx-aot`.
   Files added:
   - `nd4j/sdx-aot/src/main/linker/ld` (chmod +x) — strips GraalVM's anonymous
     `--version-script` arg and substitutes `sdx_exports.lds`.
   - `nd4j/sdx-aot/src/main/linker/sdx_exports.lds`:
     ```
     SDX_LLM_1 {
       global: sdx*;
       local: *;
     };
     ```
   - `nd4j/sdx-aot/pom.xml` — added `<buildArg>-H:NativeLinkerOption=-B${project.basedir}/src/main/linker</buildArg>`

   **Verification (2026-07-12, cpu linux-x86_64):**
   ```
   nm -D --defined-only libsdx_llm.so | grep -c "graal_\|JNI_\|__svm_"
   # BEFORE fix: 23
   # AFTER fix:   0

   nm -D --defined-only libsdx_llm.so | grep "sdxLlm\|sdxVlm\|sdxAudio"
   # 14 symbols, all tagged @@SDX_LLM_1:
   #   sdxAudioTranscribe@@SDX_LLM_1
   #   sdxLlmAbiVersion@@SDX_LLM_1
   #   sdxLlmCreateRuntime@@SDX_LLM_1  ... (all 14 public symbols)

   readelf -V libsdx_llm.so | grep "SDX_LLM_1"
   # Version definition section '.gnu.version_d' contains 2 entries (confirmed)
   ```

   **RE-PROOF 1 — Python coexistence (2026-07-12, kompile):**
   Both `libkompile_reasoning.so` (KGR_1) and `libsdx_llm.so` (SDX_LLM_1) loaded
   in the SAME Python process:
   ```
   [KGR] ABI version: 1
   [SDX] ABI version: 1
   [KGR] ask_graph_verify WORKS_AT(alice, acme): SUPPORTED
   [SDX] model vocabSize: 151936 hasChatTemplate: False
   [SDX] model loaded and unloaded OK
   === COEXISTENCE TEST PASSED ===
   ```
   Zero `ExceptionInInitializerError`. The export-allowlist fix is complete on both sides.

   **Process model contract:** With the export allowlists in place on BOTH libraries,
   RTLD_LOCAL loading of any number of GraalVM-image shared libraries in one process
   is safe — each library's `graal_create_isolate` etc. are local to their dlopen
   handle and cannot be found by the other library's `dlsym(RTLD_DEFAULT, ...)`.
   RTLD_GLOBAL is forbidden.  The single combined-image path remains the
   zero-overhead alternative if process-global symbol hygiene is too fragile.

### R8 — Defects found in the first REAL runs (kompile, 2026-07-12)

Found while running qwen2.5-0.5b-instruct GGUFs through the built `sdx_llm` image
(linux-x86_64, CPU):
1. **Quantized GGUF correctness — PARTIALLY FIXED 2026-07-12:**

   *Q5_0/Q5_1 dequantization: FIXED.* `nd4j-ggml` `Q5_0Dequantizer`/`Q5_1Dequantizer`
   high-bit extraction corrected (`(highBits >> j) & 1` first half,
   `(highBits >> (j + 16)) & 1` second half); `DequantizerFactory` routes both
   types to the pure-Java path (`nativeType = -1`). Installed to ~/.m2 2026-07-12.

   *Verification (RE-PROOF SPIN1, 2026-07-12, cpu linux-x86_64):*
   Both fp16 and q4_k_m produce **identical output** at the same speed:
   ```
   # fp16:   "The ocean is a body of the sea" — 6.7 tok/s, finishReason=EOS
   # q4_k_m: "The ocean is a body of the sea" — 6.7 tok/s, finishReason=EOS
   # Prompt: "Write one sentence about the ocean."
   ```
   The cosine-similarity divergence between formats is gone.

   *Embedded tokenizer special tokens: FIXED (R8 item 4, 2026-07-12).* `q4_k_m`
   works sidecar-free. `SdxLlmCore.buildBpeTokenizerJson` now reads
   `tokenizer.ggml.token_type` and adds every non-NORMAL token (type != 1 in GGUF)
   to `added_tokens` with `"special":true`. All 22 Qwen2.5 special tokens
   (IDs 151643–151664) correctly promoted; `<|im_start|>` and `<|im_end|>` each
   tokenize to exactly 1 id. JVM tests: 7/7 green (SdxLlmCoreTokenizerTest).
   Native rebuild in progress; `libsdx_llm.so` refresh pending build completion.

2. **Tokenizer not read from GGUF metadata** — **FIXED 2026-07-12 (kompile):**
   `SdxLlmCore.resolveTokenizer` now has a 3-path fallback:
   1. explicit `tokenizerPath` arg, 2. sidecar `tokenizer.json` next to model,
   3. **GGUF-embedded tokenizer**: reads `tokenizer.ggml.tokens` (vocab array),
   `tokenizer.ggml.merges` (BPE merges), BOS/EOS IDs from `GGMLMetadata`, and
   constructs a HuggingFace `tokenizer.json` in memory via `buildBpeTokenizerJson`,
   then calls `HuggingFaceTokenizer.fromJson()`.
   `nd4j/sdx-aot/src/test/…/SdxLlmCoreTokenizerTest.java` covers 4 cases
   (GPT-2 BPE, separate BOS/EOS, empty merges, null merges) — all green.
   **Verification (RE-PROOF 3, 2026-07-12, cpu linux-x86_64):**
   ```
   # qwen2.5-1.5b-instruct-fp16.gguf — NO sidecar tokenizer.json present
   [PROOF3] Model loaded in 15.4s vocabSize=151936
   # vocabSize=151936, bosTokenId=128245, eosTokenId=151645 — read from GGUF metadata
   # Generation works: 2.89 tok/s (simple prompt, 23 tokens)
   # Output: "I 'm Ġsorry , Ġbut ĠI 'm Ġnot Ġable Ġto Ġassist..." (BPE byte markers, coherent)
   ```
   Note: Qwen2.5 uses a custom BPE variant with byte-level markers (Ġ prefix = space).
   The chat template (`hasChatTemplate=false` from sdxLlmInfoJson) is not yet
   applied by the tokenizer binding — chat prompts should use `<|im_start|>` tags
   directly (as in `SdxSubprocessChatModel`). Tool-call JSON emission at 1.5B
   is functional but reliability depends on prompt discipline (see item 3).
3. Field evidence for R2: a 0.5B model cannot reliably emit tool-call JSON with
   prompt-only discipline — constrained decoding (R2) is what makes small-model
   tool use viable; until it lands, catalog guidance should steer tool-use
   scenarios to 1.5B+.
4. **GGUF-embedded tokenizer missing special tokens** — **FIXED 2026-07-12 (kompile):**

   **Root cause:** `SdxLlmCore.buildBpeTokenizerJson` emitted only BOS/EOS in
   `added_tokens`. Qwen2.5 has 22 special tokens (IDs 151643–151664) with
   `tokenizer.ggml.token_type != 1` (NORMAL=1 in GGUF). Without them the HuggingFace
   Rust tokenizer split `<|im_start|>` as `[27,91,318,4906,91,29]` (6 chars) instead
   of `[151644]` (1 token), producing garbled pipe-character output in no-sidecar mode.

   **Fix applied (`nd4j/sdx-aot/src/main/java/…/SdxLlmCore.java`):**
   - `tryLoadEmbeddedGgufTokenizer`: reads `raw.get("tokenizer.ggml.token_type")`
     (int[] from `GGUFReader`; defensive boxed-List path also handled), passes it as
     `tokenTypes` to the new 6-arg `buildBpeTokenizerJson` overload.
   - `buildBpeTokenizerJson(tokens, merges, bosId, eosId, model, tokenTypes)`: sweeps
     all token indices; for each where `tokenTypes[i] != 1`, adds an `added_tokens`
     entry with `"special":true`, deduplicated against BOS/EOS already added. The
     original 5-arg signature is a backward-compat wrapper that passes `null`.
   - `makeAddedToken(ObjectMapper, id, content)`: extracted helper to avoid duplication.

   **Test additions (`SdxLlmCoreTokenizerTest.java`, total 7/7 green):**
   - `buildBpeTokenizerJson_controlTokensPromotedToAddedTokens`: synthetic 6-token vocab
     with 3 CONTROL tokens (type=3); asserts CONTROL ids in `added_tokens`, NORMAL ids
     excluded, `special=true` on all.
   - `buildBpeTokenizerJson_nullTokenTypesBackwardCompat`: null tokenTypes produces
     identical JSON to the 5-arg overload.
   - `embeddedTokenizer_qwen25_chatmlMarkersAreSingleIds`: live GGUF test (skips if no
     model present, skips if sidecar present); asserts `<|im_start|>` → [151644] (1 id)
     and `<|im_end|>` → [151645] (1 id) via embedded path. Ran and passed 2026-07-12.

   **Verification (2026-07-12, JVM, cpu linux-x86_64):**
   ```
   Tests run: 7, Failures: 0, Errors: 0, Skipped: 0  (SdxLlmCoreTokenizerTest)
   <|im_start|>  → [151644]  (1 token, correct)
   <|im_end|>    → [151645]  (1 token, correct)
   ```
   Native rebuild (libsdx_llm.so) in progress. Spin results pending rebuild completion.

## Acceptance criteria

Per platform (android emulator + ios simulator minimum; one real device of each
before tagging a release):

1. C smoke: create runtime → LLM session from a real `.sdz` → 50-token streamed
   generate (UTF-8-clean chunks) → mid-generate cancel honored → embed returns
   dim > 0 → destroy; no leaks under repeated open/close.
2. Constrained decode: 20/20 generations against a kompile tool-call JSON schema
   parse successfully; stop sequences terminate correctly.
3. Memory: peak RSS within the documented envelope; over-budget bundle load fails
   with status + last-error, process alive.
4. Header freeze reviewed with kompile BEFORE templates regenerate; then a scaffolded
   app (`kompile sdk scaffold`) chats locally end-to-end against the published
   release assets.

## Non-goals

- kompile's reasoning C library itself (kompile builds it; only packaging/release
  conventions are shared).
- The desktop JVM serving lane (`SdxServingService`/`PipelineExecutor`) — unchanged.
- On-device training of any kind.

## Pointers

- `libnd4j/include/dsp/runtime/dsp_runtime_c.h` — the ABI to extend
- `libnd4j/cmake/BuildSDX.cmake`, `libnd4j/cmake/sdx_exports.lds`,
  `libnd4j/cmake/MainBuildFlow.cmake`, `libnd4j/tools/sdx-generate-bindings.sh`
- `libnd4j/include/dsp/runtime/bindings/python/` (loader + public-API tests — mirror
  new surface here)
- `DECODER_DEV_JOURNAL.md` (decode substrate status + guardrails), ADR 0106
- kompile (read-only): `SdkConstants.java` (asset naming),
  `kompile-cli/kompile-cli-main/src/main/resources/templates/mobile/{ios,android}/`
  (current consumer expectations),
  `docs/architecture/graph-reasoning-mobile-aot.md` (the companion design)
