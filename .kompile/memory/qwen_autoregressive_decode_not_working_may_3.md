---
name: Qwen autoregressive decode NOT working May 3
description: Qwen prefill works (token 271) but full autoregressive decode produces garbage — same MHA regression as VLM
type: project
---

# Qwen Autoregressive Decode NOT Working (May 3 2026)

## Status
- **Prefill (single forward pass)**: WORKING — token 271 (Paris) confirmed at commit 41a3b05be4
- **Full autoregressive decode**: BROKEN — produces `<think>\n\nGlobal B.1 P un <<CAA. P1` (garbage)

## Root Cause
Same as VLM regression: `onnx_multi_head_attention.cpp` workspace buffer removal + syncToDevice removal. The prefill-only test (TestQwenLayerDiagnostics) passes because it doesn't do decode steps — it only checks the argmax of the last prefill position. But actual generation (GenerationPipeline) does autoregressive decode which exercises the KV concat path repeatedly.

## Test Clarification
- `TestQwenLayerDiagnostics#testFrancePromptVerbose` — single prefill, checks token 271. PASSES.
- `TestQwen35Pipeline#testQwen35Pipeline` — full generation with SLOT_BY_SLOT + CUDA_GRAPHS. FAILS (garbage + Triton OOM).
- `TestQwen35Pipeline#testQwen35ReferencePrompts` — validates "Paris" appears in output. Never tested.

## Chat Template Note
Qwen3.5 generates `<think>` (token 151649) as first output due to chat template. This is EXPECTED. The real problem is garbage AFTER `<think>`.

## Fix
Same as VLM: restored AttentionWorkspace buffer + syncToDevice in onnx_multi_head_attention.cpp. Build in progress.

**How to apply:** After build, run `TestQwen35Pipeline#testQwen35Pipeline` and check for coherent output after `<think>` token. Also run `TestQwenLayerDiagnostics#testFrancePromptVerbose` as prefill sanity check.
