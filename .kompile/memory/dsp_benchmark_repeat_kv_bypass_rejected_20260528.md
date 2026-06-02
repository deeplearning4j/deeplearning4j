---
name: dsp_benchmark_repeat_kv_bypass_rejected_20260528
description: Benchmark 10 repeat_kv bypass candidate rejected due CUDA illegal memory access in 250-token benchmark
type: project
---

Benchmark 10 candidate attempted to bypass SmolDocling repeat_kv by feeding present.N.key/value directly into onnx_multi_head_attention and adding rank-4 BHSD K/V support in native MHA. A focused platform-tests check comparing raw BHSD GQA K/V to repeated K/V passed with maxDiff=0.0, but the required 250-token benchmark crashed before generation.

Command: cd platform-tests && ./run-benchmark.sh --tokens 250 --skip-audit 2>&1 | tee /tmp/dsp-benchmark-repeat-kv-bypass-250-20260528.log

Failure: JVM SIGABRT during frozen transition after warmup decode. Native error: dspPublishThreadCompletionEvent: cudaEventRecord failed: an illegal memory access was encountered (700). The script printed 62.87 tok/s, but that is invalid because CORRECTNESS=CRASH and 0 tokens were generated.

Decision: reject and revert repeat_kv bypass plus rank-4 MHA support/test. Keep note: simple direct raw KV path is not replay-safe as implemented; if revisited, investigate buffer lifetime/shape ownership around present.N.key/value producers and MHA consuming those same concat outputs under freeze/replay.
