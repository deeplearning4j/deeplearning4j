---
name: dsp_benchmark_onnx_mha_direct_output_rejected_20260528
description: Benchmark 9 direct-output onnx_multi_head_attention decode candidate regressed and was reverted
type: project
---

# DSP benchmark: onnx MHA direct output candidate rejected (2026-05-28)

Candidate: in `onnx_multi_head_attention.cpp`, use direct output reshape for `seqQ == 1` even when attention bias is present, avoiding workspace nullify + output assign after fusedGQADecode.

Build: CUDA native build passed with `/home/agibsonccc/dev-apps/mvn/bin/mvn -Pcuda -Dlibnd4j.triton=ON -Dlibnd4j.chip=cuda -Dlibnd4j.buildthreads=12 -Dlibnd4j.log=libnd4j-build.log -pl libnd4j,:nd4j-cuda-12.9 clean install -DskipTests`, log `/tmp/dsp-onnx-mha-direct-output-cuda-build-20260528.log`.

Benchmark: `platform-tests/run-benchmark.sh --tokens 250 --skip-audit`, log `/tmp/dsp-benchmark-onnx-mha-direct-output-250-20260528.log`.

Result: steady/native decode=55.17 tok/s, lateSteady=62.87 tok/s, overall=9.96. Baseline commit 4a37770da0 was steady=57.48 tok/s, lateSteady=64.96 tok/s. Replay health OK: REPLAYING, pointersStable=true, fullyReplaying=true, frozenExec=245, segments=1, captured=1/1, replays=245, captureFailures=0.

Decision: rejected because both steady and lateSteady regressed. Exact source edit reverted; rebuild baseline artifacts before next optimization.

How to apply: Do not retry the simple direct-output wrapper change for current VLM. If revisiting MHA, inspect kernel-level timings or fuse/remove upstream repeat_kv/shape nodes instead.
