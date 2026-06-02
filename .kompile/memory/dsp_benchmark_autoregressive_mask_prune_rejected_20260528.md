---
name: dsp_benchmark_autoregressive_mask_prune_rejected_20260528
description: Rejected autoregressive_decode.cu post-sample mask unmask removal after 250-token DSP benchmark regression.
type: project
---

Benchmark 8 tried exactly one native optimization in libnd4j/include/ops/declarable/helpers/cuda/autoregressive_decode.cu: remove the post-sample attentionMask/causalMask/attnMaskReformat unmask launches for kvJustWritten because the current KV position is already pre-unmasked before graph replay each step.

Build: CUDA restore/candidate builds passed with /home/agibsonccc/dev-apps/mvn/bin/mvn -Pcuda -Dlibnd4j.triton=ON -Dlibnd4j.chip=cuda -Dlibnd4j.buildthreads=12 -Dlibnd4j.log=libnd4j-build.log -pl libnd4j,:nd4j-cuda-12.9 clean install -DskipTests.

Baseline at commit 4a37770da0: steady=57.48 tok/s, lateSteady=64.96 tok/s for platform-tests/run-benchmark.sh --tokens 250.

Candidate first run with --skip-audit: steady=56.41, lateSteady=65.44, overall=11.43, replay health OK.
Candidate confirmation run with --skip-audit: steady=56.26, lateSteady=63.67, overall=17.12, native decode=56.3, replay health OK.

Decision: reject. Steady-state throughput regressed and confirmation also regressed lateSteady. Exact source edit was reverted, diff is clean, CUDA restore build passed, no commit.

Why: The kernels may not be redundant in timing terms or the measurement noise does not support removing them; future attempts should not retry this exact mask-prune change as a performance optimization unless paired with a broader replay/capture restructuring and a correctness rationale.

Milestones: failure b9033a8e, restore build 7060b38d.
