---
name: dsp-benchmark-baseline-250-may28-frozen-verify-failure
description: "Baseline 250-token DSP benchmark: lateSteady 63.78 tok/s, audit blocked by frozen DataBuffer VERIFY mutation"
type: project
---

**Command:** cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && ./run-benchmark.sh --tokens 250 2>&1 | tee /tmp/dsp-benchmark-baseline-250.log
**Result:** OPTIMAL 250 tokens, overall=23.12 tok/s, steady=56.23 tok/s, lateSteady=63.78 tok/s, native decode=249 tokens in 4428 ms (56.2 tok/s), finish=MAX_TOKENS.
**Health:** segments=1 captured=1/1 replay executions=245, Triton launches=0.
**Failures:** benchmark target assertion (<100 tok/s); built-in audit passed 10/13 suites and failed replay/training/validation. Replay/validation show DataBuffer LIFECYCLE VIOLATION from allocatePrimary on frozen output during VERIFY diagnostics.
**Root cause hypothesis:** NativeDynamicShapePlan.cpp requested output VERIFY dump calls arr->buffer()/bufferAsT<float>() on a frozen device-only output. That materializes host primary and violates frozen slot identity. Fix diagnostics to use existing primary only or skip value dump for device-only outputs; do not disable replay/VERIFY.
**Logs:** /tmp/dsp-benchmark-baseline-250.log, platform-tests/bench-output.log, platform-tests/dsp-audit-replay.log, platform-tests/dsp-audit-training.log, platform-tests/dsp-audit-validation.log
