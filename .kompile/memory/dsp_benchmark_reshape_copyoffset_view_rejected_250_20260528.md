---
name: dsp_benchmark_reshape_copyoffset_view_rejected_250_20260528
description: DSP optimization candidate benchmark result rejected
type: project
---

# DSP benchmark: reshape copy-offset view candidate rejected (2026-05-28)

Command:

```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && ./run-benchmark.sh --tokens 250 --skip-audit 2>&1 | tee /tmp/dsp-benchmark-reshape-copyoffset-view-250-20260528.log
```

Baseline from commit 4a37770da0:
- OPTIMAL 250 steady=57.48 tok/s
- lateSteady=64.96 tok/s

Candidate change:
- In `NativeDynamicShapePlan_slotexec.cpp`, allowed `reshape`/`reshape_no_copy` view creation to bypass generic input C-contiguity precheck when output shape metadata had `ARRAY_COPY_OFFSET_INPUT_0` and did not have `ARRAY_NEEDS_COPY`.

Result:
- Native decode: 249 tokens in 4920 ms (50.6 tok/s)
- OPTIMAL -> 250 tokens, overall=17.42 tok/s, steady=50.61 tok/s, lateSteady=58.00 tok/s
- Replay health: planPhase=REPLAYING, pointersStable=true, fullyReplaying=true, frozenExec=245, segments=1, replaying=1, captureFailures=0
- Target gate failed below 100 tok/s; health OK.

Decision:
- Rejected because throughput regressed versus committed baseline.
- Revert exact source lines and rebuild CUDA artifacts to restore installed state.
