---
name: dsp_benchmark_selective_value_dep_unfreeze_rejected_250_20260528
description: DSP optimization candidate benchmark result rejected
type: project
---

# DSP benchmark: selective value-dep producer unfreeze rejected (2026-05-28)

Command:

```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && ./run-benchmark.sh --tokens 250 --skip-audit 2>&1 | tee /tmp/dsp-benchmark-selective-value-dep-unfreeze-250-20260528.log
```

Baseline from commit 4a37770da0:
- OPTIMAL 250 steady=57.48 tok/s
- lateSteady=64.96 tok/s

Candidate change:
- In `NativeDynamicShapePlan_slotexec.cpp`, changed `detectFrozenConstants()` so frozen producers feeding value-dependent shape ops are only unfrozen when the output array or cached output shape dtype is UNKNOWN.
- Motivation: prior logs showed 373 `VALUE_DEP_UNFREEZE` entries for shape-control producers like concat/shape_of/stack/expand_dims.

Result:
- Native decode: 249 tokens in 5621 ms (44.3 tok/s)
- OPTIMAL -> 250 tokens, overall=10.71 tok/s, steady=44.30 tok/s, lateSteady=54.38 tok/s
- Replay health: planPhase=REPLAYING, pointersStable=true, fullyReplaying=true, frozenExec=245, segments=1, replaying=1, captureFailures=0
- Text remained coherent in the benchmark snippet, but target gate failed below 100 tok/s.

Decision:
- Rejected because throughput regressed versus committed baseline.
- Revert exact source lines and rebuild CUDA artifacts to restore installed state.
