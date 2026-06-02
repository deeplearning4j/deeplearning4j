---
name: dsp_benchmark_current_250_20260528
description: Current 250-token DSP benchmark result after DL4J regression fixes
type: project
---

Command: `cd platform-tests && ./run-benchmark.sh --tokens 250 2>&1 | tee /tmp/dsp-benchmark-current-250-20260528.log`

Result: benchmark wrapper completed; primary benchmark status FAILED due target assertion, but DSP audit passed 13/13 suites.

Metrics:
- Config: OPTIMAL
- Tokens: 250
- overall tok/s: 13.56
- steady tok/s: 57.48
- lateSteady tok/s: 64.96
- native decode: 249 tokens in 4332 ms (57.5 tok/s)
- previous baseline: steady 56.23, lateSteady 63.78
- delta: steady +1.25, lateSteady +1.18

Replay health:
- planPhase=REPLAYING
- pointersStable=true
- fullyReplaying=true
- segments=1
- captured=1/1
- replays=245
- captureFailures=0
- health OK

Notes:
- Target remains 100 tok/s; not met.
- Correctness parser reported UNKNOWN/no generated text found, but generated text was present in benchmark log and built-in DSP audit passed.
- Milestone recorded as failing benchmark target: c4a77198.
- No code change or commit was made from this measurement round.
