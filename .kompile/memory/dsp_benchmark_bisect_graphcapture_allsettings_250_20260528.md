---
name: dsp_benchmark_bisect_graphcapture_allsettings_250_20260528
description: 
type: project
---

DSP benchmark candidate BISECT_graphCapture_allSettings on 2026-05-28 used platform-tests/run-benchmark.sh --tokens 250 --config BISECT_graphCapture_allSettings --skip-audit. Result: steady=53.64 tok/s, lateSteady=60.30 tok/s, overall=11.03 tok/s. Baseline from commit 4a37770da0/default OPTIMAL: steady=57.48 tok/s, lateSteady=64.96 tok/s. Replay health OK: segments=1, captured=1/1, replays=245, capture valid=true, Triton launches=1348. Candidate regressed and was rejected; no code changes or commit.
