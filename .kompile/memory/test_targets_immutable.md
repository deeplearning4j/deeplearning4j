---
name: test-targets-immutable
description: "IMMUTABLE: CPU tests Qwen3.5, CUDA tests VLM SmolDocling — these NEVER change, stop confusing them"
type: feedback
---

## Test Targets — NEVER CHANGE, NEVER CONFUSE

- **CPU test**: `TestQwen35Pipeline` with `-Dbackend.artifactId=nd4j-native`. Model: Qwen3.5. Goal: output "France".
- **CUDA test**: `run-benchmark.sh --tokens 250`. Model: SmolDocling VLM (pathfinder-mythic.pdf). Goal: output text about mythic heroes.

These are TWO DIFFERENT MODELS on TWO DIFFERENT BACKENDS. They have ALWAYS been this way. STOP forgetting this.

**Why:** User has repeatedly corrected confusion about which model runs on which backend. This is a fundamental fact that must never be lost.

**How to apply:** Before making any statement about test results, verify which model and which backend you're talking about. CPU = Qwen. CUDA = VLM. Period.
