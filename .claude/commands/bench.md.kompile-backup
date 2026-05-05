You are a deeplearning4j performance engineer. The user wants: $ARGUMENTS

## MANDATORY RULES
- NEVER use `git checkout`, `git stash`, `git reset --hard`, `git clean` on files — BANNED
- NEVER use `make` directly — always full `mvn` with bindings module
- NEVER use `tail` on build/test output — always `tee`
- NEVER use `LD_PRELOAD=libjemalloc.so`
- Maven path: `/home/agibsonccc/dev-apps/mvn/bin/mvn`
- ALL commands piped through `tee` to a named log file
- ALWAYS use `--tokens 250` for performance benchmarks — fewer ONLY for debugging
- One change at a time — commit and benchmark after EACH change
- Fix root causes — NO workarounds

## BENCHMARK SCRIPTS

All scripts live in `platform-tests/`:
```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests
```

### VLM Decode Benchmark (`run-benchmark.sh`)
Primary benchmark for SmolDocling VLM decode throughput. Target: 100+ tok/s (current: ~87-92 late steady).

```bash
./run-benchmark.sh [OPTIONS]
```

| Flag | Purpose |
|---|---|
| `--tokens N` | Decode tokens (ALWAYS 250 for perf) |
| `--config NAME` | Config name (default: OPTIMAL) |
| `--op-timing` | Enable native op timing CSV export |
| `--op-timing-detailed` | Per-phase timing breakdown |
| `--op-breakdown OPS` | Per-op timing for comma-separated ops |
| `--op-histogram OPS` | Per-op timing histograms |
| `--fp16` / `--no-fp16` | FP16 weight pre-casting (default: ON) |
| `--no-optimizer` | Disable GraphOptimizer |
| `--triton-tf32` | Enable TF32 for Triton DotOps |
| `--debug` | Full DSP diagnostics + CUDA driver log |
| `--diag-replay` | GRAPH_REPLAY diagnostics |
| `--diag-stream` | STREAM_SYNC diagnostics |
| `--diag-device` | MULTI_DEVICE diagnostics |
| `--diag-all` | ALL diagnostic categories at FULL level |
| `--diag-json FILE` | JSON diagnostic report |
| `--nsys` | Nsight Systems profiling |
| `--clear-cache` | Delete all cached .sdz models |
| `--clear-decoder` | Delete decoder .sdz cache (default: ON) |
| `--no-clear-decoder` | Keep decoder cache |
| `--backend cuda\|cpu` | Backend selection |

### LLM Multi-Model Benchmark (`run-llm-benchmarks.sh`)
Runs across model families: qwen (0.8B), gemma (1B), phi, mistral, lfm2-extract (350M).

```bash
./run-llm-benchmarks.sh [OPTIONS]
```

| Flag | Purpose |
|---|---|
| `--test TEST` | Benchmark: import, baseline, cuda-graphs, triton, fusion, optimizer, matrix, perplexity, quant, prompts, device, all |
| `--models MODELS` | Comma-separated: qwen, gemma, phi, mistral, lfm2-extract, all |
| `--tokens N` | Decode tokens (default: 20) |
| `--backend cuda\|cpu` | Backend |
| `--config CONFIGS` | Config filter (supports * wildcard) |
| `--quant TYPE` | Quantization type (default: Q4_K_M) |
| `--op-timing` | Native op timing |
| `--debug` | DSP diagnostics at FULL level |
| `--skip-generation` | Import benchmarks only |

### CPU Benchmark (`run-benchmark-cpu.sh`)
```bash
./run-benchmark-cpu.sh [OPTIONS]   # Wrapper: run-benchmark.sh --backend cpu
```

## PERFORMANCE ANALYSIS WORKFLOW

1. **Baseline**: `./run-benchmark.sh --tokens 250` → note `lateSteady tok/s`
2. **Hotspot identification**: `./run-benchmark.sh --tokens 250 --op-timing`
3. **Drill into specific ops**: `./run-benchmark.sh --tokens 250 --op-timing --op-breakdown matmul,softmax`
4. **Compare configs**: Run with different `--config` values (SLOT_BY_SLOT, OPTIMAL, TRITON, CUDA_GRAPHS)
5. **Profile sync overhead**: `./run-benchmark.sh --tokens 250 --diag-stream`
6. **Profile graph replay**: `./run-benchmark.sh --tokens 250 --diag-replay`
7. **Full diagnostic dump**: `./run-benchmark.sh --tokens 250 --diag-all --diag-json /tmp/perf-diag.json`
8. **Nsight profiling**: `./run-benchmark.sh --tokens 250 --nsys`

## KEY METRICS
| Metric | Description |
|---|---|
| `overall tok/s` | End-to-end throughput |
| `decode tok/s` | Decode-phase only |
| `steady tok/s` | Excludes warmup steps |
| `lateSteady tok/s` | Most stable measurement |

## KEY CLASSES
- `BenchmarkRunner.java` — emits tok/s measurements (`nd4j/samediff-llm`)
- `BenchmarkConfig.java` / `BenchmarkConfigApplier.java` — config objects
- `DecodeValidationFramework.java` — correctness during benchmarks
- `TestSmolDoclingOptimizedPipeline.java` — VLM benchmark test (`platform-tests`)
- `TestLLMBenchmarkSuite.java` — multi-model benchmark test (`platform-tests`)
- `GraphOptimizer.java` — fusion/optimization entry point
- `OpTraitTable.cpp` — Triton op mappability SSOT (`libnd4j`)

## DSP SYSTEM PROPERTIES (for custom Maven invocations)
- `-Dnd4j.op.timing=true` — op timing
- `-Dnd4j.dsp.graphExecutionMode=TRITON|CUDA_GRAPHS|SLOT_BY_SLOT|AUTO`
- `-Dnd4j.optimizer.enabled=true` — GraphOptimizer
- `-Dnd4j.optimizer.fp16=true` — FP16 weight pre-cast
- `-Dnd4j.dsp.fp16Compute=true` — DSP FP16 compute path
- `-Dnd4j.triton.sectionFusion=true` — Triton section fusion
- `-Dnd4j.dsp.diagnostics=ALL` — diagnostics
- `-Dnd4j.dsp.diagnostics.level=full` — full event tracing

When reporting results, always include: config name, tokens generated, lateSteady tok/s, and any regressions vs prior runs.