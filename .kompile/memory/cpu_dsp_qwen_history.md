---
name: cpu-dsp-qwen-history
description: Reconstructed CPU DSP Qwen/TextGenerator last-good timeline and related commits
type: project
---

# CPU DSP Qwen/TextGenerator History

Reconstructed on 2026-05-01 from local git history, reflog, /tmp tee logs, and project memory.

## Related commits

- `374f45615f` (2026-02-05 12:18 +0900): earliest TextGenerator snapshot in this tree.
- `de3d47dc54` (2026-03-12 10:37 +0900): added `TestQwen35Pipeline` and another `TextGenerator` snapshot.
- `5006d67a10` / `35111b866f` (2026-03-25 20:01 +0900): fixed Qwen3.5 GDN text generation architecture issues; changed `TextGenerator`.
- `8f4cd48ede` (2026-03-26 04:26 +0900): added chat template support and llama.cpp sampling defaults for text generation.
- `d03e9acb78` (2026-04-07 06:59 +0900): CPU backend got full VLM pipeline, DSP infrastructure, OpenVINO/OneDNN optimization.
- `cc7267db18` (2026-04-25 08:17 +0900): added `TestLLMBenchmarkSuite` and `run-llm-benchmarks.sh`.
- `1b990cb8f9` (2026-04-29 10:40 +0900): accumulated decode optimizations; updated `GenerationPipeline`, `ModelIOConfig`, and `TestQwen35Pipeline`.
- `36d282016d` (2026-04-29 17:01 +0900): CPU LLM inference fixes: FlatBuffers >2GB, device targeting, FP32 dtype, auth.
- `9bb2680e2b` (2026-04-30 15:21 +0900): MKL SDPA prefill heap overrun and invalid batched GEMM strides; mapped by reflog to the last-good run window.
- `517d04fe62` (2026-05-01 12:38 +0900): `BROKEN: snapshot of working tree for investigation`; captures the later failing working tree.
- `518e704aee` (2026-05-01 12:40 +0900): reverted the broken snapshot.

## Last substantiated correct CPU DSP Qwen run

Last strong evidence is `/tmp/qwen-cpu-flash-fix.log`, finished `2026-05-01T06:16:10+09:00`. Reflog shows HEAD was still `9bb2680e2b` in that window on `ag_new_release_updates_2`.

Evidence from the log:

- `TestLLMBenchmarkSuite#testOptimalBaseline` via CPU backend (`nd4j-native`) for Qwen3.5 0.8B Q4_K_M.
- DynamicShapePlan native executor compiled the Qwen graph: 1761 slots, 487 external inputs, 37 outputs.
- Native executor mode resolved `AUTO`, `tritonAvailable=true`, `fallbackToAuto=true`.
- Qwen generated 20 tokens under `OPTIMAL` and `GenerationQualityValidator` reported `Quality check PASSED` with diversity 1.00, repetition 0.00, coherence 1.00.
- Maven result: Tests run 1, Failures 0, Errors 0, Build Success.

## Later failure evidence

`/tmp/qwen-head-test.log`, finished `2026-05-01T12:34:15+09:00`, failed before the `BROKEN` snapshot commit was created. That failing working tree was committed as `517d04fe62`.

Failure: `TestQwen35Pipeline#testQwen35ReferencePrompts` failed in DSP native execution at `rms_norm_linear`, status 50, with `MMUL cuda gemv case failed` / invalid configuration argument. The run ended with Tests run 1, Errors 1, Build Failure.

There were later slot-by-slot/diagnostic Qwen logs that Maven-passed, but several had `Quality check FAILED`, and those are not stronger evidence than the optimized CPU DSP `OPTIMAL` pass above.

## Re-run command for this milestone

```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && ./run-llm-benchmarks.sh --backend cpu --test baseline --models qwen --tokens 20 2>&1 | tee /tmp/qwen-cpu-dsp-qwen-baseline.log
```

CPU DSP build configuration is `build-config-cpu-native-dsp-triton`: run the CPU native build with `-Dlibnd4j.triton=ON`, `-Pcpu`, libnd4j and nd4j-native modules, and tee output.
