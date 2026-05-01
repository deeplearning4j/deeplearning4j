# Kompile Memory Index

## CUDA DSP Benchmarks
- [dsp-perf-recovery-commits](dsp_perf_recovery_commits.md) — [reference] Full commit chain for 10→50 tok/s recovery: infrastructure, correctness, perf commits, failed/reverted

## CUDA DSP Optimization TODOs

## CPU DSP Benchmarks
- [cpu-dsp-perf-investigation-apr29](cpu_dsp_perf_investigation_apr29.md) — [project] OpenVINO IS working, segment merge bug found (BenchmarkConfig default), compute-bound at 0.12 tok/s on Ryzen 5950X AVX2
- [cpu-dsp-perf-baseline-apr27](cpu_dsp_perf_baseline_apr27.md) — [project] CPU baseline: 3 tok/s (SmolDocling), oneDNN+OpenVINO multi-backend chain, op timing profile, active work
- [cpu-dsp-cachedshapekey-cascade-fix-apr28](cpu_dsp_cachedshapekey_cascade_fix_apr28.md) — [project] cachedShapeKey cascade bug: premature write in computeSegmentShapeKey caused fallback backends to skip compile
- [cpu-dsp-testing-two-benchmarks](cpu_dsp_testing_two_benchmarks.md) — [project] CPU DSP has 2 benchmarks: TestLLMBenchmarkSuite (Qwen 0.8B, run FIRST) and run-benchmark.sh (SmolDocling VLM, run SEC...
- [cpu-dsp-concat-path-leak](cpu_dsp_concat_path_leak.md) — [project] GGUF models use concat-based decode (no KV cache outputs) — DSP plan recompilation every token causes growing exec ti...
- [cpu-has-full-dsp](cpu_has_full_dsp.md) — [feedback] CPU has full DSP including emulated DSP — ALL configs must work on CPU, NEVER dismiss DSP failures as platform-specific

## CUDA DSP Architecture & Fixes
- [reshape-view-bypass-regression](reshape_view_bypass_regression.md) — [project] REVERTED: reshape_no_copy ARRAY_COPY_OFFSET_INPUT_0 view bypass caused -29% regression (50→35 tok/s)
- [composite-replay-schedule-analysis](composite_replay_schedule_analysis.md) — [project] Detailed analysis of composite replay schedule structure — 305 units, 93 merged groups, 93 unmerged gaps
- [openvino-model-cache-oom](openvino_model_cache_oom.md) — [project] OpenVINO modelCache_ had no eviction — CompiledModels accumulated causing C++ OOM
- [gguf-kv-cache-failed-attempt-apr27](gguf_kv_cache_failed_attempt_apr27.md) — [project] Failed attempt to add KV cache to GGUF LLaMAArchitecture — detailed analysis of what broke
- [gguf-kv-cache-status-apr27-working](gguf_kv_cache_status_apr27_working.md) — [project] GGUF in-graph KV cache working end-to-end on Qwen 0.8B — 1.83 tok/s, quality issues remain
- [dsp-ssm-op-miscompilation](dsp_ssm_op_miscompilation.md) — [project] DSP silently skips GDN/SSM ops during Triton IR emission — trait fallback masks it
- [tier1-optimization-results](tier1_optimization_results.md) — [project] Tier 1 results: 44→52 tok/s (+17.5%), plan overhead halved
- [cuda_dsp_rms_norm_linear_fused](cuda_dsp_rms_norm_linear_fused.md) — [project] Fused rms_norm_linear kernel landed — 51.88 tok/s
- [skip-rms-norm-landed](skip_rms_norm_landed.md) — [project] Fused skip_rms_norm op landed — 60 add kernels eliminated, +2.5% throughput
- [skip_rms_norm Triton support](skip_rms_norm_triton_support.md) — [project] Triton emitter for skip_rms_norm — #1 perf blocker for CUDA graph replay
- [FuseGatedMLPPattern regression](fusegatedmlppattern_regression.md) — [project] FuseGatedMLPPattern disabled — absorbing matmuls regressed 51→48.2 tok/s

## CUDA DSP Performance State
- [vlm-decode-perf-state-apr28](vlm_decode_perf_state_apr28.md) — [project] VLM decode 52 tok/s, GPU compute is bottleneck
- [vlm-perf-revised-bottleneck](vlm_perf_revised_bottleneck.md) — [project] Bottleneck is 14.8ms GPU compute inside merged CUDA graph, NOT sync gaps
- [cuda-dsp-optimization-plan](cuda_dsp_optimization_plan.md) — [project] Full optimization plan — current 53 tok/s, target 100+
- [cuda-dsp-decode-bottleneck-profile](cuda_dsp_decode_bottleneck_profile.md) — [project] Per-step decode profiling: 14ms GPU (74%), 4ms CPU launch (22%)
- [cuda-dsp-perf-regression-apr27](cuda_dsp_perf_regression_apr27.md) — [project] CUDA GPU regression 86.7→53 tok/s, nsys-verified 22-island architecture
- [nsys-gpu-kernel-profile](nsys_gpu_kernel_profile.md) — [project] nsys GPU kernel profile — 22-island architecture verified
- [triton-islands-status](triton_islands_status.md) — [project] Triton island/gap status: skip_rms_norm emitter implemented, all gaps captured
- [new-optimization-opportunities-2026-04-28](new_optimization_opportunities_2026_04_28.md) — [project] nsys-verified targets: 22 island transitions, GPU-side argmax, D2D KV scatter
- [cuda-dsp-optimization-todos](cuda_dsp_optimization_todos.md) — [project] CUDA DSP decode TODO — 53 tok/s, real targets: island reduction, GPU argmax
- [vlm-decode-loop-optimization-todo](vlm_decode_loop_optimization_todo.md) — [project] Migrate VLM decode loop from Java putScalar/H2D to GPU kernels, 5 steps
- [vlm-already-uses-native-decode](vlm_already_uses_native_decode.md) — [project] VLM benchmark already uses native autoregressive_decode C++ op

## Feedback & Negative Results
- [stream-sync-reordering-negative](stream_sync_reordering_negative.md) — [feedback] Reordering GPU work around cudaStreamSynchronize does NOT help
- [gqa-import-optimization-result](gqa_import_optimization_result.md) — [project] GroupQueryAttention→OnnxMultiHeadAttention import: 0% perf impact, reverted
