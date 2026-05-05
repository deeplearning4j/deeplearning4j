# Kompile Memory Index

## ACTIVE: TRITON Merged Capture Gap Fix (May 5 2026)

- [TRITON merged capture gap fix](triton_merged_capture_gap_fix.md) — tl_mergedCaptureActive flag allows gap ops to execute during merged CUDA graph capture
- [VLM ContextBuffers capture bug](vlm_contextbuffers_capture_bug.md) — lazy ContextBuffers init on stream 0 during capture → error 906/901

## ACTIVE: DSP Accuracy — Root Causes Found, Rebuilding (May 2 2026)

- [Session status](dsp_accuracy_session_may2_status.md) — TWO ROOT CAUSES FOUND: CPU causal mask + CUDA staging sync. Fixes applied, builds in progress
- [Test results](test_results_may2_all_failed.md) — last run: CPU=token 314, CUDA=all-zero. Rebuilding with fixes
- [Reconciliation / next if fixes fail](dsp_accuracy_next_attempts_may2_reconcile.md) — items 1-4 resolved, items 6-10 remain if needed

### Root Cause Details
- [CPU: MKL SDPA causal mask](cpu_sdpa_causal_mask_root_cause.md) — input[8] never read by PLATFORM_IMPL, all prefill attention non-causal
- [CUDA: frozen fast-path staging](cuda_frozen_fastpath_staging_root_cause.md) — graph replay reads stale capture-time data, produces zeros

## Prior Fix History (all still in code, all correct)

- [Fix plan (18 fixes)](dsp_accuracy_fix_plan.md) — 7 shared, 4 CPU, 6 CUDA, 2 not-a-bug confirmations
- [Positive fixes — do NOT revert](dsp_accuracy_positive_fixes.md) — silu aliasing, MKL heap overrun, FP16 accumulators
- [causal_conv1d kernel flip](causal_conv1d_kernel_flip_bug.md) — K-1-kk → kk for PyTorch cross-correlation

## Detailed File Analysis (reference)

- [llm_ops.cpp](dsp_regression_llm_ops_changes.md) — silu, swish_mul, rope, rms_norm_linear
- [DSP infrastructure](dsp_regression_infrastructure_changes.md) — phaseShapeInference, BFS, validation
- [Attention & SDPA](dsp_regression_attention_sdpa_changes.md) — MKL prefill bias, FlashAttention
- [Segment execution](dsp_regression_segment_execution_changes.md) — prezero, backend chain, CPU replay
- [GraphOptimizer & fusion](dsp_regression_graph_optimizer_changes.md) — DCE, NormFusion, AttentionFusion
- [autoregressive_decode.cu](dsp_regression_autoregressive_decode_changes.md) — markExternalInputVariable, printf gating
- [OpTraitTable & Java](dsp_regression_optrait_and_java_changes.md) — DATADEP traits, SameDiff, GGML
- [Helper impls](dsp_regression_helper_impl_changes.md) — rmsNorm, fusedRoPE, FlashAttention
- [Commit timeline](dsp_regression_commit_timeline.md) — Apr 29 - May 2 risk classification

## Prior Investigation Trail (historical, resolved)

- [VLM zeros investigation](vlm_decode_zeros_regression_investigation.md) — root cause found
- [CPU DSP Qwen history](cpu_dsp_qwen_history.md) — prior attempts before root cause found

## Build Config

- [CUDA build](build_config_cuda_native.md) — mvn -Pcuda with Triton
- [CPU build](build_config_cpu_native.md) — mvn -Pcpu standard
- [CPU + Triton](build_config_cpu_native_dsp_triton.md) — CPU with DSP Triton

## Feedback & Rules

- [test-targets-immutable](test_targets_immutable.md) — CPU=Qwen3.5, CUDA=SmolDocling VLM. NEVER change these.
- [dsp-accuracy-may2-post-rootcause-extra-leads](dsp_accuracy_may2_post_rootcause_extra_leads.md) — [project] May 2 post-root-cause memory search addendum: what is obsolete, what remains worth trying if CPU/CUDA fixes fail veri...
- [cpu-causal-mask-fix-verified](cpu_causal_mask_fix_verified.md) — [project] CPU causal mask fix CONFIRMED working — output changed from garbage to coherent echo. GDN layers not contributing, mo...
- [gdn-l2norm-eps-bug](gdn_l2norm_eps_bug.md) — [project] CRITICAL: GDN L2-norm eps=1e-12 vs reference eps=1e-6 — causes near-zero vector amplification corrupting GDN state
- [onednn-softplus-alpha-zero-bug](onednn_softplus_alpha_zero_bug.md) — [project] CRITICAL ROOT CAUSE: OneDNN softplus alpha=0 causes division by zero → all inf output, kills ALL GDN gate decay
- [cuda-staging-fix-not-sufficient](cuda_staging_fix_not_sufficient.md) — [project] CUDA frozen fast-path staging fix applied but NOT sufficient — still all-zero tokens [216, 49229, 0, 0, ...]
- [dsp-accuracy-cpu-fix-status](dsp_accuracy_cpu_fix_status.md) — [project] CPU Qwen3.5 accuracy fix status — softplus alpha=0 root cause found, build in progress May 2 2026
- [dsp-accuracy-cuda-fix-status](dsp_accuracy_cuda_fix_status.md) — [project] CUDA VLM accuracy fix status — staging fix not sufficient, still all-zero tokens, needs further investigation May 2 2026
- [cpu-softplus-root-cause-found](cpu_softplus_root_cause_found.md) — [project] CPU ROOT CAUSE: OneDNN softplus alpha=0 → all inf, kills GDN gate decay. Fix: alpha=1.0. Build in progress.
- [cpu-softplus-fix-partial-success](cpu_softplus_fix_partial_success.md) — [project] CPU softplus alpha=1 fix: tokens now real words but wrong ('ofof.' not France) — partial success May 2 2026
- [cpu-sdpa-causal-mask-verified-correct](cpu_sdpa_causal_mask_verified_correct.md) — [project] CPU SDPA causal mask flow verified correct — NOT the cause of 'ofof.' output
- [cpu-iargs-stop-token-parsing-fix](cpu_iargs_stop_token_parsing_fix.md) — [project] CPU iArgs parsing bug: stopTokenCount=62 instead of 2 — kvOutputIndices count mismatch between Java (0 for GGUF) and ...
- [cuda-frozen-fastpath-composite-replay-fix](cuda_frozen_fastpath_composite_replay_fix.md) — [project] CUDA VLM zeros fix: frozen fast-path in executeSlot skips re-execution during composite replay gap ops
- [cpu-qwen-iargs-parsing-fix](cpu_qwen_iargs_parsing_fix.md) — [project] CPU Qwen fix: iArgs parsing skips kvOutputIndices when hasInGraphKvCache=true (GGUF models)
- [dsp-accuracy-may2-latest-followup](dsp_accuracy_may2_latest_followup.md) — [project] May 2 latest follow-up: CPU iArgs fixed early stop but quality still wrong; causal_conv1d flip likely backwards; CUDA...
- [causal-conv1d-weight-flip-fix-may2](causal_conv1d_weight_flip_fix_may2.md) — [project] Fixed causal_conv1d weight indexing: weight[K-1-kk] for PyTorch left-padded conv semantics, affects all 18 GDN layers
- [cuda-zeros-still-broken-may2-v2](cuda_zeros_still_broken_may2_v2.md) — [project] CUDA SmolDocling still zeros after frozen fast-path fix — graph replay itself produces stale outputs, not just execut...
- [cuda-stale-gap-slot-cache-fix](cuda_stale_gap_slot_cache_fix.md) — [project] CUDA VLM zeros root cause: markExternalInputVariable invalidates captures but leaves stale activeGapSlotsCached_, com...
- [cpu-qwen-france-success](cpu_qwen_france_success.md) — [project] CPU Qwen3.5 outputs 'The capital of France is Paris' — causal_conv1d weight flip fix confirmed
- [cuda-softmax-inplace-corruption](cuda_softmax_inplace_corruption.md) — [project] Root cause of CUDA VLM all-zero tokens: fusedCausalMaskSoftmaxKernel in-place corruption
- [vlm-eos-burndown-may3](vlm_eos_burndown_may3.md) — [project] VLM EOS-on-step-2 burndown: eliminated hypotheses, remaining candidates, current state May 3 2026
- [vlm-token-parsing-fix-may3](vlm_token_parsing_fix_may3.md) — [project] Fixed VLM native token output parsing: tid==0 zero-padding hack removed, buildStopTokenIds eosId<100 guard removed
- [cuda-qwen-gdn-state-fix-may3](cuda_qwen_gdn_state_fix_may3.md) — [project] CUDA Qwen root cause: missing GDN/conv state feedback in autoregressive_decode.cu — fix applied, build in progress
- [CUDA Qwen correctness achieved May 3 2026](cuda_qwen_correctness_achieved_may_3_2026.md) — [project] CUDA Qwen outputs France (token 271) — CUTLASS stride fix + FP16 autocast removal confirmed working
- [VLM EOS root cause: mmulMxV mixed-type GEMV bug](vlm_eos_root_cause_mmulmxv_mixed_type_gemv_bug.md) — [project] mmulMxV has no mixed-type cast — HALF weight × FLOAT32 activation dispatches to usualGemv which interprets HALF as FL...
- [ConversionOptions forInference FP32 is NOT a workaround](conversionoptions_forinference_fp32_is_not_a_workaround.md) — [project] forInference() using FLOAT32 for GGML/Qwen is a legitimate precision choice, not a workaround for the FP16 autocast bug
- [VLM decode regression root cause May 3](vlm_decode_regression_root_cause_may_3.md) — [project] onnx_multi_head_attention.cpp workspace buffer removal + syncToDevice removal caused VLM EOS on step 1
- [Qwen autoregressive decode NOT working May 3](qwen_autoregressive_decode_not_working_may_3.md) — [project] Qwen prefill works (token 271) but full autoregressive decode produces garbage — same MHA regression as VLM
- [vlm-decode-native-path-restored](vlm_decode_native_path_restored.md) — [project] VLM decode routed through generateNative (C++ loop) after fixing causal mask, segment splitting, GEMV bugs
- [rms-norm-linear-fusion-multi-consumer-bug](rms_norm_linear_fusion_multi_consumer_bug.md) — [project] FuseRMSNormLinear removes rms_norm var when multiple matmuls consume it (Qwen Q/K/V shared norm)
- [slot-482-crash-root-cause-analysis](slot_482_crash_root_cause_analysis.md) — [project] VLM slot 482 (reshape_no_copy) crash: in-place fusion + multi-consumer buffer corruption in decode DSP plan
- [vlm-decode-status-may4](vlm_decode_status_may4.md) — [project] Current VLM decode status May 4: slot 482 crash with multi-consumer fix pending rebuild
- [VLM CUDA graph node count analysis](vlm_cuda_graph_node_count_analysis.md) — [project] Root cause of 60 tok/s: 2742 CUDA graph nodes cause ~5.5-11ms driver scheduling overhead per replay
- [Qwen CUDA decode garbage status](qwen_cuda_decode_garbage_status.md) — [project] Qwen first token correct (France/271) but second token produces garbage (76828) with GraphOptimizer enabled
- [Current goals and status May 4](current_goals_and_status_may_4.md) — [project] Two goals: VLM 100+ tok/s with mythic heroes output AND Qwen outputting France on CUDA
- [CUDA graph capture and replay code paths](cuda_graph_capture_and_replay_code_paths.md) — [reference] Key file locations for monolithic/composite CUDA graph capture, replay, and optimization
- [VLM op timing data OPTIMAL config](vlm_op_timing_data_optimal_config.md) — [project] Per-op timing from OPTIMAL.csv showing 250-token decode costs for performance analysis
- [VLM 552 concat ops root cause](vlm_552_concat_ops_root_cause.md) — [project] 552 concats are shape-assembly + KV cache from ONNX Attention import, ~18 per layer × 28 layers
- [VLM reshape_no_copy graph node root cause](vlm_reshape_no_copy_graph_node_root_cause.md) — [project] 604 reshape_no_copy: ~half produce graph nodes due to permute→reshape non-contiguous pattern, ARRAY_NEEDS_COPY flag
- [Triton perf benchmarks](triton_perf_benchmarks.md) — [project] VLM decode performance — current 60 tok/s monolithic CUDA graph, target 100+, key findings May 4
- [qwen-cuda-decode-root-cause-rms-norm-linear-stride](qwen_cuda_decode_root_cause_rms_norm_linear_stride.md) — [project] [project] ROOT CAUSE FOUND: CUDA Qwen second token garbage — rmsNormLinearFusedKernel assumes C-contiguous weight but...
- [Fixes applied May 4 batch](fixes_applied_may_4_batch.md) — [project] 6 fixes applied: optimizer orphan var, attention perf (nullify+syncToDevice), diagnostic, test workarounds
- [CUDA Qwen stream ordering fix](cuda_qwen_stream_ordering_fix_may4.md) — [project] GDN state feedback assign() uses wrong stream; fixed with explicit cudaMemcpyAsync on decode loop stream
