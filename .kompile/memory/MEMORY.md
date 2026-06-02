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
- [DSP native-only all-gap CUDA graph capture fix](dsp_native_only_all_gap_cuda_graph_capture_fix.md) — [project] All-gap DSP segments in AUTO/TRITON must be captured as native-only monolithic CUDA graphs, not zero-node/slot-by-slo...
- [dsp-concurrent-plan-sharing-async-nosync-clean](dsp_concurrent_plan_sharing_async_nosync_clean.md) — [project] Full DspConcurrentPlanSharingTest passes with clean async DSP diagnostics after prealloc event capture fix
- [dsp-composite-replay-async-nosync-clean](dsp_composite_replay_async_nosync_clean.md) — [project] DspCompositeReplayTest passes cleanly with async-only DSP composite replay diagnostics
- [fix-dsp-buffer-alias-external-staging-views](fix_dsp_buffer_alias_external_staging_views.md) — [project] Fixed DSP CUDA graph coverage for transparent view aliases of staged external inputs
- [dsp-benchmark-baseline-250-may28-frozen-verify-failure](dsp_benchmark_baseline_250_may28_frozen_verify_failure.md) — [project] Baseline 250-token DSP benchmark: lateSteady 63.78 tok/s, audit blocked by frozen DataBuffer VERIFY mutation
- [dsp-verify-frozen-buffer-build-success](dsp_verify_frozen_buffer_build_success.md) — [project] CUDA native build succeeded after VERIFY frozen-buffer diagnostic fix
- [dsp-merged-segment-replay-verify-fix-pass](dsp_merged_segment_replay_verify_fix_pass.md) — [project] TestDspMergedSegmentReplay#testVerifyModeNoMismatch passed after VERIFY diagnostic no-primary guard
- [dsp-validation-still-fails-dspvalidateoutputs-frozen-primary](dsp_validation_still_fails_dspvalidateoutputs_frozen_primary.md) — [project] TestDspValidation output staleness still fails in dspValidateOutputs after replay VERIFY fix
- [dsp-validateoutputs-build-success](dsp_validateoutputs_build_success.md) — [project] CUDA native build succeeded after dspValidateOutputs frozen-output duplicate fix
- [dsp-validation-validateoutputs-fix-pass](dsp_validation_validateoutputs_fix_pass.md) — [project] TestDspValidation output staleness and multi-step comparison pass after dspValidateOutputs duplicate fix
- [dsp-training-e2e-loss-zero-still-fails](dsp_training_e2e_loss_zero_still_fails.md) — [project] DspTrainingE2ETest still fails: DSP training loss collapses to zero for optimizer parity cases
- [dsp_training_mutable_external_inputs_java_build_success](dsp_training_mutable_external_inputs_java_build_success.md) — [project] Java build passed after adding training-only mutable DSP external input marking.
- [dsp_training_mutable_external_inputs_still_loss_zero](dsp_training_mutable_external_inputs_still_loss_zero.md) — [project] Training-only mutable external input marking did not fix DspTrainingE2ETest loss-zero parity failures.
- [dsp_training_diag_mutable_staging_active_but_loss_zero](dsp_training_diag_mutable_staging_active_but_loss_zero.md) — [project] DSP training diagnostics show mutable staging active, but loss-zero parity failure remains.
- [dsp_fqcn_cleanup_java_build_success](dsp_fqcn_cleanup_java_build_success.md) — [project] nd4j-api Java build passed after removing java.util FQCN declarations from DynamicShapePlanExecutor.
- [dsp_mean_sqerr_replay_friendly_cuda_build_success](dsp_mean_sqerr_replay_friendly_cuda_build_success.md) — [project] CUDA build passed after making mean_sqerr loss reductions device-side/replay-friendly.
- [dsp_mean_sqerr_replay_friendly_first_test_failed_nan](dsp_mean_sqerr_replay_friendly_first_test_failed_nan.md) — [project] First replay-friendly mean_sqerr test failed with Infinity/NaN in both reference and DSP paths.
- [dsp_mean_sqerr_count_fix_cuda_build_success](dsp_mean_sqerr_count_fix_cuda_build_success.md) — [project] CUDA build passed after fixing mean_sqerr replay-friendly count denominator.
- [dsp_mean_sqerr_count_fix_test_failed_buffer_overrun](dsp_mean_sqerr_count_fix_test_failed_buffer_overrun.md) — [project] Mean_sqerr count cast retest failed with scalar output DataBuffer canary corruption.
- [dsp_mean_sqerr_count_applyscalar_cuda_build_success](dsp_mean_sqerr_count_applyscalar_cuda_build_success.md) — [project] CUDA build passed after replacing count cast with applyScalar in mean_sqerr denominator helper.
- [dsp_mean_sqerr_count_applyscalar_test_failed_wrong_target_type](dsp_mean_sqerr_count_applyscalar_test_failed_wrong_target_ty.md) — [project] Focused DSP training test failed after mean_sqerr count applyScalar conversion
- [dsp_mean_sqerr_denominator_constant_build_failed](dsp_mean_sqerr_denominator_constant_build_failed.md) — [project] CUDA rebuild failed after mean_sqerr denominator repair
- [dsp_mean_sqerr_denominator_lvalue_cuda_build_success](dsp_mean_sqerr_denominator_lvalue_cuda_build_success.md) — [project] CUDA build succeeded after mean_sqerr denominator lvalue fix
- [dsp_training_parity_mean_sqerr_denominator_pass_but_slot_by_slot](dsp_training_parity_mean_sqerr_denominator_pass_but_slot_by_.md) — [project] Focused DSP training parity passed after mean_sqerr repair, but training still logs slot-by-slot/no-freeze
- [dsp_training_freeze_semantics_java_build_success](dsp_training_freeze_semantics_java_build_success.md) — [project] nd4j-api build passed after TrainingSession replay semantics edit
- [dsp_training_freeze_semantics_focused_pass](dsp_training_freeze_semantics_focused_pass.md) — [project] Focused DSP training parity passes with frozen replay semantics enabled
- [fix-vlm-training-pipeline-test-geometry](fix_vlm_training_pipeline_test_geometry.md) — [project] Aligned TestVlmTrainingPipeline fine-tune config geometry with its synthetic 224x224 encoder.
- [op-executioner-mean-sum-simple-investigation-2026-05-28](op_executioner_mean_sum_simple_investigation_2026_05_28.md) — [project] Investigated OpExecutionerTests.testMeanSumSimple expected [256] vs [16] in full platform-tests log
- [fix_native_multi_backend_workspace_spill_metrics](fix_native_multi_backend_workspace_spill_metrics.md) — [project] Root-caused and patched NativeMultiBackendWorkspace CPU spill metric failure in CUDA build
- [rng-shuffle-native-crash-cluster-2026-05-28](rng_shuffle_native_crash_cluster_2026_05_28.md) — [project] Root causes for CUDA RNG DeclarableOp 0xdeadbe* crashes and CUDA shuffle NDArray::specialBuffer null crashes
- [dsp_benchmark_current_250_20260528](dsp_benchmark_current_250_20260528.md) — [project] Current 250-token DSP benchmark result after DL4J regression fixes
- [dsp_benchmark_no_norm_reduction_250_20260528](dsp_benchmark_no_norm_reduction_250_20260528.md) — [project] 250-token no-normalization/no-reduction compile-all config was much slower than OPTIMAL
- [dsp_benchmark_bisect_graphcapture_allsettings_250_20260528](dsp_benchmark_bisect_graphcapture_allsettings_250_20260528.md) — [project] dsp_benchmark_bisect_graphcapture_allsettings_250_20260528
- [dsp_benchmark_where_static_output_first_250_20260528](dsp_benchmark_where_static_output_first_250_20260528.md) — [project] dsp_benchmark_where_static_output_first_250_20260528
- [dsp_benchmark_where_static_output_rejected_250_20260528](dsp_benchmark_where_static_output_rejected_250_20260528.md) — [project] dsp_benchmark_where_static_output_rejected_250_20260528
- [dsp_benchmark_reshape_copyoffset_view_rejected_250_20260528](dsp_benchmark_reshape_copyoffset_view_rejected_250_20260528.md) — [project] DSP optimization candidate benchmark result rejected
- [dsp_benchmark_selective_value_dep_unfreeze_rejected_250_20260528](dsp_benchmark_selective_value_dep_unfreeze_rejected_250_2026.md) — [project] DSP optimization candidate benchmark result rejected
- [dsp_cache_position_inplace_candidate_rejected_20260528](dsp_cache_position_inplace_candidate_rejected_20260528.md) — [project] Synthesized cache_position in-place KV candidate built but regressed VLM 250-token throughput
- [dsp_benchmark_autoregressive_mask_prune_rejected_20260528](dsp_benchmark_autoregressive_mask_prune_rejected_20260528.md) — [project] Rejected autoregressive_decode.cu post-sample mask unmask removal after 250-token DSP benchmark regression.
- [dsp_benchmark_onnx_mha_direct_output_rejected_20260528](dsp_benchmark_onnx_mha_direct_output_rejected_20260528.md) — [project] Benchmark 9 direct-output onnx_multi_head_attention decode candidate regressed and was reverted
- [dsp_benchmark_repeat_kv_bypass_rejected_20260528](dsp_benchmark_repeat_kv_bypass_rejected_20260528.md) — [project] Benchmark 10 repeat_kv bypass candidate rejected due CUDA illegal memory access in 250-token benchmark
- [triton_hopper_tablegen_dependency_fix_20260528](triton_hopper_tablegen_dependency_fix_20260528.md) — [project] Committed Triton NVHopperTransforms tablegen dependency build fix
- [lfm2-autoregressive-decode-rank0-sampling-history](lfm2_autoregressive_decode_rank0_sampling_history.md) — [project] Root cause for LFM2.5 sampling failure in Kompile staging: rank-0 token history view in native autoregressive_decode ...
- [lfm2_autoregressive_decode_rank0_sampling_fix](lfm2_autoregressive_decode_rank0_sampling_fix.md) — [project] Fix for SameDiff LFM2 sampling rank-0 failure
- [lfm2_rank0_fix_validation_2026_05_31](lfm2_rank0_fix_validation_2026_05_31.md) — [project] Validation results for LFM2 SameDiff sampling rank-0 fix
- [cuda_init_failover_available_devices_20260602](cuda_init_failover_available_devices_20260602.md) — [project] cuda_init_failover_available_devices_20260602
