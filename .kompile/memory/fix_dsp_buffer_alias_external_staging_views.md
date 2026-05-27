---
name: fix-dsp-buffer-alias-external-staging-views
description: Fixed DSP CUDA graph coverage for transparent view aliases of staged external inputs
type: project
---

DspBufferAliasAccuracyTest#testBufferAliasVaryingInput failed deepAttentionQKV/CUDA_GRAPHS after materialized reshapes were correctly captured. The remaining failure was the interleaved host-only coverage guard flagging slot 0 reshape of placeholder x as non-transparent. During capture the op views the plan-owned staging NDArray, while slotOwnership_ had classified it as SLOT_OWNED because ownership reclassification used original external inputs. Fix: keep ownership-first checks but also treat view/identity outputs as transparent when their output DataBuffer aliases a negative-source external input used for capture. Applied to CUDA graph coverage and native-only capture pre-skip helpers. Validation: CUDA+Triton build passed and DspBufferAliasAccuracyTest#testBufferAliasVaryingInput passed 63/63.
