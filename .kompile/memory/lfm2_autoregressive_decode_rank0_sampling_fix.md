---
name: lfm2_autoregressive_decode_rank0_sampling_fix
description: Fix for SameDiff LFM2 sampling rank-0 failure
type: project
---

For the LFM2.5 SameDiff native sampling crash (`NDArray::sizeAt: bad size index requested: 1 for array with rank: 0` in `autoregressive_decode`), the fix is to preserve rank when slicing generated token history in CPU/CUDA autoregressive_decode (`operator()(range, true)`) and to make CPU/CUDA sampling_penalties accept scalar inputIds by treating rank-0 as seqLen=1 with zero strides. Added regression coverage in TestSamplingPenaltiesRank3 for scalar INT64 inputIds with rank-3 logits.
