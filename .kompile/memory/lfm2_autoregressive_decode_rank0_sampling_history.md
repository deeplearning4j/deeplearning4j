---
name: lfm2-autoregressive-decode-rank0-sampling-history
description: "Root cause for LFM2.5 sampling failure in Kompile staging: rank-0 token history view in native autoregressive_decode penalty path."
type: project
---

LFM2.5 Kompile staging failure `NDArray::sizeAt: bad size index requested: 1 for array with rank: 0` is most consistent with native `autoregressive_decode` sampling/repetition-penalty path, not KV/recurrent state shape. In `libnd4j/include/ops/declarable/helpers/cuda/autoregressive_decode.cu`, when `step > 0`, it creates `NDArray* tokensSoFar = (*generatedTokenIds)(range)` where `range={0, step}`. For `step=1`, NDArray subarray logic drops unit dimensions by default and returns a rank-0 scalar view. That view is passed to `tokenSampleWithPenaltiesCuda`, then `applyPenaltiesLauncher` in `sampling_penalties.cu` checks only `idsRank == 1` else reads `inputIds->sizeAt(1)`, causing the observed rank-0 sizeAt(1) exception. CPU path has the same pattern. Fix should preserve rank with `(*generatedTokenIds)(range, true)` and/or make sampling_penalties helpers handle/validate rank-0 token-history inputs. Greedy avoids this path, explaining why greedy generation completes.
