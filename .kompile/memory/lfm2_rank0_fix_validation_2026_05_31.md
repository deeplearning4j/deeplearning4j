---
name: lfm2_rank0_fix_validation_2026_05_31
description: Validation results for LFM2 SameDiff sampling rank-0 fix
type: project
---

Validation on 2026-05-31: CUDA native rebuild succeeded with `/home/agibsonccc/dev-apps/mvn/bin/mvn -Pcuda -Dlibnd4j.triton=ON -Dlibnd4j.chip=cuda -Dlibnd4j.buildthreads=12 -Dlibnd4j.log=libnd4j-build.log -pl libnd4j,:nd4j-cuda-12.9 clean install -DskipTests` in 24:50. CUDA platform test `TestSamplingPenaltiesRank3#testRepetitionPenaltyScalarInputIds` passed (1 test), and full `TestSamplingPenaltiesRank3` passed (13 tests). Kompile `kompile-app/kompile-model-staging` rebuilt successfully with `-Dkompile.cuda=true`. End-to-end LFM2 staging repro on fixed jar loaded model with JCublasBackend and both CUDA devices visible, then `/v1/chat/completions` returned HTTP 200; old `autoregressive_decode` rank-0 failure did not recur. Output quality remains poor/invalid JSON (`{"response"="yes", "ok":true)`), which is separate from the fixed crash.
