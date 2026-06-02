---
name: dsp_mean_sqerr_count_fix_cuda_build_success
description: CUDA build passed after fixing mean_sqerr replay-friendly count denominator.
type: project
---

**Build:** /home/agibsonccc/dev-apps/mvn/bin/mvn -Pcuda -Dlibnd4j.triton=ON -Dlibnd4j.chip=cuda -Dlibnd4j.buildthreads=12 -Dlibnd4j.log=libnd4j-build.log -pl libnd4j,:nd4j-cuda-12.9 clean install -DskipTests 2>&1 | tee /tmp/dsp-mean-sqerr-count-fix-cuda-build.log
**Result:** BUILD SUCCESS.
**Change since failed test:** `reduceNonZeroWeightCount` now computes a scalar INT64 `CountNonZero`, casts to the loss floating dtype, and for scalar weights multiplies by the broadcast length. This avoids mixed float/INT64 divide and avoids using broadcast `Assign` to expand scalar weights.
**Next:** Rerun `DspTrainingE2ETest#testDspTrainingParityWithNonDsp` from platform-tests.
