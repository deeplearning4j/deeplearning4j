---
name: dsp_mean_sqerr_count_fix_test_failed_buffer_overrun
description: Mean_sqerr count cast retest failed with scalar output DataBuffer canary corruption.
type: project
---

**Test:** cd platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dtest=DspTrainingE2ETest#testDspTrainingParityWithNonDsp -Dbackend.artifactId=nd4j-cuda-12.9 2>&1 | tee /tmp/dsp-training-parity-mean-sqerr-count-fix.log
**Result:** FAIL, 3/3 errors.
**Native error:** `DataBuffer integrity check FAILED - BUFFER OVERRUN DETECTED`, scalar output buffer size 4 bytes, canary actual `0xdeadbeef00000000`, op `mean_sqerr_loss`, inputs `[8,1]`, scalar weight 1.0, reduction arg `[3]`.
**Interpretation:** The CountNonZero+`cast(..., FLOAT)` denominator path is unsafe for the scalar loss buffer. Replace cast with an arithmetic scalar transform into the FLOAT scalar denominator target, or otherwise avoid writing an INT64 scalar into a FLOAT-sized scalar buffer.
**Next:** Patch `reduceNonZeroWeightCount` to use `countLong.applyScalar(scalar::Multiply, scale, countResult)` instead of `cast`.
