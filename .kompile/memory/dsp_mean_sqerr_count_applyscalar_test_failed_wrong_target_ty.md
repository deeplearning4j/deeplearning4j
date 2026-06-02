---
name: dsp_mean_sqerr_count_applyscalar_test_failed_wrong_target_type
description: Focused DSP training test failed after mean_sqerr count applyScalar conversion
type: project
---

**Test:** DspTrainingE2ETest#testDspTrainingParityWithNonDsp\n**Command:** cd platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dtest=DspTrainingE2ETest#testDspTrainingParityWithNonDsp -Dbackend.artifactId=nd4j-cuda-12.9 2>&1 | tee /tmp/dsp-training-parity-mean-sqerr-count-applyscalar.log\n**Result:** FAIL, 0/3 passed.\n**Failure:** mean_sqerr_loss throws `NDArray::applyScalarArr method: wrong type of target array`; INT64 count source cannot applyScalar into FLOAT target.\n**Implication:** The replay-friendly denominator conversion must avoid INT64 -> FLOAT via applyScalar/cast in this path, or special-case constant scalar weights while keeping mutable reduction outputs on device.\n**Status:** Current single optimization is not valid yet; do not benchmark or move to another change until fixed and retested.
