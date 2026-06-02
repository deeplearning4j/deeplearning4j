---
name: dsp_mean_sqerr_replay_friendly_first_test_failed_nan
description: First replay-friendly mean_sqerr test failed with Infinity/NaN in both reference and DSP paths.
type: project
---

**Test:** cd platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dtest=DspTrainingE2ETest#testDspTrainingParityWithNonDsp -Dbackend.artifactId=nd4j-cuda-12.9 2>&1 | tee /tmp/dsp-training-parity-mean-sqerr-replay-friendly.log
**Result:** FAIL, 3/3 failures.
**Observed:** Ref and DSP final losses are NaN, and assertions report ref first loss Infinity / last NaN for SGD, Adam, Nesterovs.
**Interpretation:** This is not the prior replay-only zero-loss symptom. The C++ `mean_sqerr_loss` rewrite broke ordinary op execution, likely around the new device-side nonzero weight count or dividing a floating scalar by an INT64 count result.
**Next:** Fix the op math in `meanSqErr.cpp`, rebuild CUDA, and rerun the same targeted test before any benchmark.
