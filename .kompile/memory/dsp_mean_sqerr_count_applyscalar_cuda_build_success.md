---
name: dsp_mean_sqerr_count_applyscalar_cuda_build_success
description: CUDA build passed after replacing count cast with applyScalar in mean_sqerr denominator helper.
type: project
---

**Build:** /home/agibsonccc/dev-apps/mvn/bin/mvn -Pcuda -Dlibnd4j.triton=ON -Dlibnd4j.chip=cuda -Dlibnd4j.buildthreads=12 -Dlibnd4j.log=libnd4j-build.log -pl libnd4j,:nd4j-cuda-12.9 clean install -DskipTests 2>&1 | tee /tmp/dsp-mean-sqerr-count-applyscalar-cuda-build.log
**Result:** BUILD SUCCESS.
**Change since buffer overrun:** Removed `NDArray::cast` from `reduceNonZeroWeightCount`; now uses `countLong.applyScalar(scalar::Multiply, scale, countResult)` to produce the floating denominator scalar.
**Next:** Rerun focused training parity test from platform-tests.
