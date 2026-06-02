---
name: dsp_mean_sqerr_denominator_lvalue_cuda_build_success
description: CUDA build succeeded after mean_sqerr denominator lvalue fix
type: project
---

**Build:** /home/agibsonccc/dev-apps/mvn/bin/mvn -Pcuda -Dlibnd4j.triton=ON -Dlibnd4j.chip=cuda -Dlibnd4j.buildthreads=12 -Dlibnd4j.log=libnd4j-build.log -pl libnd4j,:nd4j-cuda-12.9 clean install -DskipTests 2>&1 | tee /tmp/dsp-mean-sqerr-denom-lvalue-cuda-build.log\n**Result:** PASS. libnd4j and nd4j-cuda-12.9 installed successfully.\n**Change being validated:** mean_sqerr reduction outputs now divide through device-side DivideNoNan; mode 3 denominator uses a scalar lvalue assignment instead of unsupported INT64-to-FLOAT scalar apply/cast conversion.\n**Next:** Run focused DspTrainingE2ETest#testDspTrainingParityWithNonDsp from platform-tests.
