---
name: dsp-validateoutputs-build-success
description: CUDA native build succeeded after dspValidateOutputs frozen-output duplicate fix
type: project
---

**Command:** cd /home/agibsonccc/Documents/GitHub/deeplearning4j && /home/agibsonccc/dev-apps/mvn/bin/mvn -Pcuda -Dlibnd4j.triton=ON -Dlibnd4j.chip=cuda -Dlibnd4j.buildthreads=12 -Dlibnd4j.log=libnd4j-build.log -pl libnd4j,:nd4j-cuda-12.9 clean install -DskipTests 2>&1 | tee /tmp/dsp-validate-outputs-build.log
**Result:** BUILD SUCCESS. libnd4j SUCCESS [10:05], nd4j-cuda-12.9 SUCCESS [01:23], total 11:29.
**Change under test:** DspVerifyUtils.h dspValidateOutputs duplicates frozen device-only outputs before sum/norm reductions, matching dspDetectStaleOutputs behavior.
**Next validation:** rerun TestDspValidation#testOutputStalenessDetection+testMultiStepDecodeComparison from platform-tests with tee.
