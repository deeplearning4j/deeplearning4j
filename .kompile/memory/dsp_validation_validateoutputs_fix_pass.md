---
name: dsp-validation-validateoutputs-fix-pass
description: TestDspValidation output staleness and multi-step comparison pass after dspValidateOutputs duplicate fix
type: project
---

**Command:** cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dtest=TestDspValidation#testOutputStalenessDetection+testMultiStepDecodeComparison -Dbackend.artifactId=nd4j-cuda-12.9 2>&1 | tee /tmp/dsp-validation-validateoutputs-fix.log
**Result:** PASS. Tests run: 2, Failures: 0, Errors: 0, BUILD SUCCESS, total 1:18.
**Fix validated:** DspVerifyUtils.h dspValidateOutputs now duplicates frozen device-only outputs before sum/norm reductions. The prior native Nd4jCuda.dspValidateOutputs allocatePrimary lifecycle violation is gone.
