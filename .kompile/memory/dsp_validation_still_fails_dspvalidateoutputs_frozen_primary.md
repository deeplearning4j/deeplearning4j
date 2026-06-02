---
name: dsp-validation-still-fails-dspvalidateoutputs-frozen-primary
description: TestDspValidation output staleness still fails in dspValidateOutputs after replay VERIFY fix
type: project
---

**Command:** cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dtest=TestDspValidation#testOutputStalenessDetection+testMultiStepDecodeComparison -Dbackend.artifactId=nd4j-cuda-12.9 2>&1 | tee /tmp/dsp-validation-frozen-fix.log
**Result:** FAIL. Tests run: 2, Failures: 0, Errors: 1. testMultiStepDecodeComparison passed; testOutputStalenessDetection failed.
**Error:** DataBuffer LIFECYCLE VIOLATION: allocatePrimary called on frozen DataBuffer from native Nd4jCuda.dspValidateOutputs via DspHandle.validateOutputs.
**Implication:** NativeDynamicShapePlan requested-output VERIFY value dump is fixed, but native validation code still materializes host primary on frozen device-only outputs. Next single change should inspect/fix dspValidateOutputs/DspVerifyUtils path directly.
