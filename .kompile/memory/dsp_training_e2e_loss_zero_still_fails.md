---
name: dsp-training-e2e-loss-zero-still-fails
description: "DspTrainingE2ETest still fails: DSP training loss collapses to zero for optimizer parity cases"
type: project
---

**Command:** cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dtest=DspTrainingE2ETest -Dbackend.artifactId=nd4j-cuda-12.9 2>&1 | tee /tmp/dsp-training-e2e-after-frozen-fixes.log
**Result:** FAIL. Tests run: 15, Failures: 3, Errors: 0. Failed parameterized optimizer parity cases for SGD, Adam, Nesterovs.
**Observed:** Softmax DSP losses first=1.137739, last=0.0. Assertions report loss ratio huge because refs are nonzero (SGD=1.020806, Adam=0.814623, Nesterovs=0.201982) but DSP=0.0.
**Implication:** Remaining audit blocker is training-specific correctness, separate from frozen DataBuffer diagnostic reductions.
