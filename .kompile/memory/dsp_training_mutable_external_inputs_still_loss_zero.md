---
name: dsp_training_mutable_external_inputs_still_loss_zero
description: Training-only mutable external input marking did not fix DspTrainingE2ETest loss-zero parity failures.
type: project
---

**Test:** DspTrainingE2ETest\n**Command:** `cd platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dtest=DspTrainingE2ETest -Dbackend.artifactId=nd4j-cuda-12.9 2>&1 | tee /tmp/dsp-training-e2e-mutable-vars.log`\n**Result:** BUILD FAILURE, 15 tests run, 3 failures, 0 errors.\n\n**Failures:** `testDspTrainingParityWithNonDsp` for SGD, Adam, Nesterovs. DSP loss remains exactly `0.0`; references remain nonzero.\n\n**Implication:** Marking paramsToTrain as native mutable external inputs is necessary for replay semantics but not sufficient, or the mark did not fire for the actual external inputs. Next check the tee log for mutable marking and inspect loss/slot write semantics.
