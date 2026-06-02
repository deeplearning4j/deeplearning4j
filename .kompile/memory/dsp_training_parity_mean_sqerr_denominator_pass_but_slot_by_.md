---
name: dsp_training_parity_mean_sqerr_denominator_pass_but_slot_by_slot
description: Focused DSP training parity passed after mean_sqerr repair, but training still logs slot-by-slot/no-freeze
type: project
---

**Test:** DspTrainingE2ETest#testDspTrainingParityWithNonDsp\n**Command:** cd platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dtest=DspTrainingE2ETest#testDspTrainingParityWithNonDsp -Dbackend.artifactId=nd4j-cuda-12.9 2>&1 | tee /tmp/dsp-training-parity-mean-sqerr-denom-lvalue.log\n**Result:** PASS, 3/3.\n**Observed:** Final loss no longer collapses to 0.0. The log still reports `DSP training: keeping slot-by-slot execution (no shape freeze for training)`.\n**Implication:** mean_sqerr replay-friendliness fixed the immediate loss-zero/runtime blocker, but training non-slot-by-slot/replay semantics still need inspection before full regression and benchmark.
