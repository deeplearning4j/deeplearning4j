---
name: dsp_training_freeze_semantics_focused_pass
description: Focused DSP training parity passes with frozen replay semantics enabled
type: project
---

**Test:** DspTrainingE2ETest#testDspTrainingParityWithNonDsp\n**Command:** cd platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dtest=DspTrainingE2ETest#testDspTrainingParityWithNonDsp -Dbackend.artifactId=nd4j-cuda-12.9 2>&1 | tee /tmp/dsp-training-parity-freeze-semantics.log\n**Result:** PASS, 3/3.\n**Lifecycle evidence:** Each updater case logs first DSP execution with `frozen=false`, then `DSP training: freezing shapes for batch shape to enable replay`, `FROZEN_TRANSITION`, and subsequent `DSP_EXEC_PRE ... frozen=true`.\n**Implication:** Stable-shape training now uses native frozen/replay semantics instead of the old no-freeze slot-by-slot path, while mutable trainable variables are marked and staged.
