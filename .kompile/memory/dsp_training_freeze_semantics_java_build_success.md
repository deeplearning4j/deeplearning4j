---
name: dsp_training_freeze_semantics_java_build_success
description: nd4j-api build passed after TrainingSession replay semantics edit
type: project
---

**Build:** /home/agibsonccc/dev-apps/mvn/bin/mvn install -DskipTests -pl nd4j/nd4j-backends/nd4j-api-parent/nd4j-api 2>&1 | tee /tmp/dsp-training-freeze-semantics-java-build.log\n**Result:** PASS.\n**Change:** TrainingSession now checks placeholder shape changes before DSP execution, resets the current plan for new batch shapes, and freezes after a successful DSP training execution instead of forcing no-freeze slot-by-slot behavior.\n**Next:** Rerun DspTrainingE2ETest#testDspTrainingParityWithNonDsp from platform-tests.
