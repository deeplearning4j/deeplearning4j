---
name: dsp_training_mutable_external_inputs_java_build_success
description: Java build passed after adding training-only mutable DSP external input marking.
type: project
---

**Build:** nd4j-api Java-only install\n**Command:** `/home/agibsonccc/dev-apps/mvn/bin/mvn install -DskipTests -pl nd4j/nd4j-backends/nd4j-api-parent/nd4j-api 2>&1 | tee /tmp/dsp-training-mutable-vars-java-build.log`\n**Result:** BUILD SUCCESS\n\n**Change under test:** InferenceSession exposes an empty mutable DSP external input hook by default; TrainingSession returns trainable params with gradients; DynamicShapePlanExecutor marks matching external input indices as native mutable variables for each shape-keyed plan handle.\n\n**Why:** Training weights are VARIABLE source types, not PLACEHOLDERs. Native replay only refreshes/stages mutable externals, so training AUTO/replay must explicitly mark paramsToTrain without forcing SLOT_BY_SLOT or degrading inference.
