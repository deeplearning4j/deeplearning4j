---
name: fix-vlm-training-pipeline-test-geometry
description: Aligned TestVlmTrainingPipeline fine-tune config geometry with its synthetic 224x224 encoder.
type: project
---

TestVlmTrainingPipeline used VlmFineTuneConfig factory defaults when creating pipelines for synthetic 224x224 encoder models. The VlmFineTuneConfig default imageResolution=384 with patchSize=14 fails validation, while the synthetic test model and VlmTrainingConfig defaults use 224/14. The fix wraps the factory configs with a helper that preserves trainability/LoRA fields and sets imageResolution=224, patchSize=14, maxImageTokens=256 for these tests. Targeted Maven test was not run because an existing platform-tests Maven/Surefire process was active.
