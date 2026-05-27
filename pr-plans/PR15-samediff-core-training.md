# PR15: SameDiff Core & Training

**Estimated files:** ~114
**Merge layer:** 4
**Complexity:** High
**Reviewers:** SameDiff team

## Description

SameDiff core framework: the main SameDiff and SDVariable classes,
control flow, training configuration, PEFT/LoRA support, RL alignment
trainers (PPO, DPO, GRPO, etc.), serialization, session management,
transfer learning, and training infrastructure.

## File Categories

### Core SameDiff (~8)
- `SameDiff.java`
- `SDVariable.java`
- `ControlFlow.java`
- `TrainingConfig.java`
- `TransferLearning.java`
- `TransferLearningHelper.java`
- `DifferentialFunction.java`
- `DistillationTrainer.java`
- `RLAlignmentTrainer.java`

### Config classes (~45)
PEFT configs:
- `AdaLoraConfig`, `AdapterConfig`, `DoraConfig`, `DyLoraConfig`
- `IA3Config`, `LoftQConfig`, `LohaConfig`, `LokrConfig`
- `LoraConfig`, `PeftConfig`, `PeftType`, `QLoraConfig`
- `PrefixTuningConfig`, `PromptTuningConfig`, `TaskType`

RL alignment configs:
- `DAPOConfig`, `DPOConfig`, `DrGRPOConfig`, `GRPOConfig`
- `GSPOConfig`, `KTOConfig`, `ORPOConfig`, `PPOConfig`
- `RLAlignmentConfig`, `RLPipelineConfig`, `RewardModelConfig`
- `SimPOConfig`

Training configs:
- `ContinuedPretrainingConfig`, `DistillationConfig`, `FineTuneConfiguration`
- `FP8TrainingConfig`, `GradientCheckpointConfig`, `KernelConfiguration`
- `LossScaleConfig`, `SFTConfig`
- `TtsFineTuneConfig`, `TtsTrainingConfig`
- `SDValue`, `VariableGroup`

### PEFT (~6)
- `LoftQInitializer.java`
- `LoraAdapterCache.java`
- `LoraLayer.java`
- `PeftModel.java`
- `PeftModelFactory.java`
- `package-info.java`

### RL trainers (~17)
- `DAPOTrainer`, `DPOTrainer`, `DrGRPOTrainer`, `GRPOTrainer`
- `GSPOTrainer`, `KTOTrainer`, `ORPOTrainer`, `PPOTrainer`
- `RewardModelTrainer`, `SimPOTrainer`, `VlmGRPOTrainer`
- `RewardFunction`, `SameDiffRewardFunction`, `RuleBasedRewardFunction`
- `CompositeRewardFunction`
- `SamplingStrategy`, `TopKSamplingStrategy`

### Training infrastructure (~10)
- `CheckpointManager.java`
- `CheckpointOffloadManager.java`
- `ContinuedPretrainingWorkflow.java`
- `FP8ScaleManager.java`
- `GradientAccumulator.java`
- `LossScaler.java`
- `PreferencePair.java`
- `RLAlignmentPipeline.java`
- `SFTTrainingPipeline.java`
- `TrainingResult.java`

### Session management (~11)
- `AbstractSession.java`
- `InferenceSession.java`
- `SessionMemMgr.java`
- `TrainingSession.java`
- `ArrayCacheMemoryMgr.java`
- `CleanupDiagnostics.java`
- `DependencyMap.java`
- `MultiBackendWorkspaceSessionMemMgr.java`
- `MultiGpuWorkspaceSessionMemMgr.java`
- `NoOpMemoryMgr.java`
- `WorkspaceSessionMemMgr.java`

### Serialization (~4)
- `ModelLoadingContext.java`
- `ModelSizeInfo.java`
- `SameDiffSerializer.java`
- `SDZSerializer.java`

Note: `FlatBuffersMapper.java` is assigned to **PR03** (FlatBuffers schema/serde).

### Array holders (~2)
- `SingleThreadArrayHolder.java`
- `ThreadSafeArrayHolder.java`

### Diagnostics (1)
- `DspDiagnostics.java` (Java-side DSP diagnostics)

### ADRs (3 — only those actually changed in the diff)
- `ADRs/0048 - Improved SameDiff Execution Framework.md` — DAG-based cached execution replacing initSubgraph interpreter
- `ADRs/0057 - Mixed Precision Training.md` — FP16/BF16 mixed-precision training with loss scaling
- `ADRs/0077 - PEFT and Knowledge Distillation Extensions.md` — LoRA variants, distillation API, DistillationTrainer

## Review Focus

- SameDiff.java is the central API — changes affect everything
- InferenceSession — DSP integration point
- RL trainers — new feature, needs correctness review
- PEFT/LoRA — weight modification logic
