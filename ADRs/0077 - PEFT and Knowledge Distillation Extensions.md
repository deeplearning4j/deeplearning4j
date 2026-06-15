# ADR 0077 - PEFT and Knowledge Distillation Extensions

## Status

Accepted

Proposed by: Adam Gibson

Discussed with: Development Team

## Context

The codebase has a mature PEFT framework supporting LoRA, DoRA, LoHa, LoKr, rsLoRA, and Multi-LoRA with fused C++ ops, along with some distillation building blocks (EMA Update, CenterAndSharpen, ContrastiveLoss). However, several important PEFT variants lack runtime implementations, and a dedicated knowledge distillation API is missing.

**Gaps identified:**
- PiSSA (SVD-based LoRA initialization) - NeurIPS 2024 spotlight, ~5% improvement over random init
- LoRA+ (differential learning rates) - B matrix gets higher LR for faster convergence
- BitFit (bias-only fine-tuning) - simplest PEFT method, declared but not implemented
- VeRA (shared random matrices with per-layer scaling) - extremely parameter efficient
- DyLoRA (dynamic rank training) - train once, deploy at any rank
- No KL divergence distillation loss, feature distillation, or attention distillation ops
- No DistillationTrainer API for orchestrating teacher-student training

## Decision

### Phase 1: LoRA Enhancements (Java-only)

1. **PiSSA/OLoRA Initialization**: Added to `LoraConfig.initLoraWeights` options ("pissa", "olora"). PiSSA uses SVD to initialize A,B from top-r singular values of the weight matrix. OLoRA uses QR decomposition. Both modify the base weight to W_residual = W - B@A.

2. **LoRA+ Differential Learning Rates**: Added `loraLrRatioB` field to `LoraConfig` and `learningRateMultipliers` map to `TrainingConfig`. B matrix variables are automatically tagged with the multiplier during injection.

3. **BitFit**: Implemented as `applyBitFit()` in `PeftModel`. Iterates variables, keeps only bias-named ones as VARIABLE, freezes everything else.

4. **VeRA**: New `VeraConfig` class. `applyVera()` creates ONE shared frozen random A and B matrix (from seed), with per-layer learned d and b scaling vectors. Delta: `lambda * diag(d) @ B_shared @ diag(b) @ A_shared`.

5. **DyLoRA**: New `DyLoraConfig` extending `LoraConfig` with `minRank` field. Uses same injection as LoRA; dynamic rank sampling happens during training.

### Phase 2: Knowledge Distillation Loss Ops (C++ + Java)

Three new C++ ops following the contrastive_loss pattern (header + CPU impl + CUDA impl + op registration):

1. **distillation_kl_loss**: `L = alpha * T^2 * KL(softmax(s/T) || softmax(t/T)) + (1-alpha) * CE(s, labels)`
2. **feature_distillation_loss**: `L = MSE(projection(student_hidden), teacher_hidden)` with optional projection
3. **attention_distillation_loss**: `L = MSE(student_attn, teacher_attn)` with head count alignment

Each has forward and backward (gradient) variants, CPU and CUDA implementations.

### Phase 3: Distillation Trainer API (Java-only)

1. **DistillationConfig**: Supports LOGIT_KD, FEATURE_KD, ATTENTION_KD, COMBINED types with temperature, alpha, layer mappings, loss weights, and temperature annealing.

2. **DistillationTrainer**: Orchestrates teacher forward (no grad) -> student forward -> combined loss -> backward. Supports self-distillation with EMA teacher refresh.

## Consequences

- Trainable parameter counts for VeRA are dramatically lower than LoRA (d + r per layer vs r*(d+k))
- PiSSA provides better initialization than random for LoRA, with minimal overhead (one-time SVD)
- LoRA+ is a drop-in improvement requiring only a config change
- The 3 distillation loss ops enable standard KD pipelines without custom code
- DistillationTrainer provides a high-level API matching the Hugging Face experience
- C++ distillation ops need a native build to test; Java-only PEFT changes can be tested immediately

## Files Added/Modified

### New Files
- `VeraConfig.java` - VeRA configuration
- `DyLoraConfig.java` - DyLoRA configuration
- `DistillationConfig.java` - Distillation training configuration
- `DistillationTrainer.java` - Distillation training orchestrator
- `DistillationKLLoss.java` / `DistillationKLLossBp.java` - Java wrappers
- `FeatureDistillationLoss.java` / `FeatureDistillationLossBp.java` - Java wrappers
- `AttentionDistillationLoss.java` / `AttentionDistillationLossBp.java` - Java wrappers
- C++ headers, CPU impls, CUDA impls for 3 distillation loss ops (9 files)
- 3 C++ op registration files
- `TestDistillationOps.java` - Comprehensive distillation tests

### Modified Files
- `LoraConfig.java` - Added `loraLrRatioB` field
- `LoraLayer.java` - Added PiSSA/OLoRA SVD initialization
- `PeftModel.java` - Added BitFit, VeRA, DyLoRA methods; PiSSA init wiring; LR multipliers
- `PeftConfig.java` - Added VeRA and DyLoRA to JsonSubTypes
- `TrainingConfig.java` - Added `learningRateMultipliers` field and builder methods
- `loss.h` - Added declarations for 3 new distillation loss ops
- `TestPeftOpValidation.java` - Added PiSSA, LoRA+, BitFit, VeRA, DyLoRA tests
