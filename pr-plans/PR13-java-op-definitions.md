# PR13: Java Op Definitions

**Estimated files:** ~309
**Merge layer:** 4
**Complexity:** Medium (volume, but mostly mechanical)
**Reviewers:** Java API team

## Description

Java op class definitions: all ops under `ops/impl/` (transforms, reductions,
scalars, shape ops, audio ops, updaters, LLM/attention ops), base op classes
(BaseOp, DynamicCustomOp, etc.), SameDiff namespace ops (SDBaseOps, SDNN, etc.),
activation implementations, and op codegen Kotlin source files.

## File Categories

### Op codegen Kotlin (~2+)
- `codegen/op-codegen/src/main/ops/org/nd4j/codegen/ops/SDBaseOps.kt`
- `codegen/op-codegen/src/main/ops/org/nd4j/codegen/ops/SDLoss.kt`
- `codegen/op-codegen/src/main/ops/org/nd4j/codegen/ops/NeuralNetwork.kt`
- (and other changed .kt files in op-codegen)

### Base op classes (~17)
- `BaseIndexAccumulation.java`
- `BaseOp.java`
- `BaseOpContext.java`
- `BaseReduceBoolOp.java`
- `BaseReduceFloatOp.java`
- `BaseReduceLongOp.java`
- `BaseReduceOp.java`
- `BaseReduceSameOp.java`
- `BaseScalarBoolOp.java`
- `BaseScalarOp.java`
- `BaseTransformAnyOp.java`
- `BaseTransformBoolOp.java`
- `BaseTransformFloatOp.java`
- `BaseTransformOp.java`
- `BaseTransformSameOp.java`
- `BaseTransformStrictOp.java`
- `CustomOp.java`
- `DynamicCustomOp.java`

### SameDiff namespace ops (~11)
- `SDAudio.java`
- `SDBaseOps.java`
- `SDCNN.java`
- `SDImage.java`
- `SDLinalg.java`
- `SDLoss.java`
- `SDNN.java`
- `SDRNN.java`
- `SDSignal.java`
- `SDTraining.java`
- `SDValidation.java`

### Activation implementations (~17)
- `ActivationCube.java` through `ActivationThresholdedReLU.java`

### Transform/custom ops (~128+)
Major ops in `ops/impl/transforms/custom/`:
- LLM/attention: `AutoregressiveDecode`, `FlashAttention`, `FusedRoPE`, `GatedDeltaRule`, `GroupedQueryAttention`, `KVCache*`, `MultiHeadAttention`, `RotaryPositionEncoding`, `ScaledDotProductAttention`
- Fusion: `FusedBatchGemm`, `LoRA*`, `RmsNorm*`, `SkipSimplifiedLayerNormalization`
- Quantization: `Dequantize`, `DynamicQuantize`
- Audio: `AudioNormalize`, `AudioResample`, `AWeighting`, `ChromaFeatures`, `GriffinLim`, `MelFilterbank`, `MelSpectrogram`, `MFCC`
- Standard: `Add`, `Concat`, `Gather`, `MatMul`, `Reshape`, `Softmax`, `Transpose`, etc.

### Shape ops (~20)
- `ops/impl/shape/` — Concat, Expand, Gather, Pad, Reshape, Shape, Squeeze, Tile, etc.

### Reduce/scalar/broadcast ops (~50+)
- `ops/impl/reduce/` — Mean, Sum, Max, Min, Norm, Variance, etc.
- `ops/impl/scalar/` — ScalarAdd, ScalarMul, etc.

### Updater ops (~9)
- Adam, AdaGrad, AdaDelta, RMSProp, SGD, etc.

### Other ops
- `ops/impl/controlflow/` — control flow ops
- `ops/impl/loss/` — loss ops
- `ops/impl/image/` — image ops
- `ops/impl/convolution/` — convolution ops
- `ops/custom/` — Invoke, Logdet, BarnesHutSymmetrize

## Review Focus

- New LLM/attention ops must have matching C++ implementations
- Op registration must include proper opName() and shape functions
- Activation impls must match expected mathematical behavior
