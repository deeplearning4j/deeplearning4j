# PR17: ONNX Import

**Estimated files:** ~173
**Merge layer:** 6
**Complexity:** Medium
**Reviewers:** Import/model team

## Description

ONNX model import pipeline: op mapping implementations (137 ops),
ONNX IR layer, export hooks, Microsoft ONNX extensions support,
and the import framework API.

## File Categories

### Import API framework (~13)
- `samediff-import-api/ImportGraph.java`
- `samediff-import-api/IRProtobufExtensions.java`
- `samediff-import-api/OpMappingRegistry.java`
- `samediff-import-api/` attribute rules, DefaultImportRunner
- `samediff-import-api/pom.xml`

### ONNX op implementations (~137)
All under `samediff-import-onnx/src/main/kotlin/.../implementations/`:

Standard ops: AdaptiveAvgPool, ArgMax, AveragePool, BatchNormalization,
BiasAdd, Cast, Ceil, Clip, Compress, Concat, ConstantOfShape, Conv,
ConvTranspose, Cos, CumSum, DepthToSpace, DequantizeLinear, Div,
Dropout, Einsum, Elu, Equal, Erf, Exp, Expand, Flatten, Floor,
Gather, GatherElements, GatherND, Gelu, Gemm, GlobalAveragePool,
Greater, GRU, HardSigmoid, HardSwish, Identity, If, InstanceNorm,
LayerNormalization, LeakyRelu, Less, Log, LogSoftmax, Loop, LpNorm,
LSTM, MatMul, Max, MaxPool, Mean, Min, Mod, Mul, Neg, NonMaxSuppression,
NonZero, Not, OneHot, Or, Pad, Pow, PRelu, QLinearConv, QLinearMatMul,
QuantizeLinear, Range, Reciprocal, ReduceL1, ReduceL2, ReduceLogSum,
ReduceLogSumExp, ReduceMax, ReduceMean, ReduceMin, ReduceProd, ReduceSum,
ReduceSumSquare, Relu, Reshape, Resize, ReverseSequence, RoiAlign,
Round, Scatter, ScatterElements, ScatterND, Shape, Shrink, Sigmoid,
Sign, Sin, Sinh, Size, Slice, Softmax, Softplus, SpaceToDepth, Split,
Sqrt, Squeeze, Sub, Sum, Tanh, Tile, TopK, Transpose, Trilu,
Unique, Unsqueeze, Upsample, Where, Xor, ZipMap

Microsoft extensions: GroupQueryAttention, MultiHeadAttention,
RotaryEmbedding, WindowedAttention, GroupNormalization,
FusedConv, FusedGemm, FusedMatMul, MixtureOfExperts

### ONNX definitions (~2)
- `MicrosoftOnnxExtensions.kt`
- `OnnxOpDeclarations.kt`

### ONNX export (~7)
- `OnnxExportConfig.kt`
- `OnnxExporter.kt`
- `PostExportHook.kt`
- `SameDiffToOnnxOpMapper.kt`
- `TrainingStateExporter.kt`
- `hooks/BatchNormExportHook.kt`
- `hooks/ConvExportHook.kt`

### ONNX IR (~5)
- `OnnxIRAttr.kt`
- `OnnxIRDataType.kt`
- `OnnxIRGraph.kt`
- `OnnxIRGraphRunner.kt`
- `OnnxIRTensor.kt`

### Importer (~1)
- `OnnxFrameworkImporter.kt`

### Resources (~1)
- `onnx-mapping-ruleset.pbtxt`

### TensorFlow import (~7)
- `ReverseV2.kt`, `TensorflowOpDeclarations.kt`
- `TensorflowFrameworkImporter.kt`, `TensorflowIRGraphRunner.kt`
- `TensorflowIRTensor.kt`
- `tensorflow-mapping-ruleset.pbtxt`
- `pom.xml`

### ADRs (0)
No ADRs in this PR are changed in the diff. ADRs 0002–0009 exist on master unchanged.

## Review Focus

- Microsoft ONNX extensions — GroupQueryAttention, RotaryEmbedding
  must match expected ONNX Runtime behavior
- Cast/type handling — mixed-type ops must cast explicitly
- Attention mask handling — must be FLOAT, not LONG
- Softmax axis default (opset 13+ = -1)
