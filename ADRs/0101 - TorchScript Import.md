# ADR: TorchScript and Safetensors Import to SameDiff

## Status

Accepted

## Context

PyTorch is one of the most widely used deep learning frameworks, and many pre-trained models are distributed in PyTorch formats:

1. **TorchScript (.pt)** - Serialized PyTorch models containing both architecture and weights
2. **Safetensors (.safetensors)** - A newer, safer format for storing tensor weights (developed by Hugging Face)

Users of ND4J/SameDiff need the ability to:
- Import pre-trained PyTorch models for inference
- Convert PyTorch models to SameDiff for deployment in Java/JVM environments
- Leverage the extensive ecosystem of PyTorch models (especially vision models like ResNet, VGG, EfficientNet)

## Decision

Create an `nd4j-torchscript` module that provides:

1. **Format readers** for both TorchScript and Safetensors formats
2. **Architecture detection** to automatically identify common model architectures
3. **Op mapping** to convert PyTorch operations to SameDiff equivalents
4. **Weight transformation** to handle format differences between PyTorch and ND4J

## Architecture Overview

```
nd4j-torchscript/
├── src/main/java/org/nd4j/torchscript/
│   ├── TorchScriptModelImport.java      # Public API facade
│   ├── TorchScriptImportException.java  # Custom exception
│   │
│   ├── format/                          # File format handling
│   │   ├── TorchScriptFormat.java       # Format enum
│   │   ├── TorchScriptFormatDetector.java
│   │   ├── TorchScriptDataType.java     # PyTorch dtype mappings
│   │   ├── SafetensorsReader.java       # .safetensors parser
│   │   ├── SafetensorsHeader.java
│   │   ├── TorchScriptReader.java       # .pt parser
│   │   ├── TorchScriptMetadata.java
│   │   └── TorchScriptTensorInfo.java
│   │
│   ├── convert/                         # Conversion logic
│   │   ├── TorchScriptToSameDiffConverter.java
│   │   ├── ConversionOptions.java       # Builder for options
│   │   ├── OpMappingRegistry.java       # PyTorch -> SameDiff ops
│   │   └── TensorNameMapper.java        # Name normalization
│   │
│   ├── architecture/                    # Model architecture handlers
│   │   ├── VisionArchitecture.java      # Interface
│   │   ├── ArchitectureRegistry.java    # Detection & lookup
│   │   ├── ArchitectureConfig.java
│   │   ├── ResNetArchitecture.java
│   │   ├── VGGArchitecture.java
│   │   ├── EfficientNetArchitecture.java
│   │   └── GenericCNNArchitecture.java  # Fallback
│   │
│   └── ir/                              # Intermediate representation
│       ├── TorchScriptGraph.java
│       ├── TorchScriptNode.java
│       ├── TorchScriptValue.java
│       └── PickleParser.java
```

## File Format Details

### Safetensors Format

Safetensors is a simple, safe format for storing tensors:

```
┌─────────────────────────────────────────────────────────┐
│  8 bytes: header_size (little-endian uint64)            │
├─────────────────────────────────────────────────────────┤
│  header_size bytes: JSON header                         │
│  {                                                      │
│    "tensor_name": {                                     │
│      "dtype": "F32",                                    │
│      "shape": [out, in, kH, kW],                        │
│      "data_offsets": [start, end]                       │
│    },                                                   │
│    ...                                                  │
│  }                                                      │
├─────────────────────────────────────────────────────────┤
│  Raw tensor data (concatenated, aligned)                │
└─────────────────────────────────────────────────────────┘
```

Supported data types:
| Safetensors | ND4J DataType |
|-------------|---------------|
| F32         | FLOAT         |
| F16         | HALF          |
| BF16        | BFLOAT16      |
| F64         | DOUBLE        |
| I64         | LONG          |
| I32         | INT           |
| I16         | SHORT         |
| I8          | BYTE          |
| U8          | UBYTE         |
| BOOL        | BOOL          |

### TorchScript Format (.pt)

TorchScript files are ZIP archives containing:

```
model.pt (ZIP archive)
├── data.pkl           # Pickled Python objects (model structure)
├── data/
│   ├── 0              # Tensor data files
│   ├── 1
│   └── ...
├── constants.pkl      # Model constants
└── version            # Format version
```

The pickle format requires parsing Python's pickle protocol to extract:
- Model graph structure
- Operation types and parameters
- Tensor metadata and storage locations

## Weight Format Transformations

PyTorch and ND4J use different memory layouts for certain operations:

### Convolution Weights

```
PyTorch Conv2D: [out_channels, in_channels, kernel_h, kernel_w]
ND4J Conv2D:    [kernel_h, kernel_w, in_channels, out_channels]

Transformation: permute(2, 3, 1, 0)
```

### Linear/Dense Weights

```
PyTorch Linear: [out_features, in_features]
ND4J Linear:    [in_features, out_features]

Transformation: transpose()
```

## Operation Mapping

The `OpMappingRegistry` maps PyTorch operations to SameDiff equivalents:

| PyTorch Op | SameDiff Equivalent |
|------------|---------------------|
| `aten::conv2d` | `sd.cnn().conv2d()` |
| `aten::batch_norm` | `sd.nn().batchNorm()` |
| `aten::relu` | `sd.nn().relu()` |
| `aten::max_pool2d` | `sd.cnn().maxPooling2d()` |
| `aten::avg_pool2d` | `sd.cnn().avgPooling2d()` |
| `aten::adaptive_avg_pool2d` | Custom implementation |
| `aten::linear` | `sd.mmul()` + bias add |
| `aten::dropout` | `sd.nn().dropout()` |
| `aten::flatten` | `sd.reshape()` |
| `aten::add` | `sd.math().add()` |
| `aten::mul` | `sd.math().mul()` |
| `aten::sigmoid` | `sd.nn().sigmoid()` |
| `aten::softmax` | `sd.nn().softmax()` |
| `aten::relu6` | `sd.math().min(sd.nn().relu(), 6)` |
| `aten::hardswish` | `x * hardSigmoid(x)` |
| `aten::silu` | `x * sigmoid(x)` |

## Architecture Detection

The module can automatically detect common vision architectures by analyzing tensor naming patterns:

### ResNet Detection
```java
// Look for characteristic ResNet patterns
"layer1.0.conv1.weight"    // Residual blocks
"layer1.0.downsample.0"    // Skip connections
"fc.weight"                // Final classifier
```

### VGG Detection
```java
// Look for VGG's sequential feature extractor
"features.0.weight"        // Conv layers in features
"classifier.0.weight"      // FC layers in classifier
```

### EfficientNet Detection
```java
// Look for EfficientNet's MBConv blocks
"_conv_stem.weight"
"_blocks.0._expand_conv.weight"
"_blocks.0._depthwise_conv.weight"
"_blocks.0._se_reduce.weight"  // Squeeze-excitation
```

## Usage Examples

### Basic Import

```java
// Import a safetensors file
SameDiff sd = TorchScriptModelImport.importModel("model.safetensors");

// Run inference
INDArray input = Nd4j.rand(1, 3, 224, 224);
INDArray output = sd.output(input, "output");
```

### With Conversion Options

```java
ConversionOptions options = ConversionOptions.builder()
    .targetDataType(DataType.FLOAT)
    .forTraining(false)
    .dataFormat(ConversionOptions.DataFormat.NCHW)
    .architectureOverride("resnet50")
    .build();

SameDiff sd = TorchScriptModelImport.importModel("model.pt", options);
```

### Inspect Model Before Import

```java
TorchScriptMetadata metadata = TorchScriptModelImport.inspectModel("model.safetensors");

System.out.println("Format: " + metadata.getFormat());
System.out.println("Total parameters: " + metadata.getTotalParameters());
System.out.println("Detected architecture: " + metadata.getArchitecture());

for (TorchScriptTensorInfo tensor : metadata.getTensors()) {
    System.out.println(tensor.getName() + ": " + Arrays.toString(tensor.getShape()));
}
```

### Convert to SDZ Format

```java
// Convert PyTorch model to ND4J's native format
TorchScriptModelImport.convertToSDZ("model.safetensors", "model.sdz");

// Load the converted model
SameDiff sd = SameDiff.load(new File("model.sdz"), true);
```

## Tensor Name Normalization

PyTorch tensor names use dots for hierarchy (e.g., `layer1.0.conv1.weight`). These are normalized for SameDiff:

| Original | Normalized |
|----------|------------|
| `layer1.0.conv1.weight` | `layer1_0_conv1_weight` |
| `encoder.layers.3.self_attn.bias` | `encoder_layers_3_self_attn_bias` |
| `module::layer.weight` | `module__layer_weight` |
| `0.layer.weight` | `t_0_layer_weight` |

Rules:
1. Replace `.` with `_`
2. Replace invalid characters (`::`, `-`, `/`) with `_`
3. Prefix with `t_` if name starts with a digit

## Consequences

### Positive

1. **Ecosystem Access** - Users can leverage the vast PyTorch model ecosystem
2. **Safe Format Support** - Safetensors provides security benefits over pickle-based formats
3. **Architecture Awareness** - Automatic detection simplifies import of common models
4. **Flexible Options** - ConversionOptions allows customization for different use cases
5. **Memory Efficiency** - Memory-mapped reading for large models

### Negative

1. **Limited Op Coverage** - Not all PyTorch ops are mapped (focus on CNN operations)
2. **Dynamic Graphs** - TorchScript's dynamic features may not fully translate
3. **Pickle Security** - TorchScript .pt files use pickle, which can execute arbitrary code
4. **Maintenance Burden** - Must track changes in PyTorch's serialization format

### Risks

1. **Version Compatibility** - PyTorch format changes may break import
2. **Numerical Precision** - Weight transformations may introduce small numerical differences
3. **Unsupported Ops** - Models using unmapped ops will fail or produce incorrect results

## Testing

The module includes comprehensive tests:

| Test Class | Coverage |
|------------|----------|
| `TorchScriptDataTypeTest` | Data type conversions |
| `TorchScriptFormatDetectorTest` | Format detection |
| `SafetensorsReaderTest` | Safetensors parsing |
| `TensorNameMapperTest` | Name normalization |
| `ConversionOptionsTest` | Builder pattern |
| `ArchitectureRegistryTest` | Architecture detection |
| `TorchScriptModelImportTest` | Integration tests |

## References

- [Safetensors Format Specification](https://github.com/huggingface/safetensors)
- [TorchScript Documentation](https://pytorch.org/docs/stable/jit.html)
- [Python Pickle Protocol](https://docs.python.org/3/library/pickle.html)
- [ND4J GGML Module](../nd4j-ggml/) - Similar pattern for GGML format import
