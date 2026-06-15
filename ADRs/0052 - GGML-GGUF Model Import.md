# GGML/GGUF Model Import

## Status

Implemented

Proposed by: Adam Gibson (30-12-2024)

## Context

The GGML (Georgi Gerganov Machine Learning) format and its successor GGUF (GGML Universal Format) have become the de facto standard for distributing quantized large language models (LLMs). Tools like llama.cpp use these formats extensively, and the community has produced thousands of models in GGUF format on platforms like Hugging Face.

Currently, ND4J/SameDiff lacks the ability to import models from these formats. This creates a significant barrier for users who want to:

1. Use popular quantized LLMs (LLaMA, Mistral, Qwen, etc.) within the ND4J ecosystem
2. Fine-tune GGML/GGUF models using SameDiff's autodiff capabilities
3. Convert GGML/GGUF models to ND4J's native SDZ format for optimized inference
4. Integrate with existing Java/JVM-based applications that rely on ND4J

The GGML ecosystem supports numerous quantization schemes (Q4_0, Q8_0, K-quants, etc.) that dramatically reduce model size while maintaining reasonable accuracy. Supporting these formats enables ND4J to work with models that would otherwise be too large to deploy.

## Related Work

This work builds upon and integrates with:

- [ADR 0001 - SameDiff File Format](./0001-SameDiff_File_Format.md): The SDZ format is the target output format for converted models
- [ADR 0035 - SameDiff Extended Storage Format](./0035-Samediff-Extended-Storage-Format.md): Provides sharding capabilities for large models
- [ADR 0003 - Import IR](./0003-Import_IR.md): Establishes patterns for importing external model formats

The GGUF format specification is maintained at:
- https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

## Decision

We introduce a new `nd4j-ggml` module that provides comprehensive support for importing GGML and GGUF format models into ND4J/SameDiff. The module consists of:

### Module Structure

```
nd4j/nd4j-ggml/
├── src/main/java/org/nd4j/ggml/
│   ├── GGMLModelImport.java         # Main entry point
│   ├── GGMLImportException.java     # Custom exception
│   ├── format/                       # File format handling
│   │   ├── GGMLFormat.java          # Format enum (GGML, GGUF)
│   │   ├── GGMLFormatDetector.java  # Magic byte detection
│   │   ├── GGMLDataType.java        # Type mappings
│   │   ├── GGUFReader.java          # GGUF file reader
│   │   ├── GGMLReader.java          # Legacy GGML reader
│   │   ├── GGMLMetadata.java        # Model metadata
│   │   └── GGMLTensorInfo.java      # Tensor information
│   ├── quantization/                 # Dequantization
│   │   ├── Dequantizer.java         # Interface
│   │   ├── DequantizerFactory.java  # Factory
│   │   └── Q*Dequantizer.java       # Type-specific implementations
│   ├── architecture/                 # Model architectures
│   │   ├── ModelArchitecture.java   # Interface
│   │   ├── ArchitectureRegistry.java
│   │   ├── LLaMAArchitecture.java   # LLaMA/Mistral support
│   │   └── GenericArchitecture.java # Fallback
│   └── convert/                      # Conversion logic
│       ├── ConversionOptions.java   # Configuration
│       └── GGMLToSameDiffConverter.java
└── src/test/java/                    # Comprehensive tests
```

### Key Design Decisions

#### 1. Format Support

We support both legacy GGML formats (GGML, GGMF, GGJT) and the modern GGUF format. Format detection is automatic based on magic bytes:

```java
// GGUF magic: 'GGUF' = 0x46554747
// Legacy: 'ggml' = 0x67676D6C, 'ggmf' = 0x67676D66, 'ggjt' = 0x67676A74
GGMLFormat format = GGMLFormatDetector.detect(file);
```

#### 2. Memory-Mapped I/O

For large models (7B+ parameters), we use memory-mapped I/O to avoid loading entire files into memory:

```java
MappedByteBuffer buffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, size);
```

#### 3. Configurable Quantization Handling

Users can choose how to handle quantized weights:

```java
ConversionOptions options = ConversionOptions.builder()
    .quantizationMode(QuantizationMode.DEQUANTIZE_TO_FLOAT32)  // Full precision
    // or .quantizationMode(QuantizationMode.DEQUANTIZE_TO_FLOAT16)  // Half precision
    // or .quantizationMode(QuantizationMode.PRESERVE_QUANTIZATION)  // Keep quantized
    .build();
```

#### 4. Architecture-Aware Graph Building

Different model architectures (LLaMA, BERT, GPT-2) have different layer structures. We use a registry pattern:

```java
public interface ModelArchitecture {
    String getName();
    boolean canHandle(GGMLMetadata metadata);
    SameDiff buildGraph(GGMLMetadata metadata, Map<String, INDArray> weights, ConversionOptions options);
}
```

#### 5. Tensor Name Mapping

GGML uses different naming conventions than typical PyTorch/HuggingFace models. We provide mapping:

```java
// GGML name -> SameDiff name
"blk.0.attn_q.weight" -> "model.layers.0.self_attn.q_proj.weight"
"token_embd.weight" -> "model.embed_tokens.weight"
```

### Usage Example

```java
// Simple import
SameDiff model = GGMLModelImport.importModel("llama-7b.gguf");

// With options
ConversionOptions options = ConversionOptions.builder()
    .quantizationMode(QuantizationMode.DEQUANTIZE_TO_FLOAT16)
    .forTraining(true)
    .build();
SameDiff model = GGMLModelImport.importModel("llama-7b.gguf", options);

// Convert to SDZ directly
GGMLModelImport.convertToSDZ("llama-7b.gguf", "llama-7b.sdz");

// Inspect metadata without full import
GGMLMetadata metadata = GGMLModelImport.inspectModel("llama-7b.gguf");
System.out.println("Architecture: " + metadata.getArchitecture());
System.out.println("Parameters: " + metadata.getTotalParameters());
```

## Consequences

### Advantages

1. **Access to GGML Ecosystem**: Users can now use thousands of pre-quantized models from Hugging Face and other sources.

2. **Memory Efficiency**: Support for quantized formats enables working with large models on limited hardware.

3. **Training Capability**: Dequantized models can be fine-tuned using SameDiff's autodiff capabilities.

4. **Format Conversion**: Models can be converted to ND4J's optimized SDZ format for deployment.

5. **Java/JVM Integration**: Enables LLM deployment in enterprise Java environments without Python dependencies.

6. **Consistent API**: Follows established patterns from other ND4J import modules (Keras, ONNX, TensorFlow).

7. **Extensible Architecture**: New model architectures can be added by implementing the `ModelArchitecture` interface.

### Drawbacks

1. **Quantization Loss**: Dequantizing to full precision loses the memory benefits of quantization (but enables training).

2. **Architecture Coverage**: Initially supports LLaMA-family models; other architectures require additional implementation.

3. **No Native Quantized Inference**: ND4J doesn't natively support quantized inference, so models must be dequantized.

4. **Large Model Sizes**: Dequantized models require significantly more memory than their quantized counterparts.

5. **Maintenance Burden**: GGUF format evolves; new quantization types require ongoing support.

## Appendix A: Supported Quantization Types

| Type | Bits | Block Size | Description |
|------|------|------------|-------------|
| Q4_0 | 4 | 32 | Basic 4-bit quantization |
| Q4_1 | 4 | 32 | 4-bit with minimum value |
| Q5_0 | 5 | 32 | 5-bit quantization |
| Q5_1 | 5 | 32 | 5-bit with minimum value |
| Q8_0 | 8 | 32 | 8-bit quantization |
| Q2_K | 2 | 256 | K-quant 2-bit |
| Q3_K | 3 | 256 | K-quant 3-bit |
| Q4_K | 4 | 256 | K-quant 4-bit |
| Q5_K | 5 | 256 | K-quant 5-bit |
| Q6_K | 6 | 256 | K-quant 6-bit |

## Appendix B: GGUF File Structure

```
+------------------+
| Header (24 bytes)|
|   - Magic (4)    |
|   - Version (4)  |
|   - Tensor Count |
|   - KV Count     |
+------------------+
| Metadata KV Pairs|
|   - Key (string) |
|   - Type         |
|   - Value        |
+------------------+
| Tensor Info      |
|   - Name         |
|   - Dimensions   |
|   - Type         |
|   - Offset       |
+------------------+
| Tensor Data      |
| (aligned)        |
+------------------+
```

## Appendix C: Architecture Detection

Architecture is detected from GGUF metadata key `general.architecture`:

```java
String arch = metadata.get("general.architecture");
// Returns: "llama", "mistral", "bert", "gpt2", etc.

ModelArchitecture handler = ArchitectureRegistry.detectArchitecture(metadata);
```
