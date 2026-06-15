# GGML Quantization Handling

## Status

Implemented

Proposed by: Adam Gibson (30-12-2024)

## Context

GGML models are typically distributed in quantized formats to reduce model size and memory requirements. A 7B parameter LLaMA model at full FP32 precision requires ~28GB of memory, but the same model quantized to Q4_0 requires only ~4GB.

The GGML ecosystem supports numerous quantization schemes, each with different trade-offs between model size, inference speed, and accuracy:

- **Legacy Quantization (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0)**: Block-based quantization with 32-element blocks
- **K-Quantization (Q2_K through Q6_K, Q8_K)**: Improved quantization with 256-element super-blocks and per-block scaling
- **I-Quantization (IQ2_XXS, IQ3_XXS, etc.)**: Importance-based quantization for extreme compression

When importing GGML models into ND4J/SameDiff, we must decide how to handle these quantized weights since ND4J does not natively support quantized tensor operations.

## Related Work

- [ADR 0052 - GGML/GGUF Model Import](./0052%20-%20GGML-GGUF%20Model%20Import.md): Parent ADR for GGML import
- GGML quantization reference: https://github.com/ggerganov/llama.cpp/blob/master/ggml-quants.c
- K-quants paper: https://github.com/ggerganov/llama.cpp/pull/1684

## Decision

We implement a flexible quantization handling strategy that supports three modes:

### 1. Dequantization to Full Precision

The default mode converts quantized weights back to floating-point representation:

```java
public interface Dequantizer {
    GGMLDataType getQuantType();
    int getBlockSize();
    int getBytesPerBlock();
    float[] dequantize(byte[] quantizedData, long numElements);
    INDArray dequantizeToArray(byte[] quantizedData, long[] shape, DataType targetType);
}
```

#### Q4_0 Dequantization Algorithm

Q4_0 uses 18 bytes per 32-element block: 2 bytes for FP16 scale + 16 bytes for 32 4-bit values.

```java
public float[] dequantize(byte[] data, long numElements) {
    float[] result = new float[(int) numElements];
    int numBlocks = (int) ((numElements + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for (int block = 0; block < numBlocks; block++) {
        int offset = block * BYTES_PER_BLOCK;

        // Read FP16 scale
        float scale = Float16.toFloat(readShort(data, offset));

        // Dequantize 32 4-bit values
        for (int i = 0; i < BLOCK_SIZE / 2; i++) {
            byte packed = data[offset + 2 + i];
            int low = packed & 0x0F;
            int high = (packed >> 4) & 0x0F;

            // Q4_0: values are in range [0, 15], centered at 8
            result[block * BLOCK_SIZE + i * 2] = ((low - 8) * scale);
            result[block * BLOCK_SIZE + i * 2 + 1] = ((high - 8) * scale);
        }
    }
    return result;
}
```

#### K-Quant Dequantization

K-quants use super-blocks of 256 elements with nested quantization:

```java
// Q4_K: 256-element super-blocks
// Structure: scales (12 bytes) + mins (12 bytes) + data (128 bytes)
public float[] dequantize(byte[] data, long numElements) {
    float[] result = new float[(int) numElements];
    int numSuperBlocks = (int) ((numElements + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for (int sb = 0; sb < numSuperBlocks; sb++) {
        int offset = sb * BYTES_PER_BLOCK;

        // Read super-block scale and min (FP16)
        float dScale = Float16.toFloat(readShort(data, offset));
        float dMin = Float16.toFloat(readShort(data, offset + 2));

        // Process 8 sub-blocks of 32 elements each
        for (int subBlock = 0; subBlock < 8; subBlock++) {
            // Get quantized scale and min for this sub-block
            int scaleIdx = /* ... */;
            float scale = dScale * getScale(data, offset, subBlock);
            float min = dMin * getMin(data, offset, subBlock);

            // Dequantize 32 4-bit values
            for (int i = 0; i < 32; i++) {
                int value = getQuantizedValue(data, offset, subBlock, i);
                result[sb * BLOCK_SIZE + subBlock * 32 + i] = value * scale + min;
            }
        }
    }
    return result;
}
```

### 2. Target Precision Selection

Users can choose the output precision for dequantized weights:

```java
public enum QuantizationMode {
    DEQUANTIZE_TO_FLOAT32,   // Full precision, best for training
    DEQUANTIZE_TO_FLOAT16,   // Half precision, good balance
    DEQUANTIZE_TO_BFLOAT16,  // BFloat16, optimized for certain hardware
    PRESERVE_QUANTIZATION    // Keep raw quantized data
}

ConversionOptions options = ConversionOptions.builder()
    .quantizationMode(QuantizationMode.DEQUANTIZE_TO_FLOAT16)
    .targetDataType(DataType.HALF)
    .build();
```

### 3. Quantization Preservation (Advanced)

For advanced use cases, we preserve quantization metadata for potential future native support:

```java
QuantizationInfo info = QuantizationInfo.builder()
    .quantType(GGMLDataType.GGML_TYPE_Q4_K)
    .scales(extractedScales)
    .zeroPoints(extractedZeroPoints)
    .blockSize(256)
    .build();
```

### Dequantizer Factory Pattern

```java
public class DequantizerFactory {
    private static final Map<GGMLDataType, Dequantizer> dequantizers = new ConcurrentHashMap<>();

    static {
        register(new Q4_0Dequantizer());
        register(new Q4_1Dequantizer());
        register(new Q5_0Dequantizer());
        register(new Q5_1Dequantizer());
        register(new Q8_0Dequantizer());
        register(new Q2_KDequantizer());
        register(new Q3_KDequantizer());
        register(new Q4_KDequantizer());
        register(new Q5_KDequantizer());
        register(new Q6_KDequantizer());
    }

    public static Dequantizer getDequantizer(GGMLDataType type) {
        Dequantizer dequantizer = dequantizers.get(type);
        if (dequantizer == null) {
            throw new IllegalArgumentException("No dequantizer for: " + type);
        }
        return dequantizer;
    }

    public static boolean hasDequantizer(GGMLDataType type) {
        return dequantizers.containsKey(type);
    }
}
```

## Consequences

### Advantages

1. **Flexibility**: Users can choose the right trade-off between memory and precision for their use case.

2. **Training Support**: Dequantized models can be fine-tuned using standard floating-point operations.

3. **Accuracy Preservation**: Dequantization follows the exact algorithms used by llama.cpp, ensuring consistent results.

4. **Extensibility**: New quantization types can be added by implementing the `Dequantizer` interface.

5. **Memory Options**: FP16 dequantization provides 2x memory savings over FP32 while maintaining good precision.

### Drawbacks

1. **Memory Expansion**: Dequantizing from Q4_0 to FP32 increases memory by 8x per tensor.

2. **Loss of Compression Benefits**: Dequantized models cannot leverage quantization-aware inference optimizations.

3. **Processing Overhead**: Dequantization adds import time, especially for large models.

4. **No Native Quantized Ops**: ND4J doesn't support quantized operations, so inference requires full-precision compute.

## Appendix A: Quantization Format Details

### Q4_0 Block Layout (18 bytes per 32 elements)
```
+------------------+
| Scale (FP16, 2B) |
+------------------+
| Data (16 bytes)  |
| 32 x 4-bit vals  |
+------------------+
```

### Q4_K Super-Block Layout (144 bytes per 256 elements)
```
+------------------------+
| d (FP16, 2B) - scale   |
| dmin (FP16, 2B) - min  |
+------------------------+
| scales (12 bytes)      |
| 8 x 6-bit scale values |
+------------------------+
| mins (12 bytes)        |
| 8 x 6-bit min values   |
+------------------------+
| data (128 bytes)       |
| 256 x 4-bit values     |
+------------------------+
```

### Q8_0 Block Layout (34 bytes per 32 elements)
```
+------------------+
| Scale (FP16, 2B) |
+------------------+
| Data (32 bytes)  |
| 32 x 8-bit vals  |
+------------------+
```

## Appendix B: Dequantization Accuracy

Dequantization from GGML quantized formats introduces minimal error compared to the original floating-point values. Typical reconstruction errors:

| Format | Mean Absolute Error | Max Error |
|--------|--------------------:|----------:|
| Q8_0   | ~0.001             | ~0.01     |
| Q6_K   | ~0.002             | ~0.02     |
| Q5_K   | ~0.003             | ~0.03     |
| Q4_K   | ~0.005             | ~0.05     |
| Q4_0   | ~0.008             | ~0.08     |
| Q3_K   | ~0.010             | ~0.10     |
| Q2_K   | ~0.020             | ~0.20     |

These errors are typically acceptable for inference and can be corrected during fine-tuning.
