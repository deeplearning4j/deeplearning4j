# ADR: Mixed Precision Training Support for SameDiff

## Status

Accepted

Proposed by: Adam Gibson (December 2025)

Discussed with: Development Team

## Context

Modern deep learning increasingly relies on mixed precision training to achieve faster training speeds and reduced memory usage while maintaining model accuracy. Mixed precision training uses lower-precision data types (FP16, BFLOAT16) for most computations while keeping critical operations in higher precision (FP32) to maintain numerical stability.

**The Core Challenge**: Training neural networks in pure FP16 often leads to:
- **Gradient underflow**: Small gradient values become zero when represented in FP16
- **Gradient overflow**: Large scaled gradients exceed FP16's representable range
- **Accumulation errors**: Repeated low-precision operations accumulate numerical errors

**Industry Standard**: Frameworks like PyTorch, TensorFlow, and JAX have implemented mixed precision training with loss scaling as a standard feature. SameDiff needs equivalent capabilities to remain competitive for large-scale training workloads.

**Memory Constraints**: Modern models like transformers and large CNNs often exceed GPU memory when trained in FP32. Mixed precision can reduce memory requirements by approximately 50%, enabling training of larger models or larger batch sizes.

## Decision

We implement a comprehensive mixed precision training framework for SameDiff with three core components:

1. **Loss Scaling** - Dynamic and static loss scaling to prevent gradient underflow
2. **Gradient Accumulation** - Accumulate gradients across micro-batches for effective larger batch sizes
3. **Training Configuration Integration** - Seamless integration with existing TrainingConfig API

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        TrainingConfig                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ computeDataType │  │ masterWeightDT  │  │ LossScaleConfig │  │
│  │    (FP16)       │  │     (FP32)      │  │   (Dynamic)     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Training Loop                               │
│                                                                  │
│  1. Forward Pass (FP16 compute)                                 │
│  2. Loss Computation                                            │
│  3. Loss Scaling ──────────────────────┐                        │
│  4. Backward Pass (FP16 gradients)     │                        │
│  5. Gradient Unscaling ◄───────────────┘                        │
│  6. Overflow Check ──► Scale Adjustment (if dynamic)            │
│  7. Gradient Accumulation (optional)                            │
│  8. Optimizer Update (FP32 master weights)                      │
│  9. Weight Sync (FP32 → FP16)                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Component 1: Loss Scale Configuration

The `LossScaleConfig` class provides flexible configuration for loss scaling behavior:

```java
public class LossScaleConfig {
    public enum Mode {
        NONE,      // No loss scaling
        STATIC,    // Fixed scale factor
        DYNAMIC    // Adaptive scale factor
    }

    private Mode mode = Mode.NONE;
    private double initialScale = 65536.0;      // 2^16 - good starting point
    private double minScale = 1.0;              // Floor for dynamic scaling
    private double maxScale = 65536.0;          // Ceiling for dynamic scaling
    private double growthFactor = 2.0;          // Scale multiplier on success
    private double backoffFactor = 0.5;         // Scale divisor on overflow
    private int growthInterval = 2000;          // Steps before scale increase
}
```

**Design Rationale**:
- `initialScale = 65536.0` (2^16): Provides headroom for small gradients without immediately causing overflow
- `growthInterval = 2000`: Balances between aggressive scaling and stability
- `backoffFactor = 0.5`: Conservative backoff prevents oscillation

### Component 2: Loss Scaler

The `LossScaler` class manages the actual scaling operations during training:

```java
public class LossScaler {
    // Scale loss before backward pass
    public INDArray scaleLoss(INDArray loss) {
        if (!isEnabled()) return loss;
        return loss.mul(currentScale);
    }

    // Unscale gradients before optimizer
    public void unscaleGradients(INDArray gradients) {
        if (!isEnabled()) return;
        gradients.divi(currentScale);
    }

    // Check for overflow and adjust scale
    public boolean unscaleAndCheck(INDArray gradients) {
        unscaleGradients(gradients);

        if (hasInfOrNan(gradients)) {
            if (config.getMode() == Mode.DYNAMIC) {
                decreaseScale();  // Backoff on overflow
            }
            return false;  // Signal to skip this update
        }

        if (config.getMode() == Mode.DYNAMIC) {
            consecutiveFinite++;
            if (consecutiveFinite >= config.getGrowthInterval()) {
                increaseScale();  // Grow scale after stable period
                consecutiveFinite = 0;
            }
        }
        return true;  // Gradients are valid
    }
}
```

**Dynamic Scaling Algorithm**:
1. Start with `initialScale` (typically 65536)
2. After each successful backward pass, increment `consecutiveFinite`
3. When `consecutiveFinite >= growthInterval`, multiply scale by `growthFactor`
4. On overflow (inf/nan in gradients), multiply scale by `backoffFactor` and reset `consecutiveFinite`
5. Never let scale go below `minScale` or above `maxScale`

### Component 3: Gradient Accumulator

The `GradientAccumulator` enables effective larger batch sizes by accumulating gradients across micro-batches:

```java
public class GradientAccumulator {
    private final int accumulationSteps;
    private final Map<String, INDArray> accumulatedGradients;
    private int currentStep;

    public void accumulate(String varName, INDArray gradient) {
        INDArray accumulated = accumulatedGradients.get(varName);
        if (accumulated == null) {
            // Store in FP32 for numerical stability during accumulation
            accumulated = gradient.dataType() == DataType.FLOAT
                ? gradient.dup()
                : gradient.castTo(DataType.FLOAT);
            accumulatedGradients.put(varName, accumulated);
        } else {
            accumulated.addi(gradient);
        }
    }

    public Map<String, INDArray> getAndReset() {
        // Average gradients before returning
        for (INDArray arr : accumulatedGradients.values()) {
            arr.divi(accumulationSteps);
        }
        Map<String, INDArray> result = new HashMap<>(accumulatedGradients);
        accumulatedGradients.clear();
        currentStep = 0;
        return result;
    }
}
```

**Key Design Decision**: Gradients are accumulated in FP32 regardless of compute data type. This prevents precision loss during the accumulation process, which is critical when summing many small gradients.

### Component 4: Training Configuration Integration

Mixed precision settings integrate seamlessly with the existing `TrainingConfig` builder:

```java
TrainingConfig config = TrainingConfig.builder()
    .updater(new Adam(0.001))
    // Mixed precision settings
    .computeDataType(DataType.FLOAT16)
    .masterWeightDataType(DataType.FLOAT)
    .lossScaling(LossScaleConfig.builder()
        .mode(LossScaleConfig.Mode.DYNAMIC)
        .initialScale(65536.0)
        .growthInterval(2000)
        .build())
    .gradientAccumulationSteps(4)
    // Standard settings
    .dataSetFeatureMapping("input")
    .dataSetLabelMapping("label")
    .build();
```

### Data Type Hierarchy

The framework supports three key data types for mixed precision:

| Data Type | Bits | Range | Use Case |
|-----------|------|-------|----------|
| FLOAT32 | 32 | ~1e-38 to ~1e38 | Master weights, accumulation |
| FLOAT16 | 16 | ~6e-5 to 65504 | Forward/backward compute |
| BFLOAT16 | 16 | ~1e-38 to ~1e38 | Forward/backward (wider range) |

**BFLOAT16 vs FLOAT16**:
- FLOAT16: Better precision (10 mantissa bits), smaller range
- BFLOAT16: Same range as FP32 (8 exponent bits), less precision (7 mantissa bits)
- BFLOAT16 often preferred for training due to matching FP32 range

### Overflow Detection

Efficient overflow detection is critical for dynamic loss scaling:

```java
private boolean hasInfOrNan(INDArray arr) {
    // Use efficient native operations for checking
    return arr.isInfinite().any() || arr.isNaN().any();
}
```

For large-scale training, we plan to add a fused `check_numerics` operation in libnd4j that performs this check in a single pass.

## Implementation Details

### File Locations

```
nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/
├── config/
│   └── LossScaleConfig.java       # Loss scaling configuration
├── training/
│   ├── LossScaler.java            # Loss scaling implementation
│   └── GradientAccumulator.java   # Gradient accumulation
└── TrainingConfig.java            # Extended with mixed precision fields
```

### Integration Points

**TrainingSession** modifications:
1. After loss computation: `loss = lossScaler.scaleLoss(loss)`
2. After backward pass: `valid = lossScaler.unscaleAndCheck(gradients)`
3. If `!valid`: skip optimizer update, continue to next batch
4. Before optimizer: accumulate gradients if enabled
5. After optimizer: sync master weights to compute weights

### Memory Considerations

Mixed precision training affects memory usage in several ways:

| Component | FP32 Only | Mixed Precision |
|-----------|-----------|-----------------|
| Weights | W bytes | W/2 (FP16) + W (master) = 1.5W |
| Gradients | W bytes | W/2 bytes |
| Optimizer State | 2W bytes (Adam) | 2W bytes |
| Activations | A bytes | A/2 bytes |
| **Total** | 3W + A | 3.5W + A/2 |

For models where activations dominate (most deep networks), mixed precision provides significant memory savings despite maintaining FP32 master weights.

## Usage Examples

### Basic Mixed Precision Training

```java
SameDiff sd = SameDiff.create();
// ... build model ...

TrainingConfig config = TrainingConfig.builder()
    .updater(new Adam(0.001))
    .computeDataType(DataType.FLOAT16)
    .lossScaling(LossScaleConfig.dynamicDefault())
    .dataSetFeatureMapping("input")
    .dataSetLabelMapping("label")
    .build();

sd.setTrainingConfig(config);
sd.fit(trainData, numEpochs);
```

### Mixed Precision with Gradient Accumulation

```java
// Effective batch size = 32 * 4 = 128
TrainingConfig config = TrainingConfig.builder()
    .updater(new Adam(0.001))
    .computeDataType(DataType.BFLOAT16)
    .masterWeightDataType(DataType.FLOAT)
    .lossScaling(LossScaleConfig.builder()
        .mode(Mode.DYNAMIC)
        .initialScale(32768.0)
        .build())
    .gradientAccumulationSteps(4)
    .build();
```

### Static Loss Scaling (for stable training)

```java
// When you know a good scale factor for your model
TrainingConfig config = TrainingConfig.builder()
    .updater(new SGD(0.01))
    .computeDataType(DataType.FLOAT16)
    .lossScaling(LossScaleConfig.builder()
        .mode(Mode.STATIC)
        .initialScale(1024.0)
        .build())
    .build();
```

## Consequences

### Advantages

**Performance**:
- 2-3x training speedup on hardware with FP16 acceleration (Tensor Cores)
- Reduced memory bandwidth requirements

**Memory Efficiency**:
- ~50% reduction in activation memory
- Enables training larger models or larger batch sizes

**Compatibility**:
- Seamless integration with existing TrainingConfig API
- Opt-in: existing code continues to work unchanged
- Works with all optimizers (Adam, SGD, etc.)

**Numerical Stability**:
- Dynamic loss scaling automatically adapts to training dynamics
- FP32 master weights prevent long-term precision drift
- FP32 gradient accumulation maintains accuracy

### Disadvantages

**Complexity**:
- Additional configuration options may confuse beginners
- Dynamic loss scaling adds runtime overhead for overflow checking

**Hardware Dependency**:
- Full benefits require hardware FP16 support
- CPU training sees minimal speedup (mostly memory benefits)

**Debugging**:
- Overflow events can cause seemingly random training instabilities
- Loss scale dynamics add another variable to monitor

### Limitations

**Current Implementation**:
- Master weights stored separately (future: fused optimizer updates)
- No automatic mixed precision (requires explicit configuration)
- Overflow detection not yet fused in libnd4j

**Future Enhancements**:
- Automatic precision selection per-operation
- Fused FP16 optimizer updates in libnd4j
- Distributed training integration

## Testing

The implementation includes comprehensive tests in `MixedPrecisionTrainingTest.java`:

```java
// Test categories:
- LossScaleConfig: Builder, defaults, modes
- LossScaler: Scaling, unscaling, overflow detection, dynamic adjustment
- GradientAccumulator: Basic accumulation, averaging, multi-variable
- TrainingConfig: Mixed precision configuration, validation
- Integration: End-to-end workflow with overflow simulation
```

## Migration Guide

### For Users

Existing training code continues to work unchanged. To enable mixed precision:

```java
// Before (FP32 training)
TrainingConfig config = TrainingConfig.builder()
    .updater(new Adam(0.001))
    .build();

// After (Mixed precision training)
TrainingConfig config = TrainingConfig.builder()
    .updater(new Adam(0.001))
    .computeDataType(DataType.FLOAT16)
    .lossScaling(LossScaleConfig.dynamicDefault())
    .build();
```

### Best Practices

1. **Start with dynamic loss scaling** - It adapts to your model automatically
2. **Monitor loss scale** - If it drops too low, your model may have numerical issues
3. **Use BFLOAT16 for training** - Wider range reduces overflow risk
4. **Use FLOAT16 for inference** - Better precision, overflow less of a concern
5. **Combine with gradient accumulation** - Maximize effective batch size

## Conclusion

Mixed precision training support brings SameDiff to parity with other major deep learning frameworks for efficient large-scale training. The implementation prioritizes:

- **Simplicity**: Easy opt-in with sensible defaults
- **Stability**: Dynamic loss scaling handles numerical challenges automatically
- **Flexibility**: Configurable for different use cases and hardware
- **Compatibility**: Works with existing SameDiff models and workflows

The three-component architecture (loss scaling, gradient accumulation, config integration) provides a solid foundation for future enhancements like automatic mixed precision and fused optimizer implementations.

## References

- [NVIDIA Mixed Precision Training](https://developer.nvidia.com/automatic-mixed-precision)
- [PyTorch Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html)
- [Mixed Precision Training Paper (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740)
- [BFLOAT16 Format](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)
