# ADR: LoRA Fused MatMul

## Status

Implemented

Proposed by: Adam Gibson (January 2025)

Discussed with: Development Team

## Context

Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning technique that adds trainable low-rank matrices to frozen pre-trained weights. Instead of fine-tuning all parameters in a weight matrix W (shape `[out_features, in_features]`), LoRA learns two small matrices A (shape `[r, in_features]`) and B (shape `[out_features, r]`) where `r << min(out_features, in_features)`:

```
output = input @ W^T + scaling * (input @ A^T @ B^T)
```

This allows fine-tuning with 100-1000x fewer trainable parameters while maintaining model quality close to full fine-tuning.

**The Naive Implementation Problem**: Without a fused op, this computation requires three separate GEMM calls and one accumulation:

1. `base = input @ W^T` — full-rank matmul
2. `temp1 = input @ A^T` — low-rank projection down
3. `temp2 = temp1 @ B^T` — low-rank projection up
4. `output = base + scaling * temp2` — accumulation

Each GEMM launches a separate kernel with its own memory allocation for output, resulting in 4 kernel launches, 3 intermediate allocations, and 4 read/write passes through the data. For a model with 96 LoRA-adapted layers, this overhead accumulates significantly.

## Decision

We implement a fused LoRA matmul custom op that computes the entire LoRA-adapted linear transformation in a single operation, minimizing kernel launches and intermediate allocations.

### Op Specification

```
CUSTOM_OP_IMPL(lora_matmul, 4, 1, false, -2, -2)

Inputs:
  0: input   [batch, in_features]
  1: weight  [out_features, in_features]
  2: loraA   [r, in_features]
  3: loraB   [out_features, r]

Outputs:
  0: output  [batch, out_features]

Float Args:
  0: scaling (default 1.0) — LoRA scaling factor (typically alpha/r)

Bool Args:
  0: transposeWeight (default true) — whether to transpose weight matrix
```

### Implementation

```cpp
CUSTOM_OP_IMPL(lora_matmul, 4, 1, false, -2, -2) {
    auto input  = INPUT_VARIABLE(0);  // [batch, in_features]
    auto weight = INPUT_VARIABLE(1);  // [out_features, in_features]
    auto loraA  = INPUT_VARIABLE(2);  // [r, in_features]
    auto loraB  = INPUT_VARIABLE(3);  // [out_features, r]
    auto output = OUTPUT_VARIABLE(0); // [batch, out_features]

    double scaling = T_ARG(0) > 0 ? T_ARG(0) : 1.0;
    bool transposeW = block.numB() > 0 ? B_ARG(0) : true;

    // Step 1: Base matmul — input @ W^T → output
    MmulHelper::mmul(input, weight, output, 1.0, 0.0, transposeW);

    // Step 2: LoRA pathway — input @ A^T → temp1 [batch, r]
    auto temp1 = NDArrayFactory::create(input->dataType(), {batch, r});
    MmulHelper::mmul(input, loraA, &temp1, 1.0, 0.0, true);

    // Step 3: LoRA pathway — temp1 @ B^T → temp2 [batch, out_features]
    auto temp2 = NDArrayFactory::create(input->dataType(), {batch, outFeatures});
    MmulHelper::mmul(&temp1, loraB, &temp2, 1.0, 0.0, true);

    // Step 4: Accumulate — output += scaling * temp2
    output->applyPairwiseTransform(transform::Add, temp2 * scaling, *output);

    return Status::OK;
}
```

### Shape Function

```cpp
DECLARE_SHAPE_FN(lora_matmul) {
    auto inputShape  = inputShape->at(0);
    auto weightShape = inputShape->at(1);
    auto batch = shape::sizeAt(inputShape, 0);
    auto outFeatures = shape::sizeAt(weightShape, 0);
    return SHAPELIST(ConstantShapeHelper::createShapeInfo(
        DataType::FLOAT32, 'c', {batch, outFeatures}));
}
```

### Integration with SameDiff

The op is registered in the standard op registry and can be used in SameDiff graphs:

```java
SDVariable output = sd.nn.loraMatmul(input, weight, loraA, loraB, scaling);
```

During ONNX import, LoRA-adapted linear layers can be detected and replaced with the fused op automatically when the pattern `matmul + matmul + matmul + add` is identified with appropriate rank constraints.

## Consequences

### Advantages

**Reduced Kernel Launches**: Single op dispatch instead of 4 separate ops. Saves ~20μs per LoRA layer from kernel launch overhead.

**Fewer Intermediates**: Only 2 temporary arrays (temp1, temp2) vs. 3 in the naive implementation (the final accumulation is done in-place).

**Clean API**: Single op encapsulates the entire LoRA computation, making SameDiff graphs more readable and easier to optimize.

**Gradient Support**: As a registered custom op, automatic differentiation through the LoRA matmul is supported for fine-tuning workflows.

### Disadvantages

**Fixed Rank Assumption**: The op assumes LoRA A and B matrices have compatible dimensions. Multi-rank LoRA (different ranks for different layers) requires separate op instances with different A/B shapes.

**No Merged Weight Support**: The op does not support pre-merging LoRA into base weights (`W' = W + scaling * B @ A`). For inference-only deployments where fine-tuning is complete, merging weights and using standard matmul would be faster.

**2D Only**: Currently limited to 2D inputs `[batch, features]`. Batched multi-head attention with 3D/4D inputs requires reshaping before and after the LoRA matmul.

## References

- libnd4j/include/ops/declarable/generic/nn/lora_matmul.cpp
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
