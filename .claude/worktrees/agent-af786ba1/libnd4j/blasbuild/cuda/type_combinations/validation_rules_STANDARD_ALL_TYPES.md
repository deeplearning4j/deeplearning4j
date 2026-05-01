# Type Combination Validation Rules

## Overview
This document describes the semantic validation rules applied to type combinations.

## 2-Type Combination Rules

### Always Valid
- **Same Type**: `(T, T)` for any type T
- **Bool Operations**: `(Bool, T)` or `(T, Bool)` for any type T
- **Float Pairs**: Any combination of float types (mixed precision)
- **Integer Pairs**: Any combination of integer types

### Conditional
- **Int-to-Float**: Specific promotions like `(INT32, FLOAT32)`

## 3-Type Combination Rules

### Pattern Categories

#### Identity Patterns
- `(T, T, T)` - Same type for all three

#### Mixed Precision Patterns
- `(HALF, HALF, FLOAT32)` - FP16 to FP32 accumulation
- `(BFLOAT16, BFLOAT16, FLOAT32)` - BF16 to FP32 accumulation

#### Quantization Patterns
- `(INT8/UINT8, FLOAT32, FLOAT32)` - Dequantization
- `(FLOAT32, INT8/UINT8, FLOAT32)` - Quantization scale
- `(INT8, INT8, INT32)` - INT8 accumulation

#### String Patterns
- `(UTF*, UTF*, UTF*)` - String operations
- `(UTF*, INT32/INT64, INT32/INT64)` - String indexing

## Aggressive Filtering Rules

When aggressive filtering is enabled, these patterns are blocked:
- Precision downgrades (e.g., DOUBLE to FLOAT32)
- Bool to floating point conversions
- Invalid string conversions
- Integer precision loss

## Profile: STANDARD_ALL_TYPES
