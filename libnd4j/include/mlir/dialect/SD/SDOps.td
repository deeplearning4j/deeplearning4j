//===- SDOps.td - SD Dialect Operations -----------------------*- tablegen -*-===//
//
// This program and the accompanying materials are made available under the
// terms of the Apache License, Version 2.0 which is available at
// https://www.apache.org/licenses/LICENSE-2.0.
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file defines the SD (SameDiff) dialect operations for libnd4j/nd4j.
// These operations map to the 392+ operations defined in libnd4j.
//
//===----------------------------------------------------------------------===//

#ifndef SD_DIALECT_OPS_TD
#define SD_DIALECT_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// SD Dialect Definition
//===----------------------------------------------------------------------===//

def SD_Dialect : Dialect {
    let name = "sd";
    let cppNamespace = "::mlir::sd";
    let summary = "SameDiff/ND4J operation dialect";
    let description = [{
        The SD dialect provides operations corresponding to libnd4j/nd4j
        deep learning operations. These include:
        - BLAS operations (matmul, gemm, xw_plus_b)
        - Convolution operations (conv2d, conv3d, depthwise_conv)
        - Pooling operations (maxpool, avgpool)
        - Normalization operations (batchnorm, layernorm)
        - Activation functions (relu, sigmoid, tanh, softmax)
        - Reduction operations (sum, mean, max, min)
        - Shape operations (reshape, transpose, concat)
        - Attention mechanisms (dot_product_attention, multi_head_attention)
    }];
}

//===----------------------------------------------------------------------===//
// Base Operation Classes
//===----------------------------------------------------------------------===//

// Base class for all SD operations
class SD_Op<string mnemonic, list<Trait> traits = []>
    : Op<SD_Dialect, mnemonic, traits>;

// Base class for SD operations with no side effects
class SD_PureOp<string mnemonic, list<Trait> traits = []>
    : SD_Op<mnemonic, !listconcat([Pure], traits)>;

//===----------------------------------------------------------------------===//
// BLAS Operations
//===----------------------------------------------------------------------===//

def SD_MatMulOp : SD_PureOp<"matmul"> {
    let summary = "Matrix multiplication";
    let description = [{
        Computes matrix multiplication: C = A @ B
        With optional transpose flags for A and B.
    }];

    let arguments = (ins
        AnyTensor:$a,
        AnyTensor:$b,
        DefaultValuedAttr<BoolAttr, "false">:$transposeA,
        DefaultValuedAttr<BoolAttr, "false">:$transposeB,
        DefaultValuedAttr<F64Attr, "1.0">:$alpha,
        DefaultValuedAttr<F64Attr, "0.0">:$beta
    );

    let results = (outs AnyTensor:$result);

    let assemblyFormat = [{
        $a `,` $b attr-dict `:` type($a) `,` type($b) `->` type($result)
    }];
}

def SD_XwPlusBOp : SD_PureOp<"xw_plus_b"> {
    let summary = "Fused matrix multiply and bias add";
    let description = [{
        Computes: Y = X @ W + B
        This is a fused operation for efficiency in neural networks.
    }];

    let arguments = (ins
        AnyTensor:$x,
        AnyTensor:$weights,
        AnyTensor:$bias
    );

    let results = (outs AnyTensor:$result);

    let assemblyFormat = [{
        $x `,` $weights `,` $bias attr-dict `:` type($x) `,` type($weights) `,` type($bias) `->` type($result)
    }];
}

def SD_GemmOp : SD_PureOp<"gemm"> {
    let summary = "General matrix multiply";
    let description = [{
        General matrix multiply: C = alpha * A @ B + beta * C
    }];

    let arguments = (ins
        AnyTensor:$a,
        AnyTensor:$b,
        Optional<AnyTensor>:$c,
        DefaultValuedAttr<F64Attr, "1.0">:$alpha,
        DefaultValuedAttr<F64Attr, "0.0">:$beta,
        DefaultValuedAttr<BoolAttr, "false">:$transposeA,
        DefaultValuedAttr<BoolAttr, "false">:$transposeB
    );

    let results = (outs AnyTensor:$result);
}

//===----------------------------------------------------------------------===//
// Convolution Operations
//===----------------------------------------------------------------------===//

def SD_Conv2DOp : SD_PureOp<"conv2d"> {
    let summary = "2D convolution";
    let description = [{
        Performs 2D convolution on input tensor with weights.
        Supports various padding modes and strides.
    }];

    let arguments = (ins
        AnyTensor:$input,
        AnyTensor:$weights,
        Optional<AnyTensor>:$bias,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$padding,
        I64ArrayAttr:$dilation,
        DefaultValuedAttr<I64Attr, "0">:$paddingMode,  // 0=VALID, 1=SAME, 2=CAUSAL
        DefaultValuedAttr<BoolAttr, "true">:$isNCHW,
        DefaultValuedAttr<I64Attr, "0">:$weightsFormat
    );

    let results = (outs AnyTensor:$output);
}

def SD_Conv3DOp : SD_PureOp<"conv3d"> {
    let summary = "3D convolution";
    let description = [{
        Performs 3D convolution on input tensor with weights.
    }];

    let arguments = (ins
        AnyTensor:$input,
        AnyTensor:$weights,
        Optional<AnyTensor>:$bias,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$padding,
        I64ArrayAttr:$dilation,
        DefaultValuedAttr<I64Attr, "0">:$paddingMode,
        DefaultValuedAttr<BoolAttr, "true">:$isNCDHW
    );

    let results = (outs AnyTensor:$output);
}

def SD_DepthwiseConv2DOp : SD_PureOp<"depthwise_conv2d"> {
    let summary = "Depthwise 2D convolution";
    let description = [{
        Performs depthwise separable 2D convolution.
    }];

    let arguments = (ins
        AnyTensor:$input,
        AnyTensor:$weights,
        Optional<AnyTensor>:$bias,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$padding,
        I64ArrayAttr:$dilation,
        DefaultValuedAttr<BoolAttr, "true">:$isNCHW
    );

    let results = (outs AnyTensor:$output);
}

//===----------------------------------------------------------------------===//
// Pooling Operations
//===----------------------------------------------------------------------===//

def SD_MaxPool2DOp : SD_PureOp<"maxpool2d"> {
    let summary = "2D max pooling";
    let description = [{
        Performs 2D max pooling over the input tensor.
    }];

    let arguments = (ins
        AnyTensor:$input,
        I64ArrayAttr:$kernelSize,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$padding,
        DefaultValuedAttr<BoolAttr, "true">:$isNCHW
    );

    let results = (outs AnyTensor:$output);
}

def SD_AvgPool2DOp : SD_PureOp<"avgpool2d"> {
    let summary = "2D average pooling";
    let description = [{
        Performs 2D average pooling over the input tensor.
    }];

    let arguments = (ins
        AnyTensor:$input,
        I64ArrayAttr:$kernelSize,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$padding,
        DefaultValuedAttr<BoolAttr, "true">:$isNCHW,
        DefaultValuedAttr<BoolAttr, "false">:$countIncludePad
    );

    let results = (outs AnyTensor:$output);
}

def SD_GlobalAvgPoolOp : SD_PureOp<"global_avgpool"> {
    let summary = "Global average pooling";
    let description = [{
        Performs global average pooling, reducing spatial dimensions to 1x1.
    }];

    let arguments = (ins
        AnyTensor:$input,
        DefaultValuedAttr<BoolAttr, "true">:$isNCHW
    );

    let results = (outs AnyTensor:$output);
}

//===----------------------------------------------------------------------===//
// Normalization Operations
//===----------------------------------------------------------------------===//

def SD_BatchNormOp : SD_PureOp<"batchnorm"> {
    let summary = "Batch normalization";
    let description = [{
        Applies batch normalization to the input tensor.
    }];

    let arguments = (ins
        AnyTensor:$input,
        AnyTensor:$gamma,
        AnyTensor:$beta,
        AnyTensor:$mean,
        AnyTensor:$variance,
        DefaultValuedAttr<F64Attr, "1e-5">:$epsilon,
        DefaultValuedAttr<BoolAttr, "true">:$isNCHW
    );

    let results = (outs AnyTensor:$output);
}

def SD_LayerNormOp : SD_PureOp<"layernorm"> {
    let summary = "Layer normalization";
    let description = [{
        Applies layer normalization over the last N dimensions.
    }];

    let arguments = (ins
        AnyTensor:$input,
        AnyTensor:$gamma,
        AnyTensor:$beta,
        I64ArrayAttr:$normalizedShape,
        DefaultValuedAttr<F64Attr, "1e-5">:$epsilon
    );

    let results = (outs AnyTensor:$output);
}

//===----------------------------------------------------------------------===//
// Activation Functions
//===----------------------------------------------------------------------===//

def SD_ReluOp : SD_PureOp<"relu", [Elementwise, SameOperandsAndResultType]> {
    let summary = "ReLU activation";
    let description = [{ y = max(0, x) }];

    let arguments = (ins AnyTensor:$input);
    let results = (outs AnyTensor:$output);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def SD_LeakyReluOp : SD_PureOp<"leaky_relu", [Elementwise, SameOperandsAndResultType]> {
    let summary = "Leaky ReLU activation";
    let description = [{ y = x if x > 0 else alpha * x }];

    let arguments = (ins
        AnyTensor:$input,
        DefaultValuedAttr<F64Attr, "0.01">:$alpha
    );

    let results = (outs AnyTensor:$output);
}

def SD_SigmoidOp : SD_PureOp<"sigmoid", [Elementwise, SameOperandsAndResultType]> {
    let summary = "Sigmoid activation";
    let description = [{ y = 1 / (1 + exp(-x)) }];

    let arguments = (ins AnyTensor:$input);
    let results = (outs AnyTensor:$output);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def SD_TanhOp : SD_PureOp<"tanh", [Elementwise, SameOperandsAndResultType]> {
    let summary = "Tanh activation";
    let description = [{ y = tanh(x) }];

    let arguments = (ins AnyTensor:$input);
    let results = (outs AnyTensor:$output);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def SD_GeluOp : SD_PureOp<"gelu", [Elementwise, SameOperandsAndResultType]> {
    let summary = "GELU activation";
    let description = [{ Gaussian Error Linear Unit }];

    let arguments = (ins
        AnyTensor:$input,
        DefaultValuedAttr<BoolAttr, "false">:$precise  // false = fast approximation
    );

    let results = (outs AnyTensor:$output);
}

def SD_SoftmaxOp : SD_PureOp<"softmax"> {
    let summary = "Softmax activation";
    let description = [{ Softmax along specified dimension }];

    let arguments = (ins
        AnyTensor:$input,
        DefaultValuedAttr<I64Attr, "-1">:$axis
    );

    let results = (outs AnyTensor:$output);

    let assemblyFormat = [{
        $input attr-dict `:` type($input) `->` type($output)
    }];
}

def SD_SiluOp : SD_PureOp<"silu", [Elementwise, SameOperandsAndResultType]> {
    let summary = "SiLU (Swish) activation";
    let description = [{ y = x * sigmoid(x) }];

    let arguments = (ins AnyTensor:$input);
    let results = (outs AnyTensor:$output);
}

//===----------------------------------------------------------------------===//
// Reduction Operations
//===----------------------------------------------------------------------===//

def SD_ReduceSumOp : SD_PureOp<"reduce_sum"> {
    let summary = "Sum reduction";
    let description = [{ Reduces input along specified axes by computing the sum }];

    let arguments = (ins
        AnyTensor:$input,
        I64ArrayAttr:$axes,
        DefaultValuedAttr<BoolAttr, "false">:$keepDims
    );

    let results = (outs AnyTensor:$output);
}

def SD_ReduceMeanOp : SD_PureOp<"reduce_mean"> {
    let summary = "Mean reduction";
    let description = [{ Reduces input along specified axes by computing the mean }];

    let arguments = (ins
        AnyTensor:$input,
        I64ArrayAttr:$axes,
        DefaultValuedAttr<BoolAttr, "false">:$keepDims
    );

    let results = (outs AnyTensor:$output);
}

def SD_ReduceMaxOp : SD_PureOp<"reduce_max"> {
    let summary = "Max reduction";
    let description = [{ Reduces input along specified axes by computing the maximum }];

    let arguments = (ins
        AnyTensor:$input,
        I64ArrayAttr:$axes,
        DefaultValuedAttr<BoolAttr, "false">:$keepDims
    );

    let results = (outs AnyTensor:$output);
}

def SD_ReduceMinOp : SD_PureOp<"reduce_min"> {
    let summary = "Min reduction";
    let description = [{ Reduces input along specified axes by computing the minimum }];

    let arguments = (ins
        AnyTensor:$input,
        I64ArrayAttr:$axes,
        DefaultValuedAttr<BoolAttr, "false">:$keepDims
    );

    let results = (outs AnyTensor:$output);
}

def SD_ReduceProdOp : SD_PureOp<"reduce_prod"> {
    let summary = "Product reduction";
    let description = [{ Reduces input along specified axes by computing the product }];

    let arguments = (ins
        AnyTensor:$input,
        I64ArrayAttr:$axes,
        DefaultValuedAttr<BoolAttr, "false">:$keepDims
    );

    let results = (outs AnyTensor:$output);
}

//===----------------------------------------------------------------------===//
// Shape Operations
//===----------------------------------------------------------------------===//

def SD_ReshapeOp : SD_PureOp<"reshape"> {
    let summary = "Reshape tensor";
    let description = [{ Reshapes input tensor to specified shape }];

    let arguments = (ins
        AnyTensor:$input,
        I64ArrayAttr:$shape
    );

    let results = (outs AnyTensor:$output);
}

def SD_TransposeOp : SD_PureOp<"transpose"> {
    let summary = "Transpose tensor";
    let description = [{ Permutes dimensions of the input tensor }];

    let arguments = (ins
        AnyTensor:$input,
        I64ArrayAttr:$perm
    );

    let results = (outs AnyTensor:$output);
}

def SD_ConcatOp : SD_PureOp<"concat"> {
    let summary = "Concatenate tensors";
    let description = [{ Concatenates tensors along specified axis }];

    let arguments = (ins
        Variadic<AnyTensor>:$inputs,
        I64Attr:$axis
    );

    let results = (outs AnyTensor:$output);
}

def SD_SliceOp : SD_PureOp<"slice"> {
    let summary = "Slice tensor";
    let description = [{ Extracts a slice from input tensor }];

    let arguments = (ins
        AnyTensor:$input,
        I64ArrayAttr:$begin,
        I64ArrayAttr:$size
    );

    let results = (outs AnyTensor:$output);
}

//===----------------------------------------------------------------------===//
// Attention Operations
//===----------------------------------------------------------------------===//

def SD_DotProductAttentionOp : SD_PureOp<"dot_product_attention"> {
    let summary = "Scaled dot-product attention";
    let description = [{
        Computes scaled dot-product attention:
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    }];

    let arguments = (ins
        AnyTensor:$query,
        AnyTensor:$key,
        AnyTensor:$value,
        Optional<AnyTensor>:$mask,
        DefaultValuedAttr<BoolAttr, "true">:$scaled,
        DefaultValuedAttr<F64Attr, "0.0">:$dropoutProb
    );

    let results = (outs AnyTensor:$output);
}

def SD_MultiHeadAttentionOp : SD_PureOp<"multi_head_attention"> {
    let summary = "Multi-head attention";
    let description = [{
        Performs multi-head attention as in Transformer architecture.
    }];

    let arguments = (ins
        AnyTensor:$query,
        AnyTensor:$key,
        AnyTensor:$value,
        AnyTensor:$queryWeights,
        AnyTensor:$keyWeights,
        AnyTensor:$valueWeights,
        AnyTensor:$outputWeights,
        Optional<AnyTensor>:$mask,
        I64Attr:$numHeads,
        DefaultValuedAttr<F64Attr, "0.0">:$dropoutProb
    );

    let results = (outs AnyTensor:$output);
}

//===----------------------------------------------------------------------===//
// Binary Operations (Element-wise)
//===----------------------------------------------------------------------===//

def SD_AddOp : SD_PureOp<"add", [Elementwise, Commutative]> {
    let summary = "Element-wise addition";
    let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
    let results = (outs AnyTensor:$result);
}

def SD_SubOp : SD_PureOp<"sub", [Elementwise]> {
    let summary = "Element-wise subtraction";
    let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
    let results = (outs AnyTensor:$result);
}

def SD_MulOp : SD_PureOp<"mul", [Elementwise, Commutative]> {
    let summary = "Element-wise multiplication";
    let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
    let results = (outs AnyTensor:$result);
}

def SD_DivOp : SD_PureOp<"div", [Elementwise]> {
    let summary = "Element-wise division";
    let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
    let results = (outs AnyTensor:$result);
}

//===----------------------------------------------------------------------===//
// Unary Math Operations
//===----------------------------------------------------------------------===//

def SD_ExpOp : SD_PureOp<"exp", [Elementwise, SameOperandsAndResultType]> {
    let summary = "Element-wise exponential";
    let arguments = (ins AnyTensor:$input);
    let results = (outs AnyTensor:$output);
}

def SD_LogOp : SD_PureOp<"log", [Elementwise, SameOperandsAndResultType]> {
    let summary = "Element-wise natural logarithm";
    let arguments = (ins AnyTensor:$input);
    let results = (outs AnyTensor:$output);
}

def SD_SqrtOp : SD_PureOp<"sqrt", [Elementwise, SameOperandsAndResultType]> {
    let summary = "Element-wise square root";
    let arguments = (ins AnyTensor:$input);
    let results = (outs AnyTensor:$output);
}

def SD_RsqrtOp : SD_PureOp<"rsqrt", [Elementwise, SameOperandsAndResultType]> {
    let summary = "Element-wise reciprocal square root";
    let arguments = (ins AnyTensor:$input);
    let results = (outs AnyTensor:$output);
}

def SD_AbsOp : SD_PureOp<"abs", [Elementwise, SameOperandsAndResultType]> {
    let summary = "Element-wise absolute value";
    let arguments = (ins AnyTensor:$input);
    let results = (outs AnyTensor:$output);
}

def SD_NegOp : SD_PureOp<"neg", [Elementwise, SameOperandsAndResultType]> {
    let summary = "Element-wise negation";
    let arguments = (ins AnyTensor:$input);
    let results = (outs AnyTensor:$output);
}

#endif // SD_DIALECT_OPS_TD
