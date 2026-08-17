/* ******************************************************************************
 *
 * Copyright (c) 2026 Eclipse Foundation
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef LIBND4J_VULKAN_KERNEL_EMITTER_CATALOG_H
#define LIBND4J_VULKAN_KERNEL_EMITTER_CATALOG_H

#include <config.h>
#include <system/common.h>
#include <graph/vulkan/VulkanLegacyOpCatalog.h>
#include <ops/declarable/OpDescriptor.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(HAVE_VULKAN) && HAVE_VULKAN

namespace sd {
namespace graph {

/**
 * Arithmetic or indexing recipe implemented by a Vulkan kernel emitter.
 *
 * A recipe may only select emitted math or index equations. Intrinsic family
 * comes from the concrete op's own addTraits; Vulkan argument validation and
 * lowering shape use the reusable contracts below. Values are selected only by
 * canonical descriptor hashes, never operation names.
 */
enum class VulkanKernelRecipe : uint16_t {
  MATMUL,
  TRIANGULAR_SOLVE,
  RMS_NORM,
  RMS_NORM_BP,
  FUSED_LAYER_NORM_BP,
  FUSED_ROPE,
  ROPE,
  ROPE_BP,
  SKIP_RMS_NORM,
  RMS_NORM_LINEAR,
  FUSED_GEMM_SWIGLU,
  FUSED_RMS_NORM_SWIGLU,
  DOT_PRODUCT_ATTENTION,
  FLASH_ATTENTION,
  GROUPED_QUERY_ATTENTION,
  APPLY_ALIBI,
  FUSED_ATTENTION_PROJECTION,
  FUSED_ELEMENTWISE_CHAIN,
  FUSED_BIAS_DROPOUT_RESIDUAL,
  SWISH_MUL_BP,
  FUSED_MROPE,
  VISION_EMBEDDING_MERGE,
  SWIGLU,
  GEGLU,
  REGLU,

  ADD,
  SUBTRACT,
  MULTIPLY,
  DIVIDE,
  MINIMUM,
  MAXIMUM,
  FLOOR_MOD,
  MOD,
  FLOOR_DIVIDE,
  REVERSE_DIVIDE,
  REVERSE_SUBTRACT,
  SQUARED_SUBTRACT,
  POWER,
  ATAN2,
  REVERSE_MOD,
  TRUNCATE_DIV,
  SAFE_DIVIDE,
  DIVIDE_NO_NAN,
  XDIVY,
  XLOGY,
  XLOG1PY,
  POW_DERIVATIVE,
  FMOD,
  SHIFT_LEFT,
  SHIFT_RIGHT,
  CYCLIC_SHIFT_LEFT,
  CYCLIC_SHIFT_RIGHT,
  SWISH_MUL,
  ASSIGN,
  BIAS_ADD,

  SILU,
  ABS,
  NEGATE,
  SQRT,
  TANH,
  SIGMOID,
  RELU,
  GELU,
  FAST_GELU,
  SQUARE,
  SQUARED_RELU,
  HARD_SIGMOID,
  HARD_TANH,
  RELU6,
  LEAKY_RELU,
  ELU,
  ELU_SCALAR,
  ELU_DERIVATIVE,
  RELU_DERIVATIVE,
  LEAKY_RELU_DERIVATIVE,
  SIGMOID_CROSS_ENTROPY_SMOOTHER,
  LOG_X,
  STEP,
  LSTM_CLIP,
  SQUARED_REVERSE_SUBTRACT,
  REVERSE_POWER,
  LOGICAL_NOT_BINARY,
  RELATIVE_ERROR,
  BINARY_RELATIVE_ERROR,
  BINARY_MINIMUM_ABSOLUTE_RELATIVE_ERROR,
  AXPY,
  LOG_POISSON_LOSS,
  LOG_POISSON_LOSS_FULL,
  SELU,
  SOFTPLUS,
  SOFTSIGN,
  CUBE,
  RECTIFIED_TANH,
  RATIONAL_TANH,
  TANH_DERIVATIVE,
  HARD_TANH_DERIVATIVE,
  SIGMOID_DERIVATIVE,
  SOFTSIGN_DERIVATIVE,
  TAN_DERIVATIVE,
  SELU_DERIVATIVE,
  HARD_SIGMOID_DERIVATIVE,
  RATIONAL_TANH_DERIVATIVE,
  RECTIFIED_TANH_DERIVATIVE,
  SWISH_DERIVATIVE,
  ACOSH_DERIVATIVE,
  ASINH_DERIVATIVE,
  SINH_DERIVATIVE,
  LOG_SIGMOID_DERIVATIVE,
  SPECIAL_DERIVATIVE,
  CUBE_DERIVATIVE,
  GELU_DERIVATIVE,
  PRECISE_GELU_DERIVATIVE,
  MISH_DERIVATIVE,
  STABILIZE,
  STABILIZE_FP16,
  SCALED_TANH,
  AFFINE,
  SET_RANGE,
  FLOOR,
  LOG1P,
  RINT,
  SCALE,
  PRELU,
  CLIP_BY_VALUE,
  THRESHOLDED_RELU,

  RELU_BP,
  RELU6_BP,
  THRESHOLDED_RELU_BP,
  SIGMOID_BP,
  TANH_BP,
  ELU_BP,
  SELU_BP,
  LEAKY_RELU_BP,
  SOFTPLUS_BP,
  SOFTSIGN_BP,
  HARD_SIGMOID_BP,
  HARD_TANH_BP,
  SILU_BP,
  FUSED_GELU_BP,
  SQUARED_RELU_BP,
  RECTIFIED_TANH_BP,

  EQUAL,
  NOT_EQUAL,
  LESS,
  LESS_EQUAL,
  GREATER,
  GREATER_EQUAL,
  BOOLEAN_AND,
  BOOLEAN_OR,
  BOOLEAN_XOR,
  BOOLEAN_NOT,
  EPSILON_COMPARE,
  MATCH_CONDITION,
  MATCH_CONDITION_UNARY,
  IGAMMA,
  IGAMMAC,
  IS_INF,
  IS_NAN,
  IS_FINITE,
  IS_INF_OR_NAN,
  IS_POSITIVE,
  IS_NEGATIVE,
  SIGN,
  ONES,
  ROUND,
  ONE_MINUS,
  TIMES_ONE_MINUS,
  RECIPROCAL,
  CEIL,
  COS,
  SIN,
  EXP,
  LOG,
  ACOS,
  ASIN,
  ATAN,
  SINH,
  COSH,
  TAN,
  ERF,
  ERFC,
  EXPM1,
  ACOSH,
  ASINH,
  ATANH,
  LOG_SIGMOID,
  MISH,
  RSQRT,
  SELECT,
  CAST,

  SOFTMAX,
  LOG_SOFTMAX,
  LAYER_NORM,
  NORMALIZE_MOMENTS,
  BATCH_NORM,

  // Shared value equation for layout/rank transforms and partitioning. The
  // lowering contract, not this recipe, owns operand arity and index schedule.
  COPY,
  GATHER,
  EMBEDDING_LOOKUP,
  CONCAT,
  TRANSPOSE,
  PERMUTE,
  GATHER_ND,
  TILE,
  REPEAT,
  REVERSE,
  ROLL,
  SLICE,
  STRIDED_SLICE,
  STACK,
  SPLIT,
  UNSTACK,
  TRIU,
  TRIU_BP,
  WINDOW_PARTITION,
  WINDOW_UNPARTITION,
  PULL_INDEXED_TADS,
  DISJOINT_PAIR_SHUFFLE,

  FILL_AS,
  ZEROS_AS,
  ONES_AS,
  SHAPE_OF,
  ONE_HOT,
  EYE,
  LIN_SPACE,
  RANGE,
  RANK_OF,
  SIZE_OF,
  SIZE_AT,
  MIN_MAX_DATATYPE,
  UNIFORM_RANDOM,
  RANDOM_GENERIC,

  REDUCE_SUM,
  REDUCE_LOGSUMEXP,
  REDUCE_VARIANCE,
  REDUCE_STDEV,
  REDUCE_MAX,
  REDUCE_MIN,
  REDUCE_PRODUCT,
  REDUCE_COUNT_NONZERO,
  REDUCE_COUNT_ZERO,
  REDUCE_COUNT_MATCH,
  REDUCE_ENTROPY,
  REDUCE_LOG_ENTROPY,
  REDUCE_SHANNON_ENTROPY,
  REDUCE3,

  // Canonical legacy identity whose family/op-number selects the shared
  // elementwise/reduction lowering at record time.
  LEGACY_GENERIC,
  // The canonical identity is registered, but no device equation has been
  // supplied yet. Lowerers reject this value instead of silently materializing
  // an identity or using a host implementation.
  UNSUPPORTED
};

enum class VulkanKernelFamily : uint8_t {
  UNKNOWN,
  MATMUL,
  TRIANGULAR_SOLVE,
  ATTENTION,
  ELEMENTWISE_BINARY,
  ELEMENTWISE_UNARY,
  COMPARISON,
  LOGICAL,
  TERNARY,
  CAST,
  NORMALIZATION,
  REDUCTION,
  DATA_MOVEMENT,
  CONSTANT_GENERATION
};

/**
 * Reusable Vulkan kernel shapes. This is deliberately not an operation enum:
 * descriptors with different recipes share a contract whenever their MLIR and
 * SPIR-V scheduling structure is the same.
 */
enum class VulkanLoweringContract : uint8_t {
  DEFAULT,
  // Shared by the registered fused LLM descriptors; individual arithmetic is
  // still selected by VulkanKernelRecipe.
  FUSED_LLM,
  // Shared row-reduction schedule for softmax and log-softmax.
  SOFTMAX,
  // Shared row-statistics schedule for layer_norm and fused_layer_norm.
  LAYER_NORM,
  // Shared flattened copy traversal for same-shape, reshape, and structural
  // copy descriptors; per-descriptor legality is derived from addTraits.
  LINEAR_COPY,
  // Shared single-dispatch schedule for descriptor-driven indexed TAD movement.
  // The recipe selects gather versus disjoint-pair exchange semantics.
  INDEXED_TAD_MOVEMENT,
  // Forward/back substitution over a square matrix and one or more RHS columns.
  TRIANGULAR_SOLVE
};

/** Frozen descriptor-argument encodings shared across emitter recipes. */
enum class VulkanArgumentSchema : uint8_t {
  NONE,
  OPTIONAL_SCALAR,
  OPTIONAL_SCALAR_PAIR,
  OPTIONAL_FINITE_SCALAR,
  REDUCTION_KEEPDIMS,
  STATISTICAL_REDUCTION,
  INDEX_REDUCTION,
  EPSILON,
  AXES_IARGS,
  SINGLE_IARG,
  WINDOW_GEOMETRY_IARGS,
  PERMUTATION_AXES
};

/**
 * Generic frozen-argument arity. A maximum of -1 means unbounded. These ranges
 * describe descriptor ABI, not operation identity, and allow multiple accepted
 * forms without adding a schema enum for one operation.
 */
struct VulkanArgumentCountRange {
  int16_t minimum = 0;
  int16_t maximum = -1;
};

struct VulkanArgumentSignature {
  VulkanArgumentCountRange inputs;
  VulkanArgumentCountRange outputs;
  VulkanArgumentCountRange tArgs;
  VulkanArgumentCountRange iArgs;
  VulkanArgumentCountRange bArgs;
  VulkanArgumentCountRange dArgs;
  VulkanArgumentCountRange sArgs;
};

enum VulkanArgumentValueTraits : uint8_t {
  VULKAN_ARGUMENT_VALUES_NONE = 0,
  VULKAN_ARGUMENT_VALUES_FINITE_TARGS = 1u << 0,
  VULKAN_ARGUMENT_VALUES_ORDERED_TARG_PAIR = 1u << 1,
  // If present, TArg zero must be finite and strictly positive.
  VULKAN_ARGUMENT_VALUES_POSITIVE_TARG_ZERO = 1u << 2,
  // If present, TArg zero is a probability in [0, 1).
  VULKAN_ARGUMENT_VALUES_PROBABILITY_TARG_ZERO = 1u << 3,
  // A true first BArg is legal only when TArg zero is absent or zero. This
  // captures deterministic replay for stochastic training-mode contracts.
  VULKAN_ARGUMENT_VALUES_NONZERO_TARG_REQUIRES_FALSE_BARG = 1u << 4,
  // If present, an explicitly supplied first BArg must be true. This is used
  // by fully-writing contracts whose false form intentionally leaves output
  // storage uninitialized and therefore cannot participate in replay.
  VULKAN_ARGUMENT_VALUES_PRESENT_BARG_ZERO_TRUE = 1u << 5
};

struct VulkanArgumentContract {
  VulkanArgumentSignature alternatives[2];
  uint8_t alternativeCount = 0;
  uint8_t valueTraits = VULKAN_ARGUMENT_VALUES_NONE;
};

/**
 * Generic per-input storage roles. Inputs absent from all masks use the
 * schedule's ordinary payload type. Scalar-32 operands may be floating or
 * 32-bit integer; integer-32 and integer-64 operands must use the corresponding
 * integer storage width. Loaded integer-index operands may use either width,
 * with 64-bit storage admitted only when the physical device exposes
 * shaderInt64. Structural-index operands remain real descriptor bindings but
 * are never loaded, so framework-wide index tensors can use an inert i32 MemRef
 * without requiring shaderInt64.
 */
struct VulkanOperandTypeContract {
  uint16_t scalar32InputMask = 0;
  uint16_t integer32InputMask = 0;
  uint16_t integer64InputMask = 0;
  bool requireUniformSpecialInputs = false;
  uint16_t structuralIndexInputMask = 0;
  uint16_t integerIndexInputMask = 0;
};

enum VulkanKernelDTypeSupport : uint32_t {
  VULKAN_DTYPE_NONE = 0,
  VULKAN_DTYPE_FLOAT = 1u << 0,
  VULKAN_DTYPE_SIGNED_INT32 = 1u << 1,
  VULKAN_DTYPE_UNSIGNED_INT32 = 1u << 2,
  VULKAN_DTYPE_BOOL = 1u << 3,
  VULKAN_DTYPE_INDEX = 1u << 4
};

enum VulkanKernelLayoutSupport : uint32_t {
  VULKAN_LAYOUT_DENSE_C = 1u << 0,
  VULKAN_LAYOUT_DENSE_F = 1u << 1,
  VULKAN_LAYOUT_STRIDED_VIEW = 1u << 2,
  VULKAN_LAYOUT_BROADCAST = 1u << 3
};

/**
 * Vulkan-specific emitter properties that are not intrinsic ND4J OpTraits.
 *
 * Structural family selection still comes from each descriptor's addTraits.
 * These flags describe backend contracts shared by multiple descriptors; they
 * must not grow into an operation-name or one-bit-per-op table.
 */
enum VulkanKernelEmitterTraits : uint32_t {
  VULKAN_EMITTER_TRAIT_NONE = 0,
  VULKAN_EMITTER_TRAIT_FLOAT_RESULT = 1u << 2,
  VULKAN_EMITTER_TRAIT_INDEX_RESULT = 1u << 3,
  // Unary legacy predicates write BOOL while computing in the input domain.
  VULKAN_EMITTER_TRAIT_BOOLEAN_RESULT = 1u << 4,
  // Reduction axes are intrinsically the last dimension; an optional iArg is
  // keepDims rather than an axis list.
  VULKAN_EMITTER_TRAIT_IMPLICIT_LAST_AXIS = 1u << 7,
  // Statistical reduction accepts the framework bias-correction argument.
  VULKAN_EMITTER_TRAIT_BIAS_CORRECTION_ARGUMENT = 1u << 8,
  // Dispatch one invocation for each row selected by output dimension zero.
  VULKAN_EMITTER_TRAIT_DISPATCH_FIRST_DIM = 1u << 9,
  // Dispatch one invocation for every product of dimensions before the last.
  VULKAN_EMITTER_TRAIT_DISPATCH_OUTER_ROWS = 1u << 10,
  // Dispatch exactly one invocation for a whole-kernel copy/reduction schedule.
  VULKAN_EMITTER_TRAIT_DISPATCH_SINGLE = 1u << 11,
  // Dispatch rotary pairs, heads, and flattened batch/sequence as x/y/z.
  VULKAN_EMITTER_TRAIT_DISPATCH_ROTARY_GRID = 1u << 12,
  // Dispatch one invocation for each flattened batch/sequence row.
  VULKAN_EMITTER_TRAIT_DISPATCH_BATCH_SEQUENCE = 1u << 13,
  // Dispatch one invocation for each leading batch/head/query coordinate.
  VULKAN_EMITTER_TRAIT_DISPATCH_ATTENTION_QUERY = 1u << 14,
  // The replay-safe form has no tensor inputs; frozen descriptor arguments
  // fully determine the generated values and output shape.
  VULKAN_EMITTER_TRAIT_ARGUMENT_GENERATED = 1u << 15,
  // Accepts frozen boolean arguments that affect the emitted kernel contract.
  VULKAN_EMITTER_TRAIT_BOOLEAN_PARAMETERS = 1u << 16,
  // Accepts a positive floating epsilon used by numerical stabilization.
  VULKAN_EMITTER_TRAIT_EPSILON_PARAMETER = 1u << 17,
  // Apply |x| before combining a reduction element.
  VULKAN_EMITTER_TRAIT_ABSOLUTE_INPUT = 1u << 18,
  // Apply x*x before combining a reduction element.
  VULKAN_EMITTER_TRAIT_SQUARE_INPUT = 1u << 19,
  // Divide a reduction accumulator by the selected element count.
  VULKAN_EMITTER_TRAIT_MEAN_FINALIZE = 1u << 20,
  // Apply sqrt to a completed reduction accumulator.
  VULKAN_EMITTER_TRAIT_SQRT_FINALIZE = 1u << 21,
  // NativeOpExecutioner reductions carry CUDA's full-rank TAD permutation:
  // non-reduced dimensions first, reduced dimensions as the trailing suffix.
  VULKAN_EMITTER_TRAIT_TAD_REDUCTION_PERMUTATION = 1u << 22,
  // The emitted kernel consumes caller-owned RNG state through a replay-time
  // storage-buffer binding. State is refreshed before each submission and the
  // owning generator advances only after successful replay completion.
  VULKAN_EMITTER_TRAIT_RANDOM_STATE = 1u << 23,
  // Count reductions accumulate in the backend integer domain but always
  // materialize the NativeOpExecutioner INT64 result contract.
  VULKAN_EMITTER_TRAIT_COUNT_RESULT = 1u << 24,
  // Index reductions select the first matching element in logical traversal
  // order, independent of its value.
  VULKAN_EMITTER_TRAIT_INDEX_FIRST = 1u << 25,
  // Index reductions select the last matching element in logical traversal
  // order, independent of its value.
  VULKAN_EMITTER_TRAIT_INDEX_LAST = 1u << 26
};

/**
 * Portable storage-buffer ABI for the framework's 128-bit RandomGenerator.
 * CUDA copies the complete object to device memory. Vulkan preserves the same
 * state while spelling it as four fixed-width words so shaders do not require
 * the optional Int64 capability and the byte order is platform independent.
 */
struct VulkanRandomStateWords {
  uint32_t rootLow;
  uint32_t rootHigh;
  uint32_t nodeLow;
  uint32_t nodeHigh;
};

inline constexpr size_t kVulkanRandomStateWordCount =
    sizeof(VulkanRandomStateWords) / sizeof(uint32_t);
inline constexpr size_t kVulkanRandomRootLowWord =
    offsetof(VulkanRandomStateWords, rootLow) / sizeof(uint32_t);
inline constexpr size_t kVulkanRandomNodeLowWord =
    offsetof(VulkanRandomStateWords, nodeLow) / sizeof(uint32_t);

static_assert(sizeof(VulkanRandomStateWords) == 2 * sizeof(uint64_t),
              "Vulkan RNG state ABI must preserve both 64-bit generator states");

/**
 * One canonical descriptor-hash registration.
 *
 * traits is copied from the concrete op's own addTraits/registerTypes result;
 * this catalogue never consults or populates OpTraitsTable.cpp.
 */
struct SD_LIB_EXPORT VulkanKernelEmitterInfo {
  sd::LongType descriptorHash;
  uint32_t traits;
  VulkanKernelRecipe recipe;
  // Derived from descriptor traits and an explicit reusable lowering contract.
  VulkanKernelFamily family;
  VulkanLoweringContract loweringContract;
  VulkanArgumentSchema argumentSchema;
  uint32_t emitterTraits;
  uint32_t dtypeSupport;
  uint32_t layoutSupport;
  // Descriptor-declared minimum arities are copied from the concrete op and
  // used wherever the native descriptor already expresses the ABI.
  int16_t descriptorMinimumInputs;
  int16_t descriptorMinimumOutputs;
  int16_t descriptorMinimumTArgs;
  int16_t descriptorMinimumIArgs;
  // Optional generic alternatives for ABIs the native descriptor cannot
  // express (for example, tensor operands versus frozen scalar operands).
  VulkanArgumentContract argumentContract;
  // Generic per-input storage roles used by validation, textual MLIR selection,
  // and dialect lowering. Empty masks mean the schedule's ordinary payload role.
  VulkanOperandTypeContract operandTypeContract;
  // Primary-input rank, except for zero-input constant generators where these
  // bounds describe the output rank.
  int16_t minimumRank;
  int16_t maximumRank;  // -1 means framework maximum rank
};

/** Classify a registered emitter from the op's own descriptor traits. */
SD_LIB_EXPORT VulkanKernelFamily vulkanKernelFamily(
    const VulkanKernelEmitterInfo& emitter);

inline bool hasVulkanEmitterTrait(const VulkanKernelEmitterInfo& emitter,
                                  uint32_t trait) {
  return (emitter.emitterTraits & trait) == trait;
}

inline bool hasVulkanOpTrait(const VulkanKernelEmitterInfo& emitter,
                             uint32_t trait) {
  return (emitter.traits & trait) == trait;
}

inline bool vulkanArgumentCountMatches(
    const VulkanArgumentCountRange& range, int count) {
  return count >= range.minimum &&
         (range.maximum < 0 || count <= range.maximum);
}

inline bool vulkanArgumentSignatureMatches(
    const VulkanArgumentSignature& signature, int numInputs, int numOutputs,
    int numTArgs, int numIArgs, int numBArgs, int numDArgs, int numSArgs) {
  return vulkanArgumentCountMatches(signature.inputs, numInputs) &&
         vulkanArgumentCountMatches(signature.outputs, numOutputs) &&
         vulkanArgumentCountMatches(signature.tArgs, numTArgs) &&
         vulkanArgumentCountMatches(signature.iArgs, numIArgs) &&
         vulkanArgumentCountMatches(signature.bArgs, numBArgs) &&
         vulkanArgumentCountMatches(signature.dArgs, numDArgs) &&
         vulkanArgumentCountMatches(signature.sArgs, numSArgs);
}

inline bool vulkanArgumentContractMatches(
    const VulkanKernelEmitterInfo& emitter, int numInputs, int numOutputs,
    int numTArgs, int numIArgs, int numBArgs, int numDArgs, int numSArgs) {
  if (emitter.argumentContract.alternativeCount == 0) return true;
  for (uint8_t i = 0; i < emitter.argumentContract.alternativeCount; ++i) {
    if (vulkanArgumentSignatureMatches(
            emitter.argumentContract.alternatives[i], numInputs, numOutputs,
            numTArgs, numIArgs, numBArgs, numDArgs, numSArgs)) {
      return true;
    }
  }
  return false;
}

inline bool vulkanArgumentContractAcceptsInputCount(
    const VulkanKernelEmitterInfo& emitter, int numInputs) {
  for (uint8_t i = 0; i < emitter.argumentContract.alternativeCount; ++i) {
    if (vulkanArgumentCountMatches(
            emitter.argumentContract.alternatives[i].inputs, numInputs)) {
      return true;
    }
  }
  return false;
}

inline bool vulkanArgumentContractAcceptsTensorCounts(
    const VulkanKernelEmitterInfo& emitter, int numInputs, int numOutputs) {
  if (emitter.argumentContract.alternativeCount == 0) return true;
  for (uint8_t i = 0; i < emitter.argumentContract.alternativeCount; ++i) {
    const auto& signature = emitter.argumentContract.alternatives[i];
    if (vulkanArgumentCountMatches(signature.inputs, numInputs) &&
        vulkanArgumentCountMatches(signature.outputs, numOutputs)) {
      return true;
    }
  }
  return false;
}

inline bool vulkanArgumentContractAcceptsBooleanArguments(
    const VulkanKernelEmitterInfo& emitter) {
  for (uint8_t i = 0; i < emitter.argumentContract.alternativeCount; ++i) {
    const auto& range = emitter.argumentContract.alternatives[i].bArgs;
    if (range.maximum < 0 || range.maximum > 0) return true;
  }
  return false;
}

inline bool hasVulkanMixedOperandTypeContract(
    const VulkanKernelEmitterInfo& emitter) {
  return emitter.operandTypeContract.scalar32InputMask != 0 ||
         emitter.operandTypeContract.integer32InputMask != 0 ||
         emitter.operandTypeContract.integer64InputMask != 0 ||
         emitter.operandTypeContract.integerIndexInputMask != 0;
}

inline bool hasVulkanStructuralIndexOperands(
    const VulkanKernelEmitterInfo& emitter) {
  return emitter.operandTypeContract.structuralIndexInputMask != 0;
}

inline bool vulkanInputUsesScalar32Storage(
    const VulkanKernelEmitterInfo& emitter, unsigned inputIndex) {
  return inputIndex < 16 &&
         (emitter.operandTypeContract.scalar32InputMask &
          (uint16_t{1} << inputIndex)) != 0;
}

inline bool vulkanInputUsesInteger32Storage(
    const VulkanKernelEmitterInfo& emitter, unsigned inputIndex) {
  return inputIndex < 16 &&
         (emitter.operandTypeContract.integer32InputMask &
          (uint16_t{1} << inputIndex)) != 0;
}

inline bool vulkanInputUsesInteger64Storage(
    const VulkanKernelEmitterInfo& emitter, unsigned inputIndex) {
  return inputIndex < 16 &&
         (emitter.operandTypeContract.integer64InputMask &
          (uint16_t{1} << inputIndex)) != 0;
}

inline bool vulkanInputUsesIntegerIndexStorage(
    const VulkanKernelEmitterInfo& emitter, unsigned inputIndex) {
  return inputIndex < 16 &&
         (emitter.operandTypeContract.integerIndexInputMask &
          (uint16_t{1} << inputIndex)) != 0;
}

inline bool vulkanInputIsStructuralIndex(
    const VulkanKernelEmitterInfo& emitter, unsigned inputIndex) {
  return inputIndex < 16 &&
         (emitter.operandTypeContract.structuralIndexInputMask &
          (uint16_t{1} << inputIndex)) != 0;
}

/**
 * Trait/schema-derived schedules used by validation, textual MLIR, and SPIR-V
 * lowering. These deliberately encode no descriptor hash or operation name.
 */
inline bool usesBatchedMatrixListSchedule(
    const VulkanKernelEmitterInfo& emitter) {
  return emitter.family == VulkanKernelFamily::MATMUL &&
         emitter.descriptorMinimumInputs < 0 &&
         emitter.descriptorMinimumOutputs < 0 &&
         emitter.descriptorMinimumIArgs == 2 &&
         hasVulkanOpTrait(
             emitter, sd::ops::OP_TRAIT_MATMUL |
                          sd::ops::OP_TRAIT_EXTERNAL_WORKSPACE |
                          sd::ops::OP_TRAIT_FULLY_WRITING) &&
         emitter.loweringContract == VulkanLoweringContract::DEFAULT;
}

inline bool usesDenseMatrixProductSchedule(
    const VulkanKernelEmitterInfo& emitter) {
  return emitter.family == VulkanKernelFamily::MATMUL &&
         hasVulkanOpTrait(emitter, sd::ops::OP_TRAIT_MATMUL) &&
         emitter.loweringContract == VulkanLoweringContract::DEFAULT &&
         !usesBatchedMatrixListSchedule(emitter);
}

inline bool usesIndexedAccumulationSchedule(
    const VulkanKernelEmitterInfo& emitter) {
  return emitter.family == VulkanKernelFamily::DATA_MOVEMENT &&
         hasVulkanOpTrait(
             emitter, sd::ops::OP_TRAIT_DATA_MOVEMENT |
                          sd::ops::OP_TRAIT_SCATTER_ND |
                          sd::ops::OP_TRAIT_FULLY_WRITING |
                          sd::ops::OP_TRAIT_VALUE_DEPENDENT_SHAPE) &&
         hasVulkanEmitterTrait(
             emitter, VULKAN_EMITTER_TRAIT_DISPATCH_SINGLE) &&
         hasVulkanStructuralIndexOperands(emitter) &&
         emitter.argumentSchema == VulkanArgumentSchema::NONE;
}

inline bool usesIndexedTadMovementSchedule(
    const VulkanKernelEmitterInfo& emitter) {
  return emitter.family == VulkanKernelFamily::DATA_MOVEMENT &&
         hasVulkanOpTrait(
             emitter, sd::ops::OP_TRAIT_DATA_MOVEMENT |
                          sd::ops::OP_TRAIT_DATA_DEPENDENT) &&
         hasVulkanEmitterTrait(
             emitter, VULKAN_EMITTER_TRAIT_DISPATCH_SINGLE) &&
         emitter.loweringContract ==
             VulkanLoweringContract::INDEXED_TAD_MOVEMENT &&
         emitter.argumentSchema == VulkanArgumentSchema::NONE;
}

inline bool usesLinearConcatSchedule(
    const VulkanKernelEmitterInfo& emitter) {
  return emitter.family == VulkanKernelFamily::DATA_MOVEMENT &&
         hasVulkanOpTrait(
             emitter, sd::ops::OP_TRAIT_DATA_MOVEMENT |
                          sd::ops::OP_TRAIT_CONCAT |
                          sd::ops::OP_TRAIT_FULLY_WRITING) &&
         emitter.argumentSchema == VulkanArgumentSchema::SINGLE_IARG;
}

inline bool usesVariadicAxisConcatSchedule(
    const VulkanKernelEmitterInfo& emitter) {
  return emitter.family == VulkanKernelFamily::DATA_MOVEMENT &&
         hasVulkanOpTrait(emitter, sd::ops::OP_TRAIT_CONCAT) &&
         emitter.argumentSchema == VulkanArgumentSchema::AXES_IARGS;
}

inline bool usesAxisIndexedLookupSchedule(
    const VulkanKernelEmitterInfo& emitter) {
  return emitter.family == VulkanKernelFamily::DATA_MOVEMENT &&
         hasVulkanOpTrait(
             emitter, sd::ops::OP_TRAIT_DATA_MOVEMENT |
                          sd::ops::OP_TRAIT_FULLY_WRITING |
                          sd::ops::OP_TRAIT_GATHER) &&
         emitter.argumentSchema == VulkanArgumentSchema::AXES_IARGS;
}

inline bool usesModeIndexedLookupSchedule(
    const VulkanKernelEmitterInfo& emitter) {
  return emitter.family == VulkanKernelFamily::DATA_MOVEMENT &&
         hasVulkanOpTrait(
             emitter, sd::ops::OP_TRAIT_DATA_MOVEMENT |
                          sd::ops::OP_TRAIT_FULLY_WRITING |
                          sd::ops::OP_TRAIT_GATHER) &&
         emitter.argumentSchema == VulkanArgumentSchema::SINGLE_IARG;
}

inline bool usesIndexedLookupSchedule(
    const VulkanKernelEmitterInfo& emitter) {
  return usesAxisIndexedLookupSchedule(emitter) ||
         usesModeIndexedLookupSchedule(emitter);
}

inline bool usesMultiOutputPartitionSchedule(
    const VulkanKernelEmitterInfo& emitter) {
  return emitter.family == VulkanKernelFamily::DATA_MOVEMENT &&
         hasVulkanOpTrait(
             emitter, sd::ops::OP_TRAIT_DATA_MOVEMENT |
                          sd::ops::OP_TRAIT_FULLY_WRITING |
                          sd::ops::OP_TRAIT_SPLIT) &&
         emitter.argumentSchema == VulkanArgumentSchema::AXES_IARGS;
}

inline bool usesOutputShapeBroadcastSchedule(
    const VulkanKernelEmitterInfo& emitter) {
  return emitter.family == VulkanKernelFamily::DATA_MOVEMENT &&
         hasVulkanOpTrait(
             emitter, sd::ops::OP_TRAIT_DATA_MOVEMENT |
                          sd::ops::OP_TRAIT_TILE |
                          sd::ops::OP_TRAIT_FULLY_WRITING |
                          sd::ops::OP_TRAIT_VALUE_DEPENDENT_SHAPE) &&
         emitter.loweringContract == VulkanLoweringContract::DEFAULT &&
         emitter.argumentSchema == VulkanArgumentSchema::NONE &&
         vulkanInputIsStructuralIndex(emitter, 1);
}

inline bool usesAxisPartitionSchedule(
    const VulkanKernelEmitterInfo& emitter) {
  return emitter.family == VulkanKernelFamily::DATA_MOVEMENT &&
         hasVulkanOpTrait(emitter, sd::ops::OP_TRAIT_SPLIT_V) &&
         emitter.argumentSchema == VulkanArgumentSchema::SINGLE_IARG;
}

inline bool usesRowwiseEpsilonNormalizationSchedule(
    const VulkanKernelEmitterInfo& emitter) {
  return emitter.family == VulkanKernelFamily::NORMALIZATION &&
         hasVulkanOpTrait(
             emitter, sd::ops::OP_TRAIT_NORMALIZATION |
                          sd::ops::OP_TRAIT_FULLY_WRITING) &&
         emitter.loweringContract == VulkanLoweringContract::DEFAULT &&
         hasVulkanEmitterTrait(
             emitter, VULKAN_EMITTER_TRAIT_DISPATCH_FIRST_DIM |
                          VULKAN_EMITTER_TRAIT_EPSILON_PARAMETER) &&
         emitter.argumentSchema == VulkanArgumentSchema::EPSILON;
}

inline bool usesMultiOutputNormalizationSchedule(
    const VulkanKernelEmitterInfo& emitter) {
  return emitter.family == VulkanKernelFamily::NORMALIZATION &&
         hasVulkanOpTrait(
             emitter, sd::ops::OP_TRAIT_NORMALIZATION |
                          sd::ops::OP_TRAIT_FULLY_WRITING) &&
         emitter.loweringContract == VulkanLoweringContract::DEFAULT &&
         emitter.argumentSchema ==
             VulkanArgumentSchema::OPTIONAL_FINITE_SCALAR;
}

inline bool usesNormalizationGradientSchedule(
    const VulkanKernelEmitterInfo& emitter) {
  const bool supportedDispatch =
      hasVulkanEmitterTrait(
          emitter, VULKAN_EMITTER_TRAIT_DISPATCH_OUTER_ROWS) ||
      hasVulkanEmitterTrait(emitter, VULKAN_EMITTER_TRAIT_DISPATCH_SINGLE);
  return emitter.family == VulkanKernelFamily::NORMALIZATION &&
         hasVulkanOpTrait(
             emitter, sd::ops::OP_TRAIT_NORMALIZATION |
                          sd::ops::OP_TRAIT_FULLY_WRITING |
                          sd::ops::OP_TRAIT_BACKWARD) &&
         emitter.loweringContract == VulkanLoweringContract::DEFAULT &&
         hasVulkanEmitterTrait(
             emitter, VULKAN_EMITTER_TRAIT_EPSILON_PARAMETER) &&
         emitter.argumentSchema == VulkanArgumentSchema::EPSILON &&
         supportedDispatch;
}

inline bool usesBatchwiseNormalizationSchedule(
    const VulkanKernelEmitterInfo& emitter) {
  return emitter.family == VulkanKernelFamily::NORMALIZATION &&
         hasVulkanOpTrait(
             emitter, sd::ops::OP_TRAIT_NORMALIZATION |
                          sd::ops::OP_TRAIT_FULLY_WRITING) &&
         !hasVulkanOpTrait(emitter, sd::ops::OP_TRAIT_BACKWARD) &&
         emitter.loweringContract == VulkanLoweringContract::DEFAULT &&
         hasVulkanEmitterTrait(
             emitter, VULKAN_EMITTER_TRAIT_EPSILON_PARAMETER) &&
         !hasVulkanEmitterTrait(
             emitter, VULKAN_EMITTER_TRAIT_DISPATCH_FIRST_DIM) &&
         emitter.argumentSchema == VulkanArgumentSchema::EPSILON;
}

inline bool usesBroadcastBinarySchedule(
    const VulkanKernelEmitterInfo& emitter) {
  const bool intrinsicBroadcastSemantics =
      hasVulkanOpTrait(emitter, sd::ops::OP_TRAIT_ACTIVATION) ||
      vulkanArgumentContractAcceptsBooleanArguments(emitter);
  return emitter.family == VulkanKernelFamily::ELEMENTWISE_BINARY &&
         hasVulkanOpTrait(
             emitter, sd::ops::OP_TRAIT_BINARY_ELEMENTWISE |
                          sd::ops::OP_TRAIT_FULLY_WRITING) &&
         !hasVulkanOpTrait(emitter, sd::ops::OP_TRAIT_BACKWARD) &&
         emitter.loweringContract == VulkanLoweringContract::DEFAULT &&
         intrinsicBroadcastSemantics;
}

inline bool usesWindowTransformSchedule(
    const VulkanKernelEmitterInfo& emitter) {
  return emitter.family == VulkanKernelFamily::DATA_MOVEMENT &&
         hasVulkanOpTrait(
             emitter, sd::ops::OP_TRAIT_DATA_MOVEMENT |
                          sd::ops::OP_TRAIT_FULLY_WRITING) &&
         emitter.loweringContract == VulkanLoweringContract::DEFAULT &&
         emitter.argumentSchema ==
             VulkanArgumentSchema::WINDOW_GEOMETRY_IARGS;
}

inline bool usesCachedRotarySchedule(
    const VulkanKernelEmitterInfo& emitter) {
  return emitter.family == VulkanKernelFamily::DATA_MOVEMENT &&
         hasVulkanOpTrait(
             emitter, sd::ops::OP_TRAIT_DATA_MOVEMENT |
                          sd::ops::OP_TRAIT_FULLY_WRITING) &&
         emitter.loweringContract == VulkanLoweringContract::DEFAULT &&
         hasVulkanEmitterTrait(
             emitter, VULKAN_EMITTER_TRAIT_DISPATCH_ROTARY_GRID);
}

inline bool usesRankPermutationSchedule(
    const VulkanKernelEmitterInfo& emitter) {
  return emitter.family == VulkanKernelFamily::DATA_MOVEMENT &&
         emitter.argumentSchema == VulkanArgumentSchema::PERMUTATION_AXES;
}

inline bool usesDefaultReversePermutationSchedule(
    const VulkanKernelEmitterInfo& emitter) {
  return usesRankPermutationSchedule(emitter) &&
         hasVulkanOpTrait(
             emitter, sd::ops::OP_TRAIT_DATA_MOVEMENT |
                          sd::ops::OP_TRAIT_FULLY_WRITING) &&
         !hasVulkanOpTrait(emitter, sd::ops::OP_TRAIT_VIEW_PRODUCING);
}

inline bool usesExplicitPermutationSchedule(
    const VulkanKernelEmitterInfo& emitter) {
  return usesRankPermutationSchedule(emitter) &&
         hasVulkanOpTrait(emitter, sd::ops::OP_TRAIT_VIEW_PRODUCING);
}

inline bool usesReshapeCopySchedule(
    const VulkanKernelEmitterInfo& emitter) {
  return emitter.family == VulkanKernelFamily::DATA_MOVEMENT &&
         hasVulkanOpTrait(
             emitter, sd::ops::OP_TRAIT_VIEW_PRODUCING |
                          sd::ops::OP_TRAIT_DATA_MOVEMENT |
                          sd::ops::OP_TRAIT_FULLY_WRITING |
                          sd::ops::OP_TRAIT_VALUE_DEPENDENT_SHAPE |
                          sd::ops::OP_TRAIT_DATA_DEPENDENT) &&
         emitter.loweringContract == VulkanLoweringContract::LINEAR_COPY &&
         emitter.argumentSchema == VulkanArgumentSchema::NONE;
}

inline bool usesStructuralShapeCopySchedule(
    const VulkanKernelEmitterInfo& emitter) {
  return emitter.family == VulkanKernelFamily::DATA_MOVEMENT &&
         hasVulkanOpTrait(
             emitter, sd::ops::OP_TRAIT_DATA_MOVEMENT |
                          sd::ops::OP_TRAIT_FULLY_WRITING |
                          sd::ops::OP_TRAIT_VALUE_DEPENDENT_SHAPE) &&
         !usesReshapeCopySchedule(emitter) &&
         emitter.loweringContract == VulkanLoweringContract::LINEAR_COPY &&
         emitter.argumentSchema == VulkanArgumentSchema::NONE;
}

inline bool usesSameShapeCopySchedule(
    const VulkanKernelEmitterInfo& emitter) {
  const bool copySemantic =
      hasVulkanOpTrait(
          emitter, sd::ops::OP_TRAIT_DATA_MOVEMENT |
                       sd::ops::OP_TRAIT_FULLY_WRITING) ||
      hasVulkanOpTrait(
          emitter, sd::ops::OP_TRAIT_IDENTITY |
                       sd::ops::OP_TRAIT_FULLY_WRITING);
  return emitter.loweringContract == VulkanLoweringContract::LINEAR_COPY &&
         emitter.argumentSchema == VulkanArgumentSchema::NONE &&
         !hasVulkanOpTrait(
             emitter, sd::ops::OP_TRAIT_VALUE_DEPENDENT_SHAPE) &&
         copySemantic;
}

inline bool usesTrailingPayloadCopySchedule(
    const VulkanKernelEmitterInfo& emitter) {
  return usesSameShapeCopySchedule(emitter) &&
         hasVulkanOpTrait(emitter, sd::ops::OP_TRAIT_BACKWARD);
}

inline bool usesStructuredComputeSchedule(
    const VulkanKernelEmitterInfo& emitter) {
  return emitter.loweringContract == VulkanLoweringContract::FUSED_LLM ||
         emitter.loweringContract == VulkanLoweringContract::TRIANGULAR_SOLVE ||
         usesNormalizationGradientSchedule(emitter) ||
         usesBatchwiseNormalizationSchedule(emitter) ||
         usesBroadcastBinarySchedule(emitter) ||
         usesWindowTransformSchedule(emitter);
}

inline bool usesContractMovementSchedule(
    const VulkanKernelEmitterInfo& emitter) {
  return emitter.loweringContract == VulkanLoweringContract::LINEAR_COPY ||
         usesLinearConcatSchedule(emitter) ||
         usesOutputShapeBroadcastSchedule(emitter) ||
         usesAxisPartitionSchedule(emitter);
}

/** True when the descriptor freezes scalar values into the emitted kernel. */
inline bool hasVulkanScalarArgumentSchema(
    const VulkanKernelEmitterInfo& emitter) {
  if (emitter.argumentSchema == VulkanArgumentSchema::OPTIONAL_SCALAR ||
      emitter.argumentSchema == VulkanArgumentSchema::OPTIONAL_SCALAR_PAIR ||
      emitter.argumentSchema == VulkanArgumentSchema::OPTIONAL_FINITE_SCALAR) {
    return true;
  }
  for (uint8_t i = 0; i < emitter.argumentContract.alternativeCount; ++i) {
    const auto& range = emitter.argumentContract.alternatives[i].tArgs;
    if (range.minimum > 0 || range.maximum < 0 || range.maximum > 0) {
      return true;
    }
  }
  return false;
}

/** Find the canonical emitter registration for a descriptor hash. */
SD_LIB_EXPORT const VulkanKernelEmitterInfo* findVulkanKernelEmitter(
    sd::LongType descriptorHash);

/** Stable catalogue used by coverage tests and diagnostics. */
SD_LIB_EXPORT const std::vector<VulkanKernelEmitterInfo>&
getVulkanKernelEmitterCatalog();

/** Find emitter metadata for a canonical typed NativeOpExecutioner identity. */
SD_LIB_EXPORT const VulkanKernelEmitterInfo*
findVulkanLegacyKernelEmitter(VulkanLegacyOpFamily family, int opNum);

/** Stable typed-legacy emitter catalogue used by coverage tests. */
SD_LIB_EXPORT const std::vector<VulkanKernelEmitterInfo>&
getVulkanLegacyKernelEmitterCatalog();

}  // namespace graph
}  // namespace sd

#endif  // HAVE_VULKAN
#endif  // LIBND4J_VULKAN_KERNEL_EMITTER_CATALOG_H
