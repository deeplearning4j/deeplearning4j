/* ******************************************************************************
 *
 * Copyright (c) 2026 Eclipse Foundation
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <graph/vulkan/VulkanKernelEmitterCatalog.h>

#if defined(HAVE_VULKAN) && HAVE_VULKAN

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/LegacyRandomOp.h>
#include <ops/declarable/LegacyReduceFloatOp.h>
#include <ops/declarable/LegacyReduceSameOp.h>
#include <ops/declarable/LegacyScalarOp.h>
#include <ops/declarable/LegacyTransformAnyOp.h>
#include <ops/declarable/LegacyTransformFloatOp.h>
#include <ops/declarable/LegacyTransformSameOp.h>
#include <ops/declarable/LegacyTransformStrictOp.h>
#include <ops/declarable/headers/legacy_indexing.h>
#include <ops/declarable/headers/llm.h>
#include <system/op_boilerplate.h>
#include <system/op_enums.h>

#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace sd {
namespace graph {
namespace {

constexpr uint32_t kFloat =
    VULKAN_DTYPE_FLOAT;
constexpr uint32_t kNumeric32 =
    VULKAN_DTYPE_FLOAT | VULKAN_DTYPE_SIGNED_INT32 |
    VULKAN_DTYPE_UNSIGNED_INT32;
constexpr uint32_t kNumeric32WithBool =
    kNumeric32 | VULKAN_DTYPE_BOOL;
constexpr uint32_t kNumericWithIndex =
    kNumeric32 | VULKAN_DTYPE_INDEX;
// Structural shape/sizes operands may use the framework's wider integer dtype.
// Contract lowerings bind but never load them, while payload/output storage is
// still checked separately against actual Vulkan scalar support.
constexpr uint32_t kNumeric32WithStructuralIndex = kNumericWithIndex;
constexpr uint32_t kDenseLayout =
    VULKAN_LAYOUT_DENSE_C | VULKAN_LAYOUT_DENSE_F;
constexpr uint32_t kGeneralLayout =
    VULKAN_LAYOUT_DENSE_C | VULKAN_LAYOUT_DENSE_F |
    VULKAN_LAYOUT_STRIDED_VIEW | VULKAN_LAYOUT_BROADCAST;
constexpr uint32_t kStridedLayout =
    VULKAN_LAYOUT_DENSE_C | VULKAN_LAYOUT_DENSE_F |
    VULKAN_LAYOUT_STRIDED_VIEW;

constexpr VulkanArgumentCountRange exactCount(int16_t count) {
  return VulkanArgumentCountRange{count, count};
}

constexpr VulkanArgumentCountRange countRange(
    int16_t minimum, int16_t maximum) {
  return VulkanArgumentCountRange{minimum, maximum};
}

constexpr VulkanArgumentSignature exactArguments(
    int16_t inputs, int16_t outputs, int16_t tArgs, int16_t iArgs,
    int16_t bArgs, int16_t dArgs = 0, int16_t sArgs = 0) {
  return VulkanArgumentSignature{
      exactCount(inputs), exactCount(outputs), exactCount(tArgs),
      exactCount(iArgs), exactCount(bArgs), exactCount(dArgs),
      exactCount(sArgs)};
}

constexpr VulkanArgumentSignature rangedArguments(
    int16_t minimumInputs, int16_t maximumInputs,
    int16_t minimumOutputs, int16_t maximumOutputs,
    int16_t minimumTArgs, int16_t maximumTArgs,
    int16_t minimumIArgs, int16_t maximumIArgs,
    int16_t minimumBArgs, int16_t maximumBArgs,
    int16_t minimumDArgs = 0, int16_t maximumDArgs = 0,
    int16_t minimumSArgs = 0, int16_t maximumSArgs = 0) {
  return VulkanArgumentSignature{
      countRange(minimumInputs, maximumInputs),
      countRange(minimumOutputs, maximumOutputs),
      countRange(minimumTArgs, maximumTArgs),
      countRange(minimumIArgs, maximumIArgs),
      countRange(minimumBArgs, maximumBArgs),
      countRange(minimumDArgs, maximumDArgs),
      countRange(minimumSArgs, maximumSArgs)};
}

constexpr VulkanArgumentContract singleArguments(
    const VulkanArgumentSignature& signature,
    uint8_t valueTraits = VULKAN_ARGUMENT_VALUES_NONE) {
  return VulkanArgumentContract{{signature, {}}, 1, valueTraits};
}

constexpr VulkanArgumentContract eitherArguments(
    const VulkanArgumentSignature& first,
    const VulkanArgumentSignature& second,
    uint8_t valueTraits = VULKAN_ARGUMENT_VALUES_NONE) {
  return VulkanArgumentContract{{first, second}, 2, valueTraits};
}

constexpr VulkanArgumentContract movementArguments(
    int16_t inputs, int16_t minimumIArgs, int16_t maximumIArgs,
    int16_t minimumBArgs = 0, int16_t maximumBArgs = 0) {
  return singleArguments(rangedArguments(
      inputs, inputs, 1, 1, 0, 0, minimumIArgs, maximumIArgs,
      minimumBArgs, maximumBArgs));
}

constexpr VulkanOperandTypeContract mixedOperandTypes(
    uint16_t scalar32InputMask, uint16_t integer32InputMask,
    bool requireUniformSpecialInputs = false) {
  return VulkanOperandTypeContract{
      scalar32InputMask, integer32InputMask, 0,
      requireUniformSpecialInputs, 0};
}

constexpr VulkanOperandTypeContract structuralIndexInputs(
    uint16_t structuralIndexInputMask) {
  return VulkanOperandTypeContract{
      0, 0, 0, false, structuralIndexInputMask};
}

constexpr VulkanOperandTypeContract loadedOperandTypes(
    uint16_t scalar32InputMask, uint16_t integer32InputMask,
    uint16_t integer64InputMask) {
  return VulkanOperandTypeContract{
      scalar32InputMask, integer32InputMask, integer64InputMask, false, 0, 0};
}

constexpr VulkanOperandTypeContract loadedIndexOperandTypes(
    uint16_t integerIndexInputMask) {
  return VulkanOperandTypeContract{
      0, 0, 0, false, 0, integerIndexInputMask};
}

template <typename Op>
void registerEmitter(
    std::vector<VulkanKernelEmitterInfo>& result,
    VulkanKernelRecipe recipe,
    uint32_t dtypeSupport,
    uint32_t layoutSupport,
    int16_t minimumRank = 0,
    int16_t maximumRank = -1,
    uint32_t emitterTraits = VULKAN_EMITTER_TRAIT_NONE,
    VulkanArgumentSchema argumentSchema = VulkanArgumentSchema::NONE,
    VulkanLoweringContract loweringContract = VulkanLoweringContract::DEFAULT,
    VulkanArgumentContract argumentContract = {},
    VulkanOperandTypeContract operandTypeContract = {}) {
  static Op op;
  op.initializeDescriptor();
  auto* descriptor = op.getOpDescriptor();
  VulkanKernelEmitterInfo info{
      descriptor->getHash(), descriptor->getTraits(), recipe,
      VulkanKernelFamily::UNKNOWN, loweringContract, argumentSchema,
      emitterTraits, dtypeSupport, layoutSupport,
      static_cast<int16_t>(descriptor->getNumberOfInputs()),
      static_cast<int16_t>(descriptor->getNumberOfOutputs()),
      static_cast<int16_t>(descriptor->getNumberOfTArgs()),
      static_cast<int16_t>(descriptor->getNumberOfIArgs()),
      argumentContract, operandTypeContract, minimumRank, maximumRank};
  info.family = vulkanKernelFamily(info);
  result.push_back(info);
}

template <typename Op>
void registerFusedEmitter(
    std::vector<VulkanKernelEmitterInfo>& result,
    VulkanKernelRecipe recipe,
    uint32_t dtypeSupport,
    uint32_t layoutSupport,
    int16_t minimumRank = 0,
    int16_t maximumRank = -1,
    uint32_t emitterTraits = VULKAN_EMITTER_TRAIT_NONE,
    VulkanArgumentSchema argumentSchema = VulkanArgumentSchema::NONE,
    VulkanArgumentContract argumentContract = {},
    VulkanOperandTypeContract operandTypeContract = {}) {
  registerEmitter<Op>(
      result, recipe, dtypeSupport, layoutSupport, minimumRank, maximumRank,
      emitterTraits, argumentSchema, VulkanLoweringContract::FUSED_LLM,
      argumentContract, operandTypeContract);
}

struct LegacyEmitterCatalogData {
  std::vector<VulkanKernelEmitterInfo> entries;
  std::unordered_map<VulkanLegacyOpKey, size_t, VulkanLegacyOpKeyHash> indices;
};

template <typename LegacyOp>
void registerLegacyEmitter(
    LegacyEmitterCatalogData& catalog, VulkanLegacyOpFamily family, int opNum,
    VulkanKernelRecipe recipe, uint32_t dtypeSupport, uint32_t layoutSupport,
    int16_t minimumRank = 0, int16_t maximumRank = -1,
    uint32_t emitterTraits = VULKAN_EMITTER_TRAIT_NONE,
    VulkanArgumentSchema argumentSchema = VulkanArgumentSchema::NONE,
    VulkanLoweringContract loweringContract = VulkanLoweringContract::DEFAULT,
    VulkanArgumentContract argumentContract = {}) {
  if (VulkanLegacyOpCatalog::lookup(family, opNum) == nullptr) {
    throw std::logic_error(
        "typed Vulkan emitter does not name a canonical legacy operation");
  }

  LegacyOp op(opNum);
  auto* descriptor = op.getOpDescriptor();
  VulkanKernelEmitterInfo info{
      0, descriptor->getTraits(), recipe, VulkanKernelFamily::UNKNOWN,
      loweringContract, argumentSchema, emitterTraits,
      dtypeSupport, layoutSupport,
      static_cast<int16_t>(descriptor->getNumberOfInputs()),
      static_cast<int16_t>(descriptor->getNumberOfOutputs()),
      static_cast<int16_t>(descriptor->getNumberOfTArgs()),
      static_cast<int16_t>(descriptor->getNumberOfIArgs()),
      argumentContract, {}, minimumRank, maximumRank};
  info.family = vulkanKernelFamily(info);

  const VulkanLegacyOpKey key(family, opNum);
  const size_t index = catalog.entries.size();
  if (!catalog.indices.emplace(key, index).second) {
    throw std::logic_error(
        "duplicate canonical typed identity in Vulkan emitter catalog");
  }
  catalog.entries.emplace_back(std::move(info));
}

const std::vector<VulkanKernelEmitterInfo>& buildCatalog() {
  static const std::vector<VulkanKernelEmitterInfo> catalog = [] {
    std::vector<VulkanKernelEmitterInfo> result;

#if NOT_EXCLUDED(OP_matmul)
    registerEmitter<sd::ops::matmul>(
        result, VulkanKernelRecipe::MATMUL,
        kFloat, kStridedLayout, 2, 3);
#endif
#if NOT_EXCLUDED(OP_triangular_solve)
    registerEmitter<sd::ops::triangular_solve>(
        result, VulkanKernelRecipe::TRIANGULAR_SOLVE,
        kFloat, kStridedLayout, 2, -1,
        VULKAN_EMITTER_TRAIT_BOOLEAN_PARAMETERS,
        VulkanArgumentSchema::NONE,
        VulkanLoweringContract::TRIANGULAR_SOLVE,
        singleArguments(rangedArguments(2, 2, 1, 1, 0, 0, 0, 0, 0, 2)));
#endif
#if NOT_EXCLUDED(OP_batched_gemm)
    registerEmitter<sd::ops::batched_gemm>(
        result, VulkanKernelRecipe::MATMUL,
        kFloat, kStridedLayout, 0, 1,
        VULKAN_EMITTER_TRAIT_BOOLEAN_PARAMETERS);
#endif
#if NOT_EXCLUDED(OP_rms_norm)
    registerEmitter<sd::ops::rms_norm>(
        result, VulkanKernelRecipe::RMS_NORM,
         kFloat, kStridedLayout, 2, 2,
         VULKAN_EMITTER_TRAIT_DISPATCH_FIRST_DIM |
                   VULKAN_EMITTER_TRAIT_EPSILON_PARAMETER,
        VulkanArgumentSchema::EPSILON);
#endif
#if NOT_EXCLUDED(OP_rms_norm)
    registerEmitter<sd::ops::rms_norm_bp>(
        result, VulkanKernelRecipe::RMS_NORM_BP,
        kFloat, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_DISPATCH_OUTER_ROWS |
            VULKAN_EMITTER_TRAIT_EPSILON_PARAMETER,
        VulkanArgumentSchema::EPSILON, VulkanLoweringContract::DEFAULT,
        singleArguments(
            rangedArguments(2, 2, 1, 1, 0, 1, 0, 0, 0, 0),
            VULKAN_ARGUMENT_VALUES_FINITE_TARGS));
#endif
#if NOT_EXCLUDED(OP_fused_rope)
    registerEmitter<sd::ops::fused_rope>(
        result, VulkanKernelRecipe::FUSED_ROPE,
        kFloat, kStridedLayout, 4, 4,
        VULKAN_EMITTER_TRAIT_DISPATCH_ROTARY_GRID,
        VulkanArgumentSchema::NONE);
    registerFusedEmitter<sd::ops::fused_rope_bp>(
        result, VulkanKernelRecipe::ROPE_BP,
        kFloat, kStridedLayout, 3, 4, VULKAN_EMITTER_TRAIT_NONE,
        VulkanArgumentSchema::NONE,
        singleArguments(
            rangedArguments(2, 2, 1, 1, 0, 2, 0, 3, 0, 0),
            VULKAN_ARGUMENT_VALUES_FINITE_TARGS |
                VULKAN_ARGUMENT_VALUES_POSITIVE_TARG_ZERO));
#endif
#if NOT_EXCLUDED(OP_rope)
    registerFusedEmitter<sd::ops::rope>(
        result, VulkanKernelRecipe::ROPE,
        kFloat, kStridedLayout, 3, 4, VULKAN_EMITTER_TRAIT_NONE,
        VulkanArgumentSchema::NONE,
        singleArguments(
            rangedArguments(1, 1, 1, 1, 0, 2, 0, 3, 0, 0),
            VULKAN_ARGUMENT_VALUES_FINITE_TARGS |
                VULKAN_ARGUMENT_VALUES_POSITIVE_TARG_ZERO));
    registerFusedEmitter<sd::ops::rope_bp>(
        result, VulkanKernelRecipe::ROPE_BP,
        kFloat, kStridedLayout, 3, 4, VULKAN_EMITTER_TRAIT_NONE,
        VulkanArgumentSchema::NONE,
        singleArguments(
            rangedArguments(2, 2, 1, 1, 0, 2, 0, 3, 0, 0),
            VULKAN_ARGUMENT_VALUES_FINITE_TARGS |
                VULKAN_ARGUMENT_VALUES_POSITIVE_TARG_ZERO));
#endif
#if NOT_EXCLUDED(OP_skip_rms_norm)
    registerFusedEmitter<sd::ops::skip_rms_norm>(
        result, VulkanKernelRecipe::SKIP_RMS_NORM,
        kFloat, kStridedLayout, 2, 2,
        VULKAN_EMITTER_TRAIT_DISPATCH_FIRST_DIM |
            VULKAN_EMITTER_TRAIT_EPSILON_PARAMETER,
        VulkanArgumentSchema::NONE,
        singleArguments(
            rangedArguments(3, 4, 1, 2, 0, 1, 0, 0, 0, 0),
            VULKAN_ARGUMENT_VALUES_FINITE_TARGS));
#endif
#if NOT_EXCLUDED(OP_rms_norm_linear)
    registerFusedEmitter<sd::ops::rms_norm_linear>(
        result, VulkanKernelRecipe::RMS_NORM_LINEAR,
        kFloat, kStridedLayout, 2, 2,
        VULKAN_EMITTER_TRAIT_DISPATCH_FIRST_DIM |
            VULKAN_EMITTER_TRAIT_EPSILON_PARAMETER,
        VulkanArgumentSchema::NONE,
        singleArguments(
            rangedArguments(3, 3, 1, 1, 0, 1, 0, 0, 0, 0),
            VULKAN_ARGUMENT_VALUES_FINITE_TARGS));
#endif
#if NOT_EXCLUDED(OP_fused_gemm_swiglu)
    registerFusedEmitter<sd::ops::fused_gemm_swiglu>(
        result, VulkanKernelRecipe::FUSED_GEMM_SWIGLU,
        kFloat, kStridedLayout, 2, 2,
        VULKAN_EMITTER_TRAIT_DISPATCH_FIRST_DIM,
        VulkanArgumentSchema::NONE,
        singleArguments(exactArguments(3, 1, 0, 0, 0)));
#endif
#if NOT_EXCLUDED(OP_fused_rms_norm_swiglu)
    registerFusedEmitter<sd::ops::fused_rms_norm_swiglu>(
        result, VulkanKernelRecipe::FUSED_RMS_NORM_SWIGLU,
        kFloat, kStridedLayout, 3, 3,
        VULKAN_EMITTER_TRAIT_DISPATCH_BATCH_SEQUENCE |
            VULKAN_EMITTER_TRAIT_EPSILON_PARAMETER,
        VulkanArgumentSchema::NONE,
        singleArguments(
            rangedArguments(4, 4, 1, 1, 0, 1, 0, 0, 0, 0),
            VULKAN_ARGUMENT_VALUES_FINITE_TARGS));
#endif
#if NOT_EXCLUDED(OP_dot_product_attention)
    registerFusedEmitter<sd::ops::dot_product_attention>(
        result, VulkanKernelRecipe::DOT_PRODUCT_ATTENTION,
        kFloat, kStridedLayout, 3, 4,
        VULKAN_EMITTER_TRAIT_DISPATCH_ATTENTION_QUERY,
        VulkanArgumentSchema::NONE,
        singleArguments(rangedArguments(3, 4, 1, 2, 0, 0, 2, 2, 0, 0)));
#endif
#if NOT_EXCLUDED(OP_flash_attention)
    registerFusedEmitter<sd::ops::flash_attention>(
        result, VulkanKernelRecipe::FLASH_ATTENTION,
        kFloat, kStridedLayout, 3, 4,
        VULKAN_EMITTER_TRAIT_BOOLEAN_PARAMETERS,
        VulkanArgumentSchema::NONE,
        singleArguments(
            rangedArguments(3, 3, 1, 1, 0, 1, 0, 2, 0, 1),
            VULKAN_ARGUMENT_VALUES_FINITE_TARGS));
#endif
#if NOT_EXCLUDED(OP_grouped_query_attention)
    registerFusedEmitter<sd::ops::grouped_query_attention>(
        result, VulkanKernelRecipe::GROUPED_QUERY_ATTENTION,
        kFloat, kStridedLayout, 3, 4,
        VULKAN_EMITTER_TRAIT_BOOLEAN_PARAMETERS,
        VulkanArgumentSchema::NONE,
        singleArguments(
            rangedArguments(3, 3, 1, 1, 0, 1, 0, 2, 0, 1),
            VULKAN_ARGUMENT_VALUES_FINITE_TARGS));
#endif
#if NOT_EXCLUDED(OP_apply_alibi)
    registerFusedEmitter<sd::ops::apply_alibi>(
        result, VulkanKernelRecipe::APPLY_ALIBI,
        kFloat, kStridedLayout, 4, 4, VULKAN_EMITTER_TRAIT_NONE,
        VulkanArgumentSchema::NONE,
        singleArguments(rangedArguments(1, 1, 1, 1, 0, 0, 0, 1, 0, 0)));
#endif
#if NOT_EXCLUDED(OP_fused_attention_projection)
    registerFusedEmitter<sd::ops::fused_attention_projection>(
        result, VulkanKernelRecipe::FUSED_ATTENTION_PROJECTION,
        kFloat, kStridedLayout, 3, 4, VULKAN_EMITTER_TRAIT_NONE,
        VulkanArgumentSchema::NONE,
        singleArguments(rangedArguments(2, 3, 1, 1, 0, 0, 0, 0, 0, 0)));
#endif
#if NOT_EXCLUDED(OP_fused_elementwise_chain)
    registerFusedEmitter<sd::ops::fused_elementwise_chain>(
        result, VulkanKernelRecipe::FUSED_ELEMENTWISE_CHAIN,
        kFloat, kStridedLayout, 0, -1, VULKAN_EMITTER_TRAIT_NONE,
        VulkanArgumentSchema::NONE,
        singleArguments(
            rangedArguments(1, 9, 1, 1, 0, 2, 1, 8, 0, 0),
            VULKAN_ARGUMENT_VALUES_FINITE_TARGS));
#endif
#if NOT_EXCLUDED(OP_fused_bias_dropout_residual)
    registerFusedEmitter<sd::ops::fused_bias_dropout_residual>(
        result, VulkanKernelRecipe::FUSED_BIAS_DROPOUT_RESIDUAL,
        kFloat, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_BOOLEAN_PARAMETERS,
        VulkanArgumentSchema::NONE,
        singleArguments(
            rangedArguments(3, 3, 1, 1, 0, 1, 0, 1, 0, 1),
            VULKAN_ARGUMENT_VALUES_FINITE_TARGS |
                VULKAN_ARGUMENT_VALUES_PROBABILITY_TARG_ZERO |
                VULKAN_ARGUMENT_VALUES_NONZERO_TARG_REQUIRES_FALSE_BARG));
#endif
#if NOT_EXCLUDED(OP_swish_mul)
    registerFusedEmitter<sd::ops::swish_mul_bp>(
        result, VulkanKernelRecipe::SWISH_MUL_BP,
        kFloat, kStridedLayout, 0, -1, VULKAN_EMITTER_TRAIT_NONE,
        VulkanArgumentSchema::NONE,
        singleArguments(exactArguments(3, 2, 0, 0, 0)));
#endif
#if NOT_EXCLUDED(OP_fused_mrope)
    registerFusedEmitter<sd::ops::fused_mrope>(
        result, VulkanKernelRecipe::FUSED_MROPE,
        kNumeric32, kStridedLayout, 4, 4,
        VULKAN_EMITTER_TRAIT_DISPATCH_ROTARY_GRID,
        VulkanArgumentSchema::NONE,
        singleArguments(
            rangedArguments(4, 4, 1, 1, 0, 1, 0, 4, 0, 0),
            VULKAN_ARGUMENT_VALUES_FINITE_TARGS),
        mixedOperandTypes(0x0Eu, 0, true));
#endif
#if NOT_EXCLUDED(OP_vision_embedding_merge)
    registerFusedEmitter<sd::ops::vision_embedding_merge>(
        result, VulkanKernelRecipe::VISION_EMBEDDING_MERGE,
        kNumeric32, kStridedLayout, 3, 3,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::NONE,
        singleArguments(exactArguments(3, 1, 0, 1, 0)),
        mixedOperandTypes(0, 0x04u));
#endif
#if NOT_EXCLUDED(OP_swiglu)
    registerFusedEmitter<sd::ops::swiglu>(
        result, VulkanKernelRecipe::SWIGLU,
        kFloat, kStridedLayout, 1, -1, VULKAN_EMITTER_TRAIT_NONE,
        VulkanArgumentSchema::NONE,
        singleArguments(exactArguments(1, 1, 0, 0, 0)));
#endif
#if NOT_EXCLUDED(OP_geglu)
    registerFusedEmitter<sd::ops::geglu>(
        result, VulkanKernelRecipe::GEGLU,
        kFloat, kStridedLayout, 1, -1, VULKAN_EMITTER_TRAIT_NONE,
        VulkanArgumentSchema::NONE,
        singleArguments(exactArguments(1, 1, 0, 0, 0)));
#endif
#if NOT_EXCLUDED(OP_reglu)
    registerFusedEmitter<sd::ops::reglu>(
        result, VulkanKernelRecipe::REGLU,
        kFloat, kStridedLayout, 1, -1, VULKAN_EMITTER_TRAIT_NONE,
        VulkanArgumentSchema::NONE,
        singleArguments(exactArguments(1, 1, 0, 0, 0)));
#endif

#if NOT_EXCLUDED(OP_add)
    registerEmitter<sd::ops::add>(
        result, VulkanKernelRecipe::ADD,
         kNumeric32, kGeneralLayout);
#endif
#if NOT_EXCLUDED(OP_add_inplace)
    registerEmitter<sd::ops::add_inplace>(
        result, VulkanKernelRecipe::ADD,
         kFloat, kStridedLayout);
#endif
#if NOT_EXCLUDED(OP_subtract)
    registerEmitter<sd::ops::subtract>(
        result, VulkanKernelRecipe::SUBTRACT,
         kNumeric32, kGeneralLayout);
#endif
#if NOT_EXCLUDED(OP_multiply)
    registerEmitter<sd::ops::multiply>(
        result, VulkanKernelRecipe::MULTIPLY,
         kNumeric32, kGeneralLayout);
#endif
#if NOT_EXCLUDED(OP_divide)
    registerEmitter<sd::ops::divide>(
        result, VulkanKernelRecipe::DIVIDE,
         kNumeric32, kGeneralLayout);
#endif
#if NOT_EXCLUDED(OP_realdiv)
    registerEmitter<sd::ops::realdiv>(
        result, VulkanKernelRecipe::DIVIDE,
         kNumeric32, kGeneralLayout);
#endif
#if NOT_EXCLUDED(OP_minimum)
    registerEmitter<sd::ops::minimum>(
        result, VulkanKernelRecipe::MINIMUM,
         kNumeric32, kGeneralLayout);
#endif
#if NOT_EXCLUDED(OP_maximum)
    registerEmitter<sd::ops::maximum>(
        result, VulkanKernelRecipe::MAXIMUM,
         kNumeric32, kGeneralLayout);
#endif
#if NOT_EXCLUDED(OP_floormod)
    registerEmitter<sd::ops::floormod>(
        result, VulkanKernelRecipe::FLOOR_MOD,
         kNumeric32, kGeneralLayout);
#endif
#if NOT_EXCLUDED(OP_mod)
    registerEmitter<sd::ops::mod>(
        result, VulkanKernelRecipe::MOD,
         kNumeric32, kGeneralLayout);
#endif
#if NOT_EXCLUDED(OP_floordiv)
    registerEmitter<sd::ops::floordiv>(
        result, VulkanKernelRecipe::FLOOR_DIVIDE,
         kNumeric32, kGeneralLayout);
#endif
#if NOT_EXCLUDED(OP_reversedivide)
    registerEmitter<sd::ops::reversedivide>(
        result, VulkanKernelRecipe::REVERSE_DIVIDE,
         kNumeric32, kGeneralLayout);
#endif
#if NOT_EXCLUDED(OP_reversesubtract)
    registerEmitter<sd::ops::reversesubtract>(
        result, VulkanKernelRecipe::REVERSE_SUBTRACT,
         kNumeric32, kGeneralLayout);
#endif
#if NOT_EXCLUDED(OP_squaredsubtract)
    registerEmitter<sd::ops::squaredsubtract>(
        result, VulkanKernelRecipe::SQUARED_SUBTRACT,
         kNumeric32, kGeneralLayout);
#endif
#if NOT_EXCLUDED(OP_Pow)
    registerEmitter<sd::ops::Pow>(
        result, VulkanKernelRecipe::POWER,
         kFloat, kGeneralLayout);
#endif
#if NOT_EXCLUDED(OP_tf_atan2)
    registerEmitter<sd::ops::tf_atan2>(
        result, VulkanKernelRecipe::ATAN2,
         kFloat, kGeneralLayout);
#endif
#if NOT_EXCLUDED(OP_swish_mul)
    registerEmitter<sd::ops::swish_mul>(
        result, VulkanKernelRecipe::SWISH_MUL,
         kFloat, kStridedLayout,
        0, -1);
#endif
#if NOT_EXCLUDED(OP_silu_and_mul)
    registerEmitter<sd::ops::silu_and_mul>(
        result, VulkanKernelRecipe::SWISH_MUL,
         kFloat, kStridedLayout,
        0, -1);
#endif
#if NOT_EXCLUDED(OP_biasadd)
    registerEmitter<sd::ops::biasadd>(
        result, VulkanKernelRecipe::BIAS_ADD,
        kNumeric32, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_BOOLEAN_PARAMETERS,
        VulkanArgumentSchema::NONE, VulkanLoweringContract::DEFAULT,
        singleArguments(rangedArguments(2, 2, 1, 1, 0, 0, 0, 0, 0, 1)));
#endif
#if NOT_EXCLUDED(OP_assign)
    registerEmitter<sd::ops::assign>(
        result, VulkanKernelRecipe::ASSIGN,
        kNumericWithIndex, kGeneralLayout);
#endif

#if NOT_EXCLUDED(OP_silu)
    registerEmitter<sd::ops::silu>(
        result, VulkanKernelRecipe::SILU,
         kFloat, kStridedLayout);
    registerEmitter<sd::ops::silu_bp>(
        result, VulkanKernelRecipe::SILU_BP,
         kFloat, kStridedLayout);
#endif
#if NOT_EXCLUDED(OP_fused_gelu)
    registerEmitter<sd::ops::fused_gelu>(
        result, VulkanKernelRecipe::FAST_GELU,
         kFloat, kStridedLayout,
        0, -1);
    registerEmitter<sd::ops::fused_gelu_bp>(
        result, VulkanKernelRecipe::FUSED_GELU_BP,
         kFloat, kStridedLayout,
        0, -1);
#endif
#if NOT_EXCLUDED(OP_tanh)
    registerEmitter<sd::ops::tanh>(
        result, VulkanKernelRecipe::TANH,
         kFloat, kStridedLayout);
    registerEmitter<sd::ops::tanh_bp>(
        result, VulkanKernelRecipe::TANH_BP,
         kFloat, kStridedLayout);
#endif
#if NOT_EXCLUDED(OP_sigmoid)
    registerEmitter<sd::ops::sigmoid>(
        result, VulkanKernelRecipe::SIGMOID,
         kFloat, kStridedLayout);
    registerEmitter<sd::ops::sigmoid_bp>(
        result, VulkanKernelRecipe::SIGMOID_BP,
         kFloat, kStridedLayout);
#endif
#if NOT_EXCLUDED(OP_relu)
    registerEmitter<sd::ops::relu>(
        result, VulkanKernelRecipe::RELU,
         kNumeric32, kStridedLayout,
        0, -1, VULKAN_EMITTER_TRAIT_NONE,
        VulkanArgumentSchema::OPTIONAL_SCALAR);
    registerEmitter<sd::ops::relu_bp>(
        result, VulkanKernelRecipe::RELU_BP,
         kFloat, kStridedLayout);
#endif
#if NOT_EXCLUDED(OP_square)
    registerEmitter<sd::ops::square>(
        result, VulkanKernelRecipe::SQUARE,
         kNumeric32, kStridedLayout);
#endif
#if NOT_EXCLUDED(OP_squared_relu)
    registerEmitter<sd::ops::squared_relu>(
        result, VulkanKernelRecipe::SQUARED_RELU,
         kFloat, kStridedLayout);
    registerEmitter<sd::ops::squared_relu_bp>(
        result, VulkanKernelRecipe::SQUARED_RELU_BP,
         kFloat, kStridedLayout);
#endif
#if NOT_EXCLUDED(OP_hardsigmoid)
    registerEmitter<sd::ops::hardsigmoid>(
        result, VulkanKernelRecipe::HARD_SIGMOID,
         kFloat, kStridedLayout);
    registerEmitter<sd::ops::hardsigmoid_bp>(
        result, VulkanKernelRecipe::HARD_SIGMOID_BP,
         kFloat, kStridedLayout);
#endif
#if NOT_EXCLUDED(OP_hardtanh)
    registerEmitter<sd::ops::hardtanh>(
        result, VulkanKernelRecipe::HARD_TANH,
         kFloat, kStridedLayout);
    registerEmitter<sd::ops::hardtanh_bp>(
        result, VulkanKernelRecipe::HARD_TANH_BP,
         kFloat, kStridedLayout);
#endif
#if NOT_EXCLUDED(OP_relu6)
    registerEmitter<sd::ops::relu6>(
        result, VulkanKernelRecipe::RELU6,
         kNumeric32, kStridedLayout,
        0, -1, VULKAN_EMITTER_TRAIT_NONE,
        VulkanArgumentSchema::OPTIONAL_SCALAR);
    registerEmitter<sd::ops::relu6_bp>(
        result, VulkanKernelRecipe::RELU6_BP,
         kFloat, kStridedLayout);
#endif
#if NOT_EXCLUDED(OP_lrelu)
    registerEmitter<sd::ops::lrelu>(
        result, VulkanKernelRecipe::LEAKY_RELU,
         kFloat, kStridedLayout,
        0, -1, VULKAN_EMITTER_TRAIT_NONE,
        VulkanArgumentSchema::OPTIONAL_SCALAR);
    registerEmitter<sd::ops::lrelu_bp>(
        result, VulkanKernelRecipe::LEAKY_RELU_BP,
         kFloat, kStridedLayout,
        0, -1, VULKAN_EMITTER_TRAIT_NONE,
        VulkanArgumentSchema::OPTIONAL_SCALAR);
#endif
#if NOT_EXCLUDED(OP_elu)
    registerEmitter<sd::ops::elu>(
        result, VulkanKernelRecipe::ELU,
         kFloat, kStridedLayout,
        0, -1, VULKAN_EMITTER_TRAIT_NONE,
        VulkanArgumentSchema::OPTIONAL_SCALAR);
    registerEmitter<sd::ops::elu_bp>(
        result, VulkanKernelRecipe::ELU_BP,
         kFloat, kStridedLayout,
        0, -1, VULKAN_EMITTER_TRAIT_NONE,
        VulkanArgumentSchema::OPTIONAL_SCALAR);
#endif
#if NOT_EXCLUDED(OP_selu)
    registerEmitter<sd::ops::selu>(
        result, VulkanKernelRecipe::SELU,
         kFloat, kStridedLayout);
    registerEmitter<sd::ops::selu_bp>(
        result, VulkanKernelRecipe::SELU_BP,
         kFloat, kStridedLayout);
#endif
#if NOT_EXCLUDED(OP_softplus)
    registerEmitter<sd::ops::softplus>(
        result, VulkanKernelRecipe::SOFTPLUS,
         kFloat, kStridedLayout);
    registerEmitter<sd::ops::softplus_bp>(
        result, VulkanKernelRecipe::SOFTPLUS_BP,
         kFloat, kStridedLayout);
#endif
#if NOT_EXCLUDED(OP_softsign)
    registerEmitter<sd::ops::softsign>(
        result, VulkanKernelRecipe::SOFTSIGN,
         kFloat, kStridedLayout);
    registerEmitter<sd::ops::softsign_bp>(
        result, VulkanKernelRecipe::SOFTSIGN_BP,
         kFloat, kStridedLayout);
#endif
#if NOT_EXCLUDED(OP_identity)
    registerEmitter<sd::ops::identity>(
        result, VulkanKernelRecipe::COPY,
        kNumeric32, kGeneralLayout, 0, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::NONE,
        VulkanLoweringContract::LINEAR_COPY);
    registerEmitter<sd::ops::identity_bp>(
        result, VulkanKernelRecipe::COPY,
        kNumeric32, kGeneralLayout, 0, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::NONE,
        VulkanLoweringContract::LINEAR_COPY);
#endif
#if NOT_EXCLUDED(OP_identity_n)
    registerEmitter<sd::ops::identity_n>(
        result, VulkanKernelRecipe::COPY,
        kNumeric32, kGeneralLayout, 0, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::NONE,
        VulkanLoweringContract::LINEAR_COPY);
#endif
#if NOT_EXCLUDED(OP_cube)
    registerEmitter<sd::ops::cube>(
        result, VulkanKernelRecipe::CUBE,
         kNumeric32, kStridedLayout);
#endif
#if NOT_EXCLUDED(OP_rectifiedtanh)
    registerEmitter<sd::ops::rectifiedtanh>(
        result, VulkanKernelRecipe::RECTIFIED_TANH,
         kNumeric32, kStridedLayout,
        0, -1,  VULKAN_EMITTER_TRAIT_FLOAT_RESULT);
    registerEmitter<sd::ops::rectifiedtanh_bp>(
        result, VulkanKernelRecipe::RECTIFIED_TANH_BP,
         kFloat, kStridedLayout);
#endif
#if NOT_EXCLUDED(OP_rationaltanh)
    registerEmitter<sd::ops::rationaltanh>(
        result, VulkanKernelRecipe::RATIONAL_TANH,
         kNumeric32, kStridedLayout,
        0, -1,  VULKAN_EMITTER_TRAIT_FLOAT_RESULT);
#endif
#if NOT_EXCLUDED(OP_Floor)
    registerEmitter<sd::ops::Floor>(
        result, VulkanKernelRecipe::FLOOR,
         kNumeric32, kStridedLayout);
#endif
#if NOT_EXCLUDED(OP_Log1p)
    registerEmitter<sd::ops::Log1p>(
        result, VulkanKernelRecipe::LOG1P,
         kNumeric32, kStridedLayout,
        0, -1,  VULKAN_EMITTER_TRAIT_FLOAT_RESULT);
#endif
#if NOT_EXCLUDED(OP_rint)
    registerEmitter<sd::ops::rint>(
        result, VulkanKernelRecipe::RINT,
         kNumeric32, kStridedLayout,
        0, -1,  VULKAN_EMITTER_TRAIT_FLOAT_RESULT);
#endif
#if NOT_EXCLUDED(OP_scale)
    registerEmitter<sd::ops::scale>(
        result, VulkanKernelRecipe::SCALE,
         kFloat, kStridedLayout,
        0, -1, VULKAN_EMITTER_TRAIT_NONE,
        VulkanArgumentSchema::OPTIONAL_FINITE_SCALAR);
#endif
#if NOT_EXCLUDED(OP_prelu)
    registerEmitter<sd::ops::prelu>(
        result, VulkanKernelRecipe::PRELU,
        kNumeric32, kStridedLayout, 2, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::NONE,
        VulkanLoweringContract::DEFAULT,
        singleArguments(rangedArguments(2, 2, 1, 1, 0, 0, 0, -1, 0, 0)));
#endif
#if NOT_EXCLUDED(OP_clipbyvalue)
    registerEmitter<sd::ops::clipbyvalue>(
        result, VulkanKernelRecipe::CLIP_BY_VALUE,
         kNumeric32, kStridedLayout,
        0, -1, VULKAN_EMITTER_TRAIT_NONE,
        VulkanArgumentSchema::NONE, VulkanLoweringContract::DEFAULT,
        eitherArguments(
            exactArguments(1, 1, 2, 0, 0),
            exactArguments(3, 1, 0, 0, 0),
            VULKAN_ARGUMENT_VALUES_FINITE_TARGS |
                VULKAN_ARGUMENT_VALUES_ORDERED_TARG_PAIR));
#endif
#if NOT_EXCLUDED(OP_thresholdedrelu)
    registerEmitter<sd::ops::thresholdedrelu>(
        result, VulkanKernelRecipe::THRESHOLDED_RELU,
         kFloat, kStridedLayout,
        0, -1, VULKAN_EMITTER_TRAIT_NONE,
        VulkanArgumentSchema::OPTIONAL_SCALAR);
    registerEmitter<sd::ops::thresholdedrelu_bp>(
        result, VulkanKernelRecipe::THRESHOLDED_RELU_BP,
         kFloat, kStridedLayout,
        0, -1, VULKAN_EMITTER_TRAIT_NONE,
        VulkanArgumentSchema::OPTIONAL_SCALAR);
#endif

    // BOOL descriptors are admitted here; the recorder validates the real
    // storageBuffer8BitAccess feature before claiming a dispatch.
#if NOT_EXCLUDED(OP_equals)
    registerEmitter<sd::ops::equals>(
        result, VulkanKernelRecipe::EQUAL,
        kNumeric32WithBool, kGeneralLayout);
#endif
#if NOT_EXCLUDED(OP_not_equals)
    registerEmitter<sd::ops::not_equals>(
        result, VulkanKernelRecipe::NOT_EQUAL,
        kNumeric32WithBool, kGeneralLayout);
#endif
#if NOT_EXCLUDED(OP_less)
    registerEmitter<sd::ops::less>(
        result, VulkanKernelRecipe::LESS,
        kNumeric32WithBool, kGeneralLayout);
#endif
#if NOT_EXCLUDED(OP_less_equal)
    registerEmitter<sd::ops::less_equal>(
        result, VulkanKernelRecipe::LESS_EQUAL,
        kNumeric32WithBool, kGeneralLayout);
#endif
#if NOT_EXCLUDED(OP_greater)
    registerEmitter<sd::ops::greater>(
        result, VulkanKernelRecipe::GREATER,
        kNumeric32WithBool, kGeneralLayout);
#endif
#if NOT_EXCLUDED(OP_greater_equal)
    registerEmitter<sd::ops::greater_equal>(
        result, VulkanKernelRecipe::GREATER_EQUAL,
        kNumeric32WithBool, kGeneralLayout);
#endif
#if NOT_EXCLUDED(OP_boolean_and)
    registerEmitter<sd::ops::boolean_and>(
        result, VulkanKernelRecipe::BOOLEAN_AND,
        kNumeric32WithBool, kGeneralLayout);
#endif
#if NOT_EXCLUDED(OP_boolean_or)
    registerEmitter<sd::ops::boolean_or>(
        result, VulkanKernelRecipe::BOOLEAN_OR,
        kNumeric32WithBool, kGeneralLayout);
#endif
#if NOT_EXCLUDED(OP_boolean_xor)
    registerEmitter<sd::ops::boolean_xor>(
        result, VulkanKernelRecipe::BOOLEAN_XOR,
        kNumeric32WithBool, kGeneralLayout);
#endif
#if NOT_EXCLUDED(OP_boolean_not)
    registerEmitter<sd::ops::boolean_not>(
        result, VulkanKernelRecipe::BOOLEAN_NOT,
        kNumeric32WithBool, kStridedLayout);
#endif
#if NOT_EXCLUDED(OP_select)
    registerEmitter<sd::ops::select>(
        result, VulkanKernelRecipe::SELECT,
        kNumeric32WithBool, kGeneralLayout);
#endif
#if NOT_EXCLUDED(OP_cast)
    registerEmitter<sd::ops::cast>(
        result, VulkanKernelRecipe::CAST,
        kNumeric32, kStridedLayout);
#endif

#if NOT_EXCLUDED(OP_softmax)
    registerEmitter<sd::ops::softmax>(
        result, VulkanKernelRecipe::SOFTMAX,
         kFloat, kStridedLayout, 2, 2,
         VULKAN_EMITTER_TRAIT_DISPATCH_FIRST_DIM,
         VulkanArgumentSchema::NONE, VulkanLoweringContract::SOFTMAX);
#endif
#if NOT_EXCLUDED(OP_log_softmax)
    registerEmitter<sd::ops::log_softmax>(
        result, VulkanKernelRecipe::LOG_SOFTMAX,
         kFloat, kStridedLayout, 2, 2,
         VULKAN_EMITTER_TRAIT_DISPATCH_FIRST_DIM,
         VulkanArgumentSchema::NONE, VulkanLoweringContract::SOFTMAX);
#endif
#if NOT_EXCLUDED(OP_layer_norm)
    registerEmitter<sd::ops::layer_norm>(
        result, VulkanKernelRecipe::LAYER_NORM,
         kFloat, kStridedLayout, 2, 2,
         VULKAN_EMITTER_TRAIT_DISPATCH_FIRST_DIM,
         VulkanArgumentSchema::NONE, VulkanLoweringContract::LAYER_NORM);
#endif
#if NOT_EXCLUDED(OP_normalize_moments)
    registerEmitter<sd::ops::normalize_moments>(
        result, VulkanKernelRecipe::NORMALIZE_MOMENTS,
        kNumeric32, kStridedLayout, 0, -1,
        VULKAN_EMITTER_TRAIT_NONE,
        VulkanArgumentSchema::OPTIONAL_FINITE_SCALAR);
#endif
#if NOT_EXCLUDED(OP_batchnorm)
    registerEmitter<sd::ops::batchnorm>(
        result, VulkanKernelRecipe::BATCH_NORM,
        kFloat, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_EPSILON_PARAMETER,
        VulkanArgumentSchema::EPSILON, VulkanLoweringContract::DEFAULT,
        singleArguments(
            rangedArguments(3, 5, 1, 1, 1, 1, 2, -1, 0, 0),
            VULKAN_ARGUMENT_VALUES_FINITE_TARGS));
#endif
#if NOT_EXCLUDED(OP_fused_layer_norm)
    registerEmitter<sd::ops::fused_layer_norm>(
        result, VulkanKernelRecipe::LAYER_NORM,
         kFloat, kStridedLayout, 2, 2,
         VULKAN_EMITTER_TRAIT_DISPATCH_FIRST_DIM |
                  VULKAN_EMITTER_TRAIT_EPSILON_PARAMETER,
         VulkanArgumentSchema::EPSILON, VulkanLoweringContract::LAYER_NORM);
#endif
#if NOT_EXCLUDED(OP_fused_layer_norm)
    // Parameter gradients require a cross-row reduction.  The rank-one form
    // has exactly one normalization row, so one portable Vulkan invocation
    // can write dX and dGamma without floating atomics.
    registerEmitter<sd::ops::fused_layer_norm_bp>(
        result, VulkanKernelRecipe::FUSED_LAYER_NORM_BP,
        kFloat, kStridedLayout, 1, 1,
        VULKAN_EMITTER_TRAIT_DISPATCH_SINGLE |
            VULKAN_EMITTER_TRAIT_EPSILON_PARAMETER,
        VulkanArgumentSchema::EPSILON, VulkanLoweringContract::DEFAULT,
        singleArguments(
            rangedArguments(3, 3, 2, 2, 0, 1, 0, 0, 0, 0),
            VULKAN_ARGUMENT_VALUES_FINITE_TARGS));
#endif
#if NOT_EXCLUDED(OP_legacy_pull_rows)
    registerEmitter<sd::ops::legacy_pull_rows>(
        result, VulkanKernelRecipe::PULL_INDEXED_TADS,
        kNumeric32 | VULKAN_DTYPE_INDEX, kStridedLayout, 1, 2,
        VULKAN_EMITTER_TRAIT_DISPATCH_SINGLE, VulkanArgumentSchema::NONE,
        VulkanLoweringContract::INDEXED_TAD_MOVEMENT,
        singleArguments(exactArguments(2, 1, 0, 2, 0)),
        loadedOperandTypes(uint16_t{1} << 0, 0, uint16_t{1} << 1));
#endif
#if NOT_EXCLUDED(OP_legacy_shuffle)
    registerEmitter<sd::ops::legacy_shuffle>(
        result, VulkanKernelRecipe::DISJOINT_PAIR_SHUFFLE,
        kNumeric32, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_DISPATCH_SINGLE, VulkanArgumentSchema::NONE,
        VulkanLoweringContract::INDEXED_TAD_MOVEMENT,
        singleArguments(rangedArguments(
            2, -1, 1, -1, 0, 0, 3, -1, 0, 0)));
#endif
#if NOT_EXCLUDED(OP_gather)
    registerEmitter<sd::ops::gather>(
        result, VulkanKernelRecipe::GATHER,
        kNumeric32, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::AXES_IARGS);
#endif
#if NOT_EXCLUDED(OP_embedding_lookup)
    registerEmitter<sd::ops::embedding_lookup>(
        result, VulkanKernelRecipe::EMBEDDING_LOOKUP,
        kNumeric32, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::SINGLE_IARG);
#endif
#if NOT_EXCLUDED(OP_concat)
    registerEmitter<sd::ops::concat>(
        result, VulkanKernelRecipe::CONCAT,
        kNumeric32, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::AXES_IARGS);
#endif
#if NOT_EXCLUDED(OP_squeeze)
    registerEmitter<sd::ops::squeeze>(
        result, VulkanKernelRecipe::COPY,
        kNumeric32, kGeneralLayout, 0, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::AXES_IARGS,
        VulkanLoweringContract::LINEAR_COPY);
#endif
#if NOT_EXCLUDED(OP_expand_dims)
    registerEmitter<sd::ops::expand_dims>(
        result, VulkanKernelRecipe::COPY,
        kNumeric32, kGeneralLayout, 0, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::SINGLE_IARG,
        VulkanLoweringContract::LINEAR_COPY);
#endif
#if NOT_EXCLUDED(OP_stop_gradient)
    registerEmitter<sd::ops::stop_gradient>(
        result, VulkanKernelRecipe::COPY,
        kNumeric32, kGeneralLayout, 0, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::NONE,
        VulkanLoweringContract::LINEAR_COPY);
#endif
#if NOT_EXCLUDED(OP_reshape)
    registerEmitter<sd::ops::reshape>(
        result, VulkanKernelRecipe::COPY,
        kNumeric32WithStructuralIndex, kGeneralLayout, 0, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::NONE,
        VulkanLoweringContract::LINEAR_COPY, {},
        structuralIndexInputs(uint16_t{1} << 1));
#endif
#if NOT_EXCLUDED(OP_linear_copy)
    registerEmitter<sd::ops::linear_copy>(
        result, VulkanKernelRecipe::COPY,
        kNumeric32WithStructuralIndex, kDenseLayout, 0, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::NONE,
        VulkanLoweringContract::LINEAR_COPY, {},
        structuralIndexInputs(uint16_t{1} << 1));
#endif
#if NOT_EXCLUDED(OP_flatten)
    registerEmitter<sd::ops::flatten>(
        result, VulkanKernelRecipe::COPY,
        kNumeric32, kGeneralLayout, 0, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::SINGLE_IARG);
#endif
#if NOT_EXCLUDED(OP_broadcast_to)
    registerEmitter<sd::ops::broadcast_to>(
        result, VulkanKernelRecipe::COPY,
        kNumeric32WithStructuralIndex, kGeneralLayout, 0, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::NONE,
        VulkanLoweringContract::DEFAULT, {},
        structuralIndexInputs(uint16_t{1} << 1));
#endif
#if NOT_EXCLUDED(OP_transpose)
    registerEmitter<sd::ops::transpose>(
        result, VulkanKernelRecipe::TRANSPOSE,
        kNumeric32, kStridedLayout, 2, 4,
        VULKAN_EMITTER_TRAIT_NONE,
        VulkanArgumentSchema::PERMUTATION_AXES);
#endif
#if NOT_EXCLUDED(OP_permute)
    registerEmitter<sd::ops::permute>(
        result, VulkanKernelRecipe::PERMUTE,
        kNumeric32, kStridedLayout, 2, 4,
        VULKAN_EMITTER_TRAIT_NONE,
        VulkanArgumentSchema::PERMUTATION_AXES);
#endif
#if NOT_EXCLUDED(OP_gather_nd)
    registerEmitter<sd::ops::gather_nd>(
        result, VulkanKernelRecipe::GATHER_ND,
        kNumeric32, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::NONE,
        VulkanLoweringContract::DEFAULT,
        movementArguments(2, 0, 0, 0, 1));
#endif
#if NOT_EXCLUDED(OP_scatter_nd)
    registerEmitter<sd::ops::scatter_nd>(
        result, VulkanKernelRecipe::COPY,
        kNumeric32WithStructuralIndex, kGeneralLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_DISPATCH_SINGLE |
            VULKAN_EMITTER_TRAIT_BOOLEAN_PARAMETERS,
        VulkanArgumentSchema::NONE, VulkanLoweringContract::DEFAULT, {},
        structuralIndexInputs(uint16_t{1} << 2));
#endif
#if NOT_EXCLUDED(OP_tile)
    registerEmitter<sd::ops::tile>(
        result, VulkanKernelRecipe::TILE,
        kNumeric32, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::NONE,
        VulkanLoweringContract::DEFAULT, movementArguments(1, 1, -1));
#endif
#if NOT_EXCLUDED(OP_repeat)
    registerEmitter<sd::ops::repeat>(
        result, VulkanKernelRecipe::REPEAT,
        kNumeric32, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::NONE,
        VulkanLoweringContract::DEFAULT, movementArguments(1, 2, -1));
#endif
#if NOT_EXCLUDED(OP_reverse)
    registerEmitter<sd::ops::reverse>(
        result, VulkanKernelRecipe::REVERSE,
        kNumeric32, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::NONE,
        VulkanLoweringContract::DEFAULT, movementArguments(1, 0, -1));
#endif
#if NOT_EXCLUDED(OP_roll)
    registerEmitter<sd::ops::roll>(
        result, VulkanKernelRecipe::ROLL,
        kNumericWithIndex, kStridedLayout, 0, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::NONE,
        VulkanLoweringContract::DEFAULT,
        eitherArguments(
            rangedArguments(1, 1, 1, 1, 0, 0, 1, -1, 0, 0),
            rangedArguments(2, 3, 1, 1, 0, 0, 0, 0, 0, 0)),
        loadedIndexOperandTypes((uint16_t{1} << 1) | (uint16_t{1} << 2)));
#endif
#if NOT_EXCLUDED(OP_slice)
    registerEmitter<sd::ops::slice>(
        result, VulkanKernelRecipe::SLICE,
        kNumeric32, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::NONE,
        VulkanLoweringContract::DEFAULT, movementArguments(1, 2, -1));
#endif
#if NOT_EXCLUDED(OP_strided_slice)
    registerEmitter<sd::ops::strided_slice>(
        result, VulkanKernelRecipe::STRIDED_SLICE,
        kNumeric32, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::NONE,
        VulkanLoweringContract::DEFAULT, movementArguments(1, 8, -1));
#endif
#if NOT_EXCLUDED(OP_stack)
    registerEmitter<sd::ops::stack>(
        result, VulkanKernelRecipe::STACK,
         kNumeric32, kStridedLayout, 0, -1);
#endif
#if NOT_EXCLUDED(OP_split)
    registerEmitter<sd::ops::split>(
        result, VulkanKernelRecipe::SPLIT,
        kNumeric32, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::AXES_IARGS);
#endif
#if NOT_EXCLUDED(OP_split_v)
    registerEmitter<sd::ops::split_v>(
        result, VulkanKernelRecipe::COPY,
        kNumeric32WithStructuralIndex, kGeneralLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::SINGLE_IARG,
        VulkanLoweringContract::DEFAULT, {},
        structuralIndexInputs(uint16_t{1} << 1));
#endif
#if NOT_EXCLUDED(OP_unstack)
    registerEmitter<sd::ops::unstack>(
        result, VulkanKernelRecipe::UNSTACK,
        kNumeric32, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::AXES_IARGS);
#endif
#if NOT_EXCLUDED(OP_triu)
    registerEmitter<sd::ops::triu>(
        result, VulkanKernelRecipe::TRIU,
        kFloat, kStridedLayout, 2, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::NONE,
        VulkanLoweringContract::DEFAULT, movementArguments(1, 0, 1));
    registerEmitter<sd::ops::triu_bp>(
        result, VulkanKernelRecipe::TRIU_BP,
        kFloat, kStridedLayout, 0, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::NONE,
        VulkanLoweringContract::DEFAULT, movementArguments(2, 0, 1));
#endif
#if NOT_EXCLUDED(OP_win_part)
    registerEmitter<sd::ops::win_part>(
        result, VulkanKernelRecipe::WINDOW_PARTITION,
        kFloat, kStridedLayout, 4, 4,
        VULKAN_EMITTER_TRAIT_NONE,
        VulkanArgumentSchema::WINDOW_GEOMETRY_IARGS,
        VulkanLoweringContract::DEFAULT,
        singleArguments(exactArguments(1, 1, 0, 1, 0)));
#endif
#if NOT_EXCLUDED(OP_win_unpart)
    registerEmitter<sd::ops::win_unpart>(
        result, VulkanKernelRecipe::WINDOW_UNPARTITION,
        kFloat, kStridedLayout, 4, 4,
        VULKAN_EMITTER_TRAIT_NONE,
        VulkanArgumentSchema::WINDOW_GEOMETRY_IARGS,
        VulkanLoweringContract::DEFAULT,
        singleArguments(exactArguments(1, 1, 0, 3, 0)));
#endif

#if NOT_EXCLUDED(OP_fill_as)
    registerEmitter<sd::ops::fill_as>(
        result, VulkanKernelRecipe::FILL_AS,
        kNumeric32, kStridedLayout, 0, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::NONE,
        VulkanLoweringContract::DEFAULT,
        eitherArguments(
            exactArguments(1, 1, 1, 0, 0),
            exactArguments(1, 1, 0, 1, 0),
            VULKAN_ARGUMENT_VALUES_FINITE_TARGS));
#endif
#if NOT_EXCLUDED(OP_zeros_as)
    registerEmitter<sd::ops::zeros_as>(
        result, VulkanKernelRecipe::ZEROS_AS,
        kNumeric32, kStridedLayout, 0, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::NONE,
        VulkanLoweringContract::DEFAULT,
        singleArguments(exactArguments(1, 1, 0, 0, 0)));
#endif
#if NOT_EXCLUDED(OP_ones_as)
    registerEmitter<sd::ops::ones_as>(
        result, VulkanKernelRecipe::ONES_AS,
        kNumeric32, kStridedLayout, 0, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::NONE,
        VulkanLoweringContract::DEFAULT,
        singleArguments(
            rangedArguments(1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1)));
#endif
#if NOT_EXCLUDED(OP_create)
    registerEmitter<sd::ops::create>(
        result, VulkanKernelRecipe::ZEROS_AS,
        kNumeric32WithStructuralIndex, kStridedLayout, 0, -1,
        VULKAN_EMITTER_TRAIT_BOOLEAN_PARAMETERS, VulkanArgumentSchema::NONE,
        VulkanLoweringContract::DEFAULT,
        singleArguments(
            rangedArguments(1, 1, 1, 1, 0, 0, 2, 2, 0, 1, 0, 1),
            VULKAN_ARGUMENT_VALUES_PRESENT_BARG_ZERO_TRUE),
        structuralIndexInputs(uint16_t{1}));
#endif
#if NOT_EXCLUDED(OP_shape_of)
    registerEmitter<sd::ops::shape_of>(
        result, VulkanKernelRecipe::SHAPE_OF,
        kNumeric32, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::NONE,
        VulkanLoweringContract::DEFAULT,
        singleArguments(exactArguments(1, 1, 0, 1, 0)));
#endif
#if NOT_EXCLUDED(OP_onehot)
    registerEmitter<sd::ops::onehot>(
        result, VulkanKernelRecipe::ONE_HOT,
        kNumeric32, kStridedLayout, 0, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::NONE,
        VulkanLoweringContract::DEFAULT,
        singleArguments(
            rangedArguments(1, 1, 1, 1, 0, 2, 2, 2, 0, 0, 0, 1),
            VULKAN_ARGUMENT_VALUES_FINITE_TARGS));
#endif
#if NOT_EXCLUDED(OP_eye)
    registerEmitter<sd::ops::eye>(
        result, VulkanKernelRecipe::EYE,
        kFloat, kStridedLayout, 2, -1,
        VULKAN_EMITTER_TRAIT_ARGUMENT_GENERATED, VulkanArgumentSchema::NONE,
        VulkanLoweringContract::DEFAULT,
        singleArguments(
            rangedArguments(0, 0, 1, 1, 0, 1, 1, -1, 0, 0),
            VULKAN_ARGUMENT_VALUES_FINITE_TARGS));
#endif
#if NOT_EXCLUDED(OP_lin_space)
    registerEmitter<sd::ops::lin_space>(
        result, VulkanKernelRecipe::LIN_SPACE,
        kNumeric32, kStridedLayout, 1, 1,
        VULKAN_EMITTER_TRAIT_ARGUMENT_GENERATED, VulkanArgumentSchema::NONE,
        VulkanLoweringContract::DEFAULT,
        singleArguments(
            rangedArguments(0, 0, 1, 1, 2, 2, 1, 1, 0, 1, 0, 1),
            VULKAN_ARGUMENT_VALUES_FINITE_TARGS));
#endif
#if NOT_EXCLUDED(OP_range)
    registerEmitter<sd::ops::range>(
        result, VulkanKernelRecipe::RANGE,
        kNumeric32, kStridedLayout, 1, 1,
        VULKAN_EMITTER_TRAIT_ARGUMENT_GENERATED, VulkanArgumentSchema::NONE,
        VulkanLoweringContract::DEFAULT,
        eitherArguments(
            rangedArguments(0, 0, 1, 1, 0, 0, 1, 3, 0, 0, 0, 1),
            rangedArguments(0, 0, 1, 1, 1, 3, 0, 0, 0, 0, 0, 1),
            VULKAN_ARGUMENT_VALUES_FINITE_TARGS));
#endif
#if NOT_EXCLUDED(OP_rank)
    registerEmitter<sd::ops::rank>(
        result, VulkanKernelRecipe::RANK_OF,
        kNumeric32, kStridedLayout, 0, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::NONE,
        VulkanLoweringContract::DEFAULT,
        singleArguments(exactArguments(1, 1, 0, 0, 0)));
#endif
#if NOT_EXCLUDED(OP_size)
    registerEmitter<sd::ops::size>(
        result, VulkanKernelRecipe::SIZE_OF,
        kNumeric32, kStridedLayout, 0, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::NONE,
        VulkanLoweringContract::DEFAULT,
        singleArguments(exactArguments(1, 1, 0, 0, 0)));
#endif
#if NOT_EXCLUDED(OP_size_at)
    registerEmitter<sd::ops::size_at>(
        result, VulkanKernelRecipe::SIZE_AT,
        kNumeric32, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::NONE,
        VulkanLoweringContract::DEFAULT,
        singleArguments(exactArguments(1, 1, 0, 1, 0)));
#endif
#if NOT_EXCLUDED(OP_min_max_datatype)
    registerEmitter<sd::ops::min_max_datatype>(
        result, VulkanKernelRecipe::MIN_MAX_DATATYPE,
        kNumeric32, kStridedLayout, 0, 0,
        VULKAN_EMITTER_TRAIT_ARGUMENT_GENERATED, VulkanArgumentSchema::NONE,
        VulkanLoweringContract::DEFAULT,
        singleArguments(exactArguments(0, 1, 0, 2, 0)));
#endif

#if NOT_EXCLUDED(OP_reduce_sum)
    registerEmitter<sd::ops::reduce_sum>(
        result, VulkanKernelRecipe::REDUCE_SUM,
        kNumeric32, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_BOOLEAN_PARAMETERS,
        VulkanArgumentSchema::REDUCTION_KEEPDIMS);
#endif
#if NOT_EXCLUDED(OP_reduce_mean)
    registerEmitter<sd::ops::reduce_mean>(
        result, VulkanKernelRecipe::REDUCE_SUM,
        kFloat, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_MEAN_FINALIZE |
            VULKAN_EMITTER_TRAIT_BOOLEAN_PARAMETERS,
        VulkanArgumentSchema::REDUCTION_KEEPDIMS);
#endif
#if NOT_EXCLUDED(OP_mean_square)
    registerEmitter<sd::ops::mean_square>(
        result, VulkanKernelRecipe::REDUCE_SUM,
        kFloat, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_IMPLICIT_LAST_AXIS |
            VULKAN_EMITTER_TRAIT_SQUARE_INPUT |
            VULKAN_EMITTER_TRAIT_MEAN_FINALIZE);
#endif
#if NOT_EXCLUDED(OP_reduce_norm1)
    registerEmitter<sd::ops::reduce_norm1>(
        result, VulkanKernelRecipe::REDUCE_SUM,
        kNumeric32, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_FLOAT_RESULT |
            VULKAN_EMITTER_TRAIT_ABSOLUTE_INPUT |
            VULKAN_EMITTER_TRAIT_BOOLEAN_PARAMETERS,
        VulkanArgumentSchema::REDUCTION_KEEPDIMS);
#endif
#if NOT_EXCLUDED(OP_reduce_norm2)
    registerEmitter<sd::ops::reduce_norm2>(
        result, VulkanKernelRecipe::REDUCE_SUM,
        kNumeric32, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_FLOAT_RESULT |
            VULKAN_EMITTER_TRAIT_SQUARE_INPUT |
            VULKAN_EMITTER_TRAIT_SQRT_FINALIZE |
            VULKAN_EMITTER_TRAIT_BOOLEAN_PARAMETERS,
        VulkanArgumentSchema::REDUCTION_KEEPDIMS);
#endif
#if NOT_EXCLUDED(OP_reduce_norm_max)
    registerEmitter<sd::ops::reduce_norm_max>(
        result, VulkanKernelRecipe::REDUCE_MAX,
        kNumeric32, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_FLOAT_RESULT |
            VULKAN_EMITTER_TRAIT_ABSOLUTE_INPUT |
            VULKAN_EMITTER_TRAIT_BOOLEAN_PARAMETERS,
        VulkanArgumentSchema::REDUCTION_KEEPDIMS);
#endif
#if NOT_EXCLUDED(OP_reduce_sqnorm)
    registerEmitter<sd::ops::reduce_sqnorm>(
        result, VulkanKernelRecipe::REDUCE_SUM,
        kNumeric32, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_FLOAT_RESULT |
            VULKAN_EMITTER_TRAIT_SQUARE_INPUT |
            VULKAN_EMITTER_TRAIT_BOOLEAN_PARAMETERS,
        VulkanArgumentSchema::REDUCTION_KEEPDIMS);
#endif
#if NOT_EXCLUDED(OP_reduce_logsumexp)
    registerEmitter<sd::ops::reduce_logsumexp>(
        result, VulkanKernelRecipe::REDUCE_LOGSUMEXP,
        kNumeric32, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_FLOAT_RESULT,
        VulkanArgumentSchema::REDUCTION_KEEPDIMS);
#endif
#if NOT_EXCLUDED(OP_reduce_variance)
    registerEmitter<sd::ops::reduce_variance>(
        result, VulkanKernelRecipe::REDUCE_VARIANCE,
        kNumeric32, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_FLOAT_RESULT |
            VULKAN_EMITTER_TRAIT_BIAS_CORRECTION_ARGUMENT |
            VULKAN_EMITTER_TRAIT_BOOLEAN_PARAMETERS,
        VulkanArgumentSchema::STATISTICAL_REDUCTION);
#endif
#if NOT_EXCLUDED(OP_reduce_stdev)
    registerEmitter<sd::ops::reduce_stdev>(
        result, VulkanKernelRecipe::REDUCE_STDEV,
        kNumeric32, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_FLOAT_RESULT |
            VULKAN_EMITTER_TRAIT_BIAS_CORRECTION_ARGUMENT |
            VULKAN_EMITTER_TRAIT_BOOLEAN_PARAMETERS,
        VulkanArgumentSchema::STATISTICAL_REDUCTION);
#endif
#if NOT_EXCLUDED(OP_reduce_max)
    registerEmitter<sd::ops::reduce_max>(
        result, VulkanKernelRecipe::REDUCE_MAX,
        kNumeric32, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_BOOLEAN_PARAMETERS,
        VulkanArgumentSchema::REDUCTION_KEEPDIMS);
#endif
#if NOT_EXCLUDED(OP_reduce_min)
    registerEmitter<sd::ops::reduce_min>(
        result, VulkanKernelRecipe::REDUCE_MIN,
        kNumeric32, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_BOOLEAN_PARAMETERS,
        VulkanArgumentSchema::REDUCTION_KEEPDIMS);
#endif
#if NOT_EXCLUDED(OP_reduce_prod)
    registerEmitter<sd::ops::reduce_prod>(
        result, VulkanKernelRecipe::REDUCE_PRODUCT,
        kNumeric32, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_BOOLEAN_PARAMETERS,
        VulkanArgumentSchema::REDUCTION_KEEPDIMS);
#endif
#if NOT_EXCLUDED(OP_argmax)
    registerEmitter<sd::ops::argmax>(
        result, VulkanKernelRecipe::REDUCE_MAX,
        kNumeric32, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_INDEX_RESULT |
            VULKAN_EMITTER_TRAIT_BOOLEAN_PARAMETERS,
        VulkanArgumentSchema::INDEX_REDUCTION);
#endif
#if NOT_EXCLUDED(OP_argmin)
    registerEmitter<sd::ops::argmin>(
        result, VulkanKernelRecipe::REDUCE_MIN,
        kNumeric32, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_INDEX_RESULT |
            VULKAN_EMITTER_TRAIT_BOOLEAN_PARAMETERS,
        VulkanArgumentSchema::INDEX_REDUCTION);
#endif
#if NOT_EXCLUDED(OP_argamax)
    registerEmitter<sd::ops::argamax>(
        result, VulkanKernelRecipe::REDUCE_MAX,
        kNumeric32, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_INDEX_RESULT |
            VULKAN_EMITTER_TRAIT_ABSOLUTE_INPUT |
            VULKAN_EMITTER_TRAIT_BOOLEAN_PARAMETERS,
        VulkanArgumentSchema::INDEX_REDUCTION);
#endif
#if NOT_EXCLUDED(OP_argamin)
    registerEmitter<sd::ops::argamin>(
        result, VulkanKernelRecipe::REDUCE_MIN,
        kNumeric32, kStridedLayout, 1, -1,
        VULKAN_EMITTER_TRAIT_INDEX_RESULT |
            VULKAN_EMITTER_TRAIT_ABSOLUTE_INPUT |
            VULKAN_EMITTER_TRAIT_BOOLEAN_PARAMETERS,
        VulkanArgumentSchema::INDEX_REDUCTION);
#endif

    return result;
  }();
  return catalog;
}

const LegacyEmitterCatalogData& buildLegacyCatalog() {
  static const LegacyEmitterCatalogData catalog = [] {
    LegacyEmitterCatalogData result;

    registerLegacyEmitter<sd::ops::LegacyTransformSameOp>(
        result, VulkanLegacyOpFamily::TRANSFORM_SAME,
        static_cast<int>(sd::transform::Abs),
        VulkanKernelRecipe::ABS, kFloat, kGeneralLayout);
    registerLegacyEmitter<sd::ops::LegacyTransformSameOp>(
        result, VulkanLegacyOpFamily::TRANSFORM_SAME,
        static_cast<int>(sd::transform::Neg),
        VulkanKernelRecipe::NEGATE, kFloat, kGeneralLayout);
    registerLegacyEmitter<sd::ops::LegacyTransformSameOp>(
        result, VulkanLegacyOpFamily::TRANSFORM_SAME,
        static_cast<int>(sd::transform::Cube),
        VulkanKernelRecipe::CUBE, kNumeric32, kGeneralLayout);
    registerLegacyEmitter<sd::ops::LegacyTransformStrictOp>(
        result, VulkanLegacyOpFamily::TRANSFORM_STRICT,
        static_cast<int>(sd::transform::Sigmoid),
        VulkanKernelRecipe::SIGMOID, kFloat, kGeneralLayout);
    registerLegacyEmitter<sd::ops::LegacyTransformStrictOp>(
        result, VulkanLegacyOpFamily::TRANSFORM_STRICT,
        static_cast<int>(sd::transform::SoftPlus),
        VulkanKernelRecipe::SOFTPLUS, kFloat, kGeneralLayout);
    registerLegacyEmitter<sd::ops::LegacyTransformStrictOp>(
        result, VulkanLegacyOpFamily::TRANSFORM_STRICT,
        static_cast<int>(sd::transform::Rint),
        VulkanKernelRecipe::RINT, kFloat, kGeneralLayout);
    registerLegacyEmitter<sd::ops::LegacyTransformStrictOp>(
        result, VulkanLegacyOpFamily::TRANSFORM_STRICT,
        static_cast<int>(sd::transform::Tanh),
        VulkanKernelRecipe::TANH, kFloat, kGeneralLayout);
    registerLegacyEmitter<sd::ops::LegacyTransformStrictOp>(
        result, VulkanLegacyOpFamily::TRANSFORM_STRICT,
        static_cast<int>(sd::transform::Swish),
        VulkanKernelRecipe::SILU, kFloat, kGeneralLayout);
    registerLegacyEmitter<sd::ops::LegacyTransformStrictOp>(
        result, VulkanLegacyOpFamily::TRANSFORM_STRICT,
        static_cast<int>(sd::transform::GELU),
        VulkanKernelRecipe::FAST_GELU, kFloat, kGeneralLayout);
    registerLegacyEmitter<sd::ops::LegacyTransformStrictOp>(
        result, VulkanLegacyOpFamily::TRANSFORM_STRICT,
        static_cast<int>(sd::transform::PreciseGELU),
        VulkanKernelRecipe::GELU, kFloat, kGeneralLayout);
    registerLegacyEmitter<sd::ops::LegacyTransformFloatOp>(
        result, VulkanLegacyOpFamily::TRANSFORM_FLOAT,
        static_cast<int>(sd::transform::Sqrt),
        VulkanKernelRecipe::SQRT, kFloat, kGeneralLayout,
        0, -1, VULKAN_EMITTER_TRAIT_FLOAT_RESULT);
    registerLegacyEmitter<sd::ops::LegacyTransformAnyOp>(
        result, VulkanLegacyOpFamily::TRANSFORM_ANY,
        static_cast<int>(sd::transform::Assign),
        VulkanKernelRecipe::ASSIGN, kNumericWithIndex, kGeneralLayout,
        0, -1, VULKAN_EMITTER_TRAIT_NONE, VulkanArgumentSchema::NONE,
        VulkanLoweringContract::LINEAR_COPY);

    registerLegacyEmitter<sd::ops::LegacyRandomOp>(
        result, VulkanLegacyOpFamily::RANDOM,
        static_cast<int>(sd::random::UniformDistribution),
        VulkanKernelRecipe::UNIFORM_RANDOM, kFloat, kStridedLayout,
        0, -1, VULKAN_EMITTER_TRAIT_RANDOM_STATE,
        VulkanArgumentSchema::NONE, VulkanLoweringContract::DEFAULT,
        singleArguments(
            exactArguments(0, 1, 2, 0, 0),
            VULKAN_ARGUMENT_VALUES_FINITE_TARGS |
                VULKAN_ARGUMENT_VALUES_ORDERED_TARG_PAIR));

    registerLegacyEmitter<sd::ops::LegacyScalarOp>(
        result, VulkanLegacyOpFamily::SCALAR,
        static_cast<int>(sd::scalar::Add),
        VulkanKernelRecipe::ADD, kNumeric32, kGeneralLayout);
    registerLegacyEmitter<sd::ops::LegacyScalarOp>(
        result, VulkanLegacyOpFamily::SCALAR,
        static_cast<int>(sd::scalar::Multiply),
        VulkanKernelRecipe::MULTIPLY, kNumeric32, kGeneralLayout);
    registerLegacyEmitter<sd::ops::LegacyScalarOp>(
        result, VulkanLegacyOpFamily::SCALAR,
        static_cast<int>(sd::scalar::LeakyRELU),
        VulkanKernelRecipe::LEAKY_RELU, kFloat, kGeneralLayout);
    registerLegacyEmitter<sd::ops::LegacyScalarOp>(
        result, VulkanLegacyOpFamily::SCALAR,
        static_cast<int>(sd::scalar::RELU),
        VulkanKernelRecipe::MAXIMUM, kFloat, kGeneralLayout);

    constexpr uint32_t reductionParameters =
        VULKAN_EMITTER_TRAIT_BOOLEAN_PARAMETERS |
        VULKAN_EMITTER_TRAIT_TAD_REDUCTION_PERMUTATION;
    registerLegacyEmitter<sd::ops::LegacyReduceSameOp>(
        result, VulkanLegacyOpFamily::REDUCE_SAME,
        static_cast<int>(sd::reduce::Sum),
        VulkanKernelRecipe::REDUCE_SUM, kNumeric32, kStridedLayout,
        1, -1, reductionParameters,
        VulkanArgumentSchema::REDUCTION_KEEPDIMS);
    registerLegacyEmitter<sd::ops::LegacyReduceSameOp>(
        result, VulkanLegacyOpFamily::REDUCE_SAME,
        static_cast<int>(sd::reduce::Max),
        VulkanKernelRecipe::REDUCE_MAX, kNumeric32, kStridedLayout,
        1, -1, reductionParameters,
        VulkanArgumentSchema::REDUCTION_KEEPDIMS);
    registerLegacyEmitter<sd::ops::LegacyReduceSameOp>(
        result, VulkanLegacyOpFamily::REDUCE_SAME,
        static_cast<int>(sd::reduce::Min),
        VulkanKernelRecipe::REDUCE_MIN, kNumeric32, kStridedLayout,
        1, -1, reductionParameters,
        VulkanArgumentSchema::REDUCTION_KEEPDIMS);
    registerLegacyEmitter<sd::ops::LegacyReduceSameOp>(
        result, VulkanLegacyOpFamily::REDUCE_SAME,
        static_cast<int>(sd::reduce::Prod),
        VulkanKernelRecipe::REDUCE_PRODUCT, kNumeric32, kStridedLayout,
        1, -1, reductionParameters,
        VulkanArgumentSchema::REDUCTION_KEEPDIMS);
    registerLegacyEmitter<sd::ops::LegacyReduceFloatOp>(
        result, VulkanLegacyOpFamily::REDUCE_FLOAT,
        static_cast<int>(sd::reduce::Mean),
        VulkanKernelRecipe::REDUCE_SUM, kFloat, kStridedLayout,
        1, -1,
        reductionParameters | VULKAN_EMITTER_TRAIT_MEAN_FINALIZE,
        VulkanArgumentSchema::REDUCTION_KEEPDIMS);
    registerLegacyEmitter<sd::ops::LegacyReduceFloatOp>(
        result, VulkanLegacyOpFamily::REDUCE_FLOAT,
        static_cast<int>(sd::reduce::Norm1),
        VulkanKernelRecipe::REDUCE_SUM, kNumeric32, kStridedLayout,
        1, -1,
        reductionParameters | VULKAN_EMITTER_TRAIT_FLOAT_RESULT |
            VULKAN_EMITTER_TRAIT_ABSOLUTE_INPUT,
        VulkanArgumentSchema::REDUCTION_KEEPDIMS);
    registerLegacyEmitter<sd::ops::LegacyReduceFloatOp>(
        result, VulkanLegacyOpFamily::REDUCE_FLOAT,
        static_cast<int>(sd::reduce::Norm2),
        VulkanKernelRecipe::REDUCE_SUM, kNumeric32, kStridedLayout,
        1, -1,
        reductionParameters | VULKAN_EMITTER_TRAIT_FLOAT_RESULT |
            VULKAN_EMITTER_TRAIT_SQUARE_INPUT |
            VULKAN_EMITTER_TRAIT_SQRT_FINALIZE,
        VulkanArgumentSchema::REDUCTION_KEEPDIMS);

    return result;
  }();
  return catalog;
}

const std::unordered_map<sd::LongType, size_t>& catalogIndex() {
  static const std::unordered_map<sd::LongType, size_t> index = [] {
    std::unordered_map<sd::LongType, size_t> result;
    const auto& catalog = buildCatalog();
    result.reserve(catalog.size());
    for (size_t i = 0; i < catalog.size(); ++i) {
      const auto inserted = result.emplace(catalog[i].descriptorHash, i);
      if (!inserted.second) {
        throw std::logic_error(
            "duplicate canonical descriptor hash in Vulkan emitter catalog");
      }
    }
    return result;
  }();
  return index;
}

}  // namespace

VulkanKernelFamily vulkanKernelFamily(
    const VulkanKernelEmitterInfo& emitter) {
  const uint32_t traits = emitter.traits;
  auto hasAny = [&](uint32_t mask) { return (traits & mask) != 0; };
  if (emitter.loweringContract == VulkanLoweringContract::TRIANGULAR_SOLVE) {
    return VulkanKernelFamily::TRIANGULAR_SOLVE;
  }
  if (hasAny(sd::ops::OP_TRAIT_MATMUL)) {
    return VulkanKernelFamily::MATMUL;
  }
  if (hasAny(sd::ops::OP_TRAIT_ATTENTION)) {
    return VulkanKernelFamily::ATTENTION;
  }
  if (hasAny(sd::ops::OP_TRAIT_NORMALIZATION)) {
    return VulkanKernelFamily::NORMALIZATION;
  }
  if (hasAny(sd::ops::OP_TRAIT_REDUCTION)) {
    return VulkanKernelFamily::REDUCTION;
  }
  if (hasAny(sd::ops::OP_TRAIT_CONSTANT_GENERATION)) {
    return VulkanKernelFamily::CONSTANT_GENERATION;
  }
  constexpr uint32_t kDataMovementTraits =
      sd::ops::OP_TRAIT_DATA_MOVEMENT |
      sd::ops::OP_TRAIT_VIEW_PRODUCING |
      sd::ops::OP_TRAIT_GATHER |
      sd::ops::OP_TRAIT_GATHER_ND |
      sd::ops::OP_TRAIT_CONCAT |
      sd::ops::OP_TRAIT_SPLIT |
      sd::ops::OP_TRAIT_SPLIT_V |
      sd::ops::OP_TRAIT_STACK |
      sd::ops::OP_TRAIT_SLICE |
      sd::ops::OP_TRAIT_TILE |
      sd::ops::OP_TRAIT_SCATTER_ND |
      sd::ops::OP_TRAIT_SCATTER_ND_UPDATE;
  if (hasAny(kDataMovementTraits)) {
    return VulkanKernelFamily::DATA_MOVEMENT;
  }
  if (hasAny(sd::ops::OP_TRAIT_CAST)) {
    return VulkanKernelFamily::CAST;
  }
  if (hasAny(sd::ops::OP_TRAIT_TERNARY_ELEMENTWISE)) {
    return VulkanKernelFamily::TERNARY;
  }
  if (hasAny(sd::ops::OP_TRAIT_COMPARISON)) {
    return VulkanKernelFamily::COMPARISON;
  }
  if (hasAny(sd::ops::OP_TRAIT_LOGICAL)) {
    return VulkanKernelFamily::LOGICAL;
  }
  if (hasAny(sd::ops::OP_TRAIT_BINARY_ELEMENTWISE)) {
    return VulkanKernelFamily::ELEMENTWISE_BINARY;
  }
  if (hasAny(sd::ops::OP_TRAIT_UNARY_ELEMENTWISE)) {
    return VulkanKernelFamily::ELEMENTWISE_UNARY;
  }
  throw std::logic_error(
      "Vulkan emitter descriptor has no routable addTraits family");
}

const VulkanKernelEmitterInfo* findVulkanKernelEmitter(
    sd::LongType descriptorHash) {
  const auto& index = catalogIndex();
  const auto found = index.find(descriptorHash);
  return found == index.end() ? nullptr : &buildCatalog()[found->second];
}

const std::vector<VulkanKernelEmitterInfo>& getVulkanKernelEmitterCatalog() {
  return buildCatalog();
}

const VulkanKernelEmitterInfo* findVulkanLegacyKernelEmitter(
    VulkanLegacyOpFamily family, int opNum) {
  const auto& catalog = buildLegacyCatalog();
  const auto found = catalog.indices.find(VulkanLegacyOpKey(family, opNum));
  return found == catalog.indices.end()
             ? nullptr
             : &catalog.entries[found->second];
}

const std::vector<VulkanKernelEmitterInfo>&
getVulkanLegacyKernelEmitterCatalog() {
  return buildLegacyCatalog().entries;
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_VULKAN
