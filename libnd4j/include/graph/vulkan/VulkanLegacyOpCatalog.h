/* ******************************************************************************
 *
 * Copyright (c) 2026 Eclipse Foundation
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef LIBND4J_VULKAN_LEGACY_OP_CATALOG_H
#define LIBND4J_VULKAN_LEGACY_OP_CATALOG_H

#include <system/common.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace sd {
namespace graph {

/**
 * NativeOpExecutioner legacy operation surface.
 *
 * Families that share op numbers remain distinct identities. In particular,
 * bool and integer broadcast, pairwise, and scalar surfaces must never be
 * inferred from operand data types.
 */
enum class VulkanLegacyOpFamily : uint8_t {
  BROADCAST,
  BROADCAST_BOOL,
  BROADCAST_INT,
  PAIRWISE,
  PAIRWISE_BOOL,
  PAIRWISE_INT,
  SCALAR,
  SCALAR_BOOL,
  SCALAR_INT,
  TRANSFORM_SAME,
  TRANSFORM_STRICT,
  TRANSFORM_FLOAT,
  TRANSFORM_BOOL,
  TRANSFORM_ANY,
  REDUCE_SAME,
  REDUCE_FLOAT,
  REDUCE_BOOL,
  REDUCE_LONG,
  REDUCE3,
  INDEX_REDUCE,
  SUMMARY_STATS,
  RANDOM
};

struct SD_LIB_EXPORT VulkanLegacyOpKey {
  VulkanLegacyOpFamily family;
  int opNum;

  constexpr VulkanLegacyOpKey(VulkanLegacyOpFamily familyValue, int opNumValue) noexcept
      : family(familyValue), opNum(opNumValue) {}

  constexpr bool operator==(const VulkanLegacyOpKey& other) const noexcept {
    return family == other.family && opNum == other.opNum;
  }

  constexpr bool operator!=(const VulkanLegacyOpKey& other) const noexcept {
    return !(*this == other);
  }
};

struct SD_LIB_EXPORT VulkanLegacyOpKeyHash {
  std::size_t operator()(const VulkanLegacyOpKey& key) const noexcept;
};

/**
 * Immutable identity metadata derived from one canonical legacy_ops.h tuple.
 *
 * diagnosticName is stringized from the canonical macro token. It is never
 * consulted by lookup or dispatch.
 */
class SD_LIB_EXPORT VulkanLegacyOpInfo {
 public:
  VulkanLegacyOpInfo(VulkanLegacyOpKey key,
                     const char* diagnosticName) noexcept
      : _key(key), _diagnosticName(diagnosticName) {}

  constexpr const VulkanLegacyOpKey& key() const noexcept { return _key; }
  constexpr VulkanLegacyOpFamily family() const noexcept { return _key.family; }
  constexpr int opNum() const noexcept { return _key.opNum; }
  constexpr const char* diagnosticName() const noexcept { return _diagnosticName; }

 private:
  const VulkanLegacyOpKey _key;
  const char* const _diagnosticName;
};

/** Generic DSP legacy type-code mapping for the typed Vulkan families. */
SD_LIB_EXPORT std::optional<VulkanLegacyOpFamily>
vulkanLegacyFamilyFromTypeCode(int legacyOpType);
SD_LIB_EXPORT std::optional<int>
vulkanLegacyTypeCode(VulkanLegacyOpFamily family);

class SD_LIB_EXPORT VulkanLegacyOpCatalog {
 public:
  VulkanLegacyOpCatalog() = delete;

  /** Returns nullptr when the typed legacy identity is not canonical. */
  static const VulkanLegacyOpInfo* lookup(VulkanLegacyOpFamily family, int opNum);

  /** Complete immutable catalog, in canonical family/list order. */
  static const std::vector<VulkanLegacyOpInfo>& entries();
};

/**
 * What a legacy random op (RANDOM_OPS in loops/legacy_ops.h) reads besides the
 * generator, per ops/random_ops.h and ops/special_random_ops.h: how many extra
 * arguments, and whether it reads x and y. The special samplers read y alone
 * and take y == z to mean "no y"; the ops that may read x run without it too;
 * GammaDistribution reads x, then y only together with x, each optional.
 *
 * requiredInputs counts the inputs the op cannot run without, x first. DropOut,
 * DropOutInverted and AlphaDropOut have no z-only form: the native entry point
 * without x is a stub that writes -1. ProbablisticMerge and Choice read both x
 * and y. Every other op has a form without its optional inputs.
 */
struct VulkanLegacyRandomOperands {
  int extraArguments;
  bool readsX;
  bool readsY;
  int requiredInputs = 0;
};

inline std::optional<VulkanLegacyRandomOperands> vulkanLegacyRandomOperands(
    int opNum) {
  switch (opNum) {
    case 0:  // UniformDistribution: from, to
      return VulkanLegacyRandomOperands{2, false, false};
    case 1:   // DropOut: p
    case 2:   // DropOutInverted: p
      return VulkanLegacyRandomOperands{1, true, false, 1};
    case 7:  // BernoulliDistribution: p, or x
      return VulkanLegacyRandomOperands{1, true, false};
    case 3:  // ProbablisticMerge: threshold
      return VulkanLegacyRandomOperands{1, true, true, 2};
    case 4:  // Linspace: from, to, step
      return VulkanLegacyRandomOperands{3, false, false};
    case 5:  // Choice: source x, probabilities y
      return VulkanLegacyRandomOperands{0, true, true, 2};
    case 6:   // GaussianDistribution: mean, stddev
    case 10:  // LogNormalDistribution: mean, stddev
    case 11:  // TruncatedNormalDistribution: mean, stddev
    case 8:   // BinomialDistribution: trials, p
    case 9:   // BinomialDistributionEx: trials, p
      return VulkanLegacyRandomOperands{2, false, true};
    case 12:  // AlphaDropOut: p, a, b, alphaPrime
      return VulkanLegacyRandomOperands{4, true, false, 1};
    case 13:  // ExponentialDistribution: lambda, or x
    case 14:  // ExponentialDistributionInv: lambda, or x
    case 15:  // PoissonDistribution: lambda, or x (the rate of each element)
      return VulkanLegacyRandomOperands{1, true, false};
    case 16:  // GammaDistribution: alpha, beta; x replaces alpha and, with it, y replaces beta
      return VulkanLegacyRandomOperands{2, true, true};
    default:
      return std::nullopt;
  }
}

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_VULKAN_LEGACY_OP_CATALOG_H
