/* ******************************************************************************
 *
 * Copyright (c) 2026 Eclipse Foundation
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <graph/vulkan/VulkanLegacyOpCatalog.h>

#include <graph/LegacyOpTypeCodes.h>
#include <loops/legacy_ops.h>
#include <system/op_boilerplate.h>

#include <array>
#include <stdexcept>
#include <unordered_map>

namespace sd {
namespace graph {
namespace {

struct CatalogData {
  std::vector<VulkanLegacyOpInfo> entries;
  std::unordered_map<VulkanLegacyOpKey, std::size_t, VulkanLegacyOpKeyHash> indices;
};

struct LegacyFamilyTypeCode {
  VulkanLegacyOpFamily family;
  int typeCode;
};

constexpr std::array<LegacyFamilyTypeCode, 22> kLegacyFamilyTypeCodes{{
    {VulkanLegacyOpFamily::BROADCAST, LEGACY_BROADCAST},
    {VulkanLegacyOpFamily::BROADCAST_BOOL, LEGACY_BROADCAST_BOOL},
    {VulkanLegacyOpFamily::PAIRWISE, LEGACY_PAIRWISE_TRANSFORM},
    {VulkanLegacyOpFamily::PAIRWISE_BOOL, LEGACY_PAIRWISE_BOOL},
    {VulkanLegacyOpFamily::SCALAR, LEGACY_SCALAR},
    {VulkanLegacyOpFamily::SCALAR_BOOL, LEGACY_SCALAR_BOOL},
    {VulkanLegacyOpFamily::TRANSFORM_SAME, LEGACY_TRANSFORM_SAME},
    {VulkanLegacyOpFamily::TRANSFORM_STRICT, LEGACY_TRANSFORM_STRICT},
    {VulkanLegacyOpFamily::TRANSFORM_FLOAT, LEGACY_TRANSFORM_FLOAT},
    {VulkanLegacyOpFamily::TRANSFORM_BOOL, LEGACY_TRANSFORM_BOOL},
    {VulkanLegacyOpFamily::TRANSFORM_ANY, LEGACY_TRANSFORM_ANY},
    {VulkanLegacyOpFamily::BROADCAST_INT, LEGACY_BROADCAST_INT},
    {VulkanLegacyOpFamily::PAIRWISE_INT, LEGACY_PAIRWISE_INT},
    {VulkanLegacyOpFamily::SCALAR_INT, LEGACY_SCALAR_INT},
    {VulkanLegacyOpFamily::REDUCE_SAME, LEGACY_REDUCE_SAME},
    {VulkanLegacyOpFamily::REDUCE_FLOAT, LEGACY_REDUCE_FLOAT},
    {VulkanLegacyOpFamily::REDUCE_BOOL, LEGACY_REDUCE_BOOL},
    {VulkanLegacyOpFamily::REDUCE_LONG, LEGACY_REDUCE_LONG},
    {VulkanLegacyOpFamily::REDUCE3, LEGACY_REDUCE3},
    {VulkanLegacyOpFamily::INDEX_REDUCE, LEGACY_INDEX_REDUCE},
    {VulkanLegacyOpFamily::SUMMARY_STATS, LEGACY_STATS},
    {VulkanLegacyOpFamily::RANDOM, LEGACY_RANDOM},
}};

void registerEntry(CatalogData& catalog, VulkanLegacyOpFamily family,
                   int opNum, const char* diagnosticName) {
  const VulkanLegacyOpKey key(family, opNum);
  const std::size_t index = catalog.entries.size();
  const auto inserted = catalog.indices.emplace(key, index);
  if (!inserted.second) {
    throw std::logic_error("Duplicate Vulkan legacy operation identity");
  }
  catalog.entries.emplace_back(key, diagnosticName);
}

#define SD_VULKAN_LEGACY_FIRST(tuple) SD_VULKAN_LEGACY_FIRST_IMPL tuple
#define SD_VULKAN_LEGACY_FIRST_IMPL(opNum, token) opNum
#define SD_VULKAN_LEGACY_SECOND(tuple) SD_VULKAN_LEGACY_SECOND_IMPL tuple
#define SD_VULKAN_LEGACY_SECOND_IMPL(opNum, token) token
#define SD_VULKAN_LEGACY_STRINGIZE(value) SD_VULKAN_LEGACY_STRINGIZE_IMPL(value)
#define SD_VULKAN_LEGACY_STRINGIZE_IMPL(value) #value
#define SD_VULKAN_LEGACY_REGISTER(family, tuple)                            \
  registerEntry(catalog, family, SD_VULKAN_LEGACY_FIRST(tuple),             \
                SD_VULKAN_LEGACY_STRINGIZE(SD_VULKAN_LEGACY_SECOND(tuple)));

#define SD_VULKAN_LEGACY_REGISTER_LIST(family, list) \
  FOR_EACH_DIRECT(SD_VULKAN_LEGACY_REGISTER, family, OPS_A(list))

const CatalogData& catalogData() {
  static const CatalogData data = [] {
    CatalogData catalog;

    SD_VULKAN_LEGACY_REGISTER_LIST(VulkanLegacyOpFamily::BROADCAST, BROADCAST_OPS)
    SD_VULKAN_LEGACY_REGISTER_LIST(VulkanLegacyOpFamily::BROADCAST_BOOL, BROADCAST_BOOL_OPS)
    SD_VULKAN_LEGACY_REGISTER_LIST(VulkanLegacyOpFamily::BROADCAST_INT, BROADCAST_INT_OPS)
    SD_VULKAN_LEGACY_REGISTER_LIST(VulkanLegacyOpFamily::PAIRWISE, PAIRWISE_TRANSFORM_OPS)
    SD_VULKAN_LEGACY_REGISTER_LIST(VulkanLegacyOpFamily::PAIRWISE_BOOL, PAIRWISE_BOOL_OPS)
    SD_VULKAN_LEGACY_REGISTER_LIST(VulkanLegacyOpFamily::PAIRWISE_INT, PAIRWISE_INT_OPS)
    SD_VULKAN_LEGACY_REGISTER_LIST(VulkanLegacyOpFamily::SCALAR, SCALAR_OPS)
    SD_VULKAN_LEGACY_REGISTER_LIST(VulkanLegacyOpFamily::SCALAR_BOOL, SCALAR_BOOL_OPS)
    SD_VULKAN_LEGACY_REGISTER_LIST(VulkanLegacyOpFamily::SCALAR_INT, SCALAR_INT_OPS)
    SD_VULKAN_LEGACY_REGISTER_LIST(VulkanLegacyOpFamily::TRANSFORM_SAME, TRANSFORM_SAME_OPS)
    SD_VULKAN_LEGACY_REGISTER_LIST(VulkanLegacyOpFamily::TRANSFORM_STRICT, TRANSFORM_STRICT_OPS)
    SD_VULKAN_LEGACY_REGISTER_LIST(VulkanLegacyOpFamily::TRANSFORM_FLOAT, TRANSFORM_FLOAT_OPS)
    SD_VULKAN_LEGACY_REGISTER_LIST(VulkanLegacyOpFamily::TRANSFORM_BOOL, TRANSFORM_BOOL_OPS)
    SD_VULKAN_LEGACY_REGISTER_LIST(VulkanLegacyOpFamily::TRANSFORM_ANY, TRANSFORM_ANY_OPS)
    SD_VULKAN_LEGACY_REGISTER_LIST(VulkanLegacyOpFamily::REDUCE_SAME, REDUCE_SAME_OPS)
    SD_VULKAN_LEGACY_REGISTER_LIST(VulkanLegacyOpFamily::REDUCE_FLOAT, REDUCE_FLOAT_OPS)
    SD_VULKAN_LEGACY_REGISTER_LIST(VulkanLegacyOpFamily::REDUCE_BOOL, REDUCE_BOOL_OPS)
    SD_VULKAN_LEGACY_REGISTER_LIST(VulkanLegacyOpFamily::REDUCE_LONG, REDUCE_LONG_OPS)
    SD_VULKAN_LEGACY_REGISTER_LIST(VulkanLegacyOpFamily::REDUCE3, REDUCE3_OPS)
    SD_VULKAN_LEGACY_REGISTER_LIST(VulkanLegacyOpFamily::INDEX_REDUCE, INDEX_REDUCE_OPS)
    SD_VULKAN_LEGACY_REGISTER_LIST(VulkanLegacyOpFamily::SUMMARY_STATS, SUMMARY_STATS_OPS)
    SD_VULKAN_LEGACY_REGISTER_LIST(VulkanLegacyOpFamily::RANDOM, RANDOM_OPS)

    return catalog;
  }();
  return data;
}

#undef SD_VULKAN_LEGACY_REGISTER_LIST
#undef SD_VULKAN_LEGACY_REGISTER
#undef SD_VULKAN_LEGACY_STRINGIZE_IMPL
#undef SD_VULKAN_LEGACY_STRINGIZE
#undef SD_VULKAN_LEGACY_SECOND_IMPL
#undef SD_VULKAN_LEGACY_SECOND
#undef SD_VULKAN_LEGACY_FIRST_IMPL
#undef SD_VULKAN_LEGACY_FIRST

}  // namespace

std::optional<VulkanLegacyOpFamily>
vulkanLegacyFamilyFromTypeCode(int legacyOpType) {
  for (const auto& mapping : kLegacyFamilyTypeCodes) {
    if (mapping.typeCode == legacyOpType) return mapping.family;
  }
  return std::nullopt;
}

std::optional<int> vulkanLegacyTypeCode(VulkanLegacyOpFamily family) {
  for (const auto& mapping : kLegacyFamilyTypeCodes) {
    if (mapping.family == family) return mapping.typeCode;
  }
  return std::nullopt;
}

std::size_t VulkanLegacyOpKeyHash::operator()(const VulkanLegacyOpKey& key) const noexcept {
  const std::size_t family = static_cast<std::size_t>(key.family);
  const std::size_t opNum = std::hash<int>{}(key.opNum);
  return opNum ^ (family + 0x9e3779b9U + (opNum << 6U) + (opNum >> 2U));
}

const VulkanLegacyOpInfo* VulkanLegacyOpCatalog::lookup(VulkanLegacyOpFamily family, int opNum) {
  const auto& data = catalogData();
  const auto found = data.indices.find(VulkanLegacyOpKey(family, opNum));
  if (found == data.indices.end()) return nullptr;
  return &data.entries[found->second];
}

const std::vector<VulkanLegacyOpInfo>& VulkanLegacyOpCatalog::entries() { return catalogData().entries; }

}  // namespace graph
}  // namespace sd
