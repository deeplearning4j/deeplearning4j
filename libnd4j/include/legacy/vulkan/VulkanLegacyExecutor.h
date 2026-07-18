/* ******************************************************************************
 *
 * Copyright (c) 2026 Eclipse Foundation
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef LIBND4J_VULKAN_LEGACY_EXECUTOR_H
#define LIBND4J_VULKAN_LEGACY_EXECUTOR_H

#include <config.h>
#include <system/common.h>

#if defined(SD_VULKAN) && defined(HAVE_VULKAN) && HAVE_VULKAN

#include <graph/vulkan/VulkanLegacyOpCatalog.h>

#include <string>
#include <vector>

namespace sd {

class LaunchContext;

namespace graph {

/**
 * Non-owning tensor ABI used by NativeOpExecutioner entry points.
 *
 * hostData/deviceData and both shape pointers remain owned by the caller.
 * Wrapping them for descriptor execution never transfers allocation ownership.
 */
struct SD_LIB_EXPORT VulkanLegacyTensor {
  void* hostData = nullptr;
  void* deviceData = nullptr;
  const sd::LongType* hostShapeInfo = nullptr;
  const sd::LongType* deviceShapeInfo = nullptr;
};

/**
 * Common non-owning operand and argument payload consumed by the Vulkan
 * descriptor recorder. Identity is deliberately kept out of this base so
 * generated legacy keys and direct descriptor hashes cannot be confused.
 */
struct SD_LIB_EXPORT VulkanInvocationArguments {
  std::vector<VulkanLegacyTensor> inputs;
  std::vector<VulkanLegacyTensor> outputs;
  std::vector<sd::LongType> integerArguments;
  std::vector<double> floatingArguments;
  std::vector<bool> booleanArguments;

  // Random execution ABI. Both pointers are caller-owned and remain non-owning.
  // They are meaningful only for generated legacy random operations.
  sd::Pointer randomState = nullptr;
  void* randomExtraArguments = nullptr;
};

/**
 * One typed legacy operation invocation identified by the generated
 * (family, opNum) key from legacy_ops.h. Operation names never participate in
 * routing.
 */
struct SD_LIB_EXPORT VulkanLegacyInvocation : VulkanInvocationArguments {
  VulkanLegacyOpFamily family;
  int opNum;

  VulkanLegacyInvocation(VulkanLegacyOpFamily familyValue, int opNumValue)
      : family(familyValue), opNum(opNumValue) {}
};

/**
 * Direct canonical descriptor invocation for NativeOps utilities that are not
 * members of a generated legacy family. This avoids inventing fake legacy op
 * numbers while preserving the same eager/DSP recorder path.
 */
struct SD_LIB_EXPORT VulkanDescriptorInvocation : VulkanInvocationArguments {
  sd::LongType descriptorHash;

  explicit VulkanDescriptorInvocation(sd::LongType descriptorHashValue)
      : descriptorHash(descriptorHashValue) {}
};

/**
 * Execute a canonical legacy key through the same descriptor-driven Vulkan
 * recorder used by eager custom operations and DSP replay.
 *
 * Returns VALIDATION when the canonical legacy operation has no exact
 * descriptor bridge or its operand metadata is not supported. No CPU execution
 * path exists here.
 */
SD_LIB_EXPORT Status executeVulkanLegacy(
    sd::LaunchContext* launchContext,
    const VulkanLegacyInvocation& invocation,
    std::string* errorMessage = nullptr);

/** Execute a canonical descriptor hash through the same Vulkan recorder. */
SD_LIB_EXPORT Status executeVulkanDescriptor(
    sd::LaunchContext* launchContext,
    const VulkanDescriptorInvocation& invocation,
    std::string* errorMessage = nullptr);

/** Execute or throw a diagnostic containing the exact typed legacy identity. */
SD_LIB_EXPORT void requireVulkanLegacyExecution(
    sd::LaunchContext* launchContext,
    const VulkanLegacyInvocation& invocation);

/** Execute or throw a diagnostic containing the exact descriptor hash. */
SD_LIB_EXPORT void requireVulkanDescriptorExecution(
    sd::LaunchContext* launchContext,
    const VulkanDescriptorInvocation& invocation);

}  // namespace graph
}  // namespace sd

#endif  // SD_VULKAN && HAVE_VULKAN
#endif  // LIBND4J_VULKAN_LEGACY_EXECUTOR_H
