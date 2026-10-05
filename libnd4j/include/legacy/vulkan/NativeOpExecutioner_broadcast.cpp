/* ******************************************************************************
 *
 * Copyright (c) 2026 Eclipse Foundation
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <config.h>

#if defined(SD_VULKAN) && defined(HAVE_VULKAN) && HAVE_VULKAN

#include <legacy/NativeOpExecutioner.h>
#include <legacy/vulkan/VulkanLegacyExecutor.h>

namespace sd {
namespace {

graph::VulkanLegacyTensor inputTensor(const void* h, const void* d, const sd::LongType* hs,
                                      const sd::LongType* ds) {
  return {const_cast<void*>(h), const_cast<void*>(d), hs, ds};
}

graph::VulkanLegacyTensor outputTensor(void* h, void* d, const sd::LongType* hs,
                                       const sd::LongType* ds) {
  return {h, d, hs, ds};
}

void validateDerivedTadPair(const sd::LongType* tadShapeInfo,
                            const sd::LongType* tadOffsets) {
  if ((tadShapeInfo == nullptr) != (tadOffsets == nullptr)) {
    THROW_EXCEPTION(
        "Vulkan legacy broadcast received incomplete derived TAD metadata");
  }
  // TAD shape/offset buffers are backend indexing caches. Vulkan reconstructs
  // indexing from canonical operand shapes and semantic dimensions.
}

void execBroadcastVulkan(
    sd::LaunchContext* lc, graph::VulkanLegacyOpFamily family, int opNum,
    const void* hX, const sd::LongType* hXS, const void* dX, const sd::LongType* dXS,
    const void* hY, const sd::LongType* hYS, const void* dY, const sd::LongType* dYS,
    void* hZ, const sd::LongType* hZS, void* dZ, const sd::LongType* dZS,
    void* extraParams, const sd::LongType* dimensions, sd::LongType dimensionLength,
    const sd::LongType* tadShape, const sd::LongType* tadOffsets,
    const sd::LongType* tadShapeZ, const sd::LongType* tadOffsetsZ,
    bool inverse = false) {
  if (extraParams != nullptr) {
    THROW_EXCEPTION(
        "Vulkan legacy broadcast descriptor execution cannot infer floating "
        "argument count from non-null extraParams");
  }
  if (dimensionLength < 0) {
    THROW_EXCEPTION(
        "Vulkan legacy broadcast descriptor execution requires a non-negative "
        "dimension length");
  }
  if (dimensionLength > 0 && dimensions == nullptr) {
    THROW_EXCEPTION(
        "Vulkan legacy broadcast descriptor execution received a null dimension "
        "array with a non-zero length");
  }
  validateDerivedTadPair(tadShape, tadOffsets);
  validateDerivedTadPair(tadShapeZ, tadOffsetsZ);

  graph::VulkanLegacyInvocation invocation(family, opNum);
  invocation.inputs.emplace_back(inputTensor(hX, dX, hXS, dXS));
  invocation.inputs.emplace_back(inputTensor(hY, dY, hYS, dYS));
  invocation.outputs.emplace_back(outputTensor(hZ, dZ, hZS, dZS));
  if (dimensionLength > 0) {
    invocation.integerArguments.assign(dimensions, dimensions + dimensionLength);
  }
  // Operand order remains op(X, Y); only the operand projected onto TAD axes changes.
  invocation.booleanArguments.emplace_back(inverse);
  graph::requireVulkanLegacyExecution(lc, invocation);
}

}  // namespace

void NativeOpExecutioner::execBroadcast(
    sd::LaunchContext* lc, int opNum, const void* hX, const sd::LongType* hXS,
    const void* dX, const sd::LongType* dXS, const void* hY,
    const sd::LongType* hYS, const void* dY, const sd::LongType* dYS, void* hZ,
    const sd::LongType* hZS, void* dZ, const sd::LongType* dZS,
    sd::LongType* dimensions, sd::LongType dimensionLength,
    const sd::LongType* tadShape, const sd::LongType* tadOffsets,
    const sd::LongType* tadShapeZ, const sd::LongType* tadOffsetsZ) {
  execBroadcastVulkan(lc, graph::VulkanLegacyOpFamily::BROADCAST, opNum, hX, hXS, dX,
                      dXS, hY, hYS, dY, dYS, hZ, hZS, dZ, dZS, nullptr, dimensions,
                      dimensionLength, tadShape, tadOffsets, tadShapeZ, tadOffsetsZ);
}

void NativeOpExecutioner::execBroadcast(
    sd::LaunchContext* lc, int opNum, const void* hX, const sd::LongType* hXS,
    const void* dX, const sd::LongType* dXS, const void* hY,
    const sd::LongType* hYS, const void* dY, const sd::LongType* dYS, void* hZ,
    const sd::LongType* hZS, void* dZ, const sd::LongType* dZS) {
  execBroadcastVulkan(lc, graph::VulkanLegacyOpFamily::BROADCAST, opNum, hX, hXS, dX,
                      dXS, hY, hYS, dY, dYS, hZ, hZS, dZ, dZS, nullptr, nullptr, 0,
                      nullptr, nullptr, nullptr, nullptr);
}

void NativeOpExecutioner::execInverseBroadcast(
    sd::LaunchContext* lc, int opNum, const void* hX, const sd::LongType* hXS,
    const void* dX, const sd::LongType* dXS, const void* hY,
    const sd::LongType* hYS, const void* dY, const sd::LongType* dYS, void* hZ,
    const sd::LongType* hZS, void* dZ, const sd::LongType* dZS,
    sd::LongType* dimensions, sd::LongType dimensionLength,
    const sd::LongType* tadShape, const sd::LongType* tadOffsets,
    const sd::LongType* tadShapeZ, const sd::LongType* tadOffsetsZ) {
  execBroadcastVulkan(lc, graph::VulkanLegacyOpFamily::BROADCAST, opNum, hX, hXS, dX,
                      dXS, hY, hYS, dY, dYS, hZ, hZS, dZ, dZS, nullptr, dimensions,
                      dimensionLength, tadShape, tadOffsets, tadShapeZ, tadOffsetsZ, true);
}

void NativeOpExecutioner::execBroadcastBool(
    sd::LaunchContext* lc, int opNum, const void* hX, const sd::LongType* hXS,
    const void* dX, const sd::LongType* dXS, const void* hY,
    const sd::LongType* hYS, const void* dY, const sd::LongType* dYS, void* hZ,
    const sd::LongType* hZS, void* dZ, const sd::LongType* dZS, void* extraParams,
    sd::LongType* dimensions, sd::LongType dimensionLength,
    const sd::LongType* tadShape, const sd::LongType* tadOffsets,
    const sd::LongType* tadShapeZ, const sd::LongType* tadOffsetsZ) {
  execBroadcastVulkan(lc, graph::VulkanLegacyOpFamily::BROADCAST_BOOL, opNum, hX, hXS,
                      dX, dXS, hY, hYS, dY, dYS, hZ, hZS, dZ, dZS, extraParams,
                      dimensions, dimensionLength, tadShape, tadOffsets, tadShapeZ,
                      tadOffsetsZ);
}

void NativeOpExecutioner::execBroadcastBool(
    sd::LaunchContext* lc, int opNum, const void* hX, const sd::LongType* hXS,
    const void* dX, const sd::LongType* dXS, const void* hY,
    const sd::LongType* hYS, const void* dY, const sd::LongType* dYS, void* hZ,
    const sd::LongType* hZS, void* dZ, const sd::LongType* dZS,
    void* extraParams) {
  execBroadcastVulkan(lc, graph::VulkanLegacyOpFamily::BROADCAST_BOOL, opNum, hX, hXS,
                      dX, dXS, hY, hYS, dY, dYS, hZ, hZS, dZ, dZS, extraParams,
                      nullptr, 0, nullptr, nullptr, nullptr, nullptr);
}

void NativeOpExecutioner::execInverseBroadcastBool(
    sd::LaunchContext* lc, int opNum, const void* hX, const sd::LongType* hXS,
    const void* dX, const sd::LongType* dXS, const void* hY,
    const sd::LongType* hYS, const void* dY, const sd::LongType* dYS, void* hZ,
    const sd::LongType* hZS, void* dZ, const sd::LongType* dZS,
    void* extraParams, sd::LongType* dimensions, sd::LongType dimensionLength,
    const sd::LongType* tadShape, const sd::LongType* tadOffsets,
    const sd::LongType* tadShapeZ, const sd::LongType* tadOffsetsZ) {
  execBroadcastVulkan(lc, graph::VulkanLegacyOpFamily::BROADCAST_BOOL, opNum, hX, hXS,
                      dX, dXS, hY, hYS, dY, dYS, hZ, hZS, dZ, dZS, extraParams,
                      dimensions, dimensionLength, tadShape, tadOffsets, tadShapeZ,
                      tadOffsetsZ, true);
}

void NativeOpExecutioner::execBroadcastInt(
    sd::LaunchContext* lc, int opNum, const void* hX, const sd::LongType* hXS,
    const void* dX, const sd::LongType* dXS, const void* hY,
    const sd::LongType* hYS, const void* dY, const sd::LongType* dYS, void* hZ,
    const sd::LongType* hZS, void* dZ, const sd::LongType* dZS,
    sd::LongType* dimensions, sd::LongType dimensionLength,
    const sd::LongType* tadShape, const sd::LongType* tadOffsets,
    const sd::LongType* tadShapeZ, const sd::LongType* tadOffsetsZ) {
  execBroadcastVulkan(lc, graph::VulkanLegacyOpFamily::BROADCAST_INT, opNum, hX, hXS,
                      dX, dXS, hY, hYS, dY, dYS, hZ, hZS, dZ, dZS, nullptr,
                      dimensions, dimensionLength, tadShape, tadOffsets, tadShapeZ,
                      tadOffsetsZ);
}

void NativeOpExecutioner::execBroadcastInt(
    sd::LaunchContext* lc, int opNum, const void* hX, const sd::LongType* hXS,
    const void* dX, const sd::LongType* dXS, const void* hY,
    const sd::LongType* hYS, const void* dY, const sd::LongType* dYS, void* hZ,
    const sd::LongType* hZS, void* dZ, const sd::LongType* dZS) {
  execBroadcastVulkan(lc, graph::VulkanLegacyOpFamily::BROADCAST_INT, opNum, hX, hXS,
                      dX, dXS, hY, hYS, dY, dYS, hZ, hZS, dZ, dZS, nullptr,
                      nullptr, 0, nullptr, nullptr, nullptr, nullptr);
}

void NativeOpExecutioner::execInverseBroadcastInt(
    sd::LaunchContext* lc, int opNum, const void* hX, const sd::LongType* hXS,
    const void* dX, const sd::LongType* dXS, const void* hY,
    const sd::LongType* hYS, const void* dY, const sd::LongType* dYS, void* hZ,
    const sd::LongType* hZS, void* dZ, const sd::LongType* dZS,
    sd::LongType* dimensions, sd::LongType dimensionLength,
    const sd::LongType* tadShape, const sd::LongType* tadOffsets,
    const sd::LongType* tadShapeZ, const sd::LongType* tadOffsetsZ) {
  execBroadcastVulkan(lc, graph::VulkanLegacyOpFamily::BROADCAST_INT, opNum, hX, hXS,
                      dX, dXS, hY, hYS, dY, dYS, hZ, hZS, dZ, dZS, nullptr,
                      dimensions, dimensionLength, tadShape, tadOffsets, tadShapeZ,
                      tadOffsetsZ, true);
}

namespace {
void execBroadcastVulkan(sd::LaunchContext* lc, graph::VulkanLegacyOpFamily family,
    int opNum, const sd::LegacyTensorArg& x, const sd::LegacyTensorArg& y,
    const sd::LegacyTensorArg& z, void* extraParams,
    const sd::LongType* dimensions, sd::LongType dimensionLength,
    const sd::LongType* tadShape, const sd::LongType* tadOffsets,
    const sd::LongType* tadShapeZ, const sd::LongType* tadOffsetsZ,
    bool inverse = false) {
  if (extraParams != nullptr)
    THROW_EXCEPTION("Vulkan legacy broadcast descriptor execution cannot infer floating argument count from non-null extraParams");
  if (dimensionLength < 0 || (dimensionLength > 0 && dimensions == nullptr))
    THROW_EXCEPTION("Vulkan legacy broadcast requires valid semantic dimensions");
  validateDerivedTadPair(tadShape, tadOffsets);
  validateDerivedTadPair(tadShapeZ, tadOffsetsZ);
  graph::VulkanLegacyInvocation invocation(family, opNum);
  invocation.inputs.emplace_back(graph::VulkanLegacyTensor::fromArg(x));
  invocation.inputs.emplace_back(graph::VulkanLegacyTensor::fromArg(y));
  invocation.outputs.emplace_back(graph::VulkanLegacyTensor::fromArg(z));
  if (dimensionLength > 0)
    invocation.integerArguments.assign(dimensions, dimensions + dimensionLength);
  // Operand order remains op(X, Y); only the operand projected onto TAD axes changes.
  invocation.booleanArguments.emplace_back(inverse);
  graph::requireVulkanLegacyExecution(lc, invocation);
}
}  // namespace

#define SD_VULKAN_BROADCAST(NAME, FAMILY, INVERSE) \
void NativeOpExecutioner::NAME(sd::LaunchContext* lc, int opNum, \
    const sd::LegacyTensorArg& x, const sd::LegacyTensorArg& y, const sd::LegacyTensorArg& z, \
    sd::LongType* dimension, sd::LongType dimensionLength, \
    const sd::LongType* tadShape, const sd::LongType* tadOffsets, \
    const sd::LongType* tadShapeZ, const sd::LongType* tadOffsetsZ) { \
  execBroadcastVulkan(lc, graph::VulkanLegacyOpFamily::FAMILY, opNum, x, y, z, nullptr, \
      dimension, dimensionLength, tadShape, tadOffsets, tadShapeZ, tadOffsetsZ, INVERSE); \
}
SD_VULKAN_BROADCAST(execBroadcast, BROADCAST, false)
SD_VULKAN_BROADCAST(execInverseBroadcast, BROADCAST, true)
SD_VULKAN_BROADCAST(execBroadcastInt, BROADCAST_INT, false)
SD_VULKAN_BROADCAST(execInverseBroadcastInt, BROADCAST_INT, true)
#undef SD_VULKAN_BROADCAST
#define SD_VULKAN_BROADCAST_SIMPLE(NAME, FAMILY) \
void NativeOpExecutioner::NAME(sd::LaunchContext* lc, int opNum, \
    const sd::LegacyTensorArg& x, const sd::LegacyTensorArg& y, const sd::LegacyTensorArg& z) { \
  execBroadcastVulkan(lc, graph::VulkanLegacyOpFamily::FAMILY, opNum, x, y, z, nullptr, \
      nullptr, 0, nullptr, nullptr, nullptr, nullptr); \
}
SD_VULKAN_BROADCAST_SIMPLE(execBroadcast, BROADCAST)
SD_VULKAN_BROADCAST_SIMPLE(execBroadcastInt, BROADCAST_INT)
#undef SD_VULKAN_BROADCAST_SIMPLE
#define SD_VULKAN_BROADCAST_BOOL(NAME, INVERSE) \
void NativeOpExecutioner::NAME(sd::LaunchContext* lc, int opNum, \
    const sd::LegacyTensorArg& x, const sd::LegacyTensorArg& y, const sd::LegacyTensorArg& z, \
    void* extraParams, sd::LongType* dimension, sd::LongType dimensionLength, \
    const sd::LongType* tadShape, const sd::LongType* tadOffsets, \
    const sd::LongType* tadShapeZ, const sd::LongType* tadOffsetsZ) { \
  execBroadcastVulkan(lc, graph::VulkanLegacyOpFamily::BROADCAST_BOOL, opNum, x, y, z, extraParams, \
      dimension, dimensionLength, tadShape, tadOffsets, tadShapeZ, tadOffsetsZ, INVERSE); \
}
SD_VULKAN_BROADCAST_BOOL(execBroadcastBool, false)
SD_VULKAN_BROADCAST_BOOL(execInverseBroadcastBool, true)
#undef SD_VULKAN_BROADCAST_BOOL
void NativeOpExecutioner::execBroadcastBool(sd::LaunchContext* lc, int opNum,
    const sd::LegacyTensorArg& x, const sd::LegacyTensorArg& y, const sd::LegacyTensorArg& z,
    void* extraParams) {
  execBroadcastVulkan(lc, graph::VulkanLegacyOpFamily::BROADCAST_BOOL, opNum, x, y, z, extraParams,
      nullptr, 0, nullptr, nullptr, nullptr, nullptr);
}

}  // namespace sd

#endif  // SD_VULKAN && HAVE_VULKAN
