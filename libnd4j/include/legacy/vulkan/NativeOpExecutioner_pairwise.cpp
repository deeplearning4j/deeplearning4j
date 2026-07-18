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

graph::VulkanLegacyTensor inputTensor(const void* hostData, const void* deviceData,
                                      const sd::LongType* hostShapeInfo,
                                      const sd::LongType* deviceShapeInfo) {
  return {const_cast<void*>(hostData), const_cast<void*>(deviceData), hostShapeInfo,
          deviceShapeInfo};
}

graph::VulkanLegacyTensor outputTensor(void* hostData, void* deviceData,
                                       const sd::LongType* hostShapeInfo,
                                       const sd::LongType* deviceShapeInfo) {
  return {hostData, deviceData, hostShapeInfo, deviceShapeInfo};
}

void execPairwiseVulkan(
    sd::LaunchContext* launchContext, graph::VulkanLegacyOpFamily family, int opNum,
    const void* hX, const sd::LongType* hXShapeInfo, const void* dX,
    const sd::LongType* dXShapeInfo, const void* hY, const sd::LongType* hYShapeInfo,
    const void* dY, const sd::LongType* dYShapeInfo, void* hZ,
    const sd::LongType* hZShapeInfo, void* dZ, const sd::LongType* dZShapeInfo,
    void* extraParams) {
  if (extraParams != nullptr) {
    THROW_EXCEPTION(
        "Vulkan legacy descriptor execution cannot infer floating argument count from "
        "non-null extraParams");
  }

  graph::VulkanLegacyInvocation invocation(family, opNum);
  invocation.inputs.emplace_back(inputTensor(hX, dX, hXShapeInfo, dXShapeInfo));
  invocation.inputs.emplace_back(inputTensor(hY, dY, hYShapeInfo, dYShapeInfo));
  invocation.outputs.emplace_back(outputTensor(hZ, dZ, hZShapeInfo, dZShapeInfo));
  graph::requireVulkanLegacyExecution(launchContext, invocation);
}

}  // namespace

void NativeOpExecutioner::execPairwiseTransform(
    sd::LaunchContext* lc, int opNum, const void* hX, const sd::LongType* hXShapeInfo,
    const void* dX, const sd::LongType* dXShapeInfo, const void* hY,
    const sd::LongType* hYShapeInfo, const void* dY, const sd::LongType* dYShapeInfo,
    void* hZ, const sd::LongType* hZShapeInfo, void* dZ,
    const sd::LongType* dZShapeInfo, void* extraParams) {
  execPairwiseVulkan(lc, graph::VulkanLegacyOpFamily::PAIRWISE, opNum, hX, hXShapeInfo,
                     dX, dXShapeInfo, hY, hYShapeInfo, dY, dYShapeInfo, hZ, hZShapeInfo,
                     dZ, dZShapeInfo, extraParams);
}

void NativeOpExecutioner::execPairwiseBoolTransform(
    sd::LaunchContext* lc, int opNum, const void* hX, const sd::LongType* hXShapeInfo,
    const void* dX, const sd::LongType* dXShapeInfo, const void* hY,
    const sd::LongType* hYShapeInfo, const void* dY, const sd::LongType* dYShapeInfo,
    void* hZ, const sd::LongType* hZShapeInfo, void* dZ,
    const sd::LongType* dZShapeInfo, void* extraParams) {
  execPairwiseVulkan(lc, graph::VulkanLegacyOpFamily::PAIRWISE_BOOL, opNum, hX,
                     hXShapeInfo, dX, dXShapeInfo, hY, hYShapeInfo, dY, dYShapeInfo, hZ,
                     hZShapeInfo, dZ, dZShapeInfo, extraParams);
}

void NativeOpExecutioner::execPairwiseIntTransform(
    sd::LaunchContext* lc, int opNum, const void* hX, const sd::LongType* hXShapeInfo,
    const void* dX, const sd::LongType* dXShapeInfo, const void* hY,
    const sd::LongType* hYShapeInfo, const void* dY, const sd::LongType* dYShapeInfo,
    void* hZ, const sd::LongType* hZShapeInfo, void* dZ,
    const sd::LongType* dZShapeInfo, void* extraParams) {
  execPairwiseVulkan(lc, graph::VulkanLegacyOpFamily::PAIRWISE_INT, opNum, hX,
                     hXShapeInfo, dX, dXShapeInfo, hY, hYShapeInfo, dY, dYShapeInfo, hZ,
                     hZShapeInfo, dZ, dZShapeInfo, extraParams);
}

}  // namespace sd

#endif  // SD_VULKAN && HAVE_VULKAN
