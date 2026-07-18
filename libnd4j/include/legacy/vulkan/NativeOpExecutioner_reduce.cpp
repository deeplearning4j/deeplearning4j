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

#include <cstddef>
#include <string>

namespace sd {
namespace {

graph::VulkanLegacyTensor inputTensor(const void* hostData, const void* deviceData,
                                      const sd::LongType* hostShapeInfo,
                                      const sd::LongType* deviceShapeInfo) {
  return {const_cast<void*>(hostData), const_cast<void*>(deviceData),
          hostShapeInfo, deviceShapeInfo};
}

graph::VulkanLegacyTensor outputTensor(void* hostData, void* deviceData,
                                       const sd::LongType* hostShapeInfo,
                                       const sd::LongType* deviceShapeInfo) {
  return {hostData, deviceData, hostShapeInfo, deviceShapeInfo};
}

void requireNoOpaqueExtraParameters(const void* extraParameters) {
  if (extraParameters != nullptr) {
    THROW_EXCEPTION(
        "Vulkan legacy reduction descriptor execution cannot infer the typed "
        "argument count from non-null extra parameters");
  }
}

void appendDimensions(graph::VulkanLegacyInvocation& invocation,
                      const sd::LongType* dimensions,
                      sd::LongType dimensionLength) {
  if (dimensionLength < 0) {
    THROW_EXCEPTION("Vulkan legacy reduction received a negative dimension length");
  }
  if (dimensionLength > 0 && dimensions == nullptr) {
    THROW_EXCEPTION("Vulkan legacy reduction received no dimension data");
  }

  invocation.integerArguments.reserve(
      invocation.integerArguments.size() +
      static_cast<std::size_t>(dimensionLength));
  for (sd::LongType index = 0; index < dimensionLength; ++index) {
    invocation.integerArguments.emplace_back(dimensions[index]);
  }
}

void validateDerivedTadPair(const sd::LongType* tadShapeInfo,
                            const sd::LongType* tadOffsets,
                            const char* operand) {
  if ((tadShapeInfo == nullptr) != (tadOffsets == nullptr)) {
    std::string message =
        "Vulkan legacy reduction received incomplete derived TAD metadata for ";
    message += operand;
    THROW_EXCEPTION(message.c_str());
  }
  // TAD shape/offset buffers are a backend-specific indexing cache, not an
  // operation argument. Vulkan reconstructs indexing from the canonical
  // operand shapes and dimensions above, so these derived buffers are neither
  // dereferenced nor transported across the backend boundary.
}

void executeUnaryReduction(
    sd::LaunchContext* launchContext, graph::VulkanLegacyOpFamily family,
    int opNum, const void* hX, const sd::LongType* hXShapeInfo,
    const void* dX, const sd::LongType* dXShapeInfo, void* extraParameters,
    void* hZ, const sd::LongType* hZShapeInfo, void* dZ,
    const sd::LongType* dZShapeInfo, const sd::LongType* dimensions,
    sd::LongType dimensionLength, const bool* biasCorrected = nullptr) {
  requireNoOpaqueExtraParameters(extraParameters);

  graph::VulkanLegacyInvocation invocation(family, opNum);
  invocation.inputs.emplace_back(
      inputTensor(hX, dX, hXShapeInfo, dXShapeInfo));
  invocation.outputs.emplace_back(
      outputTensor(hZ, dZ, hZShapeInfo, dZShapeInfo));
  appendDimensions(invocation, dimensions, dimensionLength);
  const bool keepDims =
      shape::rank(hZShapeInfo) == shape::rank(hXShapeInfo);
  invocation.booleanArguments.emplace_back(keepDims);
  if (biasCorrected != nullptr) {
    invocation.booleanArguments.emplace_back(*biasCorrected);
  }
  graph::requireVulkanLegacyExecution(launchContext, invocation);
}

void executeBinaryReduction(
    sd::LaunchContext* launchContext, int opNum, const void* hX,
    const sd::LongType* hXShapeInfo, const void* dX,
    const sd::LongType* dXShapeInfo, void* extraParameters, const void* hY,
    const sd::LongType* hYShapeInfo, const void* dY,
    const sd::LongType* dYShapeInfo, void* hZ,
    const sd::LongType* hZShapeInfo, void* dZ,
    const sd::LongType* dZShapeInfo, const sd::LongType* dimensions,
    sd::LongType dimensionLength) {
  requireNoOpaqueExtraParameters(extraParameters);

  graph::VulkanLegacyInvocation invocation(
      graph::VulkanLegacyOpFamily::REDUCE3, opNum);
  invocation.inputs.emplace_back(
      inputTensor(hX, dX, hXShapeInfo, dXShapeInfo));
  invocation.inputs.emplace_back(
      inputTensor(hY, dY, hYShapeInfo, dYShapeInfo));
  invocation.outputs.emplace_back(
      outputTensor(hZ, dZ, hZShapeInfo, dZShapeInfo));
  appendDimensions(invocation, dimensions, dimensionLength);
  graph::requireVulkanLegacyExecution(launchContext, invocation);
}

}  // namespace

void NativeOpExecutioner::execReduceSame(
    sd::LaunchContext* lc, int opNum, const void* hX,
    const sd::LongType* hXShapeInfo, const void* dX,
    const sd::LongType* dXShapeInfo, void* extraParams, void* hZ,
    const sd::LongType* hZShapeInfo, void* dZ,
    const sd::LongType* dZShapeInfo, sd::LongType* dimension,
    sd::LongType dimensionLength) {
  executeUnaryReduction(
      lc, graph::VulkanLegacyOpFamily::REDUCE_SAME, opNum, hX, hXShapeInfo,
      dX, dXShapeInfo, extraParams, hZ, hZShapeInfo, dZ, dZShapeInfo,
      dimension, dimensionLength);
}

void NativeOpExecutioner::execReduceFloat(
    sd::LaunchContext* lc, int opNum, const void* hX,
    const sd::LongType* hXShapeInfo, const void* dX,
    const sd::LongType* dXShapeInfo, void* extraParams, void* hZ,
    const sd::LongType* hZShapeInfo, void* dZ,
    const sd::LongType* dZShapeInfo, sd::LongType* dimension,
    sd::LongType dimensionLength) {
  executeUnaryReduction(
      lc, graph::VulkanLegacyOpFamily::REDUCE_FLOAT, opNum, hX, hXShapeInfo,
      dX, dXShapeInfo, extraParams, hZ, hZShapeInfo, dZ, dZShapeInfo,
      dimension, dimensionLength);
}

void NativeOpExecutioner::execReduceBool(
    sd::LaunchContext* lc, int opNum, const void* hX,
    const sd::LongType* hXShapeInfo, const void* dX,
    const sd::LongType* dXShapeInfo, void* extraParams, void* hZ,
    const sd::LongType* hZShapeInfo, void* dZ,
    const sd::LongType* dZShapeInfo, sd::LongType* dimension,
    sd::LongType dimensionLength) {
  executeUnaryReduction(
      lc, graph::VulkanLegacyOpFamily::REDUCE_BOOL, opNum, hX, hXShapeInfo,
      dX, dXShapeInfo, extraParams, hZ, hZShapeInfo, dZ, dZShapeInfo,
      dimension, dimensionLength);
}

void NativeOpExecutioner::execReduceLong(
    sd::LaunchContext* lc, int opNum, const void* hX,
    const sd::LongType* hXShapeInfo, const void* dX,
    const sd::LongType* dXShapeInfo, void* extraParams, void* hZ,
    const sd::LongType* hZShapeInfo, void* dZ,
    const sd::LongType* dZShapeInfo, sd::LongType* dimension,
    sd::LongType dimensionLength) {
  executeUnaryReduction(
      lc, graph::VulkanLegacyOpFamily::REDUCE_LONG, opNum, hX, hXShapeInfo,
      dX, dXShapeInfo, extraParams, hZ, hZShapeInfo, dZ, dZShapeInfo,
      dimension, dimensionLength);
}

void NativeOpExecutioner::execReduceSameScalar(
    sd::LaunchContext* lc, int opNum, const void* hX,
    const sd::LongType* hXShapeInfo, const void* dX,
    const sd::LongType* dXShapeInfo, void* extraParams, void* hZ,
    const sd::LongType* hZShapeInfo, void* dZ,
    const sd::LongType* dZShapeInfo) {
  executeUnaryReduction(
      lc, graph::VulkanLegacyOpFamily::REDUCE_SAME, opNum, hX, hXShapeInfo,
      dX, dXShapeInfo, extraParams, hZ, hZShapeInfo, dZ, dZShapeInfo,
      nullptr, 0);
}

void NativeOpExecutioner::execReduceFloatScalar(
    sd::LaunchContext* lc, int opNum, const void* hX,
    const sd::LongType* hXShapeInfo, const void* dX,
    const sd::LongType* dXShapeInfo, void* extraParams, void* hZ,
    const sd::LongType* hZShapeInfo, void* dZ,
    const sd::LongType* dZShapeInfo) {
  executeUnaryReduction(
      lc, graph::VulkanLegacyOpFamily::REDUCE_FLOAT, opNum, hX, hXShapeInfo,
      dX, dXShapeInfo, extraParams, hZ, hZShapeInfo, dZ, dZShapeInfo,
      nullptr, 0);
}

void NativeOpExecutioner::execReduceBoolScalar(
    sd::LaunchContext* lc, int opNum, const void* hX,
    const sd::LongType* hXShapeInfo, const void* dX,
    const sd::LongType* dXShapeInfo, void* extraParams, void* hZ,
    const sd::LongType* hZShapeInfo, void* dZ,
    const sd::LongType* dZShapeInfo) {
  executeUnaryReduction(
      lc, graph::VulkanLegacyOpFamily::REDUCE_BOOL, opNum, hX, hXShapeInfo,
      dX, dXShapeInfo, extraParams, hZ, hZShapeInfo, dZ, dZShapeInfo,
      nullptr, 0);
}

void NativeOpExecutioner::execReduceLongScalar(
    sd::LaunchContext* lc, int opNum, const void* hX,
    const sd::LongType* hXShapeInfo, const void* dX,
    const sd::LongType* dXShapeInfo, void* extraParams, void* hZ,
    const sd::LongType* hZShapeInfo, void* dZ,
    const sd::LongType* dZShapeInfo) {
  executeUnaryReduction(
      lc, graph::VulkanLegacyOpFamily::REDUCE_LONG, opNum, hX, hXShapeInfo,
      dX, dXShapeInfo, extraParams, hZ, hZShapeInfo, dZ, dZShapeInfo,
      nullptr, 0);
}

void NativeOpExecutioner::execIndexReduceScalar(
    sd::LaunchContext* lc, int opNum, const void* hX,
    const sd::LongType* hXShapeInfo, const void* dX,
    const sd::LongType* dXShapeInfo, void* extraParams, void* hZ,
    const sd::LongType* hZShapeInfo, void* dZ,
    const sd::LongType* dZShapeInfo) {
  executeUnaryReduction(
      lc, graph::VulkanLegacyOpFamily::INDEX_REDUCE, opNum, hX, hXShapeInfo,
      dX, dXShapeInfo, extraParams, hZ, hZShapeInfo, dZ, dZShapeInfo,
      nullptr, 0);
}

void NativeOpExecutioner::execIndexReduce(
    sd::LaunchContext* lc, int opNum, const void* hX,
    const sd::LongType* hXShapeInfo, const void* dX,
    const sd::LongType* dXShapeInfo, void* extraParams, void* hZ,
    const sd::LongType* hZShapeInfo, void* dZ,
    const sd::LongType* dZShapeInfo, sd::LongType* dimension,
    sd::LongType dimensionLength, const sd::LongType* tadShapeInfo,
    const sd::LongType* tadOffsets) {
  validateDerivedTadPair(tadShapeInfo, tadOffsets, "index-reduce input");
  executeUnaryReduction(
      lc, graph::VulkanLegacyOpFamily::INDEX_REDUCE, opNum, hX, hXShapeInfo,
      dX, dXShapeInfo, extraParams, hZ, hZShapeInfo, dZ, dZShapeInfo,
      dimension, dimensionLength);
}

void NativeOpExecutioner::execReduce3Scalar(
    sd::LaunchContext* lc, int opNum, const void* hX,
    const sd::LongType* hXShapeInfo, const void* dX,
    const sd::LongType* dXShapeInfo, void* extraParamsVals, const void* hY,
    const sd::LongType* hYShapeInfo, const void* dY,
    const sd::LongType* dYShapeInfo, void* hZ,
    const sd::LongType* hZShapeInfo, void* dZ,
    const sd::LongType* dZShapeInfo) {
  executeBinaryReduction(
      lc, opNum, hX, hXShapeInfo, dX, dXShapeInfo, extraParamsVals, hY,
      hYShapeInfo, dY, dYShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo,
      nullptr, 0);
}

void NativeOpExecutioner::execReduce3(
    sd::LaunchContext* lc, int opNum, const void* hX,
    const sd::LongType* hXShapeInfo, const void* dX,
    const sd::LongType* dXShapeInfo, void* extraParamsVals, const void* hY,
    const sd::LongType* hYShapeInfo, const void* dY,
    const sd::LongType* dYShapeInfo, void* hZ,
    const sd::LongType* hZShapeInfo, void* dZ,
    const sd::LongType* dZShapeInfo) {
  executeBinaryReduction(
      lc, opNum, hX, hXShapeInfo, dX, dXShapeInfo, extraParamsVals, hY,
      hYShapeInfo, dY, dYShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo,
      nullptr, 0);
}

void NativeOpExecutioner::execReduce3(
    sd::LaunchContext* lc, int opNum, const void* hX,
    const sd::LongType* hXShapeInfo, const void* dX,
    const sd::LongType* dXShapeInfo, void* extraParamsVals, const void* hY,
    const sd::LongType* hYShapeInfo, const void* dY,
    const sd::LongType* dYShapeInfo, void* hZ,
    const sd::LongType* hZShapeInfo, void* dZ,
    const sd::LongType* dZShapeInfo, sd::LongType* dimension,
    sd::LongType dimensionLength, const sd::LongType* xTadOnlyShapeInfo,
    const sd::LongType* xTadOffsets,
    const sd::LongType* yTadOnlyShapeInfo,
    const sd::LongType* yTadOffsets) {
  validateDerivedTadPair(xTadOnlyShapeInfo, xTadOffsets, "reduce3 X");
  validateDerivedTadPair(yTadOnlyShapeInfo, yTadOffsets, "reduce3 Y");
  executeBinaryReduction(
      lc, opNum, hX, hXShapeInfo, dX, dXShapeInfo, extraParamsVals, hY,
      hYShapeInfo, dY, dYShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo,
      dimension, dimensionLength);
}

void NativeOpExecutioner::execReduce3All(
    sd::LaunchContext* lc, int opNum, const void* hX,
    const sd::LongType* hXShapeInfo, const void* dX,
    const sd::LongType* dXShapeInfo, void* extraParamsVals, const void* hY,
    const sd::LongType* hYShapeInfo, const void* dY,
    const sd::LongType* dYShapeInfo, void* hZ,
    const sd::LongType* hZShapeInfo, void* dZ,
    const sd::LongType* dZShapeInfo, sd::LongType* dimension,
    sd::LongType dimensionLength, const sd::LongType* xTadShapeInfo,
    const sd::LongType* xOffsets, const sd::LongType* yTadShapeInfo,
    const sd::LongType* yOffsets) {
  validateDerivedTadPair(xTadShapeInfo, xOffsets, "reduce3-all X");
  validateDerivedTadPair(yTadShapeInfo, yOffsets, "reduce3-all Y");
  executeBinaryReduction(
      lc, opNum, hX, hXShapeInfo, dX, dXShapeInfo, extraParamsVals, hY,
      hYShapeInfo, dY, dYShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo,
      dimension, dimensionLength);
}

void NativeOpExecutioner::execReduce3TAD(
    sd::LaunchContext* lc, int opNum, const void* hX,
    const sd::LongType* hXShapeInfo, const void* dX,
    const sd::LongType* dXShapeInfo, void* extraParamsVals, const void* hY,
    const sd::LongType* hYShapeInfo, const void* dY,
    const sd::LongType* dYShapeInfo, void* hZ,
    const sd::LongType* hZShapeInfo, void* dZ,
    const sd::LongType* dZShapeInfo, sd::LongType* dimension,
    sd::LongType dimensionLength, const sd::LongType* tadShapeInfo,
    const sd::LongType* tadOffsets,
    const sd::LongType* yTadShapeInfo,
    const sd::LongType* yTadOffsets) {
  validateDerivedTadPair(tadShapeInfo, tadOffsets, "reduce3-TAD X");
  validateDerivedTadPair(yTadShapeInfo, yTadOffsets, "reduce3-TAD Y");
  executeBinaryReduction(
      lc, opNum, hX, hXShapeInfo, dX, dXShapeInfo, extraParamsVals, hY,
      hYShapeInfo, dY, dYShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo,
      dimension, dimensionLength);
}

void NativeOpExecutioner::execSummaryStats(
    sd::LaunchContext* lc, int opNum, const void* hX,
    sd::LongType* hXShapeInfo, const void* dX,
    sd::LongType* dXShapeInfo, void* extraParams, void* hZ,
    sd::LongType* hZShapeInfo, void* dZ, sd::LongType* dZShapeInfo,
    sd::LongType* dimension, sd::LongType dimensionLength,
    sd::LongType* tadShapeInfo, sd::LongType* tadOffsets,
    bool biasCorrected) {
  validateDerivedTadPair(tadShapeInfo, tadOffsets, "summary-stat input");
  executeUnaryReduction(
      lc, graph::VulkanLegacyOpFamily::SUMMARY_STATS, opNum, hX, hXShapeInfo,
      dX, dXShapeInfo, extraParams, hZ, hZShapeInfo, dZ, dZShapeInfo,
      dimension, dimensionLength, &biasCorrected);
}

void NativeOpExecutioner::execSummaryStats(
    sd::LaunchContext* lc, int opNum, const void* hX,
    sd::LongType* hXShapeInfo, const void* dX,
    sd::LongType* dXShapeInfo, void* extraParams, void* hZ,
    sd::LongType* hZShapeInfo, void* dZ, sd::LongType* dZShapeInfo,
    bool biasCorrected) {
  executeUnaryReduction(
      lc, graph::VulkanLegacyOpFamily::SUMMARY_STATS, opNum, hX, hXShapeInfo,
      dX, dXShapeInfo, extraParams, hZ, hZShapeInfo, dZ, dZShapeInfo,
      nullptr, 0, &biasCorrected);
}

void NativeOpExecutioner::execSummaryStatsScalar(
    sd::LaunchContext* lc, int opNum, const void* hX,
    sd::LongType* hXShapeInfo, const void* dX,
    sd::LongType* dXShapeInfo, void* extraParams, void* hZ,
    sd::LongType* hZShapeInfo, void* dZ, sd::LongType* dZShapeInfo,
    bool biasCorrected) {
  executeUnaryReduction(
      lc, graph::VulkanLegacyOpFamily::SUMMARY_STATS, opNum, hX, hXShapeInfo,
      dX, dXShapeInfo, extraParams, hZ, hZShapeInfo, dZ, dZShapeInfo,
      nullptr, 0, &biasCorrected);
}

}  // namespace sd

#endif  // SD_VULKAN && HAVE_VULKAN
