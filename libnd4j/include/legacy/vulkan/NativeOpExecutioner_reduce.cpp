/* ******************************************************************************
 *
 * Copyright (c) 2026 Eclipse Foundation
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <config.h>

#if defined(SD_VULKAN) && defined(HAVE_VULKAN) && HAVE_VULKAN

#include <array/ArrayOptions.h>
#include <system/op_enums.h>
#include <system/type_boilerplate.h>
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

template <typename Z>
void appendReduce3Epsilon(graph::VulkanLegacyInvocation& invocation, const void* extraParameters) {
  // Native Reduce3 extras are output-typed; slots 0/1 are accumulator scratch,
  // and EqualsWithEps alone has a mathematical parameter in slot 2.
  invocation.floatingArguments.emplace_back(
      static_cast<double>(static_cast<const Z*>(extraParameters)[2]));
}

void configureReduce3(graph::VulkanLegacyInvocation& invocation, const void* extraParameters,
                      const sd::LongType* outputShapeInfo, const sd::LongType* dimensions,
                      sd::LongType dimensionLength, bool allPairs) {
  appendDimensions(invocation, dimensions, dimensionLength);
  invocation.booleanArguments = {false, allPairs};
  if (extraParameters == nullptr) return;
  if (graph::VulkanLegacyOpCatalog::lookup(invocation.family, invocation.opNum) == nullptr)
    THROW_EXCEPTION("Vulkan Reduce3 received an unknown extra-parameter ABI");
  if (invocation.opNum == static_cast<int>(sd::reduce3::EqualsWithEps)) {
    if (outputShapeInfo == nullptr)
      THROW_EXCEPTION("Vulkan Reduce3 extra parameters require the output dtype");
    BUILD_SINGLE_SELECTOR(sd::ArrayOptions::dataType(outputShapeInfo), appendReduce3Epsilon,
                          (invocation, extraParameters), SD_FLOAT_TYPES);
  }
  // The other canonical Reduce3 ops supply scratch initialization only. Do not
  // transport it as TArgs or interpret it as keepDims/configuration.
}

void executeBinaryReduction(
    sd::LaunchContext* launchContext, int opNum, const void* hX,
    const sd::LongType* hXShapeInfo, const void* dX,
    const sd::LongType* dXShapeInfo, void* extraParameters, const void* hY,
    const sd::LongType* hYShapeInfo, const void* dY,
    const sd::LongType* dYShapeInfo, void* hZ,
    const sd::LongType* hZShapeInfo, void* dZ,
    const sd::LongType* dZShapeInfo, const sd::LongType* dimensions,
    sd::LongType dimensionLength, bool allPairs = false) {
  graph::VulkanLegacyInvocation invocation(
      graph::VulkanLegacyOpFamily::REDUCE3, opNum);
  invocation.inputs.emplace_back(
      inputTensor(hX, dX, hXShapeInfo, dXShapeInfo));
  invocation.inputs.emplace_back(
      inputTensor(hY, dY, hYShapeInfo, dYShapeInfo));
  invocation.outputs.emplace_back(
      outputTensor(hZ, dZ, hZShapeInfo, dZShapeInfo));
  configureReduce3(invocation, extraParameters, hZShapeInfo, dimensions, dimensionLength, allPairs);
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
      dimension, dimensionLength, true);
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

namespace {
void executeUnaryReduction(sd::LaunchContext* lc, graph::VulkanLegacyOpFamily family,
    int opNum, const sd::LegacyTensorArg& x, void* extraParams, const sd::LegacyTensorArg& z,
    const sd::LongType* dimension, sd::LongType dimensionLength, const bool* biasCorrected = nullptr) {
  requireNoOpaqueExtraParameters(extraParams);
  graph::VulkanLegacyInvocation invocation(family, opNum);
  invocation.inputs.emplace_back(graph::VulkanLegacyTensor::fromArg(x));
  invocation.outputs.emplace_back(graph::VulkanLegacyTensor::fromArg(z));
  appendDimensions(invocation, dimension, dimensionLength);
  invocation.booleanArguments.emplace_back(shape::rank(z.hostShapeInfo) == shape::rank(x.hostShapeInfo));
  if (biasCorrected != nullptr) invocation.booleanArguments.emplace_back(*biasCorrected);
  graph::requireVulkanLegacyExecution(lc, invocation);
}

void executeBinaryReduction(sd::LaunchContext* lc, int opNum,
    const sd::LegacyTensorArg& x, void* extraParams, const sd::LegacyTensorArg& y,
    const sd::LegacyTensorArg& z, const sd::LongType* dimension, sd::LongType dimensionLength,
    bool allPairs = false) {
  graph::VulkanLegacyInvocation invocation(graph::VulkanLegacyOpFamily::REDUCE3, opNum);
  invocation.inputs.emplace_back(graph::VulkanLegacyTensor::fromArg(x));
  invocation.inputs.emplace_back(graph::VulkanLegacyTensor::fromArg(y));
  invocation.outputs.emplace_back(graph::VulkanLegacyTensor::fromArg(z));
  configureReduce3(invocation, extraParams, z.hostShapeInfo, dimension, dimensionLength, allPairs);
  graph::requireVulkanLegacyExecution(lc, invocation);
}
}  // namespace

#define SD_VULKAN_REDUCE(NAME, FAMILY) \
void NativeOpExecutioner::NAME(sd::LaunchContext* lc, int opNum, \
    const sd::LegacyTensorArg& x, void* extraParams, const sd::LegacyTensorArg& z, \
    sd::LongType* dimension, sd::LongType dimensionLength) { \
  executeUnaryReduction(lc, graph::VulkanLegacyOpFamily::FAMILY, opNum, x, extraParams, z, \
                        dimension, dimensionLength); \
}
SD_VULKAN_REDUCE(execReduceFloat, REDUCE_FLOAT)
SD_VULKAN_REDUCE(execReduceSame, REDUCE_SAME)
SD_VULKAN_REDUCE(execReduceBool, REDUCE_BOOL)
SD_VULKAN_REDUCE(execReduceLong, REDUCE_LONG)
#undef SD_VULKAN_REDUCE
#define SD_VULKAN_REDUCE_SCALAR(NAME, FAMILY) \
void NativeOpExecutioner::NAME(sd::LaunchContext* lc, int opNum, \
    const sd::LegacyTensorArg& x, void* extraParams, const sd::LegacyTensorArg& z) { \
  executeUnaryReduction(lc, graph::VulkanLegacyOpFamily::FAMILY, opNum, x, extraParams, z, nullptr, 0); \
}
SD_VULKAN_REDUCE_SCALAR(execReduceFloatScalar, REDUCE_FLOAT)
SD_VULKAN_REDUCE_SCALAR(execReduceSameScalar, REDUCE_SAME)
SD_VULKAN_REDUCE_SCALAR(execReduceBoolScalar, REDUCE_BOOL)
SD_VULKAN_REDUCE_SCALAR(execReduceLongScalar, REDUCE_LONG)
SD_VULKAN_REDUCE_SCALAR(execIndexReduceScalar, INDEX_REDUCE)
#undef SD_VULKAN_REDUCE_SCALAR
void NativeOpExecutioner::execIndexReduce(sd::LaunchContext* lc, int opNum,
    const sd::LegacyTensorArg& x, void* extraParams, const sd::LegacyTensorArg& z,
    sd::LongType* dimension, sd::LongType dimensionLength,
    const sd::LongType* tadShapeInfo, const sd::LongType* tadOffsets) {
  validateDerivedTadPair(tadShapeInfo, tadOffsets, "index-reduce input");
  executeUnaryReduction(lc, graph::VulkanLegacyOpFamily::INDEX_REDUCE, opNum, x, extraParams, z,
                        dimension, dimensionLength);
}
#define SD_VULKAN_REDUCE3_SIMPLE(NAME) \
void NativeOpExecutioner::NAME(sd::LaunchContext* lc, int opNum, const sd::LegacyTensorArg& x, \
    void* extraParamsVals, const sd::LegacyTensorArg& y, const sd::LegacyTensorArg& z) { \
  executeBinaryReduction(lc, opNum, x, extraParamsVals, y, z, nullptr, 0); \
}
SD_VULKAN_REDUCE3_SIMPLE(execReduce3)
SD_VULKAN_REDUCE3_SIMPLE(execReduce3Scalar)
#undef SD_VULKAN_REDUCE3_SIMPLE
#define SD_VULKAN_REDUCE3(NAME, ALL_PAIRS) \
void NativeOpExecutioner::NAME(sd::LaunchContext* lc, int opNum, const sd::LegacyTensorArg& x, \
    void* extraParamsVals, const sd::LegacyTensorArg& y, const sd::LegacyTensorArg& z, \
    sd::LongType* dimension, sd::LongType dimensionLength, \
    const sd::LongType* xTadShapeInfo, const sd::LongType* xTadOffsets, \
    const sd::LongType* yTadShapeInfo, const sd::LongType* yTadOffsets) { \
  validateDerivedTadPair(xTadShapeInfo, xTadOffsets, "reduce3 X"); \
  validateDerivedTadPair(yTadShapeInfo, yTadOffsets, "reduce3 Y"); \
  executeBinaryReduction(lc, opNum, x, extraParamsVals, y, z, dimension, dimensionLength, ALL_PAIRS); \
}
SD_VULKAN_REDUCE3(execReduce3, false)
SD_VULKAN_REDUCE3(execReduce3All, true)
SD_VULKAN_REDUCE3(execReduce3TAD, false)
#undef SD_VULKAN_REDUCE3
void NativeOpExecutioner::execSummaryStats(sd::LaunchContext* lc, int opNum,
    const sd::LegacyTensorArg& x, void* extraParams, const sd::LegacyTensorArg& z,
    sd::LongType* dimension, sd::LongType dimensionLength,
    sd::LongType* tadShapeInfo, sd::LongType* tadOffsets, bool biasCorrected) {
  validateDerivedTadPair(tadShapeInfo, tadOffsets, "summary-stat input");
  executeUnaryReduction(lc, graph::VulkanLegacyOpFamily::SUMMARY_STATS, opNum, x, extraParams, z,
                        dimension, dimensionLength, &biasCorrected);
}
#define SD_VULKAN_SUMMARY(NAME) \
void NativeOpExecutioner::NAME(sd::LaunchContext* lc, int opNum, const sd::LegacyTensorArg& x, \
    void* extraParams, const sd::LegacyTensorArg& z, bool biasCorrected) { \
  executeUnaryReduction(lc, graph::VulkanLegacyOpFamily::SUMMARY_STATS, opNum, x, extraParams, z, \
                        nullptr, 0, &biasCorrected); \
}
SD_VULKAN_SUMMARY(execSummaryStats)
SD_VULKAN_SUMMARY(execSummaryStatsScalar)
#undef SD_VULKAN_SUMMARY

}  // namespace sd

#endif  // SD_VULKAN && HAVE_VULKAN
