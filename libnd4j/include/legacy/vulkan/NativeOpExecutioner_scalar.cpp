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
        "Vulkan legacy scalar descriptor execution cannot infer the typed "
        "argument count from non-null extra parameters");
  }
}

void appendDimensions(graph::VulkanLegacyInvocation& invocation,
                      const sd::LongType* dimensions,
                      sd::LongType dimensionLength) {
  if (dimensionLength < 0) {
    THROW_EXCEPTION("Vulkan legacy scalar received a negative dimension length");
  }
  if (dimensionLength > 0 && dimensions == nullptr) {
    THROW_EXCEPTION("Vulkan legacy scalar received no dimension data");
  }

  invocation.integerArguments.reserve(static_cast<std::size_t>(dimensionLength));
  for (sd::LongType index = 0; index < dimensionLength; ++index) {
    invocation.integerArguments.emplace_back(dimensions[index]);
  }
}

void validateDerivedTadPair(const sd::LongType* tadShapeInfo,
                            const sd::LongType* tadOffsets,
                            const char* operand) {
  if ((tadShapeInfo == nullptr) != (tadOffsets == nullptr)) {
    std::string message =
        "Vulkan legacy scalar received incomplete derived TAD metadata for ";
    message += operand;
    THROW_EXCEPTION(message.c_str());
  }
  // TAD shape/offset buffers are derived backend indexing caches. The Vulkan
  // path reconstructs operation indexing from canonical operand shapes and
  // semantic dimensions, so these buffers are not transported or dereferenced.
}

void executeScalar(
    sd::LaunchContext* launchContext, graph::VulkanLegacyOpFamily family,
    int opNum, const void* hX, const sd::LongType* hXShapeInfo,
    const void* dX, const sd::LongType* dXShapeInfo, void* hZ,
    const sd::LongType* hZShapeInfo, void* dZ,
    const sd::LongType* dZShapeInfo, const void* hScalars,
    const sd::LongType* hScalarShapeInfo, const void* dScalars,
    const sd::LongType* dScalarShapeInfo, void* extraParameters,
    const sd::LongType* dimensions, sd::LongType dimensionLength,
    const sd::LongType* tadShapeInfo, const sd::LongType* tadOffsets,
    const sd::LongType* tadShapeInfoZ, const sd::LongType* tadOffsetsZ) {
  requireNoOpaqueExtraParameters(extraParameters);
  validateDerivedTadPair(tadShapeInfo, tadOffsets, "input");
  validateDerivedTadPair(tadShapeInfoZ, tadOffsetsZ, "output");

  graph::VulkanLegacyInvocation invocation(family, opNum);
  invocation.inputs.emplace_back(
      inputTensor(hX, dX, hXShapeInfo, dXShapeInfo));
  invocation.inputs.emplace_back(
      inputTensor(hScalars, dScalars, hScalarShapeInfo, dScalarShapeInfo));
  invocation.outputs.emplace_back(
      outputTensor(hZ, dZ, hZShapeInfo, dZShapeInfo));
  appendDimensions(invocation, dimensions, dimensionLength);
  graph::requireVulkanLegacyExecution(launchContext, invocation);
}

}  // namespace

void NativeOpExecutioner::execScalar(
    sd::LaunchContext* lc, int opNum, const void* hX,
    const sd::LongType* hXShapeInfo, const void* dX,
    const sd::LongType* dXShapeInfo, void* hZ,
    const sd::LongType* hZShapeInfo, void* dZ,
    const sd::LongType* dZShapeInfo, const void* hScalar,
    const sd::LongType* hScalarShapeInfo, const void* dScalar,
    const sd::LongType* dScalarShapeInfo, void* extraParams,
    bool allowParallelism) {
  (void)allowParallelism;
  executeScalar(
      lc, graph::VulkanLegacyOpFamily::SCALAR, opNum, hX, hXShapeInfo, dX,
      dXShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo, hScalar,
      hScalarShapeInfo, dScalar, dScalarShapeInfo, extraParams, nullptr, 0,
      nullptr, nullptr, nullptr, nullptr);
}

void NativeOpExecutioner::execScalarBool(
    sd::LaunchContext* lc, int opNum, const void* hX,
    const sd::LongType* hXShapeInfo, const void* dX,
    const sd::LongType* dXShapeInfo, void* hZ,
    const sd::LongType* hZShapeInfo, void* dZ,
    const sd::LongType* dZShapeInfo, const void* hScalar,
    const sd::LongType* hScalarShapeInfo, const void* dScalar,
    const sd::LongType* dScalarShapeInfo, void* extraParams,
    bool allowParallelism) {
  (void)allowParallelism;
  executeScalar(
      lc, graph::VulkanLegacyOpFamily::SCALAR_BOOL, opNum, hX, hXShapeInfo,
      dX, dXShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo, hScalar,
      hScalarShapeInfo, dScalar, dScalarShapeInfo, extraParams, nullptr, 0,
      nullptr, nullptr, nullptr, nullptr);
}

void NativeOpExecutioner::execScalarInt(
    sd::LaunchContext* lc, int opNum, const void* hX,
    const sd::LongType* hXShapeInfo, const void* dX,
    const sd::LongType* dXShapeInfo, void* hZ,
    const sd::LongType* hZShapeInfo, void* dZ,
    const sd::LongType* dZShapeInfo, const void* hScalar,
    const sd::LongType* hScalarShapeInfo, const void* dScalar,
    const sd::LongType* dScalarShapeInfo, void* extraParams,
    bool allowParallelism) {
  (void)allowParallelism;
  executeScalar(
      lc, graph::VulkanLegacyOpFamily::SCALAR_INT, opNum, hX, hXShapeInfo, dX,
      dXShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo, hScalar,
      hScalarShapeInfo, dScalar, dScalarShapeInfo, extraParams, nullptr, 0,
      nullptr, nullptr, nullptr, nullptr);
}

void NativeOpExecutioner::execScalar(
    sd::LaunchContext* lc, int opNum, const void* hX,
    const sd::LongType* hXShapeInfo, const void* dX,
    const sd::LongType* dXShapeInfo, void* extraParams, void* hZ,
    const sd::LongType* hZShapeInfo, void* dZ,
    const sd::LongType* dZShapeInfo, const void* hScalars,
    const sd::LongType* hScalarShapeInfo, const void* dScalars,
    const sd::LongType* dScalarShapeInfo, sd::LongType* dimension,
    sd::LongType dimensionLength, const sd::LongType* tadShapeInfo,
    const sd::LongType* tadOffsets, const sd::LongType* tadShapeInfoZ,
    const sd::LongType* tadOffsetsZ) {
  executeScalar(
      lc, graph::VulkanLegacyOpFamily::SCALAR, opNum, hX, hXShapeInfo, dX,
      dXShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo, hScalars,
      hScalarShapeInfo, dScalars, dScalarShapeInfo, extraParams, dimension,
      dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ);
}

void NativeOpExecutioner::execScalarBool(
    sd::LaunchContext* lc, int opNum, const void* hX,
    const sd::LongType* hXShapeInfo, const void* dX,
    const sd::LongType* dXShapeInfo, void* extraParams, void* hZ,
    const sd::LongType* hZShapeInfo, void* dZ,
    const sd::LongType* dZShapeInfo, const void* hScalars,
    const sd::LongType* hScalarShapeInfo, const void* dScalars,
    const sd::LongType* dScalarShapeInfo, sd::LongType* dimension,
    sd::LongType dimensionLength, const sd::LongType* tadShapeInfo,
    const sd::LongType* tadOffsets, const sd::LongType* tadShapeInfoZ,
    const sd::LongType* tadOffsetsZ) {
  executeScalar(
      lc, graph::VulkanLegacyOpFamily::SCALAR_BOOL, opNum, hX, hXShapeInfo,
      dX, dXShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo, hScalars,
      hScalarShapeInfo, dScalars, dScalarShapeInfo, extraParams, dimension,
      dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ);
}

void NativeOpExecutioner::execScalarInt(
    sd::LaunchContext* lc, int opNum, const void* hX,
    const sd::LongType* hXShapeInfo, const void* dX,
    const sd::LongType* dXShapeInfo, void* extraParams, void* hZ,
    const sd::LongType* hZShapeInfo, void* dZ,
    const sd::LongType* dZShapeInfo, const void* hScalars,
    const sd::LongType* hScalarShapeInfo, const void* dScalars,
    const sd::LongType* dScalarShapeInfo, sd::LongType* dimension,
    sd::LongType dimensionLength, const sd::LongType* tadShapeInfo,
    const sd::LongType* tadOffsets, const sd::LongType* tadShapeInfoZ,
    const sd::LongType* tadOffsetsZ) {
  executeScalar(
      lc, graph::VulkanLegacyOpFamily::SCALAR_INT, opNum, hX, hXShapeInfo, dX,
      dXShapeInfo, hZ, hZShapeInfo, dZ, dZShapeInfo, hScalars,
      hScalarShapeInfo, dScalars, dScalarShapeInfo, extraParams, dimension,
      dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ);
}

}  // namespace sd

#endif  // SD_VULKAN && HAVE_VULKAN
