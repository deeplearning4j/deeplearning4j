/* ******************************************************************************
 *
 * Copyright (c) 2026 Eclipse Foundation
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <config.h>

#if defined(SD_VULKAN) && defined(HAVE_VULKAN) && HAVE_VULKAN

// Selective rendering must precede NativeOpExecutioner.h.
#include <system/selective_rendering/core.h>
#include <system/selective_rendering/float_types.h>
#include <system/selective_rendering/bfloat_types.h>

#include <graph/RandomGenerator.h>
#include <helpers/helper_generator.h>
#include <helpers/shape.h>
#include <legacy/NativeOpExecutioner.h>
#include <legacy/vulkan/VulkanLegacyExecutor.h>
#include <loops/legacy_ops.h>
#include <loops/random.h>
#include <system/op_boilerplate.h>
#include <types/types.h>

namespace sd {
namespace {

graph::VulkanLegacyTensor randomInputTensor(
    const void* hostData, const void* deviceData,
    const sd::LongType* hostShapeInfo,
    const sd::LongType* deviceShapeInfo) {
  return {const_cast<void*>(hostData), const_cast<void*>(deviceData),
          hostShapeInfo, deviceShapeInfo};
}

graph::VulkanLegacyTensor randomOutputTensor(
    void* hostData, void* deviceData,
    const sd::LongType* hostShapeInfo,
    const sd::LongType* deviceShapeInfo) {
  return {hostData, deviceData, hostShapeInfo, deviceShapeInfo};
}

bool sharesStorage(const graph::VulkanLegacyTensor& a,
                   const graph::VulkanLegacyTensor& b) {
  if (a.array != nullptr && b.array != nullptr) {
    auto* aa = const_cast<NDArray*>(a.array);
    auto* ba = const_cast<NDArray*>(b.array);
    // Match the raw CPU/CUDA pointer identity convention, not allocation identity alone.
    const auto aOffset = sd::LegacyTensorArg::withShape(
        a.array, a.hostShapeInfo, a.deviceShapeInfo, a.relativeElementOffset).absoluteElementOffset();
    const auto bOffset = sd::LegacyTensorArg::withShape(
        b.array, b.hostShapeInfo, b.deviceShapeInfo, b.relativeElementOffset).absoluteElementOffset();
    return aa->getDataBuffer() == ba->getDataBuffer() && aOffset == bOffset;
  }
  return (a.hostData != nullptr && a.hostData == b.hostData) ||
         (a.deviceData != nullptr && a.deviceData == b.deviceData);
}

// The invocation carries what the native op reads (graph::vulkanLegacyRandomOperands):
// x as given, since an in-place op reads and writes z; y, except that the special
// samplers (which read y alone) take a y that is z to mean "no y", whereas any other
// op reads the y it is given even in place (ProbablisticMerge merging into y); and
// every extra argument, in the op's type. Only the uniform distribution's range used
// to be passed, so every other op ran with the lowering's defaults (dropout p = 1,
// Bernoulli p = 0.5, a standard normal). An op that cannot run without an input
// (DropOut without x, Choice without y) is refused here: the native entry points
// without it are stubs that write -1.
template <typename X>
void executeRandom(
    sd::LaunchContext* launchContext, int opNum, sd::Pointer state,
    const graph::VulkanLegacyTensor* x, const graph::VulkanLegacyTensor* y,
    const graph::VulkanLegacyTensor& output, void* extraArguments) {
  const auto operands = graph::vulkanLegacyRandomOperands(opNum);
  if (!operands.has_value() ||
      graph::VulkanLegacyOpCatalog::lookup(
          graph::VulkanLegacyOpFamily::RANDOM, opNum) == nullptr) {
    THROW_EXCEPTION("Vulkan execRandom received a non-canonical opNum");
  }

  graph::VulkanLegacyInvocation invocation(
      graph::VulkanLegacyOpFamily::RANDOM, opNum);
  const bool yMeansNoYWhenZ = operands->readsY && !operands->readsX;
  if (operands->readsX && x != nullptr) invocation.inputs.emplace_back(*x);
  if (operands->readsY && y != nullptr &&
      !(yMeansNoYWhenZ && sharesStorage(*y, output))) {
    invocation.inputs.emplace_back(*y);
  }
  if (static_cast<int>(invocation.inputs.size()) < operands->requiredInputs) {
    const std::string message =
        "Vulkan execRandom: legacy random op " + std::to_string(opNum) +
        " needs " + std::to_string(operands->requiredInputs) +
        " input array(s), got " + std::to_string(invocation.inputs.size());
    THROW_EXCEPTION(message.c_str());
  }
  invocation.outputs.emplace_back(output);
  invocation.randomState = state;
  invocation.randomExtraArguments = extraArguments;
  if (operands->extraArguments > 0) {
    if (extraArguments == nullptr) {
      THROW_EXCEPTION("Vulkan random execution requires the op's extra arguments");
    }
    auto* values = reinterpret_cast<const X*>(extraArguments);
    for (int i = 0; i < operands->extraArguments; ++i) {
      invocation.floatingArguments.push_back(static_cast<double>(values[i]));
    }
  }

  graph::requireVulkanLegacyExecution(launchContext, invocation);
}

void requireRandomState(sd::Pointer state) {
  if (state == nullptr) {
    THROW_EXCEPTION(
        "execRandom: stateHost is nullptr - RandomGenerator pointer is invalid");
  }
}

}  // namespace

void NativeOpExecutioner::execRandom(
    sd::LaunchContext* lc, int opNum, sd::Pointer state, void* hZ,
    const sd::LongType* hZShapeInfo, void* dZ,
    const sd::LongType* dZShapeInfo, void* extraArguments) {
  requireRandomState(state);

  const auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  const auto output =
      randomOutputTensor(hZ, dZ, hZShapeInfo, dZShapeInfo);
  BUILD_SINGLE_SELECTOR(
      zType, executeRandom,
      (lc, opNum, state, nullptr, nullptr, output, extraArguments),
      SD_FLOAT_TYPES);

}

void NativeOpExecutioner::execRandom(
    sd::LaunchContext* lc, int opNum, sd::Pointer state, const void* hX,
    const sd::LongType* hXShapeInfo, const void* dX,
    const sd::LongType* dXShapeInfo, void* hZ,
    const sd::LongType* hZShapeInfo, void* dZ,
    const sd::LongType* dZShapeInfo, void* extraArguments) {
  requireRandomState(state);

  const auto x = randomInputTensor(hX, dX, hXShapeInfo, dXShapeInfo);
  const auto output =
      randomOutputTensor(hZ, dZ, hZShapeInfo, dZShapeInfo);
  const auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  // The op reads x as z's type (the CPU and CUDA entry points refuse another type the same way)
  if (sd::ArrayOptions::dataType(hXShapeInfo) != zType)
    THROW_EXCEPTION("execRandom: x must have the output's data type");
  BUILD_SINGLE_SELECTOR(
      zType, executeRandom,
      (lc, opNum, state, &x, nullptr, output, extraArguments),
      SD_FLOAT_TYPES);

}

void NativeOpExecutioner::execRandom(
    sd::LaunchContext* lc, int opNum, sd::Pointer state, const void* hX,
    const sd::LongType* hXShapeInfo, const void* dX,
    const sd::LongType* dXShapeInfo, const void* hY,
    const sd::LongType* hYShapeInfo, const void* dY,
    const sd::LongType* dYShapeInfo, void* hZ,
    const sd::LongType* hZShapeInfo, void* dZ,
    const sd::LongType* dZShapeInfo, void* extraArguments) {
  requireRandomState(state);

  const auto x = randomInputTensor(hX, dX, hXShapeInfo, dXShapeInfo);
  const auto y = randomInputTensor(hY, dY, hYShapeInfo, dYShapeInfo);
  const auto output =
      randomOutputTensor(hZ, dZ, hZShapeInfo, dZShapeInfo);
  const auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  // The op reads x and y as z's type (the CPU and CUDA entry points refuse another type the same way)
  if (sd::ArrayOptions::dataType(hXShapeInfo) != zType ||
      sd::ArrayOptions::dataType(hYShapeInfo) != zType)
    THROW_EXCEPTION("execRandom: x and y must have the output's data type");
  BUILD_SINGLE_SELECTOR(
      zType, executeRandom,
      (lc, opNum, state, &x, &y, output, extraArguments),
      SD_FLOAT_TYPES);

}

void NativeOpExecutioner::execRandom(sd::LaunchContext* lc, int opNum, sd::Pointer state,
    const sd::LegacyTensorArg& z, void* extraArguments) {
  requireRandomState(state);
  const auto output = graph::VulkanLegacyTensor::fromArg(z);
  const auto zType = sd::ArrayOptions::dataType(z.hostShapeInfo);
  BUILD_SINGLE_SELECTOR(zType, executeRandom,
      (lc, opNum, state, nullptr, nullptr, output, extraArguments), SD_FLOAT_TYPES);
}

void NativeOpExecutioner::execRandom(sd::LaunchContext* lc, int opNum, sd::Pointer state,
    const sd::LegacyTensorArg& x, const sd::LegacyTensorArg& z, void* extraArguments) {
  requireRandomState(state);
  const auto input = graph::VulkanLegacyTensor::fromArg(x);
  const auto output = graph::VulkanLegacyTensor::fromArg(z);
  const auto zType = sd::ArrayOptions::dataType(z.hostShapeInfo);
  if (sd::ArrayOptions::dataType(x.hostShapeInfo) != zType)
    THROW_EXCEPTION("execRandom: x must have the output's data type");
  BUILD_SINGLE_SELECTOR(zType, executeRandom,
      (lc, opNum, state, &input, nullptr, output, extraArguments), SD_FLOAT_TYPES);
}

void NativeOpExecutioner::execRandom(sd::LaunchContext* lc, int opNum, sd::Pointer state,
    const sd::LegacyTensorArg& x, const sd::LegacyTensorArg& y,
    const sd::LegacyTensorArg& z, void* extraArguments) {
  requireRandomState(state);
  const auto inputX = graph::VulkanLegacyTensor::fromArg(x);
  const auto inputY = graph::VulkanLegacyTensor::fromArg(y);
  const auto output = graph::VulkanLegacyTensor::fromArg(z);
  const auto zType = sd::ArrayOptions::dataType(z.hostShapeInfo);
  if (sd::ArrayOptions::dataType(x.hostShapeInfo) != zType ||
      sd::ArrayOptions::dataType(y.hostShapeInfo) != zType)
    THROW_EXCEPTION("execRandom: x and y must have the output's data type");
  BUILD_SINGLE_SELECTOR(zType, executeRandom,
      (lc, opNum, state, &inputX, &inputY, output, extraArguments), SD_FLOAT_TYPES);
}

}  // namespace sd

#endif  // SD_VULKAN && HAVE_VULKAN
