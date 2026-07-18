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

template <typename RandomOp>
void freezeRandomArguments(RandomOp*, sd::Pointer, void*,
                           graph::VulkanLegacyInvocation&) {}

template <typename X>
void freezeRandomArguments(randomOps::UniformDistribution<X>*,
                           sd::Pointer, void* extraArguments,
                           graph::VulkanLegacyInvocation& invocation) {
  if (extraArguments == nullptr) {
    THROW_EXCEPTION("Vulkan uniform random execution requires range arguments");
  }
  auto* range = reinterpret_cast<X*>(extraArguments);
  invocation.floatingArguments = {
      static_cast<double>(range[0]), static_cast<double>(range[1])};
}

template <typename RandomOp>
void executeRandomTyped(
    sd::LaunchContext* launchContext, int opNum, sd::Pointer state,
    const graph::VulkanLegacyTensor* inputs, size_t inputCount,
    const graph::VulkanLegacyTensor& output, void* extraArguments) {
  graph::VulkanLegacyInvocation invocation(
      graph::VulkanLegacyOpFamily::RANDOM, opNum);
  if (inputCount != 0) invocation.inputs.assign(inputs, inputs + inputCount);
  invocation.outputs.emplace_back(output);
  invocation.randomState = state;
  invocation.randomExtraArguments = extraArguments;
  freezeRandomArguments(static_cast<RandomOp*>(nullptr), state,
                        extraArguments, invocation);

  graph::requireVulkanLegacyExecution(launchContext, invocation);
}

template <typename X>
void executeRandom(
    sd::LaunchContext* launchContext, int opNum, sd::Pointer state,
    const graph::VulkanLegacyTensor* inputs, size_t inputCount,
    const graph::VulkanLegacyTensor& output, void* extraArguments) {
  using namespace randomOps;

  if (graph::VulkanLegacyOpCatalog::lookup(
          graph::VulkanLegacyOpFamily::RANDOM, opNum) == nullptr) {
    THROW_EXCEPTION("Vulkan execRandom received a non-canonical opNum");
  }

  DISPATCH_BY_OPNUM_T(
      executeRandomTyped,
      PARAMS(launchContext, opNum, state, inputs, inputCount, output,
             extraArguments),
      RANDOM_OPS);
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
      (lc, opNum, state, nullptr, 0, output, extraArguments),
      SD_FLOAT_TYPES);

}

void NativeOpExecutioner::execRandom(
    sd::LaunchContext* lc, int opNum, sd::Pointer state, const void* hX,
    const sd::LongType* hXShapeInfo, const void* dX,
    const sd::LongType* dXShapeInfo, void* hZ,
    const sd::LongType* hZShapeInfo, void* dZ,
    const sd::LongType* dZShapeInfo, void* extraArguments) {
  requireRandomState(state);

  const graph::VulkanLegacyTensor inputs[] = {
      randomInputTensor(hX, dX, hXShapeInfo, dXShapeInfo)};
  const auto output =
      randomOutputTensor(hZ, dZ, hZShapeInfo, dZShapeInfo);
  const auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  BUILD_SINGLE_SELECTOR(
      zType, executeRandom,
      (lc, opNum, state, inputs, 1, output, extraArguments),
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

  const graph::VulkanLegacyTensor inputs[] = {
      randomInputTensor(hX, dX, hXShapeInfo, dXShapeInfo),
      randomInputTensor(hY, dY, hYShapeInfo, dYShapeInfo)};
  const auto output =
      randomOutputTensor(hZ, dZ, hZShapeInfo, dZShapeInfo);
  const auto zType = sd::ArrayOptions::dataType(hZShapeInfo);
  BUILD_SINGLE_SELECTOR(
      zType, executeRandom,
      (lc, opNum, state, inputs, 2, output, extraArguments),
      SD_FLOAT_TYPES);

}

}  // namespace sd

#endif  // SD_VULKAN && HAVE_VULKAN
