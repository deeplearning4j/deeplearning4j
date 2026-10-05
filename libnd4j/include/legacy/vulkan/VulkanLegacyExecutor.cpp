/* ******************************************************************************
 *
 * Copyright (c) 2026 Eclipse Foundation
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <config.h>

#if defined(SD_VULKAN) && defined(HAVE_VULKAN) && HAVE_VULKAN

#include <array/NDArray.h>
#include <array/ArrayOptions.h>
#include <system/op_enums.h>
#include <system/type_boilerplate.h>
#include <execution/vulkan/VulkanExecutionStream.h>
#include <execution/vulkan/VulkanLaunchContext.h>
#include <graph/Context.h>
#include <graph/vulkan/VulkanEagerExecutor.h>
#include <helpers/shape.h>
#include <legacy/vulkan/VulkanLegacyExecutor.h>

#include <legacy/NativeOpExecutioner.h>
#include <limits>
#include <sstream>
#include <utility>

namespace sd {
namespace graph {
namespace {

template <typename T>
void appendTypedExtraParameters(VulkanLegacyInvocation& invocation,
                                const void* extraParams, int count) {
  const auto* values = static_cast<const T*>(extraParams);
  for (int index = 0; index < count; ++index)
    invocation.floatingArguments.push_back(static_cast<double>(values[index]));
}

void setError(std::string* target, const std::string& message) {
  if (target != nullptr) *target = message;
}

VulkanExecutionStream* resolveStream(sd::LaunchContext* launchContext,
                                     std::string* errorMessage) {
  if (launchContext == nullptr) {
    setError(errorMessage, "Vulkan legacy execution has no launch context");
    return nullptr;
  }

  const int deviceId = launchContext->getDeviceID();
  void* contextOwned = vulkanExecutionStream(launchContext);
  VulkanExecutionStream* stream = nullptr;
  if (contextOwned != nullptr) {
    stream = VulkanExecutionStream::fromOpaque(contextOwned, false);
  } else {
    stream = VulkanExecutionStream::defaultExecution(deviceId);
  }

  if (stream == nullptr || !stream->isActive() ||
      stream->deviceId() != deviceId) {
    setError(errorMessage,
             "Vulkan legacy execution could not resolve the exact-device stream");
    return nullptr;
  }
  return stream;
}

bool validateTensor(const VulkanLegacyTensor& tensor,
                    const char* role,
                    std::string* errorMessage) {
  if (tensor.hostShapeInfo == nullptr) {
    setError(errorMessage,
             std::string("Vulkan legacy ") + role +
                 " tensor has no host shape information");
    return false;
  }

  if (tensor.array != nullptr) {
    auto* original = const_cast<NDArray*>(tensor.array);
    if (ArrayOptions::dataType(tensor.hostShapeInfo) != original->dataType()) {
      setError(errorMessage, "Vulkan legacy effective shape changed storage dtype");
      return false;
    }
    auto* buffer = original->getDataBuffer();
    // The recorder owns device allocation and coherence preparation on this exact
    // DataBuffer. Do not require special() before it has prepared the operand.
    if (!ArrayOptions::isEmpty(const_cast<sd::LongType*>(tensor.hostShapeInfo)) &&
        shape::length(tensor.hostShapeInfo) > 0 && buffer == nullptr) {
      setError(errorMessage, "Vulkan legacy metadata operand has no original DataBuffer");
      return false;
    }
    return true;
  }
  const sd::LongType length = shape::length(tensor.hostShapeInfo);
  if (length > 0 && tensor.deviceData == nullptr) {
    setError(errorMessage,
             std::string("Vulkan legacy ") + role +
                 " tensor has no Vulkan device allocation");
    return false;
  }
  return true;
}

sd::LongType checkedAdd(sd::LongType a, sd::LongType b) {
  const auto maximum = std::numeric_limits<sd::LongType>::max();
  const auto minimum = std::numeric_limits<sd::LongType>::min();
  if ((b > 0 && a > maximum - b) || (b < 0 && a < minimum - b))
    THROW_EXCEPTION("Vulkan legacy tensor element offset overflow");
  return a + b;
}

// Validate the stride-reachable backing span, not merely logical tensor length.
void validateBackingSpan(NDArray* original, const sd::LongType* shapeInfo,
                         sd::LongType offset) {
  const auto rank = shape::rank(shapeInfo);
  if (rank < 0 || rank > SD_MAX_RANK)
    THROW_EXCEPTION("Vulkan legacy effective shape has invalid rank");
  const auto* sizes = shape::shapeOf(shapeInfo);
  const auto* strides = shape::stride(shapeInfo);
  bool empty = ArrayOptions::isEmpty(const_cast<sd::LongType*>(shapeInfo));
  for (int axis = 0; axis < rank; ++axis) {
    if (sizes[axis] < 0) THROW_EXCEPTION("Vulkan legacy effective shape has negative dimension");
    empty = empty || sizes[axis] == 0;
  }
  auto* buffer = original->getDataBuffer();
  const auto elements = buffer == nullptr ? size_t(0) : buffer->getLenInBytes() / original->sizeOfT();
  if (empty) {
    if (offset < 0 || static_cast<size_t>(offset) > elements)
      THROW_EXCEPTION("Vulkan legacy empty tensor offset exceeds original allocation");
    return;
  }
  sd::LongType lowest = offset, highest = offset;
  for (int axis = 0; axis < rank; ++axis) {
    const auto count = sizes[axis] - 1;
    const auto stride = strides[axis];
    if (count > 0 &&
        (stride > std::numeric_limits<sd::LongType>::max() / count ||
         stride < std::numeric_limits<sd::LongType>::min() / count))
      THROW_EXCEPTION("Vulkan legacy tensor stride span overflow");
    const auto delta = count * stride;
    if (delta < 0) lowest = checkedAdd(lowest, delta);
    else highest = checkedAdd(highest, delta);
  }
  if (lowest < 0 || highest < 0 || static_cast<size_t>(highest) >= elements)
    THROW_EXCEPTION("Vulkan legacy tensor stride span exceeds original allocation");
}

// Manual ownership: only metadata adapters/raw wrappers are deleted, never caller arrays.
struct TensorAdapters {
  std::vector<NDArray*> owned;
  ~TensorAdapters() { for (auto* array : owned) delete array; }
  TensorAdapters() = default;
  TensorAdapters(const TensorAdapters&) = delete;
  TensorAdapters& operator=(const TensorAdapters&) = delete;
};

NDArray* wrapTensor(const VulkanLegacyTensor& tensor,
                   sd::LaunchContext* launchContext, TensorAdapters& adapters) {
  NDArray* result;
  if (tensor.array != nullptr) {
    auto* original = const_cast<NDArray*>(tensor.array);
    const auto offset = checkedAdd(original->offset(), tensor.relativeElementOffset);
    validateBackingSpan(original, tensor.hostShapeInfo, offset);
    if (tensor.relativeElementOffset == 0 && tensor.hostShapeInfo == original->shapeInfo())
      return original;
    result = new NDArray(original->getDataBuffer(),
                         const_cast<sd::LongType*>(tensor.hostShapeInfo), launchContext, offset);
  } else {
    // Explicitly raw APIs provide allocation-base pointers and have no array metadata.
    result = new NDArray(tensor.hostData, tensor.deviceData,
                         const_cast<sd::LongType*>(tensor.hostShapeInfo), launchContext,
                         /*isBuffAlloc=*/false, /*isBuffDAlloc=*/false, /*offset=*/0);
  }
  adapters.owned.push_back(result);  // Capacity is reserved before any allocation.
  return result;
}

template <typename Execute>
Status executeInvocation(sd::LaunchContext* launchContext,
                         const VulkanInvocationArguments& invocation,
                         std::string* errorMessage, Execute&& execute) {
#if !defined(HAVE_MLIR) || !HAVE_MLIR
  setError(errorMessage, "Vulkan execution requires MLIR support");
  return Status::VALIDATION;
#else
  VulkanExecutionStream* stream = resolveStream(launchContext, errorMessage);
  if (stream == nullptr) return Status::KERNEL_FAILURE;

  TensorAdapters adapters;
  adapters.owned.reserve(invocation.inputs.size() + invocation.outputs.size());

  Context context(1);
  for (size_t index = 0; index < invocation.inputs.size(); ++index) {
    if (!validateTensor(invocation.inputs[index], "input", errorMessage)) {
      return Status::VALIDATION;
    }
    context.setInputArray(static_cast<int>(index),
                          wrapTensor(invocation.inputs[index], launchContext, adapters),
                          /*removable=*/false);
  }
  for (size_t index = 0; index < invocation.outputs.size(); ++index) {
    if (!validateTensor(invocation.outputs[index], "output", errorMessage)) {
      return Status::VALIDATION;
    }
    context.setOutputArray(static_cast<int>(index),
                           wrapTensor(invocation.outputs[index], launchContext, adapters),
                           /*removable=*/false);
  }

  context.setIArguments(invocation.integerArguments);
  context.setTArguments(invocation.floatingArguments);
  context.setBArguments(invocation.booleanArguments);
  return execute(context, *stream);
#endif
}

Status executeDescriptorHash(
    sd::LaunchContext* launchContext, sd::LongType descriptorHash,
    const VulkanInvocationArguments& invocation, std::string* errorMessage) {
  return executeInvocation(
      launchContext, invocation, errorMessage,
      [&](Context& context, VulkanExecutionStream& stream) {
        return VulkanEagerExecutor::execute(descriptorHash, context, stream,
                                            errorMessage);
      });
}

}  // namespace

VulkanLegacyTensor VulkanLegacyTensor::fromArg(const sd::LegacyTensorArg& tensor) {
  if (tensor.array == nullptr || tensor.hostShapeInfo == nullptr)
    THROW_EXCEPTION("Vulkan legacy metadata requires the original NDArray and effective shape");
  tensor.absoluteElementOffset();
  return {nullptr, nullptr, tensor.hostShapeInfo, tensor.deviceShapeInfo,
          tensor.array, tensor.relativeElementOffset};
}

void appendVulkanLegacyExtraParameters(VulkanLegacyInvocation& invocation,
                                      const void* extraParams,
                                      const sd::LongType* inputShapeInfo) {
  if (extraParams == nullptr) return;
  int count = 0;
  if (invocation.family == VulkanLegacyOpFamily::TRANSFORM_BOOL &&
      invocation.opNum == static_cast<int>(sd::transform::MatchConditionBool))
    count = 3;  // comparison, epsilon, mode, encoded in the input dtype
  else if (invocation.family == VulkanLegacyOpFamily::PAIRWISE_BOOL &&
           invocation.opNum == static_cast<int>(sd::pairwise::MatchCondition))
    count = 2;  // epsilon, mode; comparison comes from y
  else
    THROW_EXCEPTION("Vulkan legacy operation has no declared extraParams ABI");
  if (inputShapeInfo == nullptr)
    THROW_EXCEPTION("Vulkan legacy extraParams require input shape information");
  BUILD_SINGLE_SELECTOR(sd::ArrayOptions::dataType(inputShapeInfo),
                        appendTypedExtraParameters,
                        (invocation, extraParams, count), SD_NUMERIC_TYPES);
}

Status executeVulkanLegacy(sd::LaunchContext* launchContext,
                           const VulkanLegacyInvocation& invocation,
                           std::string* errorMessage) {
  if (VulkanLegacyOpCatalog::lookup(invocation.family,
                                    invocation.opNum) == nullptr) {
    setError(errorMessage,
             "Vulkan legacy execution received a non-canonical typed identity");
    return Status::VALIDATION;
  }

  return executeInvocation(
      launchContext, invocation, errorMessage,
      [&](Context& context, VulkanExecutionStream& stream) {
        return VulkanEagerExecutor::execute(
            invocation.family, invocation.opNum, context, stream,
            reinterpret_cast<RandomGenerator*>(invocation.randomState),
            errorMessage);
      });
}

Status executeVulkanDescriptor(sd::LaunchContext* launchContext,
                               const VulkanDescriptorInvocation& invocation,
                               std::string* errorMessage) {
  return executeDescriptorHash(launchContext, invocation.descriptorHash,
                               invocation, errorMessage);
}

void requireVulkanLegacyExecution(sd::LaunchContext* launchContext,
                                  const VulkanLegacyInvocation& invocation) {
  std::string errorMessage;
  const Status status =
      executeVulkanLegacy(launchContext, invocation, &errorMessage);
  if (status == Status::OK) return;

  std::ostringstream message;
  message << "Vulkan legacy execution failed"
          << " (family=" << static_cast<int>(invocation.family)
          << ", opNum=" << invocation.opNum << ')';
  if (!errorMessage.empty()) message << ": " << errorMessage;
  THROW_EXCEPTION(message.str().c_str());
}

void requireVulkanDescriptorExecution(
    sd::LaunchContext* launchContext,
    const VulkanDescriptorInvocation& invocation) {
  std::string errorMessage;
  const Status status =
      executeVulkanDescriptor(launchContext, invocation, &errorMessage);
  if (status == Status::OK) return;

  std::ostringstream message;
  message << "Vulkan descriptor execution failed"
          << " (hash=" << invocation.descriptorHash << ')';
  if (!errorMessage.empty()) message << ": " << errorMessage;
  THROW_EXCEPTION(message.str().c_str());
}

}  // namespace graph
}  // namespace sd

#endif  // SD_VULKAN && HAVE_VULKAN
