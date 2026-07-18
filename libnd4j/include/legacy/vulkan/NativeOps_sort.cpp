/* ******************************************************************************
 *
 * Copyright (c) 2026 Eclipse Foundation
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <config.h>

#if defined(SD_VULKAN) && defined(HAVE_VULKAN) && HAVE_VULKAN

#include <array/NDArray.h>
#include <execution/LaunchContext.h>
#include <execution/vulkan/VulkanExecutionStream.h>
#include <execution/vulkan/VulkanLaunchContext.h>
#include <graph/Context.h>
#include <graph/vulkan/VulkanEagerExecutor.h>
#include <legacy/NativeOps.h>
#include <ops/declarable/headers/legacy_sort.h>

#include <string>
#include <vector>


namespace {

using sd::graph::VulkanExecutionStream;

VulkanExecutionStream* resolveSortStream(sd::Pointer* extraPointers,
                                         sd::LaunchContext* launchContext,
                                         int deviceId) {
  void* contextOwned = vulkanExecutionStream(launchContext);
  VulkanExecutionStream* stream = nullptr;
  if (extraPointers != nullptr && extraPointers[1] != nullptr) {
    stream = VulkanExecutionStream::fromOpaque(extraPointers[1], false);
  } else if (launchContext != nullptr && contextOwned != nullptr) {
    stream = VulkanExecutionStream::fromOpaque(contextOwned, false);
  } else {
    stream = VulkanExecutionStream::defaultExecution(deviceId);
  }

  if (stream == nullptr || !stream->isActive() ||
      stream->deviceId() != deviceId) {
    THROW_EXCEPTION(
        "Vulkan sorting could not resolve an active exact-device execution stream");
  }
  return stream;
}

template <typename Op>
void executeSortDescriptor(sd::Pointer* extraPointers,
                           const std::vector<sd::NDArray*>& inputs,
                           const std::vector<sd::NDArray*>& outputs,
                           const std::vector<sd::LongType>& dimensions,
                           bool descending) {
  if (inputs.empty() || outputs.empty()) {
    THROW_EXCEPTION("Vulkan sorting requires input and output tensors");
  }

  auto* launchContext = inputs[0]->getContext();
  if (launchContext == nullptr) launchContext = sd::LaunchContext::defaultContext();
  const int deviceId = inputs[0]->getDataBuffer()->deviceId();
  auto* stream = resolveSortStream(extraPointers, launchContext, deviceId);

  for (auto* input : inputs) {
    if (input == nullptr || input->specialBuffer() == nullptr) {
      THROW_EXCEPTION("Vulkan sorting requires device-resident input buffers");
    }
    if (input->getDataBuffer()->deviceId() != deviceId) {
      THROW_EXCEPTION("Vulkan sorting inputs must belong to one device");
    }
  }
  for (auto* output : outputs) {
    if (output == nullptr || output->specialBuffer() == nullptr) {
      THROW_EXCEPTION("Vulkan sorting requires device-resident output buffers");
    }
    if (output->getDataBuffer()->deviceId() != deviceId) {
      THROW_EXCEPTION("Vulkan sorting outputs must belong to one device");
    }
  }

  sd::NDArray::prepareSpecialUse(outputs, inputs);

  sd::graph::Context context(0);
  context.setInputArrays(static_cast<int>(inputs.size()),
                         const_cast<sd::NDArray**>(inputs.data()), false);
  context.setOutputArrays(static_cast<int>(outputs.size()),
                          const_cast<sd::NDArray**>(outputs.data()), false);
  context.setIArguments(dimensions);
  context.setBArguments({descending});

  Op descriptor;
  std::string errorMessage;
  const auto status = sd::graph::VulkanEagerExecutor::execute(
      descriptor.getOpHash(), context, *stream, &errorMessage);
  if (status != sd::Status::OK) {
    if (errorMessage.empty()) {
      errorMessage = "Vulkan sorting emitter execution failed";
    }
    THROW_EXCEPTION(errorMessage.c_str());
  }

  sd::NDArray::registerSpecialUse(outputs, inputs);
}

template <typename Function>
void withSortErrors(Function&& function) {
  try {
    function();
  } catch (const std::exception& error) {
    auto* reference = sd::LaunchContext::defaultContext()->errorReference();
    reference->setErrorCode(1);
    reference->setErrorMessage(error.what());
  }
}

std::vector<sd::LongType> dimensionVector(const sd::LongType* dimensions,
                                          sd::LongType length) {
  if (length <= 0 || dimensions == nullptr) {
    THROW_EXCEPTION("Vulkan TAD sorting requires at least one dimension");
  }
  return {dimensions, dimensions + length};
}

std::vector<sd::LongType> dimensionVector(OpaqueNDArray dimension) {
  if (dimension == nullptr || dimension->dataType() != sd::DataType::INT64) {
    THROW_EXCEPTION("Vulkan TAD sorting dimensions must be an INT64 tensor");
  }
  return dimensionVector(dimension->bufferAsT<sd::LongType>(),
                         dimension->lengthOf());
}

void requireEqualLengths(OpaqueNDArray keys, OpaqueNDArray values,
                         const char* operation) {
  if (keys == nullptr || values == nullptr) {
    THROW_EXCEPTION("Vulkan paired sorting requires non-null arrays");
  }
  if (keys->lengthOf() != values->lengthOf()) {
    std::string message(operation);
    message += ": keys and values must have the same size";
    THROW_EXCEPTION(message.c_str());
  }
}

}  // namespace

void sort(sd::Pointer* extraPointers, OpaqueNDArray x, bool descending) {
  withSortErrors([&] {
    if (x == nullptr || x->isEmpty()) return;
    executeSortDescriptor<sd::ops::legacy_sort>(
        extraPointers, {x}, {x}, {}, descending);
  });
}

void sortTad(sd::Pointer* extraPointers, OpaqueNDArray x,
             sd::LongType* dimension, sd::LongType dimensionLength,
             sd::LongType* tadShapeInfo, sd::LongType* tadOffsets,
             bool descending) {
  withSortErrors([&] {
    if (x == nullptr || x->isEmpty()) return;
    executeSortDescriptor<sd::ops::legacy_sort_tad>(
        extraPointers, {x}, {x},
        dimensionVector(dimension, dimensionLength), descending);
  });
}

void sortByKey(sd::Pointer* extraPointers, OpaqueNDArray x, OpaqueNDArray y,
               bool descending) {
  withSortErrors([&] {
    requireEqualLengths(x, y, "sortByKey");
    if (x->isEmpty()) return;
    executeSortDescriptor<sd::ops::legacy_sort_by_key>(
        extraPointers, {x, y}, {x, y}, {}, descending);
  });
}

void sortByValue(sd::Pointer* extraPointers, OpaqueNDArray x, OpaqueNDArray y,
                 bool descending) {
  withSortErrors([&] {
    requireEqualLengths(x, y, "sortByValue");
    if (x->isEmpty()) return;
    executeSortDescriptor<sd::ops::legacy_sort_by_value>(
        extraPointers, {x, y}, {x, y}, {}, descending);
  });
}

void sortTadByKey(sd::Pointer* extraPointers, OpaqueNDArray x, OpaqueNDArray y,
                  OpaqueNDArray dimension, bool descending) {
  withSortErrors([&] {
    requireEqualLengths(x, y, "sortTadByKey");
    if (x->isEmpty()) return;
    executeSortDescriptor<sd::ops::legacy_sort_tad_by_key>(
        extraPointers, {x, y}, {x, y}, dimensionVector(dimension), descending);
  });
}

void sortTadByValue(sd::Pointer* extraPointers, OpaqueNDArray x,
                    OpaqueNDArray y, OpaqueNDArray dimension,
                    bool descending) {
  withSortErrors([&] {
    requireEqualLengths(x, y, "sortTadByValue");
    if (x->isEmpty()) return;
    executeSortDescriptor<sd::ops::legacy_sort_tad_by_value>(
        extraPointers, {x, y}, {x, y}, dimensionVector(dimension), descending);
  });
}


#endif  // SD_VULKAN && HAVE_VULKAN
