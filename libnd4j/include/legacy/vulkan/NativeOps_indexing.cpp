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
#include <helpers/shape.h>
#include <legacy/NativeOps.h>
#include <ops/declarable/headers/legacy_indexing.h>

#include <limits>
#include <string>
#include <vector>


namespace {

using sd::graph::VulkanExecutionStream;

VulkanExecutionStream* resolveIndexingStream(sd::Pointer* extraPointers,
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
        "Vulkan indexed-TAD movement could not resolve an active "
        "exact-device execution stream");
  }
  return stream;
}

void requireLogicalShapeMatch(sd::NDArray* source,
                              sd::NDArray* destination,
                              const char* operation) {
  if (source->rankOf() != destination->rankOf()) {
    std::string message(operation);
    message += ": source and destination ranks differ";
    THROW_EXCEPTION(message.c_str());
  }

  for (int dimension = 0; dimension < source->rankOf(); ++dimension) {
    if (source->sizeAt(dimension) != destination->sizeAt(dimension)) {
      std::string message(operation);
      message += ": source and destination shapes differ";
      THROW_EXCEPTION(message.c_str());
    }
  }
}

void requireExactAlias(sd::NDArray* source,
                       sd::NDArray* destination,
                       const char* operation) {
  requireLogicalShapeMatch(source, destination, operation);
  if (source->getDataBuffer() != destination->getDataBuffer() ||
      source->offset() != destination->offset()) {
    std::string message(operation);
    message += ": source and destination must alias the exact same storage";
    THROW_EXCEPTION(message.c_str());
  }

  for (int dimension = 0; dimension < source->rankOf(); ++dimension) {
    if (source->strideAt(dimension) != destination->strideAt(dimension)) {
      std::string message(operation);
      message += ": source and destination alias strides differ";
      THROW_EXCEPTION(message.c_str());
    }
  }
}

void requireDeviceArray(sd::NDArray* array, int deviceId, const char* role) {
  if (array == nullptr || array->getDataBuffer() == nullptr) {
    std::string message("Vulkan indexed-TAD movement requires a valid ");
    message += role;
    message += " array";
    THROW_EXCEPTION(message.c_str());
  }
  if (array->lengthOf() > 0 && array->specialBuffer() == nullptr) {
    std::string message("Vulkan indexed-TAD movement requires a device-resident ");
    message += role;
    message += " buffer";
    THROW_EXCEPTION(message.c_str());
  }
  if (array->getDataBuffer()->deviceId() != deviceId) {
    std::string message("Vulkan indexed-TAD movement ");
    message += role;
    message += " arrays must belong to the execution device";
    THROW_EXCEPTION(message.c_str());
  }
}

void requireDimensions(sd::NDArray* array,
                       const std::vector<sd::LongType>& dimensions,
                       const char* operation) {
  if (dimensions.empty()) {
    std::string message(operation);
    message += ": dimensions must not be empty";
    THROW_EXCEPTION(message.c_str());
  }

  std::vector<bool> seen(static_cast<size_t>(array->rankOf()), false);
  for (const auto dimension : dimensions) {
    if (dimension < 0 || dimension >= array->rankOf()) {
      std::string message(operation);
      message += ": dimension is outside the array rank";
      THROW_EXCEPTION(message.c_str());
    }
    const auto index = static_cast<size_t>(dimension);
    if (seen[index]) {
      std::string message(operation);
      message += ": dimensions must not contain duplicates";
      THROW_EXCEPTION(message.c_str());
    }
    seen[index] = true;
  }
}

std::vector<sd::LongType> sharedDimensions(OpaqueNDArray dimension) {
  if (dimension == nullptr || dimension->dataType() != sd::DataType::INT64) {
    THROW_EXCEPTION("Vulkan shuffle dimensions must be an INT64 vector");
  }
  if (dimension->rankOf() != 1 || dimension->lengthOf() <= 0) {
    THROW_EXCEPTION(
        "Vulkan shuffle follows the NativeOps shared dimension-vector contract");
  }

  auto* values = static_cast<sd::NDArray*>(dimension);
  std::vector<sd::LongType> result;
  result.reserve(static_cast<size_t>(dimension->lengthOf()));
  for (sd::LongType index = 0; index < dimension->lengthOf(); ++index) {
    result.push_back(values->e<sd::LongType>(index));
  }
  return result;
}

sd::LongType indexedItemCount(
    sd::NDArray* array,
    const std::vector<sd::LongType>& dimensions) {
  if (array->rankOf() == 1) return array->lengthOf();

  sd::LongType tadLength = 1;
  for (const auto dimension : dimensions) {
    const auto dimensionSize =
        array->sizeAt(static_cast<int>(dimension));
    if (dimensionSize <= 0 ||
        tadLength > std::numeric_limits<sd::LongType>::max() / dimensionSize) {
      THROW_EXCEPTION(
          "Vulkan shuffle TAD length exceeds the supported integer range");
    }
    tadLength *= dimensionSize;
  }

  if (array->lengthOf() % tadLength != 0) {
    THROW_EXCEPTION(
        "Vulkan shuffle dimensions do not define an integral TAD partition");
  }
  return array->lengthOf() / tadLength;
}

template <typename Op>
void executeIndexingDescriptor(
    sd::Pointer* extraPointers, const std::vector<sd::NDArray*>& inputs,
    const std::vector<sd::NDArray*>& outputs,
    const std::vector<sd::LongType>& integerArguments) {
  if (inputs.empty() || outputs.empty() || inputs[0] == nullptr ||
      inputs[0]->getDataBuffer() == nullptr) {
    THROW_EXCEPTION(
        "Vulkan indexed-TAD movement requires input and output tensors");
  }

  auto* launchContext = inputs[0]->getContext();
  if (launchContext == nullptr) {
    launchContext = sd::LaunchContext::defaultContext();
  }
  const int deviceId = inputs[0]->getDataBuffer()->deviceId();
  auto* stream =
      resolveIndexingStream(extraPointers, launchContext, deviceId);

  for (auto* input : inputs) {
    requireDeviceArray(input, deviceId, "input");
  }
  for (auto* output : outputs) {
    requireDeviceArray(output, deviceId, "output");
  }
  sd::NDArray::prepareSpecialUse(outputs, inputs);

  sd::graph::Context context(0);
  context.setInputArrays(static_cast<int>(inputs.size()),
                         const_cast<sd::NDArray**>(inputs.data()), false);
  context.setOutputArrays(static_cast<int>(outputs.size()),
                          const_cast<sd::NDArray**>(outputs.data()), false);
  context.setIArguments(integerArguments);

  Op descriptor;
  std::string errorMessage;
  const auto status = sd::graph::VulkanEagerExecutor::execute(
      descriptor.getOpHash(), context, *stream, &errorMessage);
  if (status != sd::Status::OK) {
    if (errorMessage.empty()) {
      errorMessage = "Vulkan indexed-TAD emitter execution failed";
    }
    THROW_EXCEPTION(errorMessage.c_str());
  }

  sd::NDArray::registerSpecialUse(outputs, inputs);
}

template <typename Function>
void withIndexingErrors(Function&& function) {
  try {
    function();
  } catch (const std::exception& error) {
    auto* reference = sd::LaunchContext::defaultContext()->errorReference();
    reference->setErrorCode(1);
    reference->setErrorMessage(error.what());
  }
}

}  // namespace

void pullRows(sd::Pointer* extraPointers, OpaqueNDArray x, OpaqueNDArray z,
              sd::LongType n, OpaqueNDArray indexes,
              sd::LongType dimension) {
  withIndexingErrors([&] {
    if (x == nullptr || z == nullptr || indexes == nullptr) {
      THROW_EXCEPTION("Vulkan pullRows requires non-null arrays");
    }
    if (n < 0 || indexes->lengthOf() < n) {
      THROW_EXCEPTION(
          "Vulkan pullRows index count is outside the index tensor length");
    }
    if (n == 0) return;
    if (indexes->dataType() != sd::DataType::INT64) {
      THROW_EXCEPTION("Vulkan pullRows indexes must use INT64 storage");
    }
    if (x->dataType() != z->dataType()) {
      THROW_EXCEPTION(
          "Vulkan pullRows source and destination data types must match");
    }
    if (x->rankOf() < 1 || x->rankOf() > 2 ||
        z->rankOf() != x->rankOf()) {
      THROW_EXCEPTION(
          "Vulkan pullRows follows the established rank-1/2 NativeOps contract");
    }
    if (dimension < 0 || dimension >= x->rankOf() ||
        dimension >= z->rankOf()) {
      THROW_EXCEPTION("Vulkan pullRows dimension is outside the array rank");
    }

    if (x->rankOf() == 1) {
      if (z->lengthOf() != n) {
        THROW_EXCEPTION(
            "Vulkan pullRows rank-1 destination length must equal the index count");
      }
    } else if (dimension == 1) {
      if (z->sizeAt(0) != n || z->sizeAt(1) != x->sizeAt(1)) {
        THROW_EXCEPTION(
            "Vulkan pullRows destination shape does not match retained rows");
      }
    } else if (dimension == 0) {
      if (z->sizeAt(0) != x->sizeAt(0) || z->sizeAt(1) != n) {
        THROW_EXCEPTION(
            "Vulkan pullRows destination shape does not match retained columns");
      }
    } else {
      THROW_EXCEPTION(
          "Vulkan pullRows rank-2 dimension must be zero or one");
    }

    executeIndexingDescriptor<sd::ops::legacy_pull_rows>(
        extraPointers, {x, indexes}, {z}, {n, dimension});
  });
}

void shuffle(sd::Pointer* extras, OpaqueNDArrayArr x, OpaqueNDArrayArr z,
             int N, OpaqueNDArray dimension, OpaqueNDArray shuffleMap) {
  withIndexingErrors([&] {
    if (N <= 0 || x == nullptr || z == nullptr || shuffleMap == nullptr) {
      THROW_EXCEPTION(
          "Vulkan shuffle requires non-null arrays and a positive array count");
    }
    if (shuffleMap->dataType() != sd::DataType::INT32) {
      THROW_EXCEPTION("Vulkan shuffle map must use INT32 storage");
    }

    const auto dimensions = sharedDimensions(dimension);
    std::vector<sd::NDArray*> inputs;
    std::vector<sd::NDArray*> outputs;
    std::vector<sd::LongType> integerArguments;
    inputs.reserve(static_cast<size_t>(N + 1));
    outputs.reserve(static_cast<size_t>(N));
    integerArguments.push_back(N);

    sd::LongType commonItemCount = -1;
    for (int arrayIndex = 0; arrayIndex < N; ++arrayIndex) {
      auto* source = x[arrayIndex];
      auto* destination = z[arrayIndex];
      if (source == nullptr || destination == nullptr) {
        THROW_EXCEPTION("Vulkan shuffle received a null payload array");
      }
      if (source->dataType() != destination->dataType()) {
        THROW_EXCEPTION(
            "Vulkan shuffle source and destination data types must match per array");
      }

      requireExactAlias(source, destination, "Vulkan shuffle");
      requireDimensions(source, dimensions, "Vulkan shuffle");
      requireDimensions(destination, dimensions, "Vulkan shuffle");

      const auto itemCount = indexedItemCount(source, dimensions);
      if (commonItemCount < 0) {
        commonItemCount = itemCount;
      } else if (itemCount != commonItemCount) {
        THROW_EXCEPTION(
            "Vulkan shuffle arrays must expose the same indexed item count");
      }

      integerArguments.push_back(
          static_cast<sd::LongType>(dimensions.size()));
      integerArguments.insert(integerArguments.end(), dimensions.begin(),
                              dimensions.end());
      inputs.push_back(source);
      outputs.push_back(destination);
    }

    if (shuffleMap->lengthOf() < commonItemCount) {
      THROW_EXCEPTION(
          "Vulkan shuffle map is shorter than the indexed item count");
    }

    inputs.push_back(shuffleMap);
    executeIndexingDescriptor<sd::ops::legacy_shuffle>(
        extras, inputs, outputs, integerArguments);
  });
}


#endif  // SD_VULKAN && HAVE_VULKAN
