/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <config.h>

#if defined(SD_VULKAN) && defined(HAVE_VULKAN) && HAVE_VULKAN

#include <array/DataTypeUtils.h>
#include <dsp/NativeOpsDsp.h>
#include <execution/vulkan/VulkanExecutionStream.h>
#include <graph/vulkan/VulkanDeviceContext.h>
#include <graph/vulkan/VulkanDeviceManager.h>
#include <graph/vulkan/VulkanMemoryPool.h>
#include <legacy/NativeOps.h>

#include <cstdint>
#include <exception>
#include <limits>
#include <string>
#include <utility>
#include <vector>


namespace {

constexpr int kCollectiveSuccess = 0;
constexpr int kCollectiveInvalidArgument = -1;
constexpr int kCollectiveCapabilityUnavailable = -2;
constexpr int kCollectiveSubmissionFailed = -3;

struct VulkanCollectiveCommunicator {
  int worldSize;
  int rank;
  int deviceId;
};

void collectiveError(int code, const std::string& message) {
  const auto status = code == kCollectiveInvalidArgument
                          ? sd::Status::BAD_INPUT
                          : sd::Status::KERNEL_FAILURE;
  safeSetErrorContext(static_cast<int>(status), message.c_str());
}

bool validSingletonParameters(int numRanks, int rankId, int deviceId,
                              std::string& error) {
  if (numRanks != 1 || rankId != 0) {
    error =
        "Vulkan collective capability unavailable: this backend exposes no "
        "shared multi-physical-device Vulkan device-group transport; only "
        "worldSize=1, rank=0 is supported";
    return false;
  }

  auto& manager = sd::graph::VulkanDeviceManager::getInstance();
  if (!manager.initialize()) {
    error = "Vulkan collective initialization failed: Vulkan is unavailable";
    return false;
  }
  if (deviceId < 0 || deviceId >= manager.deviceCount()) {
    error = "Vulkan collective initialization failed: deviceId is outside the "
            "enumerated Vulkan device range";
    return false;
  }
  auto* context = sd::graph::VulkanDeviceContext::getContext(deviceId);
  if (context == nullptr || context->isLost()) {
    error =
        "Vulkan collective initialization failed: device context is unavailable or lost";
    return false;
  }
  return true;
}

VulkanCollectiveCommunicator* communicator(sd::Pointer handle,
                                           const char* operation) {
  if (handle == nullptr) {
    collectiveError(kCollectiveInvalidArgument,
                    std::string("Vulkan ") + operation +
                        " failed: communicator handle is null");
    return nullptr;
  }
  auto* comm = reinterpret_cast<VulkanCollectiveCommunicator*>(handle);
  if (comm->worldSize != 1 || comm->rank != 0) {
    collectiveError(
        kCollectiveCapabilityUnavailable,
        std::string("Vulkan ") + operation +
            " capability unavailable: communicator is not a supported "
            "world-size-one communicator");
    return nullptr;
  }
  return comm;
}

bool elementSize(int dataType, uint64_t& size) {
  try {
    size = static_cast<uint64_t>(
        sd::DataTypeUtils::sizeOf(static_cast<sd::DataType>(dataType)));
    return size != 0;
  } catch (const std::exception&) {
    return false;
  }
}

bool checkedByteCount(sd::LongType count, int dataType, VkDeviceSize& bytes,
                      const char* operation) {
  if (count < 0) {
    collectiveError(kCollectiveInvalidArgument,
                    std::string("Vulkan ") + operation +
                        " failed: element count is negative");
    return false;
  }

  uint64_t width = 0;
  if (!elementSize(dataType, width)) {
    collectiveError(kCollectiveInvalidArgument,
                    std::string("Vulkan ") + operation +
                        " failed: unsupported data type ordinal " +
                        std::to_string(dataType));
    return false;
  }

  const uint64_t elements = static_cast<uint64_t>(count);
  if (elements > std::numeric_limits<uint64_t>::max() / width) {
    collectiveError(kCollectiveInvalidArgument,
                    std::string("Vulkan ") + operation +
                        " failed: byte count overflow");
    return false;
  }
  bytes = static_cast<VkDeviceSize>(elements * width);
  return true;
}

sd::graph::VulkanExecutionStream* resolveCollectiveStream(
    sd::Pointer streamHandle, int deviceId, const char* operation) {
  auto* stream =
      streamHandle == nullptr
          ? sd::graph::VulkanExecutionStream::currentOrDefault(deviceId)
          : sd::graph::VulkanExecutionStream::fromOpaque(streamHandle, false);
  if (stream == nullptr || !stream->isActive()) {
    collectiveError(kCollectiveInvalidArgument,
                    std::string("Vulkan ") + operation +
                        " failed: execution stream is unavailable");
    return nullptr;
  }
  if (stream->deviceId() != deviceId) {
    collectiveError(kCollectiveInvalidArgument,
                    std::string("Vulkan ") + operation +
                        " failed: stream does not belong to the communicator "
                        "device");
    return nullptr;
  }
  return stream;
}

int enqueueSingletonCopy(VulkanCollectiveCommunicator* comm,
                         sd::Pointer sendBuffer, sd::Pointer receiveBuffer,
                         sd::LongType count, int dataType,
                         sd::Pointer streamHandle, const char* operation) {
  if (sendBuffer == nullptr || receiveBuffer == nullptr) {
    collectiveError(kCollectiveInvalidArgument,
                    std::string("Vulkan ") + operation +
                        " failed: send or receive buffer is null");
    return kCollectiveInvalidArgument;
  }

  VkDeviceSize bytes = 0;
  if (!checkedByteCount(count, dataType, bytes, operation)) {
    return kCollectiveInvalidArgument;
  }
  if (bytes == 0) return kCollectiveSuccess;

  auto& pool = sd::graph::VulkanMemoryPool::getInstance();
  sd::graph::VulkanAllocRecord source;
  sd::graph::VulkanAllocRecord destination;
  if (!pool.queryRecord(sendBuffer, source) ||
      !pool.queryRecord(receiveBuffer, destination)) {
    collectiveError(
        kCollectiveInvalidArgument,
        std::string("Vulkan ") + operation +
            " failed: buffers must be VulkanMemoryPool device allocations");
    return kCollectiveInvalidArgument;
  }
  if (source.deviceId != comm->deviceId ||
      destination.deviceId != comm->deviceId) {
    collectiveError(
        kCollectiveInvalidArgument,
        std::string("Vulkan ") + operation +
            " failed: buffer ownership does not match the communicator device");
    return kCollectiveInvalidArgument;
  }
  if (bytes > source.logicalSize || bytes > destination.logicalSize) {
    collectiveError(kCollectiveInvalidArgument,
                    std::string("Vulkan ") + operation +
                        " failed: requested transfer exceeds a buffer");
    return kCollectiveInvalidArgument;
  }

  auto* stream =
      resolveCollectiveStream(streamHandle, comm->deviceId, operation);
  if (stream == nullptr) return kCollectiveInvalidArgument;
  if (sendBuffer == receiveBuffer) return kCollectiveSuccess;

  if (!pool.copyDeviceToDeviceAsync(receiveBuffer, sendBuffer, bytes, stream)) {
    collectiveError(kCollectiveSubmissionFailed,
                    std::string("Vulkan ") + operation +
                        " failed: device-local transfer submission failed");
    return kCollectiveSubmissionFailed;
  }
  return kCollectiveSuccess;
}

int unsupportedPointToPoint(const char* operation) {
  collectiveError(
      kCollectiveCapabilityUnavailable,
      std::string("Vulkan ") + operation +
          " capability unavailable: no Vulkan point-to-point transport is "
          "configured; host staging and synthetic peer support are forbidden");
  return kCollectiveCapabilityUnavailable;
}

struct PendingCollective {
  VulkanCollectiveCommunicator* communicator;
  sd::Pointer sendBuffer;
  sd::Pointer receiveBuffer;
  sd::LongType count;
  int dataType;
  sd::Pointer stream;
  const char* operation;
};

thread_local bool collectiveGroupOpen = false;
thread_local std::vector<PendingCollective> pendingCollectives;

int enqueueOrGroup(VulkanCollectiveCommunicator* comm,
                   sd::Pointer sendBuffer, sd::Pointer receiveBuffer,
                   sd::LongType count, int dataType, sd::Pointer stream,
                   const char* operation) {
  if (!collectiveGroupOpen) {
    return enqueueSingletonCopy(comm, sendBuffer, receiveBuffer, count,
                                dataType, stream, operation);
  }
  pendingCollectives.push_back(
      {comm, sendBuffer, receiveBuffer, count, dataType, stream, operation});
  return kCollectiveSuccess;
}

}  // namespace

sd::Pointer ncclCommInit(int numRanks, int rankId, int deviceId) {
  std::string error;
  if (!validSingletonParameters(numRanks, rankId, deviceId, error)) {
    const int code =
        numRanks == 1 && rankId == 0 ? kCollectiveInvalidArgument
                                     : kCollectiveCapabilityUnavailable;
    collectiveError(code, error);
    return nullptr;
  }

  auto* comm = new VulkanCollectiveCommunicator{1, 0, deviceId};
  return reinterpret_cast<sd::Pointer>(comm);
}

sd::Pointer ncclCommInitWithId(int numRanks, int rankId,
                               sd::Pointer uniqueId) {
  (void)uniqueId;
  if (numRanks == 1 && rankId == 0) {
    collectiveError(
        kCollectiveCapabilityUnavailable,
        "Vulkan ncclCommInitWithId capability unavailable: unique-ID bootstrap "
        "is a multi-process transport operation and no Vulkan transport is "
        "configured");
  } else {
    collectiveError(
        kCollectiveCapabilityUnavailable,
        "Vulkan ncclCommInitWithId capability unavailable: no shared Vulkan "
        "device-group or external transport is configured");
  }
  return nullptr;
}

sd::Pointer ncclGetUniqueId() {
  collectiveError(
      kCollectiveCapabilityUnavailable,
      "Vulkan ncclGetUniqueId capability unavailable: unique-ID bootstrap "
      "requires a configured multi-process transport");
  return nullptr;
}

void ncclCommDestroy(sd::Pointer commHandle) {
  delete reinterpret_cast<VulkanCollectiveCommunicator*>(commHandle);
}

int ncclDoAllReduce(sd::Pointer commHandle, sd::Pointer sendBuf,
                    sd::Pointer recvBuf, sd::LongType numElements,
                    int dataType, int reduceOp, sd::Pointer stream) {
  auto* comm = communicator(commHandle, "AllReduce");
  if (comm == nullptr) return kCollectiveInvalidArgument;
  if (reduceOp < 0 || reduceOp > 4) {
    collectiveError(kCollectiveInvalidArgument,
                    "Vulkan AllReduce failed: unsupported reduction operation");
    return kCollectiveInvalidArgument;
  }
  return enqueueOrGroup(comm, sendBuf, recvBuf, numElements, dataType,
                        stream, "AllReduce");
}

int ncclDoAllGather(sd::Pointer commHandle, sd::Pointer sendBuf,
                    sd::Pointer recvBuf, sd::LongType sendCount,
                    int dataType, sd::Pointer stream) {
  auto* comm = communicator(commHandle, "AllGather");
  if (comm == nullptr) return kCollectiveInvalidArgument;
  return enqueueOrGroup(comm, sendBuf, recvBuf, sendCount, dataType,
                        stream, "AllGather");
}

int ncclDoReduceScatter(sd::Pointer commHandle, sd::Pointer sendBuf,
                        sd::Pointer recvBuf, sd::LongType recvCount,
                        int dataType, int reduceOp, sd::Pointer stream) {
  auto* comm = communicator(commHandle, "ReduceScatter");
  if (comm == nullptr) return kCollectiveInvalidArgument;
  if (reduceOp < 0 || reduceOp > 4) {
    collectiveError(
        kCollectiveInvalidArgument,
        "Vulkan ReduceScatter failed: unsupported reduction operation");
    return kCollectiveInvalidArgument;
  }
  return enqueueOrGroup(comm, sendBuf, recvBuf, recvCount, dataType,
                        stream, "ReduceScatter");
}

int ncclDoSend(sd::Pointer commHandle, sd::Pointer sendBuf,
               sd::LongType numElements, int dataType, int peerRank,
               sd::Pointer stream) {
  (void)commHandle;
  (void)sendBuf;
  (void)numElements;
  (void)dataType;
  (void)peerRank;
  (void)stream;
  return unsupportedPointToPoint("Send");
}

int ncclDoRecv(sd::Pointer commHandle, sd::Pointer recvBuf,
               sd::LongType numElements, int dataType, int peerRank,
               sd::Pointer stream) {
  (void)commHandle;
  (void)recvBuf;
  (void)numElements;
  (void)dataType;
  (void)peerRank;
  (void)stream;
  return unsupportedPointToPoint("Recv");
}

int ncclGroupStart() {
  if (collectiveGroupOpen) {
    collectiveError(kCollectiveInvalidArgument,
                    "Vulkan ncclGroupStart failed: a group is already open on "
                    "this thread");
    return kCollectiveInvalidArgument;
  }
  pendingCollectives.clear();
  collectiveGroupOpen = true;
  return kCollectiveSuccess;
}

int ncclGroupEnd() {
  if (!collectiveGroupOpen) {
    collectiveError(kCollectiveInvalidArgument,
                    "Vulkan ncclGroupEnd failed: no group is open on this "
                    "thread");
    return kCollectiveInvalidArgument;
  }

  collectiveGroupOpen = false;
  auto operations = std::move(pendingCollectives);
  pendingCollectives.clear();

  int firstError = kCollectiveSuccess;
  for (const auto& operation : operations) {
    const int status = enqueueSingletonCopy(
        operation.communicator, operation.sendBuffer, operation.receiveBuffer,
        operation.count, operation.dataType, operation.stream,
        operation.operation);
    if (status != kCollectiveSuccess && firstError == kCollectiveSuccess) {
      firstError = status;
    }
  }
  return firstError;
}


#endif  // SD_VULKAN && HAVE_VULKAN
