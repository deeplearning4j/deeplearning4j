/* ******************************************************************************
 *
 * Copyright (c) 2026 Eclipse Foundation
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <config.h>

#if defined(SD_VULKAN) && defined(HAVE_VULKAN) && HAVE_VULKAN

#include <execution/LaunchContext.h>
#include <execution/vulkan/VulkanExecutionStream.h>
#include <graph/vulkan/VulkanDeviceManager.h>
#include <graph/vulkan/VulkanMemoryPool.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/shape.h>
#include <helpers/vulkan/VulkanRandomBuffer.h>
#include <legacy/NativeOpExecutioner.h>
#include <legacy/NativeOps.h>

#include <cstdint>
#include <type_traits>

namespace sd {
namespace random {

static_assert(std::is_base_of<RandomBuffer, VulkanRandomBuffer>::value,
              "Vulkan random state must preserve the common RandomBuffer ABI");

VulkanRandomBuffer::VulkanRandomBuffer(sd::LongType seed, sd::LongType size,
                                       uint64_t* hostBuffer,
                                       uint64_t* deviceBuffer, int deviceId)
    : RandomBuffer(seed, size, hostBuffer), _deviceId(deviceId) {
  setDeviceBuffer(deviceBuffer);
  _metadata = sd::graph::VulkanMemoryPool::getInstance().allocate(
      deviceId, sizeof(RandomBuffer));
  if (_metadata == nullptr) {
    THROW_EXCEPTION("Vulkan random metadata allocation failed");
  }
}

sd::Pointer VulkanRandomBuffer::metadataPointer() const {
  return reinterpret_cast<sd::Pointer>(_metadata);
}

VulkanRandomBuffer::~VulkanRandomBuffer() {
  if (_metadata == nullptr) return;
  sd::graph::VulkanMemoryPool::getInstance().freeSynchronized(_metadata);
  _metadata = nullptr;
}

void VulkanRandomBuffer::propagateToDevice(void* opaqueStream) {
  auto* stream =
      sd::graph::VulkanExecutionStream::fromOpaque(opaqueStream, false);
  auto* commonState = static_cast<RandomBuffer*>(this);
  if (stream == nullptr || !stream->isActive() ||
      stream->deviceId() != _deviceId ||
      !stream->enqueueCopy(_metadata, commonState, sizeof(RandomBuffer), 1)) {
    THROW_EXCEPTION("Vulkan random metadata transfer failed");
  }
}

}  // namespace random
}  // namespace sd


namespace {

struct RandomBinding {
  uint64_t *host;
  uint64_t *device;
  int deviceId;
  sd::graph::VulkanExecutionStream *stream;
};

RandomBinding resolveRandomBinding(sd::Pointer *extraPointers,
                                   sd::Pointer deviceBuffer) {
  if (extraPointers == nullptr || extraPointers[0] == nullptr ||
      extraPointers[1] == nullptr || deviceBuffer == nullptr) {
    THROW_EXCEPTION("Vulkan random state received an invalid binding");
  }

  auto &pool = sd::graph::VulkanMemoryPool::getInstance();
  sd::graph::VulkanAllocRecord record;
  if (!pool.queryRecord(deviceBuffer, record)) {
    THROW_EXCEPTION("Vulkan random device buffer is not pool-owned");
  }

  auto *stream = sd::graph::VulkanExecutionStream::fromOpaque(
      extraPointers[1], false);
  if (stream == nullptr || !stream->isActive() ||
      stream->deviceId() != record.deviceId) {
    THROW_EXCEPTION("Vulkan random stream does not own the device buffer");
  }

  return {reinterpret_cast<uint64_t *>(extraPointers[0]),
          reinterpret_cast<uint64_t *>(deviceBuffer), record.deviceId, stream};
}

sd::random::VulkanRandomBuffer* asVulkanRandomBuffer(
    sd::Pointer opaqueBuffer) {
  auto* common =
      reinterpret_cast<sd::random::RandomBuffer*>(opaqueBuffer);
  if (common == nullptr) {
    THROW_EXCEPTION("Vulkan random state received a null buffer");
  }
  return static_cast<sd::random::VulkanRandomBuffer*>(common);
}

sd::graph::VulkanExecutionStream* resolveStateStream(
    sd::Pointer* extraPointers, sd::random::VulkanRandomBuffer* buffer) {
  if (extraPointers == nullptr || extraPointers[1] == nullptr ||
      buffer == nullptr) {
    THROW_EXCEPTION("Vulkan random state received an invalid stream");
  }

  sd::graph::VulkanAllocRecord record;
  if (!sd::graph::VulkanMemoryPool::getInstance().queryRecord(
          buffer->metadataPointer(), record)) {
    THROW_EXCEPTION("Vulkan random metadata is not pool-owned");
  }

  auto* stream = sd::graph::VulkanExecutionStream::fromOpaque(
      extraPointers[1], false);
  if (stream == nullptr || !stream->isActive() ||
      stream->deviceId() != record.deviceId ||
      stream->deviceId() != buffer->deviceId()) {
    THROW_EXCEPTION("Vulkan random stream does not own the metadata");
  }
  return stream;
}

void copyRandomValues(sd::random::RandomBuffer *buffer,
                      sd::graph::VulkanExecutionStream *stream) {
  const auto bytes =
      static_cast<VkDeviceSize>(buffer->getSize()) * sizeof(uint64_t);
  if (!stream->enqueueCopy(buffer->getDeviceBuffer(), buffer->getBuffer(),
                           bytes, 1)) {
    THROW_EXCEPTION("Vulkan random value transfer failed");
  }
}

}  // namespace

void execRandom(sd::Pointer *extraPointers, int opNum, sd::Pointer stateHost,
                OpaqueNDArray z, void *extraArguments) {
  try {
    z->prepareSpecialUse({z}, {});

    auto lc = sd::LaunchContext::defaultContext();

    NativeOpExecutioner::execRandom(
        lc, opNum, stateHost,
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
        sd::ConstantShapeHelper::getInstance()
            .bufferForShapeInfo(z->shapeInfo())
            ->special(),
        extraArguments);

    z->registerSpecialUse({z}, {});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(
        e.what());
  }
}

void execRandom2(sd::Pointer *extraPointers, int opNum, sd::Pointer stateHost,
                 OpaqueNDArray x, OpaqueNDArray z, void *extraArguments) {
  try {
    x->prepareSpecialUse({z}, {x});

    auto lc = sd::LaunchContext::defaultContext();

    NativeOpExecutioner::execRandom(
        lc, opNum, stateHost,
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
        x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
        sd::ConstantShapeHelper::getInstance()
            .bufferForShapeInfo(x->shapeInfo())
            ->special(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
        sd::ConstantShapeHelper::getInstance()
            .bufferForShapeInfo(z->shapeInfo())
            ->special(),
        extraArguments);

    x->registerSpecialUse({z}, {x});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(
        e.what());
  }
}

void execRandom3(sd::Pointer *extraPointers, int opNum, sd::Pointer stateHost,
                 OpaqueNDArray x, OpaqueNDArray y, OpaqueNDArray z,
                 void *extraArguments) {
  try {
    x->prepareSpecialUse({z}, {x, y});

    auto lc = sd::LaunchContext::defaultContext();
    NativeOpExecutioner::execRandom(
        lc, opNum, stateHost,
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
        x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
        x->specialShapeInfo(),
        shape::isEmptyConst(y->shapeInfo()) ? nullptr : y->buffer(),
        y->shapeInfo(),
        shape::isEmptyConst(y->shapeInfo()) ? nullptr : y->specialBuffer(),
        y->specialShapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
        z->specialShapeInfo(), extraArguments);

    x->registerSpecialUse({z}, {x, y});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(
        e.what());
  }
}

sd::Pointer initRandom(sd::Pointer *extraPointers, long seed, long bufferSize,
                       sd::Pointer ptrToBuffer) {
  if (bufferSize <= 0) {
    THROW_EXCEPTION("Vulkan random buffer size must be positive");
  }

  const RandomBinding binding =
      resolveRandomBinding(extraPointers, ptrToBuffer);
  auto* buffer = new sd::random::VulkanRandomBuffer(
      seed, bufferSize, binding.host, binding.device, binding.deviceId);
  try {
    buffer->propagateToDevice(binding.stream);

    sd::random::Xoroshiro128 generator(
        static_cast<sd::random::RandomBuffer*>(buffer));
    generator.refreshBuffer();
    copyRandomValues(buffer, binding.stream);
  } catch (...) {
    binding.stream->synchronize();
    delete buffer;
    throw;
  }
  return static_cast<sd::random::RandomBuffer*>(buffer);
}

void destroyRandom(sd::Pointer ptrBuffer) {
  if (ptrBuffer == nullptr) return;
  auto* buffer = asVulkanRandomBuffer(ptrBuffer);

  sd::graph::VulkanAllocRecord record;
  auto& pool = sd::graph::VulkanMemoryPool::getInstance();
  if (!pool.queryRecord(buffer->metadataPointer(), record) ||
      record.deviceId != buffer->deviceId() ||
      !sd::graph::VulkanExecutionStream::synchronizeDevice(record.deviceId)) {
    THROW_EXCEPTION("Vulkan random state synchronization failed");
  }
  delete buffer;
}

void refreshBuffer(sd::Pointer* extraPointers, long seed,
                   sd::Pointer ptrRandom) {
  auto* buffer = asVulkanRandomBuffer(ptrRandom);
  auto* stream = resolveStateStream(extraPointers, buffer);
  if (!stream->synchronize()) {
    THROW_EXCEPTION("Vulkan random stream synchronization failed");
  }

  buffer->setSeed(seed);
  buffer->setOffset(0);
  buffer->propagateToDevice(stream);

  auto* common = static_cast<sd::random::RandomBuffer*>(buffer);
  sd::random::Xoroshiro128 generator(common);
  generator.refreshBuffer();
  copyRandomValues(common, stream);
}

void reSeedBuffer(sd::Pointer* extraPointers, long seed,
                  sd::Pointer ptrRandom) {
  auto* buffer = asVulkanRandomBuffer(ptrRandom);
  auto* stream = resolveStateStream(extraPointers, buffer);
  if (!stream->synchronize()) {
    THROW_EXCEPTION("Vulkan random stream synchronization failed");
  }

  buffer->reSeed(seed);
  buffer->setOffset(0);
  buffer->propagateToDevice(stream);
}


#endif  // SD_VULKAN && HAVE_VULKAN
