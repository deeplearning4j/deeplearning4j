/* ******************************************************************************
 * NDArray_core.cu - Core NDArray CUDA functions (sync, buffer, basic ops)
 * Split from NDArray.cu to reduce object file size for large binary builds
 ******************************************************************************/

#include <array/NDArray.h>
#include <array/NDArrayFactory.h>

#include <helpers/PointersManager.h>
#include <execution/AffinityManager.h>
#include <system/Environment.h>
#include <loops/special_kernels.h>

#if defined(SD_GCC_FUNCTRACE)
#include <exceptions/backward.hpp>
using namespace backward;
#endif

namespace sd {

void* NDArray::platformBuffer() { return specialBuffer(); }

void NDArray::syncToDevice() {
  if(_buffer == nullptr) return;

  auto currentDeviceId = AffinityManager::currentDeviceId();
  auto bufferDeviceId = _buffer->deviceId();

  if (currentDeviceId != _deviceId || currentDeviceId != bufferDeviceId) {
    const_cast<NDArray*>(this)->setShapeInfo(this->shapeInfo());
    _buffer->migrate();
    _deviceId = currentDeviceId;
  }

  _buffer->syncToSpecial();
}

void NDArray::syncToHost() { if(_buffer != nullptr) _buffer->syncToPrimary(getContext()); }
void NDArray::forceSyncToHost() { if(_buffer != nullptr) _buffer->syncToPrimary(getContext(), true); }
void NDArray::tickWriteHost() { if(_buffer != nullptr) _buffer->writePrimary(); }
void NDArray::tickWriteDevice() { if(_buffer != nullptr) _buffer->writeSpecial(); }
void NDArray::tickReadHost() { if(_buffer != nullptr) _buffer->readPrimary(); }
void NDArray::tickReadDevice() { if(_buffer != nullptr) _buffer->readSpecial(); }

void NDArray::tickBothActual() {
  if(_buffer == nullptr) return;
  _buffer->writePrimary();
  _buffer->readSpecial();
}

bool NDArray::isActualOnHostSide() { return _buffer == nullptr ? true : _buffer->isPrimaryActual(); }
bool NDArray::isActualOnDeviceSide() { return _buffer == nullptr ? true : _buffer->isSpecialActual(); }

void NDArray::makeBothBuffersActual() {
  if (!isActualOnHostSide()) syncToHost();
  if (!isActualOnDeviceSide()) syncToDevice();
}

void NDArray::synchronize(const char* msg) {
  // During CUDA graph capture, stream synchronization is illegal and breaks
  // capture. Captured kernels are ordered by stream semantics, so explicit
  // sync is not required inside the captured region.
  if (tl_graphExecutionActive) {
    return;
  }

  auto res = cudaStreamSynchronize(*(getContext()->getCudaStream()));
  if (res != 0) {
    std::string message = msg + std::string(": synchronization failed !");
    THROW_EXCEPTION(message.c_str());
  }
}

void NDArray::syncShape() {
  // During CUDA graph capture, use async copy on the active capture stream.
  // Using legacy stream 0 here can invalidate capture when the graph is being
  // recorded on a different stream.
  if (tl_graphExecutionActive) {
    cudaStream_t stream = tl_graphCaptureStream;
    if (stream == nullptr && getContext() != nullptr && getContext()->getCudaStream() != nullptr) {
      stream = *(getContext()->getCudaStream());
    }
    if (stream == nullptr) {
      stream = cudaStreamPerThread;
    }
    cudaMemcpyAsync(const_cast<LongType*>(specialShapeInfo()), shapeInfo(),
                    shape::shapeInfoByteLength(shapeInfo()), cudaMemcpyHostToDevice, stream);
    return;
  }
  // Use cudaStreamPerThread instead of synchronous cudaMemcpy on the legacy
  // stream (stream 0). Stream 0 causes error 906 when another thread on the
  // same device is mid-CUDA-graph-capture.
  cudaMemcpyAsync(const_cast<LongType*>(specialShapeInfo()), shapeInfo(),
                  shape::shapeInfoByteLength(shapeInfo()), cudaMemcpyHostToDevice, cudaStreamPerThread);
  cudaStreamSynchronize(cudaStreamPerThread);
}

void* NDArray::specialBuffer() {
  if (_buffer == nullptr) {
    return nullptr;
  }

  auto currentDeviceId = AffinityManager::currentDeviceId();
  auto bufferDeviceId = _buffer->deviceId();

  void* specialBuf = _buffer->special();

  if (specialBuf == nullptr || bufferDeviceId != currentDeviceId) {
    syncToDevice();
    tickReadHost();
    specialBuf = _buffer->special();
    if (specialBuf == nullptr) {
      return nullptr;
    }
  }

  return static_cast<int8_t*>(specialBuf) + (offset() * sizeOfT());
}

void NDArray::prepareSpecialUse(const std::vector<NDArray*>& writeList,
                                const std::vector<NDArray*>& readList, bool synchronizeWritables) {
  for (const auto& a : readList)
    if (a != nullptr) a->syncToDevice();

  for (const auto& a : writeList) {
    if (a != nullptr) {
      auto dataBuffer = a->getDataBuffer();
      if (dataBuffer != nullptr) {
        dataBuffer->allocateSpecial();
      }
      if (synchronizeWritables) a->syncToDevice();
    }
  }
}

void NDArray::registerSpecialUse(const std::vector<NDArray*>& writeList,
                                 const std::vector<NDArray*>& readList) {
  for (const auto& p : readList)
    if (p != nullptr) p->tickReadDevice();

  for (const auto& p : writeList)
    if (p != nullptr) p->tickWriteDevice();
}

void NDArray::preparePrimaryUse(const std::vector<NDArray*>& writeList,
                                const std::vector<NDArray*>& readList, bool synchronizeWritables) {
  for (const auto& a : readList)
    if (a != nullptr) a->syncToHost();

  for (const auto& a : writeList) {
    if (a != nullptr) {
      auto dataBuffer = a->getDataBuffer();
      if (dataBuffer != nullptr) {
        dataBuffer->allocatePrimary();
      }
      if (synchronizeWritables) a->syncToHost();
    }
  }
}

void NDArray::registerPrimaryUse(const std::vector<NDArray*>& writeList,
                                 const std::vector<NDArray*>& readList) {
  for (const auto& p : readList)
    if (p != nullptr) p->tickReadHost();

  for (const auto& p : writeList)
    if (p != nullptr) p->tickWriteHost();
}

void NDArray::printBufferDebug(const char* msg, sd::LongType offset, sd::LongType limit) {
  if (msg) sd_printf("%s:\n", msg);

  if(limit < 0) limit = lengthOf();

  sd_printf("NDArray: Shape=[", 0);
  for (int i = 0; i < rankOf(); i++) {
    sd_printf("%lld", (long long)sizeAt(i));
    if (i < rankOf() - 1) sd_printf(",", 0);
  }
  sd_printf("], DataType=%s,  Order=%c\n",
            DataTypeUtils::asString(dataType()).c_str(), ordering());

#if defined(SD_GCC_FUNCTRACE)
  printf("========================================================\n");
  Printer p;
  StackTrace st;
  st.load_here();
  p.print(st);
  printf("========================================================\n");
  fflush(stdout);
#endif

  if (_buffer != nullptr) {
    _buffer->printBufferDebug("Buffer contents", offset, limit);
  } else {
    sd_printf("Buffer is nullptr\n", 0);
  }
}

void NDArray::swapUnsafe(NDArray& other) {
  auto xType = this->dataType();

  if (xType != other.dataType())
    THROW_EXCEPTION("NDArray::swapUnsage method: both arrays must have the same data type");

  if (specialBuffer() == nullptr || other.specialBuffer() == nullptr)
    THROW_EXCEPTION("NDArray::swapUnsafe method: input array should not be empty!");

  if (lengthOf() != other.lengthOf())
    THROW_EXCEPTION("NDArray::swapUnsafe method: input arrays should have the same length!");

  PointersManager manager(getContext(), "NDArray::swapUnsafe");

  prepareSpecialUse({&other, this}, {&other, this});
  BUILD_SINGLE_SELECTOR(xType, templatedSwapUnsafe,
                        (specialBuffer(), specialShapeInfo(), other.specialBuffer(), other.specialShapeInfo(),
                            getContext()->getCudaStream()),
                        SD_COMMON_TYPES);
  registerSpecialUse({&other, this}, {&other, this});

  manager.synchronize();
}

}  // namespace sd
