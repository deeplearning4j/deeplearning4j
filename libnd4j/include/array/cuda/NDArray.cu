/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef NDARRAY_CPP
#define NDARRAY_CPP
#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <exceptions/cuda_exception.h>
#include <exceptions/datatype_exception.h>
#include <helpers/ArrayUtils.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/MmulHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>
#include <helpers/logger.h>
#include <helpers/threshold.h>
#include <indexing/IndicesList.h>
#include <indexing/NDIndex.h>
#include <legacy/NativeOpExecutioner.h>
#include <loops/broadcasting.h>
#include <loops/pairwise_transform.h>
#include <loops/random.h>
#include <loops/special_kernels.h>
#include <loops/transform_same.h>
#include <memory/MemoryRegistrator.h>
#include <memory/Workspace.h>
#include <ops/gemm.h>
#include <ops/ops.h>
#include <ops/specials_cuda.h>

#include <array/NDArray.hXX>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace sd {

void* NDArray::platformBuffer() { return specialBuffer(); }
void const* NDArray::platformBuffer() const { return specialBuffer(); }

sd::LongType const* NDArray::platformShapeInfo() const { return specialShapeInfo(); }

void NDArray::syncToDevice() const {
  auto currentDeviceId = AffinityManager::currentDeviceId();
  if (currentDeviceId != _deviceId) {
    // first of all we update shapeInfo
    const_cast<NDArray*>(this)->setShapeInfo(this->shapeInfo());

    // now we actually migrate data buffer
    _buffer->migrate();
  }

  _buffer->syncToSpecial();
}

void NDArray::syncToHost() const { _buffer->syncToPrimary(getContext()); }
void NDArray::tickWriteHost() const { _buffer->writePrimary(); }
void NDArray::tickWriteDevice() const { _buffer->writeSpecial(); }
void NDArray::tickReadHost() const { _buffer->readPrimary(); }
void NDArray::tickReadDevice() const { _buffer->readSpecial(); }
void NDArray::tickBothActual() const {
  _buffer->writePrimary();
  _buffer->readSpecial();
}
bool NDArray::isActualOnHostSide() const { return _buffer->isPrimaryActual(); }
bool NDArray::isActualOnDeviceSide() const { return _buffer->isSpecialActual(); }
void NDArray::makeBothBuffersActual() const {
  if (!isActualOnHostSide()) syncToHost();
  if (!isActualOnDeviceSide()) syncToDevice();
}

///////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL static void fillAsTriangularCuda(const void* vx, const sd::LongType* xShapeInfo, void* vz,
                                           const sd::LongType* zShapeInfo, const T val, const int lower,
                                           const int upper, char direction, bool includeEdges) {
  const auto x = reinterpret_cast<const T*>(vx);
  auto z = reinterpret_cast<T*>(vz);

  __shared__ sd::LongType zRank, xRank, areSameOffsets,
      *sharedMem;                              // xRank == zRank always, except when xRank = 1, in this case zRank = 2
  __shared__ sd::LongType zLen, totalThreads;  // xLen == zLen, except when xRank = 1, in this case zLen = 2*xLen

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    sharedMem = reinterpret_cast<sd::LongType *>(shmem);
    areSameOffsets = shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo);
    xRank = shape::rank(xShapeInfo);
    zRank = shape::rank(zShapeInfo);
    zLen = shape::length(zShapeInfo);
    totalThreads = gridDim.x * blockDim.x;
  }
  __syncthreads();

  auto coords = sharedMem + threadIdx.x * zRank;

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  bool dirU = direction == 'u';
  bool dirL = direction == 'l';
  for (sd::LongType i = tid; i < zLen; i += totalThreads) {
    shape::index2coords(i, zShapeInfo, coords);
    const auto zOffset = shape::getOffset(zShapeInfo, coords);
    auto row = coords[zRank - 2];
    auto col = coords[zRank - 1];
    auto lCompare = includeEdges ? row + lower <= col : row + lower < col;
    auto uCompare = includeEdges ? row + upper >= col : row + upper > col;
    if (dirU && lCompare || dirL && uCompare) {
      z[zOffset] = val;
    } else if (vx != vz) {  // when x and z are different arrays
      if (xRank != zRank) coords[0] = coords[1];
      const auto xOffset = areSameOffsets ? zOffset : shape::getOffset(xShapeInfo, coords);
      z[zOffset] = x[xOffset];
    }
  }
}

///////////////////////////////////////////////////////////////////
template <typename T>
void NDArray::fillAsTriangular(const float val, int lower, int upper, NDArray& target, const char direction,
                               const bool includeEdges) {
  if (isS()) THROW_EXCEPTION("NDArray::fillAsTriangular: you can't use this method on String array!");

  if (!isSameShape(target) &&
      !(rankOf() == 1 && target.rankOf() == 2 && sizeAt(0) == target.sizeAt(0) && sizeAt(0) == target.sizeAt(1)))
    throw std::string("NDArray::fillAsTriangular method: wrong shape of target array !");

  const int threadsPerBlock = SD_MAX_NUM_THREADS / 4;
  const int blocksPerGrid = (target.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
  const int sharedMem = threadsPerBlock * sizeof(int) * target.rankOf() + 128;

  PointersManager manager(getContext(), "NDArray::fillAsTriangular");

  NDArray::prepareSpecialUse({&target}, {this});
  fillAsTriangularCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *getContext()->getCudaStream()>>>(
      platformBuffer(), platformShapeInfo(), target.platformBuffer(), target.platformShapeInfo(), static_cast<T>(val),
      lower, upper, direction, includeEdges);
  NDArray::registerSpecialUse({&target}, {this});

  manager.synchronize();
}
BUILD_SINGLE_TEMPLATE(template SD_LIB_EXPORT void NDArray::fillAsTriangular,
                      (const float val, int lower, int upper, NDArray& target, const char direction,
                          const bool includeEdges),
                      SD_COMMON_TYPES);

////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL static void identityMatrixCuda(void* vx, const sd::LongType* xShapeInfo, const T val) {
  auto x = reinterpret_cast<T*>(vx);

  __shared__ sd::LongType rank, *sharedMem;
  __shared__ sd::LongType len, totalThreads;  // xLen == zLen, except when xRank = 1, in this case zLen = 2*xLen

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    sharedMem = reinterpret_cast<sd::LongType *>(shmem);
    rank = shape::rank(xShapeInfo);
    len = shape::length(xShapeInfo);
    totalThreads = gridDim.x * blockDim.x;
  }
  __syncthreads();

  auto coords = sharedMem + threadIdx.x * rank;

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (sd::LongType i = tid; i < len; i += totalThreads) {
    shape::index2coords(i, xShapeInfo, coords);
    const auto offset = shape::getOffset(xShapeInfo, coords);

    if (coords[rank - 2] == coords[rank - 1])  // row == col -> on diagonal
      x[offset] = val;
    else
      x[offset] = static_cast<T>(0);
  }
}

///////////////////////////////////////////////////////////////////
template <typename T>
static void identityMatrixCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                       const cudaStream_t* stream, void* vx, const sd::LongType* xShapeInfo,
                                       const float val) {
  identityMatrixCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, static_cast<T>(val));
}
BUILD_SINGLE_TEMPLATE(template void identityMatrixCudaLauncher,
                      (const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                          const cudaStream_t* stream, void* vx, const sd::LongType* xShapeInfo, const float val),
                      SD_COMMON_TYPES);

////////////////////////////////////////////////////////////////////////
void NDArray::setIdentity() {
  if (isS()) THROW_EXCEPTION("NDArray::setIdentity: you can't use this method on String array!");


  const int threadsPerBlock = SD_MAX_NUM_THREADS / 4;
  const int blocksPerGrid = (lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
  const int sharedMem = threadsPerBlock * sizeof(sd::LongType) * rankOf() + 128;

  PointersManager manager(getContext(), "NDArray::setIdentity");

  syncToDevice();
  BUILD_SINGLE_SELECTOR(dataType(), identityMatrixCudaLauncher,
                        (blocksPerGrid, threadsPerBlock, sharedMem, getContext()->getCudaStream(), platformBuffer(),
                            platformShapeInfo(), 1.f),
                        SD_COMMON_TYPES);
  tickWriteDevice();

  manager.synchronize();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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

////////////////////////////////////////////////////////////////////////
void NDArray::synchronize(const char* msg) const {
  auto res = cudaStreamSynchronize(*(getContext()->getCudaStream()));
  if (res != 0) {
    std::string message = msg + std::string(": synchronization failed !");
    THROW_EXCEPTION(message.c_str());
  }
}

////////////////////////////////////////////////////////////////////////
void NDArray::prepareSpecialUse(const std::vector<const NDArray*>& writeList,
                                const std::vector<const NDArray*>& readList, bool synchronizeWritables) {
  for (const auto& a : readList)
    if (a != nullptr) a->syncToDevice();

  for (const auto& a : writeList) {
    if (a != nullptr) {
      a->getDataBuffer()->allocateSpecial();
      if (synchronizeWritables) a->syncToDevice();
    }
  }
}

////////////////////////////////////////////////////////////////////////
void NDArray::registerSpecialUse(const std::vector<const NDArray*>& writeList,
                                 const std::vector<const NDArray*>& readList) {
  for (const auto& p : readList)
    if (p != nullptr) p->tickReadDevice();

  for (const auto& p : writeList)
    if (p != nullptr) p->tickWriteDevice();
}

////////////////////////////////////////////////////////////////////////
void NDArray::preparePrimaryUse(const std::vector<const NDArray*>& writeList,
                                const std::vector<const NDArray*>& readList, bool synchronizeWritables) {
  for (const auto& a : readList)
    if (a != nullptr) a->syncToHost();

  for (const auto& a : writeList) {
    if (a != nullptr) {
      a->getDataBuffer()->allocatePrimary();
      if (synchronizeWritables) a->syncToHost();
    }
  }
}

////////////////////////////////////////////////////////////////////////
void NDArray::registerPrimaryUse(const std::vector<const NDArray*>& writeList,
                                 const std::vector<const NDArray*>& readList) {
  for (const auto& p : readList)
    if (p != nullptr) p->tickReadHost();

  for (const auto& p : writeList)
    if (p != nullptr) p->tickWriteHost();
}

//////////////////////////////////////////////////////////////////////////
void NDArray::syncShape() const {
  cudaMemcpy(const_cast<sd::LongType*>(specialShapeInfo()), shapeInfo(), shape::shapeInfoByteLength(shapeInfo()),
             cudaMemcpyHostToDevice);
}

//////////////////////////////////////////////////////////////////////////
void const* NDArray::specialBufferWithOffset(sd::LongType offset) const {
  return specialBuffer() != nullptr ? static_cast<int8_t const*>(specialBuffer()) + (offset * sizeOfT()) : nullptr;
}

void* NDArray::specialBufferWithOffset(sd::LongType offset) {
  return specialBuffer() != nullptr ? static_cast<int8_t*>(specialBuffer()) + (offset * sizeOfT()) : nullptr;
}

//////////////////////////////////////////////////////////////////////////
// change an array by repeating it the number of times given by reps.
NDArray NDArray::tile(const std::vector<sd::LongType>& reps) const {
  int dim = reps.size();
  sd::LongType product = 1;
  for (const auto& item : reps) product *= item;

  if (product < 1) THROW_EXCEPTION("NDArray::tile method: one of the elements in reps array is zero !");

  int rankOld = rankOf();
  int diff = rankOld - dim;
  if (product == 1) {  // in this case 2 possibilities are present: just reshape or nothing to do
    NDArray result(*this);
    if (diff < 0) {                               // reshape to higher dimension
      std::vector<sd::LongType> shapeNew = reps;  // need to have unities at first "diff" positions of new shape
      memcpy(&shapeNew[-diff], result.shapeInfo() + 1,
             rankOld * sizeof(sd::LongType));  // put old shape numbers at rest of positions
      result.reshapei(ordering(), shapeNew);
    }
    return result;  // nothing to do, if diff >= 0 -> identity tile
  }

  // evaluate shapeInfo for resulting array
  auto newShapeInfo = ShapeUtils::evalTileShapeInfo(*this, reps, getContext()->getWorkspace());
  // create new buffer, in any case the memory amount new buffer points to is bigger then those for old _buffer
  std::shared_ptr<DataBuffer> newBuff = std::make_shared<DataBuffer>(shape::length(newShapeInfo) * sizeOfT(),
                                                                     dataType(), getContext()->getWorkspace(), true);
  // assign new shape and new buffer to resulting array
  ShapeDescriptor *descriptor = new ShapeDescriptor(newShapeInfo);
  NDArray result(newBuff,descriptor , getContext());
  delete descriptor;
  // fill newBuff, loop through all elements of newBuff
  // looping through buffer() goes automatically by means of getSubArrayIndex applying
  const auto resultLen = result.lengthOf();
  auto xType = this->dataType();
  auto stream = getContext()->getCudaStream();

  prepareSpecialUse({&result}, {this});
  BUILD_SINGLE_SELECTOR(xType, tileKernelH,
                        (this->specialBuffer(), this->specialShapeInfo(), result.specialBuffer(),
                            result.specialShapeInfo(), resultLen, stream),
                        SD_COMMON_TYPES);
  registerSpecialUse({&result}, {this});

  return result;
}

//////////////////////////////////////////////////////////////////////////
// change an array by repeating it the number of times given by reps.
void NDArray::tile(const std::vector<sd::LongType>& reps, NDArray& target) const {
  auto repProd = shape::prodLong(reps.data(), reps.size());
  if (repProd < 1) THROW_EXCEPTION("NDArray::tile: reps can't contain 0s");

  // evaluate true tile shapeInfo for comparison with target shapeInfo
  auto newShapeInfo = ShapeUtils::evalTileShapeInfo(*this, reps, getContext()->getWorkspace());
  if (!shape::equalsSoft(newShapeInfo, target.shapeInfo())) {
    THROW_EXCEPTION("NDArray::tile method - shapeInfo of target array is not suitable for tile operation !");
  }

  // fill newBuff, loop through all elements of newBuff
  // looping through buffer() goes automatically by means of getSubArrayIndex applying
  const int ews = target.ews();
  const int targetLen = target.lengthOf();
  auto stream = getContext()->getCudaStream();

  prepareSpecialUse({&target}, {this});
  BUILD_SINGLE_SELECTOR_TWICE(
      target.dataType(), tileKernelHH,
      (specialBuffer(), specialShapeInfo(), target.specialBuffer(), target.specialShapeInfo(), targetLen, ews, stream),
      SD_COMMON_TYPES);
  registerSpecialUse({&target}, {this});
}

//////////////////////////////////////////////////////////////////////////
void NDArray::tile(NDArray& target) const {
  if (rankOf() > target.rankOf())
    THROW_EXCEPTION(
        "NDArray::tile method - rank of target array must be bigger or equal to the rank of this array !");

  if (!ShapeUtils::areShapesBroadcastable(*this, target))
    THROW_EXCEPTION("NDArray::tile method - shapeInfo of target array is not suitable for tile operation !");

  // fill newBuff, loop through all elements of newBuff
  // looping through getBuffer() goes automatically by means of getSubArrayIndex applying
  const auto ews = target.ews();
  const auto targetLen = target.lengthOf();
  auto stream = getContext()->getCudaStream();

  prepareSpecialUse({&target}, {this});
  BUILD_SINGLE_SELECTOR_TWICE(
      target.dataType(), tileKernelHH,
      (specialBuffer(), specialShapeInfo(), target.specialBuffer(), target.specialShapeInfo(), targetLen, ews, stream),
      SD_COMMON_TYPES);
  registerSpecialUse({&target}, {this});
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
SD_KERNEL static void repeatCuda(const void* vx, const sd::LongType* xShapeInfo, void* vz,
                                 const sd::LongType* zShapeInfo, const sd::LongType* repeats, const sd::LongType repSize,
                                 const int axis) {
  const X* x = reinterpret_cast<const X*>(vx);
  Z* z = reinterpret_cast<Z*>(vz);

  __shared__ sd::LongType rank, *sharedMem;
  __shared__ sd::LongType zLen, totalThreads;  // xLen = zLen

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    sharedMem = reinterpret_cast<sd::LongType *>(shmem);

    rank = shape::rank(zShapeInfo);    // xRank = zRank
    zLen = shape::length(zShapeInfo);  // xLen <= zLen

    totalThreads = gridDim.x * blockDim.x;
  }

  __syncthreads();

  auto coords = sharedMem + threadIdx.x * rank;

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (sd::LongType i = tid; i < zLen; i += totalThreads) {
    shape::index2coords(i, zShapeInfo, coords);

    const auto zOffset = shape::getOffset(zShapeInfo, coords);

    if (repSize > 1) {
      for (sd::LongType j = 0; j < repSize; ++j) {
        coords[axis] -= repeats[j];
        if (coords[axis] < 0) {
          coords[axis] = j;
          break;
        }
      }
    } else
      coords[axis] /= repeats[0];

    z[zOffset] = x[shape::getOffset(xShapeInfo, coords)];
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
static void repeatCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                               const cudaStream_t* stream, const void* vx, const sd::LongType* xShapeInfo, void* vz,
                               const sd::LongType* zShapeInfo, const sd::LongType* repeats, const sd::LongType repSize, const sd::LongType axis) {
  repeatCuda<X, Z>
  <<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vz, zShapeInfo, repeats, repSize, axis);
}
BUILD_DOUBLE_TEMPLATE(template void repeatCudaLauncher,
                      (const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                          const cudaStream_t* stream, const void* vx, const sd::LongType* xShapeInfo, void* vz,
                          const sd::LongType* zShapeInfo, const sd::LongType* repeats, const sd::LongType repSize, const sd::LongType axis),
                      SD_COMMON_TYPES, SD_COMMON_TYPES);

//////////////////////////////////////////////////////////////////////////
// create new array by repeating it the number of times given by repeats
NDArray NDArray::repeat(const int axis, const std::vector<sd::LongType>& repeats) const {
  NDArray output('c', ShapeUtils::evalRepeatShape(axis, repeats, *this), dataType(), getContext());

  const int threadsPerBlock = SD_MAX_NUM_THREADS / 2;
  const int blocksPerGrid = (output.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
  const sd::LongType sharedMem = output.rankOf() * sizeof(sd::LongType) * threadsPerBlock + 128;

  PointersManager manager(getContext(), "NDArray::repeat(const int axis, const std::vector<int>& repeats)");

  const sd::LongType* reps = reinterpret_cast<sd::LongType*>(manager.replicatePointer(repeats.data(), repeats.size() * sizeof(sd::LongType)));

  prepareSpecialUse({&output}, {this});
  BUILD_SINGLE_SELECTOR_TWICE(
      dataType(), repeatCudaLauncher,
      (blocksPerGrid, threadsPerBlock, sharedMem, getContext()->getCudaStream(), specialBuffer(), specialShapeInfo(),
          output.specialBuffer(), output.specialShapeInfo(), reps, repeats.size(), axis),
      SD_COMMON_TYPES);
  prepareSpecialUse({&output}, {this});

  manager.synchronize();

  return output;
}

//////////////////////////////////////////////////////////////////////////
// fill array by repeating it the number of times given by repeats
void NDArray::repeat(const int axis, const std::vector<sd::LongType>& repeats, NDArray& target) const {
  if (!target.isSameShape(ShapeUtils::evalRepeatShape(axis, repeats, *this)))
    THROW_EXCEPTION(
        "NDArray::repeat(const int axis, const std::vector<int>& repeats, NDArray& target) method: wrong shape of "
        "target array!");

  const sd::LongType threadsPerBlock = SD_MAX_NUM_THREADS / 2;
  const sd::LongType blocksPerGrid = (target.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
  const sd::LongType sharedMem = target.rankOf() * sizeof(sd::LongType) * threadsPerBlock + 128;

  PointersManager manager(getContext(), "NDArray::repeat(const int axis, const std::vector<int>& repeats)");

  const sd::LongType* reps = reinterpret_cast<sd::LongType*>(manager.replicatePointer(repeats.data(), repeats.size() * sizeof(sd::LongType)));

  prepareSpecialUse({&target}, {this});
  BUILD_DOUBLE_SELECTOR(
      dataType(), target.dataType(), repeatCudaLauncher,
      (blocksPerGrid, threadsPerBlock, sharedMem, getContext()->getCudaStream(), specialBuffer(), specialShapeInfo(),
          target.specialBuffer(), target.specialShapeInfo(), reps, repeats.size(), axis),
      SD_COMMON_TYPES, SD_COMMON_TYPES);
  prepareSpecialUse({&target}, {this});

  manager.synchronize();
}

////////////////////////////////////////////////////////////////////////
void* NDArray::specialBuffer() {
  if (_buffer->special() == nullptr) {
    syncToDevice();
    tickReadHost();
  }
  // FIXME: this should be fixed once CUDA backend added
  return static_cast<int8_t*>(_buffer->special()) + (_offset * sizeOfT());
}

////////////////////////////////////////////////////////////////////////
void const* NDArray::specialBuffer() const {
  if (_buffer->special() == nullptr) {
    syncToDevice();
    tickReadHost();
  }
  // FIXME: this should be fixed once CUDA backend added
  return static_cast<int8_t*>(_buffer->special()) + (_offset * sizeOfT());
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray::printCurrentBuffer(const bool host, const char* msg, const int precision) const {
  if (!isScalar() && _length == 0) {
    printf("NDArray::printActualBuffer: array length is zero !\n");
    return;
  }

  if(isScalar()) {
    if(host) {

      if (msg) printf("%s", msg);

      if (buffer() == nullptr ) {
        printf("NDArray::printActualBuffer: host buffer is nullptr !\n");
        return;
      }

      const T* buff = bufferAsT<T>();
      if (msg) printf("%s", msg);
      printf("%.*f\n", precision, (double)buff[getOffset(0)]);
      return;
    } else {
      if (msg) printf("%s", msg);

      if (specialBuffer() == nullptr) {
        printf("NDArray::printSpecialBuffer: special buffer is nullptr !\n");
        return;
      }



      const auto sizeOfBuffer = sizeOfT();

      void* pHost = operator new(sizeOfBuffer);

      cudaMemcpyAsync(pHost, specialBuffer(), sizeOfBuffer, cudaMemcpyDeviceToHost, *getContext()->getCudaStream());

      cudaError_t cudaResult = cudaStreamSynchronize(*getContext()->getCudaStream());
      auto cast = reinterpret_cast<T*>(pHost);
      if (cudaResult != 0) THROW_EXCEPTION("NDArray::printSpecialBuffer: cudaStreamSynchronize failed!");
      printf("%.*f\n", precision, (double)cast[0]);

      return;
    }

  }



  if (msg) printf("%s", msg);

  if (host) {
    if (buffer() == nullptr || _length == 0) {
      printf("NDArray::printActualBuffer: host buffer is nullptr !\n");
      return;
    }

    const T* buff = bufferAsT<T>();
    for (sd::LongType i = 0; i < _length; i++) printf("%.*f, ", precision, (double)buff[getOffset(i)]);
    printf("\n");
  } else {
    if (specialBuffer() == nullptr) {
      printf("NDArray::printSpecialBuffer: special buffer is nullptr !\n");
      return;
    }

    const auto sizeOfBuffer = sizeOfT() * (getOffset(_length - 1) + 1);

    void* pHost = operator new(sizeOfBuffer);

    cudaMemcpyAsync(pHost, specialBuffer(), sizeOfBuffer, cudaMemcpyDeviceToHost, *getContext()->getCudaStream());

    cudaError_t cudaResult = cudaStreamSynchronize(*getContext()->getCudaStream());
    if (cudaResult != 0) THROW_EXCEPTION("NDArray::printSpecialBuffer: cudaStreamSynchronize failed!");

    for (sd::LongType i = 0; i < _length; i++)
      printf("%.*f, ", precision, (double)reinterpret_cast<T*>(pHost)[getOffset(i)]);
    printf("\n");

    operator delete(pHost);
  }
}
template void NDArray::printCurrentBuffer<int>(const bool host, const char* msg, const int precision) const;
template void NDArray::printCurrentBuffer<float>(const bool host, const char* msg, const int precision) const;
template void NDArray::printCurrentBuffer<double>(const bool host, const char* msg, const int precision) const;



}  // end namespace sd
#endif
