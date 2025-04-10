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

#include "execution/cuda/LaunchDims.h"


namespace sd {



void* NDArray::platformBuffer() { return specialBuffer(); }



void NDArray::syncToDevice()  {
  auto currentDeviceId = AffinityManager::currentDeviceId();
  if (currentDeviceId != _deviceId) {
    // first of all we update shapeInfo
    const_cast<NDArray*>(this)->setShapeInfo(this->shapeInfo());

    // now we actually migrate data buffer
    _buffer->migrate();
  }

  _buffer->syncToSpecial();
}

void NDArray::syncToHost()  { if(!isEmpty()) _buffer->syncToPrimary(getContext()); }
void NDArray::tickWriteHost()  { if(!isEmpty()) _buffer->writePrimary(); }
void NDArray::tickWriteDevice()  { if(!isEmpty()) _buffer->writeSpecial(); }
void NDArray::tickReadHost()  { if(!isEmpty()) _buffer->readPrimary(); }
void NDArray::tickReadDevice()  { if(!isEmpty()) _buffer->readSpecial(); }
void NDArray::tickBothActual()  {
  _buffer->writePrimary();
  _buffer->readSpecial();
}
bool NDArray::isActualOnHostSide()  { return _buffer->isPrimaryActual(); }
bool NDArray::isActualOnDeviceSide()  { return _buffer->isSpecialActual(); }
void NDArray::makeBothBuffersActual()  {
  if (!isActualOnHostSide()) syncToHost();
  if (!isActualOnDeviceSide()) syncToDevice();
}

///////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL static void fillAsTriangularCuda(const void* vx, const LongType* xShapeInfo, void* vz,
                                           const LongType* zShapeInfo, const T val, const int lower,
                                           const int upper, char direction, bool includeEdges) {
  const auto x = reinterpret_cast<const T*>(vx);
  auto z = reinterpret_cast<T*>(vz);

  __shared__ LongType zRank, xRank, areSameOffsets, *sharedMem;  // xRank == zRank always, except when xRank = 1, in this case zRank = 2
  __shared__ LongType zLen, totalThreads;  // xLen == zLen, except when xRank = 1, in this case zLen = 2*xLen
  __shared__ LongType *zShape;
  __shared__ LongType *zStride;
  __shared__ LongType *xShape;
  __shared__ LongType *xStride;
  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    sharedMem = reinterpret_cast<LongType*>(shmem);
    areSameOffsets = shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo);
    xRank = shape::rank(xShapeInfo);
    zRank = shape::rank(zShapeInfo);
    zLen = shape::length(zShapeInfo);
    totalThreads = gridDim.x * blockDim.x;
    zShape = shape::shapeOf(zShapeInfo);
    zStride = shape::stride(zShapeInfo);
    xShape = shape::shapeOf(xShapeInfo);
    xStride = shape::stride(xShapeInfo);
  }
  __syncthreads();

  auto coords = sharedMem + threadIdx.x * zRank;

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  bool dirU = direction == 'u';
  bool dirL = direction == 'l';
  for (LongType i = tid; i < zLen; i += totalThreads) {
    INDEX2COORDS(i, zRank, zShape, coords);

    LongType zOffset;
    COORDS2INDEX(zRank, zStride, coords, zOffset);

    auto row = coords[zRank - 2];
    auto col = coords[zRank - 1];
    auto lCompare = includeEdges ? row + lower <= col : row + lower < col;
    auto uCompare = includeEdges ? row + upper >= col : row + upper > col;
    if (dirU && lCompare || dirL && uCompare) {
      z[zOffset] = val;
    } else if (vx != vz) {  // when x and z are different arrays
      if (xRank != zRank) coords[0] = coords[1];
      LongType xOffset;
      COORDS2INDEX(xRank, xStride, coords, xOffset);
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
  int len = target.isScalar() ? 1 : target.lengthOf();
  const int blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
  const int sharedMem = threadsPerBlock * sizeof(int) * target.rankOf() + 128;
  dim3 launchDims = getFillTriLaunchDims(target.lengthOf(), target.rankOf());
  PointersManager manager(getContext(), "NDArray::fillAsTriangular");

  prepareSpecialUse({&target}, {this});
  fillAsTriangularCuda<T><<<launchDims.y, launchDims.x, launchDims.z, *getContext()->getCudaStream()>>>(
      platformBuffer(), specialShapeInfo(), target.platformBuffer(), target.specialShapeInfo(), static_cast<T>(val),
      lower, upper, direction, includeEdges);
  registerSpecialUse({&target}, {this});
  sd::DebugHelper::checkGlobalErrorCode("fillTriangular  failed");

  manager.synchronize();
}
BUILD_SINGLE_TEMPLATE(template SD_LIB_EXPORT void NDArray::fillAsTriangular,
                      (const float val, int lower, int upper, NDArray& target, const char direction,
                          const bool includeEdges),
                      SD_COMMON_TYPES);

////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL static void identityMatrixCuda(void* vx, const LongType* xShapeInfo, const T val) {
  auto x = reinterpret_cast<T*>(vx);

  // Shared memory variables
  __shared__ LongType rank;
  __shared__ LongType len;
  __shared__ LongType totalThreads;
  __shared__ const LongType* shapePtr;
  __shared__ const LongType* stridePtr;
  __shared__ LongType* sharedMem;

  // Initialize shared variables in thread 0
  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    sharedMem = reinterpret_cast<LongType*>(shmem);

    // Cache rank and length
    rank = shape::rank(xShapeInfo);
    len = shape::length(xShapeInfo);

    // Cache pointers to shape and stride arrays
    shapePtr = shape::shapeOf(xShapeInfo);
    stridePtr = shape::stride(xShapeInfo);

    // Calculate total number of threads
    totalThreads = gridDim.x * blockDim.x;
  }
  __syncthreads();

  // Each thread has its own coordinates array in shared memory
  auto coords = sharedMem + threadIdx.x * rank;

  // Calculate global thread ID
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Iterate over assigned elements
  for (LongType i = tid; i < len; i += totalThreads) {
    // Convert linear index to multi-dimensional coordinates using cached shape
    INDEX2COORDS(i, rank, shapePtr, coords);

    // Compute linear offset from coordinates using cached stride
    LongType offset;
    COORDS2INDEX(rank, stridePtr, coords, offset);

    // Check if the current position is on the diagonal (row == col)
    if (coords[rank - 2] == coords[rank - 1]) {  // Assuming 0-based indexing
      x[offset] = val;
    }
    else {
      x[offset] = static_cast<T>(0);
    }
  }
}

///////////////////////////////////////////////////////////////////
template <typename T>
static void identityMatrixCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                       const cudaStream_t* stream, void* vx, const LongType* xShapeInfo,
                                       const float val) {
  identityMatrixCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, static_cast<T>(val));
  sd::DebugHelper::checkGlobalErrorCode("identityMatrix  failed");

}
BUILD_SINGLE_TEMPLATE(template void identityMatrixCudaLauncher,
                      (const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                          const cudaStream_t* stream, void* vx, const sd::LongType* xShapeInfo, const float val),
                      SD_COMMON_TYPES);

////////////////////////////////////////////////////////////////////////
void NDArray::setIdentity() {
  if (isS()) THROW_EXCEPTION("NDArray::setIdentity: you can't use this method on String array!");

  int len = isScalar() ? 1 : lengthOf();
  dim3 launchDims = getIdentityLaunchDims(len, rankOf());

  PointersManager manager(getContext(), "NDArray::setIdentity");

  syncToDevice();
  BUILD_SINGLE_SELECTOR(dataType(), identityMatrixCudaLauncher,
                        (launchDims.y, launchDims.x,launchDims.z, getContext()->getCudaStream(), platformBuffer(),
                            specialShapeInfo(), 1.f),
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
void NDArray::synchronize(const char* msg)  {
  auto res = cudaStreamSynchronize(*(getContext()->getCudaStream()));
  if (res != 0) {
    std::string message = msg + std::string(": synchronization failed !");
    THROW_EXCEPTION(message.c_str());
  }
}


// NDArray implementation for .cu file
void NDArray::printBufferDebug(const char* msg, sd::LongType offset, sd::LongType limit) {
  if (msg) sd_printf("%s:\n", msg);

  if(limit < 0) limit = lengthOf();

  // Print array info
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
  // Print buffer state
  if (_buffer != nullptr) {
    _buffer->printBufferDebug("Buffer contents", offset, limit);
  } else {
    sd_printf("Buffer is nullptr\n", 0);
  }
}

////////////////////////////////////////////////////////////////////////
void NDArray::prepareSpecialUse(const std::vector<NDArray*>& writeList,
                                const std::vector<NDArray*>& readList, bool synchronizeWritables) {



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
void NDArray::registerSpecialUse(const std::vector<NDArray*>& writeList,
                                 const std::vector<NDArray*>& readList) {


  for (const auto& p : readList)
    if (p != nullptr) p->tickReadDevice();

  for (const auto& p : writeList)
    if (p != nullptr) p->tickWriteDevice();


}

////////////////////////////////////////////////////////////////////////
void NDArray::preparePrimaryUse(const std::vector<NDArray*>& writeList,
                                const std::vector<NDArray*>& readList, bool synchronizeWritables) {
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
void NDArray::registerPrimaryUse(const std::vector<NDArray*>& writeList,
                                 const std::vector<NDArray*>& readList) {
  for (const auto& p : readList)
    if (p != nullptr) p->tickReadHost();

  for (const auto& p : writeList)
    if (p != nullptr) p->tickWriteHost();

}

//////////////////////////////////////////////////////////////////////////
void NDArray::syncShape()  {
  cudaMemcpy(const_cast<LongType*>(specialShapeInfo()), shapeInfo(), shape::shapeInfoByteLength(shapeInfo()),
             cudaMemcpyHostToDevice);
}

//////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////
// change an array by repeating it the number of times given by reps.
NDArray NDArray::tile(const std::vector<LongType>& reps)  {
  int dim = reps.size();
  LongType product = 1;
  for (const auto& item : reps) product *= item;

  if (product < 1) THROW_EXCEPTION("NDArray::tile method: one of the elements in reps array is zero !");

  int rankOld = rankOf();
  int diff = rankOld - dim;
  if (product == 1) {  // in this case 2 possibilities are present: just reshape or nothing to do
    NDArray result(*this);
    if (diff < 0) {                               // reshape to higher dimension
      std::vector<LongType> shapeNew = reps;  // need to have unities at first "diff" positions of new shape
      memcpy(&shapeNew[-diff], result.shapeInfo() + 1,
             rankOld * sizeof(LongType));  // put old shape numbers at rest of positions
      result.reshapei(ordering(), shapeNew);
    }
    return result;  // nothing to do, if diff >= 0 -> identity tile
  }

  // evaluate shapeInfo for resulting array
  auto newShapeInfo = ShapeUtils::evalTileShapeInfo(*this, reps, getContext()->getWorkspace());
  // create new buffer, in any case the memory amount new buffer points to is bigger then those for old _buffer
  DataBuffer *  newBuff = new DataBuffer(shape::length(newShapeInfo) * sizeOfT(),
                                         dataType(), getContext()->getWorkspace(), true);
  // assign new shape and new buffer to resulting array
  NDArray result(newBuff,const_cast<sd::LongType *>(newShapeInfo) , getContext());
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
void NDArray::tile(const std::vector<LongType>& reps, NDArray& target)  {
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
      (specialBuffer(), specialShapeInfo(), target.specialBuffer(), target.specialShapeInfo(), targetLen, stream),
      SD_COMMON_TYPES);
  registerSpecialUse({&target}, {this});
}

//////////////////////////////////////////////////////////////////////////
void NDArray::tile(NDArray& target)  {
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
      (specialBuffer(), specialShapeInfo(), target.specialBuffer(), target.specialShapeInfo(), targetLen, stream),
      SD_COMMON_TYPES);
  registerSpecialUse({&target}, {this});
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
SD_KERNEL static void repeatCuda(const void* vx, const LongType* xShapeInfo, void* vz,
                                 const LongType* zShapeInfo, const LongType* repeats, const LongType repSize,
                                 const int axis) {
  const X* x = reinterpret_cast<const X*>(vx);
  Z* z = reinterpret_cast<Z*>(vz);

  __shared__ LongType rank, *sharedMem;
  __shared__ LongType zLen, totalThreads;  // xLen = zLen

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    sharedMem = reinterpret_cast<LongType*>(shmem);

    rank = shape::rank(zShapeInfo);    // xRank = zRank
    zLen = shape::length(zShapeInfo);  // xLen <= zLen

    totalThreads = gridDim.x * blockDim.x;
  }

  __syncthreads();

  auto coords = sharedMem + threadIdx.x * rank;

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (LongType i = tid; i < zLen; i += totalThreads) {
    INDEX2COORDS(i, rank, shape::shapeOf(zShapeInfo), coords);

    LongType zOffset;
    COORDS2INDEX(rank, shape::stride(zShapeInfo), coords, zOffset);

    if (repSize > 1) {
      for (LongType j = 0; j < repSize; ++j) {
        coords[axis] -= repeats[j];
        if (coords[axis] < 0) {
          coords[axis] = j;
          break;
        }
      }
    } else
      coords[axis] /= repeats[0];

    LongType xOffset;
    COORDS2INDEX(rank, shape::stride(xShapeInfo), coords, xOffset);

    z[zOffset] = x[xOffset];
  }
}
//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
static void repeatCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                               const cudaStream_t* stream, const void* vx, const LongType* xShapeInfo, void* vz,
                               const LongType* zShapeInfo, const LongType* repeats, const LongType repSize, const LongType axis) {
  repeatCuda<X, Z>
  <<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vz, zShapeInfo, repeats, repSize, axis);
  DebugHelper::checkGlobalErrorCode("NDArray repeat cuda failed(...) failed");

}
BUILD_DOUBLE_TEMPLATE(template void repeatCudaLauncher,
                      (const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                          const cudaStream_t* stream, const void* vx, const sd::LongType* xShapeInfo, void* vz,
                          const sd::LongType* zShapeInfo, const sd::LongType* repeats, const sd::LongType repSize, const sd::LongType axis),
                      SD_COMMON_TYPES, SD_COMMON_TYPES);

//////////////////////////////////////////////////////////////////////////
// create new array by repeating it the number of times given by repeats
NDArray NDArray::repeat(const int axis, const std::vector<LongType>& repeats)  {
  auto nonConst = const_cast<NDArray *>(this);
  std::vector<sd::LongType> shape = ShapeUtils::evalRepeatShape(axis, repeats, *nonConst);
  NDArray output('c',shape, dataType(), getContext());
  dim3 launchDims = getRepeatLaunchDims(output.lengthOf(), output.rankOf());

  PointersManager manager(getContext(), "NDArray::repeat(const int axis, const std::vector<int>& repeats)");

  const LongType* reps = reinterpret_cast<LongType*>(manager.replicatePointer(repeats.data(), repeats.size() * sizeof(LongType)));

  prepareSpecialUse({&output}, {this});
  BUILD_SINGLE_SELECTOR_TWICE(
      dataType(), repeatCudaLauncher,
      (launchDims.y, launchDims.x, launchDims.z, getContext()->getCudaStream(), specialBuffer(), specialShapeInfo(),
          output.specialBuffer(), output.specialShapeInfo(), reps, repeats.size(), axis),
      SD_COMMON_TYPES);
  prepareSpecialUse({&output}, {this});

  manager.synchronize();

  return output;
}

//////////////////////////////////////////////////////////////////////////
// fill array by repeating it the number of times given by repeats
void NDArray::repeat(const int axis, const std::vector<LongType>& repeats, NDArray& target)  {
  auto nonConst = const_cast<NDArray *>(this);
  std::vector<sd::LongType> shape = ShapeUtils::evalRepeatShape(axis, repeats, *nonConst);

  if (!target.isSameShape(shape))
    THROW_EXCEPTION(
        "NDArray::repeat(const int axis, const std::vector<int>& repeats, NDArray& target) method: wrong shape of "
        "target array!");

  dim3 launchDims = getRepeatLaunchDims(target.lengthOf(), target.rankOf());

  PointersManager manager(getContext(), "NDArray::repeat(const int axis, const std::vector<int>& repeats)");

  const LongType* reps = reinterpret_cast<LongType*>(manager.replicatePointer(repeats.data(), repeats.size() * sizeof(LongType)));

  prepareSpecialUse({&target}, {this});
  BUILD_DOUBLE_SELECTOR(
      dataType(), target.dataType(), repeatCudaLauncher,
      (launchDims.y, launchDims.x, launchDims.z, getContext()->getCudaStream(), specialBuffer(), specialShapeInfo(),
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
  return static_cast<int8_t*>(_buffer->special()) + (offset() * sizeOfT());
}



//////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray::printCurrentBuffer(const bool host, const char* msg, const int precision)  {
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
      cudaDeviceSynchronize();
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
    for (LongType i = 0; i < _length; i++) printf("%.*f, ", precision, (double)buff[getOffset(i)]);
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

    for (LongType i = 0; i < _length; i++)
      printf("%.*f, ", precision, (double)reinterpret_cast<T*>(pHost)[getOffset(i)]);
    printf("\n");

    operator delete(pHost);
  }
}
#define PRINT_BUFFER(T) template void NDArray::printCurrentBuffer<GET_SECOND(T)>(const bool host, const char* msg, const int precision);
ITERATE_LIST((SD_COMMON_TYPES),PRINT_BUFFER)


}  // end namespace sd
#endif
