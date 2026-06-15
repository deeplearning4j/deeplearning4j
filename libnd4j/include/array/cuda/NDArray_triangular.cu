/* ******************************************************************************
 * NDArray_triangular.cu - Triangular and identity matrix operations
 * Split from NDArray.cu to reduce object file size for large binary builds
 ******************************************************************************/

#include <array/NDArray.h>
#include <helpers/PointersManager.h>
#include <helpers/DebugHelper.h>
#include "execution/cuda/LaunchDims.h"

namespace sd {

///////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL static void fillAsTriangularCuda(const void* vx, const LongType* xShapeInfo, void* vz,
                                           const LongType* zShapeInfo, const T val, const int lower,
                                           const int upper, char direction, bool includeEdges) {
  const auto x = reinterpret_cast<const T*>(vx);
  auto z = reinterpret_cast<T*>(vz);

  __shared__ LongType zRank, xRank, areSameOffsets, *sharedMem;
  __shared__ LongType zLen, totalThreads;
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
    } else if (vx != vz) {
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
    THROW_EXCEPTION("NDArray::fillAsTriangular method: wrong shape of target array !");

  dim3 launchDims = getFillTriLaunchDims(target.lengthOf(), target.rankOf());

  prepareSpecialUse({&target}, {this});
  fillAsTriangularCuda<T><<<launchDims.y, launchDims.x, launchDims.z, *getContext()->getCudaStream()>>>(
      platformBuffer(), specialShapeInfo(), target.platformBuffer(), target.specialShapeInfo(), static_cast<T>(val),
      lower, upper, direction, includeEdges);
  registerSpecialUse({&target}, {this});
  sd::DebugHelper::checkGlobalErrorCode("fillTriangular failed");
  // Don't sync - let CUDA operations run asynchronously
}

BUILD_SINGLE_TEMPLATE(SD_LIB_EXPORT void NDArray::fillAsTriangular,
                      (const float val, int lower, int upper, NDArray& target, const char direction,
                          const bool includeEdges),
                      SD_COMMON_TYPES);

////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL static void identityMatrixCuda(void* vx, const LongType* xShapeInfo, const T val) {
  auto x = reinterpret_cast<T*>(vx);

  __shared__ LongType rank;
  __shared__ LongType len;
  __shared__ LongType totalThreads;
  __shared__ const LongType* shapePtr;
  __shared__ const LongType* stridePtr;
  __shared__ LongType* sharedMem;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    sharedMem = reinterpret_cast<LongType*>(shmem);
    rank = shape::rank(xShapeInfo);
    len = shape::length(xShapeInfo);
    shapePtr = shape::shapeOf(xShapeInfo);
    stridePtr = shape::stride(xShapeInfo);
    totalThreads = gridDim.x * blockDim.x;
  }
  __syncthreads();

  auto coords = sharedMem + threadIdx.x * rank;
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (LongType i = tid; i < len; i += totalThreads) {
    INDEX2COORDS(i, rank, shapePtr, coords);

    LongType offset;
    COORDS2INDEX(rank, stridePtr, coords, offset);

    if (coords[rank - 2] == coords[rank - 1]) {
      x[offset] = val;
    } else {
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
  sd::DebugHelper::checkGlobalErrorCode("identityMatrix failed");
}

BUILD_SINGLE_TEMPLATE(void identityMatrixCudaLauncher,
                      (const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                          const cudaStream_t* stream, void* vx, const sd::LongType* xShapeInfo, const float val),
                      SD_COMMON_TYPES);

////////////////////////////////////////////////////////////////////////
void NDArray::setIdentity() {
  if (isS()) THROW_EXCEPTION("NDArray::setIdentity: you can't use this method on String array!");

  int len = isScalar() ? 1 : lengthOf();
  dim3 launchDims = getIdentityLaunchDims(len, rankOf());

  syncToDevice();
  BUILD_SINGLE_SELECTOR(dataType(), identityMatrixCudaLauncher,
                        (launchDims.y, launchDims.x, launchDims.z, getContext()->getCudaStream(), platformBuffer(),
                            specialShapeInfo(), 1.f),
                        SD_COMMON_TYPES);
  tickWriteDevice();
  // Don't sync - let CUDA operations run asynchronously
}

}  // namespace sd
