/* ******************************************************************************
 * NDArray_repeat.cu - Repeat operations (BUILD_DOUBLE_TEMPLATE = 169 instantiations)
 * Split from NDArray.cu to reduce object file size for large binary builds
 ******************************************************************************/

#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>
#include <helpers/DebugHelper.h>
#include "execution/cuda/LaunchDims.h"

namespace sd {

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
SD_KERNEL static void repeatCuda(const void* vx, const LongType* xShapeInfo, void* vz,
                                 const LongType* zShapeInfo, const LongType* repeats, const LongType repSize,
                                 const int axis) {
  const X* x = reinterpret_cast<const X*>(vx);
  Z* z = reinterpret_cast<Z*>(vz);

  __shared__ LongType rank, *sharedMem;
  __shared__ LongType zLen, totalThreads;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    sharedMem = reinterpret_cast<LongType*>(shmem);

    rank = shape::rank(zShapeInfo);
    zLen = shape::length(zShapeInfo);

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
BUILD_DOUBLE_TEMPLATE(void repeatCudaLauncher,
                      (const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                          const cudaStream_t* stream, const void* vx, const sd::LongType* xShapeInfo, void* vz,
                          const sd::LongType* zShapeInfo, const sd::LongType* repeats, const sd::LongType repSize, const sd::LongType axis),
                      SD_COMMON_TYPES, SD_COMMON_TYPES);

//////////////////////////////////////////////////////////////////////////
// create new array by repeating it the number of times given by repeats
NDArray NDArray::repeat(const int axis, const std::vector<LongType>& repeats) {
  auto nonConst = const_cast<NDArray *>(this);
  std::vector<sd::LongType> shape = ShapeUtils::evalRepeatShape(axis, repeats, *nonConst);
  NDArray output('c', shape, dataType(), getContext());
  dim3 launchDims = getRepeatLaunchDims(output.lengthOf(), output.rankOf());

  PointersManager manager(getContext(), "NDArray::repeat(const int axis, const std::vector<int>& repeats)");

  const LongType* reps = reinterpret_cast<LongType*>(manager.replicatePointer(repeats.data(), repeats.size() * sizeof(LongType)));

  prepareSpecialUse({&output}, {this});
  BUILD_SINGLE_SELECTOR_TWICE(
      dataType(), repeatCudaLauncher,
      (launchDims.y, launchDims.x, launchDims.z, getContext()->getCudaStream(), specialBuffer(), specialShapeInfo(),
          output.specialBuffer(), output.specialShapeInfo(), reps, repeats.size(), axis),
      SD_COMMON_TYPES);
  registerSpecialUse({&output}, {this});

  manager.synchronize();

  return output;
}

//////////////////////////////////////////////////////////////////////////
// fill array by repeating it the number of times given by repeats
void NDArray::repeat(const int axis, const std::vector<LongType>& repeats, NDArray& target) {
  auto nonConst = const_cast<NDArray *>(this);
  std::vector<sd::LongType> shape = ShapeUtils::evalRepeatShape(axis, repeats, *nonConst);

  if (!target.isSameShape(shape))
    THROW_EXCEPTION(
        "NDArray::repeat(const int axis, const std::vector<int>& repeats, NDArray& target) method: wrong shape of "
        "target array!");

  dim3 launchDims = getRepeatLaunchDims(target.lengthOf(), target.rankOf());

  PointersManager manager(getContext(), "NDArray::repeat(const int axis, const std::vector<int>& repeats)");

  const LongType* reps = reinterpret_cast<LongType*>(manager.replicatePointer(repeats.data(), repeats.size() * sizeof(LongType)));
  auto targetDataType = target.dataType();
  auto selfDType = dataType();
  prepareSpecialUse({&target}, {this});
  BUILD_DOUBLE_SELECTOR(
      dataType(), target.dataType(), repeatCudaLauncher,
      (launchDims.y, launchDims.x, launchDims.z, getContext()->getCudaStream(), specialBuffer(), specialShapeInfo(),
          target.specialBuffer(), target.specialShapeInfo(), reps, repeats.size(), axis),
      SD_COMMON_TYPES, SD_COMMON_TYPES);
  registerSpecialUse({&target}, {this});

  manager.synchronize();
}

}  // namespace sd
