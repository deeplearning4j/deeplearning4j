#include <exceptions/cuda_exception.h>
#include <execution/cuda/LaunchDims.h>
#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/assign.h>

#include "helpers/DebugHelper.h"
#include "helpers/ShapeUtils.h"

namespace sd {
namespace ops {
namespace helpers {

template <typename X, typename Z>
SD_KERNEL static void assignKernel(const void* vx, const LongType* xShapeInfo, void* vz, const LongType* zShapeInfo,
                                   const LongType xOffset, const LongType zOffset) {
  const auto x = reinterpret_cast<const X*>(vx);
  auto z = reinterpret_cast<Z*>(vz);

  __shared__ LongType len, totalThreads;
  __shared__ int rank;

  if (threadIdx.x == 0) {
    len = shape::length(zShapeInfo);
    totalThreads = gridDim.x * blockDim.x;
    rank = shape::rank(zShapeInfo);
  }
  __syncthreads();

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  LongType xCoords[SD_MAX_RANK], zCoords[SD_MAX_RANK];

  for (LongType i = tid; i < len; i += totalThreads) {
    INDEX2COORDS(i, rank, shape::shapeOf(zShapeInfo), zCoords);
    INDEX2COORDS(i, rank, shape::shapeOf(xShapeInfo), xCoords);

    LongType xIndex, zIndex;
    COORDS2INDEX(rank, shape::stride(xShapeInfo), xCoords, xIndex);
    COORDS2INDEX(rank, shape::stride(zShapeInfo), zCoords, zIndex);

    z[zIndex] = static_cast<Z>(x[xIndex]);
  }
}

template <typename X, typename Z>
SD_HOST static void assignCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                       const cudaStream_t* stream, const void* vx, const LongType* xShapeInfo,
                                       void* vz, const LongType* zShapeInfo, const LongType xOffset, const LongType zOffset) {
  assignKernel<X, Z><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vz, zShapeInfo, xOffset, zOffset);
  DebugHelper::checkGlobalErrorCode("assignKernel(...) failed");
}

void assign(sd::LaunchContext* context, sd::NDArray* target, sd::NDArray* source) {
  if (target->lengthOf() != source->lengthOf()) {
    std::string errorMsg = "assign helper: Source and target arrays must have the same length. ";
    errorMsg += "Source shape: " + ShapeUtils::shapeAsString(source) + ", ";
    errorMsg += "Target shape: " + ShapeUtils::shapeAsString(target) + ", ";
    errorMsg += "Source datatype: " + DataTypeUtils::asString(source->dataType()) + ", ";
    errorMsg += "Target datatype: " + DataTypeUtils::asString(target->dataType());
    THROW_EXCEPTION(errorMsg.c_str());
  }

  NDArray::prepareSpecialUse({target}, {source});

  auto xType = source->dataType();
  auto zType = target->dataType();

  dim3 launchDims = traceDims(target->lengthOf());

  PointersManager manager(context, "helpers::assign");

  BUILD_DOUBLE_SELECTOR(xType, zType, assignCudaLauncher,
                        (launchDims.x, launchDims.y, launchDims.z, context->getCudaStream(),
                         source->specialBuffer(), source->specialShapeInfo(),
                         target->specialBuffer(), target->specialShapeInfo(),
                         source->offset(), target->offset()),
                        SD_COMMON_TYPES, SD_COMMON_TYPES);

  manager.synchronize();
  NDArray::registerSpecialUse({target}, {source});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd