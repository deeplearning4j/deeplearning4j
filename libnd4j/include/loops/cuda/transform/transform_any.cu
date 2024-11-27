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

//
// @author raver119@gmail.com
//
#include <helpers/DebugHelper.h>
#include <loops/legacy_ops.h>
#include <loops/transform_any.h>
#include <system/Environment.h>
#include <system/op_boilerplate.h>
#include <types/types.h>
#include <execution/cuda/DeviceValidator.h>


using namespace simdOps;



template <typename X, typename Z, typename OpType>
SD_KERNEL void transformAnySimple(const void *x, const sd::LongType *xShapeInfo,sd::LongType xRank, void *params, void *z,
                                  const sd::LongType *zShapeInfo, sd::LongType zRank,
                                  sd::LongType *allocationPointer,
                                  void *reductionPointer, const sd::LongType *tadShapeInfo,
                                  const sd::LongType *tadOffsets) {
  functions::transform::TransformAny<X, Z>::template transformCuda<OpType>(
      x, xShapeInfo, params, z, zShapeInfo, allocationPointer, reductionPointer, tadShapeInfo, tadOffsets);
}

namespace functions {
namespace transform {

template <typename X, typename Y>
SD_HOST void TransformAny<X, Y>::executeTransformShaped(dim3 launchDims, cudaStream_t *stream, const int opNum,
                                                        const void *x, const sd::LongType *xShape, sd::LongType xRank,
                                                        void *extraParams, void *z, const sd::LongType *zShape,
                                                        sd::LongType zRank, sd::LongType *allocationPointer, void *reductionPointer,
                                                        const sd::LongType *tadShapeInfo,
                                                        const sd::LongType *tadOffsets) {


  DISPATCH_BY_OPNUM_TT(intermediateShaped,
                       PARAMS(launchDims, stream, x, xShape, xRank, extraParams, z, zShape, zRank, allocationPointer,
                              reductionPointer, tadShapeInfo, tadOffsets),
                       TRANSFORM_ANY_OPS);

  sd::DebugHelper::checkErrorCode(stream, "transformAny executeTransformShaped(...) failed");
}

template <typename X, typename Z>
template <typename OpType>
SD_DEVICE void TransformAny<X, Z>::transformCuda(const void *vx, const sd::LongType *xShapeInfo, void *vparams,
                                                 void *vz, const sd::LongType *zShapeInfo,
                                                 sd::LongType *allocationPointer,
                                                 void *vreductionPointer, const sd::LongType *tadShapeInfo,
                                                 const sd::LongType *tadOffsets) {
  auto x = reinterpret_cast<const X *>(vx);
  auto z = reinterpret_cast<Z *>(vz);
  auto params = reinterpret_cast<X *>(vparams);

  if(x == nullptr || z == nullptr) {
    return;
  }
  __shared__ sd::LongType length;

  if (threadIdx.x == 0) {
    length = shape::length(xShapeInfo);
  }
  __syncthreads();

  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  int totalThreads = gridDim.x * blockDim.x;

  for (sd::LongType i = tid; i < length; i += totalThreads) {
    sd::LongType xCoords[SD_MAX_RANK];
    sd::LongType zCoords[SD_MAX_RANK];
    sd::LongType xOffset;
    sd::LongType zOffset;

    INDEX2COORDS(i, shape::rank(xShapeInfo), xShapeInfo, xCoords);
    COORDS2INDEX(shape::rank(xShapeInfo), shape::shapeOf(xShapeInfo), xCoords, xOffset);
    INDEX2COORDS(i, shape::rank(zShapeInfo), zShapeInfo, zCoords);
    COORDS2INDEX(shape::rank(zShapeInfo), shape::shapeOf(zShapeInfo), zCoords, zOffset);

    z[zOffset] = OpType::op(x[xOffset], params);
  }
};
template <typename X, typename Z>
template <typename OpType>
SD_HOST void TransformAny<X, Z>::intermediateShaped(dim3 launchDims, cudaStream_t *stream, const void *x,
                                                    const sd::LongType *xShape, sd::LongType xRank, void *extraParams, void *z,
                                                    const sd::LongType *zShape, sd::LongType zRank,
                                                    sd::LongType *allocationPointer,
                                                    void *reductionPointer, const sd::LongType *tadShapeInfo,
                                                    const sd::LongType *tadOffsets) {

  if(stream == nullptr)
    THROW_EXCEPTION("Found null stream when executing transformAny");


  transformAnySimple<X, Z, OpType><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
      x, xShape, xRank, extraParams, z, zShape, zRank, allocationPointer, reductionPointer, tadShapeInfo, tadOffsets);

  sd::DebugHelper::checkErrorCode(stream, "transformAny(...) failed");
}

BUILD_DOUBLE_TEMPLATE(template class TransformAny, , SD_COMMON_TYPES, SD_COMMON_TYPES);
}  // namespace transform
}  // namespace functions
