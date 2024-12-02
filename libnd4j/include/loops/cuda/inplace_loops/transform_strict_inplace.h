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
//  @author raver119@gmail.com
//

#ifndef DEV_TESTS_TRANSFORM_FLOAT_INPLACE_H
#define DEV_TESTS_TRANSFORM_FLOAT_INPLACE_H
#include <helpers/shape.h>
#include <ops.h>
#include <system/op_boilerplate.h>
#include <types/types.h>

using namespace simdOps;

#define LOCAL_TRANSFORM_STRICT_OPS (23, Exp), (24, Log)

namespace functions {
namespace transform {
template <typename X>
class TransformStrictInplace {
 public:
  static SD_INLINE SD_DEVICE void transformCudaLegacy(int opNum, void *dy, sd::LongType *shapeInfo, void *params,
                                                      void *result, sd::LongType *zShapeInfo, int *allocationPointer,
                                                      void *reductionPointer, sd::LongType *tadShapeInfo,
                                                      sd::LongType *tadOffsets);

  template <typename OpClass>
  static SD_INLINE SD_DEVICE void transformCuda(void *vdy, sd::LongType *shapeInfo, void *vparams, void *vresult,
                                                sd::LongType *zShapeInfo, int *allocationPointer,
                                                void *vreductionPointer, sd::LongType *tadShapeInfo,
                                                sd::LongType *tadOffsets);
};

template <typename X>
template <typename OpType>
SD_INLINE SD_DEVICE void TransformStrictInplace<X>::transformCuda(void *vdy, sd::LongType *shapeInfo, void *vparams,
                                                                  void *vresult, sd::LongType *zShapeInfo,
                                                                  int *allocationPointer, void *vreductionPointer,
                                                                  sd::LongType *tadShapeInfo,
                                                                  sd::LongType *tadOffsets) {
  auto dy = static_cast<X *>(vdy);
  auto result = static_cast<X *>(vresult);
  auto params = static_cast<X *>(vparams);
  auto reductionPointer = static_cast<X *>(vreductionPointer);

  auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ sd::LongType length;
  if (threadIdx.x == 0) length = shape::length(shapeInfo);
  __syncthreads();

  for (sd::LongType i = tid; i < length; i += gridDim.x * blockDim.x) {
    sd::LongType xCoords[SD_MAX_RANK];
    sd::LongType zCoords[SD_MAX_RANK];
    sd::LongType xOffset2;
    sd::LongType zOffset2;

    INDEX2COORDS(i, shape::rank(shapeInfo), shape::shapeOf(shapeInfo), xCoords);
    COORDS2INDEX(shape::rank(shapeInfo), shape::stride(shapeInfo), xCoords, xOffset2);
    INDEX2COORDS(i, shape::rank(zShapeInfo), shape::shapeOf(zShapeInfo), zCoords);
    COORDS2INDEX(shape::rank(zShapeInfo), shape::stride(zShapeInfo), zCoords, zOffset2);

    result[zOffset2] = OpType::op(dy[xOffset2], params);
  }
}

template <typename X>
SD_INLINE SD_DEVICE void TransformStrictInplace<X>::transformCudaLegacy(
    int opNum, void *dy, sd::LongType *shapeInfo, void *params, void *result, sd::LongType *zShapeInfo,
    int *allocationPointer, void *reductionPointer, sd::LongType *tadShapeInfo, sd::LongType *tadOffsets) {
  DISPATCH_BY_OPNUM_T(
      transformCuda,
      PARAMS(dy, shapeInfo, params, result, zShapeInfo, allocationPointer, reductionPointer, tadShapeInfo, tadOffsets),
      LOCAL_TRANSFORM_STRICT_OPS);
}
}  // namespace transform
}  // namespace functions

#undef LOCAL_TRANSFORM_STRICT_OPS
#endif  // DEV_TESTS_TRANSFORM_FLOAT_INPLACE_H
