/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See
 * the License for the specific language governing permissions and limitations
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
  static SD_INLINE SD_DEVICE void transformCudaLegacy(
      int opNum,
      void* dy,
      sd::LongType* shapeInfo,
      void* params,
      void* result,
      sd::LongType* zShapeInfo,
      int* allocationPointer,
      void* reductionPointer,
      sd::LongType* tadShapeInfo,
      sd::LongType* tadOffsets);

  template <typename OpClass>
  static SD_INLINE SD_DEVICE void transformCuda(
      void* vdy,
      sd::LongType* shapeInfo,
      void* vparams,
      void* vresult,
      sd::LongType* zShapeInfo,
      int* allocationPointer,
      void* vreductionPointer,
      sd::LongType* tadShapeInfo,
      sd::LongType* tadOffsets);
};

template <typename X>
template <typename OpClass>
SD_INLINE SD_DEVICE void TransformStrictInplace<X>::transformCuda(
    void* vdy,
    sd::LongType* shapeInfo,
    void* vparams,
    void* vresult,
    sd::LongType* zShapeInfo,
    int* allocationPointer,
    void* vreductionPointer,
    sd::LongType* tadShapeInfo,
    sd::LongType* tadOffsets) {

  auto dy    = static_cast<X*>(vdy);
  auto result= static_cast<X*>(vresult);
  auto params= static_cast<X*>(vparams);
  // reductionPointer is unused in this basic transform?

  const int tid          = blockIdx.x * blockDim.x + threadIdx.x;
  const int totalThreads = gridDim.x * blockDim.x;

  // Cache shape info in shared memory to avoid repeated calls
  __shared__ sd::LongType length;
  __shared__ int rank;
  __shared__ const sd::LongType* shapePtr;
  __shared__ const sd::LongType* stridePtr;

  __shared__ int zRank;
  __shared__ const sd::LongType* zShapePtr;
  __shared__ const sd::LongType* zStridePtr;

  if (threadIdx.x == 0) {
    length    = shape::length(shapeInfo);
    rank      = shape::rank(shapeInfo);
    shapePtr  = shape::shapeOf(shapeInfo);
    stridePtr = shape::stride(shapeInfo);

    zRank      = shape::rank(zShapeInfo);
    zShapePtr  = shape::shapeOf(zShapeInfo);
    zStridePtr = shape::stride(zShapeInfo);
  }
  __syncthreads();

  for (sd::LongType i = tid; i < length; i += totalThreads) {
    sd::LongType inCoords[SD_MAX_RANK];
    sd::LongType outCoords[SD_MAX_RANK];
    sd::LongType inOffset;
    sd::LongType outOffset;

    INDEX2COORDS(i, rank, shapePtr, inCoords);
    COORDS2INDEX(rank, stridePtr, inCoords, inOffset);

    INDEX2COORDS(i, zRank, zShapePtr, outCoords);
    COORDS2INDEX(zRank, zStridePtr, outCoords, outOffset);

    // Apply transform op in place
    result[outOffset] = OpClass::op(dy[inOffset], params);
  }
}

template <typename X>
SD_INLINE SD_DEVICE void TransformStrictInplace<X>::transformCudaLegacy(
    int opNum,
    void* dy,
    sd::LongType* shapeInfo,
    void* params,
    void* result,
    sd::LongType* zShapeInfo,
    int* allocationPointer,
    void* reductionPointer,
    sd::LongType* tadShapeInfo,
    sd::LongType* tadOffsets) {

  DISPATCH_BY_OPNUM_T(
      transformCuda,
      PARAMS(dy, shapeInfo, params, result, zShapeInfo, allocationPointer, reductionPointer, tadShapeInfo, tadOffsets),
      LOCAL_TRANSFORM_STRICT_OPS);
}

}  // namespace transform
}  // namespace functions

#undef LOCAL_TRANSFORM_STRICT_OPS
#endif  // DEV_TESTS_TRANSFORM_FLOAT_INPLACE_H
