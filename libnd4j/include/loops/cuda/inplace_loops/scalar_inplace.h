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

#ifndef DEV_TESTS_SCALAR_INPLACE_H
#define DEV_TESTS_SCALAR_INPLACE_H
#include <helpers/shape.h>
#include <ops.h>
#include <system/op_boilerplate.h>
#include <types/types.h>

using namespace simdOps;

namespace functions {
namespace scalar {
template <typename X, typename Y, typename Z>
class ScalarInplace {
 public:
  static SD_INLINE SD_DEVICE void transformCudaLegacy(int opNum, void *vscalar, void *vy, sd::LongType *yShapeInfo,
                                                      void *vparams, void *vz, sd::LongType *zShapeInfo,
                                                      int *allocationBuffer);

  template <typename OpClass>
  static SD_INLINE SD_DEVICE void transformCuda(void *vscalar, void *vy, sd::LongType *yShapeInfo, void *vparams,
                                                void *vz, sd::LongType *zShapeInfo, int *allocationBuffer);
};

template <typename X, typename Y, typename Z>
SD_INLINE SD_DEVICE void ScalarInplace<X, Y, Z>::transformCudaLegacy(int opNum, void *vscalar, void *vy,
                                                                     sd::LongType *yShapeInfo, void *vparams, void *vz,
                                                                     sd::LongType *zShapeInfo, int *allocationBuffer) {
  DISPATCH_BY_OPNUM_TTT(transformCuda, PARAMS(vscalar, vy, yShapeInfo, vparams, vz, zShapeInfo, allocationBuffer),
                        SCALAR_OPS);
}

template <typename X, typename Y, typename Z>
template <typename OpType>
SD_INLINE SD_DEVICE void ScalarInplace<X, Y, Z>::transformCuda(void *vscalar, void *vy, sd::LongType *yShapeInfo,
                                                               void *vparams, void *vz, sd::LongType *zShapeInfo,
                                                               int *allocationBuffer) {
  auto scalar = reinterpret_cast<X *>(vscalar)[0];
  auto y = reinterpret_cast<Y *>(vy);
  auto params = reinterpret_cast<Z *>(vparams);
  auto z = reinterpret_cast<Z *>(vz);

  int totalThreads = gridDim.x * blockDim.x;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ sd::LongType length;
  if (threadIdx.x == 0) length = shape::length(yShapeInfo);
  __syncthreads();

  for (sd::LongType i = tid; i < length; i += totalThreads) {
    sd::LongType yCoords[SD_MAX_RANK];
    sd::LongType zCoords[SD_MAX_RANK];
    sd::LongType yOffset;
    sd::LongType zOffset;

    INDEX2COORDS(i, shape::rank(yShapeInfo), yShapeInfo, yCoords);
    COORDS2INDEX(shape::rank(yShapeInfo), shape::shapeOf(yShapeInfo), yCoords, yOffset);
    INDEX2COORDS(i, shape::rank(zShapeInfo), zShapeInfo, zCoords);
    COORDS2INDEX(shape::rank(zShapeInfo), shape::shapeOf(zShapeInfo), zCoords, zOffset);

    z[zOffset] = OpType::op(y[yOffset], scalar, params);
  }
}
}  // namespace scalar
}  // namespace functions

#endif  // DEV_TESTS_SCALAR_INPLACE_H
