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
// @author Yurii Shyrma, created on 15.11.2018
//
#include <loops/special_kernels.h>


namespace sd {

///////////////////////////////////////////////////////////////////////
template <typename T>
SD_DEVICE void pullRowsKernel(void *vx, void *vz, LongType len, LongType *indexes, LongType  *tadShapeInfo,
                              LongType  *tadOffsets, LongType  *zTadShapeInfo, LongType  *zTadOffsets) {
  auto x = reinterpret_cast<T *>(vx);
  auto z = reinterpret_cast<T *>(vz);
  auto tadLength = shape::length(tadShapeInfo);

  for (size_t idx = blockIdx.x; idx < len; idx += gridDim.x) {
    T *rX = x + tadOffsets[indexes[idx]];
    T *rZ = z + zTadOffsets[idx];

    for (size_t i = threadIdx.x; i < tadLength; i += blockDim.x) {
      sd::LongType xCoords[SD_MAX_RANK];
      sd::LongType zCoords[SD_MAX_RANK];
      sd::LongType xOffset;
      sd::LongType zOffset;

      INDEX2COORDS(i, shape::rank(tadShapeInfo), shape::shapeOf(tadShapeInfo), xCoords);
      COORDS2INDEX(shape::rank(tadShapeInfo), shape::stride(tadShapeInfo), xCoords, xOffset);
      INDEX2COORDS(i, shape::rank(zTadShapeInfo), shape::shapeOf(zTadShapeInfo), zCoords);
      COORDS2INDEX(shape::rank(zTadShapeInfo), shape::stride(zTadShapeInfo), zCoords, zOffset);

      rZ[zOffset] = rX[xOffset];
    }
  }
}
///////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void execPullRowsKernel(void *vx, void *vz, LongType len, LongType *indexes, LongType  *tadShapeInfo,
                                  LongType  *tadOffsets, LongType  *zTadShapeInfo,
                                  LongType  *zTadOffsets) {
  pullRowsKernel<T>(vx, vz, len, indexes, tadShapeInfo, tadOffsets, zTadShapeInfo, zTadOffsets);
}

///////////////////////////////////////////////////////////////////////
template <typename T>
SD_HOST void pullRowsKernelGeneric(dim3 &launchDims, cudaStream_t *stream, void *vx, void *vz, LongType len,
                                   LongType *indexes, LongType  *tadShapeInfo, LongType  *tadOffsets,
                                   LongType  *zTadShapeInfo, LongType  *zTadOffsets) {
  execPullRowsKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(vx, vz, len, indexes, tadShapeInfo,
                                                                               tadOffsets, zTadShapeInfo, zTadOffsets);
  DebugHelper::checkErrorCode(stream, "pullRows(...) failed");
}

BUILD_SINGLE_TEMPLATE(template void pullRowsKernelGeneric,
                      (dim3 & launchDims, cudaStream_t *stream, void *vx, void *vz, sd::LongType len,
                       sd::LongType *indexes, sd::LongType  *tadShapeInfo, sd::LongType  *tadOffsets,
                       sd::LongType  *zTadShapeInfo, sd::LongType  *zTadOffsets),
                      SD_COMMON_TYPES);
}  // namespace sd
