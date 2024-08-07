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

////////////////////////////////////////////////////////////////////////
template <typename T>
SD_DEVICE void tearKernel(void* vx, LongType const* xShapeInfo, Pointer* targets, LongType const* zShapeInfo,
                          LongType const* tadShapeInfo, LongType const* tadOffsets) {
  __shared__ LongType tadLength;
  __shared__ int tadEWS;
  __shared__ int zEWS;
  //        __shared__ int tadRank;
  __shared__ LongType numTads;
  //        __shared__ int zRank;
  //        __shared__        sd::LongType *tadShape;
  //        __shared__        sd::LongType *tadStride;
  //        __shared__        sd::LongType const* zShape;
  //        __shared__        sd::LongType const* zStride;
  __shared__ T* x;
  if (threadIdx.x == 0) {
    tadLength = shape::length(tadShapeInfo);
    tadEWS = shape::elementWiseStride(tadShapeInfo);
    zEWS = shape::elementWiseStride(zShapeInfo);
    numTads = shape::length(xShapeInfo) / tadLength;
    x = static_cast<T*>(vx);
  }
  __syncthreads();

  for (LongType r = blockIdx.x; r < numTads; r += gridDim.x) {
    T* z = (T*)targets[r];
    T* s = x + tadOffsets[r];

    if (zEWS > 0 && tadEWS > 0) {
      for (LongType i = threadIdx.x; i < tadLength; i += blockDim.x) z[i * zEWS] = s[i * tadEWS];
    } else {
      for (LongType j = threadIdx.x; j < tadLength; j += blockDim.x) {
        auto xOffset = shape::getIndexOffset(j, tadShapeInfo);
        auto zOffset = shape::getIndexOffset(j, zShapeInfo);

        z[zOffset] = s[xOffset];
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void execTearKernel(void* vx, LongType const* xShapeInfo, Pointer* targets, LongType const* zShapeInfo,
                              LongType const* tadShapeInfo, LongType const* tadOffsets) {
  tearKernel<T>(vx, xShapeInfo, targets, zShapeInfo, tadShapeInfo, tadOffsets);
}

////////////////////////////////////////////////////////////////////////
template <typename T>
SD_HOST void tearKernelGeneric(dim3& launchDims, cudaStream_t* stream, void* vx, LongType const* xShapeInfo,
                               Pointer* targets, LongType const* zShapeInfo, LongType const* tadShapeInfo,
                               LongType const* tadOffsets) {
  execTearKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(vx, xShapeInfo, targets, zShapeInfo,
                                                                           tadShapeInfo, tadOffsets);
  DebugHelper::checkErrorCode(stream, "tear(...) failed");
}

BUILD_SINGLE_TEMPLATE(template void tearKernelGeneric,
                      (dim3 & launchDims, cudaStream_t* stream, void* vx, sd::LongType const* xShapeInfo,
                       sd::Pointer* targets, sd::LongType const* zShapeInfo, sd::LongType const* tadShapeInfo,
                       sd::LongType const* tadOffsets),
                      SD_COMMON_TYPES);
}  // namespace sd
