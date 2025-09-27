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
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See
 * the License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author raver119@gmail.com
// @author Yurii Shyrma, created on 27.11.2018
//
#include <loops/special_kernels.h>
#include <ops/declarable/helpers/flatten.h>

    namespace sd {

  template <typename T>
  SD_KERNEL void flattenKernel(
      Pointer* extraPointers,
      int dOffset,
      char order,
      void* vz,
      LongType* zShapeInfo,
      void* vy,
      LongType* yShapeInfo) {

    auto z = reinterpret_cast<T*>(vz);
    auto y = reinterpret_cast<T*>(vy);

    __shared__ sd::LongType yLen;
    __shared__ int yRank;
    __shared__ const sd::LongType* yShapePtr;
    __shared__ const sd::LongType* yStridePtr;

    if (threadIdx.x == 0) {
      yLen       = shape::length(yShapeInfo);
      yRank      = shape::rank(yShapeInfo);
      yShapePtr  = shape::shapeOf(yShapeInfo);
      yStridePtr = shape::stride(yShapeInfo);
    }
    __syncthreads();

    const auto tid          = blockIdx.x * blockDim.x + threadIdx.x;
    const auto totalThreads = gridDim.x * blockDim.x;

    for (auto i = tid; i < yLen; i += totalThreads) {
      sd::LongType coords[SD_MAX_RANK];
      sd::LongType yOffset;

      INDEX2COORDS(i, yRank, yShapePtr, coords);
      COORDS2INDEX(yRank, yStridePtr, coords, yOffset);

      z[i + dOffset] = y[yOffset];
    }
  }

  template <typename T>
  SD_HOST void flattenKernelGeneric(
      dim3& launchDims,
      cudaStream_t* stream,
      Pointer* extraPointers,
      int dOffset,
      char order,
      void* vz,
      LongType* zShapeInfo,
      void* vy,
      LongType* yShapeInfo) {

    flattenKernel<T>
        <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
            extraPointers,
            dOffset,
            order,
            vz,
            zShapeInfo,
            vy,
            yShapeInfo);

    DebugHelper::checkErrorCode(stream, "flattenGeneric(...) failed");
  }

  BUILD_SINGLE_TEMPLATE(
       void flattenKernelGeneric,
      (dim3 & launchDims,
       cudaStream_t* stream,
       sd::Pointer* extraPointers,
       int dOffset,
       char order,
       void* vz,
       sd::LongType* zShapeInfo,
       void* vy,
       sd::LongType* yShapeInfo),
      SD_COMMON_TYPES);

}  // namespace sd
