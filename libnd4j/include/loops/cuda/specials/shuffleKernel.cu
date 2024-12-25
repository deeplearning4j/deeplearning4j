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
// @author Yurii Shyrma, created on 15.11.2018
//
#include <loops/special_kernels.h>

    namespace sd {
    template <typename T>
    SD_KERNEL void execShuffleKernel(
        void** vdX,
        LongType** dxShapeInfo,
        void** vdZ,
        int N,
        int* shuffleMap,
        LongType** tadOnlyShapeInfo,
        LongType** tadOffsets) {

      const int tid = threadIdx.x + blockIdx.x * blockDim.x;
      const int totalThreads = gridDim.x * blockDim.x;

      // Main array shape data
      __shared__ int xRank;
      __shared__ LongType xLength;
      __shared__ const LongType* xShape;
      __shared__ const LongType* xStride;

      // TAD shape data
      __shared__ int tadRank;
      __shared__ int tadLength;
      __shared__ int numTads;
      __shared__ const LongType* tadShape;
      __shared__ const LongType* tadStride;

      // Current shape info pointer
      __shared__ LongType* xShapeInfo;

      // Process each array
      for (int arrIndex = 0; arrIndex < N; arrIndex++) {
        auto x = reinterpret_cast<T*>(vdX[arrIndex]);
        auto z = reinterpret_cast<T*>(vdZ[arrIndex]);

        if (threadIdx.x == 0) {
          // Cache main array shape data
          xShapeInfo = dxShapeInfo[arrIndex];
          xRank = shape::rank(xShapeInfo);
          xLength = shape::length(xShapeInfo);
          xShape = shape::shapeOf(xShapeInfo);
          xStride = shape::stride(xShapeInfo);

          // Cache TAD shape data
          tadLength = static_cast<int>(shape::length(tadOnlyShapeInfo[arrIndex]));
          tadRank = shape::rank(tadOnlyShapeInfo[arrIndex]);
          tadShape = shape::shapeOf(tadOnlyShapeInfo[arrIndex]);
          tadStride = shape::stride(tadOnlyShapeInfo[arrIndex]);
          numTads = static_cast<int>(xLength / tadLength);
        }
        __syncthreads();

        // Rank-1 case: treat as vector
        if (xRank == 1) {
          for (LongType elem = tid; elem < xLength; elem += totalThreads) {
            const int swapIndex = shuffleMap[elem];
            if (swapIndex >= 0 && swapIndex < xLength) {
              sd::LongType xCoords[SD_MAX_RANK];
              sd::LongType swapCoords[SD_MAX_RANK];
              sd::LongType xOffset;
              sd::LongType swapOffset;

              // Use cached shape data for coordinate transforms
              INDEX2COORDS(elem, xRank, xShape, xCoords);
              COORDS2INDEX(xRank, xStride, xCoords, xOffset);

              INDEX2COORDS(swapIndex, xRank, xShape, swapCoords);
              COORDS2INDEX(xRank, xStride, swapCoords, swapOffset);

              // Swap values
              T oldVal = x[xOffset];
              x[xOffset] = x[swapOffset];
              x[swapOffset] = oldVal;
            }
          }
        }
        else {
          // TAD-based processing
          for (LongType r = blockIdx.x; r < numTads; r += gridDim.x) {
            const int swapTarget = shuffleMap[r];
            if (swapTarget >= 0) {
              const auto oldOffset = tadOffsets[arrIndex][r];
              const auto newOffset = tadOffsets[arrIndex][swapTarget];

              // Pointers to TADs
              auto rX = x + oldOffset;
              auto rY = x + newOffset;
              auto zX = z + oldOffset;
              auto zY = z + newOffset;

              // Process elements within TAD
              for (LongType i = threadIdx.x; i < tadLength; i += blockDim.x) {
                sd::LongType xCoords[SD_MAX_RANK];
                sd::LongType xOffset;
                sd::LongType yCoords[SD_MAX_RANK];
                sd::LongType yOffset;

                // Use cached TAD shape data for coordinate transforms
                INDEX2COORDS(i, tadRank, tadShape, xCoords);
                COORDS2INDEX(tadRank, tadStride, xCoords, xOffset);

                INDEX2COORDS(i, tadRank, tadShape, yCoords);
                COORDS2INDEX(tadRank, tadStride, yCoords, yOffset);

                // Add TAD base offsets
                xOffset += oldOffset;
                yOffset += newOffset;

                // Perform swap via z array
                T oldVal = x[xOffset];
                z[xOffset] = x[yOffset];
                z[yOffset] = oldVal;
              }
            }
          }
        }
        __syncthreads();
      }
    }
  template <typename T>
  SD_HOST void shuffleKernelGeneric(
      dim3 &launchDims,
      cudaStream_t *stream,
      void** vdX,
      LongType** xShapeInfo,
      void** vdZ,
      int N,
      int* shuffleMap,
      LongType** tadOnlyShapeInfo,
      LongType** tadOffsets) {

    execShuffleKernel<T>
        <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
            vdX,
            xShapeInfo,
            vdZ,
            N,
            shuffleMap,
            tadOnlyShapeInfo,
            tadOffsets);

    DebugHelper::checkErrorCode(stream, "shuffleGeneric(...) failed");
  }

  BUILD_SINGLE_TEMPLATE(
      template void shuffleKernelGeneric,
      (dim3 & launchDims,
       cudaStream_t *stream,
       void** vdX,
       sd::LongType** xShapeInfo,
       void** vdZ,
       int N,
       int* shuffleMap,
       sd::LongType** tadOnlyShapeInfo,
       sd::LongType** tadOffsets),
      SD_COMMON_TYPES);

}  // namespace sd
