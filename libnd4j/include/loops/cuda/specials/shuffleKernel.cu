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
SD_KERNEL void execShuffleKernel(void **vdX, LongType **dxShapeInfo, void **vdZ, int N, int *shuffleMap,
                                 LongType **tadOnlyShapeInfo, LongType **tadOffsets) {
  // we assume that shuffle map for each X contains pair TAD Y
  auto dX = reinterpret_cast<T **>(vdX);
  auto dZ = reinterpret_cast<T **>(vdZ);

  __shared__ int tadLength;
  __shared__ int xRank;
  __shared__ int numTads;
  __shared__ LongType *xShapeInfo;
  __shared__ LongType xLength;

  for (int f = 0; f < N; f++) {
    auto x = reinterpret_cast<T *>(dX[f]);
    auto z = reinterpret_cast<T *>(dZ[f]);

    if (threadIdx.x == 0) {
      tadLength = shape::length(tadOnlyShapeInfo[f]);
      xShapeInfo = dxShapeInfo[f];
      xRank = shape::rank(xShapeInfo);
      xLength = shape::length(xShapeInfo);
      numTads = xLength / tadLength;
    }
    __syncthreads();

    if (xRank == 1) {
      int tid = threadIdx.x + blockIdx.x * blockDim.x;
      for (int r = tid; r < xLength; r += gridDim.x * blockDim.x) {
        auto swapIndex = shuffleMap[r];
        if (swapIndex >= 0 && swapIndex < xLength) {
          sd::LongType xCoords[SD_MAX_RANK];
          sd::LongType swapCoords[SD_MAX_RANK];
          sd::LongType xOffset;
          sd::LongType swapOffset;

          INDEX2COORDS(r, xRank, xShapeInfo, xCoords);
          COORDS2INDEX(xRank, shape::shapeOf(xShapeInfo), xCoords, xOffset);
          INDEX2COORDS(swapIndex, xRank, xShapeInfo, swapCoords);
          COORDS2INDEX(xRank, shape::shapeOf(xShapeInfo), swapCoords, swapOffset);

          T oldX = x[xOffset];
          x[xOffset] = x[swapOffset];
          x[swapOffset] = oldX;
        }
      }
    } else {
      // we roll over the pairs of TADs, thus limit is numTads / 2
      for (LongType r = blockIdx.x; r < numTads; r += gridDim.x) {
        if (shuffleMap[r] >= 0) {
          auto oldOffset = tadOffsets[f][r];
          auto newOffset = tadOffsets[f][shuffleMap[r]];

          auto rX = x + oldOffset;
          auto rY = x + newOffset;

          auto zX = z + oldOffset;
          auto zY = z + newOffset;

          for (LongType i = threadIdx.x; i < tadLength; i += blockDim.x) {
            sd::LongType xCoords[SD_MAX_RANK];
            sd::LongType yCoords[SD_MAX_RANK];
            sd::LongType xOffset;
            sd::LongType yOffset;

            INDEX2COORDS(i, shape::rank(tadOnlyShapeInfo[f]), tadOnlyShapeInfo[f], xCoords);
            COORDS2INDEX(shape::rank(tadOnlyShapeInfo[f]), shape::shapeOf(tadOnlyShapeInfo[f]), xCoords, xOffset);
            INDEX2COORDS(i, shape::rank(tadOnlyShapeInfo[f]), tadOnlyShapeInfo[f], yCoords);
            COORDS2INDEX(shape::rank(tadOnlyShapeInfo[f]), shape::shapeOf(tadOnlyShapeInfo[f]), yCoords, yOffset);

            xOffset += oldOffset;
            yOffset += newOffset;

            T oldX = x[xOffset];
            z[xOffset] = x[yOffset];
            z[yOffset] = oldX;
          }
        }
      }
    }
    __syncthreads();
  }
}
////////////////////////////////////////////////////////////////////////
template <typename T>
SD_HOST void shuffleKernelGeneric(dim3 &launchDims, cudaStream_t *stream, void **vdX, LongType **xShapeInfo,
                                  void **vdZ, int N, int *shuffleMap, LongType **tadOnlyShapeInfo, LongType **tadOffsets) {
  execShuffleKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(vdX, xShapeInfo, vdZ, N, shuffleMap,
                                                                              tadOnlyShapeInfo, tadOffsets);
  DebugHelper::checkErrorCode(stream, "shuffleGeneric(...) failed");
}

BUILD_SINGLE_TEMPLATE(template void shuffleKernelGeneric,
                      (dim3 & launchDims, cudaStream_t *stream, void **vdX, sd::LongType **xShapeInfo, void **vdZ,
                       int N, int *shuffleMap, sd::LongType **tadOnlyShapeInfo, sd::LongType **tadOffsets),
                      SD_COMMON_TYPES);
}  // namespace sd
