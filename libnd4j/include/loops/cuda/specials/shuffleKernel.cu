/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
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

namespace nd4j {

////////////////////////////////////////////////////////////////////////
    template<typename T>
    __global__ void execShuffleKernel(void **vdX, Nd4jLong **dxShapeInfo,
                                      void **vdZ,
                                      int N,
                                      int *shuffleMap,
                                      Nd4jLong **tadOnlyShapeInfo, Nd4jLong **tadOffsets) {

        // we assume that shuffle map for each X contains pair TAD Y
        auto dX = reinterpret_cast<T **>(vdX);
        auto dZ = reinterpret_cast<T **>(vdZ);

        __shared__ int tadLength;
        __shared__ int xRank;
        __shared__ int tadEWS;
        __shared__ int numTads;
        __shared__ Nd4jLong* xShapeInfo;
        __shared__ Nd4jLong xLength;

        for (int f = 0; f < N; f++) {
            auto x = reinterpret_cast<T *>(dX[f]);
            auto z = reinterpret_cast<T *>(dZ[f]);

            if (threadIdx.x == 0) {
                tadLength = shape::length(tadOnlyShapeInfo[f]);
                tadEWS = shape::elementWiseStride(tadOnlyShapeInfo[f]);
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
                        int idx = r * tadEWS;
                        int swap = swapIndex * tadEWS;
                        T oldX = x[idx];
                        x[idx] = x[swap];
                        x[swap] = oldX;
                    }
                }
            } else {
                // we roll over the pairs of TADs, thus limit is numTads / 2
                for (uint r = blockIdx.x; r < numTads; r += gridDim.x) {
                    if (shuffleMap[r] >= 0) {
                        auto oldOffset = tadOffsets[f][r];
                        auto newOffset = tadOffsets[f][shuffleMap[r]];

                        auto rX = x + oldOffset;
                        auto rY = x + newOffset;

                        auto zX = z + oldOffset;
                        auto zY = z + newOffset;

                        // so we're going to change TAD[oldOffset] with TAD[newOffset]
                        if (tadEWS == 1) {
                            for (Nd4jLong i = threadIdx.x; i < tadLength; i += blockDim.x) {
                                T oldX = rX[i];
                                rX[i] = rY[i];
                                zY[i] = oldX;
                            }

                        } else {
                            for (Nd4jLong i = threadIdx.x; i < tadLength; i += blockDim.x) {

                                auto xOffset = shape::getIndexOffset(i, tadOnlyShapeInfo[f], tadLength);
                                auto yOffset = newOffset + xOffset;
                                xOffset += oldOffset;

                                T oldX = x[xOffset];
                                z[xOffset] = x[yOffset];
                                z[yOffset] = oldX;
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

////////////////////////////////////////////////////////////////////////
    template<typename T>
    __host__ void shuffleKernelGeneric(dim3 &launchDims, cudaStream_t *stream,
                                       void **vdX, Nd4jLong **xShapeInfo,
                                       void **vdZ,
                                       int N,
                                       int *shuffleMap,
                                       Nd4jLong **tadOnlyShapeInfo, Nd4jLong **tadOffsets) {

        execShuffleKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(vdX, xShapeInfo, vdZ, N, shuffleMap, tadOnlyShapeInfo, tadOffsets);
    }

    BUILD_SINGLE_TEMPLATE(template void ND4J_EXPORT shuffleKernelGeneric, (dim3 & launchDims, cudaStream_t * stream, void * *vdX, Nd4jLong * *xShapeInfo, void **vdZ, int N, int * shuffleMap, Nd4jLong * *tadOnlyShapeInfo, Nd4jLong * *tadOffsets), LIBND4J_TYPES);
}