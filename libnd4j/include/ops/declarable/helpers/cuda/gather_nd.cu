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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 20.04.2018
//


#include<ops/declarable/helpers/transforms.h>
#include <array/ResultSet.h>
#include <helpers/ShapeUtils.h>
#include <numeric>
#include <NDArrayFactory.h>
#include <helpers/TAD.h>
#include <exceptions/cuda_exception.h>
#include <PointersManager.h>
#include <ConstantTadHelper.h>

namespace nd4j {
    namespace ops {
        namespace helpers {
            ///////////////////////////////////////////////////////////////////
// x - input, y - indices, z - output
            template<typename X, typename Y>
            __global__ static void gatherNDCuda(const void *vx, const Nd4jLong *xShapeInfo,
                                                const void *vy, const Nd4jLong *yShapeInfo,
                                                void *vz, const Nd4jLong *zShapeInfo) {

                const auto x = reinterpret_cast<const X*>(vx);
                const auto y = reinterpret_cast<const Y*>(vy);
                auto z = reinterpret_cast<X*>(vz);

                __shared__ int xRank, yRank, zRank, maxRank, yLastDim;
                __shared__ Nd4jLong zLen, totalThreads, *sharedMem;

                if (threadIdx.x == 0) {
                    extern __shared__ unsigned char shmem[];
                    sharedMem = reinterpret_cast<Nd4jLong*>(shmem);

                    xRank   = shape::rank(xShapeInfo);
                    yRank   = shape::rank(yShapeInfo);
                    zRank   = shape::rank(zShapeInfo);
                    maxRank = nd4j::math::nd4j_max<int>(yRank, nd4j::math::nd4j_max<int>(xRank, zRank));

                    zLen     = shape::length(zShapeInfo);
                    yLastDim = yShapeInfo[yRank];

                    totalThreads = gridDim.x * blockDim.x;
                }
                __syncthreads();

                auto coord = sharedMem + threadIdx.x * maxRank;

                Nd4jLong *zCoordStart, *xCoordStart;

                if(yLastDim == xRank) {
                    zCoordStart = coord;
                    xCoordStart = coord;
                }
                if(zRank >= xRank) {
                    zCoordStart = coord;
                    xCoordStart = coord + zRank - xRank;
                }
                else {
                    zCoordStart = coord + xRank - zRank;
                    xCoordStart = coord;
                }

                const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

                for (Nd4jLong i = tid; i < zLen; i += totalThreads) {

                    shape::index2coords(i, zShapeInfo, zCoordStart);

                    const auto zOffset = shape::getOffset(zShapeInfo, zCoordStart);

                    // last y coordinate
                    int coordToRestore;
                    if(yLastDim != xRank)
                        coordToRestore = static_cast<int>(zCoordStart[yRank - 1]);

                    zCoordStart[yRank - 1] = 0; // last y coordinate
                    const auto yOffset = shape::getOffset(yShapeInfo, zCoordStart);

                    //restore z coordinate
                    if(yLastDim != xRank)
                        zCoordStart[yRank - 1] = coordToRestore;

                    // construct coordinates for x
                    for(uint j = 0; j < yLastDim; ++j)
                        xCoordStart[j] = y[yOffset + j * yShapeInfo[2 * yRank]];   // last stride

                    const auto xOffset = shape::getOffset(xShapeInfo, xCoordStart);

                    z[zOffset] = x[xOffset];
                    printf("z[%lld] = x[%lld] = %f\n", zOffset, xOffset, (float) z[zOffset]);
                }
            }

///////////////////////////////////////////////////////////////////
            template<typename X, typename Y>
            static void gatherNDCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                                             const void *vx, const Nd4jLong *xShapeInfo,
                                             const void *vy, const Nd4jLong *yShapeInfo,
                                             void *vz, const Nd4jLong *zShapeInfo) {

                gatherNDCuda<X,Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo);
            }

///////////////////////////////////////////////////////////////////
            void gatherND(nd4j::LaunchContext * context, NDArray& input, NDArray& indices, NDArray& output) {

                const int maxRank = nd4j::math::nd4j_max<int>(indices.rankOf(), nd4j::math::nd4j_max<int>(input.rankOf(), output.rankOf()));

                const int threadsPerBlock = 256;
                const int blocksPerGrid = (output.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
                const int sharedMem = 8 * threadsPerBlock * maxRank + 128;

                const auto xType = input.dataType();
                const auto yType = indices.dataType();

                PointersManager manager(context, "gatherND");

                NDArray::prepareSpecialUse({&output}, {&input, &indices});
                BUILD_DOUBLE_SELECTOR(xType, yType, gatherNDCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), indices.getSpecialBuffer(), indices.getSpecialShapeInfo(), output.getSpecialBuffer(), output.getSpecialShapeInfo()), LIBND4J_TYPES, INDEXING_TYPES);
                NDArray::registerSpecialUse({&output}, {&input, &indices});

                manager.synchronize();
            }
        }
    }
}