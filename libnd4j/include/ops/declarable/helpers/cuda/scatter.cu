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
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/scatter.h>
#include <numeric>
#include <helpers/ShapeUtils.h>
#include <helpers/PointersManager.h>
#include <TAD.h>


namespace nd4j {
    namespace ops {
        namespace helpers {
            template<typename T>
            __global__ static void scatterCuda(const int opCode, const int numOfSubArrs,
                                                     void* vx, const Nd4jLong *xShapeInfo, const Nd4jLong *xOffsets,
                                                     void* vy, const Nd4jLong *yShapeInfo, const Nd4jLong *yOffsets,
                                                     const int* indexes, const bool locking) {

                __shared__ T *x, *y;
                __shared__ Nd4jLong arrLenX, arrLenY;

                for (int e = 0; e < numOfSubArrs; e++ ) {

                    const auto xIndex = indexes[e];
                    const bool isOwner = xIndex < gridDim.x ? blockIdx.x == xIndex : blockIdx.x == xIndex % gridDim.x;

                    if (!isOwner)
                        continue;

                    if (threadIdx.x == 0) {
                        x = reinterpret_cast<T*>(vx) + xOffsets[xIndex];
                        y = reinterpret_cast<T*>(vy) + yOffsets[e];
                        arrLenX = shape::length(xShapeInfo);
                        arrLenY = shape::length(yShapeInfo);
                    }
                    __syncthreads();

                    if (arrLenX != arrLenY)
                        return;

                    for (Nd4jLong i = threadIdx.x; i < arrLenX; i += blockDim.x) {

                        const auto xOffset = shape::getIndexOffset(i, xShapeInfo, arrLenX);
                        const auto yOffset = shape::getIndexOffset(i, yShapeInfo, arrLenY);

                        switch (opCode) {
                            case 0:
                                x[xOffset] += y[yOffset];
                                break;
                            case 1:
                                x[xOffset] -= y[yOffset];
                                break;
                            case 2:
                                x[xOffset] *= y[yOffset];
                                break;
                            case 3:
                                x[xOffset] /= y[yOffset];
                                break;
                            case 4:
                                x[xOffset] = y[yOffset] - x[xOffset];
                                break;
                            case 5:
                                x[xOffset] = y[yOffset] / x[xOffset];
                                break;
                            case 6:
                                x[xOffset] = y[yOffset];
                                break;
                            default:
                                continue;
                        }
                    }
                    __syncthreads();
                }
            }


            template <typename T>
            void scatter_(graph::LaunchContext *context, pairwise::Ops op, const NDArray& indices, const NDArray& updates, NDArray& output, const bool lock) {
                int axis = 0;
                shape::TAD tadX;
                tadX.init(output.getShapeInfo(), &axis, 1);
                tadX.createTadOnlyShapeInfo();
                tadX.createOffsets();

                shape::TAD tadY;
                tadY.init(updates.getShapeInfo(), &axis, 1);
                tadY.createTadOnlyShapeInfo();
                tadY.createOffsets();

                PointersManager manager(context, "scatter");
                auto psX = reinterpret_cast<Nd4jLong *>(manager.replicatePointer(tadX.tadOnlyShapeInfo, shape::shapeInfoByteLength(tadX.tadOnlyShapeInfo)));
                auto psY = reinterpret_cast<Nd4jLong *>(manager.replicatePointer(tadY.tadOnlyShapeInfo, shape::shapeInfoByteLength(tadY.tadOnlyShapeInfo)));

                auto poX = reinterpret_cast<Nd4jLong *>(manager.replicatePointer(tadX.tadOffsets, tadX.numTads * sizeof(Nd4jLong)));
                auto poY = reinterpret_cast<Nd4jLong *>(manager.replicatePointer(tadY.tadOffsets, tadY.numTads * sizeof(Nd4jLong)));

                NDArray::prepareSpecialUse({&output}, {&updates, &indices});

                scatterCuda<T><<<512, 512, 1024, *context->getCudaStream()>>>(op, indices.lengthOf(), output.getSpecialBuffer(), psX, poX, updates.getSpecialBuffer(), psY, poY, reinterpret_cast<int *>(indices.getSpecialBuffer()), lock);

                NDArray::registerSpecialUse({&output}, {&updates, &indices});
                manager.synchronize();
            }

            void scatter(graph::LaunchContext *context, pairwise::Ops op, const NDArray& indices, const NDArray& updates, NDArray& output, const bool lock) {
                BUILD_SINGLE_SELECTOR(output.dataType(), scatter_, (context, op, indices, updates, output, lock), LIBND4J_TYPES);
            }

            void scatterND(graph::LaunchContext *context, pairwise::Ops op, const NDArray& indices, const NDArray& updates, NDArray& output, const bool lock) {

            }

            void scatterForLoss(graph::LaunchContext *context, const NDArray& indices, const NDArray& updates, NDArray& output, const bool calcGrad) {

            }
        }
    }
}