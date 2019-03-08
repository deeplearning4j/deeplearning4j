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
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ConstantTadHelper.h>


namespace nd4j {
    namespace ops {
        namespace helpers {
            template<typename T, bool locking>
            __global__ static void scatterCuda(const int opCode, const int numOfSubArrs,
                                                     void* vx, const Nd4jLong *xShapeInfo, const Nd4jLong *xOffsets,
                                                     void* vy, const Nd4jLong *yShapeInfo, const Nd4jLong *yOffsets,
                                                     const int* indexes, unsigned int arrLenX, unsigned int arrLenY) {

                __shared__ T *x, *y;

                if (locking) {

                    for (int e = 0; e < numOfSubArrs; e++) {

                        const auto xIndex = indexes[e];
                        const bool isOwner = xIndex < gridDim.x ? blockIdx.x == xIndex : blockIdx.x == xIndex % gridDim.x;

                        if (!isOwner)
                            continue;

                        if (threadIdx.x == 0) {
                            x = reinterpret_cast<T *>(vx) + xOffsets[xIndex];
                            y = reinterpret_cast<T *>(vy) + yOffsets[e];
                        }
                        __syncthreads();

                        for (Nd4jLong i = threadIdx.x; i < arrLenX; i += blockDim.x) {

                            const auto xOffset = shape::getIndexOffset(i, xShapeInfo, arrLenX);
                            const auto yOffset = shape::getIndexOffset(i, yShapeInfo, arrLenY);

                            switch (opCode) {
                                case pairwise::Add:
                                    x[xOffset] += y[yOffset];
                                    break;
                                case pairwise::Subtract:
                                    x[xOffset] -= y[yOffset];
                                    break;
                                case pairwise::Multiply:
                                    x[xOffset] *= y[yOffset];
                                    break;
                                case pairwise::Divide:
                                    x[xOffset] /= y[yOffset];
                                    break;
                                case pairwise::ReverseSubtract:
                                    x[xOffset] = y[yOffset] - x[xOffset];
                                    break;
                                case pairwise::ReverseDivide:
                                    x[xOffset] = y[yOffset] / x[xOffset];
                                    break;
                                case pairwise::Copy2:
                                case pairwise::CopyPws:
                                    x[xOffset] = y[yOffset];
                                    break;
                                default:
                                    continue;
                            }
                        }
                        __syncthreads();
                    }
                } else {
                    for (int e = blockIdx.x; e < numOfSubArrs; e+= gridDim.x) {

                        if (threadIdx.x == 0) {
                            const auto xIndex = indexes[e];
                            x = reinterpret_cast<T *>(vx) + xOffsets[xIndex];
                            y = reinterpret_cast<T *>(vy) + yOffsets[e];
                        }
                        __syncthreads();

                        for (Nd4jLong i = threadIdx.x; i < arrLenX; i += blockDim.x) {
                            const auto xOffset = shape::getIndexOffset(i, xShapeInfo, arrLenX);
                            const auto yOffset = shape::getIndexOffset(i, yShapeInfo, arrLenY);

                            switch (opCode) {
                                case pairwise::Add:
                                    x[xOffset] += y[yOffset];
                                    break;
                                case pairwise::Subtract:
                                    x[xOffset] -= y[yOffset];
                                    break;
                                case pairwise::Multiply:
                                    x[xOffset] *= y[yOffset];
                                    break;
                                case pairwise::Divide:
                                    x[xOffset] /= y[yOffset];
                                    break;
                                case pairwise::ReverseSubtract:
                                    x[xOffset] = y[yOffset] - x[xOffset];
                                    break;
                                case pairwise::ReverseDivide:
                                    x[xOffset] = y[yOffset] / x[xOffset];
                                    break;
                                case pairwise::Copy2:
                                case pairwise::CopyPws:
                                    x[xOffset] = y[yOffset];
                                    break;
                                default:
                                    continue;
                            }
                        }
                        __syncthreads();
                    }
                }
            }


            template <typename T>
            void scatter_(graph::LaunchContext *context, pairwise::Ops op, const NDArray& indices, const NDArray& updates, NDArray& output, const bool lock) {
                std::vector<int> dims = {0};
                auto inverted = ShapeUtils::evalDimsToExclude(output.rankOf(), dims);

                auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output.getShapeInfo(), inverted);
                auto packY = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(updates.getShapeInfo(), inverted);

                auto psX = packX.specialShapeInfo();
                auto psY = packY.specialShapeInfo();

                PointersManager manager(context, "scatter");

                auto poX = packX.specialOffsets();
                auto poY = packY.specialOffsets();

                NDArray::prepareSpecialUse({&output}, {&updates, &indices});

                unsigned int tadLengthX = shape::length(packX.primaryShapeInfo());
                unsigned int tadLengthY = shape::length(packY.primaryShapeInfo());
                if (tadLengthX != tadLengthY)
                    throw std::runtime_error("scatter: Lengths of TADs must be equal");

                auto blockSize = nd4j::math::nd4j_max<int>(32, nd4j::math::nd4j_min<int>(tadLengthX, 1024));

                if (lock)
                    scatterCuda<T, true><<<512, blockSize, 1024, *context->getCudaStream()>>>(op, indices.lengthOf(), output.getSpecialBuffer(), psX, poX, updates.getSpecialBuffer(), psY, poY, reinterpret_cast<int *>(indices.getSpecialBuffer()), tadLengthX, tadLengthY);
                else
                    scatterCuda<T, false><<<512, blockSize, 1024, *context->getCudaStream()>>>(op, indices.lengthOf(), output.getSpecialBuffer(), psX, poX, updates.getSpecialBuffer(), psY, poY, reinterpret_cast<int *>(indices.getSpecialBuffer()), tadLengthX, tadLengthY);

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