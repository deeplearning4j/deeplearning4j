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
#include <array/NDArrayFactory.h>
#include <helpers/TAD.h>
#include <exceptions/cuda_exception.h>
#include <helpers/PointersManager.h>
#include <helpers/ConstantTadHelper.h>

namespace sd {
    namespace ops {
        namespace helpers {
            template <typename X, typename Y>
            static _CUDA_G void scatterSimpleKernel(void *vx, Nd4jLong *xTadShape, Nd4jLong *xTadOffsets, Nd4jLong xLength, Nd4jLong numTads, void *vi, Nd4jLong *iShapeInfo, Nd4jLong iLength, void *vu, Nd4jLong *uShapeInfo, Nd4jLong uLength) {
                auto u = reinterpret_cast<X*>(vu);
                auto indices = reinterpret_cast<Y*>(vi);

                auto tid = threadIdx.x + blockIdx.x * blockDim.x;
                for (int i = tid; i < iLength; i += blockDim.x * gridDim.x) {
                    auto x = reinterpret_cast<X*>(vx) + xTadOffsets[i];
                    auto idx = indices[shape::getIndexOffset(i, iShapeInfo)];

                    x[shape::getIndexOffset(idx, xTadShape)] = u[shape::getIndexOffset(i, uShapeInfo)];
                }
            }


            template <typename X, typename Y>
            void scatterSimple_(sd::LaunchContext * context, const int opId, NDArray& input, const NDArray& updates, const NDArray& indices, const std::vector<int>& dimensions) {

                auto dims = ShapeUtils::evalDimsToExclude(input.rankOf(), dimensions);
                auto packX = ConstantTadHelper::getInstance()->tadForDimensions(input.getShapeInfo(), dims);

                auto xLength = shape::length(packX.primaryShapeInfo());
                auto iLength = indices.lengthOf();
                auto uLength = updates.lengthOf();

                scatterSimpleKernel<X,Y><<<256, 256, 1024, *context->getCudaStream()>>>(input.getSpecialBuffer(), packX.platformShapeInfo(), packX.platformOffsets(), xLength, packX.numberOfTads(), indices.getSpecialBuffer(), indices.getSpecialShapeInfo(), iLength, updates.getSpecialBuffer(), updates.getSpecialShapeInfo(), uLength);
            }


            void scatterSimple(sd::LaunchContext * context, const int opId, NDArray& input, const NDArray& updates, const NDArray& indices, const std::vector<int>& dimensions) {
                auto xType = input.dataType();
                auto yType = indices.dataType();

                if (opId != 6)
                    throw std::runtime_error("scatterSimple: only copy op is supported");

                NDArray::prepareSpecialUse({&input}, {&updates, &indices});

                BUILD_DOUBLE_SELECTOR(xType, yType, scatterSimple_, (context, opId, input, updates, indices, dimensions), LIBND4J_TYPES, INDEXING_TYPES);

                NDArray::registerSpecialUse({&input}, {&updates, &indices});
            }
        }
    }
}