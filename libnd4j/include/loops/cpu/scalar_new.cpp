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
// Created by raver on 9/2/2018.
//

#include <loops/scalar_new.h>
#include <ops/ops.h>
#include <types/types.h>

namespace functions {
    namespace scalar {
        template <typename X, typename Y>
        void NewScalarTransform<X, Y>::transform(const int opNum, X *x, Nd4jLong *xShapeInfo, X *result, Nd4jLong *resultShapeInfo, Y scalar, X *extraParams) {
            // NEW DISPATCH BY OP_NUM
            NewScalarTransform<X,Y>::template transform<simdOps::NewAdd<X, Y>>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams);
        }

        template <typename X, typename Y>
        template<typename OpType>
        void NewScalarTransform<X, Y>::transform(X *x, Nd4jLong xStride, X *result, Nd4jLong resultStride, Y scalar, X *extraParams, const Nd4jLong n) {
            int num_threads = 1;
            Nd4jLong span = 100;// (n / num_threads) + 8;

            if (xStride == 1 && resultStride == 1) {
                if (num_threads > 1) {
#pragma omp parallel num_threads(num_threads) if (num_threads>1) proc_bind(AFFINITY) default(shared)
                    {
                        Nd4jLong tid = omp_get_thread_num();
                        Nd4jLong start = span * tid;
                        Nd4jLong end = span * (tid + 1);
                        if (end > n) end = n;
#pragma omp simd
                        for (Nd4jLong i = start; i < end; i++) {
                            result[i] = OpType::op(x[i], scalar, extraParams);
                        }
                    }
                } else {
#pragma omp simd
                    for (Nd4jLong i = 0; i < n; i++) {
                        result[i] = OpType::op(x[i], scalar, extraParams);
                    }
                }
            }

            else {
                if (num_threads > 1) {
#pragma omp parallel num_threads(num_threads) if (num_threads>1) proc_bind(AFFINITY) default(shared)
                    {
                        Nd4jLong tid = omp_get_thread_num();
                        Nd4jLong start = span * tid;
                        Nd4jLong end = span * (tid + 1);
                        if (end > n) end = n;
#pragma omp simd
                        for (Nd4jLong i = start; i < end; i++) {
                            result[i * resultStride] = OpType::op(x[i * xStride], scalar, extraParams);
                        }
                    }
                } else {
#pragma omp simd
                    for (Nd4jLong i = 0; i < n; i++) {
                        result[i * resultStride] = OpType::op(x[i * xStride], scalar, extraParams);
                    }
                }
            }
        }

        template <typename X, typename Y>
        template<typename OpType>
        void NewScalarTransform<X, Y>::transform(X *x, Nd4jLong *xShapeInfo, X *result, Nd4jLong *resultShapeInfo, Y scalar, X *extraParams) {
            // actual implementation
            char xOrdering = shape::order(xShapeInfo);
            char resultOrdering = shape::order(resultShapeInfo);
            auto xElementWiseStride = shape::elementWiseStride(xShapeInfo);

            if (xElementWiseStride == 1 && shape::elementWiseStride(resultShapeInfo) == 1 && xOrdering == resultOrdering) {
                transform<OpType>(x, 1, result, 1, scalar, extraParams, shape::length(xShapeInfo));
                return;
            }

            auto resultElementWiseStride = shape::elementWiseStride(resultShapeInfo);

            const Nd4jLong n = shape::length(xShapeInfo);

            if(xElementWiseStride >= 1 && resultElementWiseStride >= 1) {
                transform<OpType>(x,xElementWiseStride,result,resultElementWiseStride,scalar,extraParams,n);
            } else {
                Nd4jLong xIdx[MAX_RANK];
                Nd4jLong resultIdx[MAX_RANK];

                auto xShape = shape::shapeOf(xShapeInfo);
                auto resultShape = shape::shapeOf(resultShapeInfo);

                auto xStride = shape::stride(xShapeInfo);
                auto resultStride = shape::stride(resultShapeInfo);
                int xRank = shape::rank(xShapeInfo);
                int resultRank = shape::rank(resultShapeInfo);

#pragma omp parallel for schedule(guided) if (n > ELEMENT_THRESHOLD) proc_bind(AFFINITY) default(shared)
                for (Nd4jLong i = 0; i < n; i++) {
                    shape::ind2sub(xRank, xShape, i, n, xIdx);
                    shape::ind2sub(resultRank, resultShape, i, n, resultIdx);

                    auto xOffset2 = shape::getOffset(0, xShape, xStride, xIdx, xRank);
                    auto resultOffset2 = shape::getOffset(0, resultShape, resultStride, resultIdx, resultRank);

                    result[resultOffset2] = OpType::op(x[xOffset2], scalar, extraParams);
                }
            }
        }

        BUILD_DOUBLE_TEMPLATE(template class NewScalarTransform, , LIBND4J_TYPES, LIBND4J_TYPES);
    }
}