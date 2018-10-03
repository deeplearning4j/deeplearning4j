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

#include <ops/ops.h>
#include <helpers/shape.h>
#include <helpers/TAD.h>
#include <ops/declarable/helpers/prefix.h>

namespace nd4j {
    namespace ops {
        namespace helpers {
            template <typename T>
            static void __prefix(scalar::Ops op, void* vx, Nd4jLong* xShapeInfo, void* vz, Nd4jLong* zShapeInfo, bool exclusive, bool reverse) {
                auto x = reinterpret_cast<T *>(vx);
                auto z = reinterpret_cast<T *>(vz);
                auto length = shape::length(xShapeInfo);

                T prevSum = op == scalar::Add ? (T) 0 : (T) 1;
                T sum = prevSum;
                                
                if (reverse) {
                    if (shape::elementWiseStride(xShapeInfo) == 1 && shape::elementWiseStride(zShapeInfo) == 1 &&
                        shape::order(xShapeInfo) == 'c' && shape::order(zShapeInfo) == 'c') {

                        for (Nd4jLong e = length - 1; e >= 0; --e) {
                            sum = op == scalar::Add ? simdOps::Add<T, T, T>::op(sum, x[e]) : simdOps::Multiply<T, T, T>::op(sum, x[e]);
                            if (!exclusive)
                                prevSum = sum;

                            z[e] = prevSum;

                            prevSum = sum;
                        }
                    } else {
                        Nd4jLong xCoord[MAX_RANK];
                        Nd4jLong zCoord[MAX_RANK];

                        int xRank = shape::rank(xShapeInfo);
                        int zRank = shape::rank(zShapeInfo);

                        auto xShape = shape::shapeOf(xShapeInfo);
                        auto zShape = shape::shapeOf(zShapeInfo);

                        auto xStride = shape::stride(xShapeInfo);
                        auto zStride = shape::stride(zShapeInfo);

                        for (Nd4jLong e = length - 1; e >= 0; --e) {
                            shape::ind2subC(xRank, xShape, e, length, xCoord);
                            shape::ind2subC(zRank, zShape, e, length, zCoord);

                            auto xOffset = shape::getOffset(0, xShape, xStride, xCoord, xRank);
                            auto zOffset = shape::getOffset(0, zShape, zStride, zCoord, zRank);

                            sum = op == scalar::Add ? simdOps::Add<T, T, T>::op(sum, x[xOffset]) : simdOps::Multiply<T, T, T>::op(sum, x[xOffset]);
                            if (!exclusive)
                                prevSum = sum;

                            z[zOffset] = prevSum;

                            prevSum = sum;
                        }
                    }
                } else {
                    if (shape::elementWiseStride(xShapeInfo) == 1 && shape::elementWiseStride(zShapeInfo) == 1 &&
                        shape::order(xShapeInfo) == 'c' && shape::order(zShapeInfo) == 'c') {                        

                        for (int e = 0; e < length; e++) {
                            sum = op == scalar::Add ? simdOps::Add<T, T, T>::op(sum, x[e]) : simdOps::Multiply<T, T, T>::op(sum, x[e]);

                            if (!exclusive)
                                prevSum = sum;

                            z[e] = prevSum;

                            prevSum = sum;
                        }
                    } else {
                        Nd4jLong xCoord[MAX_RANK];
                        Nd4jLong zCoord[MAX_RANK];

                        auto xRank = shape::rank(xShapeInfo);
                        auto zRank = shape::rank(zShapeInfo);

                        auto xShape = shape::shapeOf(xShapeInfo);
                        auto zShape = shape::shapeOf(zShapeInfo);

                        auto xStride = shape::stride(xShapeInfo);
                        auto zStride = shape::stride(zShapeInfo);

                        for (int e = 0; e < length; e++) {
                            shape::ind2subC(xRank, xShape, e, length, xCoord);
                            shape::ind2subC(zRank, zShape, e, length, zCoord);

                            auto xOffset = shape::getOffset(0, xShape, xStride, xCoord, xRank);
                            auto zOffset = shape::getOffset(0, zShape, zStride, zCoord, zRank);

                            sum = op == scalar::Add ? simdOps::Add<T, T, T>::op(sum, x[xOffset]) : simdOps::Multiply<T, T, T>::op(sum, x[xOffset]);

                            if (!exclusive)
                                prevSum = sum;

                            z[zOffset] = prevSum;

                            prevSum = sum;
                        }
                    }
                }
            };

            template <typename T>
            static void __prefix(scalar::Ops op, NDArray* x, NDArray* z, std::vector<int>& dims, bool exclusive, bool reverse) {
                auto xTads = x->allTensorsAlongDimension(dims);
                auto zTads = z->allTensorsAlongDimension(dims);
                auto t = xTads->size();

// #pragma omp parallel for schedule(guided)
                for (int e = 0; e < t; e++) {
                    auto tx = xTads->at(e);
                    auto tz = zTads->at(e);

                    __prefix<T>(op, tx->buffer(), tx->shapeInfo(), tz->buffer(), tz->shapeInfo(), exclusive, reverse);
                }

                delete xTads;
                delete zTads;
            };

            template <typename T>
            static void __prefix(scalar::Ops op, NDArray* x, NDArray* z, bool exclusive, bool reverse) {
                    __prefix<T>(op, x->buffer(), x->shapeInfo(), z->buffer(), z->shapeInfo(), exclusive, reverse);
            };

            void _prefix(scalar::Ops op, NDArray* x, NDArray* z, bool exclusive, bool reverse) {
                BUILD_SINGLE_SELECTOR(x->dataType(), __prefix, (op, x, z, exclusive, reverse), LIBND4J_TYPES);
            }

            void _prefix(scalar::Ops op, NDArray* x, NDArray* z, std::vector<int>& dims, bool exclusive, bool reverse) {
                BUILD_SINGLE_SELECTOR(x->dataType(), __prefix, (op, x, z, dims, exclusive, reverse), LIBND4J_TYPES);
            }

            BUILD_SINGLE_TEMPLATE(template void __prefix, (scalar::Ops op, void* vx, Nd4jLong* xShapeInfo, void* vz, Nd4jLong* zShapeInfo, bool exclusive, bool reverse), LIBND4J_TYPES);
            BUILD_SINGLE_TEMPLATE(template void __prefix, (scalar::Ops op, NDArray* x, NDArray* z, std::vector<int>& dims, bool exclusive, bool reverse), LIBND4J_TYPES);
            BUILD_SINGLE_TEMPLATE(template void __prefix, (scalar::Ops op, NDArray* x, NDArray* z, bool exclusive, bool reverse), LIBND4J_TYPES);



        }
    }
}