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
            static void prefix_(scalar::Ops op, const void* vx, Nd4jLong* xShapeInfo, void* vz, Nd4jLong* zShapeInfo, bool exclusive, bool reverse) {
                const auto x = reinterpret_cast<const T *>(vx);
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
                    }
                    else {

                        for (Nd4jLong e = length - 1; e >= 0; --e) {

                            auto xOffset = shape::getIndexOffset(e, xShapeInfo);
                            auto zOffset = shape::getIndexOffset(e, zShapeInfo);
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
                    }
                    else {

                        for (int e = 0; e < length; e++) {

                            auto xOffset = shape::getIndexOffset(e, xShapeInfo);
                            auto zOffset = shape::getIndexOffset(e, zShapeInfo);
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
            static void prefix_(scalar::Ops op, const NDArray* x, NDArray* z, const std::vector<int>& dims, bool exclusive, bool reverse) {
                auto xTads = x->allTensorsAlongDimension(dims);
                auto zTads = z->allTensorsAlongDimension(dims);
                auto t = xTads.size();

                for (int e = 0; e < t; e++) {
                    auto tx = xTads.at(e);
                    auto tz = zTads.at(e);

                    prefix_<T>(op, tx->buffer(), tx->shapeInfo(), tz->buffer(), tz->shapeInfo(), exclusive, reverse);
                }
            };

            template <typename T>
            static void prefix_(scalar::Ops op, const NDArray* x, NDArray* z, bool exclusive, bool reverse) {
                    prefix_<T>(op, x->getBuffer(), x->getShapeInfo(), z->buffer(), z->shapeInfo(), exclusive, reverse);
            };

            void prefix(nd4j::LaunchContext * context, scalar::Ops op, const NDArray* x, NDArray* z, bool exclusive, bool reverse) {
                BUILD_SINGLE_SELECTOR(x->dataType(), prefix_, (op, x, z, exclusive, reverse), LIBND4J_TYPES);
            }

            void prefix(nd4j::LaunchContext * context, scalar::Ops op, const NDArray* x, NDArray* z, const std::vector<int>& dims, bool exclusive, bool reverse) {
                BUILD_SINGLE_SELECTOR(x->dataType(), prefix_, (op, x, z, dims, exclusive, reverse), LIBND4J_TYPES);
            }

            BUILD_SINGLE_TEMPLATE(template void prefix_, (scalar::Ops op, const void* vx, Nd4jLong* xShapeInfo, void* vz, Nd4jLong* zShapeInfo, bool exclusive, bool reverse), LIBND4J_TYPES);
            BUILD_SINGLE_TEMPLATE(template void prefix_, (scalar::Ops op, const NDArray* x, NDArray* z, const std::vector<int>& dims, bool exclusive, bool reverse), LIBND4J_TYPES);
            BUILD_SINGLE_TEMPLATE(template void prefix_, (scalar::Ops op, const NDArray* x, NDArray* z, bool exclusive, bool reverse), LIBND4J_TYPES);



        }
    }
}