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
// Created by raver119 on 15.12.17.
//

#include <op_boilerplate.h>
#include <loops/random.h>

namespace functions {
    namespace random {

        template<typename X>
        template<typename OpClass>
        void RandomFunction<X>::execTransform(Nd4jPointer state, X *x, Nd4jLong *xShapeBuffer, X *y, Nd4jLong *yShapeBuffer, X *z, Nd4jLong *zShapeBuffer, X *extraArguments) {

            if (OpClass::requiresSpecial) {
                OpClass::specialOp(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments);
                return;
            }

            auto length = shape::length(zShapeBuffer);
            auto xEWS = shape::elementWiseStride(xShapeBuffer);
            auto yEWS = shape::elementWiseStride(yShapeBuffer);
            auto zEWS = shape::elementWiseStride(zShapeBuffer);

            nd4j::random::RandomBuffer *buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (state);

            int elementsPerThread = length / ELEMENT_THRESHOLD;
            int _threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
            _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

            if (xEWS >= 1 && yEWS >= 1 && zEWS >= 1) {
                if (xEWS == 1 && yEWS == 1 && zEWS == 1) {
#pragma omp parallel for num_threads(_threads) if (_threads > 1) schedule(guided)
                    for (Nd4jLong e = 0; e < length; e++) {
                        z[e] = OpClass::op(x[e], y[e], e, length, buffer, extraArguments);
                    }

                } else {
#pragma omp parallel for num_threads(_threads) if (_threads > 1) schedule(guided)
                    for (Nd4jLong e = 0; e < length; e++) {
                        z[e * zEWS] = OpClass::op(x[e * xEWS], y[e * yEWS], e, length, buffer, extraArguments);
                    }
                }
            } else {
                // ind2sub branch
                Nd4jLong xCoord[MAX_RANK];
                Nd4jLong yCoord[MAX_RANK];
                Nd4jLong zCoord[MAX_RANK];

                int xRank = shape::rank(xShapeBuffer);
                int yRank = shape::rank(yShapeBuffer);
                int zRank = shape::rank(zShapeBuffer);

                auto xShape = shape::shapeOf(xShapeBuffer);
                auto yShape = shape::shapeOf(yShapeBuffer);
                auto zShape = shape::shapeOf(zShapeBuffer);

                auto xStride = shape::stride(xShapeBuffer);
                auto yStride = shape::stride(yShapeBuffer);
                auto zStride = shape::stride(zShapeBuffer);

#pragma omp parallel for num_threads(_threads) if (_threads > 1) schedule(guided) private(xCoord, yCoord, zCoord)
                for (Nd4jLong i = 0; i < length; i++) {
                    shape::ind2sub(xRank, xShape, i, length, xCoord);
                    shape::ind2sub(yRank, yShape, i, length, yCoord);
                    shape::ind2sub(zRank, zShape, i, length, zCoord);

                    auto xOffset2 = shape::getOffset(0, xShape, xStride, xCoord, xRank);
                    auto yOffset2 = shape::getOffset(0, yShape, yStride, yCoord, yRank);
                    auto zOffset2 = shape::getOffset(0, zShape, zStride, zCoord, zRank);


                    z[zOffset2] = OpClass::op(x[xOffset2], y[yOffset2], i, length, buffer, extraArguments);
                }
            }

            // update rng state
            buffer->rewindH(length);
        };



        template<typename X>
        template<typename OpClass>
        void RandomFunction<X>::execTransform(Nd4jPointer state, X *x, Nd4jLong *xShapeBuffer, X *z, Nd4jLong *zShapeBuffer, X *extraArguments) {
            auto length = shape::length(zShapeBuffer);
            auto xEWS = shape::elementWiseStride(xShapeBuffer);
            auto zEWS = shape::elementWiseStride(zShapeBuffer);

            nd4j::random::RandomBuffer *buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (state);

            Nd4jLong elementsPerThread = length / ELEMENT_THRESHOLD;
            Nd4jLong _threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
            _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

            if (xEWS >= 1 && zEWS >= 1) {
                if (xEWS == 1 && zEWS == 1) {
#pragma omp parallel for num_threads(_threads) if (_threads > 1) schedule(guided)
                    for (Nd4jLong e = 0; e < length; e++) {
                        z[e] = OpClass::op(x[e], e, length,  buffer, extraArguments);
                    }

                } else {
#pragma omp parallel for num_threads(_threads) if (_threads > 1) schedule(guided)
                    for (Nd4jLong e = 0; e < length; e++) {
                        z[e * zEWS] = OpClass::op(x[e * xEWS], e, length, buffer, extraArguments);
                    }
                }
            } else {
                // ind2sub branch
                Nd4jLong xCoord[MAX_RANK];
                Nd4jLong zCoord[MAX_RANK];

                int xRank = shape::rank(xShapeBuffer);
                int zRank = shape::rank(zShapeBuffer);

                auto xShape = shape::shapeOf(xShapeBuffer);
                auto zShape = shape::shapeOf(zShapeBuffer);

                auto xStride = shape::stride(xShapeBuffer);
                auto zStride = shape::stride(zShapeBuffer);

#pragma omp parallel for num_threads(_threads) if (_threads > 1) schedule(guided) private(zCoord, xCoord)
                for (Nd4jLong i = 0; i < length; i++) {
                    shape::ind2sub(xRank, xShape, i, length, xCoord);
                    shape::ind2sub(zRank, zShape, i, length, zCoord);

                    auto xOffset2 = shape::getOffset(0, xShape, xStride, xCoord, xRank);
                    auto zOffset2 = shape::getOffset(0, zShape, zStride, zCoord, zRank);

                    z[zOffset2] = OpClass::op(x[xOffset2], i, length, buffer, extraArguments);
                }
            }

            // update rng state
            buffer->rewindH(length);
        }


        template<typename X>
        template<typename OpClass>
        void RandomFunction<X>::execTransform(Nd4jPointer state, X *z, Nd4jLong  *zShapeBuffer, X *extraArguments) {
            auto length = shape::length(zShapeBuffer);
            auto ews = shape::elementWiseStride(zShapeBuffer);

            nd4j::random::RandomBuffer *buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (state);

            Nd4jLong elementsPerThread = length / ELEMENT_THRESHOLD;
            Nd4jLong _threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
            _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

            if (ews >= 1) {
                if (ews == 1) {
#pragma omp parallel for num_threads(_threads) if (_threads > 1) schedule(guided)
                    for (Nd4jLong x = 0; x < length; x++) {
                        z[x] = OpClass::op(x, length, buffer, extraArguments);
                    }

                } else {
#pragma omp parallel for num_threads(_threads) if (_threads > 1) schedule(guided)
                    for (Nd4jLong x = 0; x < length; x++) {
                        z[x * ews] = OpClass::op(x, length, buffer, extraArguments);
                    }
                }
            } else {
                // ind2sub branch
                Nd4jLong zCoord[MAX_RANK];

                int zRank = shape::rank(zShapeBuffer);
                auto zShape = shape::shapeOf(zShapeBuffer);
                auto zStride = shape::stride(zShapeBuffer);

#pragma omp parallel for num_threads(_threads) if (_threads > 1) schedule(guided) private(zCoord)
                for (Nd4jLong i = 0; i < length; i++) {
                    shape::ind2sub(zRank, zShape, i, length, zCoord);

                    auto zOffset2 = shape::getOffset(0, zShape, zStride, zCoord, zRank);
                    z[zOffset2] = OpClass::op(i, length, buffer,  extraArguments);
                }
            }

            // update rng state
            buffer->rewindH(length);
        }

        template<typename X>
        void RandomFunction<X>::execTransform(int opNum, Nd4jPointer state, X *x, Nd4jLong *xShapeBuffer, X *z, Nd4jLong *zShapeBuffer, X *extraArguments) {
            DISPATCH_BY_OPNUM_T(execTransform, PARAMS(state, x, xShapeBuffer, z, zShapeBuffer, extraArguments), RANDOM_OPS)
        }

        template<typename X>
        void RandomFunction<X>::execTransform(int opNum, Nd4jPointer state, X *x, Nd4jLong *xShapeBuffer, X *y, Nd4jLong *yShapeBuffer, X *z, Nd4jLong *zShapeBuffer, X *extraArguments) {
            DISPATCH_BY_OPNUM_T(execTransform, PARAMS(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments), RANDOM_OPS)
        }

        template<typename X>
        void RandomFunction<X>::execTransform(int opNum, Nd4jPointer state, X *z, Nd4jLong *zShapeBuffer, X *extraArguments) {
            DISPATCH_BY_OPNUM_T(execTransform, PARAMS(state, z, zShapeBuffer, extraArguments), RANDOM_OPS)
        }

        // FIXME: eventually we might want to get rid of that
#ifndef __CLION_IDE__
        BUILD_CALL_1(template void RandomFunction<float>::execTransform, float, (Nd4jPointer state, float *x, Nd4jLong *xShapeBuffer, float *y, Nd4jLong *yShapeBuffer, float *z, Nd4jLong *zShapeBuffer, float *extraArguments), RANDOM_OPS)
        BUILD_CALL_1(template void RandomFunction<float16>::execTransform, float16, (Nd4jPointer state, float16 *x, Nd4jLong *xShapeBuffer, float16 *y, Nd4jLong *yShapeBuffer, float16 *z, Nd4jLong *zShapeBuffer, float16 *extraArguments), RANDOM_OPS)
        BUILD_CALL_1(template void RandomFunction<double>::execTransform, double, (Nd4jPointer state, double *x, Nd4jLong *xShapeBuffer, double *y, Nd4jLong *yShapeBuffer, double *z, Nd4jLong *zShapeBuffer, double *extraArguments), RANDOM_OPS)

        BUILD_CALL_1(template void RandomFunction<float>::execTransform, float, (Nd4jPointer state, float *x, Nd4jLong *xShapeBuffer, float *z, Nd4jLong *zShapeBuffer, float *extraArguments), RANDOM_OPS)
        BUILD_CALL_1(template void RandomFunction<float16>::execTransform, float16, (Nd4jPointer state, float16 *x, Nd4jLong *xShapeBuffer, float16 *z, Nd4jLong *zShapeBuffer, float16 *extraArguments), RANDOM_OPS)
        BUILD_CALL_1(template void RandomFunction<double>::execTransform, double, (Nd4jPointer state, double *x, Nd4jLong *xShapeBuffer, double *z, Nd4jLong *zShapeBuffer, double *extraArguments), RANDOM_OPS)

        BUILD_CALL_1(template void RandomFunction<float>::execTransform, float, (Nd4jPointer state, float *z, Nd4jLong *zShapeBuffer, float *extraArguments), RANDOM_OPS)
        BUILD_CALL_1(template void RandomFunction<float16>::execTransform, float16, (Nd4jPointer state, float16 *z, Nd4jLong *zShapeBuffer, float16 *extraArguments), RANDOM_OPS)
        BUILD_CALL_1(template void RandomFunction<double>::execTransform, double, (Nd4jPointer state, double *z, Nd4jLong *zShapeBuffer, double *extraArguments), RANDOM_OPS)
#endif

        template class ND4J_EXPORT RandomFunction<float>;
        template class ND4J_EXPORT RandomFunction<float16>;
        template class ND4J_EXPORT RandomFunction<double>;
    }
}