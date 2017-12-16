//
// Created by raver119 on 15.12.17.
//

#include <op_boilerplate.h>
#include <loops/random.h>

namespace functions {
    namespace random {

        template<typename T>
        template<typename OpClass>
        void RandomFunction<T>::execTransform(Nd4jPointer state, T *x, int *xShapeBuffer, T *y, int *yShapeBuffer, T *z, int *zShapeBuffer, T *extraArguments) {

            if (OpClass::requiresSpecial) {
                OpClass::specialOp(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments);
                return;
            }

            Nd4jIndex length = shape::length(zShapeBuffer);
            int xEWS = shape::elementWiseStride(xShapeBuffer);
            int yEWS = shape::elementWiseStride(yShapeBuffer);
            int zEWS = shape::elementWiseStride(zShapeBuffer);

            nd4j::random::RandomBuffer *buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (state);

            int elementsPerThread = length / ELEMENT_THRESHOLD;
            int _threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
            _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

            if (xEWS >= 1 && yEWS >= 1 && zEWS >= 1) {
                if (xEWS == 1 && yEWS == 1 && zEWS == 1) {
#pragma omp parallel for num_threads(_threads) if (_threads > 1) schedule(guided)
                    for (Nd4jIndex e = 0; e < length; e++) {
                        z[e] = OpClass::op(x[e], y[e], e, length, buffer, extraArguments);
                    }

                } else {
#pragma omp parallel for num_threads(_threads) if (_threads > 1) schedule(guided)
                    for (Nd4jIndex e = 0; e < length; e++) {
                        z[e * zEWS] = OpClass::op(x[e * xEWS], y[e * yEWS], e, length, buffer, extraArguments);
                    }
                }
            } else {
                // ind2sub branch
                int xCoord[MAX_RANK];
                int yCoord[MAX_RANK];
                int zCoord[MAX_RANK];

                int xRank = shape::rank(xShapeBuffer);
                int yRank = shape::rank(yShapeBuffer);
                int zRank = shape::rank(zShapeBuffer);

                int *xShape = shape::shapeOf(xShapeBuffer);
                int *yShape = shape::shapeOf(yShapeBuffer);
                int *zShape = shape::shapeOf(zShapeBuffer);

                int *xStride = shape::stride(xShapeBuffer);
                int *yStride = shape::stride(yShapeBuffer);
                int *zStride = shape::stride(zShapeBuffer);

                int xOffset = shape::offset(xShapeBuffer);
                int yOffset = shape::offset(yShapeBuffer);
                int zOffset = shape::offset(zShapeBuffer);

#pragma omp parallel for num_threads(_threads) if (_threads > 1) schedule(guided) private(xCoord, yCoord, zCoord)
                for (Nd4jIndex i = 0; i < length; i++) {
                    shape::ind2sub(xRank, xShape, i, xCoord);
                    shape::ind2sub(yRank, yShape, i, yCoord);
                    shape::ind2sub(zRank, zShape, i, zCoord);

                    Nd4jIndex xOffset2 = shape::getOffset(xOffset, xShape, xStride, xCoord, xRank);
                    Nd4jIndex yOffset2 = shape::getOffset(yOffset, yShape, yStride, yCoord, yRank);
                    Nd4jIndex zOffset2 = shape::getOffset(zOffset, zShape, zStride, zCoord, zRank);


                    z[zOffset2] = OpClass::op(x[xOffset2], y[yOffset2], i, length, buffer, extraArguments);
                }
            }

            // update rng state
            buffer->rewindH(length);
        };



        template<typename T>
        template<typename OpClass>
        void RandomFunction<T>::execTransform(Nd4jPointer state, T *x, int *xShapeBuffer, T *z, int *zShapeBuffer, T *extraArguments) {
            Nd4jIndex length = shape::length(zShapeBuffer);
            int xEWS = shape::elementWiseStride(xShapeBuffer);
            int zEWS = shape::elementWiseStride(zShapeBuffer);

            nd4j::random::RandomBuffer *buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (state);

            Nd4jIndex elementsPerThread = length / ELEMENT_THRESHOLD;
            Nd4jIndex _threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
            _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

            if (xEWS >= 1 && zEWS >= 1) {
                if (xEWS == 1 && zEWS == 1) {
#pragma omp parallel for num_threads(_threads) if (_threads > 1) schedule(guided)
                    for (Nd4jIndex e = 0; e < length; e++) {
                        z[e] = OpClass::op(x[e], e, length,  buffer, extraArguments);
                    }

                } else {
#pragma omp parallel for num_threads(_threads) if (_threads > 1) schedule(guided)
                    for (Nd4jIndex e = 0; e < length; e++) {
                        z[e * zEWS] = OpClass::op(x[e * xEWS], e, length, buffer, extraArguments);
                    }
                }
            } else {
                // ind2sub branch
                int xCoord[MAX_RANK];
                int zCoord[MAX_RANK];

                int xRank = shape::rank(xShapeBuffer);
                int zRank = shape::rank(zShapeBuffer);

                int *xShape = shape::shapeOf(xShapeBuffer);
                int *zShape = shape::shapeOf(zShapeBuffer);

                int *xStride = shape::stride(xShapeBuffer);
                int *zStride = shape::stride(zShapeBuffer);

                int xOffset = shape::offset(xShapeBuffer);
                int zOffset = shape::offset(zShapeBuffer);

#pragma omp parallel for num_threads(_threads) if (_threads > 1) schedule(guided) private(zCoord, xCoord)
                for (Nd4jIndex i = 0; i < length; i++) {
                    shape::ind2sub(xRank, xShape, i, xCoord);
                    shape::ind2sub(zRank, zShape, i, zCoord);

                    Nd4jIndex xOffset2 = shape::getOffset(xOffset, xShape, xStride, xCoord, xRank);
                    Nd4jIndex zOffset2 = shape::getOffset(zOffset, zShape, zStride, zCoord, zRank);

                    z[zOffset2] = OpClass::op(x[xOffset2], i, length, buffer, extraArguments);
                }
            }

            // update rng state
            buffer->rewindH(length);
        }


        template<typename T>
        template<typename OpClass>
        void RandomFunction<T>::execTransform(Nd4jPointer state, T *z, int *zShapeBuffer, T *extraArguments) {
            Nd4jIndex length = shape::length(zShapeBuffer);
            int ews = shape::elementWiseStride(zShapeBuffer);

            nd4j::random::RandomBuffer *buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (state);

            Nd4jIndex elementsPerThread = length / ELEMENT_THRESHOLD;
            Nd4jIndex _threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
            _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

            if (ews >= 1) {
                if (ews == 1) {
#pragma omp parallel for num_threads(_threads) if (_threads > 1) schedule(guided)
                    for (Nd4jIndex x = 0; x < length; x++) {
                        z[x] = OpClass::op(x, length, buffer, extraArguments);
                    }

                } else {
#pragma omp parallel for num_threads(_threads) if (_threads > 1) schedule(guided)
                    for (Nd4jIndex x = 0; x < length; x++) {
                        z[x * ews] = OpClass::op(x, length, buffer, extraArguments);
                    }
                }
            } else {
                // ind2sub branch
                int zCoord[MAX_RANK];

                int zRank = shape::rank(zShapeBuffer);
                int *zShape = shape::shapeOf(zShapeBuffer);
                int *zStride = shape::stride(zShapeBuffer);
                int zOffset = shape::offset(zShapeBuffer);

#pragma omp parallel for num_threads(_threads) if (_threads > 1) schedule(guided) private(zCoord)
                for (Nd4jIndex i = 0; i < length; i++) {
                    shape::ind2sub(zRank, zShape, i, zCoord);
                    Nd4jIndex zOffset2 = shape::getOffset(zOffset, zShape, zStride, zCoord, zRank);
                    z[zOffset2] = OpClass::op(i, length, buffer,  extraArguments);
                }
            }

            // update rng state
            buffer->rewindH(length);
        }

        template<typename T>
        void RandomFunction<T>::execTransform(int opNum, Nd4jPointer state, T *x, int *xShapeBuffer, T *z, int *zShapeBuffer, T *extraArguments) {
            DISPATCH_BY_OPNUM(execTransform, PARAMS(state, x, xShapeBuffer, z, zShapeBuffer, extraArguments), RANDOM_OPS)
        }

        template<typename T>
        void RandomFunction<T>::execTransform(int opNum, Nd4jPointer state, T *x, int *xShapeBuffer, T *y, int *yShapeBuffer, T *z, int *zShapeBuffer, T *extraArguments) {
            DISPATCH_BY_OPNUM(execTransform, PARAMS(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments), RANDOM_OPS)
        }

        template<typename T>
        void RandomFunction<T>::execTransform(int opNum, Nd4jPointer state, T *z, int *zShapeBuffer, T *extraArguments) {
            DISPATCH_BY_OPNUM(execTransform, PARAMS(state, z, zShapeBuffer, extraArguments), RANDOM_OPS)
        }

        // FIXME: eventually we might want to get rid of that
#ifndef __CLION_IDE__
        BUILD_CALL_1(template void RandomFunction<float>::execTransform, float, (Nd4jPointer state, float *x, int *xShapeBuffer, float *y, int *yShapeBuffer, float *z, int *zShapeBuffer, float *extraArguments), RANDOM_OPS)
        BUILD_CALL_1(template void RandomFunction<float16>::execTransform, float16, (Nd4jPointer state, float16 *x, int *xShapeBuffer, float16 *y, int *yShapeBuffer, float16 *z, int *zShapeBuffer, float16 *extraArguments), RANDOM_OPS)
        BUILD_CALL_1(template void RandomFunction<double>::execTransform, double, (Nd4jPointer state, double *x, int *xShapeBuffer, double *y, int *yShapeBuffer, double *z, int *zShapeBuffer, double *extraArguments), RANDOM_OPS)

        BUILD_CALL_1(template void RandomFunction<float>::execTransform, float, (Nd4jPointer state, float *x, int *xShapeBuffer, float *z, int *zShapeBuffer, float *extraArguments), RANDOM_OPS)
        BUILD_CALL_1(template void RandomFunction<float16>::execTransform, float16, (Nd4jPointer state, float16 *x, int *xShapeBuffer, float16 *z, int *zShapeBuffer, float16 *extraArguments), RANDOM_OPS)
        BUILD_CALL_1(template void RandomFunction<double>::execTransform, double, (Nd4jPointer state, double *x, int *xShapeBuffer, double *z, int *zShapeBuffer, double *extraArguments), RANDOM_OPS)

        BUILD_CALL_1(template void RandomFunction<float>::execTransform, float, (Nd4jPointer state, float *z, int *zShapeBuffer, float *extraArguments), RANDOM_OPS)
        BUILD_CALL_1(template void RandomFunction<float16>::execTransform, float16, (Nd4jPointer state, float16 *z, int *zShapeBuffer, float16 *extraArguments), RANDOM_OPS)
        BUILD_CALL_1(template void RandomFunction<double>::execTransform, double, (Nd4jPointer state, double *z, int *zShapeBuffer, double *extraArguments), RANDOM_OPS)
#endif

        template class ND4J_EXPORT RandomFunction<float>;
        template class ND4J_EXPORT RandomFunction<float16>;
        template class ND4J_EXPORT RandomFunction<double>;
    }
}