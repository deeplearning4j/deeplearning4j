//
// @author raver119@gmail.com
//

#ifndef LIBND4J_RANDOM_H
#define LIBND4J_RANDOM_H

#define RANDOM_OPS \
        (0, randomOps::BoundedDistribution) ,\
        (1, randomOps::DropOut) ,\
        (2, randomOps::DropOutInverted) ,\
        (3, randomOps::ProbablisticMerge)

#include <shape.h>
#include <helpers/helper_random.h>
#include <ops/random_ops.h>



namespace functions {
    namespace random {

        template<typename T>
        class RandomFunction {
        public:

            template<typename OpClass>
            static inline void execTransform(Nd4jPointer state, T *x, int *xShapeBuffer, T *y, int *yShapeBuffer, T *z, int *zShapeBuffer, T *extraArguments) {
                int length = shape::length(zShapeBuffer);
                int xEWS = shape::elementWiseStride(xShapeBuffer);
                int yEWS = shape::elementWiseStride(yShapeBuffer);
                int zEWS = shape::elementWiseStride(zShapeBuffer);

                int elementsPerThread = length / ELEMENT_THRESHOLD;
                int _threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
                _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

                if (xEWS >= 1 && yEWS >= 1 && zEWS >= 1) {
                    if (xEWS == 1 && yEWS == 1 && zEWS == 1) {
#pragma omp parallel for num_threads(_threads) if (_threads > 1) schedule(guided)
                        for (int e = 0; e < length; e++) {
                            z[e] = OpClass::op(x[e], y[e], e, nullptr, extraArguments);
                        }

                    } else {
#pragma omp parallel for num_threads(_threads) if (_threads > 1) schedule(guided)
                        for (int e = 0; e < length; e++) {
                            z[e * zEWS] = OpClass::op(x[e * xEWS], y[e * yEWS], e, nullptr, extraArguments);
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
                    for (int i = 0; i < length; i++) {
                        shape::ind2sub(xRank, xShape, i, xCoord);
                        shape::ind2sub(yRank, yShape, i, yCoord);
                        shape::ind2sub(zRank, zShape, i, zCoord);

                        Nd4jIndex xOffset2 = shape::getOffset(xOffset, xShape, xStride, xCoord, xRank);
                        Nd4jIndex yOffset2 = shape::getOffset(yOffset, yShape, yStride, yCoord, yRank);
                        Nd4jIndex zOffset2 = shape::getOffset(zOffset, zShape, zStride, zCoord, zRank);


                        z[zOffset2] = OpClass::op(x[xOffset2], y[yOffset2], i, nullptr, extraArguments);
                    }
                }
            }


            template<typename OpClass>
            static inline void execTransform(Nd4jPointer state, T *x, int *xShapeBuffer, T *z, int *zShapeBuffer, T *extraArguments) {
                int length = shape::length(zShapeBuffer);
                int xEWS = shape::elementWiseStride(xShapeBuffer);
                int zEWS = shape::elementWiseStride(zShapeBuffer);

                int elementsPerThread = length / ELEMENT_THRESHOLD;
                int _threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
                _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

                if (xEWS >= 1 && zEWS >= 1) {
                    if (xEWS == 1 && zEWS == 1) {
#pragma omp parallel for num_threads(_threads) if (_threads > 1) schedule(guided)
                        for (int e = 0; e < length; e++) {
                            z[e] = OpClass::op(x[e], e, nullptr, extraArguments);
                        }

                    } else {
#pragma omp parallel for num_threads(_threads) if (_threads > 1) schedule(guided)
                        for (int e = 0; e < length; e++) {
                            z[e * zEWS] = OpClass::op(x[e * xEWS], e, nullptr, extraArguments);
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
                    for (int i = 0; i < length; i++) {
                        shape::ind2sub(xRank, xShape, i, xCoord);
                        shape::ind2sub(zRank, zShape, i, zCoord);

                        Nd4jIndex xOffset2 = shape::getOffset(xOffset, xShape, xStride, xCoord, xRank);
                        Nd4jIndex zOffset2 = shape::getOffset(zOffset, zShape, zStride, zCoord, zRank);

                        z[zOffset2] = OpClass::op(x[xOffset2], i, nullptr, extraArguments);
                    }
                }
            }

            template<typename OpClass>
            static inline void execTransform(Nd4jPointer state, T *z, int *zShapeBuffer, T *extraArguments) {
                int length = shape::length(zShapeBuffer);
                int ews = shape::elementWiseStride(zShapeBuffer);

                int elementsPerThread = length / ELEMENT_THRESHOLD;
                int _threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
                _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

                if (ews >= 1) {
                    if (ews == 1) {
#pragma omp parallel for num_threads(_threads) if (_threads > 1) schedule(guided)
                        for (int x = 0; x < length; x++) {
                            z[x] = OpClass::op(z[x], x, nullptr, extraArguments);
                        }

                    } else {
#pragma omp parallel for num_threads(_threads) if (_threads > 1) schedule(guided)
                        for (int x = 0; x < length; x++) {
                            z[x * ews] = OpClass::op(z[x * ews], x, nullptr, extraArguments);
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
                    for (int i = 0; i < length; i++) {
                        shape::ind2sub(zRank, zShape, i, zCoord);
                        Nd4jIndex zOffset2 = shape::getOffset(zOffset, zShape, zStride, zCoord, zRank);
                        z[zOffset2] = OpClass::op(z[zOffset2], i, nullptr,  extraArguments);
                    }
                }
            }

            static inline void execTransform(int opNum, Nd4jPointer state, T *x, int *xShapeBuffer, T *z, int *zShapeBuffer, T *extraArguments) {
                DISPATCH_BY_OPNUM(execTransform, PARAMS(state, x, xShapeBuffer, z, zShapeBuffer, extraArguments), RANDOM_OPS)
            }

            static inline void execTransform(int opNum, Nd4jPointer state, T *x, int *xShapeBuffer, T *y, int *yShapeBuffer, T *z, int *zShapeBuffer, T *extraArguments) {
                DISPATCH_BY_OPNUM(execTransform, PARAMS(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments), RANDOM_OPS)
            }

            static inline void execTransform(int opNum, Nd4jPointer state, T *z, int *zShapeBuffer, T *extraArguments) {
                DISPATCH_BY_OPNUM(execTransform, PARAMS(state, z, zShapeBuffer, extraArguments), RANDOM_OPS)
            }
        };
    }
}

#endif //LIBND4J_RANDOM_H
