//
// @author raver119@gmail.com
//

#ifndef LIBND4J_RANDOM_H
#define LIBND4J_RANDOM_H



#include <helpers/shape.h>
#include <helpers/helper_random.h>
#include <ops/random_ops.h>
#include <ops/special_random_ops.h>

#include "legacy_ops.h"


namespace functions {
    namespace random {

        template<typename T>
        class RandomFunction {
        public:


#ifdef __CUDACC__
            template<typename OpClass>
            __device__ static inline void execTransformCuda(Nd4jPointer state, T *x, int *xShapeBuffer, T *y, int *yShapeBuffer, T *z, int *zShapeBuffer, T *extraArguments) {
                if (OpClass::requiresSpecial) {
                    OpClass::specialOpCuda(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments);
                    return;
                } else {

                __shared__ Nd4jIndex length;
                __shared__ int xEWS;
                __shared__ int yEWS;
                __shared__ int zEWS;

                __shared__ nd4j::random::RandomBuffer *buffer;
                __shared__ unsigned char *cB;
                __shared__ unsigned char *dB;
                __shared__ nd4j::random::RandomBuffer *devBuffer;
                if (threadIdx.x == 0) {
                    length = shape::length(zShapeBuffer);
                    xEWS = shape::elementWiseStride(xShapeBuffer);
                    yEWS = shape::elementWiseStride(yShapeBuffer);
                    zEWS = shape::elementWiseStride(zShapeBuffer);

                    extern __shared__ unsigned char shmem[];
                    buffer = (nd4j::random::RandomBuffer *) shmem;
                    cB = shmem;
                    devBuffer = reinterpret_cast<nd4j::random::RandomBuffer *> (state);
                    dB = reinterpret_cast<unsigned char *> (state);
                }
                __syncthreads();

                // using this loop instead of memcpy
                for (int e = threadIdx.x; e < sizeof(nd4j::random::RandomBuffer); e+= blockDim.x) {
                    cB[e] = dB[e];
                }
                __syncthreads();


                int tid = blockIdx.x * blockDim.x + threadIdx.x;

                if (xEWS >= 1 && yEWS >= 1 && zEWS >= 1) {
                    for (Nd4jIndex e = tid; e < length; e += blockDim.x * gridDim.x) {
                        z[e * zEWS] = OpClass::op(x[e * xEWS], y[e * yEWS], e, length, buffer, extraArguments);
                    }
                } else {
                    // negative ews
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

                    for (Nd4jIndex i = tid; i < length; i += blockDim.x * gridDim.x) {
                        shape::ind2sub(xRank, xShape, i, xCoord);
                        shape::ind2sub(yRank, yShape, i, yCoord);
                        shape::ind2sub(zRank, zShape, i, zCoord);

                        Nd4jIndex xOffset2 = shape::getOffset(xOffset, xShape, xStride, xCoord, xRank);
                        Nd4jIndex yOffset2 = shape::getOffset(yOffset, yShape, yStride, yCoord, yRank);
                        Nd4jIndex zOffset2 = shape::getOffset(zOffset, zShape, zStride, zCoord, zRank);


                        z[zOffset2] = OpClass::op(x[xOffset2], y[yOffset2], i, length, buffer, extraArguments);
                    }
                }

                __syncthreads();
                devBuffer->rewind(length);
                }
            }
#endif

            template<typename OpClass>
            static inline void execTransform(Nd4jPointer state, T *x, int *xShapeBuffer, T *y, int *yShapeBuffer, T *z, int *zShapeBuffer, T *extraArguments) {

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
            }


#ifdef __CUDACC__
            template<typename OpClass>
            __device__ static inline void execTransformCuda(Nd4jPointer state, T *x, int *xShapeBuffer, T *z, int *zShapeBuffer, T *extraArguments) {
                __shared__ Nd4jIndex length;
                __shared__ int xEWS;
                __shared__ int zEWS;

                __shared__ nd4j::random::RandomBuffer *buffer;
                __shared__ unsigned char *cB;
                __shared__ unsigned char *dB;
                __shared__ nd4j::random::RandomBuffer *devBuffer;
                if (threadIdx.x == 0) {
                    extern __shared__ unsigned char shmem[];
                    buffer = (nd4j::random::RandomBuffer *) shmem;
                    cB = shmem;
                    devBuffer = reinterpret_cast<nd4j::random::RandomBuffer *> (state);
                    dB = reinterpret_cast<unsigned char *> (state);

                    length = shape::length(zShapeBuffer);
                    xEWS = shape::elementWiseStride(xShapeBuffer);
                    zEWS = shape::elementWiseStride(zShapeBuffer);
                }
                __syncthreads();

                // using this loop instead of memcpy
                for (int e = threadIdx.x; e < sizeof(nd4j::random::RandomBuffer); e+= blockDim.x) {
                    cB[e] = dB[e];
                }
                __syncthreads();


                if (xEWS >= 1 && zEWS >= 1) {
                    for (Nd4jIndex e = blockIdx.x * blockDim.x + threadIdx.x; e < length; e += blockDim.x * gridDim.x) {
                        z[e * zEWS] = OpClass::op(x[e * xEWS], e, length, buffer, extraArguments);
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

                    for (Nd4jIndex i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += blockDim.x * gridDim.x) {
                        shape::ind2sub(xRank, xShape, i, xCoord);
                        shape::ind2sub(zRank, zShape, i, zCoord);

                        Nd4jIndex xOffset2 = shape::getOffset(xOffset, xShape, xStride, xCoord, xRank);
                        Nd4jIndex zOffset2 = shape::getOffset(zOffset, zShape, zStride, zCoord, zRank);

                        z[zOffset2] = OpClass::op(x[xOffset2], i, length, buffer, extraArguments);
                    }
                }

                __syncthreads();
                devBuffer->rewind(length);
            }
#endif


            template<typename OpClass>
            static inline void execTransform(Nd4jPointer state, T *x, int *xShapeBuffer, T *z, int *zShapeBuffer, T *extraArguments) {
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

#ifdef __CUDACC__
            template<typename OpClass>
            __device__ static inline void execTransformCuda(Nd4jPointer state, T *z, int *zShapeBuffer, T *extraArguments) {
                Nd4jIndex length = shape::length(zShapeBuffer);
                int ews = shape::elementWiseStride(zShapeBuffer);

                __shared__ nd4j::random::RandomBuffer *buffer;
                __shared__ unsigned char *cB;
                __shared__ unsigned char *dB;
                __shared__ nd4j::random::RandomBuffer *devBuffer;
                if (threadIdx.x == 0) {
                    extern __shared__ unsigned char shmem[];
                    buffer = (nd4j::random::RandomBuffer *) shmem;
                    cB = shmem;
                    devBuffer = reinterpret_cast<nd4j::random::RandomBuffer *> (state);
                    dB = reinterpret_cast<unsigned char *> (state);
                }
                __syncthreads();

                // using this loop instead of memcpy
                for (int e = threadIdx.x; e < sizeof(nd4j::random::RandomBuffer); e+= blockDim.x) {
                    cB[e] = dB[e];
                }
                __syncthreads();

                int tid = blockIdx.x * blockDim.x + threadIdx.x;

                if (ews >= 1) {
                    for (Nd4jIndex x = tid; x < length; x += blockDim.x * gridDim.x) {
                        z[x * ews] = OpClass::op(x, length, buffer, extraArguments);
                    }
                } else {
                    // ind2sub branch
                    int zCoord[MAX_RANK];

                    int zRank = shape::rank(zShapeBuffer);
                    int *zShape = shape::shapeOf(zShapeBuffer);
                    int *zStride = shape::stride(zShapeBuffer);
                    int zOffset = shape::offset(zShapeBuffer);

                    for (Nd4jIndex i = tid; i < length; i += blockDim.x * gridDim.x) {
                        shape::ind2sub(zRank, zShape, i, zCoord);
                        Nd4jIndex zOffset2 = shape::getOffset(zOffset, zShape, zStride, zCoord, zRank);
                        z[zOffset2] = OpClass::op(i, length, buffer,  extraArguments);
                    }
                }

                __syncthreads();
                devBuffer->rewind(length);
            }
#endif

            template<typename OpClass>
            static inline void execTransform(Nd4jPointer state, T *z, int *zShapeBuffer, T *extraArguments) {
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

#ifdef __CUDACC__

template <typename T, typename OpClass>
__device__ void randomSingleGeneric(
        Nd4jPointer state,
        T *z,
        int *zShapeBuffer,
        T *extraArguments) {


    functions::random::RandomFunction<T>::template execTransformCuda<OpClass>(
            state,
            z,
            zShapeBuffer,
            extraArguments);
}

template <typename T, typename OpClass>
__device__ void randomDoubleGeneric(
        Nd4jPointer state,
        T *x,
        int *xShapeBuffer,
        T *z,
        int *zShapeBuffer,
        T *extraArguments) {


    functions::random::RandomFunction<T>::template execTransformCuda<OpClass>(
            state,
            x,
            xShapeBuffer,
            z,
            zShapeBuffer,
            extraArguments);
}


template <typename T, typename OpClass>
__device__ void randomTripleGeneric(
        Nd4jPointer state,
        T *x,
        int *xShapeBuffer,
        T *y,
        int *yShapeBuffer,
        T *z,
        int *zShapeBuffer,
        T *extraArguments) {


    functions::random::RandomFunction<T>::template execTransformCuda<OpClass>(
            state,
            x,
            xShapeBuffer,
            y,
            yShapeBuffer,
            z,
            zShapeBuffer,
            extraArguments);
}


// here we generate kernels for target operations
DISPATCH_KERNEL_SIMPLE(randomSingle_, randomSingleGeneric, float, INPUT(Nd4jPointer state, float *z, int *zShapeBuffer, float *extraArguments), PARAMS(state, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomSingle_, randomSingleGeneric, double, INPUT(Nd4jPointer state, double *z, int *zShapeBuffer, double *extraArguments), PARAMS(state, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomSingle_, randomSingleGeneric, float16, INPUT(Nd4jPointer state, float16 *z, int *zShapeBuffer, float16 *extraArguments), PARAMS(state, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

DISPATCH_KERNEL_SIMPLE(randomDouble_, randomDoubleGeneric, float, INPUT(Nd4jPointer state, float *x, int *xShapeBuffer, float *z, int *zShapeBuffer, float *extraArguments), PARAMS(state, x, xShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomDouble_, randomDoubleGeneric, double, INPUT(Nd4jPointer state, double *x, int *xShapeBuffer, double *z, int *zShapeBuffer, double *extraArguments), PARAMS(state, x, xShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomDouble_, randomDoubleGeneric, float16, INPUT(Nd4jPointer state, float16 *x, int *xShapeBuffer, float16 *z, int *zShapeBuffer, float16 *extraArguments), PARAMS(state, x, xShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

DISPATCH_KERNEL_SIMPLE(randomTriple_, randomTripleGeneric, float, INPUT(Nd4jPointer state, float *x, int *xShapeBuffer, float *y, int *yShapeBuffer, float *z, int *zShapeBuffer, float *extraArguments), PARAMS(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomTriple_, randomTripleGeneric, double, INPUT(Nd4jPointer state, double *x, int *xShapeBuffer, double *y, int *yShapeBuffer, double *z, int *zShapeBuffer, double *extraArguments), PARAMS(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomTriple_, randomTripleGeneric, float16, INPUT(Nd4jPointer state, float16 *x, int *xShapeBuffer, float16 *y, int *yShapeBuffer, float16 *z, int *zShapeBuffer, float16 *extraArguments), PARAMS(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

#endif

#endif //LIBND4J_RANDOM_H
