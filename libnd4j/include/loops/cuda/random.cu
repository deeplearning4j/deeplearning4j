//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#include <loops/random.h>
#include <dll.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_launch_config.h>
#include <helpers/DebugHelper.h>

template <typename T, typename OpClass>
static inline __device__ void randomSingleGeneric(
        Nd4jPointer state,
        T *z,
        Nd4jLong *zShapeBuffer,
        T *extraArguments) {


    functions::random::RandomFunction<T>::template execTransformCuda<OpClass>(
            state,
            z,
            zShapeBuffer,
            extraArguments);
}

template <typename T, typename OpClass>
static inline __device__ void randomDoubleGeneric(
        Nd4jPointer state,
        T *x,
        Nd4jLong *xShapeBuffer,
        T *z,
        Nd4jLong *zShapeBuffer,
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
static inline __device__ void randomTripleGeneric(
        Nd4jPointer state,
        T *x,
        Nd4jLong *xShapeBuffer,
        T *y,
        Nd4jLong *yShapeBuffer,
        T *z,
        Nd4jLong *zShapeBuffer,
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


#ifndef __CLION_IDE__
// here we generate kernels for target operations
DISPATCH_KERNEL_SIMPLE(randomSingle_, randomSingleGeneric, float, INPUT(Nd4jPointer state, float *z, Nd4jLong *zShapeBuffer, float *extraArguments), PARAMS(state, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomSingle_, randomSingleGeneric, double, INPUT(Nd4jPointer state, double *z, Nd4jLong *zShapeBuffer, double *extraArguments), PARAMS(state, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomSingle_, randomSingleGeneric, float16, INPUT(Nd4jPointer state, float16 *z, Nd4jLong *zShapeBuffer, float16 *extraArguments), PARAMS(state, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

DISPATCH_KERNEL_SIMPLE(randomDouble_, randomDoubleGeneric, float, INPUT(Nd4jPointer state, float *x, Nd4jLong *xShapeBuffer, float *z, Nd4jLong *zShapeBuffer, float *extraArguments), PARAMS(state, x, xShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomDouble_, randomDoubleGeneric, double, INPUT(Nd4jPointer state, double *x, Nd4jLong *xShapeBuffer, double *z, Nd4jLong *zShapeBuffer, double *extraArguments), PARAMS(state, x, xShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomDouble_, randomDoubleGeneric, float16, INPUT(Nd4jPointer state, float16 *x, Nd4jLong *xShapeBuffer, float16 *z, Nd4jLong *zShapeBuffer, float16 *extraArguments), PARAMS(state, x, xShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

DISPATCH_KERNEL_SIMPLE(randomTriple_, randomTripleGeneric, float, INPUT(Nd4jPointer state, float *x, Nd4jLong *xShapeBuffer, float *y, Nd4jLong *yShapeBuffer, float *z, Nd4jLong *zShapeBuffer, float *extraArguments), PARAMS(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomTriple_, randomTripleGeneric, double, INPUT(Nd4jPointer state, double *x, Nd4jLong *xShapeBuffer, double *y, Nd4jLong *yShapeBuffer, double *z, Nd4jLong *zShapeBuffer, double *extraArguments), PARAMS(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomTriple_, randomTripleGeneric, float16, INPUT(Nd4jPointer state, float16 *x, Nd4jLong *xShapeBuffer, float16 *y, Nd4jLong *yShapeBuffer, float16 *z, Nd4jLong *zShapeBuffer, float16 *extraArguments), PARAMS(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

#endif

namespace functions {
    namespace random {
            template<typename T>
            template<typename OpClass>
            void _CUDA_D RandomFunction<T>::execTransformCuda(Nd4jPointer state, T *x, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *z, Nd4jLong *zShapeBuffer, T *extraArguments) {
                if (OpClass::requiresSpecial) {
                    OpClass::specialOpCuda(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments);
                    return;
                } else {

                __shared__ Nd4jLong length;
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
                    for (Nd4jLong e = tid; e < length; e += blockDim.x * gridDim.x) {
                        z[e * zEWS] = OpClass::op(x[e * xEWS], y[e * yEWS], e, length, buffer, extraArguments);
                    }
                } else {
                    // negative ews
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

                    for (Nd4jLong i = tid; i < length; i += blockDim.x * gridDim.x) {
                        shape::ind2sub(xRank, xShape, i, xCoord);
                        shape::ind2sub(yRank, yShape, i, yCoord);
                        shape::ind2sub(zRank, zShape, i, zCoord);

                        auto xOffset2 = shape::getOffset(0, xShape, xStride, xCoord, xRank);
                        auto yOffset2 = shape::getOffset(0, yShape, yStride, yCoord, yRank);
                        auto zOffset2 = shape::getOffset(0, zShape, zStride, zCoord, zRank);

                        z[zOffset2] = OpClass::op(x[xOffset2], y[yOffset2], i, length, buffer, extraArguments);
                    }
                }

                __syncthreads();
                devBuffer->rewind(length);
                }
            };


            template<typename T>
            template<typename OpClass>
            void _CUDA_D RandomFunction<T>::execTransformCuda(Nd4jPointer state, T *x, Nd4jLong *xShapeBuffer, T *z, Nd4jLong *zShapeBuffer, T *extraArguments) {
                __shared__ Nd4jLong length;
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
                    for (Nd4jLong e = blockIdx.x * blockDim.x + threadIdx.x; e < length; e += blockDim.x * gridDim.x) {
                        z[e * zEWS] = OpClass::op(x[e * xEWS], e, length, buffer, extraArguments);
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

                    for (Nd4jLong i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += blockDim.x * gridDim.x) {
                        shape::ind2sub(xRank, xShape, i, xCoord);
                        shape::ind2sub(zRank, zShape, i, zCoord);

                        auto xOffset2 = shape::getOffset(0, xShape, xStride, xCoord, xRank);
                        auto zOffset2 = shape::getOffset(0, zShape, zStride, zCoord, zRank);

                        z[zOffset2] = OpClass::op(x[xOffset2], i, length, buffer, extraArguments);
                    }
                }

                __syncthreads();
                devBuffer->rewind(length);
            }


            template<typename T>
            template<typename OpClass>
            void _CUDA_D RandomFunction<T>::execTransformCuda(Nd4jPointer state, T *z, Nd4jLong *zShapeBuffer, T *extraArguments) {
                Nd4jLong length = shape::length(zShapeBuffer);
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
                    for (Nd4jLong x = tid; x < length; x += blockDim.x * gridDim.x) {
                        z[x * ews] = OpClass::op(x, length, buffer, extraArguments);
                    }
                } else {
                    // ind2sub branch
                    Nd4jLong zCoord[MAX_RANK];

                    int zRank = shape::rank(zShapeBuffer);
                    auto zShape = shape::shapeOf(zShapeBuffer);
                    auto zStride = shape::stride(zShapeBuffer);

                    for (Nd4jLong i = tid; i < length; i += blockDim.x * gridDim.x) {
                        shape::ind2sub(zRank, zShape, i, zCoord);

                        auto zOffset2 = shape::getOffset(0, zShape, zStride, zCoord, zRank);

                        z[zOffset2] = OpClass::op(i, length, buffer,  extraArguments);
                    }
                }

                __syncthreads();
                devBuffer->rewind(length);
            }

        template <>
        _CUDA_H void RandomFunction<float>::executeCudaSingle(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, Nd4jPointer stateHost, float *z, Nd4jLong *zShapeBuffer, float *extraArguments) {
            cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

            nd4j::random::RandomBuffer *buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (stateHost);
            Nd4jPointer state = buffer->getDevicePointer();

            // this macro builds bunch of IF/ELSE selectors for kernel launch
            DISPATCH_SIMPLE(randomSingle, float, PARAMS(state, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

            DEBUG_KERNEL(stream, opNum);
        }

        template <>
        _CUDA_H void RandomFunction<float16>::executeCudaSingle(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, Nd4jPointer stateHost, float16 *z, Nd4jLong *zShapeBuffer, float16 *extraArguments) {
            cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

            nd4j::random::RandomBuffer *buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (stateHost);
            Nd4jPointer state = buffer->getDevicePointer();

            // this macro builds bunch of IF/ELSE selectors for kernel launch
            DISPATCH_SIMPLE(randomSingle, float16, PARAMS(state, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

            DEBUG_KERNEL(stream, opNum);
        }

        template <>
        _CUDA_H void RandomFunction<double>::executeCudaSingle(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, Nd4jPointer stateHost, double *z, Nd4jLong *zShapeBuffer, double *extraArguments) {
            cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

            nd4j::random::RandomBuffer *buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (stateHost);
            Nd4jPointer state = buffer->getDevicePointer();

            // this macro builds bunch of IF/ELSE selectors for kernel launch
            DISPATCH_SIMPLE(randomSingle, double, PARAMS(state, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

            DEBUG_KERNEL(stream, opNum);
        }

        template <>
        _CUDA_H void RandomFunction<float>::executeCudaDouble(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, Nd4jPointer stateHost, float *x, Nd4jLong *xShapeBuffer, float *z, Nd4jLong *zShapeBuffer, float *extraArguments) {
            cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

            nd4j::random::RandomBuffer *buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (stateHost);
            Nd4jPointer state = buffer->getDevicePointer();

            // this macro builds bunch of IF/ELSE selectors for kernel launch
            DISPATCH_SIMPLE(randomDouble, float, PARAMS(state, x, xShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

            DEBUG_KERNEL(stream, opNum);
        }


        template <>
        _CUDA_H void RandomFunction<float16>::executeCudaDouble(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, Nd4jPointer stateHost, float16 *x, Nd4jLong *xShapeBuffer, float16 *z, Nd4jLong *zShapeBuffer, float16 *extraArguments) {
            cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

            nd4j::random::RandomBuffer *buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (stateHost);
            Nd4jPointer state = buffer->getDevicePointer();

            // this macro builds bunch of IF/ELSE selectors for kernel launch
            DISPATCH_SIMPLE(randomDouble, float16, PARAMS(state, x, xShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

            DEBUG_KERNEL(stream, opNum);
        }

        template <>
        _CUDA_H void RandomFunction<double>::executeCudaDouble(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, Nd4jPointer stateHost, double *x, Nd4jLong *xShapeBuffer, double *z, Nd4jLong *zShapeBuffer, double *extraArguments) {
            cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

            nd4j::random::RandomBuffer *buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (stateHost);
            Nd4jPointer state = buffer->getDevicePointer();

            // this macro builds bunch of IF/ELSE selectors for kernel launch
            DISPATCH_SIMPLE(randomDouble, double, PARAMS(state, x, xShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

            DEBUG_KERNEL(stream, opNum);
        }

        template <>
        _CUDA_H void RandomFunction<float>::executeCudaTriple(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, Nd4jPointer stateHost, float *x, Nd4jLong *xShapeBuffer, float *y, Nd4jLong *yShapeBuffer, float *z, Nd4jLong *zShapeBuffer, float *extraArguments) {
            cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

            nd4j::random::RandomBuffer *buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (stateHost);
            Nd4jPointer state = buffer->getDevicePointer();

            // this macro builds bunch of IF/ELSE selectors for kernel launch
            DISPATCH_SIMPLE(randomTriple, float, PARAMS(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

            DEBUG_KERNEL(stream, opNum);
        }

        template <>
        _CUDA_H void RandomFunction<float16>::executeCudaTriple(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, Nd4jPointer stateHost, float16 *x, Nd4jLong *xShapeBuffer, float16 *y, Nd4jLong *yShapeBuffer, float16 *z, Nd4jLong *zShapeBuffer, float16 *extraArguments) {
            cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

            nd4j::random::RandomBuffer *buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (stateHost);
            Nd4jPointer state = buffer->getDevicePointer();

            // this macro builds bunch of IF/ELSE selectors for kernel launch
            DISPATCH_SIMPLE(randomTriple, float16, PARAMS(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

            DEBUG_KERNEL(stream, opNum);
        }



        template <>
        _CUDA_H void RandomFunction<double>::executeCudaTriple(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, Nd4jPointer stateHost, double *x, Nd4jLong *xShapeBuffer, double *y, Nd4jLong *yShapeBuffer, double *z, Nd4jLong *zShapeBuffer, double *extraArguments) {
            cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

            nd4j::random::RandomBuffer *buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (stateHost);
            Nd4jPointer state = buffer->getDevicePointer();

            // this macro builds bunch of IF/ELSE selectors for kernel launch
            DISPATCH_SIMPLE(randomTriple, double, PARAMS(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

            DEBUG_KERNEL(stream, opNum);
        }



        template class ND4J_EXPORT RandomFunction<float>;
        template class ND4J_EXPORT RandomFunction<float16>;
        template class ND4J_EXPORT RandomFunction<double>;
    }
}
