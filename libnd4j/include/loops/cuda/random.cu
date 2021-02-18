/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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

#include <system/op_boilerplate.h>
#include <loops/random.h>
#include <system/dll.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helpers/DebugHelper.h>
#include <ops/specials_cuda.h>

using namespace randomOps;

template <typename T, typename OpClass>
static inline __device__ void randomSingleGeneric(
        Nd4jPointer state,
        void *z,
        Nd4jLong const* zShapeBuffer,
        void *extraArguments) {


    functions::random::RandomFunction<T>::template execTransformCuda<OpClass>(
            state,
            z,
            zShapeBuffer,
            extraArguments);
}

template <typename T, typename OpClass>
static inline __device__ void randomDoubleGeneric(
        Nd4jPointer state,
        void const* x,
        Nd4jLong const* xShapeBuffer,
        void *z,
        Nd4jLong const* zShapeBuffer,
        void *extraArguments) {


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
        void const* x,
        Nd4jLong const* xShapeBuffer,
        void const* y,
        Nd4jLong const* yShapeBuffer,
        void *z,
        Nd4jLong const* zShapeBuffer,
        void *extraArguments) {


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
DISPATCH_KERNEL_SIMPLE(randomSingle_, randomSingleGeneric, float, INPUT(Nd4jPointer state, void *z, Nd4jLong const* zShapeBuffer, void *extraArguments), PARAMS(state, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomSingle_, randomSingleGeneric, double, INPUT(Nd4jPointer state, void *z, Nd4jLong const* zShapeBuffer, void *extraArguments), PARAMS(state, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomSingle_, randomSingleGeneric, float16, INPUT(Nd4jPointer state, void *z, Nd4jLong const* zShapeBuffer, void *extraArguments), PARAMS(state, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomSingle_, randomSingleGeneric, bfloat16, INPUT(Nd4jPointer state, void *z, Nd4jLong const* zShapeBuffer, void *extraArguments), PARAMS(state, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

DISPATCH_KERNEL_SIMPLE(randomDouble_, randomDoubleGeneric, float, INPUT(Nd4jPointer state, void const* x, Nd4jLong const* xShapeBuffer, void *z, Nd4jLong const* zShapeBuffer, void *extraArguments), PARAMS(state, x, xShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomDouble_, randomDoubleGeneric, double, INPUT(Nd4jPointer state, void const* x, Nd4jLong const* xShapeBuffer, void *z, Nd4jLong const* zShapeBuffer, void *extraArguments), PARAMS(state, x, xShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomDouble_, randomDoubleGeneric, float16, INPUT(Nd4jPointer state, void const* x, Nd4jLong const* xShapeBuffer, void *z, Nd4jLong const* zShapeBuffer, void *extraArguments), PARAMS(state, x, xShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomDouble_, randomDoubleGeneric, bfloat16, INPUT(Nd4jPointer state, void const* x, Nd4jLong const* xShapeBuffer, void *z, Nd4jLong const* zShapeBuffer, void *extraArguments), PARAMS(state, x, xShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

DISPATCH_KERNEL_SIMPLE(randomTriple_, randomTripleGeneric, float, INPUT(Nd4jPointer state, void const* x, Nd4jLong const* xShapeBuffer, void const* y, Nd4jLong const* yShapeBuffer, void *z, Nd4jLong const* zShapeBuffer, void *extraArguments), PARAMS(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomTriple_, randomTripleGeneric, double, INPUT(Nd4jPointer state, void const* x, Nd4jLong const* xShapeBuffer, void const* y, Nd4jLong const* yShapeBuffer, void *z, Nd4jLong const* zShapeBuffer, void *extraArguments), PARAMS(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomTriple_, randomTripleGeneric, float16, INPUT(Nd4jPointer state, void const* x, Nd4jLong const* xShapeBuffer, void const* y, Nd4jLong const* yShapeBuffer, void *z, Nd4jLong const* zShapeBuffer, void *extraArguments), PARAMS(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomTriple_, randomTripleGeneric, bfloat16, INPUT(Nd4jPointer state, void const* x, Nd4jLong const* xShapeBuffer, void const* y, Nd4jLong const* yShapeBuffer, void *z, Nd4jLong const* zShapeBuffer, void *extraArguments), PARAMS(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

#endif

namespace functions {
    namespace random {
            template<typename T>
            template<typename OpClass>
            void _CUDA_D RandomFunction<T>::execTransformCuda(Nd4jPointer state, void const* vx, Nd4jLong const* xShapeBuffer, void const* vy, Nd4jLong const* yShapeBuffer, void *vz, Nd4jLong const* zShapeBuffer, void *vextraArguments) {

                auto x = reinterpret_cast<T const*>(vx);
                auto y = reinterpret_cast<T const*>(vy);
                auto z = reinterpret_cast<T*>(vz);
                auto extraArguments = reinterpret_cast<T*>(vextraArguments);

                if (OpClass::requiresSpecial) {
                    OpClass::specialOpCuda(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments);
                    return;
                } else {

                __shared__ Nd4jLong length;
                __shared__ int xEWS;
                __shared__ int yEWS;
                __shared__ int zEWS;
                __shared__ char xOrder;
                __shared__ char yOrder;
                __shared__ char zOrder;

                __shared__ sd::graph::RandomGenerator *buffer;
                __shared__ unsigned char *cB;
                __shared__ unsigned char *dB;
                sd::graph::RandomGenerator *devBuffer;
                if (threadIdx.x == 0) {
                    length = shape::length(zShapeBuffer);
                    xEWS = shape::elementWiseStride(xShapeBuffer);
                    yEWS = shape::elementWiseStride(yShapeBuffer);
                    zEWS = shape::elementWiseStride(zShapeBuffer);
                    xOrder = shape::order(xShapeBuffer);
                    yOrder = shape::order(yShapeBuffer);
                    zOrder = shape::order(zShapeBuffer);

                    extern __shared__ unsigned char shmem[];
                    buffer = (sd::graph::RandomGenerator *) shmem;
                    cB = shmem;
                    devBuffer = reinterpret_cast<sd::graph::RandomGenerator *> (state);
                    dB = reinterpret_cast<unsigned char *> (state);
                }
                __syncthreads();

                // using this loop instead of memcpy
                for (int e = threadIdx.x; e < sizeof(sd::graph::RandomGenerator); e+= blockDim.x)
                    cB[e] = dB[e];

                __syncthreads();


                int tid = blockIdx.x * blockDim.x + threadIdx.x;

                if (xEWS >= 1 && yEWS >= 1 && zEWS >= 1 && xOrder == yOrder && xOrder == zOrder) {
                    for (Nd4jLong e = tid; e < length; e += blockDim.x * gridDim.x) {
                        z[e * zEWS] = OpClass::op(x[e * xEWS], y[e * yEWS], e, length, buffer, extraArguments);
                    }
                } else {
                    for (Nd4jLong i = tid; i < length; i += blockDim.x * gridDim.x) {

                        auto xOffset2 = shape::getIndexOffset(i, xShapeBuffer);
                        auto yOffset2 = shape::getIndexOffset(i, yShapeBuffer);
                        auto zOffset2 = shape::getIndexOffset(i, zShapeBuffer);

                            z[zOffset2] = OpClass::op(x[xOffset2], y[yOffset2], i, length, buffer, extraArguments);
                        }
                    }
                }
            };


            template<typename T>
            template<typename OpClass>
            void _CUDA_D RandomFunction<T>::execTransformCuda(Nd4jPointer state, void const* vx, Nd4jLong const* xShapeBuffer, void* vz, Nd4jLong const* zShapeBuffer, void *vextraArguments) {

                auto x = reinterpret_cast<T const*>(vx);
                auto z = reinterpret_cast<T*>(vz);
                auto extraArguments = reinterpret_cast<T*>(vextraArguments);

                __shared__ Nd4jLong length;
                __shared__ int xEWS;
                __shared__ int zEWS;
                __shared__ char xOrder;
                __shared__ char zOrder;

                __shared__ sd::graph::RandomGenerator *buffer;
                __shared__ unsigned char *cB;
                __shared__ unsigned char *dB;
                __shared__ sd::graph::RandomGenerator *devBuffer;

                if (threadIdx.x == 0) {
                    extern __shared__ unsigned char shmem[];
                    buffer = (sd::graph::RandomGenerator *) shmem;
                    cB = shmem;
                    devBuffer = reinterpret_cast<sd::graph::RandomGenerator *> (state);
                    dB = reinterpret_cast<unsigned char *> (state);

                    length = shape::length(zShapeBuffer);
                    xEWS = shape::elementWiseStride(xShapeBuffer);
                    zEWS = shape::elementWiseStride(zShapeBuffer);
                    xOrder = shape::order(xShapeBuffer);
                    zOrder = shape::order(zShapeBuffer);
                }
                __syncthreads();

                // using this loop instead of memcpy
                for (int e = threadIdx.x; e < sizeof(sd::graph::RandomGenerator); e+= blockDim.x)
                    cB[e] = dB[e];

                __syncthreads();


                if (xEWS >= 1 && zEWS >= 1 && xOrder == zOrder) {
                    for (Nd4jLong e = blockIdx.x * blockDim.x + threadIdx.x; e < length; e += blockDim.x * gridDim.x) {
                        z[e * zEWS] = OpClass::op(x[e * xEWS], e, length, buffer, extraArguments);
                    }
                } else {

                    for (Nd4jLong i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += blockDim.x * gridDim.x) {

                        auto xOffset2 = shape::getIndexOffset(i, xShapeBuffer);
                        auto zOffset2 = shape::getIndexOffset(i, zShapeBuffer);

                        z[zOffset2] = OpClass::op(x[xOffset2], i, length, buffer, extraArguments);
                    }
                }
            }


            template<typename T>
            template<typename OpClass>
            void _CUDA_D RandomFunction<T>::execTransformCuda(Nd4jPointer state, void *vz, Nd4jLong const* zShapeBuffer, void *vextraArguments) {

                auto z = reinterpret_cast<T*>(vz);
                auto extraArguments = reinterpret_cast<T*>(vextraArguments);

                __shared__ Nd4jLong length;
                __shared__ Nd4jLong ews;
                __shared__ sd::graph::RandomGenerator *buffer;
                __shared__ unsigned char *cB;
                __shared__ unsigned char *dB;
                __shared__ sd::graph::RandomGenerator *devBuffer;

                if (threadIdx.x == 0) {
                    extern __shared__ unsigned char shmem[];
                    buffer = (sd::graph::RandomGenerator *) shmem;
                    cB = shmem;
                    devBuffer = reinterpret_cast<sd::graph::RandomGenerator *> (state);
                    dB = reinterpret_cast<unsigned char *> (state);
                    length = shape::length(zShapeBuffer);
                    ews = shape::elementWiseStride(zShapeBuffer);
                }
                __syncthreads();

                // using this loop instead of memcpy
                for (int e = threadIdx.x; e < sizeof(sd::graph::RandomGenerator); e+= blockDim.x)
                    cB[e] = dB[e];

                __syncthreads();

                int tid = blockIdx.x * blockDim.x + threadIdx.x;

                if (ews > 0) {
                    for (Nd4jLong i = tid; i < length; i += blockDim.x * gridDim.x) {
                        z[i * ews] = OpClass::op(i, length, buffer, extraArguments);
                    }
                } else {

                    for (Nd4jLong i = tid; i < length; i += blockDim.x * gridDim.x) {
                        auto zOffset2 = shape::getIndexOffset(i, zShapeBuffer);
                        z[zOffset2] = OpClass::op(i, length, buffer, extraArguments);
                    }
                }
            }

        template <>
        _CUDA_H void RandomFunction<float>::executeCudaSingle(dim3& launchDims, cudaStream_t* stream, int opNum, Nd4jPointer stateHost, void *vz, Nd4jLong const* zShapeBuffer, void *vextraArguments) {

            auto z = reinterpret_cast<float*>(vz);
            auto extraArguments = reinterpret_cast<float*>(vextraArguments);

            // this macro builds bunch of IF/ELSE selectors for kernel launch
            DISPATCH_SIMPLE(randomSingle, float, PARAMS(stateHost, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

            DEBUG_KERNEL(stream, opNum);
        }

        template <>
        _CUDA_H void RandomFunction<float16>::executeCudaSingle(dim3& launchDims, cudaStream_t* stream, int opNum, Nd4jPointer stateHost, void *vz, Nd4jLong const* zShapeBuffer, void *vextraArguments) {

            auto z = reinterpret_cast<float16*>(vz);
            auto extraArguments = reinterpret_cast<float16*>(vextraArguments);

            // this macro builds bunch of IF/ELSE selectors for kernel launch
            DISPATCH_SIMPLE(randomSingle, float16, PARAMS(stateHost, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

            DEBUG_KERNEL(stream, opNum);
        }

        template <>
        _CUDA_H void RandomFunction<bfloat16>::executeCudaSingle(dim3& launchDims, cudaStream_t* stream, int opNum, Nd4jPointer stateHost, void *vz, Nd4jLong const* zShapeBuffer, void *vextraArguments) {

            auto z = reinterpret_cast<bfloat16*>(vz);
            auto extraArguments = reinterpret_cast<bfloat16*>(vextraArguments);

            // this macro builds bunch of IF/ELSE selectors for kernel launch
            DISPATCH_SIMPLE(randomSingle, bfloat16, PARAMS(stateHost, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

            DEBUG_KERNEL(stream, opNum);
        }

        template <>
        _CUDA_H void RandomFunction<double>::executeCudaSingle(dim3& launchDims, cudaStream_t *stream, int opNum, Nd4jPointer stateHost, void *vz, Nd4jLong const* zShapeBuffer, void *vextraArguments) {

            auto z = reinterpret_cast<double*>(vz);
            auto extraArguments = reinterpret_cast<double*>(vextraArguments);

            // this macro builds bunch of IF/ELSE selectors for kernel launch
            DISPATCH_SIMPLE(randomSingle, double, PARAMS(stateHost, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

            DEBUG_KERNEL(stream, opNum);
        }

        template <>
        _CUDA_H void RandomFunction<float>::executeCudaDouble(dim3& launchDims, cudaStream_t* stream, int opNum, Nd4jPointer stateHost, void const* vx, Nd4jLong const* xShapeBuffer, void *vz, Nd4jLong const* zShapeBuffer, void *vextraArguments) {

            auto x = reinterpret_cast<float const*>(vx);
            auto z = reinterpret_cast<float*>(vz);
            auto extraArguments = reinterpret_cast<float*>(vextraArguments);

            // this macro builds bunch of IF/ELSE selectors for kernel launch
            DISPATCH_SIMPLE(randomDouble, float, PARAMS(stateHost, x, xShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

            DEBUG_KERNEL(stream, opNum);
        }


        template <>
        _CUDA_H void RandomFunction<float16>::executeCudaDouble(dim3& launchDims, cudaStream_t* stream, int opNum, Nd4jPointer stateHost, void const* vx, Nd4jLong const* xShapeBuffer, void *vz, Nd4jLong const* zShapeBuffer, void *vextraArguments) {

            auto x = reinterpret_cast<float16 const*>(vx);
            auto z = reinterpret_cast<float16*>(vz);
            auto extraArguments = reinterpret_cast<float16*>(vextraArguments);

            // this macro builds bunch of IF/ELSE selectors for kernel launch
            DISPATCH_SIMPLE(randomDouble, float16, PARAMS(stateHost, x, xShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

            DEBUG_KERNEL(stream, opNum);
        }

        template <>
        _CUDA_H void RandomFunction<bfloat16>::executeCudaDouble(dim3& launchDims, cudaStream_t* stream, int opNum, Nd4jPointer stateHost, void const* vx, Nd4jLong const* xShapeBuffer, void *vz, Nd4jLong const* zShapeBuffer, void *vextraArguments) {

            auto x = reinterpret_cast<bfloat16 const*>(vx);
            auto z = reinterpret_cast<bfloat16*>(vz);
            auto extraArguments = reinterpret_cast<bfloat16*>(vextraArguments);

            // this macro builds bunch of IF/ELSE selectors for kernel launch
            DISPATCH_SIMPLE(randomDouble, bfloat16, PARAMS(stateHost, x, xShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

            DEBUG_KERNEL(stream, opNum);
        }

        template <>
        _CUDA_H void RandomFunction<double>::executeCudaDouble(dim3& launchDims, cudaStream_t* stream, int opNum, Nd4jPointer stateHost, void const* vx, Nd4jLong const* xShapeBuffer, void *vz, Nd4jLong const* zShapeBuffer, void *vextraArguments) {

            auto x = reinterpret_cast<double const*>(vx);
            auto z = reinterpret_cast<double*>(vz);
            auto extraArguments = reinterpret_cast<double*>(vextraArguments);

            // this macro builds bunch of IF/ELSE selectors for kernel launch
            DISPATCH_SIMPLE(randomDouble, double, PARAMS(stateHost, x, xShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

            DEBUG_KERNEL(stream, opNum);
        }

        template <>
        _CUDA_H void RandomFunction<float>::executeCudaTriple(dim3& launchDims, cudaStream_t* stream, int opNum, Nd4jPointer stateHost, void const* vx, Nd4jLong const* xShapeBuffer, void const* vy, Nd4jLong const* yShapeBuffer, void *vz, Nd4jLong const* zShapeBuffer, void *vextraArguments) {

            auto x = reinterpret_cast<float const*>(vx);
            auto y = reinterpret_cast<float const*>(vy);
            auto z = reinterpret_cast<float*>(vz);
            auto extraArguments = reinterpret_cast<float*>(vextraArguments);

            // this macro builds bunch of IF/ELSE selectors for kernel launch
            DISPATCH_SIMPLE(randomTriple, float, PARAMS(stateHost, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

            DEBUG_KERNEL(stream, opNum);
        }

        template <>
        _CUDA_H void RandomFunction<float16>::executeCudaTriple(dim3& launchDims, cudaStream_t* stream, int opNum, Nd4jPointer stateHost, void const* vx, Nd4jLong const* xShapeBuffer, void const* vy, Nd4jLong const* yShapeBuffer, void *vz, Nd4jLong const* zShapeBuffer, void *vextraArguments) {

            auto x = reinterpret_cast<float16 const*>(vx);
            auto y = reinterpret_cast<float16 const*>(vy);
            auto z = reinterpret_cast<float16*>(vz);
            auto extraArguments = reinterpret_cast<float16*>(vextraArguments);

            // this macro builds bunch of IF/ELSE selectors for kernel launch
            DISPATCH_SIMPLE(randomTriple, float16, PARAMS(stateHost, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

            DEBUG_KERNEL(stream, opNum);
        }

        template <>
        _CUDA_H void RandomFunction<bfloat16>::executeCudaTriple(dim3& launchDims, cudaStream_t* stream, int opNum, Nd4jPointer stateHost, void const* vx, Nd4jLong const* xShapeBuffer, void const* vy, Nd4jLong const* yShapeBuffer, void *vz, Nd4jLong const* zShapeBuffer, void *vextraArguments) {

            auto x = reinterpret_cast<bfloat16 const*>(vx);
            auto y = reinterpret_cast<bfloat16 const*>(vy);
            auto z = reinterpret_cast<bfloat16*>(vz);
            auto extraArguments = reinterpret_cast<bfloat16*>(vextraArguments);

            // this macro builds bunch of IF/ELSE selectors for kernel launch
            DISPATCH_SIMPLE(randomTriple, bfloat16, PARAMS(stateHost, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

            DEBUG_KERNEL(stream, opNum);
        }



        template <>
        _CUDA_H void RandomFunction<double>::executeCudaTriple(dim3& launchDims, cudaStream_t* stream, int opNum, Nd4jPointer stateHost, void const* vx, Nd4jLong const* xShapeBuffer, void const* vy, Nd4jLong const* yShapeBuffer, void *vz, Nd4jLong const* zShapeBuffer, void *vextraArguments) {

            auto x = reinterpret_cast<double const*>(vx);
            auto y = reinterpret_cast<double const*>(vy);
            auto z = reinterpret_cast<double*>(vz);
            auto extraArguments = reinterpret_cast<double*>(vextraArguments);

            // this macro builds bunch of IF/ELSE selectors for kernel launch
            DISPATCH_SIMPLE(randomTriple, double, PARAMS(stateHost, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

            DEBUG_KERNEL(stream, opNum);
        }

        BUILD_SINGLE_TEMPLATE(template class ND4J_EXPORT RandomFunction, , FLOAT_TYPES);
    }
}
