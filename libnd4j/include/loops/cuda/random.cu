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
#include <helpers/DebugHelper.h>
#include <loops/random.h>
#include <ops/specials_cuda.h>
#include <system/common.h>
#include <system/op_boilerplate.h>

using namespace randomOps;

template <typename T, typename OpClass>
static SD_INLINE SD_DEVICE void randomSingleGeneric(sd::Pointer state, void* z, sd::LongType const* zShapeBuffer,
                                                    void* extraArguments) {
  functions::random::RandomFunction<T>::template execTransformCuda<OpClass>(state, z, zShapeBuffer, extraArguments);
}

template <typename T, typename OpClass>
static SD_INLINE SD_DEVICE void randomDoubleGeneric(sd::Pointer state, void const* x, sd::LongType const* xShapeBuffer,
                                                    void* z, sd::LongType const* zShapeBuffer, void* extraArguments) {
  functions::random::RandomFunction<T>::template execTransformCuda<OpClass>(state, x, xShapeBuffer, z, zShapeBuffer,
                                                                            extraArguments);
}

template <typename T, typename OpClass>
static SD_INLINE SD_DEVICE void randomTripleGeneric(sd::Pointer state, void const* x, sd::LongType const* xShapeBuffer,
                                                    void const* y, sd::LongType const* yShapeBuffer, void* z,
                                                    sd::LongType const* zShapeBuffer, void* extraArguments) {
  functions::random::RandomFunction<T>::template execTransformCuda<OpClass>(state, x, xShapeBuffer, y, yShapeBuffer, z,
                                                                            zShapeBuffer, extraArguments);
}

#ifndef __CLION_IDE__
// here we generate kernels for target operations
DISPATCH_KERNEL_SIMPLE(randomSingle_, randomSingleGeneric, float,
                       INPUT(sd::Pointer state, void* z, sd::LongType const* zShapeBuffer, void* extraArguments),
                       PARAMS(state, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomSingle_, randomSingleGeneric, double,
                       INPUT(sd::Pointer state, void* z, sd::LongType const* zShapeBuffer, void* extraArguments),
                       PARAMS(state, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomSingle_, randomSingleGeneric, float16,
                       INPUT(sd::Pointer state, void* z, sd::LongType const* zShapeBuffer, void* extraArguments),
                       PARAMS(state, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomSingle_, randomSingleGeneric, bfloat16,
                       INPUT(sd::Pointer state, void* z, sd::LongType const* zShapeBuffer, void* extraArguments),
                       PARAMS(state, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

DISPATCH_KERNEL_SIMPLE(randomDouble_, randomDoubleGeneric, float,
                       INPUT(sd::Pointer state, void const* x, sd::LongType const* xShapeBuffer, void* z,
                             sd::LongType const* zShapeBuffer, void* extraArguments),
                       PARAMS(state, x, xShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomDouble_, randomDoubleGeneric, double,
                       INPUT(sd::Pointer state, void const* x, sd::LongType const* xShapeBuffer, void* z,
                             sd::LongType const* zShapeBuffer, void* extraArguments),
                       PARAMS(state, x, xShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomDouble_, randomDoubleGeneric, float16,
                       INPUT(sd::Pointer state, void const* x, sd::LongType const* xShapeBuffer, void* z,
                             sd::LongType const* zShapeBuffer, void* extraArguments),
                       PARAMS(state, x, xShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomDouble_, randomDoubleGeneric, bfloat16,
                       INPUT(sd::Pointer state, void const* x, sd::LongType const* xShapeBuffer, void* z,
                             sd::LongType const* zShapeBuffer, void* extraArguments),
                       PARAMS(state, x, xShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

DISPATCH_KERNEL_SIMPLE(randomTriple_, randomTripleGeneric, float,
                       INPUT(sd::Pointer state, void const* x, sd::LongType const* xShapeBuffer, void const* y,
                             sd::LongType const* yShapeBuffer, void* z, sd::LongType const* zShapeBuffer,
                             void* extraArguments),
                       PARAMS(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments),
                       OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomTriple_, randomTripleGeneric, double,
                       INPUT(sd::Pointer state, void const* x, sd::LongType const* xShapeBuffer, void const* y,
                             sd::LongType const* yShapeBuffer, void* z, sd::LongType const* zShapeBuffer,
                             void* extraArguments),
                       PARAMS(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments),
                       OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomTriple_, randomTripleGeneric, float16,
                       INPUT(sd::Pointer state, void const* x, sd::LongType const* xShapeBuffer, void const* y,
                             sd::LongType const* yShapeBuffer, void* z, sd::LongType const* zShapeBuffer,
                             void* extraArguments),
                       PARAMS(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments),
                       OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomTriple_, randomTripleGeneric, bfloat16,
                       INPUT(sd::Pointer state, void const* x, sd::LongType const* xShapeBuffer, void const* y,
                             sd::LongType const* yShapeBuffer, void* z, sd::LongType const* zShapeBuffer,
                             void* extraArguments),
                       PARAMS(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments),
                       OPS_A(RANDOM_OPS))

#endif

namespace functions {
namespace random {
template <typename T>
template <typename OpClass>
void SD_DEVICE RandomFunction<T>::execTransformCuda(sd::Pointer state, void const* vx, sd::LongType const* xShapeBuffer,
                                                    void const* vy, sd::LongType const* yShapeBuffer, void* vz,
                                                    sd::LongType const* zShapeBuffer, void* vextraArguments) {
  auto x = reinterpret_cast<T const*>(vx);
  auto y = reinterpret_cast<T const*>(vy);
  auto z = reinterpret_cast<T*>(vz);
  auto extraArguments = reinterpret_cast<T*>(vextraArguments);

  if (OpClass::requiresSpecial) {
    OpClass::specialOpCuda(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments);
    return;
  } else {
    __shared__ sd::LongType length;
    __shared__ int xEWS;
    __shared__ int yEWS;
    __shared__ int zEWS;
    __shared__ char xOrder;
    __shared__ char yOrder;
    __shared__ char zOrder;

    __shared__ sd::graph::RandomGenerator* buffer;
    __shared__ unsigned char* cB;
    __shared__ unsigned char* dB;
    sd::graph::RandomGenerator* devBuffer;
    if (threadIdx.x == 0) {
      length = shape::length(zShapeBuffer);
      xEWS = shape::elementWiseStride(xShapeBuffer);
      yEWS = shape::elementWiseStride(yShapeBuffer);
      zEWS = shape::elementWiseStride(zShapeBuffer);
      xOrder = shape::order(xShapeBuffer);
      yOrder = shape::order(yShapeBuffer);
      zOrder = shape::order(zShapeBuffer);

      extern __shared__ unsigned char shmem[];
      buffer = (sd::graph::RandomGenerator*)shmem;
      cB = shmem;
      devBuffer = reinterpret_cast<sd::graph::RandomGenerator*>(state);
      dB = reinterpret_cast<unsigned char*>(state);
    }
    __syncthreads();

    // using this loop instead of memcpy
    for (int e = threadIdx.x; e < sizeof(sd::graph::RandomGenerator); e += blockDim.x) cB[e] = dB[e];

    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (xEWS >= 1 && yEWS >= 1 && zEWS >= 1 && xOrder == yOrder && xOrder == zOrder) {
      for (sd::LongType e = tid; e < length; e += blockDim.x * gridDim.x) {
        z[e * zEWS] = OpClass::op(x[e * xEWS], y[e * yEWS], e, length, buffer, extraArguments);
      }
    } else {
      for (sd::LongType i = tid; i < length; i += blockDim.x * gridDim.x) {
        auto xOffset2 = shape::getIndexOffset(i, xShapeBuffer);
        auto yOffset2 = shape::getIndexOffset(i, yShapeBuffer);
        auto zOffset2 = shape::getIndexOffset(i, zShapeBuffer);

        z[zOffset2] = OpClass::op(x[xOffset2], y[yOffset2], i, length, buffer, extraArguments);
      }
    }
  }
};

template <typename T>
template <typename OpClass>
void SD_DEVICE RandomFunction<T>::execTransformCuda(sd::Pointer state, void const* vx, sd::LongType const* xShapeBuffer,
                                                    void* vz, sd::LongType const* zShapeBuffer, void* vextraArguments) {
  auto x = reinterpret_cast<T const*>(vx);
  auto z = reinterpret_cast<T*>(vz);
  auto extraArguments = reinterpret_cast<T*>(vextraArguments);

  __shared__ sd::LongType length;
  __shared__ int xEWS;
  __shared__ int zEWS;
  __shared__ char xOrder;
  __shared__ char zOrder;

  __shared__ sd::graph::RandomGenerator* buffer;
  __shared__ unsigned char* cB;
  __shared__ unsigned char* dB;
  __shared__ sd::graph::RandomGenerator* devBuffer;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    buffer = (sd::graph::RandomGenerator*)shmem;
    cB = shmem;
    devBuffer = reinterpret_cast<sd::graph::RandomGenerator*>(state);
    dB = reinterpret_cast<unsigned char*>(state);

    length = shape::length(zShapeBuffer);
    xEWS = shape::elementWiseStride(xShapeBuffer);
    zEWS = shape::elementWiseStride(zShapeBuffer);
    xOrder = shape::order(xShapeBuffer);
    zOrder = shape::order(zShapeBuffer);
  }
  __syncthreads();

  // using this loop instead of memcpy
  for (int e = threadIdx.x; e < sizeof(sd::graph::RandomGenerator); e += blockDim.x) cB[e] = dB[e];

  __syncthreads();

  if (xEWS >= 1 && zEWS >= 1 && xOrder == zOrder) {
    for (sd::LongType e = blockIdx.x * blockDim.x + threadIdx.x; e < length; e += blockDim.x * gridDim.x) {
      z[e * zEWS] = OpClass::op(x[e * xEWS], e, length, buffer, extraArguments);
    }
  } else {
    for (sd::LongType i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += blockDim.x * gridDim.x) {
      auto xOffset2 = shape::getIndexOffset(i, xShapeBuffer);
      auto zOffset2 = shape::getIndexOffset(i, zShapeBuffer);

      z[zOffset2] = OpClass::op(x[xOffset2], i, length, buffer, extraArguments);
    }
  }
}

template <typename T>
template <typename OpClass>
void SD_DEVICE RandomFunction<T>::execTransformCuda(sd::Pointer state, void* vz, sd::LongType const* zShapeBuffer,
                                                    void* vextraArguments) {
  auto z = reinterpret_cast<T*>(vz);
  auto extraArguments = reinterpret_cast<T*>(vextraArguments);

  __shared__ sd::LongType length;
  __shared__ sd::LongType ews;
  __shared__ sd::graph::RandomGenerator* buffer;
  __shared__ unsigned char* cB;
  __shared__ unsigned char* dB;
  __shared__ sd::graph::RandomGenerator* devBuffer;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    buffer = (sd::graph::RandomGenerator*)shmem;
    cB = shmem;
    devBuffer = reinterpret_cast<sd::graph::RandomGenerator*>(state);
    dB = reinterpret_cast<unsigned char*>(state);
    length = shape::length(zShapeBuffer);
    ews = shape::elementWiseStride(zShapeBuffer);
  }
  __syncthreads();

  // using this loop instead of memcpy
  for (int e = threadIdx.x; e < sizeof(sd::graph::RandomGenerator); e += blockDim.x) cB[e] = dB[e];

  __syncthreads();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (ews > 0) {
    for (sd::LongType i = tid; i < length; i += blockDim.x * gridDim.x) {
      z[i * ews] = OpClass::op(i, length, buffer, extraArguments);
    }
  } else {
    for (sd::LongType i = tid; i < length; i += blockDim.x * gridDim.x) {
      auto zOffset2 = shape::getIndexOffset(i, zShapeBuffer);
      z[zOffset2] = OpClass::op(i, length, buffer, extraArguments);
    }
  }
}

template <>
SD_HOST void RandomFunction<float>::executeCudaSingle(dim3& launchDims, cudaStream_t* stream, int opNum,
                                                      sd::Pointer stateHost, void* vz, sd::LongType const* zShapeBuffer,
                                                      void* vextraArguments) {
  auto z = reinterpret_cast<float*>(vz);
  auto extraArguments = reinterpret_cast<float*>(vextraArguments);

  // this macro builds bunch of IF/ELSE selectors for kernel launch
  DISPATCH_SIMPLE(randomSingle, float, PARAMS(stateHost, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

  sd::DebugHelper::checkErrorCode(stream, "RandomFunction executeCudaSingle(...) failed");
}

template <>
SD_HOST void RandomFunction<float16>::executeCudaSingle(dim3& launchDims, cudaStream_t* stream, int opNum,
                                                        sd::Pointer stateHost, void* vz,
                                                        sd::LongType const* zShapeBuffer, void* vextraArguments) {
  auto z = reinterpret_cast<float16*>(vz);
  auto extraArguments = reinterpret_cast<float16*>(vextraArguments);

  // this macro builds bunch of IF/ELSE selectors for kernel launch
  DISPATCH_SIMPLE(randomSingle, float16, PARAMS(stateHost, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

  sd::DebugHelper::checkErrorCode(stream, "RandomFunction executeCudaSingle(...) failed");
}

template <>
SD_HOST void RandomFunction<bfloat16>::executeCudaSingle(dim3& launchDims, cudaStream_t* stream, int opNum,
                                                         sd::Pointer stateHost, void* vz,
                                                         sd::LongType const* zShapeBuffer, void* vextraArguments) {
  auto z = reinterpret_cast<bfloat16*>(vz);
  auto extraArguments = reinterpret_cast<bfloat16*>(vextraArguments);

  // this macro builds bunch of IF/ELSE selectors for kernel launch
  DISPATCH_SIMPLE(randomSingle, bfloat16, PARAMS(stateHost, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

  sd::DebugHelper::checkErrorCode(stream, "RandomFunction executeCudaSingle(...) failed");
}

template <>
SD_HOST void RandomFunction<double>::executeCudaSingle(dim3& launchDims, cudaStream_t* stream, int opNum,
                                                       sd::Pointer stateHost, void* vz,
                                                       sd::LongType const* zShapeBuffer, void* vextraArguments) {
  auto z = reinterpret_cast<double*>(vz);
  auto extraArguments = reinterpret_cast<double*>(vextraArguments);

  // this macro builds bunch of IF/ELSE selectors for kernel launch
  DISPATCH_SIMPLE(randomSingle, double, PARAMS(stateHost, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

  sd::DebugHelper::checkErrorCode(stream, "RandomFunction executeCudaSingle(...) failed");
}

template <>
SD_HOST void RandomFunction<float>::executeCudaDouble(dim3& launchDims, cudaStream_t* stream, int opNum,
                                                      sd::Pointer stateHost, void const* vx,
                                                      sd::LongType const* xShapeBuffer, void* vz,
                                                      sd::LongType const* zShapeBuffer, void* vextraArguments) {
  auto x = reinterpret_cast<float const*>(vx);
  auto z = reinterpret_cast<float*>(vz);
  auto extraArguments = reinterpret_cast<float*>(vextraArguments);

  // this macro builds bunch of IF/ELSE selectors for kernel launch
  DISPATCH_SIMPLE(randomDouble, float, PARAMS(stateHost, x, xShapeBuffer, z, zShapeBuffer, extraArguments),
                  OPS_A(RANDOM_OPS))

  sd::DebugHelper::checkErrorCode(stream, "RandomFunction executeCudaSingle(...) failed");
}

template <>
SD_HOST void RandomFunction<float16>::executeCudaDouble(dim3& launchDims, cudaStream_t* stream, int opNum,
                                                        sd::Pointer stateHost, void const* vx,
                                                        sd::LongType const* xShapeBuffer, void* vz,
                                                        sd::LongType const* zShapeBuffer, void* vextraArguments) {
  auto x = reinterpret_cast<float16 const*>(vx);
  auto z = reinterpret_cast<float16*>(vz);
  auto extraArguments = reinterpret_cast<float16*>(vextraArguments);

  // this macro builds bunch of IF/ELSE selectors for kernel launch
  DISPATCH_SIMPLE(randomDouble, float16, PARAMS(stateHost, x, xShapeBuffer, z, zShapeBuffer, extraArguments),
                  OPS_A(RANDOM_OPS))

  sd::DebugHelper::checkErrorCode(stream, "RandomFunction executeCudaSingle(...) failed");
}

template <>
SD_HOST void RandomFunction<bfloat16>::executeCudaDouble(dim3& launchDims, cudaStream_t* stream, int opNum,
                                                         sd::Pointer stateHost, void const* vx,
                                                         sd::LongType const* xShapeBuffer, void* vz,
                                                         sd::LongType const* zShapeBuffer, void* vextraArguments) {
  auto x = reinterpret_cast<bfloat16 const*>(vx);
  auto z = reinterpret_cast<bfloat16*>(vz);
  auto extraArguments = reinterpret_cast<bfloat16*>(vextraArguments);

  // this macro builds bunch of IF/ELSE selectors for kernel launch
  DISPATCH_SIMPLE(randomDouble, bfloat16, PARAMS(stateHost, x, xShapeBuffer, z, zShapeBuffer, extraArguments),
                  OPS_A(RANDOM_OPS))

  sd::DebugHelper::checkErrorCode(stream, "RandomFunction executeCudaSingle(...) failed");
}

template <>
SD_HOST void RandomFunction<double>::executeCudaDouble(dim3& launchDims, cudaStream_t* stream, int opNum,
                                                       sd::Pointer stateHost, void const* vx,
                                                       sd::LongType const* xShapeBuffer, void* vz,
                                                       sd::LongType const* zShapeBuffer, void* vextraArguments) {
  auto x = reinterpret_cast<double const*>(vx);
  auto z = reinterpret_cast<double*>(vz);
  auto extraArguments = reinterpret_cast<double*>(vextraArguments);

  // this macro builds bunch of IF/ELSE selectors for kernel launch
  DISPATCH_SIMPLE(randomDouble, double, PARAMS(stateHost, x, xShapeBuffer, z, zShapeBuffer, extraArguments),
                  OPS_A(RANDOM_OPS))

  sd::DebugHelper::checkErrorCode(stream, "RandomFunction executeCudaSingle(...) failed");
}

template <>
SD_HOST void RandomFunction<float>::executeCudaTriple(dim3& launchDims, cudaStream_t* stream, int opNum,
                                                      sd::Pointer stateHost, void const* vx,
                                                      sd::LongType const* xShapeBuffer, void const* vy,
                                                      sd::LongType const* yShapeBuffer, void* vz,
                                                      sd::LongType const* zShapeBuffer, void* vextraArguments) {
  auto x = reinterpret_cast<float const*>(vx);
  auto y = reinterpret_cast<float const*>(vy);
  auto z = reinterpret_cast<float*>(vz);
  auto extraArguments = reinterpret_cast<float*>(vextraArguments);

  // this macro builds bunch of IF/ELSE selectors for kernel launch
  DISPATCH_SIMPLE(randomTriple, float,
                  PARAMS(stateHost, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments),
                  OPS_A(RANDOM_OPS))

  sd::DebugHelper::checkErrorCode(stream, "RandomFunction executeCudaSingle(...) failed");
}

template <>
SD_HOST void RandomFunction<float16>::executeCudaTriple(dim3& launchDims, cudaStream_t* stream, int opNum,
                                                        sd::Pointer stateHost, void const* vx,
                                                        sd::LongType const* xShapeBuffer, void const* vy,
                                                        sd::LongType const* yShapeBuffer, void* vz,
                                                        sd::LongType const* zShapeBuffer, void* vextraArguments) {
  auto x = reinterpret_cast<float16 const*>(vx);
  auto y = reinterpret_cast<float16 const*>(vy);
  auto z = reinterpret_cast<float16*>(vz);
  auto extraArguments = reinterpret_cast<float16*>(vextraArguments);

  // this macro builds bunch of IF/ELSE selectors for kernel launch
  DISPATCH_SIMPLE(randomTriple, float16,
                  PARAMS(stateHost, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments),
                  OPS_A(RANDOM_OPS))

  sd::DebugHelper::checkErrorCode(stream, "RandomFunction executeCudaSingle(...) failed");
}

template <>
SD_HOST void RandomFunction<bfloat16>::executeCudaTriple(dim3& launchDims, cudaStream_t* stream, int opNum,
                                                         sd::Pointer stateHost, void const* vx,
                                                         sd::LongType const* xShapeBuffer, void const* vy,
                                                         sd::LongType const* yShapeBuffer, void* vz,
                                                         sd::LongType const* zShapeBuffer, void* vextraArguments) {
  auto x = reinterpret_cast<bfloat16 const*>(vx);
  auto y = reinterpret_cast<bfloat16 const*>(vy);
  auto z = reinterpret_cast<bfloat16*>(vz);
  auto extraArguments = reinterpret_cast<bfloat16*>(vextraArguments);

  // this macro builds bunch of IF/ELSE selectors for kernel launch
  DISPATCH_SIMPLE(randomTriple, bfloat16,
                  PARAMS(stateHost, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments),
                  OPS_A(RANDOM_OPS))

  sd::DebugHelper::checkErrorCode(stream, "RandomFunction executeCudaSingle(...) failed");
}

template <>
SD_HOST void RandomFunction<double>::executeCudaTriple(dim3& launchDims, cudaStream_t* stream, int opNum,
                                                       sd::Pointer stateHost, void const* vx,
                                                       sd::LongType const* xShapeBuffer, void const* vy,
                                                       sd::LongType const* yShapeBuffer, void* vz,
                                                       sd::LongType const* zShapeBuffer, void* vextraArguments) {
  auto x = reinterpret_cast<double const*>(vx);
  auto y = reinterpret_cast<double const*>(vy);
  auto z = reinterpret_cast<double*>(vz);
  auto extraArguments = reinterpret_cast<double*>(vextraArguments);

  // this macro builds bunch of IF/ELSE selectors for kernel launch
  DISPATCH_SIMPLE(randomTriple, double,
                  PARAMS(stateHost, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments),
                  OPS_A(RANDOM_OPS))

  sd::DebugHelper::checkErrorCode(stream, "RandomFunction executeCudaSingle(...) failed");
}

BUILD_SINGLE_TEMPLATE(template class RandomFunction, , SD_FLOAT_TYPES);
}  // namespace random
}  // namespace functions
