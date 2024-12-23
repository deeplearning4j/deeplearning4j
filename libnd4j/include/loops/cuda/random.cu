/******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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

// The single/double/triple generic function names
// remain the same, but we will show the
// reworked approach inside the execTransformCuda code.

template <typename T, typename OpClass>
static SD_INLINE SD_DEVICE void randomSingleGeneric(
    sd::Pointer state,
    void* z,
    sd::LongType const* zShapeBuffer,
    void* extraArguments) {
  functions::random::RandomFunction<T>::template execTransformCuda<OpClass>(
      state, z, zShapeBuffer, extraArguments
  );
}

template <typename T, typename OpClass>
static SD_INLINE SD_DEVICE void randomDoubleGeneric(
    sd::Pointer state,
    void const* x,
    sd::LongType const* xShapeBuffer,
    void* z,
    sd::LongType const* zShapeBuffer,
    void* extraArguments) {
  functions::random::RandomFunction<T>::template execTransformCuda<OpClass>(
      state, x, xShapeBuffer, z, zShapeBuffer, extraArguments
  );
}

template <typename T, typename OpClass>
static SD_INLINE SD_DEVICE void randomTripleGeneric(
    sd::Pointer state,
    void const* x,
    sd::LongType const* xShapeBuffer,
    void const* y,
    sd::LongType const* yShapeBuffer,
    void* z,
    sd::LongType const* zShapeBuffer,
    void* extraArguments) {
  functions::random::RandomFunction<T>::template execTransformCuda<OpClass>(
      state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments
  );
}

// The DISPATCH_KERNEL_SIMPLE macros remain unaltered
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
                             sd::LongType const* yShapeBuffer, void* z, sd::LongType const* zShapeBuffer, void* extraArguments),
                       PARAMS(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments),
                       OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomTriple_, randomTripleGeneric, double,
                       INPUT(sd::Pointer state, void const* x, sd::LongType const* xShapeBuffer, void const* y,
                             sd::LongType const* yShapeBuffer, void* z, sd::LongType const* zShapeBuffer, void* extraArguments),
                       PARAMS(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments),
                       OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomTriple_, randomTripleGeneric, float16,
                       INPUT(sd::Pointer state, void const* x, sd::LongType const* xShapeBuffer, void const* y,
                             sd::LongType const* yShapeBuffer, void* z, sd::LongType const* zShapeBuffer, void* extraArguments),
                       PARAMS(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments),
                       OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomTriple_, randomTripleGeneric, bfloat16,
                       INPUT(sd::Pointer state, void const* x, sd::LongType const* xShapeBuffer, void const* y,
                             sd::LongType const* yShapeBuffer, void* z, sd::LongType const* zShapeBuffer, void* extraArguments),
                       PARAMS(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments),
                       OPS_A(RANDOM_OPS))

namespace functions {
namespace random {

// Below we rewrite the execTransformCuda methods with shape-of, stride-of, rank caching.
template <typename T>
template <typename OpClass>
SD_DEVICE void RandomFunction<T>::execTransformCuda(
    sd::Pointer state,
    void const* vx,
    sd::LongType const* xShapeBuffer,
    void const* vy,
    sd::LongType const* yShapeBuffer,
    void* vz,
    sd::LongType const* zShapeBuffer,
    void* vextraArguments)
{
  auto x = reinterpret_cast<const T*>(vx);
  auto y = reinterpret_cast<const T*>(vy);
  auto z = reinterpret_cast<T*>(vz);
  auto extraArguments = reinterpret_cast<T*>(vextraArguments);

  // If special op is needed, do that first
  if (OpClass::requiresSpecial) {
    OpClass::specialOpCuda(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments);
    return;
  } else {
    // We do shape caching in shared memory
    __shared__ sd::LongType length;
    __shared__ int xRank, yRank, zRank;
    __shared__ const sd::LongType* xShapePtr;
    __shared__ const sd::LongType* yShapePtr;
    __shared__ const sd::LongType* zShapePtr;
    __shared__ const sd::LongType* xStridePtr;
    __shared__ const sd::LongType* yStridePtr;
    __shared__ const sd::LongType* zStridePtr;

    // also copy the random generator from global to shared memory
    __shared__ sd::graph::RandomGenerator* rng;
    __shared__ unsigned char* cB;
    __shared__ unsigned char* dB;

    if (threadIdx.x == 0) {
      length    = shape::length(zShapeBuffer);

      xRank     = shape::rank(xShapeBuffer);
      yRank     = shape::rank(yShapeBuffer);
      zRank     = shape::rank(zShapeBuffer);

      xShapePtr = shape::shapeOf(xShapeBuffer);
      yShapePtr = shape::shapeOf(yShapeBuffer);
      zShapePtr = shape::shapeOf(zShapeBuffer);

      xStridePtr= shape::stride(xShapeBuffer);
      yStridePtr= shape::stride(yShapeBuffer);
      zStridePtr= shape::stride(zShapeBuffer);

      extern __shared__ unsigned char sharedMem[];
      rng = reinterpret_cast<sd::graph::RandomGenerator*>(sharedMem);
      cB  = sharedMem;
      dB  = reinterpret_cast<unsigned char*>(state);
    }
    __syncthreads();

    // copy global random generator to block's shared memory
    for (int e = threadIdx.x; e < (int)sizeof(sd::graph::RandomGenerator); e += blockDim.x)
      cB[e] = dB[e];

    __syncthreads();

    // do the actual transform
    const int tid          = blockDim.x * blockIdx.x + threadIdx.x;
    const int totalThreads = blockDim.x * gridDim.x;

    for (sd::LongType i = tid; i < length; i += totalThreads) {
      sd::LongType coordsX[SD_MAX_RANK];
      sd::LongType coordsY[SD_MAX_RANK];
      sd::LongType coordsZ[SD_MAX_RANK];

      sd::LongType xOffset, yOffset, zOffset;

      INDEX2COORDS(i, xRank, xShapePtr, coordsX);
      COORDS2INDEX(xRank, xStridePtr, coordsX, xOffset);

      INDEX2COORDS(i, yRank, yShapePtr, coordsY);
      COORDS2INDEX(yRank, yStridePtr, coordsY, yOffset);

      INDEX2COORDS(i, zRank, zShapePtr, coordsZ);
      COORDS2INDEX(zRank, zStridePtr, coordsZ, zOffset);

      // op: we pass in the generator from shared memory
      z[zOffset] = OpClass::op(x[xOffset], y[yOffset], i, length, rng, extraArguments);
    }
  }
}

template <typename T>
template <typename OpClass>
SD_DEVICE void RandomFunction<T>::execTransformCuda(
    sd::Pointer state,
    void const* vx,  // x array
    sd::LongType const* xShapeBuffer,
    void* vz,        // z array
    sd::LongType const* zShapeBuffer,
    void* vextraArguments)
{
  auto x = reinterpret_cast<const T*>(vx);
  auto z = reinterpret_cast<T*>(vz);
  auto extraArgs = reinterpret_cast<T*>(vextraArguments);

  // shape caching in shared memory
  __shared__ sd::LongType length;
  __shared__ int xRank, zRank;
  __shared__ const sd::LongType* xShapePtr;
  __shared__ const sd::LongType* xStridePtr;
  __shared__ const sd::LongType* zShapePtr;
  __shared__ const sd::LongType* zStridePtr;

  __shared__ sd::graph::RandomGenerator* rng;
  __shared__ unsigned char* cB;
  __shared__ unsigned char* dB;

  if (threadIdx.x == 0) {
    length     = shape::length(zShapeBuffer);

    xRank      = shape::rank(xShapeBuffer);
    zRank      = shape::rank(zShapeBuffer);

    xShapePtr  = shape::shapeOf(xShapeBuffer);
    xStridePtr = shape::stride(xShapeBuffer);
    zShapePtr  = shape::shapeOf(zShapeBuffer);
    zStridePtr = shape::stride(zShapeBuffer);

    extern __shared__ unsigned char sharedMem[];
    rng = reinterpret_cast<sd::graph::RandomGenerator*>(sharedMem);
    cB  = sharedMem;
    dB  = reinterpret_cast<unsigned char*>(state);
  }
  __syncthreads();

  for (int e = threadIdx.x; e < (int)sizeof(sd::graph::RandomGenerator); e += blockDim.x)
    cB[e] = dB[e];

  __syncthreads();

  const int tid          = blockDim.x * blockIdx.x + threadIdx.x;
  const int totalThreads = blockDim.x * gridDim.x;

  for (sd::LongType i = tid; i < length; i += totalThreads) {
    sd::LongType coordsX[SD_MAX_RANK];
    sd::LongType coordsZ[SD_MAX_RANK];
    sd::LongType xOffset, zOffset;

    INDEX2COORDS(i, xRank, xShapePtr, coordsX);
    COORDS2INDEX(xRank, xStridePtr, coordsX, xOffset);

    INDEX2COORDS(i, zRank, zShapePtr, coordsZ);
    COORDS2INDEX(zRank, zStridePtr, coordsZ, zOffset);

    z[zOffset] = OpClass::op(x[xOffset], i, length, rng, extraArgs);
  }
}

template <typename T>
template <typename OpClass>
SD_DEVICE void RandomFunction<T>::execTransformCuda(
    sd::Pointer state,
    void* vz,
    sd::LongType const* zShapeBuffer,
    void* vextraArguments)
{
  auto z = reinterpret_cast<T*>(vz);
  auto extraArgs = reinterpret_cast<T*>(vextraArguments);

  // shape caching in shared memory
  __shared__ sd::LongType length;
  __shared__ int zRank;
  __shared__ const sd::LongType* zShapePtr;
  __shared__ const sd::LongType* zStridePtr;

  __shared__ sd::graph::RandomGenerator* rng;
  __shared__ unsigned char* cB;
  __shared__ unsigned char* dB;

  if (threadIdx.x == 0) {
    length   = shape::length(zShapeBuffer);
    zRank    = shape::rank(zShapeBuffer);
    zShapePtr= shape::shapeOf(zShapeBuffer);
    zStridePtr= shape::stride(zShapeBuffer);

    extern __shared__ unsigned char sharedMem[];
    rng = reinterpret_cast<sd::graph::RandomGenerator*>(sharedMem);
    cB  = sharedMem;
    dB  = reinterpret_cast<unsigned char*>(state);
  }
  __syncthreads();

  for (int e = threadIdx.x; e < (int)sizeof(sd::graph::RandomGenerator); e += blockDim.x)
    cB[e] = dB[e];

  __syncthreads();

  const int tid          = blockDim.x * blockIdx.x + threadIdx.x;
  const int totalThreads = blockDim.x * gridDim.x;

  for (sd::LongType i = tid; i < length; i += totalThreads) {
    sd::LongType coordsZ[SD_MAX_RANK];
    sd::LongType zOffset;

    INDEX2COORDS(i, zRank, zShapePtr, coordsZ);
    COORDS2INDEX(zRank, zStridePtr, coordsZ, zOffset);

    z[zOffset] = OpClass::op(i, length, rng, extraArgs);
  }
}

// Everything else remains the same, except we've replaced the repeated shape calls
// in each loop with shared memory caching within each execTransformCuda method.

namespace functions {
namespace random {

// We keep the rest of the code that calls these execTransformCuda methods
// via macros and method calls. The macros remain the same as well.
BUILD_SINGLE_TEMPLATE(template class RandomFunction, , SD_FLOAT_TYPES);

}  // namespace random
}  // namespace functions

