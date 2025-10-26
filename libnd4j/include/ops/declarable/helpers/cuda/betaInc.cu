/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author Yurii Shyrma (iuriish@yahoo.com)
//
#include <array/DataTypeUtils.h>
#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/betaInc.h>

#include <cmath>


#include "execution/cuda/LaunchDims.h"

namespace sd {
namespace ops {
namespace helpers {

///////////////////////////////////////////////////////////////////
// modified Lentz’s algorithm for continued fractions,
// reference: Lentz, W.J. 1976, “Generating Bessel Functions in Mie Scattering Calculations Using Continued Fractions,”
template <typename T>
SD_DEVICE T continuedFractionCuda(const T a, const T b, const T x) {
  extern __shared__ unsigned char shmem[];
  T* coeffs = reinterpret_cast<T*>(shmem);

  const T min = DataTypeUtils::min<T>() / DataTypeUtils::eps<T>();
  const T aPlusb = a + b;
  T val, aPlus2i;

  T t2 = coeffs[1];
  T t1 = coeffs[0];
  if (math::sd_abs<T,T>(t1) < min) t1 = min;
  t1 = static_cast<T>(1) / t1;
  T result = t1;

  for (LongType i = 1; i <= maxIter; ++i) {
    const LongType i2 = 2 * i;
    aPlus2i = a + static_cast<T>(i2);

    // t1
    t1 = static_cast<T>(1) + coeffs[i2] * t1;
    if (math::sd_abs<T,T>(t1) < min) t1 = min;
    t1 = static_cast<T>(1) / t1;
    // t2
    t2 = static_cast<T>(1) + coeffs[i2] / t2;
    if (math::sd_abs<T,T>(t2) < min) t2 = min;
    // result
    result *= t2 * t1;
    // t1
    t1 = static_cast<T>(1) + coeffs[i2 + 1] * t1;
    if (math::sd_abs<T,T>(t1) < min) t1 = min;
    t1 = static_cast<T>(1) / t1;
    // t2
    t2 = static_cast<T>(1) + coeffs[i2 + 1] / t2;
    if (math::sd_abs<T,T>(t2) < min) t2 = min;
    // result
    val = t2 * t1;
    result *= val;

    // condition to stop loop
    if (math::sd_abs<T,T>(val - static_cast<T>(1)) <= DataTypeUtils::eps<T>()) return result;
  }

  return DataTypeUtils::infOrMax<T>();  // no convergence, more iterations is required, return infinity
}

///////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void betaIncForArrayCuda(const void* va, const LongType* aShapeInfo, const void* vb,
                                   const LongType* bShapeInfo, const void* vx, const LongType* xShapeInfo,
                                   void* vz, const LongType* zShapeInfo) {
  extern __shared__ unsigned char shmem[];
  T* sharedMem = reinterpret_cast<T*>(shmem);
  T* z = reinterpret_cast<T*>(vz);

  __shared__ LongType aLen, bLen, xLen, zLen, aOffset, bOffset, xOffset, zOffset;
  __shared__ int aRank, bRank, xRank, zRank;
  __shared__ const LongType *aShape, *bShape, *xShape, *zShape;
  __shared__ const LongType *aStride, *bStride, *xStride, *zStride;
  __shared__ T a, b, x;
  __shared__ bool symmCond;

  const LongType j = blockIdx.x;  // one block per each element

  if (threadIdx.x == 0) {
    // Cache lengths
    aLen = shape::length(aShapeInfo);
    bLen = shape::length(bShapeInfo);
    xLen = shape::length(xShapeInfo);
    zLen = shape::length(zShapeInfo);

    // Cache ranks
    aRank = shape::rank(aShapeInfo);
    bRank = shape::rank(bShapeInfo);
    xRank = shape::rank(xShapeInfo);
    zRank = shape::rank(zShapeInfo);

    // Cache shapes
    aShape = shape::shapeOf(aShapeInfo);
    bShape = shape::shapeOf(bShapeInfo);
    xShape = shape::shapeOf(xShapeInfo);
    zShape = shape::shapeOf(zShapeInfo);

    // Cache strides
    aStride = shape::stride(aShapeInfo);
    bStride = shape::stride(bShapeInfo);
    xStride = shape::stride(xShapeInfo);
    zStride = shape::stride(zShapeInfo);

    LongType aCoords[SD_MAX_RANK];
    LongType bCoords[SD_MAX_RANK];
    LongType xCoords[SD_MAX_RANK];
    LongType zCoords[SD_MAX_RANK];

    INDEX2COORDS(j, aRank, aShape, aCoords);
    COORDS2INDEX(aRank, aStride, aCoords, aOffset);
    INDEX2COORDS(j, bRank, bShape, bCoords);
    COORDS2INDEX(bRank, bStride, bCoords, bOffset);
    INDEX2COORDS(j, xRank, xShape, xCoords);
    COORDS2INDEX(xRank, xStride, xCoords, xOffset);
    INDEX2COORDS(j, zRank, zShape, zCoords);
    COORDS2INDEX(zRank, zStride, zCoords, zOffset);

    if (aOffset >= aLen || bOffset >= bLen || xOffset >= xLen || zOffset >= zLen)
      return;

    a = *(reinterpret_cast<const T*>(va) + aOffset);
    b = *(reinterpret_cast<const T*>(vb) + bOffset);
    x = *(reinterpret_cast<const T*>(vx) + xOffset);
    symmCond = x > (a + T(1)) / (a + b + T(2));

    if (symmCond) {  // swap a and b, x = 1 - x
      T temp = a;
      a = b;
      b = temp;
      x = T(1) - x;
    }
  }
  __syncthreads();

  // t^{n-1} * (1 - t)^{n-1} is symmetric function with respect to x = 0.5
  if (zOffset < zLen && a == b && x == T(0.5)) {
    z[zOffset] = T(0.5);
    return;
  }

  if (zOffset < zLen && (x == T(0) || x == T(1))) {
    if (symmCond) {
      z[zOffset] = T(1) - x;
    } else {
      z[zOffset] = x;
    }
    return;
  }

  // calculate two coefficients per thread
  if (threadIdx.x != 0) {
    const int i = threadIdx.x;
    const T aPlus2i = a + T(2) * T(i);
    sharedMem[2 * i] = T(i) * (b - T(i)) * x / ((aPlus2i - T(1)) * aPlus2i);
    sharedMem[2 * i + 1] = -(a + T(i)) * (a + b + T(i)) * x / ((aPlus2i + T(1)) * aPlus2i);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    const T gammaPart = static_cast<T>(lgamma(a)) + static_cast<T>(lgamma(b)) - static_cast<T>(lgamma(a + b));
    const T front = math::sd_exp<T, T>(math::sd_log<T, T>(x) * a + math::sd_log<T, T>(T(1) - x) * b - gammaPart);

    sharedMem[0] = T(1) - (a + b) * x / (a + T(1));
    sharedMem[1] = T(1);

    z[zOffset] = front * continuedFractionCuda(a, b, x) / a;

    if (symmCond) {  // symmetry relation
      z[zOffset] = T(1) - z[zOffset];
    }
  }
}
///////////////////////////////////////////////////////////////////
template <typename T>
static void betaIncForArrayCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                        const cudaStream_t* stream, const void* va, const LongType* aShapeInfo,
                                        const void* vb, const LongType* bShapeInfo, const void* vx,
                                        const LongType* xShapeInfo, void* vz, const LongType* zShapeInfo) {
  betaIncForArrayCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(va, aShapeInfo, vb, bShapeInfo, vx,
                                                                                 xShapeInfo, vz, zShapeInfo);
  sd::DebugHelper::checkGlobalErrorCode("betaInc  failed");

}

///////////////////////////////////////////////////////////////////
// overload betaInc for arrays, shapes of a, b and x must be the same !!!
void betaInc(LaunchContext* context, NDArray& a, NDArray& b, NDArray& x, NDArray& output) {
  dim3 launchDims = getBetaInc(maxIter,output.lengthOf(),output.sizeOfT());

  const auto xType = x.dataType();

  PointersManager manager(context, "betaInc");

  NDArray::prepareSpecialUse({&output}, {&a, &b, &x});
  BUILD_SINGLE_SELECTOR(xType, betaIncForArrayCudaLauncher,
                        (launchDims.y, launchDims.x, launchDims.z, context->getCudaStream(), a.specialBuffer(),
                            a.specialShapeInfo(), b.specialBuffer(), b.specialShapeInfo(), x.specialBuffer(),
                            x.specialShapeInfo(), output.specialBuffer(), output.specialShapeInfo()),
                        SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({&output}, {&a, &b, &x});

  manager.synchronize();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
