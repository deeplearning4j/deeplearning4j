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

/*
 * This set of methods provides dataType conversions in all possible directions supported:
 *  FP8, FP16, FLOAT, DOUBLE, INT8, UINT8, UINT16,
 *
 * @author raver119@gmail.com
 */

#ifndef LIBND4J_TYPE_CONVERSIONS_H
#define LIBND4J_TYPE_CONVERSIONS_H


#include <math/templatemath.h>
#include <ops/ops.h>
#include <system/Environment.h>
#include <types/float16.h>
#include <types/float8.h>
#include <types/int16.h>
#include <types/int8.h>
#include <types/uint16.h>
#include <types/uint8.h>
#include <execution/Threads.h>

#define LOG_NUM_BANKS 4
#define NUM_BANKS 256
namespace sd {

typedef union {
  float f_;
  int i_;
} FloatBits;

class SD_LIB_HIDDEN TypeCast {
 public:
  template <typename S, typename T>
  static SD_INLINE SD_HOST void convertGeneric(Pointer *extras, void *dx, LongType N, void *dz) {
    auto x = reinterpret_cast<S *>(dx);
    auto z = reinterpret_cast<T *>(dz);

    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        z[i] = static_cast<T>(static_cast<float>(x[i]));
      }
    };
    samediff::Threads::parallel_for(func, 0, N);
  }

  template <typename T>
  static SD_HOST void convertToThreshold(Pointer *extras, void *dx, LongType N, void *dz);

  template <typename T>
  static SD_HOST void convertFromThreshold(Pointer *extras, const void *dx, LongType N, void *dz);

  SD_INLINE static SD_HOST LongType estimateQuantizedSize(LongType rawSize) {
    if (rawSize <= 0) THROW_EXCEPTION("Input size for quantization can't be <= 0");

    // 2 fp32 values for max/min, and rawSize number of BYTES
    return 8 + rawSize;
  }

  template <typename T>
  static SD_HOST void convertToQuantized(Pointer *extras, void *dx, LongType N, void *dz);

  template <typename T>
  static SD_HOST void convertFromQuantized(Pointer *extras, void *dx, LongType N, void *dz);

#ifdef __CUDACC__
  template <typename S, typename T>
  static SD_HOST void convertGenericCuda(Pointer *extras, void *dx, LongType N, void *dz);
#endif
};

SD_INLINE SD_HOST_DEVICE bool isPowerOfTwo(int n) { return ((n & (n - 1)) == 0); }

SD_INLINE SD_HOST_DEVICE int floorPow2(int n) {
#ifdef WIN32
  // method 2
  return 1 << static_cast<int>(logb(static_cast<float>(n)));
#else
  // method 1
  // float nf = (float)n;
  // return 1 << (((*(int*)&nf) >> 23) - 127);
  int exp;
  frexp(static_cast<float>(n), &exp);
  return 1 << (exp - 1);
#endif
}

#ifdef __CUDACC__
SD_DEVICE __inline__ int pow2i(int e) { return 1 << e; }

template <typename T>
SD_HOST void encoderKernelP1Generic(dim3 &launchDims, cudaStream_t *stream, const void *dx, LongType N, void *dz,
                                    float threshold);

template <typename T>
SD_HOST void encoderKernelP3Generic(dim3 &launchDims, cudaStream_t *stream, void *dx, int *offsets, LongType N,
                                    void *dz);




SD_KERNEL void uniformAdd(int *g_data, int *uniforms, int n, int blockOffset, int baseIndex);

template <bool storeSum, bool isNP2>
SD_KERNEL void prescan(int *g_odata, const int *g_idata, int *g_blockSums, int n, int blockIndex, int baseIndex);

template <bool storeSum, bool isNP2>
SD_HOST void prescanLauncher(dim3 &blocks, dim3 &threads, int shmem, cudaStream_t *stream, int *g_odata,
                             const int *g_idata, int *g_blockSums, int n, int blockIndex, int baseIndex);

template <typename S, typename T>
SD_KERNEL void convertKernel(void *dx, LongType N, void *dz);
#endif

}  // namespace sd

#endif  // LIBND4J_TYPE_CONVERSIONS_H
