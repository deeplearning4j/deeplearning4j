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
//
//
#include <helpers/DebugHelper.h>
#include <loops/type_conversions.h>
#include <types/types.h>

namespace sd {
template <typename S, typename T>
void TypeCast::convertGenericCuda(Pointer *extras, void *dx, LongType N, void *dz) {
  auto stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

  sd::convertKernel<S, T><<<256, 1024, 1024, *stream>>>(dx, N, dz);
  DebugHelper::checkErrorCode(stream, "convertGeneric(...) failed");
};

template <typename S, typename T>
SD_DEVICE void convertKernelGeneric(S *x, LongType N, T *z) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (LongType i = tid; i < N; i += blockDim.x * gridDim.x) {
    // despite it's stupid, it simplifies conversion to bottom dtypes
    // FIXME: get rid of through-float though
    z[i] = static_cast<T>(static_cast<float>(x[i]));
  }
};

  // Define this to more rigorously avoid bank conflicts, even at the lower (root) levels of the tree
  //#define ZERO_BANK_CONFLICTS

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + (index) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#endif

#ifdef CHECK_BANK_CONFLICTS
#define TEMP(index) CUT_BANK_CHECKER(temp, index)
#else
#define TEMP(index) temp[index]
#endif

template <bool isNP2>
SD_DEVICE void loadSharedChunkFromMem(int *s_data, const int *g_idata, int n, int baseIndex, int &ai, int &bi,
                                      int &mem_ai, int &mem_bi, int &bankOffsetA, int &bankOffsetB) {
  int thid = threadIdx.x;
  mem_ai = baseIndex + threadIdx.x;
  mem_bi = mem_ai + blockDim.x;

  ai = thid;
  bi = thid + blockDim.x;

  // compute spacing to avoid bank conflicts
  bankOffsetA = CONFLICT_FREE_OFFSET(ai);
  bankOffsetB = CONFLICT_FREE_OFFSET(bi);

  // Cache the computational window in shared memory
  // pad values beyond n with zeros
  s_data[ai + bankOffsetA] = g_idata[mem_ai];

  if (isNP2) {  // compile-time decision
    s_data[bi + bankOffsetB] = (bi < n) ? g_idata[mem_bi]  : static_cast<LongType>(0) ;
  } else {
    s_data[bi + bankOffsetB] = g_idata[mem_bi];
  }
}

template <bool isNP2>
SD_DEVICE void storeSharedChunkToMem(int *g_odata, int *s_data, int n, int ai, int bi, int mem_ai, int mem_bi,
                                     int bankOffsetA, int bankOffsetB) {
  __syncthreads();

  // write results to global memory
  g_odata[mem_ai] = s_data[ai + bankOffsetA];
  if (isNP2) {  // compile-time decision
    if (bi < n) g_odata[mem_bi] = s_data[bi + bankOffsetB];
  } else {
    g_odata[mem_bi] = s_data[bi + bankOffsetB];
  }
}

template <bool storeSum>
SD_DEVICE void clearLastElement(int *s_data, int *g_blockSums, int blockIndex) {
  if (threadIdx.x == 0) {
    int index = (blockDim.x << 1) - 1;
    index += CONFLICT_FREE_OFFSET(index);

    if (storeSum) {  // compile-time decision
      // write this block's total sum to the corresponding index in the blockSums array
      g_blockSums[blockIndex] = s_data[index];
    }

    // zero the last element in the scan so it will propagate back to the front
    s_data[index] = 0;
  }
}

SD_DEVICE unsigned int buildSum(int *s_data) {
  unsigned int thid = threadIdx.x;
  unsigned int stride = 1;

  // build the sum in place up the tree
  for (int d = blockDim.x; d > 0; d >>= 1) {
    __syncthreads();

    if (thid < d) {
      int i = __mul24(__mul24(2, stride), thid);
      int ai = i + stride - 1;
      int bi = ai + stride;

      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      s_data[bi] += s_data[ai];
    }

    stride *= 2;
  }

  return stride;
}

SD_DEVICE void scanRootToLeaves(int *s_data, unsigned int stride) {
  unsigned int thid = threadIdx.x;

  // traverse down the tree building the scan in place
  for (int d = 1; d <= blockDim.x; d *= 2) {
    stride >>= 1;

    __syncthreads();

    if (thid < d) {
      int i = __mul24(__mul24(2, stride), thid);
      int ai = i + stride - 1;
      int bi = ai + stride;

      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      float t = s_data[ai];
      s_data[ai] = s_data[bi];
      s_data[bi] += t;
    }
  }
}

template <bool storeSum>
SD_DEVICE void prescanBlock(int *data, int blockIndex, int *blockSums) {
  int stride = buildSum(data);  // build the sum in place up the tree
  clearLastElement<storeSum>(data, blockSums, (blockIndex == 0) ? blockIdx.x : blockIndex);
  scanRootToLeaves(data, stride);  // traverse down tree to build the scan
}

template <bool storeSum, bool isNP2>
SD_KERNEL void prescan(int *g_odata, const int *g_idata, int *g_blockSums, int n, int blockIndex, int baseIndex) {
  int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
  extern __shared__ int s_data[];

  // load data into shared memory
  loadSharedChunkFromMem<isNP2>(reinterpret_cast<int *>(s_data), g_idata, n,
                                (baseIndex == 0) ? __mul24(blockIdx.x, (blockDim.x << 1)) : baseIndex, ai, bi, mem_ai,
                                mem_bi, bankOffsetA, bankOffsetB);

  // scan the data in each block
  prescanBlock<storeSum>(s_data, blockIndex, g_blockSums);

  // write results to device memory
  storeSharedChunkToMem<isNP2>(g_odata, s_data, n, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB);
}

SD_KERNEL void uniformAdd(int *g_data, int *uniforms, int n, int blockOffset, int baseIndex) {
  __shared__ float uni;
  if (threadIdx.x == 0) uni = uniforms[blockIdx.x + blockOffset];

  unsigned int address = __mul24(blockIdx.x, (blockDim.x << 1)) + baseIndex + threadIdx.x;

  __syncthreads();

  // note two adds per thread
  g_data[address] += uni;
  g_data[address + blockDim.x] += (threadIdx.x + blockDim.x < n) * uni;
}

/*
 * This kernel does prefix sum in parallel, to calculate offsets for each block
 */
template <typename T>
SD_DEVICE inline void encoderKernelP2Generic(void *dx, LongType n, void *dz) {
  // TODO: to be remove
}

//////////////////////////////////////////////////////////////////////////
/*
 * PLEASE NOTE: This kernel doesn't allow loop for data. Basically: grid will be huge.
 */
template <typename T>
SD_KERNEL static void execEncoderKernelP1(const void *dx, LongType N, void *dz, float threshold) {
  auto x = reinterpret_cast<const T *>(dx);
  auto z = reinterpret_cast<int *>(dz);

  // basically, for phase One we want do calculation: how many eligible values we have, and which blocks will be holding
  // data
  LongType tid = blockIdx.x * blockDim.x + threadIdx.x;

  int pass = tid < N && math::sd_abs<T,T>(x[tid]) >= static_cast<T>(threshold) ? static_cast<LongType>(1)   : static_cast<LongType>(0) ;
  int bp = __syncthreads_count(pass);

  if (threadIdx.x == 0) {
    // saving out per-block passes
    z[blockIdx.x + 1] = bp;

    // saving out sum
    atomicAdd(&z[0], bp);
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
SD_HOST void encoderKernelP1Generic(dim3 &launchDims, cudaStream_t *stream, const void *dx, LongType N, void *dz,
                                    float threshold) {
  execEncoderKernelP1<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(dx, N, dz, threshold);
  DebugHelper::checkErrorCode(stream, "encoderP1(...) failed");
}
BUILD_SINGLE_TEMPLATE(template void encoderKernelP1Generic,
                      (dim3 & launchDims, cudaStream_t *stream, const void *dx, sd::LongType N, void *dz,
                       float threshold),
                      SD_FLOAT_TYPES);

//////////////////////////////////////////////////////////////////////////
/*
 * PLEASE NOTE: This kernel doesn't allow loop for data. Basically: grid will be huge.
 *
 * Based on: https://github.com/knotman90/cuStreamComp <-- efficient CUDA stream compaction algorithm
 */
template <typename T>
SD_KERNEL static void execEncoderKernelP3(void *dx, int *offsets, LongType N, void *dz) {
  auto x = reinterpret_cast<T *>(dx);
  auto z = reinterpret_cast<int *>(dz);

  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ int warpTotals[];

  // fetch block offset only once
  __shared__ float threshold;
  __shared__ FloatBits fb;
  __shared__ int bo;
  __shared__ int limit;
  if (threadIdx.x == 0) {
    limit = z[0];
    fb.i_ = z[2];
    threshold = fb.f_;
    bo = offsets[blockIdx.x];
  }
  __syncthreads();

  // out-of-limit threads do not play here
  auto value = tid < N ? x[tid] : (T)0.f;

  // out-of-limit threads just declare they have no changes
  auto pred = tid >= N ? static_cast<LongType>(0)  : math::sd_abs<T,T>(value) >= static_cast<T>(threshold) ? static_cast<LongType>(1)   : static_cast<LongType>(0) ;
  auto w_i = threadIdx.x / warpSize;  // warp index (or, warp number) - index of the Warp within TOTAL_WARPS
  auto t_i = threadIdx.x % warpSize;  // thread index within a warp
  unsigned int t_m = INT_MAX >> (warpSize - t_i - 1);  // thread mask (ERROR IN THE PAPER minus one is required)

  int b = __ballot_sync(t_m, pred);  // balres = number whose ith bit isone if the ith's thread pred is true masked up
                                     // to the current index in warp
  auto t_u = __popc(b);  // popc count the number of bit one. simply count the number predicated true BEFORE MY INDEX

  if (t_i == warpSize - 1) warpTotals[w_i] = t_u + pred;

  __syncthreads();

  int w_i_u = 0;
  for (int j = 0; j <= 5; j++) {
    unsigned int b_j =
        __ballot_sync(t_m, warpTotals[t_i] & pow2i(j));  //# of the ones in the j'th digit of the warp offsets
    w_i_u += (__popc(b_j) << j);
  }

  // we just ignore all results coming from non-0 threads
  if (w_i == 0 && t_i < blockDim.x / warpSize) warpTotals[t_i] = w_i_u;

  __syncthreads();

  // pred is always false if we're out-of-limits
  if (pred) {
    int idx = t_u + warpTotals[w_i] + bo + 4;
    if (idx < limit + 4) {
      z[idx] = value > static_cast<T>(0.0f) ? tid + 1 : -(tid + 1);
      x[tid] = value > static_cast<T>(0.0f) ? x[tid] - threshold : x[tid] + threshold;
    }
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
SD_HOST void encoderKernelP3Generic(dim3 &launchDims, cudaStream_t *stream, void *dx, int *offsets, LongType N,
                                    void *dz) {
  execEncoderKernelP3<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(dx, offsets, N, dz);
  DebugHelper::checkErrorCode(stream, "encoderP3(...) failed");
}
BUILD_SINGLE_TEMPLATE(template void encoderKernelP3Generic,
                      (dim3 & launchDims, cudaStream_t *stream, void *dx, int *offsets, sd::LongType N, void *dz),
                      SD_FLOAT_TYPES);

//////////////////////////////////////////////////////////////////////////
/*
 *   This kernel handles decode from sparse threshold array, to dense array
 *
 *   PLEASE NOTE: Z is expected to be memset to 0
 */
template <typename T>
SD_KERNEL static void execDecoderKernel(const void *dx, LongType N, void *dz) {
  auto x = reinterpret_cast<const int *>(dx);
  auto z = reinterpret_cast<T *>(dz);

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float threshold;
  __shared__ int limit;

  __shared__ FloatBits fb;
  if (threadIdx.x == 0) {
    limit = x[0];
    fb.i_ = x[2];
    threshold = fb.f_;
  }
  __syncthreads();

  for (int e = tid; e < limit; e += blockDim.x * gridDim.x) {
    int el = x[e + 4];
    int ael = sd::math::sd_abs<int,int>(el) - 1;

    // TODO: investigate, if += would work better here, as in "decoded accumulation"
    z[ael] += el > 0 ? threshold : -threshold;
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
SD_HOST void decoderKernelGeneric(dim3 &launchDims, cudaStream_t *stream, const void *dx, LongType N, void *dz) {
  execDecoderKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(dx, N, dz);
  DebugHelper::checkErrorCode(stream, "execDecoder(...) failed");
}
BUILD_SINGLE_TEMPLATE(template void decoderKernelGeneric,
                      (dim3 & launchDims, cudaStream_t *stream, const void *dx, sd::LongType N, void *dz),
                      SD_FLOAT_TYPES);

//////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL static void execCudaEncodeBitmapKernel(void *vdx, LongType N, int *dz, int *scalar, int *reductionBuffer,
                                                 float threshold) {
  auto dx = reinterpret_cast<T *>(vdx);
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  T off(0.0f);
  __shared__ int counter;
  __shared__ int *shmem;
  __shared__ T *vals;
  if (threadIdx.x == 0) {
    extern __shared__ char mem[];
    shmem = reinterpret_cast<int *>(mem);
    vals = reinterpret_cast<T *>(shmem + blockDim.x);
    counter = 0;
  }
  __syncthreads();

  LongType loopRemainder = N % (blockDim.x * gridDim.x);
  LongType loopLimit = N + (blockDim.x * gridDim.x - loopRemainder);

  for (LongType i = tid; i < loopLimit; i += blockDim.x * gridDim.x) {
    // all threads in block reading stuff
    T val = i < N ? dx[i] : off;
    T abs = math::sd_abs<T,T>(val);

    int byteId = i / 16 + 4;
    int bitId = i % 16;

    shmem[threadIdx.x] = 0;
    vals[threadIdx.x] = val;

    if (abs >= static_cast<T>(threshold) && i < N) {
      shmem[threadIdx.x] = 1 << (bitId);
      atomicAdd(&counter, 1);
      if (val < static_cast<T>(0.0f)) {
        shmem[threadIdx.x] |= 1 << (bitId + 16);
        vals[threadIdx.x] += static_cast<T>(threshold);
      } else {
        vals[threadIdx.x] -= static_cast<T>(threshold);
      }
    } else if (abs >= static_cast<T>(threshold) / static_cast<T>(2.0f) && val < static_cast<T>(0.0f) && i < N) {
      atomicAdd(&counter, 1);
      shmem[threadIdx.x] = 1 << (bitId + 16);

      vals[threadIdx.x] += static_cast<T>(threshold) / static_cast<T>(2.0f);
    }
    __syncthreads();

    if (threadIdx.x % 16 == 0 && i < N) {
      int byte = 0;
      for (int e = 0; e < 16; e++) {
        if (i + e >= N) continue;

        byte |= shmem[threadIdx.x + e];
      }
      dz[byteId] = byte;
    }
    __syncthreads();

    if (i < N) dx[i] = vals[threadIdx.x];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(scalar, counter);
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
SD_HOST void cudaEncodeBitmapGeneric(dim3 &launchDims, cudaStream_t *stream, void *vdx, LongType N, int *dz,
                                     int *scalar, int *reductionBuffer, float threshold) {
  execCudaEncodeBitmapKernel<T>
      <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(vdx, N, dz, scalar, reductionBuffer, threshold);
  DebugHelper::checkErrorCode(stream, "encodeBitmap(...) failed");
}
BUILD_SINGLE_TEMPLATE(template void cudaEncodeBitmapGeneric,
                      (dim3 & launchDims, cudaStream_t *stream, void *vdx, sd::LongType N, int *dz, int *scalar,
                       int *reductionBuffer, float threshold),
                      SD_FLOAT_TYPES);

//////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL static void execCudaDecodeBitmapKernel(const void *dx, LongType N, void *vdz) {
  auto dz = static_cast<T *>(vdz);

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ T *shmem;
  __shared__ FloatBits fb;
  __shared__ float threshold;
  __shared__ const int *x;
  if (threadIdx.x == 0) {
    extern __shared__ char mem[];
    shmem = reinterpret_cast<T *>(mem);
    x = reinterpret_cast<const int *>(dx);
    fb.i_ = x[2];
    threshold = fb.f_;
  }
  __syncthreads();

  int lim = N / 16 + 5;
  for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
    int byteId = i / 16 + 4;
    shmem[threadIdx.x] = dz[i];
    __syncthreads();

    if (threadIdx.x % 16 == 0) {
      int byte = x[byteId];

      for (int e = 0; e < 16; e++) {
        if (i + e >= N) continue;

        int bitId = (i + e) % 16;

        bool hasBit = (byte & 1 << (bitId)) != 0;
        bool hasSign = (byte & 1 << (bitId + 16)) != 0;

        if (hasBit) {
          if (hasSign)
            shmem[threadIdx.x + bitId] -= threshold;
          else
            shmem[threadIdx.x + bitId] += threshold;
        } else if (hasSign) {
          shmem[threadIdx.x + bitId] -= threshold / 2;
        }
      }
    }
    __syncthreads();

    dz[i] = shmem[threadIdx.x];
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
SD_HOST void cudaDecodeBitmapGeneric(dim3 &launchDims, cudaStream_t *stream, const void *dx, LongType N,
                                     void *vdz) {
  execCudaDecodeBitmapKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(dx, N, vdz);
  DebugHelper::checkErrorCode(stream, "cudeDecodeBitmap(...) failed");
}
BUILD_SINGLE_TEMPLATE(template void cudaDecodeBitmapGeneric,
                      (dim3 & launchDims, cudaStream_t *stream, const void *dx, sd::LongType N, void *vdz),
                      SD_FLOAT_TYPES);

template <bool storeSum, bool isNP2>
SD_HOST void prescanLauncher(dim3 &blocks, dim3 &threads, int shmem, cudaStream_t *stream, int *g_odata,
                             const int *g_idata, int *g_blockSums, int n, int blockIndex, int baseIndex) {
  shmem = sd::math::sd_max<int>(shmem, 16384);
  prescan<storeSum, isNP2>
      <<<blocks, threads, shmem, *stream>>>(g_odata, g_idata, g_blockSums, n, blockIndex, baseIndex);
  sd::DebugHelper::checkErrorCode(stream, "prescanLauncher  failed");

};

template <typename S, typename T>
SD_KERNEL void convertKernel(void *dx, LongType N, void *dz) {
  auto x = reinterpret_cast<S *>(dx);
  auto z = reinterpret_cast<T *>(dz);

  sd::convertKernelGeneric(x, N, z);
}

#define LIBND4J_BOOLS_LOCAL (randomName0, 0), (randomName1, 1)

BUILD_DOUBLE_TEMPLATE(template void TypeCast::convertGenericCuda,
                      (sd::Pointer * extras, void *dx, sd::LongType N, void *dz), SD_COMMON_TYPES_ALL,
                      SD_COMMON_TYPES_ALL);
BUILD_DOUBLE_TEMPLATE(template void prescanLauncher,
                      (dim3 & blocks, dim3 &threads, int shmem, cudaStream_t *stream, int *g_odata, const int *g_idata,
                       int *g_blockSums, int n, int blockIndex, int baseIndex),
                      LIBND4J_BOOLS_LOCAL, LIBND4J_BOOLS_LOCAL);

#undef LIBND4J_BOOLS_LOCAL
}  // namespace sd
