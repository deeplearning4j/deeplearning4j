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
// @author raver119@gmail.com
// @author Yurii Shyrma, created on 15.11.2018
//
#include <loops/special_kernels.h>


namespace sd {

///////////////////////////////////////////////////////////////////////
/**
 * This kernel accumulates X arrays, and stores z into Z
 *
 * @tparam T
 * @param x
 * @param z
 * @param n
 * @param length
 */
template <typename T>
SD_DEVICE void accumulateKernel(void **vx, void *vz, int n, const LongType length) {
  auto x = reinterpret_cast<T **>(vx);
  auto z = reinterpret_cast<T *>(vz);

  __shared__ T *shmem;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char sharedmem[];
    shmem = (T *)sharedmem;
  }
  __syncthreads();

  for (int r = blockDim.x * blockIdx.x; r < length; r += blockDim.x * gridDim.x) {
    shmem[threadIdx.x] = 0.0f;

    LongType baseIdx = r;

    // aggregation step, we roll over all arrays
    for (int ar = 0; ar < n; ar++) {
      T *cdata = (T *)x[ar];
      cdata += baseIdx;

      if (baseIdx + threadIdx.x < length) shmem[threadIdx.x] += cdata[threadIdx.x];
    }

    T *wdata = z + baseIdx;

    // saving accumulated values
    if (baseIdx + threadIdx.x < length) wdata[threadIdx.x] = shmem[threadIdx.x];
  }
}

///////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void execAccumulateKernel(void **vx, void *vz, int n, const LongType length) {
  accumulateKernel<T>(vx, vz, n, length);
}

///////////////////////////////////////////////////////////////////////
template <typename T>
SD_HOST void accumulateKernelGeneric(dim3 &launchDims, cudaStream_t *stream, void **vx, void *vz, int n,
                                     const LongType length) {
  execAccumulateKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(vx, vz, n, length);
  DebugHelper::checkErrorCode(stream, "accumulate(...) failed");
}

BUILD_SINGLE_TEMPLATE( void accumulateKernelGeneric,
                      (dim3 & launchDims, cudaStream_t *stream, void **vx, void *vz, int n, const sd::LongType length),
                      SD_COMMON_TYPES);
}  // namespace sd
