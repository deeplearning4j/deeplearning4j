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
template <typename T>
SD_DEVICE void averagingKernel(void **vdx, void *vdz, int n, LongType length, bool propagate) {
  auto dx = reinterpret_cast<T **>(vdx);
  auto dz = reinterpret_cast<T *>(vdz);

  __shared__ T *shmem;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char sharedmem[];
    shmem = (T *)sharedmem;
  }
  __syncthreads();

  // each block cycles over it's own part of arrays
  for (int r = blockDim.x * blockIdx.x; r < length; r += blockDim.x * gridDim.x) {
    shmem[threadIdx.x] = (T)0.0f;

    LongType baseIdx = r;

    // aggregation step, we roll over all arrays
    for (int ar = 0; ar < n; ar++) {
      T *cdata = (T *)dx[ar];
      cdata += baseIdx;

      if (baseIdx + threadIdx.x < length) shmem[threadIdx.x] += cdata[threadIdx.x];
    }

    // average data in shared memory
    if (baseIdx + threadIdx.x < length) shmem[threadIdx.x] /= n;

    // div step & write out step
    if (dz != nullptr) {
      T *wdata = dz + baseIdx;

      if (baseIdx + threadIdx.x < length) {
        wdata[threadIdx.x] = shmem[threadIdx.x];
      }
    }

    // propagate averaged data to all arrays
    if (propagate)
      for (int ar = 0; ar < n; ar++) {
        T *cdata = (T *)dx[ar];
        cdata += baseIdx;

        if (baseIdx + threadIdx.x < length) cdata[threadIdx.x] = shmem[threadIdx.x];
      }
  }
}

///////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void execAveragingKernel(void **vdx, void *vdz, int n, LongType length, bool propagate) {
  averagingKernel<T>(vdx, vdz, n, length, propagate);
}

///////////////////////////////////////////////////////////////////////
template <typename T>
SD_HOST void averagingKernelGeneric(dim3 &launchDims, cudaStream_t *stream, void **vdx, void *vdz, int n,
                                    LongType length, bool propagate) {
  execAveragingKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(vdx, vdz, n, length, propagate);
  DebugHelper::checkErrorCode(stream, "averaging(...) failed");
}

BUILD_SINGLE_TEMPLATE(template void averagingKernelGeneric,
                      (dim3 & launchDims, cudaStream_t *stream, void **vdx, void *vdz, int n, sd::LongType length,
                       bool propagate),
                      SD_COMMON_TYPES);
}  // namespace sd
