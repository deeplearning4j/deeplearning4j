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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 06.02.2019
// @author raver119@gmail.com
//

#ifndef CUDAMANAGER_H
#define CUDAMANAGER_H
#include <execution/LaunchContext.h>
#include <types/types.h>

#include <string>
#include <vector>

namespace sd {

/**
 * Tracks allocation source for proper cleanup.
 * This fixes the memory leak where cudaMalloc allocations were never freed
 * because we couldn't distinguish them from workspace allocations.
 */
struct AllocatedPointer {
  void* ptr;
  bool fromCudaMalloc;  // true = needs cudaFree, false = workspace-managed

  AllocatedPointer(void* p, bool fromMalloc) : ptr(p), fromCudaMalloc(fromMalloc) {}
};

class SD_LIB_EXPORT PointersManager {
  LaunchContext* _context;
  std::vector<AllocatedPointer> _allocatedPointers;  // Track allocation source
  std::string _funcName;
  bool _workspaceWasActive;  // Track if workspace was active at construction time

 public:
  PointersManager(const LaunchContext* context, const std::string& funcName = "");

  ~PointersManager();

  void* allocateDevMem(const size_t sizeInBytes);

  void* replicatePointer(const void* src, const size_t size);

  void synchronize() const;

  template <typename T>
  void printDevContentOnHost(const void* pDev, const LongType len) const;

#ifdef SD_CUDA
  template <typename T>
  static void printDevContentOnDevFromHost(const void* pDev, const LongType len, const int tid = 0);
#endif

#ifdef __CUDACC__
  template <typename T>
  static SD_INLINE SD_DEVICE void printDevContentOnDev(const void* pDev, const LongType len, const int tid = 0) {
    if (blockIdx.x * blockDim.x + threadIdx.x != tid) return;

    printf("device print out: \n");
    for (LongType i = 0; i < len; ++i) printf("%f, ", (double)reinterpret_cast<const T*>(pDev)[i]);

    printf("\n");
  }

#endif
};

}  // namespace sd

#endif  // CUDAMANAGER_H
