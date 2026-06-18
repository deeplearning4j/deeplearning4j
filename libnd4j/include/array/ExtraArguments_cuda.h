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
// Dispatch shims for ExtraArguments device allocation and H2D transfer.
// CUDA implementations live in array/cuda/ExtraArgumentsCuda.cu.
// CPU stubs are provided inline here so ExtraArguments.cpp needs no #ifdef SD_CUDA.
//

#ifndef LIBND4J_EXTRAARGUMENTS_CUDA_H
#define LIBND4J_EXTRAARGUMENTS_CUDA_H

#include <system/common.h>
#include <cstddef>

#ifdef SD_CUDA

// Declarations — definitions are in ExtraArgumentsCuda.cu
namespace sd {
namespace extra_args_detail {

// Allocate 'bytes' of device memory via CudaMemoryPool for the current device.
SD_LIB_EXPORT void* extraArgsAllocDevice(size_t bytes);

// Free a pointer previously returned by extraArgsAllocDevice via CudaMemoryPool.
SD_LIB_EXPORT void extraArgsFreeDevice(void* ptr);

// Copy 'bytes' from host src to device dst, respecting CUDA graph capture state.
// During capture: routes through the capture-safe async path (extraArgsCaptureH2D).
// Outside capture: uses cudaMemcpyAsync on cudaStreamPerThread + sync.
SD_LIB_EXPORT void extraArgsCopyH2DDispatch(void* dst, const void* src, size_t bytes);

}  // namespace extra_args_detail
}  // namespace sd

#else  // CPU stubs — all inline, no separate translation unit needed

#include <cstring>
#include <cstdint>

namespace sd {
namespace extra_args_detail {

inline void* extraArgsAllocDevice(size_t bytes) {
  return new int8_t[bytes];
}

inline void extraArgsFreeDevice(void* ptr) {
  delete[] reinterpret_cast<int8_t*>(ptr);
}

// On CPU the "device pointer" IS host memory, so H2D is just a memcpy.
inline void extraArgsCopyH2DDispatch(void* dst, const void* src, size_t bytes) {
  std::memcpy(dst, src, bytes);
}

}  // namespace extra_args_detail
}  // namespace sd

#endif  // SD_CUDA

#endif  // LIBND4J_EXTRAARGUMENTS_CUDA_H
