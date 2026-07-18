/* ******************************************************************************
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

#pragma once

//
// Device dispatch header for ResourceBinder operations.
// CUDA implementations live in ResourceBinder_cuda.cu and Vulkan implementations
// live in graph/vulkan/VulkanResourceBinder.cpp. Host builds use inline stubs.
//

#include <system/common.h>
#include <cstddef>

namespace sd {
namespace graph {

#if defined(SD_CUDA)
namespace cuda {
#elif defined(SD_VULKAN)
namespace vulkan {
#endif

#if defined(SD_CUDA) || defined(SD_VULKAN)

// ── TLS stream management ────────────────────────────────────────────────
void* ResourceBinder_getDspExecutionStream();
void  ResourceBinder_setDspExecutionStream(void* stream);

// ── Event lifecycle ──────────────────────────────────────────────────────
void* ResourceBinder_createCompletionEvent();
void  ResourceBinder_destroyCompletionEvent(void* event);
void  ResourceBinder_recordEvent(void* event, void* stream);
void  ResourceBinder_streamWaitEvent(void* stream, void* event);

// ── Stream lifecycle ─────────────────────────────────────────────────────
void* ResourceBinder_createStream();
void  ResourceBinder_destroyStream(void* stream, int deviceId);

// ── Device memory (workspaces, arg tables) ───────────────────────────────
void* ResourceBinder_deviceAlloc(size_t bytes, int deviceId);
void  ResourceBinder_deviceFree(void* ptr, int deviceId);

// ── Pinned host memory (staging buffers) ─────────────────────────────────
void* ResourceBinder_pinnedAlloc(size_t bytes);
void  ResourceBinder_pinnedFree(void* ptr);

// ── Async memcpy ─────────────────────────────────────────────────────────
void  ResourceBinder_memcpyD2HAsync(void* dst, const void* src, size_t bytes, void* stream);

#if defined(SD_CUDA)
}  // namespace cuda
using cuda::ResourceBinder_getDspExecutionStream;
using cuda::ResourceBinder_setDspExecutionStream;
using cuda::ResourceBinder_createCompletionEvent;
using cuda::ResourceBinder_destroyCompletionEvent;
using cuda::ResourceBinder_recordEvent;
using cuda::ResourceBinder_streamWaitEvent;
using cuda::ResourceBinder_createStream;
using cuda::ResourceBinder_destroyStream;
using cuda::ResourceBinder_deviceAlloc;
using cuda::ResourceBinder_deviceFree;
using cuda::ResourceBinder_pinnedAlloc;
using cuda::ResourceBinder_pinnedFree;
using cuda::ResourceBinder_memcpyD2HAsync;
#elif defined(SD_VULKAN)
}  // namespace vulkan
using vulkan::ResourceBinder_getDspExecutionStream;
using vulkan::ResourceBinder_setDspExecutionStream;
using vulkan::ResourceBinder_createCompletionEvent;
using vulkan::ResourceBinder_destroyCompletionEvent;
using vulkan::ResourceBinder_recordEvent;
using vulkan::ResourceBinder_streamWaitEvent;
using vulkan::ResourceBinder_createStream;
using vulkan::ResourceBinder_destroyStream;
using vulkan::ResourceBinder_deviceAlloc;
using vulkan::ResourceBinder_deviceFree;
using vulkan::ResourceBinder_pinnedAlloc;
using vulkan::ResourceBinder_pinnedFree;
using vulkan::ResourceBinder_memcpyD2HAsync;
#endif

#else  // Host-only stubs

static inline void* ResourceBinder_getDspExecutionStream() { return nullptr; }
static inline void  ResourceBinder_setDspExecutionStream(void*) {}

static inline void* ResourceBinder_createCompletionEvent() { return nullptr; }
static inline void  ResourceBinder_destroyCompletionEvent(void*) {}
static inline void  ResourceBinder_recordEvent(void*, void*) {}
static inline void  ResourceBinder_streamWaitEvent(void*, void*) {}

static inline void* ResourceBinder_createStream() { return nullptr; }
static inline void  ResourceBinder_destroyStream(void*, int) {}

static inline void* ResourceBinder_deviceAlloc(size_t, int) { return nullptr; }
static inline void  ResourceBinder_deviceFree(void*, int) {}

static inline void* ResourceBinder_pinnedAlloc(size_t) { return nullptr; }
static inline void  ResourceBinder_pinnedFree(void*) {}

static inline void  ResourceBinder_memcpyD2HAsync(void*, const void*, size_t, void*) {}

#endif

}  // namespace graph
}  // namespace sd
