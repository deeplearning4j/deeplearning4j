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
// Dispatch header for ResourceBinder CUDA API calls.
// .cu implementations in ResourceBinder_cuda.cu.
// CPU builds get inline stubs so callers need no #ifdef guards.
//

#include <system/common.h>
#include <cstddef>

namespace sd {
namespace graph {

#ifdef SD_CUDA

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
void* ResourceBinder_deviceAlloc(size_t bytes);
void  ResourceBinder_deviceFree(void* ptr);

// ── Pinned host memory (staging buffers) ─────────────────────────────────
void* ResourceBinder_pinnedAlloc(size_t bytes);
void  ResourceBinder_pinnedFree(void* ptr);

// ── Async memcpy ─────────────────────────────────────────────────────────
void  ResourceBinder_memcpyD2HAsync(void* dst, const void* src, size_t bytes, void* stream);

#else  // CPU stubs

static inline void* ResourceBinder_getDspExecutionStream() { return nullptr; }
static inline void  ResourceBinder_setDspExecutionStream(void*) {}

static inline void* ResourceBinder_createCompletionEvent() { return nullptr; }
static inline void  ResourceBinder_destroyCompletionEvent(void*) {}
static inline void  ResourceBinder_recordEvent(void*, void*) {}
static inline void  ResourceBinder_streamWaitEvent(void*, void*) {}

static inline void* ResourceBinder_createStream() { return nullptr; }
static inline void  ResourceBinder_destroyStream(void*, int) {}

static inline void* ResourceBinder_deviceAlloc(size_t) { return nullptr; }
static inline void  ResourceBinder_deviceFree(void*) {}

static inline void* ResourceBinder_pinnedAlloc(size_t) { return nullptr; }
static inline void  ResourceBinder_pinnedFree(void*) {}

static inline void  ResourceBinder_memcpyD2HAsync(void*, const void*, size_t, void*) {}

#endif

}  // namespace graph
}  // namespace sd
