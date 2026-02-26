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
// Global padded operator new/delete for libnd4j
//
// C++ ops in libnd4j can overrun small heap allocations by a few bytes,
// corrupting adjacent glibc malloc chunk metadata (stored inline, just
// before the next allocation). This causes "double free or corruption
// (!prev)" SIGABRT on later free/malloc calls.
//
// jemalloc avoids this (slab-based, no inline metadata), but we cannot
// require jemalloc everywhere. Instead, we over-allocate every global
// operator new by 4KB so that small overruns land in padding rather than
// in the next chunk's metadata.
//
// Hidden visibility ensures these replacements are scoped to libnd4j.so
// and do not leak into the host JVM or other shared libraries.
// On Android, the NDK libc++ declares operator new/delete with "default"
// visibility — using "hidden" here causes a visibility mismatch error.
//

#include <cstdlib>
#include <new>

// 4KB padding — enough to absorb any realistic op overrun
static constexpr size_t GLOBAL_NEW_PADDING = 4096;

#if defined(__ANDROID__) || defined(__APPLE__)
#define SD_ALLOC_VIS
#else
#define SD_ALLOC_VIS __attribute__((visibility("hidden")))
#endif

SD_ALLOC_VIS
void* operator new(size_t size) {
  void* p = std::malloc(size + GLOBAL_NEW_PADDING);
  if (!p) throw std::bad_alloc();
  return p;
}

SD_ALLOC_VIS
void* operator new[](size_t size) {
  void* p = std::malloc(size + GLOBAL_NEW_PADDING);
  if (!p) throw std::bad_alloc();
  return p;
}

SD_ALLOC_VIS
void* operator new(size_t size, const std::nothrow_t&) noexcept {
  return std::malloc(size + GLOBAL_NEW_PADDING);
}

SD_ALLOC_VIS
void* operator new[](size_t size, const std::nothrow_t&) noexcept {
  return std::malloc(size + GLOBAL_NEW_PADDING);
}

SD_ALLOC_VIS
void operator delete(void* ptr) noexcept {
  std::free(ptr);
}

SD_ALLOC_VIS
void operator delete[](void* ptr) noexcept {
  std::free(ptr);
}

SD_ALLOC_VIS
void operator delete(void* ptr, const std::nothrow_t&) noexcept {
  std::free(ptr);
}

SD_ALLOC_VIS
void operator delete[](void* ptr, const std::nothrow_t&) noexcept {
  std::free(ptr);
}

// Sized delete variants (C++14)
SD_ALLOC_VIS
void operator delete(void* ptr, size_t) noexcept {
  std::free(ptr);
}

SD_ALLOC_VIS
void operator delete[](void* ptr, size_t) noexcept {
  std::free(ptr);
}
