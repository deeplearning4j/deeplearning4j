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

#ifndef LIBND4J_POINTER_VALIDATION_H
#define LIBND4J_POINTER_VALIDATION_H

#include <cstdint>
#include <cstdio>
#include <system/common.h>

namespace sd {

static constexpr uint32_t MAGIC_DESTROYED = 0xDEADBEEF;

#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || defined(_M_ARM64)
static constexpr uintptr_t PTR_MIN_VALID = 0x10000ULL;
static constexpr uintptr_t PTR_MAX_VALID = 0x00007fffffffffffULL;
#else
static constexpr uintptr_t PTR_MIN_VALID = 0x10000UL;
static constexpr uintptr_t PTR_MAX_VALID = 0xffffffffUL;
#endif

namespace pointer_validation {

SD_INLINE uintptr_t addressForValidation(const void* ptr) {
  const auto addr = reinterpret_cast<uintptr_t>(ptr);
#if defined(__ANDROID__) && defined(__aarch64__)
  // Android uses top-byte-ignore for tagged heap pointers. The allocation tag
  // is metadata, not part of the virtual address used for range validation.
  return addr & 0x00ffffffffffffffULL;
#else
  return addr;
#endif
}

}  // namespace pointer_validation

SD_INLINE bool isValidPointer(const void* ptr) {
  const auto addr = pointer_validation::addressForValidation(ptr);
  return addr >= PTR_MIN_VALID && addr <= PTR_MAX_VALID;
}

SD_INLINE bool isAlignedPointer(const void* ptr, size_t alignment) {
  return (pointer_validation::addressForValidation(ptr) % alignment) == 0;
}

#define SD_VALIDATE_PTR(ptr, context) \
  do { \
    if (!sd::isValidPointer(ptr)) { \
      char _sd_msg[256]; \
      snprintf(_sd_msg, sizeof(_sd_msg), \
               "%s: invalid pointer %p (out of user-space range)", \
               (context), (const void*)(ptr)); \
      THROW_EXCEPTION(_sd_msg); \
    } \
  } while (0)

#define SD_VALIDATE_THIS(context) \
  do { \
    if (!sd::isValidPointer(this)) { \
      char _sd_msg[256]; \
      snprintf(_sd_msg, sizeof(_sd_msg), \
               "%s: corrupted this=%p — stale pointer (use-after-free or heap corruption)", \
               (context), (const void*)this); \
      THROW_EXCEPTION(_sd_msg); \
    } \
  } while (0)

#define SD_VALIDATE_ALIGNED(ptr, align, context) \
  do { \
    if (!sd::isAlignedPointer(ptr, align)) { \
      char _sd_msg[256]; \
      snprintf(_sd_msg, sizeof(_sd_msg), \
               "%s: misaligned pointer %p (expected %zu-byte alignment)", \
               (context), (const void*)(ptr), (size_t)(align)); \
      THROW_EXCEPTION(_sd_msg); \
    } \
  } while (0)

#define SD_VALIDATE_MAGIC(magic, expected, context) \
  do { \
    if ((magic) != (expected)) { \
      char _sd_msg[256]; \
      snprintf(_sd_msg, sizeof(_sd_msg), \
               "%s: invalid magic 0x%08x (expected 0x%08x) — use-after-free or corruption", \
               (context), (unsigned)(magic), (unsigned)(expected)); \
      THROW_EXCEPTION(_sd_msg); \
    } \
  } while (0)

}  // namespace sd

// Padded operator new/delete macro — protects adjacent glibc chunks from
// overruns by allocating SD_ALLOC_PADDING extra bytes. Place inside a
// class body (public section) to replace the per-class copy-paste blocks.
//
// The nothrow variants are hidden from JavaCPP which can't parse them.
#ifndef __JAVACPP_HACK__
#define SD_PADDED_NEW_DELETE                                                   \
  static void* operator new(size_t size) {                                     \
    return std::malloc(size + SD_ALLOC_PADDING);                               \
  }                                                                            \
  static void* operator new(size_t size, const std::nothrow_t&) noexcept {     \
    return std::malloc(size + SD_ALLOC_PADDING);                               \
  }                                                                            \
  static void operator delete(void* ptr) noexcept {                            \
    std::free(ptr);                                                            \
  }                                                                            \
  static void operator delete(void* ptr, const std::nothrow_t&) noexcept {     \
    std::free(ptr);                                                            \
  }
#else
#define SD_PADDED_NEW_DELETE
#endif

#endif  // LIBND4J_POINTER_VALIDATION_H
