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
// Consolidated FNV-1a 64-bit hashing utilities for DSP infrastructure.
// Replaces ~8 duplicate implementations scattered across plan files.
//

#ifndef LIBND4J_DSP_HASH_UTILS_H
#define LIBND4J_DSP_HASH_UTILS_H

#include <graph/DspConstants.h>
#include <system/common.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace sd {
namespace graph {
namespace dsp {

// ─── Core FNV-1a primitives ─────────────────────────────────────────────────

// Mix raw bytes into an existing FNV-1a hash (incremental).
SD_INLINE void fnv1aMix(uint64_t& hash, const void* data, size_t size) {
  const auto* bytes = static_cast<const unsigned char*>(data);
  for (size_t i = 0; i < size; i++) {
    hash ^= static_cast<uint64_t>(bytes[i]);
    hash *= FNV1A64_PRIME;
  }
}

// Mix a single 64-bit value into an existing FNV-1a hash.
SD_INLINE void fnv1aMixValue(uint64_t& hash, uint64_t value) {
  hash ^= value;
  hash *= FNV1A64_PRIME;
}

// ─── Convenience one-shot hashing ───────────────────────────────────────────

// Hash a byte buffer from scratch.
SD_INLINE uint64_t fnv1a64(const void* data, size_t len) {
  uint64_t hash = FNV1A64_OFFSET_BASIS;
  fnv1aMix(hash, data, len);
  return hash;
}

// Hash a string from scratch.
SD_INLINE uint64_t fnv1a64String(const std::string& s) {
  return fnv1a64(s.data(), s.size());
}

// ─── Slot address hashing ───────────────────────────────────────────────────
//
// Computes FNV-1a hash of buffer addresses for a range of output slots.
// Used to verify that output buffer pointers haven't been reallocated between
// CUDA graph capture and replay — stale addresses cause SIGSEGV or corruption.
//
// bufferGetter: callable(NDArray*) -> void*  returning the relevant buffer ptr.
// Example: [](NDArray* a) { return a->specialBuffer(); }
//
template <typename NDArrayT, typename BufferGetter>
SD_INLINE uint64_t computeSlotAddrHash(NDArrayT** outputSlots, int startSlot, int endSlot,
                                        int totalSlots, BufferGetter bufferGetter) {
  uint64_t hash = FNV1A64_OFFSET_BASIS;
  for (int si = startSlot; si <= endSlot && si < totalSlots; si++) {
    void* addr = (outputSlots[si] != nullptr) ? bufferGetter(outputSlots[si]) : nullptr;
    fnv1aMixValue(hash, reinterpret_cast<uintptr_t>(addr));
  }
  return hash;
}

}  // namespace dsp
}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_DSP_HASH_UTILS_H
