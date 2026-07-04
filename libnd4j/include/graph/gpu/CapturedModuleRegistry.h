/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

#ifndef LIBND4J_CAPTUREDMODULEREGISTRY_H
#define LIBND4J_CAPTUREDMODULEREGISTRY_H

#include <mutex>
#include <unordered_map>

namespace sd {
namespace graph {
namespace modreg {

/**
 * Refcount registry for GPU modules (CUmodule) referenced by captured CUDA
 * graphs. Two holder classes exist:
 *
 *  - the OWNER: the compile cache / compiled-kernel teardown that loaded the
 *    module (implicit single ref — not stored here);
 *  - HANDLES: CUDA graph replay handles whose captured kernel nodes bake the
 *    module's functions. Each capture retains once per handle.
 *
 * The module may only be unloaded when BOTH the owner released it AND no
 * handle refs remain. TritonGraphBackend is a singleton whose compiled cache
 * outlives plans, so a plan-handle death must NOT unload a cache-shared
 * module (later plans cache-hit it → CUDA 98 'invalid device function'),
 * and cache eviction must NOT unload while a live graph bakes it (host-side
 * SIGSEGV in cudaGraphLaunch). This registry arbitrates both.
 */
struct ModuleEntry {
  int handleRefs = 0;
  bool ownerReleased = false;
};

inline std::mutex& moduleRegistryMutex() {
  static std::mutex m;
  return m;
}

inline std::unordered_map<void*, ModuleEntry>& moduleRegistry() {
  static std::unordered_map<void*, ModuleEntry> r;
  return r;
}

/** A replay handle captured a kernel from this module: hold it alive. */
inline void retainForHandle(void* module) {
  if (module == nullptr) return;
  std::lock_guard<std::mutex> lock(moduleRegistryMutex());
  moduleRegistry()[module].handleRefs++;
}

/**
 * A replay handle died (or a discarded capture dropped its retain).
 * @return true when the CALLER must unload the module now (last ref, owner gone).
 */
inline bool releaseFromHandle(void* module) {
  if (module == nullptr) return false;
  std::lock_guard<std::mutex> lock(moduleRegistryMutex());
  auto it = moduleRegistry().find(module);
  if (it == moduleRegistry().end()) return false;
  it->second.handleRefs--;
  if (it->second.handleRefs <= 0 && it->second.ownerReleased) {
    moduleRegistry().erase(it);
    return true;
  }
  return false;
}

/**
 * The owning cache/teardown is done with the module.
 * @return true when the CALLER must unload now (no handle refs outstanding);
 *         false when live handles remain — the last handle unloads instead.
 */
inline bool releaseFromOwner(void* module) {
  if (module == nullptr) return false;
  std::lock_guard<std::mutex> lock(moduleRegistryMutex());
  auto it = moduleRegistry().find(module);
  if (it == moduleRegistry().end()) return true;  // never captured → normal unload
  if (it->second.handleRefs <= 0) {
    moduleRegistry().erase(it);
    return true;
  }
  it->second.ownerReleased = true;
  return false;
}

}  // namespace modreg
}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_CAPTUREDMODULEREGISTRY_H
