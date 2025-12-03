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
// Created by raver119 on 30.11.17.
//
#include <exceptions/cuda_exception.h>
#include <execution/AffinityManager.h>
#include <execution/LaunchContext.h>
#include <helpers/logger.h>
#include <thread>

// NOTE: Removed thread_local to fix "cannot allocate memory in static TLS block" error
// when JavaCPP loads library via dlopen(). This error occurs because:
// 1. dlopen() loads the library at runtime (not at program startup)
// 2. thread_local variables require space in the static TLS block
// 3. The static TLS block has limited size and cannot be extended after program start
// 4. When loaded via dlopen(), the library's TLS requirements must fit in remaining space
// 5. If sanitizers or lifecycle tracking are enabled, TLS space may already be exhausted
//
// iOS/Apple/Android builds already avoid thread_local for similar reasons.
// For CPU builds, making this a non-thread-local global is acceptable because:
// - Each LaunchContext instance maintains its own thread-safe state
// - The global contextBuffers is used as a default/fallback buffer pool
// - Proper synchronization is handled at the LaunchContext level
sd::ContextBuffers contextBuffers = sd::ContextBuffers();

#if HAVE_ONEDNN
#include <dnnl.hpp>
#endif

namespace sd {

LaunchContext::~LaunchContext() {
#if HAVE_ONEDNN
  // Intentionally NOT deleting _engine to avoid use-after-free during shutdown
  // The LaunchContext objects are kept alive in a function-local static vector (contexts())
  // to survive JVM shutdown. Deleting _engine here causes crashes because
  // OneDNN's static cleanup may have already run, making the engine invalid.
  // This is safe because LaunchContext instances are never destroyed during
  // normal operation - only during process exit when memory cleanup doesn't matter.
  // delete reinterpret_cast<dnnl::engine*>(_engine);
#endif
}

// This avoids static destruction order crashes during JVM shutdown
// CRITICAL FIX: Use pointer to prevent static destructor from running.
// The vector is intentionally leaked to avoid crashes during shutdown when:
// 1. An exception is thrown that references LaunchContext
// 2. JVM shuts down and runs static destructors
// 3. The vector's destructor tries to destroy stored contexts while they're still in use
// This is safe because LaunchContexts are only created during initialization and
// process exit cleans up all memory anyway.
std::vector<LaunchContext*>& LaunchContext::contexts() {
  static std::vector<LaunchContext*>* _contexts = new std::vector<LaunchContext*>();
  return *_contexts;
}

SD_MAP_IMPL<int, std::mutex*> LaunchContext::_deviceMutexes;
std::mutex LaunchContext::_mutex;

////////////////////////////////////////////////////////////////////////
LaunchContext::LaunchContext() {
  // default constructor, just to make clang/ranlib happy
  _workspace = nullptr;
  _deviceID = 0;
#if HAVE_ONEDNN
  _engine = new dnnl::engine(dnnl::engine::kind::cpu, 0);
#endif
}

LaunchContext::LaunchContext(sd::Pointer cudaStream, sd::Pointer reductionPointer, sd::Pointer scalarPointer,
                             sd::Pointer allocationPointer) {}

static std::mutex _lock;

LaunchContext* LaunchContext::defaultContext() {
  {
    // synchronous block goes here
    std::lock_guard<std::mutex> lock(_lock);
    // TODO: we need it to be device-aware, but only once we add NUMA support for cpu
    if (LaunchContext::contexts().empty())
      LaunchContext::contexts().emplace_back(new LaunchContext());
  }

  // return context for current device
  return LaunchContext::contexts().at(0);
}

std::mutex* LaunchContext::deviceMutex() { return &_mutex; }

void LaunchContext::swapContextBuffers(ContextBuffers& buffers) {
  //
}

bool LaunchContext::isInitialized() { return true; }

void LaunchContext::releaseBuffers() {
  //
}

sd::ErrorReference* LaunchContext::errorReference() { return contextBuffers.errorReference(); }

void* LaunchContext::engine() { return _engine; }
}  // namespace sd
