/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

#include <helpers/ShapeBufferPlatformHelper.h>
#include <helpers/cpu/CpuShapeBufferCreator.h>
#include <mutex>

// Include platform-specific headers conditionally
#if defined(SD_CUDA)
#include <helpers/cuda/CudaShapeBufferCreator.h>
#include <cuda.h>
#include <cuda_runtime.h>
#endif

// Forward declare Environment class if it's used for platform detection

namespace sd {

// This ensures ShapeBufferPlatformHelper is initialized BEFORE DirectShapeTrie or any other
// code tries to use it. The dummy struct with static member forces initialization at program startup.
struct ShapeBufferInitializer {
  ShapeBufferInitializer() {
    ShapeBufferPlatformHelper::initialize();
  }
};
static ShapeBufferInitializer _force_early_init;

void ShapeBufferPlatformHelper::initialize() {
  // Thread-safe initialization using static local mutex
  // This prevents race conditions when multiple threads call initialize() simultaneously
  static std::mutex init_mutex;
  static bool init_done = false;

  // Fast path: if already initialized, return immediately without locking
  if (init_done) {
    return;
  }

  // Slow path: acquire lock and check again
  std::lock_guard<std::mutex> lock(init_mutex);
  if (init_done) {
    return;  // Another thread completed initialization while we were waiting
  }

#if defined(SD_CUDA)
  printf("Initializing CUDA platform\n");
  fflush(stdout);
  // Switch to CUDA implementation
  ShapeBufferCreatorHelper::setCurrentCreator(&CudaShapeBufferCreator::getInstance());
#else
  printf("Initializing CPU platform\n");
  fflush(stdout);

  ShapeBufferCreatorHelper::setCurrentCreator(&CpuShapeBufferCreator::getInstance());

#endif

  // Mark as complete - must be last line after all initialization
  init_done = true;

  // Add other platforms as needed (ROCm, OpenCL, etc.)
}

} // namespace sd
