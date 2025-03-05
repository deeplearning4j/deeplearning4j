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

// Include platform-specific headers conditionally
#if defined(SD_CUDA)
#include <helpers/cuda/CudaShapeBufferCreator.h>
#include <cuda.h>
#include <cuda_runtime.h>
#endif

// Forward declare Environment class if it's used for platform detection

namespace sd {

void ShapeBufferPlatformHelper::initialize() {

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

  // Add other platforms as needed (ROCm, OpenCL, etc.)
}

} // namespace sd
