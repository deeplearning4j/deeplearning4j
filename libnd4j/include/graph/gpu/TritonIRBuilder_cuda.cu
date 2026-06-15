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

//
// TritonIRBuilder — CUDA-specific helpers that require NVCC compilation.
// Separated from TritonIRBuilder.cpp to avoid NVCC/MLIR header conflicts.
//

#include <config.h>

#if HAVE_TRITON

#include <system/Environment.h>

#include <cuda_runtime.h>

namespace sd {
namespace graph {
namespace ir_builder_internal {

int getSectionedCooperativeTargetBlocks() {
  int configured = sd::Environment::getInstance().tritonCoopTargetBlocks();
  if (configured > 0) return configured;

  int device = 0;
  if (cudaGetDevice(&device) == cudaSuccess) {
    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, device) == cudaSuccess &&
        props.multiProcessorCount > 0) {
      // One cooperative block per SM is the strictest guaranteed residency target.
      return props.multiProcessorCount;
    }
  }

  // Conservative default when device query is unavailable.
  return 128;
}

}  // namespace ir_builder_internal
}  // namespace graph
}  // namespace sd

#endif  // HAVE_TRITON
