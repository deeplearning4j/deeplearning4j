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

#ifndef LIBND4J_NVRTC_KERNEL_BUILDER_H
#define LIBND4J_NVRTC_KERNEL_BUILDER_H

#ifdef SD_CUDA

#include <array/NDArray.h>
#include <graph/NativeDynamicShapePlan.h>
#include <graph/gpu/OpCategoryTable.h>
#include <system/common.h>

#include <string>
#include <vector>

namespace sd {
namespace graph {

/**
 * Describes how a kernel parameter binds to runtime data.
 */
struct JitParamBinding {
  enum Type {
    INPUT_PTR,        // Pointer to an external input array's specialBuffer
    OUTPUT_PTR,       // Pointer to a slot output array's specialBuffer
    CROSS_SEG_PTR,    // Pointer to output from a prior segment (via outputSlots_)
    LENGTH            // int64_t element count
  };
  Type type;
  int slotIdx;            // outputSlots_[] index for OUTPUT_PTR / CROSS_SEG_PTR
  int externalInputIdx;   // externalInputs[] index for INPUT_PTR (-1 if not used)

  JitParamBinding() : type(LENGTH), slotIdx(-1), externalInputIdx(-1) {}
};

/**
 * Result of buildKernelSource(): contains the generated CUDA C++ source,
 * kernel function name, and parameter binding metadata for launch.
 */
struct JitKernelSource {
  std::string sourceCode;           // Generated CUDA C++ source
  std::string kernelName;           // __global__ function name
  std::vector<JitParamBinding> paramBindings;  // How to resolve each kernel param at launch
  bool valid;                       // false if generation failed

  JitKernelSource() : valid(false) {}
};

/**
 * Check whether all slots in [startSlot..endSlot] can be JIT-compiled
 * via NVRTC into a fused CUDA C kernel.
 *
 * Uses the unified TritonOpCategory system (OpCategoryTable.h) instead of
 * the legacy FusedElemOp codes. Accepts: BINARY_ELEMENTWISE, UNARY_ELEMENTWISE,
 * COMPARISON, LOGICAL, TERNARY, IDENTITY, CAST.
 * Rejects: MATMUL, REDUCTION, NORMALIZATION, FUSED_ATTENTION, SHAPE_MANIPULATION,
 * DATA_MOVEMENT, CONSTANT_GENERATION, UNSUPPORTED.
 *
 * @param slots       Array of NativeSlot descriptors
 * @param startSlot   First slot index (inclusive)
 * @param endSlot     Last slot index (inclusive)
 * @return true if the segment can be JIT-compiled
 */
bool canJitSegment(const NativeSlot* slots, int startSlot, int endSlot);

/**
 * Generate a CUDA C++ kernel source string for the given segment.
 *
 * The kernel processes all fusible slots in SSA order:
 * - Consecutive slot chains (output consumed by exactly one downstream slot)
 *   keep intermediates in register variables.
 * - Outputs consumed by multiple downstream slots or needed as segment outputs
 *   are stored to/loaded from global memory pointers.
 *
 * @param slots            Array of NativeSlot descriptors
 * @param startSlot        First slot index (inclusive)
 * @param endSlot          Last slot index (inclusive)
 * @param outputSlots      Current outputSlots_ array (for resolving cross-segment inputs)
 * @param totalOutputSlots Size of the outputSlots array
 * @param segmentIndex     Segment index (for unique kernel naming)
 * @return JitKernelSource with generated code and param bindings
 */
JitKernelSource buildKernelSource(const NativeSlot* slots, int startSlot, int endSlot,
                                  NDArray** outputSlots, int totalOutputSlots,
                                  int segmentIndex);

}  // namespace graph
}  // namespace sd

#endif  // SD_CUDA
#endif  // LIBND4J_NVRTC_KERNEL_BUILDER_H
