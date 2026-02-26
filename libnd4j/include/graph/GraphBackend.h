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

#ifndef LIBND4J_GRAPH_BACKEND_H
#define LIBND4J_GRAPH_BACKEND_H

#include <array/NDArray.h>
#include <system/common.h>

#include <string>
#include <vector>

namespace sd {
namespace graph {

// Forward declarations — avoid circular include with NativeDynamicShapePlan.h
struct NativeSlot;
struct GraphSegment;

/**
 * Backend-agnostic compilation audit entry.
 * Tracks whether each op in a segment was compiled by the graph backend
 * or silently skipped. Skipped ops produce stale outputs on graph replay.
 */
struct CompilationAuditEntry {
  int slotIndex = -1;
  std::string opName;
  bool wasCompiled = false;    // true if backend included this op
  std::string reason;          // why it was skipped (e.g., "unmappable op kind")
};

/**
 * Abstract graph backend interface for hardware-specific graph APIs.
 *
 * All three graph APIs (CUDA Graphs, oneDNN Graph, ACL Dynamic Fusion)
 * follow the same pattern:
 * 1. Detect fusible segments in the plan
 * 2. Compile the segment into a hardware-specific graph
 * 3. Execute the compiled graph
 *
 * NativeDynamicShapePlan selects the appropriate backend at compile time:
 * - CUDA build → CudaGraphBackend
 * - CPU build with oneDNN → OneDnnGraphBackend
 * - CPU build with ACL → AclGraphBackend
 * - Fallback → slot-by-slot execution (always works)
 */
class SD_LIB_EXPORT GraphBackend {
 public:
  virtual ~GraphBackend() = default;

  /**
   * Check if this backend is available at runtime.
   */
  virtual bool isAvailable() const = 0;

  /**
   * Check if a contiguous range of slots can be fused/captured as a graph.
   */
  virtual bool canFuseSegment(NativeSlot* slots, int start, int end) = 0;

  /**
   * Compile a graph segment into backend-specific graph representation.
   * Returns true on success.
   *
   * @param seg The segment to compile
   * @param slots All slots in the plan
   * @param externalInputs External input arrays (constants, variables, placeholders)
   * @param numExternalInputs Count of external inputs
   * @param outputSlots The plan's output slot array (intermediate results)
   * @param totalOutputSlots Total number of output slots
   * @param shapeKey Shape signature for cache invalidation
   */
  virtual bool compileSegment(GraphSegment& seg, NativeSlot* slots,
                              NDArray** externalInputs, int numExternalInputs,
                              NDArray** outputSlots, int totalOutputSlots,
                              LongType shapeKey,
                              int totalSlots = 0,
                              int* requestedOutputSlotIndices = nullptr,
                              int numRequestedOutputs = 0) = 0;

  /**
   * Execute a previously compiled segment.
   *
   * @param seg The compiled segment
   * @param slots All slots in the plan
   * @param externalInputs External input arrays
   * @param numExternalInputs Count of external inputs
   * @param outputSlots The plan's output slot array (written by execution)
   * @param totalOutputSlots Total number of output slots
   * @param stream Backend-specific execution stream (nullptr for CPU)
   */
  virtual Status executeSegment(GraphSegment& seg, NativeSlot* slots,
                                NDArray** externalInputs, int numExternalInputs,
                                NDArray** outputSlots, int totalOutputSlots,
                                void* stream) = 0;

  /**
   * Invalidate all cached compiled graphs (e.g., on shape change).
   */
  virtual void invalidateCache() = 0;

  /**
   * Get the backend name for diagnostics.
   */
  virtual const char* name() const = 0;

  /**
   * Get the compilation audit for the most recent compileSegment() call.
   * Each entry shows whether a slot was compiled by the backend or skipped.
   * Skipped ops will produce stale outputs on graph replay.
   */
  virtual std::vector<CompilationAuditEntry> getLastCompilationAudit() const = 0;
};

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_GRAPH_BACKEND_H
