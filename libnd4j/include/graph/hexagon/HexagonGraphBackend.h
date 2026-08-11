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

#ifndef LIBND4J_HEXAGON_GRAPH_BACKEND_H
#define LIBND4J_HEXAGON_GRAPH_BACKEND_H

#include <system/common.h>

#ifdef HAVE_HEXAGON_MLIR

#include <graph/GraphBackend.h>

#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sd {
namespace graph {

// Forward declarations — defined in NativeDynamicShapePlan.h
struct NativeSlot;
struct GraphSegment;

/**
 * Hexagon MLIR graph backend for the native plan executor.
 *
 * Compiles sequences of ops into fused HVX kernels using the hexagon-mlir
 * compiler. The Hexagon DSP's HVX (Hexagon Vector eXtension) unit provides
 * 1024-bit or 2048-bit SIMD operations, and TCM (Tightly Coupled Memory)
 * provides single-cycle on-chip SRAM for staging data.
 *
 * Execution flow:
 *   1. DMA input tensors from DDR to TCM
 *   2. Execute HVX kernel on TCM-resident data
 *   3. DMA output tensors from TCM back to DDR
 *
 * Segment fusion is constrained by TCM capacity (typically 256KB-1MB).
 * Segments whose working set exceeds TCM_CAPACITY are rejected.
 *
 * Singleton pattern — one instance per process.
 */
class SD_LIB_EXPORT HexagonGraphBackend : public GraphBackend {
 public:
  ~HexagonGraphBackend() override;

  // Non-copyable
  HexagonGraphBackend(const HexagonGraphBackend&) = delete;
  HexagonGraphBackend& operator=(const HexagonGraphBackend&) = delete;

  /**
   * Get the singleton instance (reference, matching TpuGraphBackend pattern).
   */
  static HexagonGraphBackend& getInstance();

  // ── GraphBackend interface ─────────────────────────────────────────────

  const char* name() const override { return "Hexagon MLIR"; }

  /**
   * Check if the Hexagon backend is available at runtime.
   * Returns true if HexagonRuntimeManager can load the runtime and
   * at least one NPU device is present.
   */
  bool isAvailable() const override;
  bool isResolvable(const GraphBackendRequest& request) const override;
  int resolutionPriority(const GraphBackendRequest& request) const override;

  /**
   * Check if a contiguous range of slots can be fused into an HVX kernel.
   * Verifies:
   *   1. All ops in the range are HVX-representable (via HexagonIRBuilder whitelist)
   *   2. The estimated working set fits within TCM_CAPACITY
   */
  bool canFuseSegment(NativeSlot* slots, int start, int end) override;
  bool canResolveSegment(const GraphBackendRequest& request,
                         NativeSlot* slots, int start, int end) override;

  /**
   * Compile a graph segment into an HVX kernel.
   *   1. Build MLIR via HexagonIRBuilder
   *   2. Compile MLIR to NPU kernel via hexagon-mlir runtime
   *   3. Store compiled kernel handle in the segment cache
   */
  bool compileSegment(GraphSegment& seg, NativeSlot* slots,
                      NDArray** externalInputs, int numExternalInputs,
                      NDArray** outputSlots, int totalOutputSlots,
                      LongType shapeKey,
                      int totalSlots = 0,
                      int* requestedOutputSlotIndices = nullptr,
                      int numRequestedOutputs = 0) override;
  bool compileSegment(const GraphBackendRequest& request,
                      GraphSegment& seg, NativeSlot* slots,
                      NDArray** externalInputs, int numExternalInputs,
                      NDArray** outputSlots, int totalOutputSlots,
                      LongType shapeKey, int totalSlots,
                      int* requestedOutputSlotIndices,
                      int numRequestedOutputs) override;

  /**
   * Execute a previously compiled segment on the Hexagon NPU.
   *   1. DMA input tensors from host DDR to NPU TCM
   *   2. Dispatch the compiled HVX kernel
   *   3. DMA output tensors from NPU TCM back to host DDR
   */
  Status executeSegment(GraphSegment& seg, NativeSlot* slots,
                        NDArray** externalInputs, int numExternalInputs,
                        NDArray** outputSlots, int totalOutputSlots,
                        void* stream) override;
  Status executeSegment(const GraphBackendRequest& request,
                        GraphSegment& seg, NativeSlot* slots,
                        NDArray** externalInputs, int numExternalInputs,
                        NDArray** outputSlots, int totalOutputSlots,
                        void* stream) override;

  /**
   * Invalidate all cached compiled kernels (e.g., on shape change).
   */
  void invalidateCache() override;

  /**
   * Get the compilation audit for the most recent compileSegment() call.
   */
  std::vector<CompilationAuditEntry> getLastCompilationAudit() const override;

  // ── Configuration ─────────────────────────────────────────────────────

  /**
   * Default TCM capacity in bytes (256KB).
   * Can be overridden at runtime via HexagonRuntimeManager::getTcmCapacity().
   */
  static constexpr size_t TCM_CAPACITY = 256 * 1024;

  /**
   * Minimum fraction of HVX-mappable ops required to attempt compilation.
   */
  static constexpr float MIN_MAPPABLE_FRACTION = 0.5f;

 private:
  HexagonGraphBackend();

  // ── Compiled kernel cache ─────────────────────────────────────────────

  struct CompiledKernel {
    void* kernelHandle = nullptr;   // Opaque kernel from HexagonRuntimeManager
    void* npuContext = nullptr;     // NPU context used for compilation
    int startSlot = -1;
    int endSlot = -1;
    std::vector<CompilationAuditEntry> audit;
  };

  struct SegmentCacheKey {
    int startSlot;
    int endSlot;
    LongType shapeKey;
    bool allowRuntimeCompilation;
    std::string artifactDirectory;
    bool operator==(const SegmentCacheKey& o) const {
      return startSlot == o.startSlot &&
             endSlot == o.endSlot &&
             shapeKey == o.shapeKey &&
             allowRuntimeCompilation == o.allowRuntimeCompilation &&
             artifactDirectory == o.artifactDirectory;
    }
  };

  struct SegmentCacheHash {
    size_t operator()(const SegmentCacheKey& k) const {
      size_t h = std::hash<int>()(k.startSlot);
      h ^= std::hash<int>()(k.endSlot) << 1;
      h ^= std::hash<LongType>()(k.shapeKey) << 2;
      h ^= std::hash<bool>()(k.allowRuntimeCompilation) << 3;
      h ^= std::hash<std::string>()(k.artifactDirectory) << 4;
      return h;
    }
  };

  std::unordered_map<SegmentCacheKey, CompiledKernel, SegmentCacheHash> cache_;
  std::unordered_set<SegmentCacheKey, SegmentCacheHash> failedCache_;
  mutable std::mutex cacheMtx_;

  // Most recent compilation audit
  std::vector<CompilationAuditEntry> lastCompilationAudit_;

  // NPU context for the backend (lazily initialized)
  void* npuContext_ = nullptr;

  /**
   * Ensure the NPU context is initialized.
   * @return true if npuContext_ is valid
   */
  bool ensureNpuContext();

  /**
   * DMA input arrays to TCM, returning TCM buffer pointers.
   * @param slot       The slot whose inputs to stage
   * @param slots      Full slot array (for cross-slot references)
   * @param externalInputs External inputs
   * @param numExternalInputs Count of external inputs
   * @param outputSlots Output slot array
   * @param totalOutputSlots Total output slots
   * @param tcmPtrs    Output: TCM pointers for each input
   * @return true if all DMA transfers succeeded
   */
  bool stageInputsToTcm(const NativeSlot& slot, NativeSlot* slots,
                         NDArray** externalInputs, int numExternalInputs,
                         NDArray** outputSlots, int totalOutputSlots,
                         std::vector<void*>& tcmPtrs);

  /**
   * DMA output from TCM back to host DDR.
   * @param outputArray Target NDArray in host DDR
   * @param tcmPtr      TCM source pointer
   * @param bytes       Number of bytes to transfer
   * @return true if DMA transfer succeeded
   */
  bool stageOutputFromTcm(NDArray* outputArray, void* tcmPtr, size_t bytes);
};

}  // namespace graph
}  // namespace sd

#endif  // HAVE_HEXAGON_MLIR
#endif  // LIBND4J_HEXAGON_GRAPH_BACKEND_H
