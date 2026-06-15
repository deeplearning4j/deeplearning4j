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

#ifndef LIBND4J_TPU_GRAPH_BACKEND_H
#define LIBND4J_TPU_GRAPH_BACKEND_H

#include <system/common.h>

#ifdef SD_TPU

#include <graph/GraphBackend.h>

#include <mutex>
#include <string>
#include <vector>

namespace sd {
namespace graph {

// Forward declarations — avoid circular include with NativeDynamicShapePlan.h
struct NativeSlot;
struct GraphSegment;

/**
 * TPU HLO graph backend for NativeDynamicShapePlan.
 *
 * Implements the GraphBackend interface for TPU hardware via PJRT:
 *   1. canFuseSegment(): checks all ops in range are HLO-representable
 *   2. compileSegment(): builds HLO via HloIRBuilder -> compiles via PjrtClientManager
 *   3. executeSegment(): binds input PJRT buffers -> executes -> copies outputs
 *
 * Singleton pattern, consistent with other graph backends.
 */
class SD_LIB_EXPORT TpuGraphBackend : public GraphBackend {
 public:
  /**
   * Get the singleton instance.
   */
  static TpuGraphBackend* getInstance();

  // ── GraphBackend interface ────────────────────────────────────────────

  /**
   * Check if TPU backend is available at runtime.
   * Returns true if PjrtClientManager can create a client and devices exist.
   */
  bool isAvailable() const override;

  /**
   * Check if all ops in the slot range [start, end) can be fused into HLO.
   * Uses HloIRBuilder's whitelist to verify each op is mappable.
   */
  bool canFuseSegment(NativeSlot* slots, int start, int end) override;

  /**
   * Compile a segment into an HLO module and XLA executable.
   * 1. Build HLO text via HloIRBuilder
   * 2. Compile via PjrtClientManager
   * 3. Store compiled handle in segment's replayHandle (TpuReplayHandle)
   */
  bool compileSegment(GraphSegment& seg, NativeSlot* slots,
                      NDArray** externalInputs, int numExternalInputs,
                      NDArray** outputSlots, int totalOutputSlots,
                      LongType shapeKey,
                      int totalSlots = 0,
                      int* requestedOutputSlotIndices = nullptr,
                      int numRequestedOutputs = 0) override;

  /**
   * Execute a previously compiled segment.
   * Binds input PJRT buffers from externalInputs and outputSlots,
   * executes the compiled module, and copies output buffers back.
   */
  Status executeSegment(GraphSegment& seg, NativeSlot* slots,
                        NDArray** externalInputs, int numExternalInputs,
                        NDArray** outputSlots, int totalOutputSlots,
                        void* stream) override;

  /**
   * Invalidate all cached compiled graphs.
   */
  void invalidateCache() override;

  /**
   * Get the backend name for diagnostics.
   */
  const char* name() const override { return "TPU HLO"; }

  /**
   * Get the compilation audit for the most recent compileSegment() call.
   */
  std::vector<CompilationAuditEntry> getLastCompilationAudit() const override;

 private:
  TpuGraphBackend();
  ~TpuGraphBackend() override = default;

  // Non-copyable
  TpuGraphBackend(const TpuGraphBackend&) = delete;
  TpuGraphBackend& operator=(const TpuGraphBackend&) = delete;

  // Compilation audit trail for the last compileSegment() call
  std::vector<CompilationAuditEntry> lastAudit_;

  // Thread safety for singleton access and audit
  mutable std::mutex mutex_;
};

}  // namespace graph
}  // namespace sd

#endif  // SD_TPU
#endif  // LIBND4J_TPU_GRAPH_BACKEND_H
