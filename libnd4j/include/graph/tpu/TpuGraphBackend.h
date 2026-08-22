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
#include <vector>

namespace sd {
namespace graph {

struct NativeSlot;
struct GraphSegment;

/** Strict StableHLO/PJRT compiler and executor for TPU graph segments. */
class SD_LIB_EXPORT TpuGraphBackend : public GraphBackend {
 public:
  static TpuGraphBackend& getInstance();

  bool isAvailable() const override;
  bool isResolvable(const GraphBackendRequest& request) const override;
  int resolutionPriority(const GraphBackendRequest& request) const override;
  GraphBackendPlanningPolicy planningPolicy(
      const GraphBackendRequest& request) const override;

  bool canResolveSlot(const GraphBackendRequest& request,
                      NativeSlot* slots, int slotIndex) override;
  bool canFuseSegment(NativeSlot* slots, int start, int end) override;

  bool compileSegment(GraphSegment& seg, NativeSlot* slots,
                      NDArray** externalInputs, int numExternalInputs,
                      NDArray** outputSlots, int totalOutputSlots,
                      LongType shapeKey, int totalSlots = 0,
                      int* requestedOutputSlotIndices = nullptr,
                      int numRequestedOutputs = 0) override;

  bool compileSegment(const GraphBackendRequest& request,
                      GraphSegment& seg, NativeSlot* slots,
                      NDArray** externalInputs, int numExternalInputs,
                      NDArray** outputSlots, int totalOutputSlots,
                      LongType shapeKey, int totalSlots,
                      int* requestedOutputSlotIndices,
                      int numRequestedOutputs) override;

  Status executeSegment(GraphSegment& seg, NativeSlot* slots,
                        NDArray** externalInputs, int numExternalInputs,
                        NDArray** outputSlots, int totalOutputSlots,
                        void* stream) override;

  void invalidateCache() override;
  const char* name() const override { return "TPU StableHLO"; }
  std::vector<CompilationAuditEntry> getLastCompilationAudit() const override;

 private:
  TpuGraphBackend();
  ~TpuGraphBackend() override = default;

  TpuGraphBackend(const TpuGraphBackend&) = delete;
  TpuGraphBackend& operator=(const TpuGraphBackend&) = delete;

  bool compileInternal(bool runtimeCompilationAllowed,
                       GraphSegment& seg, NativeSlot* slots,
                       NDArray** externalInputs, int numExternalInputs,
                       NDArray** outputSlots, int totalOutputSlots,
                       LongType shapeKey, int totalSlots,
                       int* requestedOutputSlotIndices,
                       int numRequestedOutputs);
  void auditRange(NativeSlot* slots, int start, int end, bool compiled,
                  const std::string& reason);

  std::vector<CompilationAuditEntry> lastAudit_;
  mutable std::mutex mutex_;
};

}  // namespace graph
}  // namespace sd

#endif  // SD_TPU
#endif  // LIBND4J_TPU_GRAPH_BACKEND_H
