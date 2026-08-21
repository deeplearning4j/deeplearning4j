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

#ifndef LIBND4J_ACL_GRAPH_BACKEND_H
#define LIBND4J_ACL_GRAPH_BACKEND_H

#include <graph/GraphBackend.h>
#include <graph/GraphBackendCommon.h>
#include <graph/NativeDynamicShapePlan.h>

#include <config.h>

#if HAVE_ARMCOMPUTE

#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/NEON/NEFunctions.h>
#include <arm_compute/runtime/Tensor.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sd {
namespace graph {

/**
 * ARM Compute Library backend for the native plan executor.
 *
 * Uses ACL's NEFunctions for individual ops and composes completely covered
 * operation ranges into one segment-owned execution group.
 *
 * Follows the existing armcompute platform helper patterns:
 * - Tensor info from NDArray shape/dtype
 * - import_memory() for zero-copy when possible
 * - Segment-owned configured functions keyed by shape
 */
class AclGraphBackend : public GraphBackend {
 public:
  AclGraphBackend();
  ~AclGraphBackend() override;

  const char* name() const override { return "ARM ACL"; }
  bool isAvailable() const override;
  bool isResolvable(const GraphBackendRequest& request) const override;
  int resolutionPriority(const GraphBackendRequest& request) const override;
  bool canResolveSlot(const GraphBackendRequest& request,
                      NativeSlot* slots, int slotIndex) override;
  bool canResolveSegment(const GraphBackendRequest& request,
                         NativeSlot* slots, int start, int end) override;
  bool canFuseSegment(NativeSlot* slots, int start, int end) override;

  bool compileSegment(GraphSegment& seg, NativeSlot* slots,
                      NDArray** externalInputs, int numExternalInputs,
                      NDArray** outputSlots, int totalOutputSlots,
                      LongType shapeKey,
                      int totalSlots = 0,
                      int* requestedOutputSlotIndices = nullptr,
                      int numRequestedOutputs = 0) override;

  Status executeSegment(GraphSegment& seg, NativeSlot* slots,
                        NDArray** externalInputs, int numExternalInputs,
                        NDArray** outputSlots, int totalOutputSlots,
                        void* stream) override;

  void invalidateCache() override;

  std::vector<CompilationAuditEntry> getLastCompilationAudit() const override;

  static AclGraphBackend& getInstance();

 private:
  // Helper to map NDArray DataType to ACL DataType
  static arm_compute::DataType mapDataType(DataType dt);

  // Helper to create TensorInfo from NDArray
  static arm_compute::TensorInfo getTensorInfo(NDArray* arr);

  // Helper to detect activation type from op name
  static arm_compute::ActivationLayerInfo::ActivationFunction mapActivation(const std::string& opName);

  // Metadata-only admission for operations implemented by buildFunctions().
  // Runtime tensor descriptors are validated again before ACL configuration.
  static bool isSupportedSlotContract(const NativeSlot& slot);

  // A compiled ACL function group: a sequence of ACL NEFunctions + their tensors
  struct AclFunctionGroup {
    LongType shapeKey = 0;
    int startSlot = -1;
    int endSlot = -1;
    bool valid = false;
    std::mutex executionMtx;

    struct FunctionEntry {
      std::unique_ptr<arm_compute::IFunction> function;
      // Input/output tensor wrappers (ACL Tensor objects)
      std::vector<std::shared_ptr<arm_compute::Tensor>> tensors;
    };
    std::vector<FunctionEntry> functions;

    // Mapping from slot index to ACL tensor for buffer import at execution time
    // slotIdx -> index into the tensors vector
    std::unordered_map<int, std::shared_ptr<arm_compute::Tensor>> slotToTensor;
    std::unordered_map<int, std::shared_ptr<arm_compute::Tensor>> extToTensor;
    std::unordered_set<int> producedSlots;

    // Per-slot compilation audit: tracks which ops were compiled vs skipped
    std::vector<CompilationAuditEntry> compilationAudit;

    void invalidate() {
      std::lock_guard<std::mutex> lock(executionMtx);
      valid = false;
      functions.clear();
      slotToTensor.clear();
      extToTensor.clear();
      producedSlots.clear();
    }
  };

  // Plans own compiled groups. The singleton retains weak references only for
  // explicit global invalidation, avoiding mutable cross-plan tensor sharing.
  std::vector<std::weak_ptr<AclFunctionGroup>> compiledArtifacts_;
  mutable std::mutex cacheMtx_;

  // Most recent compilation audit (updated by compileSegment)
  std::vector<CompilationAuditEntry> lastCompilationAudit_;

  // Build ACL functions for a segment
  std::shared_ptr<AclFunctionGroup> buildFunctions(
      NativeSlot* slots, int startSlot, int endSlot,
      NDArray** externalInputs, int numExternalInputs,
      NDArray** outputSlots, int totalOutputSlots);
};

}  // namespace graph
}  // namespace sd

#endif  // HAVE_ARMCOMPUTE
#endif  // LIBND4J_ACL_GRAPH_BACKEND_H
