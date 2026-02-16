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
#include <vector>

namespace sd {
namespace graph {

/**
 * ARM Compute Library backend for the native plan executor.
 *
 * Uses ACL's NEFunctions for individual ops and composes them into
 * fused execution groups where possible. For supported patterns
 * (MatMul+Bias+Activation, Conv+BN+ReLU), uses ACL's built-in fusion.
 *
 * Follows the existing armcompute platform helper patterns:
 * - Tensor info from NDArray shape/dtype
 * - import_memory() for zero-copy when possible
 * - Cached configured functions keyed by shape
 */
class AclGraphBackend : public GraphBackend {
 public:
  AclGraphBackend();
  ~AclGraphBackend() override;

  const char* name() const override { return "ARM ACL"; }
  bool isAvailable() const override;
  bool canFuseSegment(NativeSlot* slots, int start, int end) override;

  bool compileSegment(GraphSegment& seg, NativeSlot* slots,
                      NDArray** externalInputs, int numExternalInputs,
                      NDArray** outputSlots, int totalOutputSlots,
                      LongType shapeKey) override;

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

  // A compiled ACL function group: a sequence of ACL NEFunctions + their tensors
  struct AclFunctionGroup {
    LongType shapeKey;
    bool valid;

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

    // Per-slot compilation audit: tracks which ops were compiled vs skipped
    std::vector<CompilationAuditEntry> compilationAudit;

    AclFunctionGroup() : shapeKey(0), valid(false) {}
  };

  // Segment cache
  struct SegmentCacheKey {
    int startSlot;
    int endSlot;
    LongType shapeKey;
    bool operator==(const SegmentCacheKey& o) const {
      return startSlot == o.startSlot && endSlot == o.endSlot && shapeKey == o.shapeKey;
    }
  };
  struct SegmentCacheHash {
    size_t operator()(const SegmentCacheKey& k) const {
      size_t h = std::hash<int>()(k.startSlot);
      h ^= std::hash<int>()(k.endSlot) << 1;
      h ^= std::hash<LongType>()(k.shapeKey) << 2;
      return h;
    }
  };

  std::unordered_map<SegmentCacheKey, AclFunctionGroup, SegmentCacheHash> cache_;
  std::mutex cacheMtx_;

  // Most recent compilation audit (updated by compileSegment)
  std::vector<CompilationAuditEntry> lastCompilationAudit_;

  // Build ACL functions for a segment
  AclFunctionGroup buildFunctions(NativeSlot* slots, int startSlot, int endSlot,
                                  NDArray** externalInputs, int numExternalInputs,
                                  NDArray** outputSlots, int totalOutputSlots);

  // Detect fusible patterns:
  // Returns true if slots[idx] and slots[idx+1] can be fused (e.g., add+relu)
  bool canFuseActivation(NativeSlot* slots, int idx, int endSlot);
};

}  // namespace graph
}  // namespace sd

#endif  // HAVE_ARMCOMPUTE
#endif  // LIBND4J_ACL_GRAPH_BACKEND_H
