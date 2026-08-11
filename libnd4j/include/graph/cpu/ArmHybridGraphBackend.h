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

#ifndef LIBND4J_ARM_HYBRID_GRAPH_BACKEND_H
#define LIBND4J_ARM_HYBRID_GRAPH_BACKEND_H

#include <graph/GraphBackend.h>
#include <graph/GraphBackendCommon.h>
#include <graph/NativeDynamicShapePlan.h>

#if HAVE_MLIR

#include <graph/cpu/CpuIRBuilder.h>
#include <mlir/runtime/MLIREngine.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace sd {
namespace graph {

/**
 * ARM-tuned MLIR CPU graph backend.
 *
 * This backend targets Android NDK aarch64 and Linux ARM64 with host-CPU
 * compilation options for NEON, SVE/SME where available, ARM-tuned tile sizes,
 * dot-product instructions, and optional AOT generation. The historical class
 * name is retained for source compatibility; device execution belongs to each
 * dedicated device backend and is never selected or invoked here.
 *
 * Integration: Registered in NativeDynamicShapePlan's shared backend catalog
 * on ARM targets. Resolution priority is backend-owned:
 *   ACL (ARM Compute Library) > ARM MLIR > generic MLIR CPU
 */
class ArmHybridGraphBackend : public GraphBackend {
 public:
  ArmHybridGraphBackend();
  ~ArmHybridGraphBackend() override;

  const char* name() const override { return "ARM MLIR CPU"; }
  bool isAvailable() const override;
  bool isResolvable(const GraphBackendRequest& request) const override;
  int resolutionPriority(const GraphBackendRequest& request) const override;
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

  static ArmHybridGraphBackend& getInstance();

 private:
  CpuIRBuilder irBuilder_;

  struct CompiledSegment {
    std::shared_ptr<sd::mlir_runtime::CompiledKernel> kernel;
    LongType shapeKey;
    bool valid;

    // ArgMapping from GraphBackendCommon.h
    std::vector<ArgMapping> argMappings;
    std::vector<CompilationAuditEntry> compilationAudit;

    CompiledSegment() : shapeKey(0), valid(false) {}
  };

  // Segment cache (SegmentCacheKey/Hash from GraphBackendCommon.h)
  std::unordered_map<SegmentCacheKey, CompiledSegment, SegmentCacheHash> cache_;
  std::mutex cacheMtx_;
  std::vector<CompilationAuditEntry> lastCompilationAudit_;

  /// Get ARM-tuned MLIR compile options
  sd::mlir_runtime::MLIRCompileOptions getArmCompileOptions() const;
};

}  // namespace graph
}  // namespace sd

#endif  // HAVE_MLIR
#endif  // LIBND4J_ARM_HYBRID_GRAPH_BACKEND_H
