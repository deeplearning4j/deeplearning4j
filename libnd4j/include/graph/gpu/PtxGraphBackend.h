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

#ifndef LIBND4J_PTX_GRAPH_BACKEND_H
#define LIBND4J_PTX_GRAPH_BACKEND_H

#include <graph/GraphBackend.h>
#include <graph/NativeDynamicShapePlan.h>

#ifdef SD_CUDA

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace sd {
namespace graph {

/**
 * PTX template GPU backend for the native plan executor.
 *
 * Generates PTX assembly text directly via string templates for fused
 * element-wise chains. No compilation step -- the PTX text is loaded
 * directly via cuModuleLoadDataEx, which JIT-compiles PTX to SASS.
 *
 * This is the fastest "compilation" path (just string concatenation)
 * but produces less optimized code than NVRTC or Triton.
 *
 * Dispatch priority: Triton -> NVRTC -> PTX -> CUDA Graphs -> slot-by-slot
 */
class PtxGraphBackend : public GraphBackend {
 public:
  PtxGraphBackend();
  ~PtxGraphBackend() override;

  const char* name() const override { return "PTX Template"; }
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

  static PtxGraphBackend& getInstance();

 private:
  struct CompiledKernel {
    void* gpuModule;
    void* kernelFunction;

    struct ArgMapping {
      int slotIndex;
      bool isOutput;
    };
    std::vector<ArgMapping> argMap;
    std::vector<CompilationAuditEntry> audit;

    CompiledKernel() : gpuModule(nullptr), kernelFunction(nullptr) {}
  };

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

  std::unordered_map<SegmentCacheKey, CompiledKernel, SegmentCacheHash> cache_;
  std::mutex cacheMtx_;
  std::vector<CompilationAuditEntry> lastCompilationAudit_;

  static constexpr int MIN_FUSIBLE_OPS = 2;

  // Generate PTX text for a fused chain
  std::string generatePtx(NativeSlot* slots, int startSlot, int endSlot,
                           NDArray** externalInputs, int numExternalInputs,
                           NDArray** outputSlots, int totalOutputSlots,
                           CompiledKernel& result);

  // Get SM version for PTX target directive
  static int getSmVersion();
};

}  // namespace graph
}  // namespace sd

#endif  // SD_CUDA
#endif  // LIBND4J_PTX_GRAPH_BACKEND_H
