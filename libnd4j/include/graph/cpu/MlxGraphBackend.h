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

#ifndef LIBND4J_MLX_GRAPH_BACKEND_H
#define LIBND4J_MLX_GRAPH_BACKEND_H

#include <graph/GraphBackend.h>
#include <graph/GraphBackendCommon.h>
#include <graph/NativeDynamicShapePlan.h>

#include <config.h>

#if HAVE_MLX

#include <graph/cpu/MlxIRBuilder.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace sd {
namespace graph {

/**
 * MLX Apple Silicon graph backend for the native plan executor.
 *
 * Compiles fused Metal GPU kernels from NativeSlot segments using
 * MlxIRBuilder (mlx::core lazy computation graphs) evaluated via Metal.
 *
 * All phases implemented:
 *   Phase 1: Element-wise (binary, unary, comparison, logical, ternary, identity, cast)
 *   Phase 2: Reductions, matmul, normalization, shape manipulation, data movement, constant gen
 *   Phase 3: Convolution (conv2d, depthwise_conv2d), fused attention (SDPA)
 *   LLM:    RoPE (mx::fast::rope), GQA, flash attention, sliding window attention,
 *           fused_rms_norm_swiglu, KV cache update, embedding lookup, top-k
 *
 * Priority in getCpuGraphBackend():
 *   MLX (Apple Silicon) > oneDNN (Intel) > ACL (ARM) > MLIR JIT (universal)
 *
 * This is the highest-priority CPU backend on macOS/Apple Silicon.
 */
class MlxGraphBackend : public GraphBackend {
 public:
  MlxGraphBackend();
  ~MlxGraphBackend() override;

  const char* name() const override { return "MLX Apple Silicon"; }
  bool isAvailable() const override;
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

  static MlxGraphBackend& getInstance();

 private:
  MlxIRBuilder irBuilder_;

  // Compiled segment: holds the MLX graph and argument mapping
  struct CompiledSegment {
    MlxIRBuilder::MlxGraph mlxGraph;
    LongType shapeKey;
    bool valid;

    // ArgMapping from GraphBackendCommon.h
    std::vector<ArgMapping> argMappings;

    // Per-slot compilation audit
    std::vector<CompilationAuditEntry> compilationAudit;

    CompiledSegment() : shapeKey(0), valid(false) {}
  };

  // Segment cache (SegmentCacheKey/Hash from GraphBackendCommon.h)
  std::unordered_map<SegmentCacheKey, CompiledSegment, SegmentCacheHash> cache_;
  std::mutex cacheMtx_;

  std::vector<CompilationAuditEntry> lastCompilationAudit_;

  // Cached availability check result
  mutable bool availabilityChecked_ = false;
  mutable bool available_ = false;
};

}  // namespace graph
}  // namespace sd

#endif  // HAVE_MLX
#endif  // LIBND4J_MLX_GRAPH_BACKEND_H
