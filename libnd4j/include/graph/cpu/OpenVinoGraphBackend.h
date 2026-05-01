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

#ifndef LIBND4J_OPENVINO_GRAPH_BACKEND_H
#define LIBND4J_OPENVINO_GRAPH_BACKEND_H

#include <graph/GraphBackend.h>
#include <graph/GraphBackendCommon.h>
#include <graph/NativeDynamicShapePlan.h>

#include <config.h>

#if HAVE_OPENVINO

#include <openvino/openvino.hpp>
#include <openvino/op/ops.hpp>

#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace sd {
namespace graph {

/**
 * OpenVINO CPU graph backend for the native plan executor.
 *
 * Maps sequences of ops in a plan segment to OpenVINO opset13 operations,
 * compiles via ov::Core::compile_model("CPU"), caches by shape key,
 * and executes with zero-copy ov::Tensor wrappers around NDArray host buffers.
 *
 * Supports mixed segments: ops that can't be mapped to OV (SSM recurrence)
 * are executed natively via the NativeSlotExecutor callback. The segment is
 * split into OV "islands" around native ops, and execution interleaves
 * compiled OV inference with native slot-by-slot execution.
 */
class OpenVinoGraphBackend : public GraphBackend {
 public:
  // Callback type for executing native (non-OV) slot ranges
  using NativeSlotExecutor = std::function<Status(int startSlot, int endSlot)>;

  OpenVinoGraphBackend();
  ~OpenVinoGraphBackend() override;

  const char* name() const override { return "OpenVINO"; }
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

  // Native slot executor management (thread-local, set by the plan executor)
  void setNativeSlotExecutor(NativeSlotExecutor executor);
  void clearNativeSlotExecutor();

  static OpenVinoGraphBackend& getInstance();

 private:
  ov::Core core_;

  // A contiguous range of slots executed natively (not through OV)
  struct NativeRange {
    int startSlot;
    int endSlot;
  };

  // A compiled OV island covering a contiguous range of OV-compilable slots
  struct OvIsland {
    int startSlot;
    int endSlot;
    std::shared_ptr<ov::CompiledModel> compiled;
    std::shared_ptr<ov::InferRequest> request;
    std::vector<int> inputSlotMap;   // OV input index -> source slot/ext index
    std::vector<int> outputSlotMap;  // OV output index -> outputSlot index
  };

  // Execution schedule step: either an OV island or a native range
  struct ExecutionStep {
    bool isNative;
    int index;  // index into ovIslands or nativeRanges
  };

  // Cached compiled segment with mixed execution support
  struct CompiledSegment {
    LongType shapeKey;
    bool valid;

    // OV islands (compiled subgraphs between native ops)
    std::vector<OvIsland> ovIslands;
    // Native ranges (slots executed via NativeSlotExecutor)
    std::vector<NativeRange> nativeRanges;
    // Interleaved execution order
    std::vector<ExecutionStep> executionSchedule;

    // Per-slot compilation audit
    std::vector<CompilationAuditEntry> compilationAudit;

    CompiledSegment() : shapeKey(0), valid(false) {}
  };

  // Per-segment cache with optional LRU eviction.
  // 0 = unlimited (no eviction). Memory is bounded by the topology model cache
  // (modelCache_) which shares CompiledModels across segments with identical op
  // structure (e.g., transformer layers). Each segment entry only holds a
  // lightweight InferRequest (~KB). Previously 32, which caused KERNEL_FAILURE
  // when frozen plans executed evicted segments (e.g. 772 evictions on
  // 1913-slot Qwen model).
  static constexpr int kMaxCacheEntries = 0;

  // LRU list: front = MRU, back = LRU
  using CacheEntry = std::pair<SegmentCacheKey, CompiledSegment>;
  using CacheLruList = std::list<CacheEntry>;
  CacheLruList cacheLru_;
  std::unordered_map<SegmentCacheKey, CacheLruList::iterator, SegmentCacheHash> cache_;
  std::mutex cacheMtx_;

  // Evict oldest cache entries until under the cap. Must hold cacheMtx_.
  void evictCacheIfOverLimitLocked();

  // Topology-based compiled model cache.
  // Segments from different transformer layers with the same op sequence and shapes
  // share one CompiledModel.  Each segment gets its own InferRequest (cheap, ~KB).
  // Key: topology hash (op names + input shapes + output shapes), Value: compiled model.
  std::unordered_map<uint64_t, std::shared_ptr<ov::CompiledModel>> modelCache_;
  std::mutex modelCacheMtx_;

  // Compute topology hash for an island (op names + parameter shapes, slot-index-agnostic)
  uint64_t computeIslandTopologyHash(NativeSlot* slots, int startSlot, int endSlot,
                                     NDArray** externalInputs, int numExternalInputs,
                                     NDArray** outputSlots, int totalOutputSlots);

  // Most recent compilation audit
  std::vector<CompilationAuditEntry> lastCompilationAudit_;

  // Thread-local native executor (set by plan before executeSegment)
  static thread_local NativeSlotExecutor nativeExecutor_;

  // Check if an op name is mappable to an OpenVINO opset13 operation
  static bool isOpenVinoMappable(const std::string& opName);

  // Check if an op must run natively (too complex/stateful for OV decomposition)
  static bool isNativeDeferredOp(const std::string& opName);

  // Map NDArray DataType to OpenVINO element type
  static ov::element::Type mapDataType(DataType dt);

  // Build an OV model from a contiguous range of OV-compilable slots
  OvIsland buildIsland(NativeSlot* slots, int startSlot, int endSlot,
                       NDArray** externalInputs, int numExternalInputs,
                       NDArray** outputSlots, int totalOutputSlots,
                       int segStartSlot, int segEndSlot);
};

}  // namespace graph
}  // namespace sd

#endif  // HAVE_OPENVINO
#endif  // LIBND4J_OPENVINO_GRAPH_BACKEND_H
