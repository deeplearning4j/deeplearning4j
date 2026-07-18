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

#ifndef LIBND4J_ONEDNN_GRAPH_BACKEND_H
#define LIBND4J_ONEDNN_GRAPH_BACKEND_H

#include <graph/GraphBackend.h>
#include <graph/GraphBackendCommon.h>
#include <graph/NativeDynamicShapePlan.h>

#include <config.h>

#if HAVE_ONEDNN

#include <oneapi/dnnl/dnnl_graph.hpp>
#include <dnnl.hpp>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace sd {
namespace graph {

namespace dg = dnnl::graph;

/**
 * oneDNN Graph API backend for the native plan executor.
 *
 * Maps sequences of ops in a plan segment to dnnl::graph operations,
 * lets oneDNN automatically partition and fuse them, compiles the fused
 * partitions, caches by shape key, and executes with direct buffer pointers.
 *
 * Pattern follows the existing sdpa.cpp implementation.
 */
class OneDnnGraphBackend : public GraphBackend {
 public:
  OneDnnGraphBackend();
  ~OneDnnGraphBackend() override;

  const char* name() const override { return "OneDNN Graph"; }
  bool isAvailable() const override;
  bool canFuseSegment(NativeSlot* slots, int start, int end) override;

  // Type for a callback that executes a native slot range [startSlot, endSlot].
  // Used for mixed segments: unmappable ops are executed via the plan's slot-by-slot path.
  using NativeSlotExecutor = std::function<Status(int startSlot, int endSlot)>;

  // Set the executor callback for native (non-OneDNN) op ranges inside mixed segments.
  // Must be set before executeSegment() for any mixed-segment to succeed.
  void setNativeSlotExecutor(NativeSlotExecutor executor);
  void clearNativeSlotExecutor();

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

  static OneDnnGraphBackend& getInstance();

 private:
  dnnl::engine engine_;

  // Thread-local stream for reduced allocation overhead
  dnnl::stream& getThreadStream();

  // Map libnd4j op name to oneDNN graph op kind.
  // Returns dg::op::kind::LastSymbol if unmapped.
  static dg::op::kind mapOpKind(const std::string& opName);

  // Map NDArray DataType to oneDNN graph logical tensor data type.
  static dg::logical_tensor::data_type mapDataType(DataType dt);

  // Cached compiled segment: partitions + compiled partition objects
  struct CompiledSegment {
    LongType shapeKey;
    bool valid;
    bool isMixedSegment = false;  // true if segment has both OneDNN and native ops

    // One compiled partition per oneDNN partition (usually 1 for a fusible segment)
    struct PartitionEntry {
      dg::compiled_partition compiledPartition;
      std::vector<size_t> inputTensorIds;   // Logical tensor IDs for inputs
      std::vector<size_t> outputTensorIds;  // Logical tensor IDs for outputs

      // Cached tensor wrappers — avoids recreating logical_tensor + tensor every execution.
      // On stable shapes, only buffer pointers change; we rebuild tensors only if
      // buffer address differs from lastBufferPtrs[i].
      struct CachedTensor {
        dg::logical_tensor lt;
        void* lastBuffer = nullptr;
      };
      std::vector<CachedTensor> cachedInputTensors;
      std::vector<CachedTensor> cachedOutputTensors;

      // Slot range this partition covers (for mixed-segment interleaving)
      int startSlot = -1;
      int endSlot = -1;
    };
    std::vector<PartitionEntry> partitions;

    // A native slot range that must be executed via the plan's slot-by-slot path.
    // Used in mixed segments for unmappable ops between OneDNN partitions.
    struct NativeRange {
      int startSlot;
      int endSlot;
    };

    // For mixed segments: ordered execution schedule.
    // Each entry is either a partition index (>=0) or a native range index (<0, encoded as -(nativeIdx+1)).
    // Execution proceeds in order: OneDNN partitions and native ranges interleaved.
    // This preserves correct data flow when unmappable ops appear between fusible groups.
    struct ExecutionStep {
      bool isNative;       // true = native range, false = OneDNN partition
      int index;           // partition index or native range index
    };
    std::vector<ExecutionStep> executionSchedule;  // only used when isMixedSegment
    std::vector<NativeRange> nativeRanges;         // native slot ranges in schedule order

    // Tensor ID → slot mapping for wiring inputs/outputs at execution time
    // Maps tensor ID to: >=0 = outputSlot index, <0 = -(externalInputIndex+1)
    std::unordered_map<size_t, int> tensorIdToSlotMap;

    // Per-slot compilation audit: tracks which ops were compiled vs skipped
    std::vector<CompilationAuditEntry> compilationAudit;

    CompiledSegment() : shapeKey(0), valid(false), isMixedSegment(false) {}
  };

  // Per-segment cache (SegmentCacheKey/Hash from GraphBackendCommon.h)
  std::unordered_map<SegmentCacheKey, CompiledSegment, SegmentCacheHash> cache_;
  std::mutex cacheMtx_;

  // Negative cache: segment+shape combinations that failed compilation.
  // Avoids re-attempting known-bad compilations (mirrors Triton's failedCache_).
  std::unordered_set<SegmentCacheKey, SegmentCacheHash> failedCache_;

  // Most recent compilation audit (updated by compileSegment)
  std::vector<CompilationAuditEntry> lastCompilationAudit_;

  // Native slot executor: runs a range of slots via the plan's slot-by-slot path.
  // Thread-local so each calling thread gets its own executor (matches Triton model).
  static thread_local NativeSlotExecutor nativeExecutor_;

  // Minimum fraction of ops that must be OneDNN-mappable to accept a mixed segment.
  static constexpr float MIN_ONEDNN_COVERAGE = 0.3f;
  // Minimum absolute count of OneDNN-mappable ops required.
  static constexpr int MIN_MAPPABLE_OPS = 2;

  // Trait-based fallback: check OpDescriptor traits for mappability when mapOpKind returns LastSymbol.
  // Maps OpTraits flags to dg::op::kind. Returns LastSymbol if no mapping exists.
  static dg::op::kind mapOpKindFromTraits(const std::string& opName);

  // Build a dg::graph from a segment of slots (pure-OneDNN or mixed).
  // For mixed segments, unmappable ops are tracked as NativeRanges in the result.
  CompiledSegment buildGraph(NativeSlot* slots, int startSlot, int endSlot,
                             NDArray** externalInputs, int numExternalInputs,
                             NDArray** outputSlots, int totalOutputSlots);
};

}  // namespace graph
}  // namespace sd

#endif  // HAVE_ONEDNN
#endif  // LIBND4J_ONEDNN_GRAPH_BACKEND_H
