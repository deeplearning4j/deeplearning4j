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

    // One compiled partition per oneDNN partition (usually 1 for a fusible segment)
    struct PartitionEntry {
      dg::compiled_partition compiledPartition;
      std::vector<size_t> inputTensorIds;   // Logical tensor IDs for inputs
      std::vector<size_t> outputTensorIds;  // Logical tensor IDs for outputs
    };
    std::vector<PartitionEntry> partitions;

    // Tensor ID → slot mapping for wiring inputs/outputs at execution time
    // Maps tensor ID to: >=0 = outputSlot index, <0 = -(externalInputIndex+1)
    std::unordered_map<size_t, int> tensorIdToSlotMap;

    // Per-slot compilation audit: tracks which ops were compiled vs skipped
    std::vector<CompilationAuditEntry> compilationAudit;

    CompiledSegment() : shapeKey(0), valid(false) {}
  };

  // Per-segment cache (keyed by segment start/end + shape)
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

  std::unordered_map<SegmentCacheKey, CompiledSegment, SegmentCacheHash> cache_;
  std::mutex cacheMtx_;

  // Most recent compilation audit (updated by compileSegment)
  std::vector<CompilationAuditEntry> lastCompilationAudit_;

  // Build a dg::graph from a segment of slots
  CompiledSegment buildGraph(NativeSlot* slots, int startSlot, int endSlot,
                             NDArray** externalInputs, int numExternalInputs,
                             NDArray** outputSlots, int totalOutputSlots);
};

}  // namespace graph
}  // namespace sd

#endif  // HAVE_ONEDNN
#endif  // LIBND4J_ONEDNN_GRAPH_BACKEND_H
