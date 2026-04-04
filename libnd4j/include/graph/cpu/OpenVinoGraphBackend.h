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
 * OpenVINO provides broader op coverage than oneDNN Graph (~200 vs ~80 ops),
 * including Gather, GatherND, ScatterNDUpdate, Where/Select, Split, Slice,
 * and full comparison/logical op families. It also offers Snippets JIT for
 * element-wise fusion and oneDNN BRGEMM for matmul/conv.
 *
 * Priority: after oneDNN Graph (which has built-in MHA/MLP fusion patterns
 * that are faster for those specific patterns), before ARM ACL.
 */
class OpenVinoGraphBackend : public GraphBackend {
 public:
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

  static OpenVinoGraphBackend& getInstance();

 private:
  ov::Core core_;

  // Cached compiled segment
  struct CompiledSegment {
    LongType shapeKey;
    bool valid;

    std::shared_ptr<ov::CompiledModel> compiled;
    std::shared_ptr<ov::InferRequest> request;

    // Maps OV model input index -> source: >=0 = outputSlot index, <0 = -(externalInputIndex+1)
    std::vector<int> inputSlotMap;
    // Maps OV model output index -> outputSlot index
    std::vector<int> outputSlotMap;

    // Per-slot compilation audit
    std::vector<CompilationAuditEntry> compilationAudit;

    CompiledSegment() : shapeKey(0), valid(false) {}
  };

  // Per-segment cache (SegmentCacheKey/Hash from GraphBackendCommon.h)
  std::unordered_map<SegmentCacheKey, CompiledSegment, SegmentCacheHash> cache_;
  std::mutex cacheMtx_;

  // Most recent compilation audit
  std::vector<CompilationAuditEntry> lastCompilationAudit_;

  // Check if an op name is mappable to an OpenVINO opset13 operation
  static bool isOpenVinoMappable(const std::string& opName);

  // Map NDArray DataType to OpenVINO element type
  static ov::element::Type mapDataType(DataType dt);

  // Build an ov::Model from a segment of slots
  CompiledSegment buildModel(NativeSlot* slots, int startSlot, int endSlot,
                             NDArray** externalInputs, int numExternalInputs,
                             NDArray** outputSlots, int totalOutputSlots);
};

}  // namespace graph
}  // namespace sd

#endif  // HAVE_OPENVINO
#endif  // LIBND4J_OPENVINO_GRAPH_BACKEND_H
