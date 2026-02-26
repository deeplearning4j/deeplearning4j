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

#ifndef LIBND4J_MLIR_CPU_GRAPH_BACKEND_H
#define LIBND4J_MLIR_CPU_GRAPH_BACKEND_H

#include <graph/GraphBackend.h>
#include <graph/NativeDynamicShapePlan.h>

#ifdef HAVE_MLIR

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
 * MLIR JIT CPU graph backend for the native plan executor.
 *
 * Compiles fused CPU kernels from NativeSlot segments using
 * CpuIRBuilder (MLIR memref/scf/arith/math) and MLIREngine's CPU
 * lowering pipeline (→ LLVM JIT).
 *
 * Supports ALL op categories: element-wise (binary, unary, comparison,
 * logical, ternary, identity, cast), matmul, reduction, normalization,
 * data movement (gather, concat, tile, scatter_nd, split), shape
 * manipulation (permute, reshape), constant generation, and convolution.
 *
 * Priority in getCpuGraphBackend():
 *   oneDNN (Intel-optimized) > ACL (ARM-optimized) > MLIR JIT (universal)
 */
class MlirCpuGraphBackend : public GraphBackend {
 public:
  MlirCpuGraphBackend();
  ~MlirCpuGraphBackend() override;

  const char* name() const override { return "MLIR CPU JIT"; }
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

  static MlirCpuGraphBackend& getInstance();

 private:
  CpuIRBuilder irBuilder_;

  // Compiled segment: holds the JIT kernel and argument mapping
  struct CompiledSegment {
    std::shared_ptr<sd::mlir_runtime::CompiledKernel> kernel;
    LongType shapeKey;
    bool valid;

    // Argument mapping: ordered list of source indices for the kernel's buffer args.
    // <0: external input (-(idx+1))
    // >=0: outputSlot index
    struct ArgMapping {
      int sourceIndex;
      bool isOutput;
    };
    std::vector<ArgMapping> argMappings;

    // Per-slot compilation audit
    std::vector<CompilationAuditEntry> compilationAudit;

    CompiledSegment() : shapeKey(0), valid(false) {}
  };

  // Cache key
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

  std::vector<CompilationAuditEntry> lastCompilationAudit_;
};

}  // namespace graph
}  // namespace sd

#endif  // HAVE_MLIR
#endif  // LIBND4J_MLIR_CPU_GRAPH_BACKEND_H
