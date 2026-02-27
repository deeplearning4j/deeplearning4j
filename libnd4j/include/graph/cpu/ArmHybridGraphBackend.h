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
 * ARM hybrid graph backend: CPU MLIR JIT with optional Vulkan GPU offload.
 *
 * This backend targets ARM devices (Android NDK aarch64, Linux ARM64) and
 * implements a hybrid execution strategy:
 *
 * 1. CPU path (default): Uses MLIR with ARM-specific optimizations:
 *    - ARM NEON 128-bit SIMD vectorization
 *    - ARM SVE scalable vector support (where available)
 *    - ARM SME matrix extension (where available)
 *    - ARM-tuned tile sizes (16 for L1 cache, vs 32 on x86)
 *    - ARM dot product instructions (ARMv8.2+)
 *
 * 2. GPU path (optional): Uses MLIR → SPIR-V → Vulkan for offloading
 *    compute-heavy ops (matmul, conv2d, large reductions) to ARM Mali
 *    or Qualcomm Adreno GPUs.
 *
 * 3. AOT compilation: Can pre-compile kernels at build time for
 *    deployment on Android devices without shipping LLVM JIT.
 *
 * The backend selects CPU vs GPU path per-segment based on:
 *   - Op type (matmul/conv → GPU candidate, element-wise → CPU)
 *   - Tensor size (small tensors stay on CPU to avoid transfer overhead)
 *   - GPU availability (graceful fallback to CPU-only)
 *
 * Integration: Used by NativeDynamicShapePlan when running on ARM targets.
 * Priority in getCpuGraphBackend():
 *   ACL (ARM Compute Library) > ArmHybrid (MLIR) > generic MLIR CPU
 */
class ArmHybridGraphBackend : public GraphBackend {
 public:
  ArmHybridGraphBackend();
  ~ArmHybridGraphBackend() override;

  const char* name() const override { return "ARM Hybrid (MLIR CPU + Vulkan)"; }
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

  static ArmHybridGraphBackend& getInstance();

  /// Check if Vulkan compute is available on this device
  bool isVulkanAvailable() const { return vulkanAvailable_; }

  /// Set minimum tensor element count for GPU offload (default: 65536)
  void setGpuOffloadThreshold(int64_t threshold) { gpuOffloadThreshold_ = threshold; }

  /// Enable/disable Vulkan GPU offloading
  void setVulkanEnabled(bool enabled) { vulkanEnabled_ = enabled; }

 private:
  CpuIRBuilder irBuilder_;
  bool vulkanAvailable_ = false;
  bool vulkanEnabled_ = false;
  int64_t gpuOffloadThreshold_ = 65536;  // Min elements for GPU offload

  // Execution path for a compiled segment
  enum class ExecPath {
    ARM_CPU,      // ARM MLIR CPU with NEON/SVE
    VULKAN_GPU,   // SPIR-V via Vulkan compute
  };

  struct CompiledSegment {
    std::shared_ptr<sd::mlir_runtime::CompiledKernel> kernel;
    LongType shapeKey;
    bool valid;
    ExecPath execPath;

    struct ArgMapping {
      int sourceIndex;
      bool isOutput;
    };
    std::vector<ArgMapping> argMappings;
    std::vector<CompilationAuditEntry> compilationAudit;

    CompiledSegment() : shapeKey(0), valid(false), execPath(ExecPath::ARM_CPU) {}
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

  std::unordered_map<SegmentCacheKey, CompiledSegment, SegmentCacheHash> cache_;
  std::mutex cacheMtx_;
  std::vector<CompilationAuditEntry> lastCompilationAudit_;

  /// Determine whether a segment should use GPU or CPU path
  ExecPath selectExecPath(NativeSlot* slots, int start, int end,
                          NDArray** outputSlots, int totalOutputSlots);

  /// Check if Vulkan is available on the current device
  bool probeVulkanDevice();

  /// Get ARM-tuned MLIR compile options
  sd::mlir_runtime::MLIRCompileOptions getArmCompileOptions() const;
};

}  // namespace graph
}  // namespace sd

#endif  // HAVE_MLIR
#endif  // LIBND4J_ARM_HYBRID_GRAPH_BACKEND_H
