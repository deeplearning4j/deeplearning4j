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

#ifndef LIBND4J_NVRTC_GRAPH_BACKEND_H
#define LIBND4J_NVRTC_GRAPH_BACKEND_H

#include <graph/GraphBackend.h>
#include <graph/NativeDynamicShapePlan.h>
#include <graph/gpu/JitGraphBackendCommon.h>

#ifdef SD_CUDA

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace sd {
namespace graph {

/**
 * NVRTC GPU compiler backend for the native plan executor.
 *
 * Generates CUDA C source code for fused element-wise chains at runtime,
 * compiles with NVRTC (nvrtcCompileProgram), and loads the resulting PTX
 * via the CUDA Driver API. No external dependencies beyond the CUDA toolkit.
 *
 * Dispatch priority: Triton -> NVRTC -> PTX -> CUDA Graphs -> slot-by-slot
 */
class NvrtcGraphBackend : public GraphBackend {
 public:
  NvrtcGraphBackend();
  ~NvrtcGraphBackend() override;

  const char* name() const override { return "NVRTC"; }
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

  static NvrtcGraphBackend& getInstance();

 private:
  std::unordered_map<JitSegmentCacheKey, JitCompiledKernel, JitSegmentCacheHash> cache_;
  std::mutex cacheMtx_;
  std::vector<CompilationAuditEntry> lastCompilationAudit_;

  // Generate CUDA C source for a fused element-wise chain
  std::string generateCudaSource(NativeSlot* slots, int startSlot, int endSlot,
                                  NDArray** externalInputs, int numExternalInputs,
                                  NDArray** outputSlots, int totalOutputSlots,
                                  JitCompiledKernel& result);

  // Get compute capability string for current device
  static std::string getComputeArch();
};

}  // namespace graph
}  // namespace sd

#endif  // SD_CUDA
#endif  // LIBND4J_NVRTC_GRAPH_BACKEND_H
