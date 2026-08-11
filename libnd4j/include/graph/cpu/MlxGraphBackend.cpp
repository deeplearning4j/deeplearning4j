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

#include <config.h>

#if HAVE_MLX

#include <graph/cpu/MlxGraphBackend.h>
#include <graph/DspDiagnostics.h>
#include <helpers/logger.h>
#include <system/Environment.h>

#include <algorithm>
#include <cctype>
#include <mutex>

// MLX C++ API
#include <mlx/mlx.h>

namespace mx = mlx::core;

namespace sd {
namespace graph {

MlxGraphBackend::MlxGraphBackend() = default;
MlxGraphBackend::~MlxGraphBackend() = default;

MlxGraphBackend& MlxGraphBackend::getInstance() {
  static MlxGraphBackend* instance = nullptr;
  static std::once_flag initFlag;
  std::call_once(initFlag, []() {
    instance = new MlxGraphBackend();
  });
  return *instance;
}

bool MlxGraphBackend::isAvailable() const {
  if (availabilityChecked_) return available_;
  availabilityChecked_ = true;

  try {
    // Trivial MLX op + eval to verify Metal works
    auto a = mx::array({1.0f, 2.0f, 3.0f});
    auto b = mx::array({4.0f, 5.0f, 6.0f});
    auto c = mx::add(a, b);
    mx::eval(c);

    // Verify result
    auto* data = c.data<float>();
    if (data[0] != 5.0f || data[1] != 7.0f || data[2] != 9.0f) {
      DSP_DIAG(BACKEND, "MlxGraphBackend: Metal compute verification failed");
      available_ = false;
      return false;
    }

    DSP_DIAG(BACKEND, "MlxGraphBackend: Metal compute verified OK");
    available_ = true;
  } catch (...) {
    DSP_DIAG(BACKEND, "MlxGraphBackend: Metal not available (exception during verification)");
    available_ = false;
  }

  return available_;
}

bool MlxGraphBackend::isResolvable(
    const GraphBackendRequest& request) const {
  return request.executionMode == GraphExecutionMode::GEM_MLX ||
         request.executionMode == GraphExecutionMode::GEM_AUTO ||
         request.executionMode == GraphExecutionMode::GEM_PORTABLE_REPLAY;
}

int MlxGraphBackend::resolutionPriority(
    const GraphBackendRequest& request) const {
  return request.executionMode == GraphExecutionMode::GEM_MLX ? 1000 : 700;
}

bool MlxGraphBackend::canFuseSegment(NativeSlot* slots, int start, int end) {
  if (end < start) return false;
  int segSize = end - start + 1;
  if (segSize < 2) return false;

  for (int i = start; i <= end; i++) {
    if (!MlxIRBuilder::isMlxMappable(slots[i].ident.opName)) {
      DSP_DIAG(BACKEND, "MlxGraphBackend::canFuseSegment: unmappable op '%s' at slot %d",
               slots[i].ident.opName.c_str(), i);
      return false;
    }
  }

  DSP_DIAG(SEGMENT, "MlxGraphBackend::canFuseSegment [%d-%d]: all %d ops mappable", start, end, segSize);
  return true;
}

bool MlxGraphBackend::compileSegment(GraphSegment& seg, NativeSlot* slots,
                                      NDArray** externalInputs, int numExternalInputs,
                                      NDArray** outputSlots, int totalOutputSlots,
                                      LongType shapeKey,
                                      int totalSlots,
                                      int* requestedOutputSlotIndices,
                                      int numRequestedOutputs) {
  int startSlot = seg.def.startSlot;
  int endSlot = seg.def.endSlot;

  // Check cache first
  SegmentCacheKey key{startSlot, endSlot, shapeKey};
  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    auto it = cache_.find(key);
    if (it != cache_.end() && it->second.valid) {
      DSP_DIAG(JIT, "MlxGraphBackend::compileSegment [%d-%d]: cache HIT (shapeKey=0x%llx)",
               startSlot, endSlot, (long long)shapeKey);
      lastCompilationAudit_ = it->second.compilationAudit;
      return true;
    }
  }

  DSP_DIAG(COMPILE, "MlxGraphBackend::compileSegment [%d-%d]: cache MISS, compiling (shapeKey=0x%llx)",
           startSlot, endSlot, (long long)shapeKey);

  // Analyze the segment
  auto analysis = MlxIRBuilder::analyzeSegment(slots, startSlot, endSlot, totalSlots,
                                                externalInputs, numExternalInputs,
                                                outputSlots, totalOutputSlots);

  if (!analysis.canCompile) {
    DSP_DIAG(COMPILE, "MlxGraphBackend: cannot compile segment [%d-%d]: %s",
              startSlot, endSlot, analysis.failureReason.c_str());
    return false;
  }

  // Build MLX computation graph
  auto mlxGraph = irBuilder_.buildGraph(slots, startSlot, endSlot, totalSlots,
                                         externalInputs, numExternalInputs,
                                         outputSlots, totalOutputSlots);

  if (!mlxGraph.valid) {
    DSP_DIAG(COMPILE, "MlxGraphBackend: failed to build MLX graph for segment [%d-%d]",
              startSlot, endSlot);
    return false;
  }

  // Build argument mapping
  CompiledSegment compiled;
  compiled.mlxGraph = std::move(mlxGraph);
  compiled.shapeKey = shapeKey;
  compiled.valid = true;

  // Reconstruct arg ordering via shared buildArgMappings() (GraphBackendCommon.h)
  compiled.argMappings = buildArgMappings(
      slots, startSlot, endSlot,
      externalInputs, numExternalInputs,
      outputSlots, totalOutputSlots, totalSlots,
      MlxIRBuilder::computeExternallyVisibleOutputs);

  // Build compilation audit
  for (int i = startSlot; i <= endSlot; i++) {
    CompilationAuditEntry entry;
    entry.slotIndex = i;
    entry.opName = slots[i].ident.opName;
    entry.wasCompiled = MlxIRBuilder::isMlxMappable(slots[i].ident.opName);
    if (!entry.wasCompiled) {
      entry.reason = "unsupported op category";
    }
    compiled.compilationAudit.push_back(entry);
  }

  lastCompilationAudit_ = compiled.compilationAudit;

  DSP_DIAG(COMPILE, "MlxGraphBackend: compiled segment [%d-%d] with %d buffer args (%d ops fused)",
            startSlot, endSlot, static_cast<int>(compiled.argMappings.size()),
            endSlot - startSlot + 1);

  // Cache the result
  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    cache_[key] = std::move(compiled);
  }

  return true;
}

Status MlxGraphBackend::executeSegment(GraphSegment& seg, NativeSlot* slots,
                                        NDArray** externalInputs, int numExternalInputs,
                                        NDArray** outputSlots, int totalOutputSlots,
                                        void* stream) {
  int startSlot = seg.def.startSlot;
  int endSlot = seg.def.endSlot;
  SegmentCacheKey key{startSlot, endSlot, seg.def.shapeKeyState.compiledShapeKey};

  CompiledSegment* compiled = nullptr;
  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    auto it = cache_.find(key);
    if (it == cache_.end() || !it->second.valid) {
      DSP_DIAG(EXECUTE, "MlxGraphBackend::executeSegment [%d-%d]: no compiled graph found", startSlot, endSlot);
      return Status::KERNEL_FAILURE;
    }
    compiled = &it->second;
  }

  if (!compiled->mlxGraph.valid) {
    DSP_DIAG(EXECUTE, "MlxGraphBackend::executeSegment [%d-%d]: graph invalid", startSlot, endSlot);
    return Status::KERNEL_FAILURE;
  }

  DSP_DIAG(EXECUTE, "MlxGraphBackend::executeSegment [%d-%d]: executing MLX graph", startSlot, endSlot);

  // For each output in the MLX graph, evaluate and copy back to NDArray
  try {
    for (auto& [outIdx, mlxArr] : compiled->mlxGraph.outputArrays) {
      if (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots && outputSlots[outIdx]) {
        MlxIRBuilder::mlxArrayToNDArray(mlxArr, outputSlots[outIdx]);
      }
    }
  } catch (const std::exception& e) {
    DSP_DIAG(EXECUTE, "MlxGraphBackend: execution failed for segment [%d-%d]: %s",
              startSlot, endSlot, e.what());
    return Status::KERNEL_FAILURE;
  }

  return Status::OK;
}

void MlxGraphBackend::invalidateCache() {
  std::lock_guard<std::mutex> lock(cacheMtx_);
  cache_.clear();
}

std::vector<CompilationAuditEntry> MlxGraphBackend::getLastCompilationAudit() const {
  return lastCompilationAudit_;
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_MLX
