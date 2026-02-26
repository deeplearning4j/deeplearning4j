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
#include <helpers/logger.h>
#include <system/Environment.h>

#include <algorithm>
#include <cctype>

// MLX C++ API
#include <mlx/mlx.h>

namespace mx = mlx::core;

namespace sd {
namespace graph {

MlxGraphBackend::MlxGraphBackend() = default;
MlxGraphBackend::~MlxGraphBackend() = default;

MlxGraphBackend& MlxGraphBackend::getInstance() {
  static MlxGraphBackend instance;
  return instance;
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
      sd_printf("MlxGraphBackend: Metal compute verification failed\n", "");
      available_ = false;
      return false;
    }

    sd_printf("MlxGraphBackend: Metal compute verified OK\n", "");
    available_ = true;
  } catch (...) {
    sd_printf("MlxGraphBackend: Metal not available (exception during verification)\n", "");
    available_ = false;
  }

  return available_;
}

bool MlxGraphBackend::canFuseSegment(NativeSlot* slots, int start, int end) {
  if (end < start) return false;
  int segSize = end - start + 1;
  if (segSize < 2) return false;

  for (int i = start; i <= end; i++) {
    if (!MlxIRBuilder::isMlxMappable(slots[i].opName)) {
      return false;
    }
  }

  return true;
}

bool MlxGraphBackend::compileSegment(GraphSegment& seg, NativeSlot* slots,
                                      NDArray** externalInputs, int numExternalInputs,
                                      NDArray** outputSlots, int totalOutputSlots,
                                      LongType shapeKey,
                                      int totalSlots,
                                      int* requestedOutputSlotIndices,
                                      int numRequestedOutputs) {
  int startSlot = seg.startSlot;
  int endSlot = seg.endSlot;

  // Check cache first
  SegmentCacheKey key{startSlot, endSlot, shapeKey};
  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    auto it = cache_.find(key);
    if (it != cache_.end() && it->second.valid) {
      lastCompilationAudit_ = it->second.compilationAudit;
      return true;
    }
  }

  // Analyze the segment
  auto analysis = MlxIRBuilder::analyzeSegment(slots, startSlot, endSlot, totalSlots,
                                                externalInputs, numExternalInputs,
                                                outputSlots, totalOutputSlots);

  if (!analysis.canCompile) {
    sd_printf("MlxGraphBackend: cannot compile segment [%d-%d]: %s\n",
              startSlot, endSlot, analysis.failureReason.c_str());
    return false;
  }

  // Build MLX computation graph
  auto mlxGraph = irBuilder_.buildGraph(slots, startSlot, endSlot, totalSlots,
                                         externalInputs, numExternalInputs,
                                         outputSlots, totalOutputSlots);

  if (!mlxGraph.valid) {
    sd_printf("MlxGraphBackend: failed to build MLX graph for segment [%d-%d]\n",
              startSlot, endSlot);
    return false;
  }

  // Build argument mapping
  CompiledSegment compiled;
  compiled.mlxGraph = std::move(mlxGraph);
  compiled.shapeKey = shapeKey;
  compiled.valid = true;

  // Reconstruct arg ordering: inputs then outputs
  std::unordered_set<int> internalOutputs;
  for (int i = startSlot; i <= endSlot; i++) {
    for (int o = 0; o < slots[i].numOutputs; o++) {
      internalOutputs.insert(slots[i].outputSlotIndices[o]);
    }
  }

  std::unordered_set<int> seenSrc;
  // Input args
  for (int i = startSlot; i <= endSlot; i++) {
    for (int inp = 0; inp < slots[i].numInputs; inp++) {
      int srcIdx = slots[i].inputSourceIndices[inp];
      if (seenSrc.count(srcIdx)) continue;

      bool isExternal = (srcIdx < 0);
      bool isPreSegment = (!isExternal && !internalOutputs.count(srcIdx));

      if (isExternal || isPreSegment) {
        NDArray* arr = nullptr;
        if (isExternal) {
          int extIdx = -(srcIdx + 1);
          if (extIdx < numExternalInputs && externalInputs) arr = externalInputs[extIdx];
        } else {
          if (srcIdx < totalOutputSlots && outputSlots) arr = outputSlots[srcIdx];
        }
        if (!arr) continue;

        seenSrc.insert(srcIdx);
        compiled.argMappings.push_back({srcIdx, false});
      }
    }
  }

  // Output args (externally visible)
  auto externalOutputSet = MlxIRBuilder::computeExternallyVisibleOutputs(
      slots, startSlot, endSlot, totalSlots);
  for (int i = startSlot; i <= endSlot; i++) {
    for (int o = 0; o < slots[i].numOutputs; o++) {
      int outIdx = slots[i].outputSlotIndices[o];
      if (!externalOutputSet.count(outIdx)) continue;
      if (seenSrc.count(outIdx)) continue;
      seenSrc.insert(outIdx);

      NDArray* arr = nullptr;
      if (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots) arr = outputSlots[outIdx];
      if (!arr) continue;

      compiled.argMappings.push_back({outIdx, true});
    }
  }

  // Internal intermediate outputs
  for (int i = startSlot; i <= endSlot; i++) {
    for (int o = 0; o < slots[i].numOutputs; o++) {
      int outIdx = slots[i].outputSlotIndices[o];
      if (seenSrc.count(outIdx)) continue;

      NDArray* arr = nullptr;
      if (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots) arr = outputSlots[outIdx];
      if (!arr) continue;

      seenSrc.insert(outIdx);
      compiled.argMappings.push_back({outIdx, true});
    }
  }

  // Build compilation audit
  for (int i = startSlot; i <= endSlot; i++) {
    CompilationAuditEntry entry;
    entry.slotIndex = i;
    entry.opName = slots[i].opName;
    entry.wasCompiled = MlxIRBuilder::isMlxMappable(slots[i].opName);
    if (!entry.wasCompiled) {
      entry.reason = "unsupported op category";
    }
    compiled.compilationAudit.push_back(entry);
  }

  lastCompilationAudit_ = compiled.compilationAudit;

  sd_printf("MlxGraphBackend: compiled segment [%d-%d] with %d buffer args (%d ops fused)\n",
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
  int startSlot = seg.startSlot;
  int endSlot = seg.endSlot;
  SegmentCacheKey key{startSlot, endSlot, seg.shapeKey};

  CompiledSegment* compiled = nullptr;
  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    auto it = cache_.find(key);
    if (it == cache_.end() || !it->second.valid) {
      return Status::KERNEL_FAILURE;
    }
    compiled = &it->second;
  }

  if (!compiled->mlxGraph.valid) {
    return Status::KERNEL_FAILURE;
  }

  // For each output in the MLX graph, evaluate and copy back to NDArray
  try {
    for (auto& [outIdx, mlxArr] : compiled->mlxGraph.outputArrays) {
      if (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots && outputSlots[outIdx]) {
        MlxIRBuilder::mlxArrayToNDArray(mlxArr, outputSlots[outIdx]);
      }
    }
  } catch (const std::exception& e) {
    sd_printf("MlxGraphBackend: execution failed for segment [%d-%d]: %s\n",
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
