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

#ifdef HAVE_MLIR

#include <graph/cpu/MlirCpuGraphBackend.h>
#include <helpers/logger.h>
#include <system/Environment.h>

#include <algorithm>
#include <cctype>

namespace sd {
namespace graph {

MlirCpuGraphBackend::MlirCpuGraphBackend() = default;
MlirCpuGraphBackend::~MlirCpuGraphBackend() = default;

MlirCpuGraphBackend& MlirCpuGraphBackend::getInstance() {
  static MlirCpuGraphBackend instance;
  return instance;
}

bool MlirCpuGraphBackend::isAvailable() const {
  return sd::mlir_runtime::MLIREngine::getInstance().initialize();
}

bool MlirCpuGraphBackend::canFuseSegment(NativeSlot* slots, int start, int end) {
  if (end < start) return false;
  int segSize = end - start + 1;
  if (segSize < 2) return false;  // Need at least 2 ops to be worth fusing

  // Check all ops are MLIR-mappable
  for (int i = start; i <= end; i++) {
    if (!CpuIRBuilder::isMlirMappable(slots[i].opName)) {
      return false;
    }
  }

  return true;
}

bool MlirCpuGraphBackend::compileSegment(GraphSegment& seg, NativeSlot* slots,
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
  auto analysis = CpuIRBuilder::analyzeSegment(slots, startSlot, endSlot, totalSlots,
                                                externalInputs, numExternalInputs,
                                                outputSlots, totalOutputSlots);

  if (!analysis.canCompile) {
    sd_printf("MlirCpuGraphBackend: cannot compile segment [%d-%d]: %s\n",
              startSlot, endSlot, analysis.failureReason.c_str());
    return false;
  }

  // Build MLIR module
  auto& engine = sd::mlir_runtime::MLIREngine::getInstance();
  if (!engine.isInitialized() && !engine.initialize()) {
    sd_printf("MlirCpuGraphBackend: failed to initialize MLIREngine\n", "");
    return false;
  }

  auto module = irBuilder_.buildModule(engine.getContext(), slots, startSlot, endSlot,
                                       totalSlots, externalInputs, numExternalInputs,
                                       outputSlots, totalOutputSlots);
  if (!module) {
    sd_printf("MlirCpuGraphBackend: failed to build MLIR module for segment [%d-%d]\n",
              startSlot, endSlot);
    return false;
  }

  // Compile through MLIREngine's CPU pipeline
  auto kernel = engine.compileModule(std::move(module), "fused_kernel");
  if (!kernel || !kernel->isValid()) {
    sd_printf("MlirCpuGraphBackend: failed to JIT-compile segment [%d-%d]\n",
              startSlot, endSlot);
    return false;
  }

  // Build argument mapping (must match the order CpuIRBuilder used)
  CompiledSegment compiled;
  compiled.kernel = kernel;
  compiled.shapeKey = shapeKey;
  compiled.valid = true;

  // Reconstruct the same arg ordering as buildModule():
  // 1. Input args: external and pre-segment sources
  // 2. Output args: externally visible outputs
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
        // Verify the source array exists
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
  auto externalOutputSet = CpuIRBuilder::computeExternallyVisibleOutputs(
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

  // Internal intermediate outputs (needed by non-element-wise ops as full memrefs)
  // This must match the third phase of CpuIRBuilder::buildModule()
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
    entry.wasCompiled = CpuIRBuilder::isMlirMappable(slots[i].opName);
    if (!entry.wasCompiled) {
      entry.reason = "unsupported op category";
    }
    compiled.compilationAudit.push_back(entry);
  }

  lastCompilationAudit_ = compiled.compilationAudit;

  sd_printf("MlirCpuGraphBackend: compiled segment [%d-%d] with %d buffer args (%d ops fused)\n",
            startSlot, endSlot, static_cast<int>(compiled.argMappings.size()),
            endSlot - startSlot + 1);

  // Cache the result
  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    cache_[key] = std::move(compiled);
  }

  return true;
}

Status MlirCpuGraphBackend::executeSegment(GraphSegment& seg, NativeSlot* slots,
                                            NDArray** externalInputs, int numExternalInputs,
                                            NDArray** outputSlots, int totalOutputSlots,
                                            void* stream) {
  int startSlot = seg.startSlot;
  int endSlot = seg.endSlot;
  SegmentCacheKey key{startSlot, endSlot, seg.shapeKey};

  // Look up cached kernel
  CompiledSegment* compiled = nullptr;
  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    auto it = cache_.find(key);
    if (it == cache_.end() || !it->second.valid) {
      return Status::KERNEL_FAILURE;
    }
    compiled = &it->second;
  }

  if (!compiled->kernel || !compiled->kernel->isValid()) {
    return Status::KERNEL_FAILURE;
  }

  // Wire NDArray buffers to kernel arguments in the same order as compilation
  std::vector<NDArray*> inputArrays;
  std::vector<NDArray*> outputArrays;

  for (auto& mapping : compiled->argMappings) {
    NDArray* arr = nullptr;
    if (mapping.sourceIndex < 0) {
      int extIdx = -(mapping.sourceIndex + 1);
      if (extIdx < numExternalInputs && externalInputs) arr = externalInputs[extIdx];
    } else {
      if (mapping.sourceIndex < totalOutputSlots && outputSlots)
        arr = outputSlots[mapping.sourceIndex];
    }

    if (!arr) {
      sd_printf("MlirCpuGraphBackend: null array for arg sourceIndex=%d in segment [%d-%d]\n",
                mapping.sourceIndex, startSlot, endSlot);
      return Status::KERNEL_FAILURE;
    }

    if (mapping.isOutput) {
      outputArrays.push_back(arr);
    } else {
      inputArrays.push_back(arr);
    }
  }

  // Execute the JIT kernel
  bool ok = compiled->kernel->execute(inputArrays, outputArrays);
  if (!ok) {
    sd_printf("MlirCpuGraphBackend: JIT execution failed for segment [%d-%d]\n",
              startSlot, endSlot);
    return Status::KERNEL_FAILURE;
  }

  return Status::OK;
}

void MlirCpuGraphBackend::invalidateCache() {
  std::lock_guard<std::mutex> lock(cacheMtx_);
  cache_.clear();
}

std::vector<CompilationAuditEntry> MlirCpuGraphBackend::getLastCompilationAudit() const {
  return lastCompilationAudit_;
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_MLIR
