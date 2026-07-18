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

#if defined(HAVE_MLIR) && HAVE_MLIR

#include <graph/cpu/ArmHybridGraphBackend.h>
#include <graph/DspDiagnostics.h>
#include <helpers/logger.h>
#include <system/Environment.h>

#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/BuiltinOps.h"

#include <algorithm>
#include <cctype>
#include <mutex>


namespace sd {
namespace graph {

ArmHybridGraphBackend::ArmHybridGraphBackend() = default;

ArmHybridGraphBackend::~ArmHybridGraphBackend() = default;

ArmHybridGraphBackend& ArmHybridGraphBackend::getInstance() {
  static ArmHybridGraphBackend* instance = nullptr;
  static std::once_flag initFlag;
  std::call_once(initFlag, []() {
    instance = new ArmHybridGraphBackend();
  });
  return *instance;
}

bool ArmHybridGraphBackend::isAvailable() const {
  // Available if MLIR engine can initialize (always true on ARM with MLIR)
  return sd::mlir_runtime::MLIREngine::getInstance().initialize();
}

bool ArmHybridGraphBackend::canFuseSegment(NativeSlot* slots, int start, int end) {
  if (end < start) return false;
  int segSize = end - start + 1;
  if (segSize < 2) return false;

  // Check all ops are MLIR-mappable
  for (int i = start; i <= end; i++) {
    if (!CpuIRBuilder::isMlirMappable(slots[i].ident.opName)) {
      DSP_DIAG(BACKEND, "ArmHybridGraphBackend::canFuseSegment: unmappable op '%s' at slot %d",
               slots[i].ident.opName.c_str(), i);
      return false;
    }
  }

  DSP_DIAG(SEGMENT, "ArmHybridGraphBackend::canFuseSegment [%d-%d]: all %d ops mappable",
           start, end, segSize);
  return true;
}

sd::mlir_runtime::MLIRCompileOptions ArmHybridGraphBackend::getArmCompileOptions() const {
  auto opts = sd::mlir_runtime::MLIREngine::getArmAndroidDefaults();
  // Override AOT mode — for JIT execution we don't want AOT
  opts.aotMode = false;
  opts.aotTarget = sd::mlir_runtime::AOTTarget::HOST;
  return opts;
}

bool ArmHybridGraphBackend::compileSegment(GraphSegment& seg, NativeSlot* slots,
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
      DSP_DIAG(JIT, "ArmHybridGraphBackend::compileSegment [%d-%d]: cache HIT (shapeKey=0x%llx)",
               startSlot, endSlot, (long long)shapeKey);
      lastCompilationAudit_ = it->second.compilationAudit;
      return true;
    }
  }

  DSP_DIAG(COMPILE, "ArmHybridGraphBackend::compileSegment [%d-%d]: cache MISS, compiling (shapeKey=0x%llx)",
           startSlot, endSlot, (long long)shapeKey);

  // Analyze the segment
  auto analysis = CpuIRBuilder::analyzeSegment(slots, startSlot, endSlot, totalSlots,
                                                externalInputs, numExternalInputs,
                                                outputSlots, totalOutputSlots);

  if (!analysis.canCompile) {
    DSP_DIAG(COMPILE, "ArmHybridGraphBackend: cannot compile segment [%d-%d]: %s",
              startSlot, endSlot, analysis.failureReason.c_str());
    return false;
  }

  // Build MLIR module
  auto& engine = sd::mlir_runtime::MLIREngine::getInstance();
  if (!engine.isInitialized() && !engine.initialize()) {
    DSP_DIAG(BACKEND, "ArmHybridGraphBackend: failed to initialize MLIREngine");
    return false;
  }

  auto module = irBuilder_.buildModule(engine.getContext(), slots, startSlot, endSlot,
                                       totalSlots, externalInputs, numExternalInputs,
                                       outputSlots, totalOutputSlots);
  if (!module) {
    DSP_DIAG(COMPILE, "ArmHybridGraphBackend: failed to build MLIR module for segment [%d-%d]",
              startSlot, endSlot);
    return false;
  }

  // Compile with ARM-tuned options
  auto compileOpts = getArmCompileOptions();

  auto kernel = engine.compileModule(std::move(module), "fused_kernel", compileOpts);

  if (!kernel || !kernel->isValid()) {
    DSP_DIAG(COMPILE, "ArmHybridGraphBackend: failed to compile segment [%d-%d]",
              startSlot, endSlot);
    return false;
  }

  // Build argument mapping (same logic as MlirCpuGraphBackend)
  CompiledSegment compiled;
  compiled.kernel = kernel;
  compiled.shapeKey = shapeKey;
  compiled.valid = true;

  // Reconstruct arg ordering via shared buildArgMappings() (GraphBackendCommon.h)
  compiled.argMappings = buildArgMappings(
      slots, startSlot, endSlot,
      externalInputs, numExternalInputs,
      outputSlots, totalOutputSlots, totalSlots,
      CpuIRBuilder::computeExternallyVisibleOutputs);

  // Build compilation audit
  for (int i = startSlot; i <= endSlot; i++) {
    CompilationAuditEntry entry;
    entry.slotIndex = i;
    entry.opName = slots[i].ident.opName;
    entry.wasCompiled = CpuIRBuilder::isMlirMappable(slots[i].ident.opName);
    if (!entry.wasCompiled) {
      entry.reason = "unsupported op category";
    }
    compiled.compilationAudit.push_back(entry);
  }

  lastCompilationAudit_ = compiled.compilationAudit;

  DSP_DIAG(COMPILE, "ArmHybridGraphBackend: compiled ARM CPU segment [%d-%d] with %d buffer args (%d ops fused)",
            startSlot, endSlot,
            static_cast<int>(compiled.argMappings.size()),
            endSlot - startSlot + 1);

  // Cache the result
  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    cache_[key] = std::move(compiled);
  }

  return true;
}

Status ArmHybridGraphBackend::executeSegment(GraphSegment& seg, NativeSlot* slots,
                                              NDArray** externalInputs, int numExternalInputs,
                                              NDArray** outputSlots, int totalOutputSlots,
                                              void* stream) {
  int startSlot = seg.def.startSlot;
  int endSlot = seg.def.endSlot;
  SegmentCacheKey key{startSlot, endSlot, seg.def.shapeKeyState.compiledShapeKey};

  // Look up cached kernel
  CompiledSegment* compiled = nullptr;
  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    auto it = cache_.find(key);
    if (it == cache_.end() || !it->second.valid) {
      DSP_DIAG(EXECUTE, "ArmHybridGraphBackend::executeSegment [%d-%d]: no compiled kernel found",
               startSlot, endSlot);
      return Status::KERNEL_FAILURE;
    }
    compiled = &it->second;
  }

  if (!compiled->kernel || !compiled->kernel->isValid()) {
    DSP_DIAG(EXECUTE, "ArmHybridGraphBackend::executeSegment [%d-%d]: kernel invalid", startSlot, endSlot);
    return Status::KERNEL_FAILURE;
  }

  DSP_DIAG(EXECUTE, "ArmHybridGraphBackend::executeSegment [%d-%d]: executing ARM CPU kernel with %d args",
           startSlot, endSlot, (int)compiled->argMappings.size());

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
      DSP_DIAG(EXECUTE, "ArmHybridGraphBackend: null array for arg sourceIndex=%d in segment [%d-%d]",
                mapping.sourceIndex, startSlot, endSlot);
      return Status::KERNEL_FAILURE;
    }

    if (mapping.isOutput) {
      outputArrays.push_back(arr);
    } else {
      inputArrays.push_back(arr);
    }
  }

  // Execute the kernel
  bool ok = compiled->kernel->execute(inputArrays, outputArrays);
  if (!ok) {
    DSP_DIAG(EXECUTE, "ArmHybridGraphBackend: ARM CPU execution failed for segment [%d-%d]",
              startSlot, endSlot);
    return Status::KERNEL_FAILURE;
  }

  return Status::OK;
}

void ArmHybridGraphBackend::invalidateCache() {
  std::lock_guard<std::mutex> lock(cacheMtx_);
  cache_.clear();
}

std::vector<CompilationAuditEntry> ArmHybridGraphBackend::getLastCompilationAudit() const {
  return lastCompilationAudit_;
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_MLIR
