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

#ifdef HAVE_HEXAGON_MLIR

#include <graph/hexagon/HexagonGraphBackend.h>
#include <graph/hexagon/HexagonIRBuilder.h>
#include <graph/hexagon/HexagonRuntimeManager.h>
#include <graph/NativeDynamicShapePlan.h>
#include <graph/DspDiagnostics.h>

#include <cstring>

namespace sd {
namespace graph {

// ── Singleton ────────────────────────────────────────────────────────────────

HexagonGraphBackend::HexagonGraphBackend() = default;

HexagonGraphBackend::~HexagonGraphBackend() {
  invalidateCache();

  // Release NPU context
  if (npuContext_ != nullptr) {
    HexagonRuntimeManager::getInstance().releaseDevice(npuContext_);
    npuContext_ = nullptr;
  }
}

HexagonGraphBackend* HexagonGraphBackend::getInstance() {
  static HexagonGraphBackend instance;
  return &instance;
}

// ── NPU Context ─────────────────────────────────────────────────────────────

bool HexagonGraphBackend::ensureNpuContext() {
  if (npuContext_ != nullptr) return true;

  auto& runtime = HexagonRuntimeManager::getInstance();
  if (!runtime.isAvailable()) return false;

  npuContext_ = runtime.initDevice(0);
  if (npuContext_ == nullptr) {
    DSP_DIAG(BACKEND, "HexagonGraphBackend: failed to initialize NPU device 0");
    return false;
  }

  DSP_DIAG(BACKEND, "HexagonGraphBackend: NPU context initialized: %p", npuContext_);
  return true;
}

// ── Availability ─────────────────────────────────────────────────────────────

bool HexagonGraphBackend::isAvailable() const {
  auto& runtime = HexagonRuntimeManager::getInstance();
  return runtime.isAvailable() && runtime.getDeviceCount() > 0;
}

// ── Segment Fusion Check ─────────────────────────────────────────────────────

bool HexagonGraphBackend::canFuseSegment(NativeSlot* slots, int start, int end) {
  if (slots == nullptr || start >= end) return false;

  int totalOps = end - start;
  int mappableOps = 0;

  for (int i = start; i < end; i++) {
    if (HexagonIRBuilder::isHexagonMappable(slots[i].opName)) {
      mappableOps++;
    }
  }

  // Check minimum mappable fraction
  float fraction = static_cast<float>(mappableOps) / static_cast<float>(totalOps);
  if (fraction < MIN_MAPPABLE_FRACTION) {
    DSP_DIAG(SEGMENT, "HexagonGraphBackend::canFuseSegment: [%d, %d) only %.0f%% "
             "mappable (%d/%d), need %.0f%%",
             start, end, fraction * 100, mappableOps, totalOps,
             MIN_MAPPABLE_FRACTION * 100);
    return false;
  }

  // Check TCM capacity
  size_t estimatedTcm = HexagonIRBuilder::estimateTcmUsage(slots, start, end);

  // Use runtime TCM capacity if available, otherwise default
  size_t tcmCapacity = TCM_CAPACITY;
  if (npuContext_ != nullptr) {
    size_t runtimeCapacity = HexagonRuntimeManager::getInstance().getTcmCapacity(npuContext_);
    if (runtimeCapacity > 0) {
      tcmCapacity = runtimeCapacity;
    }
  }

  if (estimatedTcm > tcmCapacity) {
    DSP_DIAG(SEGMENT, "HexagonGraphBackend::canFuseSegment: [%d, %d) estimated TCM "
             "%zuKB > capacity %zuKB",
             start, end, estimatedTcm / 1024, tcmCapacity / 1024);
    return false;
  }

  DSP_DIAG(SEGMENT, "HexagonGraphBackend::canFuseSegment: [%d, %d) fusible "
           "(%d/%d mappable, %zuKB TCM est.)",
           start, end, mappableOps, totalOps, estimatedTcm / 1024);
  return true;
}

// ── Compilation ─────────────────────────────────────────────────────────────

bool HexagonGraphBackend::compileSegment(GraphSegment& seg, NativeSlot* slots,
                                          NDArray** externalInputs,
                                          int numExternalInputs,
                                          NDArray** outputSlots,
                                          int totalOutputSlots,
                                          LongType shapeKey,
                                          int totalSlots,
                                          int* requestedOutputSlotIndices,
                                          int numRequestedOutputs) {
  std::lock_guard<std::mutex> lock(cacheMtx_);

  SegmentCacheKey key{seg.startSlot, seg.endSlot, shapeKey};

  // Check negative cache
  if (failedCache_.find(key) != failedCache_.end()) {
    DSP_DIAG(COMPILE, "HexagonGraphBackend::compileSegment: [%d, %d) in failed cache, "
             "skipping", seg.startSlot, seg.endSlot);
    return false;
  }

  // Check positive cache
  auto it = cache_.find(key);
  if (it != cache_.end() && it->second.kernelHandle != nullptr) {
    DSP_DIAG(COMPILE, "HexagonGraphBackend::compileSegment: [%d, %d) cache hit",
             seg.startSlot, seg.endSlot);
    lastCompilationAudit_ = it->second.audit;
    return true;
  }

  // Ensure NPU context is initialized
  if (!ensureNpuContext()) {
    DSP_DIAG(COMPILE, "HexagonGraphBackend::compileSegment: NPU context unavailable");
    failedCache_.insert(key);
    return false;
  }

  // Build MLIR bytecode from the slot range
  std::vector<uint8_t> mlirBytecode = HexagonIRBuilder::buildModule(
      slots, seg.startSlot, seg.endSlot, externalInputs, numExternalInputs);

  if (mlirBytecode.empty()) {
    DSP_DIAG(COMPILE, "HexagonGraphBackend::compileSegment: [%d, %d) MLIR build failed",
             seg.startSlot, seg.endSlot);
    failedCache_.insert(key);
    return false;
  }

  // Compile MLIR to NPU kernel
  auto& runtime = HexagonRuntimeManager::getInstance();
  void* kernelHandle = runtime.compileKernel(
      npuContext_, mlirBytecode.data(), mlirBytecode.size());

  if (kernelHandle == nullptr) {
    DSP_DIAG(COMPILE, "HexagonGraphBackend::compileSegment: [%d, %d) kernel "
             "compilation failed", seg.startSlot, seg.endSlot);
    failedCache_.insert(key);
    return false;
  }

  // Build compilation audit
  std::vector<CompilationAuditEntry> audit;
  for (int i = seg.startSlot; i < seg.endSlot; i++) {
    CompilationAuditEntry entry;
    entry.slotIndex = i;
    entry.opName = slots[i].opName ? slots[i].opName : "";
    entry.wasCompiled = HexagonIRBuilder::isHexagonMappable(slots[i].opName);
    if (!entry.wasCompiled) {
      entry.reason = "not HVX-mappable";
    }
    audit.push_back(std::move(entry));
  }

  // Store in cache
  CompiledKernel compiled;
  compiled.kernelHandle = kernelHandle;
  compiled.npuContext = npuContext_;
  compiled.startSlot = seg.startSlot;
  compiled.endSlot = seg.endSlot;
  compiled.audit = audit;

  cache_[key] = std::move(compiled);
  lastCompilationAudit_ = audit;

  DSP_DIAG(COMPILE, "HexagonGraphBackend::compileSegment: [%d, %d) compiled "
           "successfully, kernel=%p",
           seg.startSlot, seg.endSlot, kernelHandle);
  return true;
}

// ── Execution ───────────────────────────────────────────────────────────────

Status HexagonGraphBackend::executeSegment(GraphSegment& seg, NativeSlot* slots,
                                            NDArray** externalInputs,
                                            int numExternalInputs,
                                            NDArray** outputSlots,
                                            int totalOutputSlots,
                                            void* stream) {
  std::lock_guard<std::mutex> lock(cacheMtx_);

  // Find the compiled kernel — search cache for matching segment range
  CompiledKernel* compiled = nullptr;
  for (auto& pair : cache_) {
    if (pair.first.startSlot == seg.startSlot &&
        pair.first.endSlot == seg.endSlot) {
      compiled = &pair.second;
      break;
    }
  }

  if (compiled == nullptr || compiled->kernelHandle == nullptr) {
    DSP_DIAG(EXECUTION, "HexagonGraphBackend::executeSegment: [%d, %d) no compiled "
             "kernel found", seg.startSlot, seg.endSlot);
    return Status::KERNEL_FAILURE;
  }

  auto& runtime = HexagonRuntimeManager::getInstance();

  // Collect all input/output buffer pointers as kernel arguments
  std::vector<void*> kernelArgs;

  for (int i = seg.startSlot; i < seg.endSlot; i++) {
    const auto& slot = slots[i];

    // Stage inputs
    for (int j = 0; j < slot.numInputs; j++) {
      NDArray* inputArr = nullptr;

      if (slot.inputSlotIndices != nullptr && slot.inputSlotIndices[j] >= 0) {
        // Cross-slot reference — use the output from the producing slot
        int srcSlot = slot.inputSlotIndices[j];
        if (srcSlot >= 0 && srcSlot < totalOutputSlots && outputSlots[srcSlot] != nullptr) {
          inputArr = outputSlots[srcSlot];
        }
      } else if (slot.inputArrays != nullptr && slot.inputArrays[j] != nullptr) {
        inputArr = slot.inputArrays[j];
      }

      if (inputArr != nullptr) {
        kernelArgs.push_back(inputArr->buffer());
      }
    }

    // Output buffer
    if (slot.outputArray != nullptr) {
      kernelArgs.push_back(slot.outputArray->buffer());
    }
  }

  // Dispatch to NPU
  if (!runtime.dispatchKernel(compiled->npuContext, compiled->kernelHandle,
                               kernelArgs.data(),
                               static_cast<int>(kernelArgs.size()))) {
    DSP_DIAG(EXECUTION, "HexagonGraphBackend::executeSegment: [%d, %d) dispatch failed",
             seg.startSlot, seg.endSlot);
    return Status::KERNEL_FAILURE;
  }

  // Wait for NPU completion
  if (!runtime.waitForCompletion(compiled->npuContext)) {
    DSP_DIAG(EXECUTION, "HexagonGraphBackend::executeSegment: [%d, %d) "
             "waitForCompletion failed", seg.startSlot, seg.endSlot);
    return Status::KERNEL_FAILURE;
  }

  DSP_DIAG(EXECUTION, "HexagonGraphBackend::executeSegment: [%d, %d) executed "
           "successfully", seg.startSlot, seg.endSlot);
  return Status::OK;
}

// ── Cache Management ────────────────────────────────────────────────────────

void HexagonGraphBackend::invalidateCache() {
  std::lock_guard<std::mutex> lock(cacheMtx_);

  auto& runtime = HexagonRuntimeManager::getInstance();

  // Release all compiled kernels
  for (auto& pair : cache_) {
    if (pair.second.kernelHandle != nullptr && pair.second.npuContext != nullptr) {
      runtime.releaseKernel(pair.second.npuContext, pair.second.kernelHandle);
    }
  }
  cache_.clear();
  failedCache_.clear();
  lastCompilationAudit_.clear();

  DSP_DIAG(COMPILE, "HexagonGraphBackend::invalidateCache: all caches cleared");
}

std::vector<CompilationAuditEntry> HexagonGraphBackend::getLastCompilationAudit() const {
  return lastCompilationAudit_;
}

// ── DMA Helpers ─────────────────────────────────────────────────────────────

bool HexagonGraphBackend::stageInputsToTcm(const NativeSlot& slot,
                                            NativeSlot* slots,
                                            NDArray** externalInputs,
                                            int numExternalInputs,
                                            NDArray** outputSlots,
                                            int totalOutputSlots,
                                            std::vector<void*>& tcmPtrs) {
  auto& runtime = HexagonRuntimeManager::getInstance();
  tcmPtrs.clear();

  for (int j = 0; j < slot.numInputs; j++) {
    NDArray* inputArr = nullptr;

    if (slot.inputSlotIndices != nullptr && slot.inputSlotIndices[j] >= 0) {
      int srcSlot = slot.inputSlotIndices[j];
      if (srcSlot >= 0 && srcSlot < totalOutputSlots && outputSlots[srcSlot] != nullptr) {
        inputArr = outputSlots[srcSlot];
      }
    } else if (slot.inputArrays != nullptr && slot.inputArrays[j] != nullptr) {
      inputArr = slot.inputArrays[j];
    }

    if (inputArr == nullptr) {
      tcmPtrs.push_back(nullptr);
      continue;
    }

    size_t bytes = inputArr->lengthOf() * inputArr->sizeOfT();

    // Allocate TCM for this input
    void* tcmPtr = runtime.allocateTcm(npuContext_, bytes);
    if (tcmPtr == nullptr) {
      DSP_DIAG(MEMORY, "HexagonGraphBackend: TCM alloc failed for input %d (%zu bytes)",
               j, bytes);
      // Free previously allocated TCM in this call
      for (auto* ptr : tcmPtrs) {
        if (ptr != nullptr) runtime.freeTcm(npuContext_, ptr);
      }
      tcmPtrs.clear();
      return false;
    }

    // DMA from host to TCM
    if (!runtime.dmaHostToTcm(npuContext_, tcmPtr, inputArr->buffer(), bytes)) {
      DSP_DIAG(MEMORY, "HexagonGraphBackend: DMA host->TCM failed for input %d", j);
      runtime.freeTcm(npuContext_, tcmPtr);
      for (auto* ptr : tcmPtrs) {
        if (ptr != nullptr) runtime.freeTcm(npuContext_, ptr);
      }
      tcmPtrs.clear();
      return false;
    }

    tcmPtrs.push_back(tcmPtr);
  }

  return true;
}

bool HexagonGraphBackend::stageOutputFromTcm(NDArray* outputArray,
                                              void* tcmPtr, size_t bytes) {
  if (outputArray == nullptr || tcmPtr == nullptr || bytes == 0) return false;

  auto& runtime = HexagonRuntimeManager::getInstance();
  return runtime.dmaTcmToHost(npuContext_, outputArray->buffer(), tcmPtr, bytes);
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_HEXAGON_MLIR
