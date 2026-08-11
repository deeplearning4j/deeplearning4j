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

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>

namespace sd {
namespace graph {
namespace {

std::string artifactPathForSegment(const std::string& artifactDirectory,
                                   int startSlot, int endSlot,
                                   LongType shapeKey) {
  std::ostringstream name;
  if (!artifactDirectory.empty()) {
    name << artifactDirectory;
    const char last = artifactDirectory.back();
    if (last != '/' && last != '\\') name << '/';
  }
  name << "hexagon_" << startSlot << '_' << endSlot << '_'
       << std::hex << std::setw(16) << std::setfill('0')
       << static_cast<uint64_t>(shapeKey) << ".bin";
  return name.str();
}

bool readBinaryArtifact(const std::string& path, std::vector<uint8_t>* bytes) {
  if (bytes == nullptr || path.empty()) return false;
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input.is_open()) return false;
  const std::streamoff size = input.tellg();
  if (size <= 0) return false;
  input.seekg(0, std::ios::beg);
  bytes->resize(static_cast<size_t>(size));
  input.read(reinterpret_cast<char*>(bytes->data()), size);
  return input.gcount() == size;
}

}  // namespace

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

HexagonGraphBackend& HexagonGraphBackend::getInstance() {
  static HexagonGraphBackend* instance = nullptr;
  static std::once_flag initFlag;
  std::call_once(initFlag, []() {
    instance = new HexagonGraphBackend();
  });
  return *instance;
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

bool HexagonGraphBackend::isResolvable(
    const GraphBackendRequest& request) const {
  return request.executionMode == GraphExecutionMode::GEM_HEXAGON ||
         request.executionMode == GraphExecutionMode::GEM_AUTO;
}

int HexagonGraphBackend::resolutionPriority(
    const GraphBackendRequest& request) const {
  return request.executionMode == GraphExecutionMode::GEM_HEXAGON ? 1000 : 300;
}

// ── Segment Fusion Check ─────────────────────────────────────────────────────

bool HexagonGraphBackend::canFuseSegment(NativeSlot* slots, int start, int end) {
  return canResolveSegment(
      GraphBackendRequest{GraphExecutionMode::GEM_AUTO}, slots, start, end);
}

bool HexagonGraphBackend::canResolveSegment(
    const GraphBackendRequest& request, NativeSlot* slots, int start, int end) {
  if (slots == nullptr || start > end) return false;

  int totalOps = end - start + 1;
  int mappableOps = 0;

  for (int i = start; i <= end; i++) {
    if (HexagonIRBuilder::isHexagonMappable(slots[i].ident.opName.c_str())) {
      mappableOps++;
    }
  }

  // Check minimum mappable fraction. A vendor AOT segment is authoritative
  // for op coverage; the exact shape-keyed artifact is verified during compile.
  float fraction = static_cast<float>(mappableOps) / static_cast<float>(totalOps);
  const bool strictAot =
      !request.runtimeCompilationAllowed &&
      !request.runtimeArtifactDirectory.empty();
  if (!strictAot && fraction < MIN_MAPPABLE_FRACTION) {
    DSP_DIAG(SEGMENT, "HexagonGraphBackend::canFuseSegment: [%d, %d] only %.0f%% "
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
    DSP_DIAG(SEGMENT, "HexagonGraphBackend::canFuseSegment: [%d, %d] estimated TCM "
             "%zuKB > capacity %zuKB",
             start, end, estimatedTcm / 1024, tcmCapacity / 1024);
    return false;
  }

  DSP_DIAG(SEGMENT, "HexagonGraphBackend::canFuseSegment: [%d, %d] fusible "
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
  return compileSegment(
      GraphBackendRequest{GraphExecutionMode::GEM_AUTO}, seg, slots,
      externalInputs, numExternalInputs, outputSlots, totalOutputSlots,
      shapeKey, totalSlots, requestedOutputSlotIndices, numRequestedOutputs);
}

bool HexagonGraphBackend::compileSegment(
    const GraphBackendRequest& request, GraphSegment& seg, NativeSlot* slots,
    NDArray** externalInputs, int numExternalInputs, NDArray** outputSlots,
    int totalOutputSlots, LongType shapeKey, int totalSlots,
    int* requestedOutputSlotIndices, int numRequestedOutputs) {
  std::lock_guard<std::mutex> lock(cacheMtx_);

  SegmentCacheKey key{seg.def.startSlot, seg.def.endSlot, shapeKey,
                      request.runtimeCompilationAllowed,
                      request.runtimeArtifactDirectory};

  // Check negative cache
  if (failedCache_.find(key) != failedCache_.end()) {
    DSP_DIAG(COMPILE, "HexagonGraphBackend::compileSegment: [%d, %d] in failed cache, "
             "skipping", seg.def.startSlot, seg.def.endSlot);
    return false;
  }

  // Check positive cache
  auto it = cache_.find(key);
  if (it != cache_.end() && it->second.kernelHandle != nullptr) {
    DSP_DIAG(COMPILE, "HexagonGraphBackend::compileSegment: [%d, %d] cache hit",
             seg.def.startSlot, seg.def.endSlot);
    lastCompilationAudit_ = it->second.audit;
    return true;
  }

  // Ensure NPU context is initialized
  if (!ensureNpuContext()) {
    DSP_DIAG(COMPILE, "HexagonGraphBackend::compileSegment: NPU context unavailable");
    failedCache_.insert(key);
    return false;
  }

  auto& runtime = HexagonRuntimeManager::getInstance();
  void* kernelHandle = nullptr;
  bool loadedFromAot = false;

  // Production/mobile sessions import deterministic vendor AOT kernels before
  // considering runtime compilation. The artifact name is part of the public
  // SDX bundle contract and includes segment range plus the canonical shape key.
  const std::string artifactPath = artifactPathForSegment(
      request.runtimeArtifactDirectory, seg.def.startSlot, seg.def.endSlot,
      shapeKey);
  std::vector<uint8_t> aotKernel;
  if (!request.runtimeArtifactDirectory.empty() &&
      readBinaryArtifact(artifactPath, &aotKernel)) {
    kernelHandle = runtime.loadKernel(npuContext_, aotKernel.data(),
                                      aotKernel.size());
    if (kernelHandle != nullptr) {
      loadedFromAot = true;
      DSP_DIAG(COMPILE,
               "HexagonGraphBackend::compileSegment: [%d, %d] loaded AOT "
               "kernel %s",
               seg.def.startSlot, seg.def.endSlot, artifactPath.c_str());
    }
  }

  if (kernelHandle == nullptr && !request.runtimeCompilationAllowed) {
    DSP_DIAG(COMPILE,
             "HexagonGraphBackend::compileSegment: [%d, %d] strict AOT "
             "artifact unavailable or rejected: %s",
             seg.def.startSlot, seg.def.endSlot, artifactPath.c_str());
    failedCache_.insert(key);
    return false;
  }

  if (kernelHandle == nullptr) {
    // Development path only: build IR and ask the vendor adapter to compile it.
    std::vector<uint8_t> mlirBytecode = HexagonIRBuilder::buildModule(
        slots, seg.def.startSlot, seg.def.endSlot, externalInputs,
        numExternalInputs);
    if (mlirBytecode.empty()) {
      DSP_DIAG(COMPILE,
               "HexagonGraphBackend::compileSegment: [%d, %d] MLIR build failed",
               seg.def.startSlot, seg.def.endSlot);
      failedCache_.insert(key);
      return false;
    }
    kernelHandle = runtime.compileKernel(npuContext_, mlirBytecode.data(),
                                         mlirBytecode.size());
    if (kernelHandle == nullptr) {
      DSP_DIAG(COMPILE,
               "HexagonGraphBackend::compileSegment: [%d, %d] kernel "
               "compilation failed",
               seg.def.startSlot, seg.def.endSlot);
      failedCache_.insert(key);
      return false;
    }
  }

  // Build compilation audit
  std::vector<CompilationAuditEntry> audit;
  for (int i = seg.def.startSlot; i <= seg.def.endSlot; i++) {
    CompilationAuditEntry entry;
    entry.slotIndex = i;
    entry.opName = slots[i].ident.opName.c_str();
    entry.wasCompiled = loadedFromAot ||
        HexagonIRBuilder::isHexagonMappable(slots[i].ident.opName.c_str());
    if (loadedFromAot) {
      entry.reason = "covered by vendor AOT segment";
    } else if (!entry.wasCompiled) {
      entry.reason = "not HVX-mappable";
    }
    audit.push_back(std::move(entry));
  }

  // Store in cache
  CompiledKernel compiled;
  compiled.kernelHandle = kernelHandle;
  compiled.npuContext = npuContext_;
  compiled.startSlot = seg.def.startSlot;
  compiled.endSlot = seg.def.endSlot;
  compiled.audit = audit;

  cache_[key] = std::move(compiled);
  lastCompilationAudit_ = audit;

  DSP_DIAG(COMPILE, "HexagonGraphBackend::compileSegment: [%d, %d] compiled "
           "successfully, kernel=%p",
           seg.def.startSlot, seg.def.endSlot, kernelHandle);
  return true;
}

// ── Execution ───────────────────────────────────────────────────────────────

Status HexagonGraphBackend::executeSegment(GraphSegment& seg, NativeSlot* slots,
                                            NDArray** externalInputs,
                                            int numExternalInputs,
                                            NDArray** outputSlots,
                                            int totalOutputSlots,
                                            void* stream) {
  return executeSegment(
      GraphBackendRequest{GraphExecutionMode::GEM_AUTO}, seg, slots,
      externalInputs, numExternalInputs, outputSlots, totalOutputSlots, stream);
}

Status HexagonGraphBackend::executeSegment(
    const GraphBackendRequest& request, GraphSegment& seg, NativeSlot* slots,
    NDArray** externalInputs, int numExternalInputs, NDArray** outputSlots,
    int totalOutputSlots, void* stream) {
  std::lock_guard<std::mutex> lock(cacheMtx_);

  // Resolve the exact kernel domain. A strict AOT context must never reuse a
  // process-wide entry created by a JIT-enabled context with the same shapes.
  SegmentCacheKey key{seg.def.startSlot, seg.def.endSlot,
                      seg.def.shapeKeyState.compiledShapeKey,
                      request.runtimeCompilationAllowed,
                      request.runtimeArtifactDirectory};
  auto compiledIt = cache_.find(key);
  CompiledKernel* compiled =
      compiledIt == cache_.end() ? nullptr : &compiledIt->second;

  if (compiled == nullptr || compiled->kernelHandle == nullptr) {
    DSP_DIAG(EXECUTE, "HexagonGraphBackend::executeSegment: [%d, %d] no compiled "
             "kernel found", seg.def.startSlot, seg.def.endSlot);
    return Status::KERNEL_FAILURE;
  }

  auto& runtime = HexagonRuntimeManager::getInstance();

  // Collect all input/output buffer pointers as kernel arguments.
  // Index convention for wiring.inputSourceIndices[j]:
  //   >= 0 : index into flat outputSlots array (prior op's output)
  //   <  0 : external input at -(srcIdx+1) into externalInputs
  std::vector<void*> kernelArgs;

  for (int i = seg.def.startSlot; i <= seg.def.endSlot; i++) {
    const auto& slot = slots[i];

    // Stage inputs
    for (int j = 0; j < slot.wiring.numInputs; j++) {
      NDArray* inputArr = nullptr;
      int srcIdx = slot.wiring.inputSourceIndices != nullptr
                       ? slot.wiring.inputSourceIndices[j]
                       : -1;

      if (srcIdx >= 0) {
        // Cross-slot reference — use the output from the producing slot
        if (srcIdx < totalOutputSlots && outputSlots[srcIdx] != nullptr) {
          inputArr = outputSlots[srcIdx];
        }
      } else {
        // External input: -(srcIdx + 1) into externalInputs
        int extIdx = -(srcIdx + 1);
        if (extIdx >= 0 && extIdx < numExternalInputs &&
            externalInputs[extIdx] != nullptr) {
          inputArr = externalInputs[extIdx];
        }
      }

      if (inputArr != nullptr) {
        kernelArgs.push_back(inputArr->buffer());
      }
    }

    // Output buffers: each output lives in the flat outputSlots array at
    // slot.wiring.outputSlotIndices[k].
    for (int k = 0; k < slot.wiring.numOutputs; k++) {
      int outIdx = slot.wiring.outputSlotIndices != nullptr
                       ? slot.wiring.outputSlotIndices[k]
                       : -1;
      if (outIdx >= 0 && outIdx < totalOutputSlots &&
          outputSlots[outIdx] != nullptr) {
        kernelArgs.push_back(outputSlots[outIdx]->buffer());
      }
    }
  }

  // Dispatch to NPU
  if (!runtime.dispatchKernel(compiled->npuContext, compiled->kernelHandle,
                               kernelArgs.data(),
                               static_cast<int>(kernelArgs.size()))) {
    DSP_DIAG(EXECUTE, "HexagonGraphBackend::executeSegment: [%d, %d] dispatch failed",
             seg.def.startSlot, seg.def.endSlot);
    return Status::KERNEL_FAILURE;
  }

  // Wait for NPU completion
  if (!runtime.waitForCompletion(compiled->npuContext)) {
    DSP_DIAG(EXECUTE, "HexagonGraphBackend::executeSegment: [%d, %d] "
             "waitForCompletion failed", seg.def.startSlot, seg.def.endSlot);
    return Status::KERNEL_FAILURE;
  }

  DSP_DIAG(EXECUTE, "HexagonGraphBackend::executeSegment: [%d, %d] executed "
           "successfully", seg.def.startSlot, seg.def.endSlot);
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

  for (int j = 0; j < slot.wiring.numInputs; j++) {
    NDArray* inputArr = nullptr;
    int srcIdx = slot.wiring.inputSourceIndices != nullptr
                     ? slot.wiring.inputSourceIndices[j]
                     : -1;

    if (srcIdx >= 0) {
      // Cross-slot reference: prior op's output in the flat outputSlots array
      if (srcIdx < totalOutputSlots && outputSlots[srcIdx] != nullptr) {
        inputArr = outputSlots[srcIdx];
      }
    } else {
      // External input: -(srcIdx + 1) into externalInputs
      int extIdx = -(srcIdx + 1);
      if (extIdx >= 0 && extIdx < numExternalInputs &&
          externalInputs[extIdx] != nullptr) {
        inputArr = externalInputs[extIdx];
      }
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
