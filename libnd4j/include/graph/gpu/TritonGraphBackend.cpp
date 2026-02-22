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

#if HAVE_TRITON

#include <graph/gpu/TritonGraphBackend.h>
#include <helpers/logger.h>
#include <system/common.h>

// MLIR core for ModuleOp used in compileToGpuBinary cleanup
#include <mlir/IR/BuiltinOps.h>

namespace sd {
namespace graph {

// ─── Singleton ──────────────────────────────────────────────────────────────

TritonGraphBackend& TritonGraphBackend::getInstance() {
  static TritonGraphBackend instance;
  return instance;
}

TritonGraphBackend::TritonGraphBackend() = default;

TritonGraphBackend::~TritonGraphBackend() {
  invalidateCache();
}

// ─── Availability ───────────────────────────────────────────────────────────

bool TritonGraphBackend::isAvailable() const {
  return TritonTargetDispatch::isReady();
}

// ─── Segment fusibility check ───────────────────────────────────────────────

bool TritonGraphBackend::canFuseSegment(NativeSlot* slots, int start, int end) {
  if (!isAvailable()) return false;

  int totalOps = end - start + 1;
  if (totalOps < MIN_MAPPABLE_OPS) return false;

  // ALL ops in segment must be Triton-mappable (not UNSUPPORTED).
  // We now support all categories: element-wise (binary, unary, comparison, logical,
  // ternary, identity, cast), reduction, normalization, and matmul.
  auto segmentPattern = TritonIRBuilder::classifySegment(slots, start, end);

  for (int i = start; i <= end; i++) {
    if (!TritonIRBuilder::isTritonMappable(slots[i].opName)) {
      return false;
    }
  }

  // For now, only element-wise compatible segments are fully implemented.
  // Reduction, normalization, and matmul patterns have placeholder implementations
  // that will be completed in later phases.
  if (segmentPattern == SegmentKernelPattern::ELEMENTWISE_1D) {
    return true;
  }

  // Reduction/normalization/matmul segments: accept if all ops are mappable.
  // The IR builder will handle dispatch to the appropriate kernel pattern.
  return true;
}

// ─── Compilation ────────────────────────────────────────────────────────────

bool TritonGraphBackend::compileSegment(GraphSegment& seg, NativeSlot* slots,
                                        NDArray** externalInputs, int numExternalInputs,
                                        NDArray** outputSlots, int totalOutputSlots,
                                        LongType shapeKey) {
  SegmentCacheKey key{seg.startSlot, seg.endSlot, shapeKey};

  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      // Already compiled for this shape
      lastCompilationAudit_ = it->second.audit;
      return true;
    }
  }

  // Compile the segment
  auto compiled = compileToGpuBinary(slots, seg.startSlot, seg.endSlot,
                                      externalInputs, numExternalInputs,
                                      outputSlots, totalOutputSlots);

  if (!compiled.gpuModule || !compiled.kernelFunction) {
    sd_printf("TritonGraphBackend::compileSegment: compilation failed for segment [%d-%d]\n",
              seg.startSlot, seg.endSlot);
    return false;
  }

  lastCompilationAudit_ = compiled.audit;

  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    cache_[key] = std::move(compiled);
  }

  sd_printf("TritonGraphBackend: compiled segment [%d-%d] (shape key %lld)\n",
            seg.startSlot, seg.endSlot, shapeKey);
  return true;
}

// ─── Execution ──────────────────────────────────────────────────────────────

Status TritonGraphBackend::executeSegment(GraphSegment& seg, NativeSlot* slots,
                                          NDArray** externalInputs, int numExternalInputs,
                                          NDArray** outputSlots, int totalOutputSlots,
                                          void* stream) {
  SegmentCacheKey key{seg.startSlot, seg.endSlot, seg.shapeKey};

  CompiledKernel* compiled = nullptr;
  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      sd_printf("TritonGraphBackend::executeSegment: no compiled kernel for segment [%d-%d]\n",
                seg.startSlot, seg.endSlot);
      return Status::KERNEL_FAILURE;
    }
    compiled = &it->second;
  }

  // Build kernel arguments from the arg slot mapping
  // Each argument is a pointer to the GPU buffer (specialBuffer)
  std::vector<void*> kernelArgs;
  kernelArgs.reserve(compiled->argSlotMapping.size() + 1);

  for (auto& argMapping : compiled->argSlotMapping) {
    NDArray* arr = nullptr;
    if (argMapping.slotIndex < 0) {
      // External input
      int extIdx = -(argMapping.slotIndex + 1);
      if (extIdx < numExternalInputs) {
        arr = externalInputs[extIdx];
      }
    } else {
      // Output slot
      if (argMapping.slotIndex < totalOutputSlots) {
        arr = outputSlots[argMapping.slotIndex];
      }
    }

    if (!arr) {
      sd_printf("TritonGraphBackend::executeSegment: null array for arg slot %d\n",
                argMapping.slotIndex);
      return Status::KERNEL_FAILURE;
    }

    // Pass GPU buffer pointer
    kernelArgs.push_back(arr->specialBuffer());
  }

  // Add n_elements argument
  // For element-wise ops, this is the total number of elements in the output
  LongType nElements = 0;
  for (auto& argMapping : compiled->argSlotMapping) {
    if (argMapping.isOutput) {
      int slotIdx = argMapping.slotIndex;
      if (slotIdx >= 0 && slotIdx < totalOutputSlots && outputSlots[slotIdx]) {
        nElements = outputSlots[slotIdx]->lengthOf();
        break;
      }
    }
  }
  int nElem32 = static_cast<int>(nElements);
  kernelArgs.push_back(&nElem32);

  // Compute actual grid size based on segment pattern and n_elements
  auto segmentPattern = TritonIRBuilder::classifySegment(slots, seg.startSlot, seg.endSlot);
  unsigned int actualGridX = (nElements + compiled->blockX - 1) / compiled->blockX;
  unsigned int actualGridY = compiled->gridY;
  unsigned int actualGridZ = compiled->gridZ;
  if (actualGridX == 0) actualGridX = 1;

  // For matmul patterns, grid is 2D based on M/N dimensions
  // For reduction/normalization, grid is based on outer dimensions
  // These will be refined as Phase 2-4 implementations mature

  // Dereference the stream pointer.
  // NativeDynamicShapePlan passes void* pointing to a cudaStream_t/hipStream_t
  // (which are themselves pointer types), so we dereference to get the actual handle.
  void* actualStream = (stream != nullptr) ? *static_cast<void**>(stream) : nullptr;

  // Launch the compiled kernel
  bool ok = TritonTargetDispatch::launchKernel(
      compiled->kernelFunction,
      actualGridX, actualGridY, actualGridZ,
      compiled->blockX * 32,  // threads = BLOCK_SIZE (blockX is tile size, each warp is 32 threads)
      compiled->blockY, compiled->blockZ,
      compiled->sharedMemBytes,
      actualStream,
      kernelArgs.data(),
      static_cast<int>(kernelArgs.size()));

  if (!ok) {
    sd_printf("TritonGraphBackend::executeSegment: kernel launch failed for segment [%d-%d]\n",
              seg.startSlot, seg.endSlot);
    return Status::KERNEL_FAILURE;
  }

  return Status::OK;
}

// ─── Cache invalidation ────────────────────────────────────────────────────

void TritonGraphBackend::invalidateCache() {
  std::lock_guard<std::mutex> lock(cacheMtx_);
  for (auto& entry : cache_) {
    if (entry.second.gpuModule) {
      TritonTargetDispatch::unloadModule(entry.second.gpuModule);
    }
  }
  cache_.clear();
  lastCompilationAudit_.clear();
}

// ─── Compilation audit ──────────────────────────────────────────────────────

std::vector<CompilationAuditEntry> TritonGraphBackend::getLastCompilationAudit() const {
  return lastCompilationAudit_;
}

// ─── Internal: compile to GPU binary ────────────────────────────────────────

TritonGraphBackend::CompiledKernel TritonGraphBackend::compileToGpuBinary(
    NativeSlot* slots, int startSlot, int endSlot,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots) {

  CompiledKernel result;

  // Build Triton IR
  auto irModule = irBuilder_.buildModule(slots, startSlot, endSlot,
                                          externalInputs, numExternalInputs,
                                          outputSlots, totalOutputSlots);
  if (!irModule.valid) {
    sd_printf("TritonGraphBackend: IR build failed for segment [%d-%d]\n", startSlot, endSlot);
    return result;
  }

  // Build compilation audit
  for (int i = startSlot; i <= endSlot; i++) {
    CompilationAuditEntry entry;
    entry.slotIndex = i;
    entry.opName = slots[i].opName;
    entry.wasCompiled = TritonIRBuilder::isTritonMappable(slots[i].opName);
    if (!entry.wasCompiled) {
      entry.reason = "unmappable op (not in Triton op table)";
    }
    result.audit.push_back(entry);
  }

  // Compile MLIR -> GPU binary
  auto binary = TritonTargetDispatch::compile(irModule.mlirModule, irModule.numWarps, irModule.numStages);
  if (!binary.data) {
    sd_printf("TritonGraphBackend: Triton compilation failed for segment [%d-%d]\n", startSlot, endSlot);
    // Clean up MLIR module
    if (irModule.mlirModule) {
      auto* mod = static_cast<mlir::ModuleOp*>(irModule.mlirModule);
      mod->erase();
      delete mod;
    }
    return result;
  }

  // Load binary into driver module
  result.gpuModule = TritonTargetDispatch::loadModule(binary);
  if (!result.gpuModule) {
    sd_printf("TritonGraphBackend: module load failed for segment [%d-%d]\n", startSlot, endSlot);
    delete[] static_cast<char*>(binary.data);
    return result;
  }

  // Get kernel function
  result.kernelFunction = TritonTargetDispatch::getKernelFunction(result.gpuModule, irModule.kernelName);
  if (!result.kernelFunction) {
    sd_printf("TritonGraphBackend: kernel function '%s' not found in module\n", irModule.kernelName.c_str());
    TritonTargetDispatch::unloadModule(result.gpuModule);
    result.gpuModule = nullptr;
    delete[] static_cast<char*>(binary.data);
    return result;
  }

  // Set launch config
  result.gridX = irModule.gridX;
  result.gridY = irModule.gridY;
  result.gridZ = irModule.gridZ;
  result.blockX = irModule.blockX;
  result.blockY = irModule.blockY;
  result.blockZ = irModule.blockZ;
  result.sharedMemBytes = binary.sharedMemBytes;
  result.numWarps = binary.numWarps;
  result.argSlotMapping = irModule.args;

  // Clean up
  delete[] static_cast<char*>(binary.data);
  if (irModule.mlirModule) {
    auto* mod = static_cast<mlir::ModuleOp*>(irModule.mlirModule);
    mod->erase();
    delete mod;
  }

  return result;
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_TRITON
