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

#ifdef SD_CUDA
#include <cuda_runtime.h>
#endif

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

// ─── Check if all ops in a range are Triton-mappable ────────────────────────

bool TritonGraphBackend::areAllOpsMappable(NativeSlot* slots, int start, int end) {
  for (int i = start; i <= end; i++) {
    if (!TritonIRBuilder::isTritonMappable(slots[i].opName)) {
      return false;
    }
  }
  return true;
}

// ─── Segment fusibility check ───────────────────────────────────────────────

bool TritonGraphBackend::canFuseSegment(NativeSlot* slots, int start, int end) {
  if (!isAvailable()) return false;

  int totalOps = end - start + 1;
  if (totalOps < MIN_MAPPABLE_OPS) return false;

  // ALL ops in segment must be Triton-mappable (not UNSUPPORTED).
  // We now support all categories: element-wise (binary, unary, comparison, logical,
  // ternary, identity, cast), reduction, normalization, and matmul.
  // No size limit here — segments exceeding MAX_COMPILABLE_OPS are automatically
  // split into sub-segments in compileSegment().
  for (int i = start; i <= end; i++) {
    if (!TritonIRBuilder::isTritonMappable(slots[i].opName)) {
      return false;
    }
  }

  return true;
}

// ─── Compilation ────────────────────────────────────────────────────────────

bool TritonGraphBackend::compileSegment(GraphSegment& seg, NativeSlot* slots,
                                        NDArray** externalInputs, int numExternalInputs,
                                        NDArray** outputSlots, int totalOutputSlots,
                                        LongType shapeKey,
                                        int totalSlots,
                                        int* requestedOutputSlotIndices,
                                        int numRequestedOutputs) {
  SegmentCacheKey key{seg.startSlot, seg.endSlot, shapeKey};

  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      // Already compiled for this shape
      lastCompilationAudit_ = it->second.audit;
      totalCacheHits_++;
      return true;
    }
  }

  int segmentOps = seg.endSlot - seg.startSlot + 1;
  CompiledSegment compiledSeg;

  // Use section boundaries for splitting: identify natural boundaries where
  // the op category changes (e.g., element-wise → matmul → element-wise).
  // Each sub-kernel handles one section or a group of compatible sections.
  // This produces correct kernels because each section type needs different
  // grid dimensions, shared memory, and execution patterns.
  auto sections = TritonIRBuilder::identifySections(slots, seg.startSlot, seg.endSlot,
                                                      outputSlots, totalOutputSlots,
                                                      externalInputs, numExternalInputs);

  if (sections.empty()) {
    sd_printf("TritonGraphBackend::compileSegment: no sections found for segment [%d-%d]\n",
              seg.startSlot, seg.endSlot);
    return false;
  }

  sd_printf("TritonGraphBackend: segment [%d-%d] has %d ops, %d sections\n",
            seg.startSlot, seg.endSlot, segmentOps, static_cast<int>(sections.size()));

  // Compile each section as its own sub-kernel.
  // Compatible adjacent sections (all element-wise) can be merged.
  for (int i = 0; i < static_cast<int>(sections.size()); i++) {
    int subStart = sections[i].startSlot;
    int subEnd = sections[i].endSlot;

    // Merge consecutive element-wise-compatible sections into one sub-kernel
    // (they share the same grid/block pattern and can fuse).
    // Cap merged size so the kernel arg count stays within CUDA's 4KB param limit
    // (~500 pointer args). Each op contributes ~1-2 unique buffer args on average.
    while (i + 1 < static_cast<int>(sections.size())) {
      int mergedOps = subEnd - subStart + 1 + (sections[i + 1].endSlot - sections[i + 1].startSlot + 1);
      if (mergedOps > MAX_COMPILABLE_OPS) break;  // Would exceed register/arg limits
      auto nextType = sections[i + 1].type;
      auto curType = sections[i].type;
      bool curMergeable = (curType == KernelSectionType::ELEMENTWISE ||
                           curType == KernelSectionType::IDENTITY ||
                           curType == KernelSectionType::CONSTANT_GENERATION ||
                           curType == KernelSectionType::SHAPE_MANIPULATION ||
                           curType == KernelSectionType::REDUCTION ||
                           curType == KernelSectionType::NORMALIZATION);
      bool nextMergeable = (nextType == KernelSectionType::ELEMENTWISE ||
                            nextType == KernelSectionType::IDENTITY ||
                            nextType == KernelSectionType::CONSTANT_GENERATION ||
                            nextType == KernelSectionType::SHAPE_MANIPULATION ||
                            nextType == KernelSectionType::REDUCTION ||
                            nextType == KernelSectionType::NORMALIZATION);
      if (curMergeable && nextMergeable) {
        subEnd = sections[i + 1].endSlot;
        i++;
      } else {
        break;
      }
    }

    int subOps = subEnd - subStart + 1;
    if (subOps < MIN_MAPPABLE_OPS) {
      for (int s = subStart; s <= subEnd; s++) {
        CompilationAuditEntry entry;
        entry.slotIndex = s;
        entry.opName = slots[s].opName;
        entry.wasCompiled = true;
        compiledSeg.audit.push_back(entry);
      }
      continue;
    }

    sd_printf("TritonGraphBackend: compiling sub-segment %d [%d-%d] (%d ops)\n",
              static_cast<int>(compiledSeg.subKernels.size()) + 1, subStart, subEnd, subOps);

    auto compiled = compileToGpuBinary(slots, subStart, subEnd,
                                        totalSlots,
                                        externalInputs, numExternalInputs,
                                        outputSlots, totalOutputSlots);

    if (!compiled.gpuModule || !compiled.kernelFunction) {
#ifdef SD_CUDA
      cudaGetLastError();
#endif
      sd_printf("TritonGraphBackend: sub-segment [%d-%d] compilation FAILED\n", subStart, subEnd);
      for (auto& prev : compiledSeg.subKernels) {
        if (prev.gpuModule) {
          TritonTargetDispatch::unloadModule(prev.gpuModule);
        }
      }
      return false;
    }

    compiled.startSlot_ = subStart;
    compiled.endSlot_ = subEnd;
    {
      FILE* tf = fopen("/tmp/triton_trace.txt", "a");
      if (tf) {
        fprintf(tf, "COMPILED: sub[%d-%d] %d ops, %d args, indirect=%d, numWarps=%d, blockX=%d\n",
                subStart, subEnd, subOps,
                (int)compiled.argSlotMapping.size(),
                compiled.useIndirectArgs ? 1 : 0,
                compiled.numWarps, compiled.blockX);
        fflush(tf); fclose(tf);
      }
    }
    compiledSeg.audit.insert(compiledSeg.audit.end(),
                              compiled.audit.begin(), compiled.audit.end());
    compiledSeg.subKernels.push_back(std::move(compiled));
  }

  sd_printf("TritonGraphBackend: compiled segment [%d-%d] into %d sub-kernels\n",
            seg.startSlot, seg.endSlot, static_cast<int>(compiledSeg.subKernels.size()));

  lastCompilationAudit_ = compiledSeg.audit;

  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    cache_[key] = std::move(compiledSeg);
  }

  sd_printf("TritonGraphBackend: compiled segment [%d-%d] (%d sub-kernels, shape key %lld)\n",
            seg.startSlot, seg.endSlot, (int)cache_[key].subKernels.size(), shapeKey);
  return true;
}

// ─── Execute a single compiled kernel ───────────────────────────────────────

Status TritonGraphBackend::executeSingleKernel(CompiledKernel& compiled, NativeSlot* slots,
                                                NDArray** externalInputs, int numExternalInputs,
                                                NDArray** outputSlots, int totalOutputSlots,
                                                void* stream) {
  int numBufferArgs = static_cast<int>(compiled.argSlotMapping.size());

  // Resolve all buffer pointers from the arg slot mapping
  std::vector<void*> bufferPtrs;
  bufferPtrs.reserve(numBufferArgs);

  for (auto& argMapping : compiled.argSlotMapping) {
    NDArray* arr = nullptr;
    if (argMapping.slotIndex < 0) {
      int extIdx = -(argMapping.slotIndex + 1);
      if (extIdx < numExternalInputs) arr = externalInputs[extIdx];
    } else {
      if (argMapping.slotIndex < totalOutputSlots) arr = outputSlots[argMapping.slotIndex];
    }

    if (!arr) {
      sd_printf("TritonGraphBackend::executeSingleKernel: null array for arg slot %d "
                "(sub-segment [%d-%d])\n",
                argMapping.slotIndex, compiled.startSlot_, compiled.endSlot_);
      return Status::KERNEL_FAILURE;
    }
    void* sbuf = arr->specialBuffer();
    if (!sbuf) {
      sd_printf("TritonGraphBackend::executeSingleKernel: null specialBuffer for arg slot %d "
                "(sub-segment [%d-%d], length=%lld, dtype=%d)\n",
                argMapping.slotIndex, compiled.startSlot_, compiled.endSlot_,
                (long long)arr->lengthOf(), static_cast<int>(arr->dataType()));
      FILE* tf = fopen("/tmp/triton_trace.txt", "a");
      if (tf) {
        fprintf(tf, "NULL_SBUF: slot %d sub[%d-%d] isOutput=%d\n",
                argMapping.slotIndex, compiled.startSlot_, compiled.endSlot_,
                argMapping.isOutput ? 1 : 0);
        fflush(tf); fclose(tf);
      }
      return Status::KERNEL_FAILURE;
    }
    bufferPtrs.push_back(sbuf);
  }

  // Compute n_elements from first output
  LongType nElements = 0;
  for (auto& argMapping : compiled.argSlotMapping) {
    if (argMapping.isOutput) {
      int slotIdx = argMapping.slotIndex;
      if (slotIdx >= 0 && slotIdx < totalOutputSlots && outputSlots[slotIdx]) {
        nElements = outputSlots[slotIdx]->lengthOf();
        break;
      }
    }
  }
  int nElem32 = static_cast<int>(nElements);

  // Compute grid size
  unsigned int actualGridX = (nElements + compiled.blockX - 1) / compiled.blockX;
  unsigned int actualGridY = compiled.gridY;
  unsigned int actualGridZ = compiled.gridZ;
  if (actualGridX == 0) actualGridX = 1;

  // Dereference the stream pointer
  void* actualStream = (stream != nullptr) ? *static_cast<void**>(stream) : nullptr;

  // Build kernel args — either direct (each ptr is a separate arg) or indirect
  // (all ptrs packed into a device-side i64 array, kernel receives 1 pointer)
  std::vector<void*> kernelArgs;
  void* argTableDevice = nullptr;

  if (compiled.useIndirectArgs) {
    // Pack all buffer pointers as int64 values into a device-side array.
    // The kernel signature is: @kernel(%argTable: !tt.ptr<i64>, %n_elements: i32)
    // It loads each buffer pointer from argTable[i] and casts via tt.int_to_ptr.
    std::vector<int64_t> argTableHost(numBufferArgs);
    for (int i = 0; i < numBufferArgs; i++) {
      argTableHost[i] = reinterpret_cast<int64_t>(bufferPtrs[i]);
    }

#ifdef SD_CUDA
    // Allocate device buffer for the arg table
    size_t tableBytes = numBufferArgs * sizeof(int64_t);
    auto allocErr = cudaMallocAsync(&argTableDevice, tableBytes,
                                     static_cast<cudaStream_t>(actualStream));
    if (allocErr != cudaSuccess) {
      allocErr = cudaMalloc(&argTableDevice, tableBytes);
      if (allocErr != cudaSuccess) {
        sd_printf("TritonGraphBackend: failed to allocate arg table (%d bytes): %s\n",
                  (int)tableBytes, cudaGetErrorString(allocErr));
        return Status::KERNEL_FAILURE;
      }
    }
    // Copy host → device (async on the execution stream)
    cudaMemcpyAsync(argTableDevice, argTableHost.data(), tableBytes,
                     cudaMemcpyHostToDevice, static_cast<cudaStream_t>(actualStream));
#endif

    // Kernel args: [argTablePtr, n_elements]
    kernelArgs.push_back(&argTableDevice);
    kernelArgs.push_back(&nElem32);
  } else {
    // Direct mode: each buffer pointer is a separate kernel arg + n_elements
    // cuLaunchKernel expects void** where each entry points to the actual param value.
    // bufferPtrs[i] IS the void* value; &bufferPtrs[i] is the pointer-to-pointer.
    for (int i = 0; i < numBufferArgs; i++) {
      kernelArgs.push_back(&bufferPtrs[i]);
    }
    kernelArgs.push_back(&nElem32);
  }

  // Diagnostic: dump key info for indirect arg launches
  if (compiled.useIndirectArgs) {
    FILE* tf = fopen("/tmp/triton_trace.txt", "a");
    if (tf) {
      fprintf(tf, "PRE_LAUNCH_INDIRECT: [%d-%d] argTableDevice=%p nElem=%d grid=%ux%ux%u block=%ux%ux%u nBufArgs=%d\n",
              compiled.startSlot_, compiled.endSlot_, argTableDevice, nElem32,
              actualGridX, actualGridY, actualGridZ,
              compiled.numWarps * 32, compiled.blockY, compiled.blockZ, numBufferArgs);
      // Check for null buffer pointers and dump any found
      int nullCount = 0;
      for (int i = 0; i < numBufferArgs; i++) {
        if (!bufferPtrs[i]) {
          fprintf(tf, "  NULL_BUF[%d] slot=%d isOutput=%d\n",
                  i, compiled.argSlotMapping[i].slotIndex,
                  compiled.argSlotMapping[i].isOutput ? 1 : 0);
          nullCount++;
        }
      }
      if (nullCount > 0) {
        fprintf(tf, "  TOTAL_NULL_BUFS: %d of %d\n", nullCount, numBufferArgs);
      }
      // First 3 and last 3 for quick reference
      for (int i = 0; i < numBufferArgs && i < 3; i++) {
        fprintf(tf, "  bufPtr[%d] = %p (slot %d)\n",
                i, bufferPtrs[i], compiled.argSlotMapping[i].slotIndex);
      }
      // Non-blocking stream check
#ifdef SD_CUDA
      auto peekErr = cudaPeekAtLastError();
      if (peekErr != cudaSuccess) {
        fprintf(tf, "  peekErr=%d (%s)\n", (int)peekErr, cudaGetErrorString(peekErr));
      }
      int curDevice = -1;
      cudaGetDevice(&curDevice);
      fprintf(tf, "  curDevice=%d, actualStream=%p\n", curDevice, actualStream);
#endif
      fflush(tf); fclose(tf);
    }
  }

  // Launch
  bool ok;
  if (compiled.useCooperativeLaunch) {
    ok = TritonTargetDispatch::launchCooperativeKernel(
        compiled.kernelFunction,
        actualGridX, actualGridY, actualGridZ,
        compiled.numWarps * 32,
        compiled.blockY, compiled.blockZ,
        compiled.sharedMemBytes,
        actualStream,
        kernelArgs.data(),
        static_cast<int>(kernelArgs.size()));
  } else {
    ok = TritonTargetDispatch::launchKernel(
        compiled.kernelFunction,
        actualGridX, actualGridY, actualGridZ,
        compiled.numWarps * 32,
        compiled.blockY, compiled.blockZ,
        compiled.sharedMemBytes,
        actualStream,
        kernelArgs.data(),
        static_cast<int>(kernelArgs.size()));
  }

  if (!ok) {
    sd_printf("TritonGraphBackend::executeSingleKernel: kernel launch failed for [%d-%d] "
              "(cooperative=%d, grid=%ux%ux%u, block=%ux%ux%u, sharedMem=%u)\n",
              compiled.startSlot_, compiled.endSlot_,
              compiled.useCooperativeLaunch ? 1 : 0,
              actualGridX, actualGridY, actualGridZ,
              compiled.numWarps * 32, compiled.blockY, compiled.blockZ,
              compiled.sharedMemBytes);
    {
      FILE* tf = fopen("/tmp/triton_trace.txt", "a");
      if (tf) {
        fprintf(tf, "LAUNCH_FAIL: [%d-%d] nElem=%lld grid=%ux%ux%u block=%ux%ux%u shMem=%u nArgs=%d coop=%d\n",
                compiled.startSlot_, compiled.endSlot_, (long long)nElements,
                actualGridX, actualGridY, actualGridZ,
                compiled.numWarps * 32, compiled.blockY, compiled.blockZ,
                compiled.sharedMemBytes, (int)kernelArgs.size(), compiled.useCooperativeLaunch ? 1 : 0);
#ifdef SD_CUDA
        auto err = cudaGetLastError();
        fprintf(tf, "  cudaErr=%d (%s)\n", (int)err, cudaGetErrorString(err));
#endif
        fflush(tf); fclose(tf);
      }
    }
#ifdef SD_CUDA
    if (argTableDevice) cudaFreeAsync(argTableDevice, static_cast<cudaStream_t>(actualStream));
#endif
    return Status::KERNEL_FAILURE;
  }

#ifdef SD_CUDA
  // Free the indirect arg table after kernel launch (async — kernel reads it before free executes)
  if (argTableDevice) cudaFreeAsync(argTableDevice, static_cast<cudaStream_t>(actualStream));
#endif

  return Status::OK;
}

// ─── Execution ──────────────────────────────────────────────────────────────

Status TritonGraphBackend::executeSegment(GraphSegment& seg, NativeSlot* slots,
                                          NDArray** externalInputs, int numExternalInputs,
                                          NDArray** outputSlots, int totalOutputSlots,
                                          void* stream) {
  SegmentCacheKey key{seg.startSlot, seg.endSlot, seg.shapeKey};

  CompiledSegment* compiledSeg = nullptr;
  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      sd_printf("TritonGraphBackend::executeSegment: no compiled kernel for segment [%d-%d]\n",
                seg.startSlot, seg.endSlot);
      return Status::KERNEL_FAILURE;
    }
    compiledSeg = &it->second;
  }

  // Execute all sub-kernels in sequence on the same stream.
  {
    FILE* tf = fopen("/tmp/triton_trace.txt", "a");
    if (tf) {
      fprintf(tf, "executeSegment: seg[%d-%d] %d sub-kernels\n",
              seg.startSlot, seg.endSlot, (int)compiledSeg->subKernels.size());
      fflush(tf); fclose(tf);
    }
  }
  for (int i = 0; i < (int)compiledSeg->subKernels.size(); i++) {
    auto& subKernel = compiledSeg->subKernels[i];
    auto status = executeSingleKernel(subKernel, slots,
                                       externalInputs, numExternalInputs,
                                       outputSlots, totalOutputSlots,
                                       stream);
    if (status != Status::OK) {
      sd_printf("TritonGraphBackend::executeSegment: sub-kernel %d/%d [%d-%d] failed\n",
                i + 1, (int)compiledSeg->subKernels.size(),
                subKernel.startSlot_, subKernel.endSlot_);
      FILE* tf = fopen("/tmp/triton_trace.txt", "a");
      if (tf) {
        fprintf(tf, "SUB_KERNEL_FAIL: %d/%d [%d-%d]\n",
                i + 1, (int)compiledSeg->subKernels.size(),
                subKernel.startSlot_, subKernel.endSlot_);
        fflush(tf); fclose(tf);
      }
      return status;
    }
    totalKernelLaunches_++;
  }


  return Status::OK;
}

// ─── Cache invalidation ────────────────────────────────────────────────────

void TritonGraphBackend::invalidateCache() {
  std::lock_guard<std::mutex> lock(cacheMtx_);
  for (auto& entry : cache_) {
    for (auto& kernel : entry.second.subKernels) {
      if (kernel.gpuModule) {
        TritonTargetDispatch::unloadModule(kernel.gpuModule);
      }
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
    int totalSlots,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots) {

  CompiledKernel result;

  // Build Triton IR
  auto irModule = irBuilder_.buildModule(slots, startSlot, endSlot,
                                          totalSlots,
                                          externalInputs, numExternalInputs,
                                          outputSlots, totalOutputSlots);
  if (!irModule.valid) {
#ifdef SD_CUDA
    cudaGetLastError();  // Clear sticky CUDA errors from failed IR build
#endif
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
#ifdef SD_CUDA
    cudaGetLastError();  // Clear sticky CUDA errors from failed compilation
#endif
    sd_printf("TritonGraphBackend: Triton compilation failed for segment [%d-%d]\n", startSlot, endSlot);
    // Clean up MLIR module
    if (irModule.mlirModule) {
      auto* mod = static_cast<mlir::ModuleOp*>(irModule.mlirModule);
      mod->erase();
      delete mod;
    }
    return result;
  }

  // Dump PTX for indirect-args kernels (diagnostic)
  if (irModule.useIndirectArgs && binary.data && binary.size > 0) {
    FILE* pf = fopen("/tmp/triton_ptx_indirect.ptx", "w");
    if (pf) {
      fprintf(pf, "// Indirect-args kernel [%d-%d], %d args\n",
              startSlot, endSlot, static_cast<int>(irModule.args.size()));
      fwrite(binary.data, 1, binary.size, pf);
      fflush(pf); fclose(pf);
    }
  }

  // Load binary into driver module
  result.gpuModule = TritonTargetDispatch::loadModule(binary);
  if (!result.gpuModule) {
#ifdef SD_CUDA
    cudaGetLastError();  // Clear sticky CUDA errors from failed module load
#endif
    sd_printf("TritonGraphBackend: module load failed for segment [%d-%d]\n", startSlot, endSlot);
    delete[] static_cast<char*>(binary.data);
    return result;
  }

  // Get kernel function
  result.kernelFunction = TritonTargetDispatch::getKernelFunction(result.gpuModule, irModule.kernelName);
  if (!result.kernelFunction) {
#ifdef SD_CUDA
    cudaGetLastError();  // Clear sticky CUDA errors
#endif
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
  result.useCooperativeLaunch = irModule.useCooperativeLaunch;
  result.useIndirectArgs = irModule.useIndirectArgs;

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
