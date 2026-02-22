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

#ifdef SD_CUDA

#include <graph/gpu/NvrtcGraphBackend.h>
#include <graph/gpu/GpuKernelLauncher.h>
#include <ops/declarable/helpers/fusedElementwiseChain.h>
#include <system/common.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <sstream>
#include <unordered_set>

namespace sd {
namespace graph {

// ---- Singleton ----

NvrtcGraphBackend& NvrtcGraphBackend::getInstance() {
  static NvrtcGraphBackend instance;
  return instance;
}

NvrtcGraphBackend::NvrtcGraphBackend() = default;

NvrtcGraphBackend::~NvrtcGraphBackend() {
  invalidateCache();
}

// ---- Availability ----

bool NvrtcGraphBackend::isAvailable() const {
  // NVRTC is always available on CUDA builds (ships with toolkit)
  return true;
}

// ---- Compute arch ----

std::string NvrtcGraphBackend::getComputeArch() {
  cudaDeviceProp props;
  int device = 0;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&props, device);
  return "compute_" + std::to_string(props.major * 10 + props.minor);
}

// ---- Segment fusibility ----

bool NvrtcGraphBackend::canFuseSegment(NativeSlot* slots, int start, int end) {
  int fusible = 0;
  for (int i = start; i <= end; i++) {
    int code = sd::ops::helpers::opNameToFusedCode(slots[i].opName);
    if (code >= 0) {
      fusible++;
    }
  }
  return fusible >= MIN_FUSIBLE_OPS;
}

// ---- CUDA C expression generation ----

std::string NvrtcGraphBackend::opToCudaExpr(int fusedOpCode, const std::string& in,
                                             const std::string& sec) {
  using namespace sd::ops::helpers;
  switch (fusedOpCode) {
    case FUSED_ADD:     return "(" + in + " + " + sec + ")";
    case FUSED_SUB:     return "(" + in + " - " + sec + ")";
    case FUSED_MUL:     return "(" + in + " * " + sec + ")";
    case FUSED_DIV:     return "(" + in + " / " + sec + ")";
    case FUSED_RELU:    return "(" + in + " > 0.0f ? " + in + " : 0.0f)";
    case FUSED_SIGMOID: return "(1.0f / (1.0f + expf(-" + in + ")))";
    case FUSED_TANH:    return "tanhf(" + in + ")";
    case FUSED_GELU:    return "(" + in + " * 0.5f * (1.0f + erff(" + in + " * 0.7071067811865476f)))";
    case FUSED_EXP:     return "expf(" + in + ")";
    case FUSED_LOG:     return "logf(" + in + ")";
    case FUSED_ABS:     return "fabsf(" + in + ")";
    case FUSED_NEG:     return "(-" + in + ")";
    case FUSED_SQUARE:  return "(" + in + " * " + in + ")";
    case FUSED_SQRT:    return "sqrtf(" + in + ")";
    case FUSED_SWISH:   return "(" + in + " / (1.0f + expf(-" + in + ")))";
    case FUSED_SILU:    return "(" + in + " / (1.0f + expf(-" + in + ")))";
    case FUSED_MISH:    return "(" + in + " * tanhf(logf(1.0f + expf(" + in + "))))";
    default:            return in;  // identity fallback
  }
}

// ---- CUDA C source generation ----

std::string NvrtcGraphBackend::generateCudaSource(
    NativeSlot* slots, int startSlot, int endSlot,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    CompiledKernel& result) {

  // Walk the slots, identify external inputs, intermediate SSA values, and outputs.
  //
  // Strategy: for each slot in [startSlot, endSlot], resolve its inputs:
  //   - If inputSourceIndex < 0: external input array at -(idx+1)
  //   - If inputSourceIndex >= 0: output of a prior slot (intermediate)
  //
  // Each slot's output is stored as an SSA variable (t_slotIdx).
  // External inputs and final outputs become kernel parameters.

  // Collect unique external input indices and output slot indices
  std::unordered_map<int, int> externalInputMap;  // extIdx -> paramIdx
  std::vector<int> outputSlotIndices;
  int paramIdx = 0;

  // First pass: identify all external inputs across all slots
  for (int si = startSlot; si <= endSlot; si++) {
    auto& slot = slots[si];
    for (int inp = 0; inp < slot.numInputs; inp++) {
      int srcIdx = slot.inputSourceIndices[inp];
      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        if (externalInputMap.find(extIdx) == externalInputMap.end()) {
          externalInputMap[extIdx] = paramIdx++;
        }
      }
    }
  }

  // Identify output slots (last slot's outputs, or any slot whose output is
  // consumed outside this segment)
  std::unordered_set<int> segmentSlotOutputs;
  for (int si = startSlot; si <= endSlot; si++) {
    for (int o = 0; o < slots[si].numOutputs; o++) {
      segmentSlotOutputs.insert(slots[si].outputSlotIndices[o]);
    }
  }

  // The final slot's outputs are always segment outputs
  auto& lastSlot = slots[endSlot];
  for (int o = 0; o < lastSlot.numOutputs; o++) {
    outputSlotIndices.push_back(lastSlot.outputSlotIndices[o]);
  }

  // Build kernel signature
  std::ostringstream src;
  src << "extern \"C\" __global__ void nvrtc_fused_kernel(\n";

  // Input parameters
  for (auto& [extIdx, pIdx] : externalInputMap) {
    src << "    const float* __restrict__ in" << pIdx << ",\n";
    CompiledKernel::ArgMapping am;
    am.slotIndex = -(extIdx + 1);  // negative = external
    am.isOutput = false;
    result.argMap.push_back(am);
  }

  // Output parameters
  for (int i = 0; i < static_cast<int>(outputSlotIndices.size()); i++) {
    src << "    float* __restrict__ out" << i << ",\n";
    CompiledKernel::ArgMapping am;
    am.slotIndex = outputSlotIndices[i];
    am.isOutput = true;
    result.argMap.push_back(am);
  }

  // Element count
  src << "    const int n\n";
  src << ") {\n";
  src << "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
  src << "    if (idx >= n) return;\n\n";

  // Load external inputs
  for (auto& [extIdx, pIdx] : externalInputMap) {
    src << "    float ext" << pIdx << " = in" << pIdx << "[idx];\n";
  }
  src << "\n";

  // Walk slots and emit fused ops
  std::unordered_map<int, std::string> slotOutputVars;  // outputSlotIdx -> variable name

  for (int si = startSlot; si <= endSlot; si++) {
    auto& slot = slots[si];
    int fusedCode = sd::ops::helpers::opNameToFusedCode(slot.opName);

    // Resolve first input
    std::string inputVar;
    if (slot.numInputs > 0) {
      int srcIdx = slot.inputSourceIndices[0];
      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        inputVar = "ext" + std::to_string(externalInputMap[extIdx]);
      } else {
        // Prior slot output
        auto it = slotOutputVars.find(srcIdx);
        if (it != slotOutputVars.end()) {
          inputVar = it->second;
        } else {
          inputVar = "0.0f";  // should not happen
        }
      }
    }

    // Resolve second input (for binary ops)
    std::string secVar = "0.0f";
    if (slot.numInputs > 1 && sd::ops::helpers::isBinaryFusedOp(static_cast<sd::ops::helpers::FusedElemOp>(fusedCode))) {
      int srcIdx = slot.inputSourceIndices[1];
      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        secVar = "ext" + std::to_string(externalInputMap[extIdx]);
      } else {
        auto it = slotOutputVars.find(srcIdx);
        if (it != slotOutputVars.end()) {
          secVar = it->second;
        }
      }
    }

    // Emit the op
    std::string resultVar = "t" + std::to_string(si);
    if (fusedCode >= 0) {
      src << "    float " << resultVar << " = " << opToCudaExpr(fusedCode, inputVar, secVar) << ";\n";
    } else {
      // Unsupported op: pass through input (identity)
      src << "    float " << resultVar << " = " << inputVar << ";  // unsupported: " << slot.opName << "\n";
    }

    // Map this slot's output indices to the variable
    for (int o = 0; o < slot.numOutputs; o++) {
      slotOutputVars[slot.outputSlotIndices[o]] = resultVar;
    }
  }

  // Store outputs
  src << "\n";
  for (int i = 0; i < static_cast<int>(outputSlotIndices.size()); i++) {
    int outIdx = outputSlotIndices[i];
    auto it = slotOutputVars.find(outIdx);
    if (it != slotOutputVars.end()) {
      src << "    out" << i << "[idx] = " << it->second << ";\n";
    }
  }

  src << "}\n";

  result.numArgs = static_cast<int>(result.argMap.size()) + 1;  // +1 for n
  return src.str();
}

// ---- Compilation ----

bool NvrtcGraphBackend::compileSegment(GraphSegment& seg, NativeSlot* slots,
                                        NDArray** externalInputs, int numExternalInputs,
                                        NDArray** outputSlots, int totalOutputSlots,
                                        LongType shapeKey) {
  SegmentCacheKey key{seg.startSlot, seg.endSlot, shapeKey};

  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      lastCompilationAudit_ = it->second.audit;
      return true;
    }
  }

  CompiledKernel compiled;

  // Generate CUDA C source
  std::string cudaSrc = generateCudaSource(slots, seg.startSlot, seg.endSlot,
                                            externalInputs, numExternalInputs,
                                            outputSlots, totalOutputSlots, compiled);

  if (cudaSrc.empty()) {
    sd_printf("NvrtcGraphBackend: CUDA source generation failed for segment [%d-%d]\n",
              seg.startSlot, seg.endSlot);
    return false;
  }

  // Build audit
  for (int i = seg.startSlot; i <= seg.endSlot; i++) {
    CompilationAuditEntry entry;
    entry.slotIndex = i;
    entry.opName = slots[i].opName;
    entry.wasCompiled = (sd::ops::helpers::opNameToFusedCode(slots[i].opName) >= 0);
    if (!entry.wasCompiled) {
      entry.reason = "unmappable op (not in fused elementwise table)";
    }
    compiled.audit.push_back(entry);
  }

  // Compile with NVRTC
  nvrtcProgram prog;
  nvrtcResult nvRes = nvrtcCreateProgram(&prog, cudaSrc.c_str(),
                                          "nvrtc_fused_kernel.cu",
                                          0, nullptr, nullptr);
  if (nvRes != NVRTC_SUCCESS) {
    sd_printf("NvrtcGraphBackend: nvrtcCreateProgram failed: %s\n", nvrtcGetErrorString(nvRes));
    return false;
  }

  std::string arch = "--gpu-architecture=" + getComputeArch();
  const char* opts[] = {arch.c_str()};

  nvRes = nvrtcCompileProgram(prog, 1, opts);
  if (nvRes != NVRTC_SUCCESS) {
    // Get compilation log
    size_t logSize = 0;
    nvrtcGetProgramLogSize(prog, &logSize);
    if (logSize > 1) {
      std::string log(logSize, '\0');
      nvrtcGetProgramLog(prog, &log[0]);
      sd_printf("NvrtcGraphBackend: compilation failed for segment [%d-%d]:\n%s\n",
                seg.startSlot, seg.endSlot, log.c_str());
    }
    nvrtcDestroyProgram(&prog);
    return false;
  }

  // Get PTX
  size_t ptxSize = 0;
  nvrtcGetPTXSize(prog, &ptxSize);
  std::string ptx(ptxSize, '\0');
  nvrtcGetPTX(prog, &ptx[0]);
  nvrtcDestroyProgram(&prog);

  // Load PTX module
  compiled.gpuModule = GpuKernelLauncher::loadPtxModule(ptx.c_str(), ptx.size());
  if (!compiled.gpuModule) {
    sd_printf("NvrtcGraphBackend: PTX module load failed for segment [%d-%d]\n",
              seg.startSlot, seg.endSlot);
    return false;
  }

  // Get kernel function
  compiled.kernelFunction = GpuKernelLauncher::getKernelFunc(compiled.gpuModule,
                                                              "nvrtc_fused_kernel");
  if (!compiled.kernelFunction) {
    sd_printf("NvrtcGraphBackend: kernel function not found in module\n", "");
    GpuKernelLauncher::unloadModule(compiled.gpuModule);
    compiled.gpuModule = nullptr;
    return false;
  }

  lastCompilationAudit_ = compiled.audit;

  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    cache_[key] = std::move(compiled);
  }

  sd_printf("NvrtcGraphBackend: compiled segment [%d-%d] (%zu bytes PTX, shape key %lld)\n",
            seg.startSlot, seg.endSlot, ptxSize, shapeKey);
  return true;
}

// ---- Execution ----

Status NvrtcGraphBackend::executeSegment(GraphSegment& seg, NativeSlot* slots,
                                          NDArray** externalInputs, int numExternalInputs,
                                          NDArray** outputSlots, int totalOutputSlots,
                                          void* stream) {
  SegmentCacheKey key{seg.startSlot, seg.endSlot, seg.shapeKey};

  CompiledKernel* compiled = nullptr;
  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      sd_printf("NvrtcGraphBackend::executeSegment: no compiled kernel for segment [%d-%d]\n",
                seg.startSlot, seg.endSlot);
      return Status::KERNEL_FAILURE;
    }
    compiled = &it->second;
  }

  // Build kernel arguments
  std::vector<void*> kernelArgs;
  kernelArgs.reserve(compiled->argMap.size() + 1);

  LongType nElements = 0;

  for (auto& am : compiled->argMap) {
    NDArray* arr = nullptr;
    if (am.slotIndex < 0) {
      // External input: slotIndex is -(extIdx+1), so extIdx = -(slotIndex+1)
      int extIdx = -(am.slotIndex + 1);
      if (extIdx < numExternalInputs) {
        arr = externalInputs[extIdx];
      }
    } else {
      if (am.slotIndex < totalOutputSlots) {
        arr = outputSlots[am.slotIndex];
      }
    }

    if (!arr) {
      sd_printf("NvrtcGraphBackend::executeSegment: null array for arg slot %d\n", am.slotIndex);
      return Status::KERNEL_FAILURE;
    }

    kernelArgs.push_back(arr->specialBuffer());

    if (am.isOutput && nElements == 0) {
      nElements = arr->lengthOf();
    }
  }

  int nElem32 = static_cast<int>(nElements);
  kernelArgs.push_back(&nElem32);

  // Launch config
  unsigned int blockSize = 256;
  unsigned int gridSize = (static_cast<unsigned int>(nElements) + blockSize - 1) / blockSize;
  if (gridSize == 0) gridSize = 1;

  // Dereference stream pointer (NativeDynamicShapePlan passes void* to cudaStream_t)
  void* actualStream = (stream != nullptr) ? *static_cast<void**>(stream) : nullptr;

  bool ok = GpuKernelLauncher::launchKernel(
      compiled->kernelFunction,
      gridSize, 1, 1,
      blockSize, 1, 1,
      0, actualStream,
      kernelArgs.data(),
      static_cast<int>(kernelArgs.size()));

  if (!ok) {
    sd_printf("NvrtcGraphBackend::executeSegment: kernel launch failed for segment [%d-%d]\n",
              seg.startSlot, seg.endSlot);
    return Status::KERNEL_FAILURE;
  }

  return Status::OK;
}

// ---- Cache invalidation ----

void NvrtcGraphBackend::invalidateCache() {
  std::lock_guard<std::mutex> lock(cacheMtx_);
  for (auto& entry : cache_) {
    if (entry.second.gpuModule) {
      GpuKernelLauncher::unloadModule(entry.second.gpuModule);
    }
  }
  cache_.clear();
  lastCompilationAudit_.clear();
}

// ---- Audit ----

std::vector<CompilationAuditEntry> NvrtcGraphBackend::getLastCompilationAudit() const {
  return lastCompilationAudit_;
}

}  // namespace graph
}  // namespace sd

#endif  // SD_CUDA
