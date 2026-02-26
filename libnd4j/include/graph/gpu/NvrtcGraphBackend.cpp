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
#include <graph/gpu/OpCategoryTable.h>
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
    auto cat = getOpCategoryFromName(slots[i].opName);
    if (isNvrtcJittable(cat)) {
      fusible++;
    }
  }
  return fusible >= MIN_FUSIBLE_OPS;
}

// ---- CUDA C expression generation (category-based dispatch) ----

static std::string generateBinaryExpr(const std::string& opName,
                                       const std::string& a, const std::string& b) {
  if (opName == "add" || opName == "Add") return "(" + a + " + " + b + ")";
  if (opName == "subtract" || opName == "Sub") return "(" + a + " - " + b + ")";
  if (opName == "multiply" || opName == "Mul") return "(" + a + " * " + b + ")";
  if (opName == "divide" || opName == "Div" || opName == "RealDiv")
    return "(" + b + " != 0.0f ? " + a + " / " + b + " : 0.0f)";
  if (opName == "minimum" || opName == "Min" || opName == "min_pairwise" || opName == "MinPairwise")
    return "fminf(" + a + ", " + b + ")";
  if (opName == "maximum" || opName == "Max" || opName == "max_pairwise" || opName == "MaxPairwise")
    return "fmaxf(" + a + ", " + b + ")";
  if (opName == "mod" || opName == "Mod" || opName == "floormod" || opName == "FloorMod")
    return "fmodf(" + a + ", " + b + ")";
  if (opName == "atan2" || opName == "Atan2")
    return "atan2f(" + a + ", " + b + ")";
  if (opName == "floordiv" || opName == "FloorDiv")
    return "floorf(" + a + " / " + b + ")";
  if (opName == "reversedivide" || opName == "ReverseDivide")
    return "(" + a + " != 0.0f ? " + b + " / " + a + " : 0.0f)";
  if (opName == "reversesubtract" || opName == "ReverseSubtract")
    return "(" + b + " - " + a + ")";
  if (opName == "squaredsubtract" || opName == "SquaredSubtract")
    return "((" + a + " - " + b + ") * (" + a + " - " + b + "))";
  if (opName == "multiply_no_nan" || opName == "MultiplyNoNan")
    return "(" + b + " != 0.0f ? " + a + " * " + b + " : 0.0f)";
  if (opName == "pow" || opName == "Pow")
    return "powf(" + a + ", " + b + ")";
  if (opName == "swish_mul" || opName == "SwishMul")
    return "(" + a + " / (1.0f + expf(-" + a + ")) * " + b + ")";
  // Fallback
  return "(" + a + " + " + b + ")";
}

static std::string generateUnaryExpr(const std::string& opName, const std::string& val,
                                      const NativeSlot& slot) {
  if (opName == "relu" || opName == "Relu")
    return "(" + val + " > 0.0f ? " + val + " : 0.0f)";
  if (opName == "sigmoid" || opName == "Sigmoid")
    return "(1.0f / (1.0f + expf(-" + val + ")))";
  if (opName == "tanh" || opName == "Tanh")
    return "tanhf(" + val + ")";
  if (opName == "gelu" || opName == "Gelu")
    return "(0.5f * " + val + " * (1.0f + tanhf(0.7978845608f * ("
           + val + " + 0.044715f * " + val + " * " + val + " * " + val + "))))";
  if (opName == "exp" || opName == "Exp")
    return "expf(" + val + ")";
  if (opName == "log" || opName == "Log")
    return "(" + val + " > 0.0f ? logf(" + val + ") : -1e38f)";
  if (opName == "abs" || opName == "Abs")
    return "fabsf(" + val + ")";
  if (opName == "neg" || opName == "Neg")
    return "(-" + val + ")";
  if (opName == "square" || opName == "Square")
    return "(" + val + " * " + val + ")";
  if (opName == "sqrt" || opName == "Sqrt")
    return "(" + val + " >= 0.0f ? sqrtf(" + val + ") : 0.0f)";
  if (opName == "swish" || opName == "Swish" || opName == "silu" || opName == "Silu")
    return "(" + val + " / (1.0f + expf(-" + val + ")))";
  if (opName == "mish" || opName == "Mish")
    return "(" + val + " * tanhf(logf(1.0f + expf(" + val + "))))";
  if (opName == "rsqrt" || opName == "Rsqrt")
    return "(1.0f / sqrtf(" + val + "))";
  if (opName == "reciprocal" || opName == "Reciprocal")
    return "(1.0f / " + val + ")";
  if (opName == "sign" || opName == "Sign")
    return "(" + val + " > 0.0f ? 1.0f : (" + val + " < 0.0f ? -1.0f : 0.0f))";
  if (opName == "erf" || opName == "Erf")
    return "erff(" + val + ")";
  if (opName == "erfc" || opName == "Erfc")
    return "erfcf(" + val + ")";
  if (opName == "log1p" || opName == "Log1p")
    return "log1pf(" + val + ")";
  if (opName == "ceil" || opName == "Ceil")
    return "ceilf(" + val + ")";
  if (opName == "floor" || opName == "Floor")
    return "floorf(" + val + ")";
  if (opName == "round" || opName == "Round")
    return "rintf(" + val + ")";
  if (opName == "sin" || opName == "Sin")
    return "sinf(" + val + ")";
  if (opName == "cos" || opName == "Cos")
    return "cosf(" + val + ")";
  if (opName == "elu" || opName == "Elu")
    return "(" + val + " >= 0.0f ? " + val + " : (expf(" + val + ") - 1.0f))";
  if (opName == "selu" || opName == "Selu")
    return "(" + val + " >= 0.0f ? 1.0507009873554805f * " + val
           + " : 1.0507009873554805f * 1.6732632423543772f * (expf(" + val + ") - 1.0f))";
  if (opName == "softplus" || opName == "Softplus")
    return "logf(1.0f + expf(" + val + "))";
  if (opName == "softsign" || opName == "Softsign")
    return "(" + val + " / (1.0f + fabsf(" + val + ")))";
  if (opName == "hard_sigmoid" || opName == "HardSigmoid")
    return "fminf(1.0f, fmaxf(0.0f, 0.2f * " + val + " + 0.5f))";
  if (opName == "hardtanh" || opName == "HardTanh")
    return "fminf(1.0f, fmaxf(-1.0f, " + val + "))";
  if (opName == "relu6" || opName == "Relu6")
    return "fminf(6.0f, fmaxf(0.0f, " + val + "))";
  if (opName == "leakyrelu" || opName == "LeakyRelu") {
    std::string alpha = "0.01f";
    if (slot.numTArgs > 0) {
      alpha = std::to_string(static_cast<float>(slot.tArgs[0])) + "f";
    }
    return "(" + val + " >= 0.0f ? " + val + " : " + val + " * " + alpha + ")";
  }
  // Scalar ops: second operand from tArgs[0]
  if (opName == "add_scalar") {
    std::string scalar = std::to_string(static_cast<float>(slot.tArgs[0])) + "f";
    return "(" + val + " + " + scalar + ")";
  }
  if (opName == "subtract_scalar") {
    std::string scalar = std::to_string(static_cast<float>(slot.tArgs[0])) + "f";
    return "(" + val + " - " + scalar + ")";
  }
  if (opName == "multiply_scalar") {
    std::string scalar = std::to_string(static_cast<float>(slot.tArgs[0])) + "f";
    return "(" + val + " * " + scalar + ")";
  }
  if (opName == "divide_scalar") {
    std::string scalar = std::to_string(static_cast<float>(slot.tArgs[0])) + "f";
    return "(" + val + " / " + scalar + ")";
  }
  // Fallback: identity
  return val;
}

static std::string generateComparisonExpr(const std::string& opName,
                                           const std::string& a, const std::string& b) {
  if (opName == "greater" || opName == "Greater")
    return "(" + a + " > " + b + " ? 1.0f : 0.0f)";
  if (opName == "greater_equal" || opName == "GreaterEqual")
    return "(" + a + " >= " + b + " ? 1.0f : 0.0f)";
  if (opName == "less" || opName == "Less")
    return "(" + a + " < " + b + " ? 1.0f : 0.0f)";
  if (opName == "less_equal" || opName == "LessEqual")
    return "(" + a + " <= " + b + " ? 1.0f : 0.0f)";
  if (opName == "equals" || opName == "Equals")
    return "(" + a + " == " + b + " ? 1.0f : 0.0f)";
  if (opName == "not_equals" || opName == "NotEquals")
    return "(" + a + " != " + b + " ? 1.0f : 0.0f)";
  // Fallback
  return "(" + a + " > " + b + " ? 1.0f : 0.0f)";
}

static std::string generateLogicalExpr(const std::string& opName,
                                        const std::string& a, const std::string& b) {
  if (opName == "boolean_and" || opName == "BooleanAnd" ||
      opName == "logical_and" || opName == "LogicalAnd")
    return "((" + a + " != 0.0f && " + b + " != 0.0f) ? 1.0f : 0.0f)";
  if (opName == "boolean_or" || opName == "BooleanOr" ||
      opName == "logical_or" || opName == "LogicalOr")
    return "((" + a + " != 0.0f || " + b + " != 0.0f) ? 1.0f : 0.0f)";
  if (opName == "boolean_not" || opName == "BooleanNot" ||
      opName == "logical_not" || opName == "LogicalNot")
    return "(" + a + " == 0.0f ? 1.0f : 0.0f)";
  if (opName == "boolean_xor" || opName == "BooleanXor")
    return "(((" + a + " != 0.0f) != (" + b + " != 0.0f)) ? 1.0f : 0.0f)";
  // Fallback
  return "((" + a + " != 0.0f && " + b + " != 0.0f) ? 1.0f : 0.0f)";
}

static std::string generateTernaryExpr(const std::string& opName,
                                        const std::string& cond,
                                        const std::string& trueVal,
                                        const std::string& falseVal) {
  return "(" + cond + " != 0.0f ? " + trueVal + " : " + falseVal + ")";
}

/**
 * Dispatch to the appropriate expression generator based on op category.
 */
static std::string opToCudaExpr(TritonOpCategory cat, const std::string& opName,
                                 const std::string& primary,
                                 const std::string& secondary,
                                 const std::string& tertiary,
                                 const NativeSlot& slot) {
  switch (cat) {
    case TritonOpCategory::BINARY_ELEMENTWISE:
      return generateBinaryExpr(opName, primary, secondary);
    case TritonOpCategory::UNARY_ELEMENTWISE:
      return generateUnaryExpr(opName, primary, slot);
    case TritonOpCategory::COMPARISON:
      return generateComparisonExpr(opName, primary, secondary);
    case TritonOpCategory::LOGICAL:
      return generateLogicalExpr(opName, primary, secondary);
    case TritonOpCategory::TERNARY:
      return generateTernaryExpr(opName, primary, secondary, tertiary);
    case TritonOpCategory::CAST:
    case TritonOpCategory::IDENTITY:
      return primary;  // pass-through
    default:
      return primary;  // identity fallback
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
    auto cat = getOpCategoryFromName(slot.opName);
    int inputCount = categoryInputCount(cat);

    // Helper to resolve an input source index to a variable name
    auto resolveInput = [&](int inputIdx) -> std::string {
      if (inputIdx < slot.numInputs) {
        int srcIdx = slot.inputSourceIndices[inputIdx];
        if (srcIdx < 0) {
          int extIdx = -(srcIdx + 1);
          return "ext" + std::to_string(externalInputMap[extIdx]);
        } else {
          auto it = slotOutputVars.find(srcIdx);
          if (it != slotOutputVars.end()) {
            return it->second;
          }
        }
      }
      return "0.0f";
    };

    // Resolve inputs based on category input count
    std::string inputVar = (slot.numInputs > 0) ? resolveInput(0) : "0.0f";
    std::string secVar = (inputCount >= 2 && slot.numInputs > 1) ? resolveInput(1) : "0.0f";
    std::string terVar = (inputCount >= 3 && slot.numInputs > 2) ? resolveInput(2) : "0.0f";

    // Emit the op
    std::string resultVar = "t" + std::to_string(si);
    if (isNvrtcJittable(cat)) {
      src << "    float " << resultVar << " = " << opToCudaExpr(cat, slot.opName, inputVar, secVar, terVar, slot) << ";\n";
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
                                        LongType shapeKey,
                                        int totalSlots,
                                        int* requestedOutputSlotIndices,
                                        int numRequestedOutputs) {
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
    entry.wasCompiled = isNvrtcJittable(getOpCategoryFromName(slots[i].opName));
    if (!entry.wasCompiled) {
      entry.reason = "unmappable op (not in OpCategoryTable or not NVRTC-jittable)";
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
