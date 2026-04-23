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


#include <graph/gpu/NvrtcGraphBackend.h>
#include <graph/DspDiagnostics.h>
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
  // Use sm_XX (not compute_XX) so NVRTC generates CUBIN directly.
  // This avoids PTX version mismatch between toolkit and driver
  // (e.g., NVRTC 12.9 emits PTX 8.8 but driver 570.x only JITs PTX ≤8.7).
  return "sm_" + std::to_string(props.major * 10 + props.minor);
}

// ---- Segment fusibility (delegates to shared logic) ----

bool NvrtcGraphBackend::canFuseSegment(NativeSlot* slots, int start, int end) {
  return jitCanFuseSegment(slots, start, end);
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
    return "(" + val + " * (1.0f / (1.0f + expf(-1.702f * " + val + "))))";
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
    if (slot.args.numTArgs > 0) {
      alpha = std::to_string(static_cast<float>(slot.args.tArgs[0])) + "f";
    }
    return "(" + val + " >= 0.0f ? " + val + " : " + val + " * " + alpha + ")";
  }
  // Scalar ops: second operand from tArgs[0]
  if (opName == "add_scalar") {
    std::string scalar = std::to_string(static_cast<float>(slot.args.tArgs[0])) + "f";
    return "(" + val + " + " + scalar + ")";
  }
  if (opName == "subtract_scalar") {
    std::string scalar = std::to_string(static_cast<float>(slot.args.tArgs[0])) + "f";
    return "(" + val + " - " + scalar + ")";
  }
  if (opName == "multiply_scalar") {
    std::string scalar = std::to_string(static_cast<float>(slot.args.tArgs[0])) + "f";
    return "(" + val + " * " + scalar + ")";
  }
  if (opName == "divide_scalar") {
    std::string scalar = std::to_string(static_cast<float>(slot.args.tArgs[0])) + "f";
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
    JitCompiledKernel& result) {

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
    for (int inp = 0; inp < slot.wiring.numInputs; inp++) {
      int srcIdx = slot.wiring.inputSourceIndices[inp];
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
    for (int o = 0; o < slots[si].wiring.numOutputs; o++) {
      segmentSlotOutputs.insert(slots[si].wiring.outputSlotIndices[o]);
    }
  }

  // The final slot's outputs are always segment outputs
  auto& lastSlot = slots[endSlot];
  for (int o = 0; o < lastSlot.wiring.numOutputs; o++) {
    outputSlotIndices.push_back(lastSlot.wiring.outputSlotIndices[o]);
  }

  // Build kernel signature
  std::ostringstream src;
  src << "extern \"C\" __global__ void nvrtc_fused_kernel(\n";

  // Input parameters
  for (auto& [extIdx, pIdx] : externalInputMap) {
    src << "    const float* __restrict__ in" << pIdx << ",\n";
    JitCompiledKernel::ArgMapping am;
    am.slotIndex = -(extIdx + 1);  // negative = external
    am.isOutput = false;
    result.argMap.push_back(am);
  }

  // Output parameters
  for (int i = 0; i < static_cast<int>(outputSlotIndices.size()); i++) {
    src << "    float* __restrict__ out" << i << ",\n";
    JitCompiledKernel::ArgMapping am;
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
    auto cat = getOpCategoryFromName(slot.ident.opName);
    int inputCount = categoryInputCount(cat);

    // Helper to resolve an input source index to a variable name
    auto resolveInput = [&](int inputIdx) -> std::string {
      if (inputIdx < slot.wiring.numInputs) {
        int srcIdx = slot.wiring.inputSourceIndices[inputIdx];
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
    std::string inputVar = (slot.wiring.numInputs > 0) ? resolveInput(0) : "0.0f";
    std::string secVar = (inputCount >= 2 && slot.wiring.numInputs > 1) ? resolveInput(1) : "0.0f";
    std::string terVar = (inputCount >= 3 && slot.wiring.numInputs > 2) ? resolveInput(2) : "0.0f";

    // Emit the op
    std::string resultVar = "t" + std::to_string(si);
    if (isNvrtcJittable(cat)) {
      src << "    float " << resultVar << " = " << opToCudaExpr(cat, slot.ident.opName, inputVar, secVar, terVar, slot) << ";\n";
    } else {
      // Unsupported op: pass through input (identity)
      src << "    float " << resultVar << " = " << inputVar << ";  // unsupported: " << slot.ident.opName << "\n";
    }

    // Map this slot's output indices to the variable
    for (int o = 0; o < slot.wiring.numOutputs; o++) {
      slotOutputVars[slot.wiring.outputSlotIndices[o]] = resultVar;
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
  JitSegmentCacheKey key{seg.def.startSlot, seg.def.endSlot, shapeKey};

  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      lastCompilationAudit_ = it->second.audit;
      return true;
    }
  }

  JitCompiledKernel compiled;

  // Generate CUDA C source
  std::string cudaSrc = generateCudaSource(slots, seg.def.startSlot, seg.def.endSlot,
                                            externalInputs, numExternalInputs,
                                            outputSlots, totalOutputSlots, compiled);

  if (cudaSrc.empty()) {
    DSP_DIAG(COMPILE, "NvrtcGraphBackend: CUDA source generation failed for segment [%d-%d]",
             seg.def.startSlot, seg.def.endSlot);
    return false;
  }

  // Build audit
  for (int i = seg.def.startSlot; i <= seg.def.endSlot; i++) {
    CompilationAuditEntry entry;
    entry.slotIndex = i;
    entry.opName = slots[i].ident.opName;
    entry.wasCompiled = isNvrtcJittable(getOpCategoryFromName(slots[i].ident.opName));
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
    DSP_DIAG(COMPILE, "NvrtcGraphBackend: nvrtcCreateProgram failed: %s", nvrtcGetErrorString(nvRes));
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
      DSP_DIAG(COMPILE, "NvrtcGraphBackend: compilation failed for segment [%d-%d]:\n%s",
               seg.def.startSlot, seg.def.endSlot, log.c_str());
    }
    nvrtcDestroyProgram(&prog);
    return false;
  }

  // Get CUBIN (native code, no driver JIT needed — avoids PTX version mismatch)
  size_t cubinSize = 0;
  nvrtcGetCUBINSize(prog, &cubinSize);
  if (cubinSize == 0) {
    // Fallback: if CUBIN not available (shouldn't happen with sm_XX target),
    // try PTX path
    DSP_DIAG(COMPILE, "NvrtcGraphBackend: no CUBIN output, falling back to PTX for segment [%d-%d]",
             seg.def.startSlot, seg.def.endSlot);
    size_t ptxSize = 0;
    nvrtcGetPTXSize(prog, &ptxSize);
    std::string ptx(ptxSize, '\0');
    nvrtcGetPTX(prog, &ptx[0]);
    nvrtcDestroyProgram(&prog);
    compiled.gpuModule = GpuKernelLauncher::loadPtxModule(ptx.c_str(), ptx.size());
  } else {
    std::vector<char> cubin(cubinSize);
    nvrtcGetCUBIN(prog, cubin.data());
    nvrtcDestroyProgram(&prog);
    // Load CUBIN via same cuModuleLoadDataEx (handles both PTX and CUBIN)
    compiled.gpuModule = GpuKernelLauncher::loadPtxModule(cubin.data(), cubinSize);
  }
  if (!compiled.gpuModule) {
    DSP_DIAG(COMPILE, "NvrtcGraphBackend: module load failed for segment [%d-%d]",
             seg.def.startSlot, seg.def.endSlot);
    return false;
  }

  // Get kernel function
  compiled.kernelFunction = GpuKernelLauncher::getKernelFunc(compiled.gpuModule,
                                                              "nvrtc_fused_kernel");
  if (!compiled.kernelFunction) {
    DSP_DIAG(COMPILE, "NvrtcGraphBackend: kernel function not found in module");
    GpuKernelLauncher::unloadModule(compiled.gpuModule);
    compiled.gpuModule = nullptr;
    return false;
  }

  lastCompilationAudit_ = compiled.audit;

  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    cache_[key] = std::move(compiled);
  }

  DSP_DIAG(JIT, "NvrtcGraphBackend: compiled segment [%d-%d] (shape key %lld)",
            seg.def.startSlot, seg.def.endSlot, shapeKey);
  return true;
}

// ---- Execution (delegates to shared logic) ----

Status NvrtcGraphBackend::executeSegment(GraphSegment& seg, NativeSlot* slots,
                                          NDArray** externalInputs, int numExternalInputs,
                                          NDArray** outputSlots, int totalOutputSlots,
                                          void* stream) {
  JitSegmentCacheKey key{seg.def.startSlot, seg.def.endSlot, seg.def.shapeKeyState.compiledShapeKey};
  return jitExecuteSegment(key, cache_, cacheMtx_, "NvrtcGraphBackend",
                           slots, externalInputs, numExternalInputs,
                           outputSlots, totalOutputSlots, stream);
}

// ---- Cache invalidation (delegates to shared logic) ----

void NvrtcGraphBackend::invalidateCache() {
  jitInvalidateCache(cache_, cacheMtx_, lastCompilationAudit_);
}

// ---- Audit ----

std::vector<CompilationAuditEntry> NvrtcGraphBackend::getLastCompilationAudit() const {
  return lastCompilationAudit_;
}

}  // namespace graph
}  // namespace sd

