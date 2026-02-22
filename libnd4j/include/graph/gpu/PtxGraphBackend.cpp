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

#include <graph/gpu/PtxGraphBackend.h>
#include <graph/gpu/GpuKernelLauncher.h>
#include <ops/declarable/helpers/fusedElementwiseChain.h>
#include <system/common.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <sstream>
#include <unordered_set>

namespace sd {
namespace graph {

// ---- Singleton ----

PtxGraphBackend& PtxGraphBackend::getInstance() {
  static PtxGraphBackend instance;
  return instance;
}

PtxGraphBackend::PtxGraphBackend() = default;

PtxGraphBackend::~PtxGraphBackend() {
  invalidateCache();
}

// ---- Availability ----

bool PtxGraphBackend::isAvailable() const {
  return true;  // Always available on CUDA builds
}

// ---- SM version ----

int PtxGraphBackend::getSmVersion() {
  cudaDeviceProp props;
  int device = 0;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&props, device);
  return props.major * 10 + props.minor;
}

// ---- Segment fusibility ----

bool PtxGraphBackend::canFuseSegment(NativeSlot* slots, int start, int end) {
  int fusible = 0;
  for (int i = start; i <= end; i++) {
    int code = sd::ops::helpers::opNameToFusedCode(slots[i].opName);
    if (code >= 0) {
      fusible++;
    }
  }
  return fusible >= MIN_FUSIBLE_OPS;
}

// ---- PTX instruction generation helpers ----

// Register counter for PTX float register allocation
struct PtxRegAlloc {
  int nextFloat = 0;
  int nextPred = 0;

  std::string allocFloat() { return "f" + std::to_string(nextFloat++); }
  std::string allocPred() { return "p" + std::to_string(nextPred++); }
};

/**
 * Emit PTX instructions for a single fused op.
 * @param out     Output stream for PTX instructions
 * @param ra      Register allocator
 * @param opCode  FusedElemOp code
 * @param inReg   Input float register name
 * @param secReg  Secondary input float register (for binary ops)
 * @return        Result float register name
 */
static std::string emitPtxOp(std::ostringstream& out, PtxRegAlloc& ra,
                               int opCode, const std::string& inReg,
                               const std::string& secReg) {
  using namespace sd::ops::helpers;
  std::string result = ra.allocFloat();

  switch (opCode) {
    case FUSED_ADD:
      out << "    add.f32 " << result << ", " << inReg << ", " << secReg << ";\n";
      break;
    case FUSED_SUB:
      out << "    sub.f32 " << result << ", " << inReg << ", " << secReg << ";\n";
      break;
    case FUSED_MUL:
      out << "    mul.f32 " << result << ", " << inReg << ", " << secReg << ";\n";
      break;
    case FUSED_DIV:
      out << "    div.rn.f32 " << result << ", " << inReg << ", " << secReg << ";\n";
      break;
    case FUSED_RELU:
      // max(x, 0.0f)
      out << "    max.f32 " << result << ", " << inReg << ", 0f00000000;\n";
      break;
    case FUSED_ABS:
      out << "    abs.f32 " << result << ", " << inReg << ";\n";
      break;
    case FUSED_NEG:
      out << "    neg.f32 " << result << ", " << inReg << ";\n";
      break;
    case FUSED_SQUARE: {
      out << "    mul.f32 " << result << ", " << inReg << ", " << inReg << ";\n";
      break;
    }
    case FUSED_SQRT:
      out << "    sqrt.approx.f32 " << result << ", " << inReg << ";\n";
      break;
    case FUSED_EXP: {
      // exp(x) = ex2(x * log2(e))
      // log2(e) = 1.4426950408889634 = 0x3FB8AA3B in float
      std::string t = ra.allocFloat();
      out << "    mul.f32 " << t << ", " << inReg << ", 0f3FB8AA3B;\n";
      out << "    ex2.approx.f32 " << result << ", " << t << ";\n";
      break;
    }
    case FUSED_LOG: {
      // log(x) = lg2(x) * ln(2)
      // ln(2) = 0.6931471805599453 = 0x3F317218 in float
      std::string t = ra.allocFloat();
      out << "    lg2.approx.f32 " << t << ", " << inReg << ";\n";
      out << "    mul.f32 " << result << ", " << t << ", 0f3F317218;\n";
      break;
    }
    case FUSED_SIGMOID: {
      // sigmoid(x) = 1 / (1 + exp(-x))
      // = 1 / (1 + ex2(-x * log2(e)))
      std::string neg = ra.allocFloat();
      std::string scaled = ra.allocFloat();
      std::string e2 = ra.allocFloat();
      std::string sum = ra.allocFloat();
      out << "    neg.f32 " << neg << ", " << inReg << ";\n";
      out << "    mul.f32 " << scaled << ", " << neg << ", 0f3FB8AA3B;\n";  // * log2(e)
      out << "    ex2.approx.f32 " << e2 << ", " << scaled << ";\n";
      out << "    add.f32 " << sum << ", " << e2 << ", 0f3F800000;\n";  // + 1.0f
      out << "    rcp.approx.f32 " << result << ", " << sum << ";\n";
      break;
    }
    case FUSED_TANH: {
      // tanh(x) = 2*sigmoid(2x) - 1
      // Use the PTX tanh.approx.f32 if available (sm_75+), otherwise compound
      out << "    // tanh via compound: 2*sigmoid(2x)-1\n";
      std::string x2 = ra.allocFloat();
      std::string neg2x = ra.allocFloat();
      std::string scaled = ra.allocFloat();
      std::string e2 = ra.allocFloat();
      std::string sum = ra.allocFloat();
      std::string sig = ra.allocFloat();
      std::string sig2 = ra.allocFloat();
      out << "    add.f32 " << x2 << ", " << inReg << ", " << inReg << ";\n";  // 2x
      out << "    neg.f32 " << neg2x << ", " << x2 << ";\n";
      out << "    mul.f32 " << scaled << ", " << neg2x << ", 0f3FB8AA3B;\n";
      out << "    ex2.approx.f32 " << e2 << ", " << scaled << ";\n";
      out << "    add.f32 " << sum << ", " << e2 << ", 0f3F800000;\n";
      out << "    rcp.approx.f32 " << sig << ", " << sum << ";\n";
      out << "    add.f32 " << sig2 << ", " << sig << ", " << sig << ";\n";
      out << "    sub.f32 " << result << ", " << sig2 << ", 0f3F800000;\n";  // -1.0
      break;
    }
    case FUSED_SWISH:
    case FUSED_SILU: {
      // swish(x) = x * sigmoid(x)
      std::string neg = ra.allocFloat();
      std::string scaled = ra.allocFloat();
      std::string e2 = ra.allocFloat();
      std::string sum = ra.allocFloat();
      std::string sig = ra.allocFloat();
      out << "    neg.f32 " << neg << ", " << inReg << ";\n";
      out << "    mul.f32 " << scaled << ", " << neg << ", 0f3FB8AA3B;\n";
      out << "    ex2.approx.f32 " << e2 << ", " << scaled << ";\n";
      out << "    add.f32 " << sum << ", " << e2 << ", 0f3F800000;\n";
      out << "    rcp.approx.f32 " << sig << ", " << sum << ";\n";
      out << "    mul.f32 " << result << ", " << inReg << ", " << sig << ";\n";
      break;
    }
    case FUSED_MISH: {
      // mish(x) = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
      // Approximation: compute softplus, then tanh, then multiply
      std::string xScaled = ra.allocFloat();
      std::string expX = ra.allocFloat();
      std::string sp = ra.allocFloat();
      std::string spLog = ra.allocFloat();
      // softplus: log(1 + exp(x))
      out << "    mul.f32 " << xScaled << ", " << inReg << ", 0f3FB8AA3B;\n";
      out << "    ex2.approx.f32 " << expX << ", " << xScaled << ";\n";
      out << "    add.f32 " << sp << ", " << expX << ", 0f3F800000;\n";
      out << "    lg2.approx.f32 " << spLog << ", " << sp << ";\n";
      std::string softplus = ra.allocFloat();
      out << "    mul.f32 " << softplus << ", " << spLog << ", 0f3F317218;\n";  // * ln(2)
      // tanh(softplus) via 2*sigmoid(2*softplus)-1
      std::string sp2 = ra.allocFloat();
      std::string negsp2 = ra.allocFloat();
      std::string sc2 = ra.allocFloat();
      std::string e2b = ra.allocFloat();
      std::string sm = ra.allocFloat();
      std::string sg = ra.allocFloat();
      std::string sg2 = ra.allocFloat();
      std::string th = ra.allocFloat();
      out << "    add.f32 " << sp2 << ", " << softplus << ", " << softplus << ";\n";
      out << "    neg.f32 " << negsp2 << ", " << sp2 << ";\n";
      out << "    mul.f32 " << sc2 << ", " << negsp2 << ", 0f3FB8AA3B;\n";
      out << "    ex2.approx.f32 " << e2b << ", " << sc2 << ";\n";
      out << "    add.f32 " << sm << ", " << e2b << ", 0f3F800000;\n";
      out << "    rcp.approx.f32 " << sg << ", " << sm << ";\n";
      out << "    add.f32 " << sg2 << ", " << sg << ", " << sg << ";\n";
      out << "    sub.f32 " << th << ", " << sg2 << ", 0f3F800000;\n";
      out << "    mul.f32 " << result << ", " << inReg << ", " << th << ";\n";
      break;
    }
    case FUSED_GELU: {
      // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
      // Approx: x * sigmoid(1.702 * x) (fast GELU approximation)
      // 1.702 = 0x3FD9999A
      std::string scaled = ra.allocFloat();
      std::string neg = ra.allocFloat();
      std::string negScaled = ra.allocFloat();
      std::string e2 = ra.allocFloat();
      std::string sum = ra.allocFloat();
      std::string sig = ra.allocFloat();
      out << "    mul.f32 " << scaled << ", " << inReg << ", 0f3FD9999A;\n";  // 1.702 * x
      out << "    neg.f32 " << neg << ", " << scaled << ";\n";
      out << "    mul.f32 " << negScaled << ", " << neg << ", 0f3FB8AA3B;\n";
      out << "    ex2.approx.f32 " << e2 << ", " << negScaled << ";\n";
      out << "    add.f32 " << sum << ", " << e2 << ", 0f3F800000;\n";
      out << "    rcp.approx.f32 " << sig << ", " << sum << ";\n";
      out << "    mul.f32 " << result << ", " << inReg << ", " << sig << ";\n";
      break;
    }
    default:
      // Identity fallback
      out << "    mov.f32 " << result << ", " << inReg << "; // unsupported op " << opCode << "\n";
      break;
  }

  return result;
}

// ---- PTX generation ----

std::string PtxGraphBackend::generatePtx(
    NativeSlot* slots, int startSlot, int endSlot,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    CompiledKernel& result) {

  int smVersion = getSmVersion();

  // Collect external inputs and outputs (same logic as NVRTC backend)
  std::unordered_map<int, int> externalInputMap;  // extIdx -> paramIdx
  std::vector<int> outputSlotIndices;
  int paramIdx = 0;

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

  auto& lastSlot = slots[endSlot];
  for (int o = 0; o < lastSlot.numOutputs; o++) {
    outputSlotIndices.push_back(lastSlot.outputSlotIndices[o]);
  }

  int numInputParams = static_cast<int>(externalInputMap.size());
  int numOutputParams = static_cast<int>(outputSlotIndices.size());

  // Build arg mappings
  for (auto& [extIdx, pIdx] : externalInputMap) {
    CompiledKernel::ArgMapping am;
    am.slotIndex = -(extIdx + 1);
    am.isOutput = false;
    result.argMap.push_back(am);
  }
  for (int outSlotIdx : outputSlotIndices) {
    CompiledKernel::ArgMapping am;
    am.slotIndex = outSlotIdx;
    am.isOutput = true;
    result.argMap.push_back(am);
  }

  // Count max float registers needed (estimate generously)
  int maxFloatRegs = 64;  // generous
  int maxPredRegs = 8;

  std::ostringstream ptx;
  ptx << ".version 7.0\n";
  ptx << ".target sm_" << smVersion << "\n";
  ptx << ".address_size 64\n\n";

  // Kernel entry
  ptx << ".visible .entry ptx_fused_kernel(\n";
  for (int i = 0; i < numInputParams; i++) {
    ptx << "    .param .u64 in" << i << ",\n";
  }
  for (int i = 0; i < numOutputParams; i++) {
    ptx << "    .param .u64 out" << i << ",\n";
  }
  ptx << "    .param .s32 n\n";
  ptx << ") {\n";

  // Register declarations
  ptx << "    .reg .pred p<" << maxPredRegs << ">;\n";
  ptx << "    .reg .f32 f<" << maxFloatRegs << ">;\n";
  ptx << "    .reg .b32 r<16>;\n";
  ptx << "    .reg .b64 rd<" << (numInputParams + numOutputParams + 8) << ">;\n\n";

  // Thread index computation: idx = blockIdx.x * blockDim.x + threadIdx.x
  ptx << "    mov.u32 r0, %ctaid.x;\n";
  ptx << "    mov.u32 r1, %ntid.x;\n";
  ptx << "    mov.u32 r2, %tid.x;\n";
  ptx << "    mad.lo.s32 r3, r0, r1, r2;\n";

  // Bounds check
  ptx << "    ld.param.s32 r4, [n];\n";
  ptx << "    setp.ge.s32 p0, r3, r4;\n";
  ptx << "    @p0 bra EXIT;\n\n";

  // Compute byte offset: r3 * 4 (sizeof float)
  ptx << "    cvt.u64.u32 rd0, r3;\n";
  ptx << "    shl.b64 rd0, rd0, 2;\n\n";

  // Load external inputs
  PtxRegAlloc ra;
  // Reserve a few registers for the loads
  std::unordered_map<int, std::string> extRegMap;  // extParamIdx -> float register

  for (int i = 0; i < numInputParams; i++) {
    int rdBase = i + 1;
    std::string fReg = ra.allocFloat();
    ptx << "    ld.param.u64 rd" << rdBase << ", [in" << i << "];\n";
    ptx << "    add.u64 rd" << rdBase << ", rd" << rdBase << ", rd0;\n";
    ptx << "    ld.global.f32 " << fReg << ", [rd" << rdBase << "];\n";
    extRegMap[i] = fReg;
  }
  ptx << "\n";

  // Walk slots and emit ops
  std::unordered_map<int, std::string> slotOutputRegs;  // outputSlotIdx -> float register

  for (int si = startSlot; si <= endSlot; si++) {
    auto& slot = slots[si];
    int fusedCode = sd::ops::helpers::opNameToFusedCode(slot.opName);

    // Resolve first input register
    std::string inReg;
    if (slot.numInputs > 0) {
      int srcIdx = slot.inputSourceIndices[0];
      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        inReg = extRegMap[externalInputMap[extIdx]];
      } else {
        auto it = slotOutputRegs.find(srcIdx);
        if (it != slotOutputRegs.end()) {
          inReg = it->second;
        } else {
          // Shouldn't happen; use zero register
          inReg = ra.allocFloat();
          ptx << "    mov.f32 " << inReg << ", 0f00000000; // missing input\n";
        }
      }
    }

    // Resolve second input for binary ops
    std::string secReg;
    if (slot.numInputs > 1 && fusedCode >= 0 &&
        sd::ops::helpers::isBinaryFusedOp(static_cast<sd::ops::helpers::FusedElemOp>(fusedCode))) {
      int srcIdx = slot.inputSourceIndices[1];
      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        secReg = extRegMap[externalInputMap[extIdx]];
      } else {
        auto it = slotOutputRegs.find(srcIdx);
        if (it != slotOutputRegs.end()) {
          secReg = it->second;
        } else {
          secReg = "0f00000000";
        }
      }
    }

    ptx << "    // slot " << si << ": " << slot.opName << "\n";

    std::string resultReg;
    if (fusedCode >= 0) {
      resultReg = emitPtxOp(ptx, ra, fusedCode, inReg, secReg);
    } else {
      // Identity pass-through
      resultReg = ra.allocFloat();
      ptx << "    mov.f32 " << resultReg << ", " << inReg << "; // unsupported: " << slot.opName << "\n";
    }

    for (int o = 0; o < slot.numOutputs; o++) {
      slotOutputRegs[slot.outputSlotIndices[o]] = resultReg;
    }
  }

  // Store outputs
  ptx << "\n";
  for (int i = 0; i < numOutputParams; i++) {
    int rdIdx = numInputParams + i + 1;
    int outSlotIdx = outputSlotIndices[i];
    auto it = slotOutputRegs.find(outSlotIdx);
    if (it != slotOutputRegs.end()) {
      ptx << "    ld.param.u64 rd" << rdIdx << ", [out" << i << "];\n";
      ptx << "    add.u64 rd" << rdIdx << ", rd" << rdIdx << ", rd0;\n";
      ptx << "    st.global.f32 [rd" << rdIdx << "], " << it->second << ";\n";
    }
  }

  ptx << "\nEXIT:\n";
  ptx << "    ret;\n";
  ptx << "}\n";

  return ptx.str();
}

// ---- Compilation ----

bool PtxGraphBackend::compileSegment(GraphSegment& seg, NativeSlot* slots,
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

  std::string ptxSrc = generatePtx(slots, seg.startSlot, seg.endSlot,
                                     externalInputs, numExternalInputs,
                                     outputSlots, totalOutputSlots, compiled);

  if (ptxSrc.empty()) {
    sd_printf("PtxGraphBackend: PTX generation failed for segment [%d-%d]\n",
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

  // Load PTX directly (JIT compilation by CUDA driver)
  compiled.gpuModule = GpuKernelLauncher::loadPtxModule(ptxSrc.c_str(), ptxSrc.size());
  if (!compiled.gpuModule) {
    sd_printf("PtxGraphBackend: PTX module load failed for segment [%d-%d]\n",
              seg.startSlot, seg.endSlot);
    return false;
  }

  compiled.kernelFunction = GpuKernelLauncher::getKernelFunc(compiled.gpuModule, "ptx_fused_kernel");
  if (!compiled.kernelFunction) {
    sd_printf("PtxGraphBackend: kernel function not found in module\n", "");
    GpuKernelLauncher::unloadModule(compiled.gpuModule);
    compiled.gpuModule = nullptr;
    return false;
  }

  lastCompilationAudit_ = compiled.audit;

  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    cache_[key] = std::move(compiled);
  }

  sd_printf("PtxGraphBackend: loaded segment [%d-%d] (%zu bytes PTX, shape key %lld)\n",
            seg.startSlot, seg.endSlot, ptxSrc.size(), shapeKey);
  return true;
}

// ---- Execution ----

Status PtxGraphBackend::executeSegment(GraphSegment& seg, NativeSlot* slots,
                                        NDArray** externalInputs, int numExternalInputs,
                                        NDArray** outputSlots, int totalOutputSlots,
                                        void* stream) {
  SegmentCacheKey key{seg.startSlot, seg.endSlot, seg.shapeKey};

  CompiledKernel* compiled = nullptr;
  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      sd_printf("PtxGraphBackend::executeSegment: no compiled kernel for segment [%d-%d]\n",
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
      sd_printf("PtxGraphBackend::executeSegment: null array for arg slot %d\n", am.slotIndex);
      return Status::KERNEL_FAILURE;
    }

    kernelArgs.push_back(arr->specialBuffer());

    if (am.isOutput && nElements == 0) {
      nElements = arr->lengthOf();
    }
  }

  int nElem32 = static_cast<int>(nElements);
  kernelArgs.push_back(&nElem32);

  unsigned int blockSize = 256;
  unsigned int gridSize = (static_cast<unsigned int>(nElements) + blockSize - 1) / blockSize;
  if (gridSize == 0) gridSize = 1;

  void* actualStream = (stream != nullptr) ? *static_cast<void**>(stream) : nullptr;

  bool ok = GpuKernelLauncher::launchKernel(
      compiled->kernelFunction,
      gridSize, 1, 1,
      blockSize, 1, 1,
      0, actualStream,
      kernelArgs.data(),
      static_cast<int>(kernelArgs.size()));

  if (!ok) {
    sd_printf("PtxGraphBackend::executeSegment: kernel launch failed for segment [%d-%d]\n",
              seg.startSlot, seg.endSlot);
    return Status::KERNEL_FAILURE;
  }

  return Status::OK;
}

// ---- Cache invalidation ----

void PtxGraphBackend::invalidateCache() {
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

std::vector<CompilationAuditEntry> PtxGraphBackend::getLastCompilationAudit() const {
  return lastCompilationAudit_;
}

}  // namespace graph
}  // namespace sd

#endif  // SD_CUDA
