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

#include <graph/gpu/NvrtcKernelBuilder.h>
#include <ops/declarable/helpers/fusedElementwiseChain.h>
#include <system/common.h>

#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace sd {
namespace graph {

using sd::ops::helpers::opNameToFusedCode;
using sd::ops::helpers::FusedElemOp;
using sd::ops::helpers::isBinaryFusedOp;

// ── canJitSegment ──────────────────────────────────────────────────────────

bool canJitSegment(const NativeSlot* slots, int startSlot, int endSlot) {
  if (startSlot > endSlot) return false;

  DataType commonDtype = DataType::UNKNOWN;
  bool hasNonSkippedSlot = false;

  for (int i = startSlot; i <= endSlot; i++) {
    const auto& slot = slots[i];

    // Skip frozen constants and identity ops — their outputs are pre-set
    if (slot.frozenConstantSlot || slot.isIdentityOp) continue;
    // Skip fused chain tails — head dispatches the entire chain
    if (slot.isFusedChainTail) continue;

    // Check if this op is element-wise fusible
    int fusedCode = opNameToFusedCode(slot.opName);
    if (fusedCode < 0) {
      // Not a fusible op — can't JIT this segment
      return false;
    }

    hasNonSkippedSlot = true;
  }

  // Need at least one actual op to JIT
  return hasNonSkippedSlot;
}

// ── Helper: generate the CUDA expression for a single fused op ──────────

static std::string generateOpExpression(int fusedCode, const std::string& valVar,
                                        const std::string& secVar) {
  switch (fusedCode) {
    case FusedElemOp::FUSED_ADD:
      return valVar + " + " + secVar;
    case FusedElemOp::FUSED_SUB:
      return valVar + " - " + secVar;
    case FusedElemOp::FUSED_MUL:
      return valVar + " * " + secVar;
    case FusedElemOp::FUSED_DIV:
      return "(" + secVar + " != 0.0f ? " + valVar + " / " + secVar + " : 0.0f)";
    case FusedElemOp::FUSED_RELU:
      return "(" + valVar + " > 0.0f ? " + valVar + " : 0.0f)";
    case FusedElemOp::FUSED_SIGMOID:
      return "(1.0f / (1.0f + __expf(-" + valVar + ")))";
    case FusedElemOp::FUSED_TANH:
      return "tanhf(" + valVar + ")";
    case FusedElemOp::FUSED_GELU: {
      // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
      return "(0.5f * " + valVar + " * (1.0f + tanhf(0.7978845608f * ("
             + valVar + " + 0.044715f * " + valVar + " * " + valVar + " * " + valVar + "))))";
    }
    case FusedElemOp::FUSED_EXP:
      return "__expf(" + valVar + ")";
    case FusedElemOp::FUSED_LOG:
      return "(" + valVar + " > 0.0f ? __logf(" + valVar + ") : -1e38f)";
    case FusedElemOp::FUSED_ABS:
      return "fabsf(" + valVar + ")";
    case FusedElemOp::FUSED_NEG:
      return "(-" + valVar + ")";
    case FusedElemOp::FUSED_SQUARE:
      return "(" + valVar + " * " + valVar + ")";
    case FusedElemOp::FUSED_SQRT:
      return "(" + valVar + " >= 0.0f ? sqrtf(" + valVar + ") : 0.0f)";
    case FusedElemOp::FUSED_SWISH:
    case FusedElemOp::FUSED_SILU:
      return "(" + valVar + " / (1.0f + __expf(-" + valVar + ")))";
    case FusedElemOp::FUSED_MISH:
      return "(" + valVar + " * tanhf(__logf(1.0f + __expf(" + valVar + "))))";
    case FusedElemOp::FUSED_CLIP:
      // clipMin/Max passed as extra params — handled separately
      return "fminf(fmaxf(" + valVar + ", clipMin), clipMax)";
    case FusedElemOp::FUSED_LEAKY_RELU:
      return "(" + valVar + " >= 0.0f ? " + valVar + " : " + valVar + " * " + secVar + ")";
    default:
      return valVar;  // passthrough
  }
}

// ── buildKernelSource ──────────────────────────────────────────────────────

JitKernelSource buildKernelSource(const NativeSlot* slots, int startSlot, int endSlot,
                                  NDArray** outputSlots, int totalOutputSlots,
                                  int segmentIndex) {
  JitKernelSource result;

  // Collect active (non-skipped) slots in order
  struct ActiveSlot {
    int slotIdx;        // global slot index
    int fusedCode;
    bool isBinary;
    int primaryInputSource;   // inputSourceIndices[0]
    int secondaryInputSource; // inputSourceIndices[1] for binary ops (-1 for unary)
    int outputSlotIdx;        // outputSlotIndices[0]
  };
  std::vector<ActiveSlot> activeSlots;

  for (int i = startSlot; i <= endSlot; i++) {
    const auto& slot = slots[i];
    if (slot.frozenConstantSlot || slot.isIdentityOp || slot.isFusedChainTail) continue;

    int fusedCode = opNameToFusedCode(slot.opName);
    if (fusedCode < 0) {
      result.valid = false;
      return result;
    }

    ActiveSlot as;
    as.slotIdx = i;
    as.fusedCode = fusedCode;
    as.isBinary = isBinaryFusedOp(static_cast<FusedElemOp>(fusedCode));
    as.primaryInputSource = (slot.numInputs > 0) ? slot.inputSourceIndices[0] : -1;
    as.secondaryInputSource = (slot.numInputs > 1 && as.isBinary) ? slot.inputSourceIndices[1] : -1;
    as.outputSlotIdx = (slot.numOutputs > 0) ? slot.outputSlotIndices[0] : -1;
    activeSlots.push_back(as);
  }

  if (activeSlots.empty()) {
    result.valid = false;
    return result;
  }

  // ── Analyze SSA graph: count consumers for each output slot ──
  // An output consumed by exactly 1 downstream slot in this segment
  // can stay in a register variable. Otherwise it needs a global memory pointer.
  std::unordered_map<int, int> outputConsumerCount;
  for (const auto& as : activeSlots) {
    if (as.primaryInputSource >= 0) outputConsumerCount[as.primaryInputSource]++;
    if (as.secondaryInputSource >= 0) outputConsumerCount[as.secondaryInputSource]++;
  }

  // Determine which output slots need global memory pointers
  // (consumed >1 times, or is a segment output = last slot's output)
  std::unordered_set<int> needsGlobalPtr;
  // The last active slot's output always goes to global memory
  if (!activeSlots.empty()) {
    int lastOut = activeSlots.back().outputSlotIdx;
    if (lastOut >= 0) needsGlobalPtr.insert(lastOut);
  }
  for (const auto& as : activeSlots) {
    int outIdx = as.outputSlotIdx;
    if (outIdx >= 0 && outputConsumerCount.count(outIdx) && outputConsumerCount[outIdx] > 1) {
      needsGlobalPtr.insert(outIdx);
    }
  }

  // ── Collect all external input sources (negative inputSourceIndices) ──
  // Maps external input index -> parameter index
  std::unordered_map<int, int> externalInputMap;  // extIdx -> param position
  std::vector<int> externalInputList;  // ordered list of ext indices
  for (const auto& as : activeSlots) {
    auto checkExt = [&](int src) {
      if (src < 0) {
        int extIdx = -(src + 1);
        if (externalInputMap.find(extIdx) == externalInputMap.end()) {
          externalInputMap[extIdx] = static_cast<int>(externalInputList.size());
          externalInputList.push_back(extIdx);
        }
      }
    };
    checkExt(as.primaryInputSource);
    if (as.isBinary) checkExt(as.secondaryInputSource);
  }

  // ── Collect cross-segment input sources (>= 0, from prior segments) ──
  std::unordered_map<int, int> crossSegMap;
  std::vector<int> crossSegList;
  for (const auto& as : activeSlots) {
    auto checkCross = [&](int src) {
      if (src >= 0) {
        // Check if this source is produced within this segment
        bool internal = false;
        for (const auto& other : activeSlots) {
          if (other.outputSlotIdx == src && other.slotIdx < as.slotIdx) {
            internal = true;
            break;
          }
        }
        if (!internal) {
          if (crossSegMap.find(src) == crossSegMap.end()) {
            crossSegMap[src] = static_cast<int>(crossSegList.size());
            crossSegList.push_back(src);
          }
        }
      }
    };
    checkCross(as.primaryInputSource);
    if (as.isBinary) checkCross(as.secondaryInputSource);
  }

  // ── Collect output slots that need global memory store ──
  std::vector<int> outputPtrList;
  std::unordered_map<int, int> outputPtrMap;
  for (int outIdx : needsGlobalPtr) {
    outputPtrMap[outIdx] = static_cast<int>(outputPtrList.size());
    outputPtrList.push_back(outIdx);
  }

  // ── Check for CLIP op (needs clipMin/clipMax params) ──
  bool hasClip = false;
  for (const auto& as : activeSlots) {
    if (as.fusedCode == FusedElemOp::FUSED_CLIP) {
      hasClip = true;
      break;
    }
  }

  // ── Generate kernel source ──
  std::ostringstream src;
  std::string kernelName = "jit_seg_" + std::to_string(segmentIndex) + "_"
                           + std::to_string(startSlot) + "_" + std::to_string(endSlot);

  // Build parameter list
  src << "extern \"C\" __global__ void " << kernelName << "(\n";
  src << "    long long n";

  // External input pointers
  for (size_t i = 0; i < externalInputList.size(); i++) {
    src << ",\n    const float* __restrict__ ext" << i;
  }
  // Cross-segment input pointers
  for (size_t i = 0; i < crossSegList.size(); i++) {
    src << ",\n    const float* __restrict__ cross" << i;
  }
  // Output pointers
  for (size_t i = 0; i < outputPtrList.size(); i++) {
    src << ",\n    float* __restrict__ out" << i;
  }
  // Clip params
  if (hasClip) {
    src << ",\n    float clipMin, float clipMax";
  }
  src << ") {\n";

  // Thread index
  src << "  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n";
  src << "  if (idx >= n) return;\n\n";

  // ── Generate per-slot compute ──
  // Track which output slot index maps to which register variable name
  std::unordered_map<int, std::string> slotToVar;

  for (size_t ai = 0; ai < activeSlots.size(); ai++) {
    const auto& as = activeSlots[ai];
    std::string varName = "v" + std::to_string(ai);

    // Resolve primary input
    std::string primaryExpr;
    if (as.primaryInputSource < 0) {
      int extIdx = -(as.primaryInputSource + 1);
      int paramIdx = externalInputMap[extIdx];
      primaryExpr = "ext" + std::to_string(paramIdx) + "[idx]";
    } else if (slotToVar.count(as.primaryInputSource)) {
      primaryExpr = slotToVar[as.primaryInputSource];
    } else if (crossSegMap.count(as.primaryInputSource)) {
      int paramIdx = crossSegMap[as.primaryInputSource];
      primaryExpr = "cross" + std::to_string(paramIdx) + "[idx]";
    } else {
      // Should not happen for valid segments
      primaryExpr = "0.0f";
    }

    // Resolve secondary input for binary ops
    std::string secondaryExpr = "0.0f";
    if (as.isBinary && as.secondaryInputSource != -1) {
      if (as.secondaryInputSource < 0) {
        int extIdx = -(as.secondaryInputSource + 1);
        int paramIdx = externalInputMap[extIdx];
        secondaryExpr = "ext" + std::to_string(paramIdx) + "[idx]";
      } else if (slotToVar.count(as.secondaryInputSource)) {
        secondaryExpr = slotToVar[as.secondaryInputSource];
      } else if (crossSegMap.count(as.secondaryInputSource)) {
        int paramIdx = crossSegMap[as.secondaryInputSource];
        secondaryExpr = "cross" + std::to_string(paramIdx) + "[idx]";
      }
    }

    // Generate the computation expression
    std::string opExpr = generateOpExpression(as.fusedCode, primaryExpr, secondaryExpr);

    // If output needs global memory, store it; otherwise keep in register
    src << "  float " << varName << " = " << opExpr << ";\n";

    // Track the variable for downstream slots
    if (as.outputSlotIdx >= 0) {
      slotToVar[as.outputSlotIdx] = varName;

      // Store to global memory if needed
      if (needsGlobalPtr.count(as.outputSlotIdx)) {
        int outParamIdx = outputPtrMap[as.outputSlotIdx];
        src << "  out" << outParamIdx << "[idx] = " << varName << ";\n";
      }
    }
  }

  src << "}\n";

  // ── Build param bindings ──
  result.paramBindings.clear();

  // First param: length (n)
  {
    JitParamBinding b;
    b.type = JitParamBinding::LENGTH;
    result.paramBindings.push_back(b);
  }

  // External input pointers
  for (int extIdx : externalInputList) {
    JitParamBinding b;
    b.type = JitParamBinding::INPUT_PTR;
    b.externalInputIdx = extIdx;
    result.paramBindings.push_back(b);
  }

  // Cross-segment input pointers
  for (int slotIdx : crossSegList) {
    JitParamBinding b;
    b.type = JitParamBinding::CROSS_SEG_PTR;
    b.slotIdx = slotIdx;
    result.paramBindings.push_back(b);
  }

  // Output pointers
  for (int slotIdx : outputPtrList) {
    JitParamBinding b;
    b.type = JitParamBinding::OUTPUT_PTR;
    b.slotIdx = slotIdx;
    result.paramBindings.push_back(b);
  }

  result.sourceCode = src.str();
  result.kernelName = kernelName;
  result.valid = true;
  return result;
}

}  // namespace graph
}  // namespace sd

#endif  // SD_CUDA
