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

#ifdef SD_TPU

#include <graph/tpu/HloIRBuilder.h>
#include <graph/NativeDynamicShapePlan.h>
#include <graph/DspDiagnostics.h>

#include <cstring>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace sd {
namespace graph {

// ── Op name -> HLO opcode mapping ───────────────────────────────────────────

struct HloOpMapping {
  const char* nd4jName;
  const char* hloOpcode;
};

// Whitelist of supported op mappings
static const HloOpMapping g_hloOpMappings[] = {
    {"add",              "add"},
    {"Add",              "add"},
    {"subtract",         "subtract"},
    {"Subtract",         "subtract"},
    {"multiply",         "multiply"},
    {"Multiply",         "multiply"},
    {"divide",           "divide"},
    {"Divide",           "divide"},
    {"matmul",           "dot"},
    {"Matmul",           "dot"},
    {"mmul",             "dot"},
    {"Mmul",             "dot"},
    {"relu",             "maximum"},       // relu = max(x, 0)
    {"Relu",             "maximum"},
    {"softmax",          "custom-call"},   // softmax = exp(x - max) / sum(exp)
    {"Softmax",          "custom-call"},
    {"reshape",          "reshape"},
    {"Reshape",          "reshape"},
    {"reshape_no_copy",  "reshape"},
    {"permute",          "transpose"},
    {"Permute",          "transpose"},
    {"transpose",        "transpose"},
    {"Transpose",        "transpose"},
    {"concat",           "concatenate"},
    {"Concat",           "concatenate"},
    {"gather",           "gather"},
    {"Gather",           "gather"},
    {"scatter_update",   "scatter"},
    {"scatter",          "scatter"},
    {"Scatter",          "scatter"},
    {"tanh",             "tanh"},
    {"Tanh",             "tanh"},
    {"sigmoid",          "logistic"},
    {"Sigmoid",          "logistic"},
    {nullptr,            nullptr}  // Sentinel
};

// Build a lookup set for fast O(1) checking
static std::unordered_map<std::string, const char*>& getOpLookup() {
  // -fno-threadsafe-statics: use std::call_once for thread-safe initialization.
  static std::once_flag opLookupFlag;
  static std::unordered_map<std::string, const char*> lookup;
  std::call_once(opLookupFlag, []() {
    for (int i = 0; g_hloOpMappings[i].nd4jName != nullptr; ++i) {
      lookup[g_hloOpMappings[i].nd4jName] = g_hloOpMappings[i].hloOpcode;
    }
  });
  return lookup;
}

// ── Public API ──────────────────────────────────────────────────────────────

bool HloIRBuilder::isHloMappable(const char* opName) {
  if (opName == nullptr) return false;
  auto& lookup = getOpLookup();
  return lookup.find(opName) != lookup.end();
}

const char* HloIRBuilder::mapOpToHlo(const char* opName) {
  if (opName == nullptr) return nullptr;
  auto& lookup = getOpLookup();
  auto it = lookup.find(opName);
  return (it != lookup.end()) ? it->second : nullptr;
}

std::vector<std::string> HloIRBuilder::getSupportedOps() {
  // Return unique op names (deduplicated, since we map both cases)
  std::unordered_set<std::string> seen;
  std::vector<std::string> result;
  for (int i = 0; g_hloOpMappings[i].nd4jName != nullptr; ++i) {
    std::string name = g_hloOpMappings[i].nd4jName;
    if (seen.insert(name).second) {
      result.push_back(name);
    }
  }
  return result;
}

// ── Shape/Type Formatting ───────────────────────────────────────────────────

std::string HloIRBuilder::formatHloType(DataType dt) {
  switch (dt) {
    case FLOAT32: return "f32";
    case HALF: return "f16";
    case BFLOAT16: return "bf16";
    case DOUBLE: return "f64";
    case INT8: return "s8";
    case INT16: return "s16";
    case INT32: return "s32";
    case INT64: return "s64";
    case UINT8: return "u8";
    case UINT16: return "u16";
    case UINT32: return "u32";
    case UINT64: return "u64";
    case BOOL: return "pred";
    default:
      DSP_DIAG(COMPILE, "HloIRBuilder: unsupported data type %d",
               static_cast<int>(dt));
      return "f32";  // Fallback
  }
}

std::string HloIRBuilder::formatHloShape(const NDArray* arr) {
  if (arr == nullptr) return "f32[]";

  std::ostringstream oss;
  oss << formatHloType(arr->dataType());
  oss << "[";
  for (int i = 0; i < arr->rankOf(); ++i) {
    if (i > 0) oss << ",";
    oss << arr->sizeAt(i);
  }
  oss << "]";
  return oss.str();
}

// ── Module Builder ──────────────────────────────────────────────────────────

std::string HloIRBuilder::buildModule(NativeSlot* slots, int start, int end,
                                       NDArray** externalInputs, int numExt) {
  if (slots == nullptr || start >= end) {
    DSP_DIAG(COMPILE, "HloIRBuilder::buildModule: invalid range [%d, %d)",
             start, end);
    return "";
  }

  // First pass: verify all ops in range are HLO-mappable
  for (int i = start; i < end; ++i) {
    const char* opName = slots[i].ident.opName.c_str();
    if (!isHloMappable(opName)) {
      DSP_DIAG(COMPILE, "HloIRBuilder::buildModule: slot %d op '%s' not "
               "HLO-mappable, aborting segment", i, opName);
      return "";
    }
  }

  // Build HLO module text
  // HLO text format reference:
  //   HloModule <name>
  //   ENTRY <name> (<params>) -> <return_type> {
  //     <param0> = <type> parameter(0)
  //     ...
  //     <result> = <type> <opcode>(<operands>)
  //     ROOT <result>
  //   }

  std::ostringstream hlo;
  hlo << "HloModule dsp_segment_" << start << "_" << end << "\n\n";
  hlo << "ENTRY main {\n";

  // Track SSA value names for each slot output and external input
  // External inputs: "ext_N", slot outputs: "slot_N"
  // Map from source index to SSA name
  std::unordered_map<int, std::string> ssaNames;

  // Emit parameter declarations for external inputs referenced by this segment
  std::unordered_set<int> usedExternals;
  for (int i = start; i < end; ++i) {
    for (int j = 0; j < slots[i].wiring.numInputs; ++j) {
      int srcIdx = slots[i].wiring.inputSourceIndices[j];
      if (srcIdx < 0) {
        // External input: -(srcIdx + 1)
        int extIdx = -(srcIdx + 1);
        usedExternals.insert(extIdx);
      }
    }
  }

  int paramIdx = 0;
  for (int extIdx : usedExternals) {
    std::string paramName = "ext_" + std::to_string(extIdx);

    // Format shape from the external input array
    std::string shape = "f32[]";  // Default fallback
    if (externalInputs != nullptr && extIdx >= 0 && extIdx < numExt &&
        externalInputs[extIdx] != nullptr) {
      shape = formatHloShape(externalInputs[extIdx]);
    }

    hlo << "  " << paramName << " = " << shape
        << " parameter(" << paramIdx << ")\n";

    // Map the negative source index to this SSA name
    int srcIdx = -(extIdx + 1);
    ssaNames[srcIdx] = paramName;
    paramIdx++;
  }

  // Second pass: emit HLO instructions for each slot
  std::string lastSlotName;
  for (int i = start; i < end; ++i) {
    const char* opName = slots[i].ident.opName.c_str();
    const char* hloOp = mapOpToHlo(opName);
    if (hloOp == nullptr) continue;  // Should not happen (verified above)

    std::string slotName = "slot_" + std::to_string(i);

    // Collect operand SSA names
    std::vector<std::string> operands;
    for (int j = 0; j < slots[i].wiring.numInputs; ++j) {
      int srcIdx = slots[i].wiring.inputSourceIndices[j];
      auto it = ssaNames.find(srcIdx);
      if (it != ssaNames.end()) {
        operands.push_back(it->second);
      } else {
        // Cross-segment input or prior slot — use a placeholder
        operands.push_back("slot_" + std::to_string(srcIdx));
      }
    }

    // Build operand list
    std::ostringstream opList;
    for (size_t j = 0; j < operands.size(); ++j) {
      if (j > 0) opList << ", ";
      opList << operands[j];
    }

    // Determine output shape (best-effort from cached shapes or external inputs)
    std::string outputShape = "f32[]";  // Placeholder

    // Emit the instruction
    hlo << "  " << slotName << " = " << outputShape
        << " " << hloOp << "(" << opList.str() << ")\n";

    // Register this slot's output SSA name for downstream consumers
    if (slots[i].wiring.numOutputs > 0) {
      for (int k = 0; k < slots[i].wiring.numOutputs; ++k) {
        ssaNames[slots[i].wiring.outputSlotIndices[k]] = slotName;
      }
    }

    lastSlotName = slotName;
  }

  // ROOT instruction (last slot in segment)
  if (!lastSlotName.empty()) {
    hlo << "  ROOT result = " << lastSlotName << "\n";
  }

  hlo << "}\n";

  std::string result = hlo.str();

  DSP_DIAG(COMPILE, "HloIRBuilder::buildModule: generated %zu bytes HLO for "
           "segment [%d, %d) with %d slots, %d external params",
           result.size(), start, end, end - start,
           static_cast<int>(usedExternals.size()));

  return result;
}

}  // namespace graph
}  // namespace sd

#endif  // SD_TPU
