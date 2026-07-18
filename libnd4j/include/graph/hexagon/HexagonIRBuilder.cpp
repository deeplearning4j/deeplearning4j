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

#include <graph/hexagon/HexagonIRBuilder.h>
#include <graph/NativeDynamicShapePlan.h>
#include <graph/DspDiagnostics.h>
#include <array/DataTypeUtils.h>
#include <helpers/shape.h>

#include <cstring>
#include <sstream>

namespace sd {
namespace graph {

// ---- Hexagon-mappable op whitelist ----
// These ops can be compiled to HVX vector instructions via hexagon-mlir.

static const std::unordered_set<std::string>& buildMappableOps() {
  static const std::unordered_set<std::string> ops = {
      // Elementwise arithmetic
      "add", "subtract", "multiply", "divide",
      "multiply_no_nan", "floormod", "floordiv",

      // Elementwise unary
      "abs", "neg", "sign",
      "sqrt", "rsqrt", "square",
      "exp", "log", "log1p",
      "ceil", "floor", "round",
      "sin", "cos", "tan",
      "sinh", "cosh", "tanh",
      "sigmoid", "relu", "relu6",
      "elu", "selu", "gelu",
      "silu", "swish",
      "softplus", "softsign",

      // Elementwise comparison
      "maximum", "minimum",
      "greater", "less", "greater_equal", "less_equal",
      "equals", "not_equals",

      // Reduction (simple — 1D only for HVX)
      "reduce_sum", "reduce_mean", "reduce_max", "reduce_min",

      // Type casting
      "cast",

      // Matmul (via HVX VMPY + VACC pipeline)
      "matmul", "mmul",

      // Normalization
      "rms_norm",

      // Misc
      "identity", "assign",
  };
  return ops;
}

const std::unordered_set<std::string>& HexagonIRBuilder::getMappableOps() {
  return buildMappableOps();
}

bool HexagonIRBuilder::isHexagonMappable(const char* opName) {
  if (opName == nullptr) return false;
  const auto& ops = buildMappableOps();
  return ops.find(std::string(opName)) != ops.end();
}

// ---- TCM Usage Estimation ----

size_t HexagonIRBuilder::estimateTcmUsage(NativeSlot* slots, int start, int end) {
  if (slots == nullptr || start > end) return 0;

  size_t totalBytes = 0;

  for (int i = start; i <= end; i++) {
    const auto& slot = slots[i];

    // Sum input buffer sizes.
    // inputSourceIndices >= 0 → prior slot output (already counted); skip.
    // inputSourceIndices <  0 → external input: -(idx+1) into externalInputs.
    // We have no array pointers here, so we estimate from cachedOutputShapes
    // of the producing slot when available. For external inputs the shapes
    // are not in shapeCache (those belong to the caller), so we conservatively
    // count 0 bytes for them — the estimate is non-load-bearing; it only
    // gates whether we attempt the segment at all, not correctness.
    for (int j = 0; j < slot.wiring.numInputs; j++) {
      int srcIdx = slot.wiring.inputSourceIndices != nullptr
                       ? slot.wiring.inputSourceIndices[j]
                       : -1;
      if (srcIdx >= 0) {
        // Cross-slot reference: already counted when we processed the
        // producing slot — skip to avoid double-counting.
        continue;
      }
      // External input: byte size is unknown here without an NDArray*.
      // Leave contribution as 0; caller can pass real arrays if needed.
    }

    // Sum output buffer sizes from the slot's own shapeCache when available.
    for (int k = 0; k < slot.wiring.numOutputs; k++) {
      if (k < static_cast<int>(slot.shapeCache.cachedOutputShapes.size())) {
        const LongType* si = slot.shapeCache.cachedOutputShapes[k];
        if (si != nullptr) {
          // shape::length returns element count; DataTypeUtils::sizeOf gives
          // the element byte size from the shapeInfo type field.
          totalBytes += static_cast<size_t>(shape::length(si)) *
                        DataTypeUtils::sizeOf(si);
        }
      }
    }
  }

  return totalBytes;
}

// ---- MLIR Emission ----

void HexagonIRBuilder::emitElementwise(std::string& mlirBuffer,
                                        const NativeSlot& slot, int slotIndex) {
  // Skeletal MLIR emission for elementwise ops targeting Hexagon HVX.
  // The actual MLIR dialect depends on the hexagon-mlir SDK version.
  std::ostringstream oss;
  oss << "  // slot " << slotIndex << ": " << (slot.ident.opName.empty() ? std::string("unknown") : slot.ident.opName)
      << " (elementwise -> HVX)\n";
  oss << "  %out_" << slotIndex << " = \"hexagon.hvx_elementwise\"(";

  for (int j = 0; j < slot.wiring.numInputs; j++) {
    if (j > 0) oss << ", ";
    int srcIdx = slot.wiring.inputSourceIndices != nullptr
                     ? slot.wiring.inputSourceIndices[j]
                     : -1;
    if (srcIdx >= 0) {
      oss << "%out_" << srcIdx;
    } else {
      oss << "%ext_input_" << slotIndex << "_" << j;
    }
  }

  oss << ") {op = \"" << (slot.ident.opName.empty() ? std::string("unknown") : slot.ident.opName) << "\"}"
      << " : () -> tensor<*xf32>\n";

  mlirBuffer += oss.str();
}

void HexagonIRBuilder::emitMatmul(std::string& mlirBuffer,
                                   const NativeSlot& slot, int slotIndex) {
  // Skeletal MLIR emission for matmul using HVX VMPY + VACC pipeline
  std::ostringstream oss;
  oss << "  // slot " << slotIndex << ": matmul -> HVX VMPY+VACC\n";
  oss << "  %out_" << slotIndex << " = \"hexagon.hvx_matmul\"(";

  if (slot.wiring.numInputs >= 2) {
    for (int j = 0; j < 2; j++) {
      if (j > 0) oss << ", ";
      int srcIdx = slot.wiring.inputSourceIndices != nullptr
                       ? slot.wiring.inputSourceIndices[j]
                       : -1;
      if (srcIdx >= 0) {
        oss << "%out_" << srcIdx;
      } else {
        oss << "%ext_input_" << slotIndex << "_" << j;
      }
    }
  }

  oss << ") : () -> tensor<*xf32>\n";
  mlirBuffer += oss.str();
}

void HexagonIRBuilder::emitDmaOps(std::string& mlirBuffer,
                                   const NativeSlot& slot, int slotIndex) {
  // Skeletal MLIR emission for DMA operations (DDR <-> TCM)
  std::ostringstream oss;

  // Load inputs from DDR to TCM (only external inputs need DMA load;
  // cross-slot outputs are already in TCM from the producing slot).
  for (int j = 0; j < slot.wiring.numInputs; j++) {
    int srcIdx = slot.wiring.inputSourceIndices != nullptr
                     ? slot.wiring.inputSourceIndices[j]
                     : -1;
    if (srcIdx < 0) {
      // External input (srcIdx < 0) lives in DDR and must be DMA'd into TCM.
      oss << "  %tcm_in_" << slotIndex << "_" << j
          << " = \"hexagon.dma_load\"(%ext_input_" << slotIndex << "_" << j
          << ") : () -> memref<*xf32, 1>\n";  // addrspace 1 = TCM
    }
  }

  // Store output from TCM to DDR
  oss << "  \"hexagon.dma_store\"(%out_" << slotIndex
      << ", %ddr_out_" << slotIndex << ") : () -> ()\n";

  mlirBuffer += oss.str();
}

// ---- Module Building ----

std::vector<uint8_t> HexagonIRBuilder::buildModule(NativeSlot* slots, int start, int end,
                                                     NDArray** externalInputs, int numExt) {
  if (slots == nullptr || start > end) {
    DSP_DIAG(COMPILE, "HexagonIRBuilder::buildModule: invalid range [%d, %d]",
             start, end);
    return {};
  }

  std::string mlirText;
  mlirText.reserve(4096);

  // Module header
  mlirText += "module {\n";
  mlirText += "func.func @hexagon_fused_kernel() {\n";

  int mappedOps = 0;
  int skippedOps = 0;

  for (int i = start; i <= end; i++) {
    const auto& slot = slots[i];
    const std::string& opNameStr = slot.ident.opName;
    const char* opName = opNameStr.c_str();

    if (!isHexagonMappable(opName)) {
      DSP_DIAG(COMPILE, "HexagonIRBuilder::buildModule: slot %d op '%s' not mappable, "
               "skipping", i, opName);
      skippedOps++;
      continue;
    }

    // Route to appropriate emitter
    if (std::strcmp(opName, "matmul") == 0 || std::strcmp(opName, "mmul") == 0) {
      emitDmaOps(mlirText, slot, i);
      emitMatmul(mlirText, slot, i);
    } else {
      emitDmaOps(mlirText, slot, i);
      emitElementwise(mlirText, slot, i);
    }

    mappedOps++;
  }

  mlirText += "  return\n";
  mlirText += "}\n";
  mlirText += "}\n";

  DSP_DIAG(COMPILE, "HexagonIRBuilder::buildModule: [%d, %d] -> %d mapped, %d skipped, "
           "%zu bytes MLIR text", start, end, mappedOps, skippedOps, mlirText.size());

  if (mappedOps == 0) {
    DSP_DIAG(COMPILE, "HexagonIRBuilder::buildModule: no ops mapped, returning empty");
    return {};
  }

  // Convert MLIR text to bytecode
  return serializeToBytecode(mlirText);
}

std::vector<uint8_t> HexagonIRBuilder::serializeToBytecode(const std::string& mlirText) {
  // Skeletal implementation — actual serialization requires the hexagon-mlir
  // compiler library. For now, return the MLIR text as raw bytes which the
  // HexagonRuntimeManager::compileKernel() will process.
  //
  // When the hexagon-mlir SDK is available, this should:
  //   1. Parse the MLIR text into an MLIR Module
  //   2. Run the Hexagon lowering passes (Triton dialect -> HVX dialect -> machine code)
  //   3. Serialize to bytecode format expected by the runtime
  DSP_DIAG(COMPILE, "HexagonIRBuilder::serializeToBytecode: %zu bytes MLIR text -> "
           "raw bytecode (SDK required for proper compilation)", mlirText.size());

  std::vector<uint8_t> bytecode(mlirText.begin(), mlirText.end());
  return bytecode;
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_HEXAGON_MLIR
