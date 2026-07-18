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

#ifndef LIBND4J_HEXAGON_IR_BUILDER_H
#define LIBND4J_HEXAGON_IR_BUILDER_H

#include <system/common.h>

#ifdef HAVE_HEXAGON_MLIR

#include <array/NDArray.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace sd {
namespace graph {

// Forward declarations — defined in NativeDynamicShapePlan.h
struct NativeSlot;
struct GraphSegment;

/**
 * Converts NativeSlot sequences to MLIR bytecode targeting the Hexagon NPU.
 *
 * Uses hexagon-mlir's Triton dialect for kernel compilation:
 *   - Elementwise ops -> HVX (Hexagon Vector eXtension) vector operations
 *   - Matmul ops -> HVX VMPY + VACC pipeline
 *   - Memory ops -> DMA operations for TCM staging
 *
 * Static utility class — all methods are stateless.
 */
class SD_LIB_EXPORT HexagonIRBuilder {
 public:
  // Non-instantiable static utility
  HexagonIRBuilder() = delete;

  /**
   * Build an MLIR bytecode module from a sequence of NativeSlots.
   *
   * @param slots          Full slot array
   * @param start          Start index (inclusive) in the slot array
   * @param end            End index (inclusive) in the slot array
   * @param externalInputs External input arrays (constants, variables, placeholders)
   * @param numExt         Number of external inputs
   * @return MLIR bytecode as a byte vector, or empty vector on failure
   */
  static std::vector<uint8_t> buildModule(NativeSlot* slots, int start, int end,
                                           NDArray** externalInputs, int numExt);

  /**
   * Check if a given op is mappable to Hexagon HVX instructions.
   *
   * @param opName Op name string (e.g., "add", "multiply", "matmul")
   * @return true if the op can be compiled to HVX
   */
  static bool isHexagonMappable(const char* opName);

  /**
   * Estimate TCM (Tightly Coupled Memory) usage for a slot range.
   *
   * Computes the worst-case TCM footprint by summing input/output buffer sizes
   * for all ops in the range. Used by HexagonGraphBackend::canFuseSegment() to
   * ensure the working set fits in on-chip TCM.
   *
   * @param slots  Full slot array
   * @param start  Start index (inclusive)
   * @param end    End index (inclusive)
   * @return Estimated TCM usage in bytes
   */
  static size_t estimateTcmUsage(NativeSlot* slots, int start, int end);

  /**
   * Get the set of all Hexagon-mappable op names.
   * @return Const reference to the whitelist set
   */
  static const std::unordered_set<std::string>& getMappableOps();

 private:
  /**
   * Emit MLIR for a single elementwise op (add, mul, relu, etc.).
   * @param mlirBuffer Output string buffer for MLIR text
   * @param slot       The slot to emit
   * @param slotIndex  Absolute index of this slot
   */
  static void emitElementwise(std::string& mlirBuffer,
                               const NativeSlot& slot, int slotIndex);

  /**
   * Emit MLIR for a matmul op using HVX VMPY + VACC pipeline.
   * @param mlirBuffer Output string buffer for MLIR text
   * @param slot       The slot to emit
   * @param slotIndex  Absolute index of this slot
   */
  static void emitMatmul(std::string& mlirBuffer,
                          const NativeSlot& slot, int slotIndex);

  /**
   * Emit MLIR for DMA operations (load from DDR to TCM, store from TCM to DDR).
   * @param mlirBuffer Output string buffer for MLIR text
   * @param slot       The slot to emit
   * @param slotIndex  Absolute index of this slot
   */
  static void emitDmaOps(std::string& mlirBuffer,
                          const NativeSlot& slot, int slotIndex);

  /**
   * Convert MLIR text to bytecode using the hexagon-mlir compiler.
   * @param mlirText MLIR text representation
   * @return Bytecode as a byte vector, or empty on failure
   */
  static std::vector<uint8_t> serializeToBytecode(const std::string& mlirText);
};

}  // namespace graph
}  // namespace sd

#endif  // HAVE_HEXAGON_MLIR
#endif  // LIBND4J_HEXAGON_IR_BUILDER_H
