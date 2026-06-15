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

#ifndef LIBND4J_HLO_IR_BUILDER_H
#define LIBND4J_HLO_IR_BUILDER_H

#include <system/common.h>

#ifdef SD_TPU

#include <array/NDArray.h>

#include <cstdint>
#include <string>
#include <vector>

namespace sd {
namespace graph {

// Forward declarations — avoid circular include with NativeDynamicShapePlan.h
struct NativeSlot;

/**
 * Converts a sequence of NativeSlots to a serialized HLO module text.
 *
 * Maps ND4J ops to HLO opcodes using a whitelist-based approach.
 * Only ops with known HLO equivalents are supported. Unsupported ops
 * cause the entire segment to be marked as non-fusible.
 *
 * Supported HLO opcode mappings:
 *   add, subtract, multiply, divide        -> HLO add, subtract, multiply, divide
 *   matmul / mmul                           -> HLO dot
 *   relu                                    -> HLO maximum(x, 0)
 *   softmax                                 -> HLO custom (exp/sum/div)
 *   reshape / reshape_no_copy               -> HLO reshape
 *   permute / transpose                     -> HLO transpose
 *   concat                                  -> HLO concatenate
 *   gather                                  -> HLO gather
 *   scatter                                 -> HLO scatter
 *   tanh                                    -> HLO tanh
 *   sigmoid                                 -> HLO logistic
 *
 * All methods are static — this is a stateless utility class.
 */
class SD_LIB_EXPORT HloIRBuilder {
 public:
  /**
   * Build a serialized HLO module from a range of slots.
   *
   * @param slots           All slots in the plan
   * @param start           Start slot index (inclusive)
   * @param end             End slot index (exclusive)
   * @param externalInputs  External input arrays (constants, variables, placeholders)
   * @param numExt          Number of external inputs
   * @return Serialized HLO module text, or empty string on failure
   */
  static std::string buildModule(NativeSlot* slots, int start, int end,
                                 NDArray** externalInputs, int numExt);

  /**
   * Check if a named op can be mapped to an HLO opcode.
   *
   * @param opName  The ND4J op name (e.g., "add", "matmul", "relu")
   * @return true if this op has a known HLO equivalent
   */
  static bool isHloMappable(const char* opName);

  /**
   * Get the list of all supported ND4J op names that can be mapped to HLO.
   *
   * @return vector of supported op name strings
   */
  static std::vector<std::string> getSupportedOps();

 private:
  // No instances — all static methods
  HloIRBuilder() = delete;

  /**
   * Map an ND4J op name to its HLO opcode string.
   *
   * @param opName  The ND4J op name
   * @return HLO opcode string, or nullptr if not mappable
   */
  static const char* mapOpToHlo(const char* opName);

  /**
   * Format an NDArray shape as an HLO shape string (e.g., "f32[2,3,4]").
   *
   * @param arr  The array whose shape to format
   * @return HLO shape string
   */
  static std::string formatHloShape(const NDArray* arr);

  /**
   * Format a data type as an HLO type string (e.g., "f32", "f16", "s32").
   *
   * @param dt  The ND4J data type
   * @return HLO type string
   */
  static std::string formatHloType(DataType dt);
};

}  // namespace graph
}  // namespace sd

#endif  // SD_TPU
#endif  // LIBND4J_HLO_IR_BUILDER_H
