/* ******************************************************************************
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef LIBND4J_TPU_STABLEHLOGRAPHLOWERING_H
#define LIBND4J_TPU_STABLEHLOGRAPHLOWERING_H

#include <system/common.h>

#ifdef SD_TPU

#include <array/NDArray.h>
#include <graph/GraphBackendCommon.h>

#include <string>

namespace sd {
namespace graph {

struct StableHloLoweringResult {
  bool success = false;
  std::string program;
  std::string format = "mlir";
  FunctionalGraphBoundary boundary;
  int failedSlot = -1;
  std::string error;
};

/**
 * Assembles one functional StableHLO module from NativeSlots.
 *
 * Normal op equations come exclusively from KernelSpec/KernelExpr resolved by
 * canonical descriptor hash. Op-local NativeSlot traits authorize the family
 * and safety contract. This class owns only graph boundaries, tensor types,
 * broadcast adaptation, structural recipes, and function result assembly.
 */
class SD_LIB_EXPORT StableHloGraphLowering {
 public:
  static bool canLowerSlot(const NativeSlot& slot, std::string* reason = nullptr);

  static StableHloLoweringResult lower(
      NativeSlot* slots, int start, int end,
      NDArray** externalInputs, int numExternalInputs,
      NDArray** outputSlots, int totalOutputSlots,
      int totalSlots = 0,
      int* requestedOutputSlotIndices = nullptr,
      int numRequestedOutputs = 0);

 private:
  StableHloGraphLowering() = delete;
};

}  // namespace graph
}  // namespace sd

#endif  // SD_TPU
#endif  // LIBND4J_TPU_STABLEHLOGRAPHLOWERING_H
