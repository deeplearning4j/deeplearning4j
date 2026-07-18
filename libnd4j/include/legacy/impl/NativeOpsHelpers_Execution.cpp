/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <legacy/NativeOps.h>
#include <ops/declarable/OpRegistrator.h>

#include <vector>

#if defined(SD_VULKAN)
#error "Vulkan custom-op execution must be provided by the Vulkan eager executor"
#endif


sd::Status execCustomOp(sd::Pointer *extraPointers, sd::LongType hash, OpaqueNDArrayArr inputs, int numInputs,
                        OpaqueNDArrayArr outputs, int numOutputs, double *tArgs, int numTArgs,
                        sd::LongType *iArgs, int numIArgs, bool *bArgs, int numBArgs, bool isInplace) {
#ifdef __cpp_exceptions
  try {
    const std::vector<sd::NDArray *> inputVec(inputs, inputs + numInputs);
    const std::vector<sd::NDArray *> outputVec(outputs, outputs + numOutputs);
    const std::vector<double> tArgsVec(tArgs, tArgs + numTArgs);
    const std::vector<sd::LongType> iArgsVec(iArgs, iArgs + numIArgs);
    const std::vector<bool> bArgsVec(bArgs, bArgs + numBArgs);

    auto op = sd::ops::OpRegistrator::getInstance().getOperation(hash);
    if (op == nullptr) THROW_EXCEPTION("Operation not found for the given hash.");

    return op->execute(inputVec, outputVec, tArgsVec, iArgsVec, bArgsVec, {}, isInplace);
  } catch (std::exception &e) {
    safeSetErrorContext(1, e.what());
    return sd::Status::KERNEL_FAILURE;
  }
#else
  const std::vector<sd::NDArray *> inputVec(inputs, inputs + numInputs);
  const std::vector<sd::NDArray *> outputVec(outputs, outputs + numOutputs);
  const std::vector<double> tArgsVec(tArgs, tArgs + numTArgs);
  const std::vector<sd::LongType> iArgsVec(iArgs, iArgs + numIArgs);
  const std::vector<bool> bArgsVec(bArgs, bArgs + numBArgs);

  auto op = sd::ops::OpRegistrator::getInstance().getOperation(hash);
  if (op == nullptr) {
    safeSetErrorContext(1, "Operation not found for the given hash.");
    return sd::Status::KERNEL_FAILURE;
  }

  return op->execute(inputVec, outputVec, tArgsVec, iArgsVec, bArgsVec, {}, isInplace);
#endif
}

