/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <array/CudaPointerDeallocator.h>
#include <array/PrimaryPointerDeallocator.h>
#include <helpers/ConstantHelper.h>
#include <helpers/TadCalculatorPlatform.h>
#include <helpers/cuda/CudaShapeBufferCreator.h>

#include <memory>

SD_BACKEND_ABI_NAMESPACE_BEGIN
namespace tad_platform {

TadShapeOwnership shapeOwnership() {
  return TadShapeOwnership::Owned;
}

ConstantShapeBuffer* createShape(const LongType* shapeInfo, LongType rank) {
  return CudaShapeBufferCreator::getInstance().create(
      shapeInfo, static_cast<int>(rank));
}

ConstantOffsetsBuffer* createOffsets(
    std::unique_ptr<LongType[]> offsets, LongType count) {
  if (offsets == nullptr || count < 0) {
    THROW_EXCEPTION("TadCalculator: invalid CUDA offsets");
  }

  const size_t allocationElements =
      static_cast<size_t>(count) + SD_SHAPE_ALLOC_PADDING;
  auto primary = std::make_shared<PointerWrapper>(
      offsets.release(), std::make_shared<PrimaryPointerDeallocator>());
  auto special = std::make_shared<PointerWrapper>(
      ConstantHelper::getInstance().replicatePointer(
          primary->pointer(), allocationElements * sizeof(LongType)),
      std::make_shared<CudaPointerDeallocator>());
  return new ConstantOffsetsBuffer(primary, special);
}

}  // namespace tad_platform
SD_BACKEND_ABI_NAMESPACE_END
