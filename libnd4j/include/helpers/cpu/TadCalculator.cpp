/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <array/PrimaryPointerDeallocator.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/TadCalculatorPlatform.h>

#include <memory>

SD_BACKEND_ABI_NAMESPACE_BEGIN
namespace tad_platform {

TadShapeOwnership shapeOwnership() {
  return TadShapeOwnership::CachedReference;
}

ConstantShapeBuffer* createShape(const LongType* shapeInfo, LongType) {
  // ConstantShapeHelper's legacy ABI takes a mutable pointer but only reads the
  // descriptor before copying it into the constant shape cache.
  return ConstantShapeHelper::getInstance().bufferForShapeInfo(
      const_cast<LongType*>(shapeInfo));
}

ConstantOffsetsBuffer* createOffsets(
    std::unique_ptr<LongType[]> offsets, LongType count) {
  if (offsets == nullptr || count < 0) {
    THROW_EXCEPTION("TadCalculator: invalid CPU offsets");
  }
  auto primary = std::make_shared<PointerWrapper>(
      offsets.release(), std::make_shared<PrimaryPointerDeallocator>());
  return new ConstantOffsetsBuffer(primary);
}

}  // namespace tad_platform
SD_BACKEND_ABI_NAMESPACE_END
