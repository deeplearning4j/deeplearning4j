/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <array/TadCalculator.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ShapeUtils.h>
#include <helpers/TadCalculatorPlatform.h>
#include <system/op_boilerplate.h>

#include <memory>
#include <vector>

SD_BACKEND_ABI_NAMESPACE_BEGIN

TadCalculator::TadCalculator(LongType* originalShape)
    : _originalShape(originalShape),
      _tadShape(nullptr),
      _tadOffsets(nullptr),
      _numTads(0),
      _tadShapeOwnership(tad_platform::shapeOwnership()) {}

TadCalculator::~TadCalculator() {
  delete _tadOffsets;
  _tadOffsets = nullptr;
}

void TadCalculator::createTadPack(
    const std::vector<LongType>& dimensions) {
  if (_originalShape == nullptr) {
    THROW_EXCEPTION("TadCalculator: original shape is null");
  }

  const LongType* shapeInfo =
      ConstantShapeHelper::getInstance().createFromExisting(_originalShape);
  if (shape::isEmptyConst(shapeInfo)) {
    THROW_EXCEPTION("TadCalculator: cannot create TADs for an empty array");
  }

  const LongType rank = shape::rank(shapeInfo);
  for (LongType dimension = 0; dimension < rank; ++dimension) {
    if (shape::sizeAt(shapeInfo, dimension) == 0) {
      THROW_EXCEPTION(
          "TadCalculator: cannot create TADs for a zero-sized dimension");
    }
  }

  std::unique_ptr<const std::vector<LongType>> dimensionsToExclude(
      ShapeUtils::evalDimsToExclude(
          rank, dimensions.size(), dimensions.data()));
  if (dimensionsToExclude == nullptr) {
    THROW_EXCEPTION("TadCalculator: failed to evaluate excluded dimensions");
  }

  if (dimensionsToExclude->empty()) {
    _tadShape = tad_platform::createShape(shapeInfo, rank);
    auto offsets = std::make_unique<LongType[]>(
        1 + SD_SHAPE_ALLOC_PADDING);
    offsets[0] = 0;
    _tadOffsets = tad_platform::createOffsets(std::move(offsets), 1);
    _numTads = 1;
    return;
  }

  if (dimensionsToExclude->size() == static_cast<size_t>(rank)) {
    const LongType totalElements = shape::length(shapeInfo);
    const LongType* scalarShapeInfo =
        ConstantShapeHelper::getInstance().scalarShapeInfo(
            ArrayOptions::dataType(shapeInfo));
    _tadShape = tad_platform::createShape(scalarShapeInfo, 0);

    auto offsets = std::make_unique<LongType[]>(
        static_cast<size_t>(totalElements) + SD_SHAPE_ALLOC_PADDING);
    for (LongType index = 0; index < totalElements; ++index) {
      offsets[index] = index;
    }
    _tadOffsets =
        tad_platform::createOffsets(std::move(offsets), totalElements);
    _numTads = totalElements;
    return;
  }

  const LongType numberOfSubarrays =
      ShapeUtils::getNumOfSubArrs(shapeInfo, *dimensionsToExclude);
  if (numberOfSubarrays > 0) {
    const LongType subarrayRank =
        rank - static_cast<LongType>(dimensionsToExclude->size());
    auto subarrayShape = std::make_unique<LongType[]>(
        shape::shapeInfoLength(subarrayRank) + SD_SHAPE_ALLOC_PADDING);
    auto offsets = std::make_unique<LongType[]>(
        static_cast<size_t>(numberOfSubarrays) + SD_SHAPE_ALLOC_PADDING);

    shape::calcSubArrsShapeInfoAndOffsets(
        shapeInfo, numberOfSubarrays, dimensionsToExclude->size(),
        dimensionsToExclude->data(), subarrayShape.get(), offsets.get(), false);

    _tadShape =
        tad_platform::createShape(subarrayShape.get(), subarrayRank);
    _tadOffsets = tad_platform::createOffsets(
        std::move(offsets), numberOfSubarrays);
    _numTads = numberOfSubarrays;
    return;
  }

  auto fullShape = std::make_unique<LongType[]>(
      shape::shapeInfoLength(rank) + SD_SHAPE_ALLOC_PADDING);
  shape::copyTo<LongType>(
      shape::shapeInfoLength(rank), shapeInfo, fullShape.get());

  _tadShape = tad_platform::createShape(fullShape.get(), rank);
  auto offsets = std::make_unique<LongType[]>(
      1 + SD_SHAPE_ALLOC_PADDING);
  offsets[0] = 0;
  _tadOffsets = tad_platform::createOffsets(std::move(offsets), 1);
  _numTads = 1;
}

SD_BACKEND_ABI_NAMESPACE_END
