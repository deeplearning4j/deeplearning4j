/* ******************************************************************************
 *
 * Copyright (c) 2026 Eclipse Foundation
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <system/op_boilerplate.h>

#include <array/ArrayOptions.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/shape.h>
#include <ops/declarable/headers/legacy_indexing.h>

#include <memory>
#include <vector>

namespace sd {
namespace ops {
namespace {

bool sameLogicalShape(const sd::LongType* left, const sd::LongType* right) {
  const auto leftRank = shape::rank(left);
  if (leftRank != shape::rank(right)) return false;

  const auto* leftShape = shape::shapeOf(left);
  const auto* rightShape = shape::shapeOf(right);
  for (sd::LongType dimension = 0; dimension < leftRank; ++dimension) {
    if (leftShape[dimension] != rightShape[dimension]) return false;
  }
  return true;
}

void requireDimensionVector(NDArray* array,
                            const std::vector<sd::LongType>& dimensions,
                            const char* operation) {
  REQUIRE_TRUE(!dimensions.empty(), 0, "%s: dimensions must not be empty",
               operation);

  std::vector<bool> seen(static_cast<size_t>(array->rankOf()), false);
  for (const auto dimension : dimensions) {
    REQUIRE_TRUE(dimension >= 0 && dimension < array->rankOf(), 0,
                 "%s: dimension %lld is outside rank %i", operation,
                 static_cast<long long>(dimension), array->rankOf());
    REQUIRE_TRUE(!seen[static_cast<size_t>(dimension)], 0,
                 "%s: dimension %lld is repeated", operation,
                 static_cast<long long>(dimension));
    seen[static_cast<size_t>(dimension)] = true;
  }
}

template <typename T>
void pullIndexedElements(NDArray* source, NDArray* destination,
                         const NDArray* indexes, sd::LongType count) {
  const auto* sourceBuffer = source->bufferAsT<T>();
  auto* destinationBuffer = destination->bufferAsT<T>();
  const auto* indexBuffer = const_cast<NDArray*>(indexes)->bufferAsT<sd::LongType>();
  const auto* sourceShapeInfo = source->shapeInfo();
  const auto* destinationShapeInfo = destination->shapeInfo();
  const auto* sourceShape = shape::shapeOf(sourceShapeInfo);
  const auto* destinationShape = shape::shapeOf(destinationShapeInfo);
  const auto* sourceStride = shape::stride(sourceShapeInfo);
  const auto* destinationStride = shape::stride(destinationShapeInfo);

  for (sd::LongType pullIndex = 0; pullIndex < count; ++pullIndex) {
    sd::LongType sourceCoordinates[SD_MAX_RANK];
    sd::LongType destinationCoordinates[SD_MAX_RANK];
    sd::LongType sourceOffset;
    sd::LongType destinationOffset;

    INDEX2COORDS(indexBuffer[pullIndex], 1, sourceShape, sourceCoordinates);
    COORDS2INDEX(1, sourceStride, sourceCoordinates, sourceOffset);
    INDEX2COORDS(pullIndex, 1, destinationShape,
                 destinationCoordinates);
    COORDS2INDEX(1, destinationStride, destinationCoordinates,
                 destinationOffset);
    destinationBuffer[destinationOffset] = sourceBuffer[sourceOffset];
  }
}

template <typename T>
void pullIndexedTads(
    NDArray* source, NDArray* destination, const NDArray* indexes,
    sd::LongType count, const std::shared_ptr<sd::TadPack>& sourceTads,
    const std::shared_ptr<sd::TadPack>& destinationTads) {
  const auto* sourceBuffer = source->bufferAsT<T>();
  auto* destinationBuffer = destination->bufferAsT<T>();
  const auto* indexBuffer = const_cast<NDArray*>(indexes)->bufferAsT<sd::LongType>();

  const auto* sourceTadShapeInfo = sourceTads->primaryShapeInfo();
  const auto* destinationTadShapeInfo = destinationTads->primaryShapeInfo();
  const auto* sourceTadOffsets = sourceTads->primaryOffsets();
  const auto* destinationTadOffsets = destinationTads->primaryOffsets();

  const auto tadLength = shape::length(sourceTadShapeInfo);
  const auto sourceTadRank = shape::rank(sourceTadShapeInfo);
  const auto destinationTadRank = shape::rank(destinationTadShapeInfo);
  const auto* sourceTadShape = shape::shapeOf(sourceTadShapeInfo);
  const auto* destinationTadShape = shape::shapeOf(destinationTadShapeInfo);
  const auto* sourceTadStride = shape::stride(sourceTadShapeInfo);
  const auto* destinationTadStride = shape::stride(destinationTadShapeInfo);

  for (sd::LongType pullIndex = 0; pullIndex < count; ++pullIndex) {
    const auto sourceTad = indexBuffer[pullIndex];
    const auto sourceBase = sourceTadOffsets[sourceTad];
    const auto destinationBase = destinationTadOffsets[pullIndex];

    for (sd::LongType element = 0; element < tadLength; ++element) {
      sd::LongType sourceCoordinates[SD_MAX_RANK];
      sd::LongType destinationCoordinates[SD_MAX_RANK];
      sd::LongType sourceOffset;
      sd::LongType destinationOffset;

      INDEX2COORDS(element, sourceTadRank, sourceTadShape,
                   sourceCoordinates);
      COORDS2INDEX(sourceTadRank, sourceTadStride, sourceCoordinates,
                   sourceOffset);
      INDEX2COORDS(element, destinationTadRank, destinationTadShape,
                   destinationCoordinates);
      COORDS2INDEX(destinationTadRank, destinationTadStride,
                   destinationCoordinates, destinationOffset);

      destinationBuffer[destinationBase + destinationOffset] =
          sourceBuffer[sourceBase + sourceOffset];
    }
  }
}

template <typename T>
void shuffleRankOne(NDArray* source, const int* shuffleMap) {
  auto* sourceBuffer = source->bufferAsT<T>();
  const auto* sourceShapeInfo = source->shapeInfo();
  const auto* sourceShape = shape::shapeOf(sourceShapeInfo);
  const auto* sourceStride = shape::stride(sourceShapeInfo);
  const auto sourceRank = shape::rank(sourceShapeInfo);

  for (sd::LongType element = 0; element < source->lengthOf(); ++element) {
    const auto target = shuffleMap[element];
    if (target < 0) continue;

    sd::LongType sourceCoordinates[SD_MAX_RANK];
    sd::LongType targetCoordinates[SD_MAX_RANK];
    sd::LongType sourceOffset;
    sd::LongType targetOffset;

    INDEX2COORDS(element, sourceRank, sourceShape, sourceCoordinates);
    COORDS2INDEX(sourceRank, sourceStride, sourceCoordinates, sourceOffset);
    INDEX2COORDS(target, sourceRank, sourceShape, targetCoordinates);
    COORDS2INDEX(sourceRank, sourceStride, targetCoordinates, targetOffset);

    const T oldValue = sourceBuffer[sourceOffset];
    sourceBuffer[sourceOffset] = sourceBuffer[targetOffset];
    sourceBuffer[targetOffset] = oldValue;
  }
}

template <typename T>
void shuffleTadsFromSource(
    NDArray* source, NDArray* destination, const int* shuffleMap,
    const std::shared_ptr<sd::TadPack>& sourceTads,
    const std::shared_ptr<sd::TadPack>& destinationTads) {
  const auto* sourceBuffer = source->bufferAsT<T>();
  auto* destinationBuffer = destination->bufferAsT<T>();

  const auto* sourceTadShapeInfo = sourceTads->primaryShapeInfo();
  const auto* destinationTadShapeInfo = destinationTads->primaryShapeInfo();
  const auto* sourceTadOffsets = sourceTads->primaryOffsets();
  const auto* destinationTadOffsets = destinationTads->primaryOffsets();

  const auto tadLength = shape::length(sourceTadShapeInfo);
  const auto sourceTadRank = shape::rank(sourceTadShapeInfo);
  const auto destinationTadRank = shape::rank(destinationTadShapeInfo);
  const auto* sourceTadShape = shape::shapeOf(sourceTadShapeInfo);
  const auto* destinationTadShape = shape::shapeOf(destinationTadShapeInfo);
  const auto* sourceTadStride = shape::stride(sourceTadShapeInfo);
  const auto* destinationTadStride = shape::stride(destinationTadShapeInfo);

  for (sd::LongType sourceTad = 0; sourceTad < sourceTads->numberOfTads();
       ++sourceTad) {
    const auto targetTad = shuffleMap[sourceTad];
    if (targetTad < 0) continue;

    const auto sourceBase = sourceTadOffsets[sourceTad];
    const auto targetSourceBase = sourceTadOffsets[targetTad];
    const auto destinationBase = destinationTadOffsets[sourceTad];
    const auto targetDestinationBase = destinationTadOffsets[targetTad];

    for (sd::LongType element = 0; element < tadLength; ++element) {
      sd::LongType sourceCoordinates[SD_MAX_RANK];
      sd::LongType destinationCoordinates[SD_MAX_RANK];
      sd::LongType sourceOffset;
      sd::LongType destinationOffset;

      INDEX2COORDS(element, sourceTadRank, sourceTadShape,
                   sourceCoordinates);
      COORDS2INDEX(sourceTadRank, sourceTadStride, sourceCoordinates,
                   sourceOffset);
      INDEX2COORDS(element, destinationTadRank, destinationTadShape,
                   destinationCoordinates);
      COORDS2INDEX(destinationTadRank, destinationTadStride,
                   destinationCoordinates, destinationOffset);

      const T oldValue = sourceBuffer[sourceBase + sourceOffset];
      destinationBuffer[destinationBase + destinationOffset] =
          sourceBuffer[targetSourceBase + sourceOffset];
      destinationBuffer[targetDestinationBase + destinationOffset] = oldValue;
    }
  }
}

template <typename T>
void shuffleIndexedTads(
    const NDArray* shuffleMap, NDArray* source, NDArray* destination,
    const std::vector<sd::LongType>& dimensions) {
  const auto* mapBuffer =
      const_cast<NDArray*>(shuffleMap)->bufferAsT<int>();

  if (source->rankOf() == 1) {
    shuffleRankOne<T>(source, mapBuffer);
    return;
  }

  std::vector<sd::LongType> tadDimensions(dimensions);
  auto sourceTads =
      ConstantTadHelper::getInstance().tadForDimensions(
          source->shapeInfo(), &tadDimensions);
  auto destinationTads =
      ConstantTadHelper::getInstance().tadForDimensions(
          destination->shapeInfo(), &tadDimensions);
  shuffleTadsFromSource<T>(source, destination, mapBuffer, sourceTads,
                           destinationTads);
}

}  // namespace

CUSTOM_OP_IMPL(legacy_pull_rows, 2, 1, false, 0, 2) {
  auto* source = INPUT_VARIABLE(0);
  auto* indexes = INPUT_VARIABLE(1);
  auto* destination = OUTPUT_VARIABLE(0);
  const auto count = INT_ARG(0);
  const auto dimension = INT_ARG(1);

  REQUIRE_TRUE(count >= 0, 0, "legacy_pull_rows: count must be non-negative");
  REQUIRE_TRUE(indexes->dataType() == DataType::INT64, 0,
               "legacy_pull_rows: indexes must use INT64 storage");
  REQUIRE_TRUE(indexes->lengthOf() >= count, 0,
               "legacy_pull_rows: indexes contain fewer than %lld entries",
               static_cast<long long>(count));
  REQUIRE_TRUE(source->dataType() == destination->dataType(), 0,
               "legacy_pull_rows: source and destination data types differ");
  REQUIRE_TRUE(source->rankOf() == 1 || source->rankOf() == 2, 0,
               "legacy_pull_rows: source rank must be 1 or 2");
  REQUIRE_TRUE(destination->rankOf() == source->rankOf(), 0,
               "legacy_pull_rows: source and destination ranks differ");
  REQUIRE_TRUE(dimension >= 0 && dimension < source->rankOf(), 0,
               "legacy_pull_rows: dimension %lld is outside source rank %i",
               static_cast<long long>(dimension), source->rankOf());

  const auto* indexBuffer = const_cast<NDArray*>(indexes)->bufferAsT<sd::LongType>();
  if (source->rankOf() == 1) {
    REQUIRE_TRUE(destination->lengthOf() == count, 0,
                 "legacy_pull_rows: rank-1 destination length must equal count");
    for (sd::LongType index = 0; index < count; ++index) {
      REQUIRE_TRUE(indexBuffer[index] >= 0 &&
                       indexBuffer[index] < source->lengthOf(),
                   0, "legacy_pull_rows: source element index %lld is out of range",
                   static_cast<long long>(indexBuffer[index]));
    }

    BUILD_SINGLE_SELECTOR(
        source->dataType(), pullIndexedElements,
        (source, destination, indexes, count), SD_COMMON_TYPES);
    return Status::OK;
  }

  std::vector<sd::LongType> tadDimensions{dimension};
  auto sourceTads = ConstantTadHelper::getInstance().tadForDimensions(
      source->shapeInfo(), &tadDimensions);
  auto destinationTads = ConstantTadHelper::getInstance().tadForDimensions(
      destination->shapeInfo(), &tadDimensions);

  REQUIRE_TRUE(sameLogicalShape(sourceTads->primaryShapeInfo(),
                                destinationTads->primaryShapeInfo()),
               0, "legacy_pull_rows: source and destination TAD shapes differ");
  REQUIRE_TRUE(destinationTads->numberOfTads() == count, 0,
               "legacy_pull_rows: destination TAD count must equal %lld",
               static_cast<long long>(count));

  for (sd::LongType index = 0; index < count; ++index) {
    REQUIRE_TRUE(indexBuffer[index] >= 0 &&
                     indexBuffer[index] < sourceTads->numberOfTads(),
                 0, "legacy_pull_rows: source TAD index %lld is out of range",
                 static_cast<long long>(indexBuffer[index]));
  }

  BUILD_SINGLE_SELECTOR(
      source->dataType(), pullIndexedTads,
      (source, destination, indexes, count, sourceTads, destinationTads),
      SD_COMMON_TYPES);
  return Status::OK;
}

DECLARE_TYPES(legacy_pull_rows) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_INTS, ALL_FLOATS})
      ->setAllowedInputTypes(1, DataType::INT64)
      ->setAllowedOutputTypes(0, {ALL_INTS, ALL_FLOATS})
      ->setSameMode(false);
  getOpDescriptor()->addTraits(OP_TRAIT_DATA_MOVEMENT |
                               OP_TRAIT_GATHER |
                               OP_TRAIT_FULLY_WRITING |
                               OP_TRAIT_DATA_DEPENDENT);
}

DECLARE_SHAPE_FN(legacy_pull_rows) {
  REQUIRE_TRUE(inputShape != nullptr && inputShape->size() == 2, 0,
               "legacy_pull_rows shape inference requires two inputs");
  const auto* arguments = block.getIArguments();
  REQUIRE_TRUE(arguments != nullptr && arguments->size() == 2, 0,
               "legacy_pull_rows shape inference requires count and dimension");

  const auto* sourceShapeInfo = inputShape->at(0);
  const auto count = arguments->at(0);
  const auto dimension = arguments->at(1);
  const auto rank = shape::rank(sourceShapeInfo);

  REQUIRE_TRUE(count >= 0, 0,
               "legacy_pull_rows shape inference requires non-negative count");
  REQUIRE_TRUE(rank == 1 || rank == 2, 0,
               "legacy_pull_rows shape inference requires rank 1 or 2");
  REQUIRE_TRUE(dimension >= 0 && dimension < rank, 0,
               "legacy_pull_rows shape inference received an invalid dimension");

  std::vector<sd::LongType> outputShape;
  if (rank == 1) {
    outputShape = {count};
  } else if (dimension == 1) {
    outputShape = {count, shape::sizeAt(sourceShapeInfo, 1)};
  } else {
    outputShape = {shape::sizeAt(sourceShapeInfo, 0), count};
  }

  return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
      ArrayOptions::dataType(sourceShapeInfo), shape::order(sourceShapeInfo),
      outputShape));
}

CUSTOM_OP_IMPL(legacy_shuffle, -1, -1, true, 0, -1) {
  const auto* integerArguments = block.getIArguments();
  REQUIRE_TRUE(integerArguments != nullptr && !integerArguments->empty(), 0,
               "legacy_shuffle: array count argument is required");
  const auto& arguments = *integerArguments;

  const auto arrayCount = arguments[0];
  REQUIRE_TRUE(arrayCount > 0, 0,
               "legacy_shuffle: array count must be positive");
  REQUIRE_TRUE(block.width() == static_cast<size_t>(arrayCount + 1), 0,
               "legacy_shuffle: expected map plus %lld source arrays",
               static_cast<long long>(arrayCount));

  auto* shuffleMap = INPUT_VARIABLE(static_cast<int>(arrayCount));
  REQUIRE_TRUE(shuffleMap->dataType() == DataType::INT32, 0,
               "legacy_shuffle: shuffle map must use INT32 storage");

  size_t argumentOffset = 1;
  const auto* mapBuffer =
      const_cast<NDArray*>(shuffleMap)->bufferAsT<int>();

  for (sd::LongType arrayIndex = 0; arrayIndex < arrayCount; ++arrayIndex) {
    REQUIRE_TRUE(argumentOffset < arguments.size(), 0,
                 "legacy_shuffle: missing dimension count for array %lld",
                 static_cast<long long>(arrayIndex));
    const auto dimensionCount = arguments[argumentOffset++];
    REQUIRE_TRUE(dimensionCount > 0, 0,
                 "legacy_shuffle: dimensions must not be empty");
    REQUIRE_TRUE(argumentOffset + static_cast<size_t>(dimensionCount) <=
                     arguments.size(),
                 0, "legacy_shuffle: truncated dimension vector");

    std::vector<sd::LongType> arrayDimensions(
        arguments.begin() + static_cast<ptrdiff_t>(argumentOffset),
        arguments.begin() +
            static_cast<ptrdiff_t>(argumentOffset + dimensionCount));
    argumentOffset += static_cast<size_t>(dimensionCount);

    auto* source = INPUT_VARIABLE(static_cast<int>(arrayIndex));
    auto* destination = OUTPUT_VARIABLE(static_cast<int>(arrayIndex));
    REQUIRE_TRUE(source->dataType() == destination->dataType(), 0,
                 "legacy_shuffle: source and destination data types differ");
    REQUIRE_TRUE(source->rankOf() == destination->rankOf(), 0,
                 "legacy_shuffle: source and destination ranks differ");
    requireDimensionVector(source, arrayDimensions, "legacy_shuffle");
    requireDimensionVector(destination, arrayDimensions, "legacy_shuffle");

    sd::LongType itemCount;
    if (source->rankOf() == 1) {
      REQUIRE_TRUE(source->lengthOf() == destination->lengthOf(), 0,
                   "legacy_shuffle: rank-1 source and destination lengths differ");
      itemCount = source->lengthOf();
    } else {
      auto sourceTads = ConstantTadHelper::getInstance().tadForDimensions(
          source->shapeInfo(), &arrayDimensions);
      auto destinationTads =
          ConstantTadHelper::getInstance().tadForDimensions(
              destination->shapeInfo(), &arrayDimensions);
      REQUIRE_TRUE(sameLogicalShape(sourceTads->primaryShapeInfo(),
                                    destinationTads->primaryShapeInfo()),
                   0, "legacy_shuffle: source and destination TAD shapes differ");
      REQUIRE_TRUE(sourceTads->numberOfTads() ==
                       destinationTads->numberOfTads(),
                   0, "legacy_shuffle: source and destination TAD counts differ");
      itemCount = sourceTads->numberOfTads();
    }

    REQUIRE_TRUE(shuffleMap->lengthOf() >= itemCount, 0,
                 "legacy_shuffle: map is shorter than the TAD count");
    for (sd::LongType item = 0; item < itemCount; ++item) {
      REQUIRE_TRUE(mapBuffer[item] < 0 || mapBuffer[item] < itemCount, 0,
                   "legacy_shuffle: target %i is outside the TAD range",
                   mapBuffer[item]);
    }

    BUILD_SINGLE_SELECTOR(
        source->dataType(), shuffleIndexedTads,
        (shuffleMap, source, destination, arrayDimensions),
        SD_COMMON_TYPES);
  }

  REQUIRE_TRUE(argumentOffset == arguments.size(), 0,
               "legacy_shuffle: unexpected trailing dimension arguments");
  return Status::OK;
}

DECLARE_TYPES(legacy_shuffle) {
  getOpDescriptor()
      ->setAllowedInputTypes({ALL_INTS, ALL_FLOATS})
      ->setAllowedOutputTypes({ALL_INTS, ALL_FLOATS})
      ->setSameMode(false);
  getOpDescriptor()->addTraits(OP_TRAIT_DATA_MOVEMENT |
                               OP_TRAIT_DATA_DEPENDENT);
}

DECLARE_SHAPE_FN(legacy_shuffle) {
  const auto* arguments = block.getIArguments();
  REQUIRE_TRUE(arguments != nullptr && !arguments->empty(), 0,
               "legacy_shuffle shape inference requires encoded dimensions");

  const auto arrayCount = arguments->at(0);
  REQUIRE_TRUE(arrayCount > 0, 0,
               "legacy_shuffle shape inference requires a positive array count");
  REQUIRE_TRUE(inputShape != nullptr &&
                   inputShape->size() == static_cast<size_t>(arrayCount + 1),
               0, "legacy_shuffle shape inference received the wrong input count");

  size_t argumentOffset = 1;
  for (sd::LongType arrayIndex = 0; arrayIndex < arrayCount; ++arrayIndex) {
    REQUIRE_TRUE(argumentOffset < arguments->size(), 0,
                 "legacy_shuffle shape inference is missing a dimension count");
    const auto dimensionCount = arguments->at(argumentOffset++);
    REQUIRE_TRUE(dimensionCount > 0, 0,
                 "legacy_shuffle shape inference requires non-empty dimensions");
    REQUIRE_TRUE(argumentOffset + static_cast<size_t>(dimensionCount) <=
                     arguments->size(),
                 0, "legacy_shuffle shape inference has truncated dimensions");
    argumentOffset += static_cast<size_t>(dimensionCount);
  }
  REQUIRE_TRUE(argumentOffset == arguments->size(), 0,
               "legacy_shuffle shape inference has trailing dimensions");

  auto result = SHAPELIST();
  for (sd::LongType arrayIndex = 0; arrayIndex < arrayCount; ++arrayIndex) {
    result->push_back(ConstantShapeHelper::getInstance().createFromExisting(
        inputShape->at(static_cast<size_t>(arrayIndex))));
  }
  return result;
}

}  // namespace ops
}  // namespace sd
