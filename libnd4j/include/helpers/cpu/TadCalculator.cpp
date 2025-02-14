#include <array/TadCalculator.h>
#include <helpers/ShapeUtils.h>
#include <helpers/ConstantHelper.h>
#include <helpers/ConstantShapeHelper.h>

namespace sd {

TadCalculator::TadCalculator(LongType* originalShape)
    : _originalShape(originalShape), _numTads(0) {}

void TadCalculator::createTadPack(const std::vector<LongType>& dimensions) {
  // Validate input and create shape info from original shape
  if (!_originalShape) {
    THROW_EXCEPTION("Original shape is null");
  }

  auto shapeInfo = ConstantShapeHelper::getInstance().createFromExisting(_originalShape,false);
  const LongType rank = shape::rank(shapeInfo);

  // Calculate dimensions to exclude
  const std::vector<LongType>* dimsToExclude = ShapeUtils::evalDimsToExclude(rank, dimensions.size(), dimensions.data());
  if (!dimsToExclude) {
    THROW_EXCEPTION("Failed to evaluate dimensions to exclude");
  }

  // Calculate number of sub-arrays
  const LongType numOfSubArrs = ShapeUtils::getNumOfSubArrs(shapeInfo, *dimsToExclude);

  if (numOfSubArrs > 0) {
    // Calculate sub-array rank
    const LongType subArrRank = (static_cast<size_t>(rank) == dimsToExclude->size() || false) ? rank : rank - dimsToExclude->size();

    // Allocate memory for shapes and offsets
    auto sPtr = std::make_shared<PointerWrapper>(new LongType[shape::shapeInfoLength(subArrRank)]);
    auto oPtr = std::make_shared<PointerWrapper>(new LongType[numOfSubArrs]);

    // Calculate shapes and offsets
    shape::calcSubArrsShapeInfoAndOffsets(
        shapeInfo,
        numOfSubArrs,
        dimsToExclude->size(),
        dimsToExclude->data(),
        sPtr->pointerAsT<LongType>(),
        oPtr->pointerAsT<LongType>(),
        false);  // areUnitiesInShape

    // Create shape buffer
    auto shapesBuffer = ConstantShapeHelper::getInstance().bufferForShapeInfo(sPtr->pointerAsT<LongType>());

    // Create offsets buffer
    _tadOffsets = ConstantOffsetsBuffer(oPtr);
    _tadShape = *shapesBuffer;
    _numTads = numOfSubArrs;
  } else {
    // Base case: number of sub arrays is zero, use original shape
    const LongType subArrRank = rank;

    // Allocate and copy shape info
    auto sPtr = std::make_shared<PointerWrapper>(new LongType[shape::shapeInfoLength(subArrRank)]);
    LongType* shapeInfo2 = sPtr->pointerAsT<LongType>();

    // Copy shape info
    auto nonConstant = const_cast<LongType*>(shapeInfo);
    auto nonConst2 = const_cast<LongType*>(shapeInfo2);
    shape::copyTo<LongType>(shape::shapeInfoLength(subArrRank), nonConstant, nonConst2);

    // Create base offset
    LongType* baseOffset = new LongType[1];
    baseOffset[0] = 0;
    auto oPtr = std::make_shared<PointerWrapper>(baseOffset);

    // Create buffers
    auto shapesBuffer = ConstantShapeHelper::getInstance().bufferForShapeInfo(sPtr->pointerAsT<LongType>());
    _tadOffsets = ConstantOffsetsBuffer(oPtr);
    _tadShape = *shapesBuffer;
    _numTads = 1;
  }

  delete dimsToExclude;
}

} // namespace sd