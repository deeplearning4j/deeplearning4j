#include <array/TadCalculator.h>
#include <helpers/ShapeUtils.h>
#include <helpers/ConstantHelper.h>
#include <helpers/ConstantShapeHelper.h>

namespace sd {

TadCalculator::TadCalculator(LongType* originalShape)
    : _originalShape(originalShape), _numTads(0), _tadShape(nullptr), _tadOffsets(nullptr) {}

TadCalculator::~TadCalculator() {}

void TadCalculator::createTadPack(const std::vector<LongType>& dimensions) {
  if (!_originalShape) {
    THROW_EXCEPTION("Original shape is null");
  }

  // Check for empty array
  if (shape::isEmptyConst(_originalShape)) {
    THROW_EXCEPTION("Cannot create TADs for empty array");
  }

  auto shapeInfo = ConstantShapeHelper::getInstance().createFromExisting(_originalShape);
  const LongType rank = shape::rank(shapeInfo);

  // Check for zero-sized dimensions
  for (LongType i = 0; i < rank; i++) {
    if (shape::sizeAt(shapeInfo, i) == 0) {
      THROW_EXCEPTION("Cannot create TADs for array with zero-sized dimensions");
    }
  }

  const std::vector<LongType>* dimsToExclude = ShapeUtils::evalDimsToExclude(rank, dimensions.size(), dimensions.data());
  if (!dimsToExclude) {
    THROW_EXCEPTION("Failed to evaluate dimensions to exclude");
  }

  if (dimsToExclude->size() == 0 || dimsToExclude->size() == rank) {
    const LongType totalElements = shape::length(shapeInfo);
    
    auto scalarShapeInfo = ConstantShapeHelper::getInstance().scalarShapeInfo(ArrayOptions::dataType(shapeInfo));
    auto scalarShapeBuffer = ConstantShapeHelper::getInstance().bufferForShapeInfo(scalarShapeInfo);
    
    auto oPtr = std::make_shared<PointerWrapper>(new LongType[totalElements]);
    LongType* offsets = oPtr->pointerAsT<LongType>();
    
    for (LongType i = 0; i < totalElements; ++i) {
      offsets[i] = i;
    }
    
    _tadShape = scalarShapeBuffer;
    _tadOffsets = new ConstantOffsetsBuffer(oPtr);
    _numTads = totalElements;
    
    delete dimsToExclude;
    return;
  }

  const LongType numOfSubArrs = ShapeUtils::getNumOfSubArrs(shapeInfo, *dimsToExclude);

  if (numOfSubArrs > 0) {
    const LongType subArrRank = (static_cast<size_t>(rank) == dimsToExclude->size() || false) ? rank : rank - dimsToExclude->size();

    auto sPtr = std::make_shared<PointerWrapper>(new LongType[shape::shapeInfoLength(subArrRank)]);
    auto oPtr = std::make_shared<PointerWrapper>(new LongType[numOfSubArrs]);

    shape::calcSubArrsShapeInfoAndOffsets(
        shapeInfo,
        numOfSubArrs,
        dimsToExclude->size(),
        dimsToExclude->data(),
        sPtr->pointerAsT<LongType>(),
        oPtr->pointerAsT<LongType>(),
        false);

    auto shapesBuffer = ConstantShapeHelper::getInstance().bufferForShapeInfo(sPtr->pointerAsT<LongType>());

    _tadOffsets = new ConstantOffsetsBuffer(oPtr);
    _tadShape = shapesBuffer;
    _numTads = numOfSubArrs;
  } else {
    const LongType subArrRank = rank;

    auto sPtr = std::make_shared<PointerWrapper>(new LongType[shape::shapeInfoLength(subArrRank)]);
    LongType* shapeInfo2 = sPtr->pointerAsT<LongType>();

    auto nonConstant = const_cast<LongType*>(shapeInfo);
    auto nonConst2 = const_cast<LongType*>(shapeInfo2);
    shape::copyTo<LongType>(shape::shapeInfoLength(subArrRank), nonConstant, nonConst2);

    LongType* baseOffset = new LongType[1];
    baseOffset[0] = 0;
    auto oPtr = std::make_shared<PointerWrapper>(baseOffset);

    auto shapesBuffer = ConstantShapeHelper::getInstance().bufferForShapeInfo(sPtr->pointerAsT<LongType>());
    _tadOffsets = new ConstantOffsetsBuffer(oPtr);
    _tadShape = shapesBuffer;
    _numTads = 1;
  }

  delete dimsToExclude;
}

} // namespace sd