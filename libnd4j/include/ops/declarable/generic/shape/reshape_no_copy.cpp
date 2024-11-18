//
// Created by agibsonccc on 8/30/24.
//

#include <ops/declarable/CustomOperations.h>
#include <helpers/shape.h>
#include <helpers/ShapeUtils.h>
#include <helpers/reshapeNoCopy.h>
namespace sd {
namespace  ops {
CUSTOM_OP_IMPL(reshape_no_copy, -2, 1, false, 0, -2) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  if(ArrayOptions::arrayNeedsCopy(const_cast<LongType *>(input->shapeInfo()))
     || output->dataBuffer() != input->dataBuffer()) {
    //deref avoiding copy
    NDArray &originalInput = *input;
    output->assign(originalInput);
  }
  //the rest is no op, we don't need to copy we just needed the new shape

  return Status::OK;
}

DECLARE_SHAPE_FN(reshape_no_copy) {
  auto inShape = inputShape->at(0);
  if(ArrayOptions::dataType(inShape) == UNKNOWN) {
    THROW_EXCEPTION("Illegal data type set for reshape: UNKNOWN");
  }

  DataType dtype = ArrayOptions::dataType(inShape);
  char order = shape::order(inShape);  // Default to input order
  std::vector<sd::LongType> newShape;

  if (block.width() > 1) {
    auto shapeArg = INPUT_VARIABLE(1);
    auto shapeBuffLong = shapeArg->getBufferAsVector<sd::LongType>();
    //last is the ordering
    for(int i = 0; i < shapeBuffLong.size() - 1; i++) {
      newShape.push_back(shapeBuffLong[i]);
    }

    // Handle order when shape is provided as input
    if (block.numI() > 0) {
      auto orderArg = INT_ARG(0);
      if (orderArg == RESHAPE_NO_COPY_F_ORDER_MARKER) {
        order = 'f';
      } else if (orderArg == RESHAPE_NO_COPY_C_ORDER_MARKER) {
        order = 'c';
      }
    } else {
      // Default to 'c' order if not specified
      order = 'c';
    }
  } else {
    std::vector<sd::LongType> *iArgs = block.getIArguments();
    for(LongType i = 0; i < block.numI() - 1; i++) {
      newShape.push_back(iArgs->at(i));
    }
    order = iArgs->at(iArgs->size() - 1) == RESHAPE_NO_COPY_F_ORDER_MARKER ? 'f' : 'c';
  }

  sd::LongType  len = shape::shapeInfoLength(newShape.size());
  sd::LongType *newShapeInfo = new sd::LongType[len];
  newShapeInfo[0] = newShape.size();
  shape::setShape(newShapeInfo,newShape.data());
  auto newShapeView = shape::shapeOf(newShapeInfo);

  for(int i = 0; i < newShape.size(); i++) {
    if(newShape[i] != newShapeView[i]) {
      std::string errorMessage;
      errorMessage += "Failed to set shape. ";
      errorMessage += "Shape ";
      errorMessage += std::to_string(i);
      errorMessage += ": ";
      errorMessage += std::to_string(newShape[i]);
      errorMessage += " != ";
      errorMessage += std::to_string(newShapeView[i]);
      THROW_EXCEPTION(errorMessage.c_str())
    }
  }

  if (shape::isEmptyConst(inShape)) {
    newShapeInfo[0] = newShape.size();
    shape::setShape(newShapeInfo,newShape.data());
    // If reshape is not possible without allocation, fall back to regular reshape
    shape::updateStrides(newShapeInfo, order,true);
    ArrayOptions::resetFlags(newShapeInfo);
    ArrayOptions::setDataType(newShapeInfo, dtype);
    ArrayOptions::toggleIsEmpty(newShapeInfo);
  } else {
    bool reshapeNoAllocSuccess = helpers::reshapeNoAlloc(inShape, newShape, order, newShapeInfo);
    if (!reshapeNoAllocSuccess) {
      // If reshape is not possible without allocation, fall back to regular reshape
      shape::updateStrides(newShapeInfo, order,true);
      shape::setElementWiseStride(newShapeInfo, 0);
      ArrayOptions::resetFlags(newShapeInfo);
      ArrayOptions::setDataType(newShapeInfo, dtype);
      ArrayOptions::setPropertyBit(newShapeInfo, ARRAY_NEEDS_COPY);
    } else {

      newShapeInfo[0] = newShape.size();
      shape::setElementWiseStride(newShapeInfo, 0);
      shape::setShape(newShapeInfo,newShape.data());
      ArrayOptions::resetFlags(newShapeInfo);
      //we need this in order to preserve the offset of the original buffer when creating the output array
      ArrayOptions::togglePropertyBit(newShapeInfo, ARRAY_COPY_OFFSET_INPUT_0);
      ArrayOptions::setDataType(newShapeInfo, dtype);

    }

  }

  return SHAPELIST(CONSTANT(newShapeInfo));
}

DECLARE_TYPES(reshape_no_copy) {
  getOpDescriptor()
      ->setAllowedInputTypes(sd::DataType::ANY)
      ->setAllowedOutputTypes(sd::DataType::ANY)
      ->setSameMode(true);
}
}
}