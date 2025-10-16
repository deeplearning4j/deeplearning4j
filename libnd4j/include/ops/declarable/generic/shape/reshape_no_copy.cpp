//
// Created by agibsonccc on 8/30/24.
//

#include <helpers/reshapeNoCopy.h>
#include <helpers/shape.h>
#include <ops/declarable/headers/shape.h>
namespace sd {
namespace ops {
CUSTOM_OP_IMPL(reshape_no_copy, -2, 1, false, 0, -2) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);
  //note that the calculate output shape that sets this flag does not have access to the data buffer
  if (ArrayOptions::arrayNeedsCopy(const_cast<LongType *>(output->shapeInfo()))
      || output->dataBuffer() != input->dataBuffer()) {
    //immitate a reshape operation but without triggering a copy. These helpers are to prevent stack overflows with reshape -> assign -> reshape which used to exist
    auto* inputShape = input->getShapeAsVector();
    sd::LongType  *shapeInfo = NDArray::reshapeShapeInfo(output, output->ordering(), *inputShape);
    delete inputShape;
    NDArray::copyDataForAssign(input, output, shapeInfo, false);
  }
  // the rest is no op, we don't need to copy we just needed the new shape

  return Status::OK;
}

DECLARE_SHAPE_FN(reshape_no_copy) {
  auto inShape = inputShape->at(0);
  if (ArrayOptions::dataType(inShape) == UNKNOWN) {
    THROW_EXCEPTION("Illegal data type set for reshape: UNKNOWN");
  }

  DataType dtype = ArrayOptions::dataType(inShape);
  char order = shape::order(inShape);  // Default to input order
  std::vector<sd::LongType> newShape;

  if (block.width() > 1) {
    auto shapeArg = INPUT_VARIABLE(1);
    auto shapeBuffLong = shapeArg->getBufferAsVector<sd::LongType>();
    // last is the ordering
    for (size_t i = 0; i < shapeBuffLong.size() - 1; i++) {
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
    for (size_t i = 0; i < block.numI() - 1; i++) {
      newShape.push_back(iArgs->at(i));
    }
    order = iArgs->at(iArgs->size() - 1) == RESHAPE_NO_COPY_F_ORDER_MARKER ? 'f' : 'c';
  }

  // Handle -1 in shape specification
  sd::LongType negativeOneCount = 0;
  sd::LongType negativeOneIndex = -1;
  sd::LongType totalElements = shape::length(inShape);
  sd::LongType knownDimProduct = 1;
  
  // Count -1s and calculate product of known dimensions
  for (size_t i = 0; i < newShape.size(); i++) {
    if (newShape[i] == -1) {
      negativeOneCount++;
      negativeOneIndex = i;
    } else if (newShape[i] <= 0) {
      std::string errorMessage = "Shape value is invalid: ";
      errorMessage += std::to_string(newShape[i]);
      errorMessage += " at index ";
      errorMessage += std::to_string(i);
      errorMessage += " in shape ";
      errorMessage += std::to_string(newShape.size());
      THROW_EXCEPTION(errorMessage.c_str());
    } else {
      knownDimProduct *= newShape[i];
    }
  }
  
  // Validate -1 usage
  if (negativeOneCount > 1) {
    THROW_EXCEPTION("Only one dimension can be -1 in reshape operation");
  }
  
  // Calculate the -1 dimension if present
  if (negativeOneCount == 1) {
    if (totalElements % knownDimProduct != 0) {
      std::string errorMessage = "Cannot reshape array of size ";
      errorMessage += std::to_string(totalElements);
      errorMessage += " into shape with known dimensions product ";
      errorMessage += std::to_string(knownDimProduct);
      THROW_EXCEPTION(errorMessage.c_str());
    }
    newShape[negativeOneIndex] = totalElements / knownDimProduct;
  }

  sd::LongType len = shape::shapeInfoLength(newShape.size());
  sd::LongType *newShapeInfo = new sd::LongType[len];
  newShapeInfo[0] = newShape.size();
  shape::setShape(newShapeInfo, newShape.data());
  shape::setOrder(newShapeInfo, order);
  auto newShapeView = shape::shapeOf(newShapeInfo);

  for (size_t i = 0; i < newShape.size(); i++) {
    if (newShape[i] != newShapeView[i]) {
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
    shape::setShape(newShapeInfo, newShape.data());
    // If reshape is not possible without allocation, fall back to regular reshape
    shape::updateStrides(newShapeInfo, order, true);
    ArrayOptions::resetFlags(newShapeInfo);
    ArrayOptions::setDataType(newShapeInfo, dtype);
    ArrayOptions::toggleIsEmpty(newShapeInfo);
  } else {
    bool reshapeNoAllocSuccess = helpers::reshapeNoAlloc(inShape, newShape, order, newShapeInfo);
    if (!reshapeNoAllocSuccess || shape::order(inShape) != order) {
      //we need new strides if we can't handle the copy
      shape::updateStrides(newShapeInfo, order, true);
      ArrayOptions::resetFlags(newShapeInfo);
      ArrayOptions::setDataType(newShapeInfo, dtype);
      //ensure we trigger a proper data copy
      ArrayOptions::togglePropertyBit(newShapeInfo, ARRAY_NEEDS_COPY);
    } else {
      //we set strides in the reshape alloc success already
      newShapeInfo[0] = newShape.size();
      shape::setShape(newShapeInfo, newShape.data());
      ArrayOptions::resetFlags(newShapeInfo);
      // we need this in order to preserve the offset of the original buffer when creating the output array
      ArrayOptions::togglePropertyBit(newShapeInfo, ARRAY_COPY_OFFSET_INPUT_0);
      ArrayOptions::setDataType(newShapeInfo, dtype);
    }
  }


  auto newShape2 = ConstantShapeHelper::getInstance().createFromExisting(newShapeInfo);
  delete[] newShapeInfo;
  return SHAPELIST(CONSTANT(newShape2));
}

DECLARE_TYPES(reshape_no_copy) {
  getOpDescriptor()
      ->setAllowedInputTypes(sd::DataType::ANY)
      ->setAllowedOutputTypes(sd::DataType::ANY)
      ->setSameMode(true);
}
}
}
