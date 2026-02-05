//
// Created by agibsonccc on 8/30/24.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_reshape_no_copy)
#include <helpers/reshapeNoCopy.h>
#include <helpers/shape.h>
#include <ops/declarable/headers/shape.h>
#include <cstring>
namespace sd {
namespace ops {
CUSTOM_OP_IMPL(reshape_no_copy, -2, 1, false, 0, -2) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  // Handle empty arrays - nothing to copy
  if (input->isEmpty() || input->lengthOf() == 0 || output->isEmpty() || output->lengthOf() == 0) {
    return Status::OK;
  }

  //note that the calculate output shape that sets this flag does not have access to the data buffer
  if (ArrayOptions::arrayNeedsCopy(const_cast<LongType *>(output->shapeInfo()))
      || output->dataBuffer() != input->dataBuffer()) {
    // Buffers differ (fresh allocation) or copy flag is set.
    // Use assign() which correctly iterates over logical elements,
    // handling non-contiguous views, different strides, and different buffer sizes.
    if (input->lengthOf() == output->lengthOf() && input->lengthOf() > 0) {
      output->assign(input);
    }
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

    if (shapeArg == nullptr) {
      THROW_EXCEPTION("reshape_no_copy: Shape argument (input 1) is null");
    }
    if (shapeArg->isEmpty()) {
      THROW_EXCEPTION("reshape_no_copy: Shape argument (input 1) is empty - cannot determine target shape");
    }
    if (shapeArg->lengthOf() == 0) {
      THROW_EXCEPTION("reshape_no_copy: Shape argument (input 1) has length 0 - cannot determine target shape");
    }

    auto shapeBuffLong = shapeArg->getBufferAsVector<sd::LongType>();

    // Validate that we actually got shape values
    if (shapeBuffLong.empty()) {
      std::string errorMsg = "reshape_no_copy: Shape argument returned empty vector. ";
      errorMsg += "Shape arg rank: " + std::to_string(shapeArg->rankOf());
      errorMsg += ", length: " + std::to_string(shapeArg->lengthOf());
      errorMsg += ", isEmpty: " + std::to_string(shapeArg->isEmpty());
      THROW_EXCEPTION(errorMsg.c_str());
    }

    // Copy all elements from the shape array - order is passed separately via iArgs
    for (size_t i = 0; i < shapeBuffLong.size(); i++) {
      newShape.push_back(shapeBuffLong[i]);
    }

    // Handle order when shape is provided as input - order comes from iArgs, not from shape array
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
    // Need at least 1 integer argument (the order marker) to proceed
    if (block.numI() == 0) {
      THROW_EXCEPTION("reshape_no_copy: No shape arguments provided. Need at least shape dimensions and order marker.");
    }
    // All but the last argument are shape dimensions; the last is the order marker
    if (block.numI() > 1) {
      for (size_t i = 0; i < block.numI() - 1; i++) {
        newShape.push_back(iArgs->at(i));
      }
    }
    order = iArgs->at(iArgs->size() - 1) == RESHAPE_NO_COPY_F_ORDER_MARKER ? 'f' : 'c';
  }

  // Handle empty newShape (rank-0 / scalar output)
  // This is a VALID use case when:
  // 1. Input has exactly 1 element (reshape to scalar)
  // 2. Called from noOp() in reduce operations
  // It's INVALID when:
  // 1. Input has more than 1 element (cannot reshape to scalar without losing data)
  // 2. Shape argument was corrupted
  if (newShape.empty()) {
    sd::LongType inputLength = shape::length(inShape);
    if (inputLength == 1) {
      // Valid: reshape single element to scalar
      // Create a rank-0 (scalar) shape info
      sd::LongType len = shape::shapeInfoLength(static_cast<sd::LongType>(0));  // rank 0
      // Zero-initialize to prevent uninitialized memory errors in DirectShapeTrie comparisons
      sd::LongType *newShapeInfo = new sd::LongType[len]();  // value-initialized to zero
      newShapeInfo[0] = 0;  // rank = 0
      shape::setOrder(newShapeInfo, order);
      ArrayOptions::resetFlags(newShapeInfo);
      ArrayOptions::setDataType(newShapeInfo, dtype);
      // Copy strides are not needed for scalars, but we need proper offset handling
      ArrayOptions::togglePropertyBit(newShapeInfo, ARRAY_COPY_OFFSET_INPUT_0);

      auto newShape2 = ConstantShapeHelper::getInstance().createFromExisting(newShapeInfo);
      delete[] newShapeInfo;
      return SHAPELIST(CONSTANT(newShape2));
    } else {
      // Invalid: cannot reshape multi-element array to scalar
      std::string errorMsg = "reshape_no_copy: Target shape is empty (rank-0) but input has ";
      errorMsg += std::to_string(inputLength);
      errorMsg += " elements. Cannot reshape to scalar without losing data. ";
      errorMsg += "Input shape rank: " + std::to_string(shape::rank(inShape));
      errorMsg += ", block.width: " + std::to_string(block.width());
      errorMsg += ", block.numI: " + std::to_string(block.numI());
      THROW_EXCEPTION(errorMsg.c_str());
    }
  }

  // Handle -1 and 0 in shape specification
  // ONNX semantics: 0 means "copy dimension from input at same position"
  sd::LongType negativeOneCount = 0;
  sd::LongType negativeOneIndex = -1;
  sd::LongType totalElements = shape::length(inShape);
  sd::LongType knownDimProduct = 1;
  sd::LongType inputRank = shape::rank(inShape);

  // Count -1s and calculate product of known dimensions
  // Handle 0 = copy from input (ONNX semantics)
  bool hasZeroDim = false;
  for (size_t i = 0; i < newShape.size(); i++) {
    if (newShape[i] == -1) {
      negativeOneCount++;
      negativeOneIndex = i;
    } else if (newShape[i] < 0) {
      std::string errorMessage = "Shape value is invalid: ";
      errorMessage += std::to_string(newShape[i]);
      errorMessage += " at index ";
      errorMessage += std::to_string(i);
      errorMessage += " in shape ";
      errorMessage += std::to_string(newShape.size());
      THROW_EXCEPTION(errorMessage.c_str());
    } else if (newShape[i] == 0) {
      // ONNX semantics: 0 means "copy dimension from input at same position"
      if (i < static_cast<size_t>(inputRank)) {
        sd::LongType inputDim = shape::shapeOf(inShape)[i];
        newShape[i] = inputDim;
        if (inputDim == 0) {
          hasZeroDim = true;
        } else {
          knownDimProduct *= inputDim;
        }
      } else {
        // Position beyond input rank - treat as 1 (padding case)
        newShape[i] = 1;
        knownDimProduct *= 1;
      }
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
    if (hasZeroDim || totalElements == 0) {
      // For empty arrays, set -1 dimension to 0 or 1 depending on context
      newShape[negativeOneIndex] = (totalElements == 0) ? 0 : 1;
    } else if (totalElements % knownDimProduct != 0) {
      std::string errorMessage = "Cannot reshape array of size ";
      errorMessage += std::to_string(totalElements);
      errorMessage += " into shape with known dimensions product ";
      errorMessage += std::to_string(knownDimProduct);
      THROW_EXCEPTION(errorMessage.c_str());
    } else {
      newShape[negativeOneIndex] = totalElements / knownDimProduct;
    }
  }

  // Validate that the target shape's total element count matches the input's element count.
  // This catches bugs where wrong dimensions are passed (e.g., from Java char-widening
  // or incorrect shape calculations), which would create a view with more elements than
  // the underlying buffer, causing heap corruption when ops read/write past the end.
  if (!hasZeroDim && totalElements > 0 && negativeOneCount == 0) {
    sd::LongType targetElements = 1;
    for (size_t i = 0; i < newShape.size(); i++) {
      targetElements *= newShape[i];
    }
    if (targetElements != totalElements) {
      std::string errorMessage = "reshape_no_copy: Cannot reshape array of ";
      errorMessage += std::to_string(totalElements);
      errorMessage += " elements into shape [";
      for (size_t i = 0; i < newShape.size(); i++) {
        if (i > 0) errorMessage += ", ";
        errorMessage += std::to_string(newShape[i]);
      }
      errorMessage += "] (";
      errorMessage += std::to_string(targetElements);
      errorMessage += " elements). Element counts must match.";
      THROW_EXCEPTION(errorMessage.c_str());
    }
  }

  sd::LongType len = shape::shapeInfoLength(newShape.size());
  // Zero-initialize to prevent uninitialized memory errors in DirectShapeTrie comparisons
  sd::LongType *newShapeInfo = new sd::LongType[len]();  // value-initialized to zero
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

  // Handle empty output (either from empty input or zero dimension in shape)
  if (shape::isEmptyConst(inShape) || hasZeroDim) {
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
}  // namespace ops
}  // namespace sd

#endif
