/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// Created by raver119 on 29/10/17.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_reshape)
#include <ops/declarable/headers/shape.h>
#include <cstring>
#include <cstdio>
namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
// here iArgs is a vector with (optional) negative of order as first element:
// ({-order, dim1, dim2, dim3, ...})
CUSTOM_OP_IMPL(reshape, 1, 1, false, 0, -2) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  // Fast path: if same buffer, this is just a view change - nothing to copy
  if (x->dataBuffer() == z->dataBuffer()) {
    return Status::OK;
  }

  // Special case: empty.reshape(<other empty shape>) -> return empty
  if (x->isEmpty()) {
    REQUIRE_TRUE(z->isEmpty(), 0, "Reshape: when input is empty, output must also be empty");
    return Status::OK;
  }

  const sd::LongType len = x->lengthOf();
  const sd::LongType zLen = z->lengthOf();

  // Special case: scalar/length-1 input being expanded to larger output
  // This is used in ONNX models for broadcasting scalar conditions
  if ((x->isScalar() || len == 1) && zLen > 1) {
    // Fill output with the scalar value
    z->assign(x);
    return Status::OK;
  }

  // Validate lengths match for non-scalar inputs
  if (!x->isScalar()) {
    REQUIRE_TRUE(len == zLen, 0,
                 "Reshape: lengths before and after reshape should match, but "
                 "got %i vs %i",
                 len, zLen);
  }

  // Fast path: contiguous arrays can use direct memcpy (CPU only)
  const bool xContiguous = x->ordering() == 'c' && shape::strideDescendingCAscendingF(x->shapeInfo());
  const bool zContiguous = z->ordering() == 'c' && shape::strideDescendingCAscendingF(z->shapeInfo());

#if defined(SD_CUDA)
  // GPU path: use assign which handles device buffers properly
  z->assign(x);
#else
  if (xContiguous && zContiguous) {
    // CPU path: sync to host and use memcpy
    x->syncToHost();
    std::memcpy(z->buffer(), x->buffer(), len * x->sizeOfT());
    // Mark HOST as modified and sync to DEVICE
    z->tickWriteHost();
    z->syncToDevice();
  } else {
    // Non-contiguous: use assign
    z->assign(x);
  }
#endif

  return Status::OK;
}

DECLARE_TYPES(reshape) {
  getOpDescriptor()->setAllowedInputTypes(0, ANY)->setAllowedInputTypes(1, {ALL_INTS})->setSameMode(true);
}

bool handleOptionalOrder(std::vector<LongType> &reshapeArgs, char &ordering) {
  if (reshapeArgs.size() > 0) {
    // check if any optional negative ordering value is passed
    auto optional = reshapeArgs[0];
    if (optional < 0) {
      optional = abs(optional);
      // check if passed option is allowed. (-1 -> dynamic shape)
      // in that case we will return back
      if (optional == 1) return true;
      // in this case it should obey allowed orderings
      if (optional != 'c' && optional != 'f') return false;
      reshapeArgs.erase(reshapeArgs.begin());
      // ordering was passed and ok. let's assign
      ordering = optional;
    }
  }
  // skipped
  return true;
}

LongType* handleScalarAndLength1Case(NDArray* x, std::vector<LongType>& reshapeArgs) {
  //need to handle disambiguation between empty and scalar
  if(x->isScalar() || x->lengthOf() == 1) {
    if(reshapeArgs.size() < 1) {
      return ConstantShapeHelper::getInstance().scalarShapeInfo(x->dataType());
    }

    // For scalar/length-1 input, if reshape args contain -1, replace it with 1
    std::vector<LongType> finalShape = reshapeArgs;
    for (size_t i = 0; i < finalShape.size(); i++) {
      if (finalShape[i] == -1) {
        finalShape[i] = 1;
      }
    }

    return ConstantShapeHelper::getInstance().createShapeInfo(x->dataType(), 'c', finalShape);
  }
  return nullptr;
}

void processReshapeArgs(NDArray* x, std::vector<LongType>& reshapeArgs, std::vector<LongType>& shapeNew,
                        LongType& newShapeLen, int& pos, bool& newShapeEmpty) {
  newShapeLen = 1;
  pos = -1;
  newShapeEmpty = false;

  for (size_t i = 0; i < reshapeArgs.size(); i++) {
    LongType dim = reshapeArgs[i];
    if (dim == -1) {
      REQUIRE_TRUE(pos == -1, 0, "Reshape : Only one unknown dimension (-1) is allowed.");
      pos = i;
      shapeNew.push_back(1);
    } else if (dim == 0) {
      // ONNX semantics: 0 means "copy dimension from input at same position"
      // If the position is within input's rank, copy the input dimension
      // Otherwise treat as empty/zero dimension
      if (i < static_cast<size_t>(x->rankOf())) {
        LongType inputDim = x->sizeAt(i);
        shapeNew.push_back(inputDim);
        if (inputDim == 0) {
          newShapeEmpty = true;
        } else {
          newShapeLen *= inputDim;
        }
      } else {
        // Position beyond input rank - treat as 1 (padding case)
        shapeNew.push_back(1);
      }
    } else {
      shapeNew.push_back(dim);
      newShapeLen *= dim;
    }
  }
}

void computeUnknownDimension(NDArray* x, std::vector<LongType>& shapeNew, int pos,
                             LongType newShapeLen, bool newShapeEmpty) {
  if (pos != -1) {
    LongType xLen = x->lengthOf();
    if (x->isEmpty()) {
      xLen = 1;
      // For empty shapes, calculate length considering non-zero dimensions
      for (LongType i = 0; i < x->rankOf(); ++i)  // take into account possible empty shapes
        if (x->sizeAt(i) > 0 || !newShapeEmpty) xLen *= x->sizeAt(i);
    }

    shapeNew[pos] = xLen / newShapeLen;
  }
}

LongType* handleEmptyShapeCase(NDArray* x, std::vector<LongType>& shapeNew, bool newShapeEmpty) {
  if(newShapeEmpty) {
    // shapeNew already has the computed dimensions including the -1 replacement
    // Just ensure there are no remaining negative values (shouldn't happen)
    for(size_t i = 0; i < shapeNew.size(); i++) {
      if(shapeNew[i] < 0)
        shapeNew[i] = 1;
    }
    return ConstantShapeHelper::getInstance().emptyShapeInfoWithShape(x->dataType(), shapeNew);
  }
  return nullptr;
}

DECLARE_SHAPE_FN(reshape) {
  auto x = INPUT_VARIABLE(0);
  std::vector<LongType> reshapeArgs;
  std::vector<LongType> shapeNew;
  char orderNew = 'c';
  /**
   * NOTE: The value here is negative as a flag.
   * A negative value signifies 1 of 3 values:
   * -1 -> dynamic shape
   * -99 -> c ordering
   * -102 -> f ordering
   *
   */
  if (block.width() == 1) {
    reshapeArgs = *block.getIArguments();
    if (!handleOptionalOrder(reshapeArgs, orderNew)) {
      THROW_EXCEPTION(
          "reshape:: Value passed in must be -99 or -102 for the ordering if "
          "an int array is present. -99 represents c ordering and -102 "
          "represents f ordering. This number is negative for the long array "
          "case to flag the difference between an ordering and a dimension "
          "being specified.");
    };
  } else {
    reshapeArgs = INPUT_VARIABLE(1)->getBufferAsVector<LongType>();

    // VALIDATION: Check for pointer-like values in shape array
    // This catches a corruption bug where pointer addresses are stored as shape values
    for (size_t i = 0; i < reshapeArgs.size(); i++) {
      LongType value = reshapeArgs[i];
      // Pointer values typically have high bits set (> 0x100000000000 = 256TB address space)
      // Valid shape dimensions should be much smaller (typically < 1 billion)
      if (value > 0x100000000000ULL && value < 0x8000000000000000ULL) {
        std::string errorMsg = "reshape:: Shape input array at index ";
        errorMsg += std::to_string(i);
        errorMsg += " contains a value that looks like a pointer address: ";
        errorMsg += std::to_string(value);
        errorMsg += " (hex: 0x";
        char hexBuf[32];
        snprintf(hexBuf, sizeof(hexBuf), "%lx", static_cast<unsigned long>(value));
        errorMsg += hexBuf;
        errorMsg += "). Full shape array: [";
        for (size_t j = 0; j < reshapeArgs.size(); j++) {
          if (j > 0) errorMsg += ", ";
          errorMsg += std::to_string(reshapeArgs[j]);
        }
        errorMsg += "]. This indicates the shape array's data buffer contains pointer addresses ";
        errorMsg += "instead of actual shape values. Possible causes: ";
        errorMsg += "1) Use-after-free where a scalar buffer was deallocated and memory reused, ";
        errorMsg += "2) Concat operation received corrupted scalar inputs, ";
        errorMsg += "3) Race condition during shape calculation. ";
        errorMsg += "Shape input buffer address: 0x";
        snprintf(hexBuf, sizeof(hexBuf), "%lx", reinterpret_cast<unsigned long>(INPUT_VARIABLE(1)->buffer()));
        errorMsg += hexBuf;
        THROW_EXCEPTION(errorMsg.c_str());
      }
    }

    if (block.numI() > 0) {
      // Note here that the ordering for this case can not be negative.
      // Negative is used in the long array case to be used as a flag to
      // differentiate between a 99 or 102 shaped array and
      // the ordering. You can't have a -99 or -102 shaped array.
      char potentialOrdering = (char)I_ARG(0);
      if (!handleOptionalOrder(reshapeArgs, orderNew)) {
        THROW_EXCEPTION(
            "reshape:: Value passed in must be -99 or -102 for the ordering if "
            "an int array is present. -99 represents c ordering and -102 "
            "represents f ordering. This number is negative for the long array "
            "case to flag the difference between an ordering and a dimension "
            "being specified.");
      };

      orderNew = -potentialOrdering;
    }
  }

  // Handle scalar/length 1 case
  LongType* scalarResult = handleScalarAndLength1Case(x, reshapeArgs);
  if (scalarResult != nullptr) {
    return SHAPELIST(scalarResult);
  }

  LongType newShapeLen;
  int pos;
  bool newShapeEmpty;

  // Process reshape arguments (with ONNX "0 means copy from input" support)
  processReshapeArgs(x, reshapeArgs, shapeNew, newShapeLen, pos, newShapeEmpty);

  // Compute unknown dimension if needed
  computeUnknownDimension(x, shapeNew, pos, newShapeLen, newShapeEmpty);

  // Handle empty shape case
  LongType* emptyResult = handleEmptyShapeCase(x, shapeNew, newShapeEmpty);
  if (emptyResult != nullptr) {
    return SHAPELIST(emptyResult);
  }

  auto len = shape::prodLong(shapeNew.data(), shapeNew.size());
  if(!x->isScalar() && !x->isEmpty())
  REQUIRE_TRUE(x->lengthOf() == len, 0,
               "Reshape: lengths before and after reshape should match, but "
               "got %i vs %i",
               x->lengthOf(), len);

  return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(x->dataType(), orderNew, shapeNew));
}

}  // namespace ops
}  // namespace sd

#endif
