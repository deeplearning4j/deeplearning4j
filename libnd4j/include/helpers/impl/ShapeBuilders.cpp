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
// @author raver119@gmail.com
//
#include <helpers/ShapeBuilders.h>
#include <system/env_functions.h>
#include <cstring>

#include "array/ShapeDescriptor.h"

namespace sd {

// Use the global padding constant from common.h
static constexpr LongType SHAPE_ALLOC_PADDING = SD_SHAPE_ALLOC_PADDING;

LongType* ShapeBuilders::createShapeInfoFrom(ShapeDescriptor* descriptor) {
  LongType bufferLen = shape::shapeInfoLength(descriptor->rank());
  auto ret = new LongType[bufferLen + SHAPE_ALLOC_PADDING];
  // Initialize shape info portion to zero to avoid uninitialized memory issues
  memset(ret, 0, bufferLen * sizeof(LongType));
  // Initialize guard bytes to known pattern to detect buffer overflows
  if (sd::env_isDebug()) {
    uint8_t* guardBytes = reinterpret_cast<uint8_t*>(ret) + (bufferLen * sizeof(LongType));
    memset(guardBytes, 0xAB, SHAPE_ALLOC_PADDING);
  }
  ret[0] = descriptor->rank();
  if(descriptor->rank() > 0) {
    shape::setShape(ret, descriptor->shape_strides());
    shape::setStrideConst(ret, descriptor->stridesPtr());
    shape::setOrder(ret, descriptor->order());
  } else {
    std::vector<LongType> shape = {0};
    std::vector<LongType> strides = {1};
    shape::setShape(ret,shape.data());
    shape::setStrideConst(ret, strides.data());
    shape::setOrder(ret,'c');
  }

  shape::setExtra(ret, descriptor->extra());
  if(ArrayOptions::dataType(ret) != descriptor->dataType()) {
    ArrayOptions::setDataType(ret, descriptor->dataType());
  }
  return ret;
}

LongType* ShapeBuilders::createScalarShapeInfo(const DataType dataType, memory::Workspace* workspace) {
  // there is no reason for shape info to use workspaces. we have constant shape helper for this
  // workspaces with shapebuffers also appears to cause issues when reused elsewhere.
  LongType lenOfShapeInfo = 6;
  auto newShape = new LongType[lenOfShapeInfo + SHAPE_ALLOC_PADDING];
  newShape[0] = 0;
  newShape[1] = 0;
  newShape[2] = 1;
  newShape[3] = ArrayOptions::setDataTypeValue(ArrayOptions::defaultFlag(), dataType);
  newShape[4] = 1;
  newShape[5] = 99;

   if (sd::env_isDebug()) {
    // Guard bytes go right after the shape info data (inside the padding region)
    // This allows detecting buffer overruns into the padding
    uint8_t* guardBytes = reinterpret_cast<uint8_t*>(newShape) + (lenOfShapeInfo * sizeof(LongType));
    memset(guardBytes, 0xAB, SHAPE_ALLOC_PADDING);
  }

  DataType actualType = ArrayOptions::dataType(newShape);
  if (actualType != dataType) {
    printf("ERROR: Data type mismatch in scalarShapeInfo - requested %d but got %d\n",
           DataTypeUtils::asInt(dataType), DataTypeUtils::asInt(actualType));
  }
  return newShape;
}
LongType* ShapeBuilders::createVectorShapeInfo(const DataType dataType, const LongType length,
                                               memory::Workspace* workspace) {
  //there is no reason for shape info to use workspaces. we have constant shape helper for this
  // workspaces with shapebuffers also appears to cause issues when reused elsewhere.
  LongType* newShape = new LongType[shape::shapeInfoLength(static_cast<LongType>(1)) + SHAPE_ALLOC_PADDING];

  newShape[0] = 1;
  newShape[1] = length;
  newShape[2] = 1;
  newShape[3] =  ArrayOptions::setDataTypeValue(ArrayOptions::defaultFlag(), dataType);
  newShape[4] = 1;
  newShape[5] = 99;

   if (sd::env_isDebug()) {
    // Guard bytes go right after the shape info data (inside the padding region)
    // This allows detecting buffer overruns into the padding
    uint8_t* guardBytes = reinterpret_cast<uint8_t*>(newShape) + (6 * sizeof(LongType));
    memset(guardBytes, 0xAB, SHAPE_ALLOC_PADDING);
  }

  return newShape;
}


LongType* ShapeBuilders::createShapeInfo(const DataType dataType, const char order, int rank,
                                         const LongType* shapeOnly,
                                         const LongType *strideOnly,
                                         memory::Workspace* workspace, sd::LongType extras) {
  LongType* shapeInfo = nullptr;

  if (rank == 0) {  // scalar case
    shapeInfo = createScalarShapeInfo(dataType, workspace);
  } else {
    shapeInfo = new LongType[shape::shapeInfoLength(rank) + SHAPE_ALLOC_PADDING];

    // Initialize entire buffer to zero first
    memset(shapeInfo, 0, shape::shapeInfoLength(rank) * sizeof(LongType));

    shapeInfo[0] = rank;

    // Set shape values
    for (int i = 0; i < rank; i++) {
      shapeInfo[i + 1] = shapeOnly[i];
    }

    // Set stride values
    for (int i = 0; i < rank; i++) {
      shapeInfo[i + 1 + rank] = strideOnly[i];
    }

    // Explicitly set EWS to -1 (unused) at position length-2
    shapeInfo[shape::shapeInfoLength(rank) - 2] = -1;

    // Set order (at position length-1)
    shapeInfo[shape::shapeInfoLength(rank) - 1] = order;

    if (sd::env_isDebug()) {
      uint8_t* guardBytes = reinterpret_cast<uint8_t*>(shapeInfo) + (shape::shapeInfoLength(rank) * sizeof(LongType));
      memset(guardBytes, 0xAB, SHAPE_ALLOC_PADDING);
    }
  }

  // The 'extras' parameter may not have data type flags set, which would cause
  // ArrayOptions::dataType() to return UNKNOWN, triggering validation errors.
  // We must call setDataType() AFTER setExtra() to ensure the data type is correct.
  ArrayOptions::setExtra(shapeInfo, extras);
  ArrayOptions::setDataType(shapeInfo, dataType);  // Ensure data type is set from the dataType parameter
  shape::setOrder(shapeInfo, order);
  return shapeInfo;
}

LongType* ShapeBuilders::copyShapeInfoWithNewType(const LongType* inShapeInfo, const DataType newType) {
  int rank = shape::rank(inShapeInfo);
  LongType* newShapeInfo = new LongType[shape::shapeInfoLength(rank) + SHAPE_ALLOC_PADDING];

  // Copy the basic shape structure
  memcpy(newShapeInfo, inShapeInfo, shape::shapeInfoByteLength(inShapeInfo));

  if (sd::env_isDebug()) {
    uint8_t* guardBytes = reinterpret_cast<uint8_t*>(newShapeInfo) + (shape::shapeInfoLength(rank) * sizeof(LongType));
    memset(guardBytes, 0xAB, SHAPE_ALLOC_PADDING);
  }

  // Update the data type while preserving other properties
  LongType currentExtra = ArrayOptions::extra(inShapeInfo);
  LongType newExtra = ArrayOptions::setDataTypeValue(
      ArrayOptions::propertyWithoutDataTypeValue(currentExtra),
      newType
  );
  ArrayOptions::setExtra(newShapeInfo, newExtra);

  return newShapeInfo;
}



////////////////////////////////////////////////////////////////////////////////
LongType  * ShapeBuilders::createShapeInfo(const DataType dataType, const char order, int rank, const LongType* shapeOnly,
                                           memory::Workspace* workspace, bool empty)  {
  LongType* shapeInfo = nullptr;

  if (rank == 0) {  // scalar case
    shapeInfo = createScalarShapeInfo(dataType, workspace);
  } else {
    shapeInfo = new LongType[shape::shapeInfoLength(rank) + SHAPE_ALLOC_PADDING];
    // Initialize to zero to avoid uninitialized memory issues
    memset(shapeInfo, 0, shape::shapeInfoLength(rank) * sizeof(LongType));

    if (sd::env_isDebug()) {
      uint8_t* guardBytes = reinterpret_cast<uint8_t*>(shapeInfo) + (shape::shapeInfoLength(rank) * sizeof(LongType));
      memset(guardBytes, 0xAB, SHAPE_ALLOC_PADDING);
    }

    shapeInfo[0] = rank;
    for (int i = 0; i < rank; i++) {
      shapeInfo[i + 1] = shapeOnly[i];
    }

    ArrayOptions::resetFlags(shapeInfo);
    
    // IMPORTANT: Set ARRAY_EMPTY flag BEFORE updateStrides so strides are calculated correctly
    if (empty) {
      ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
    }
    
    shape::updateStrides(shapeInfo, order, false);
  }

  ArrayOptions::setDataType(shapeInfo, dataType);

  // ARRAY_EMPTY already set above if needed

  return shapeInfo;
}

LongType* ShapeBuilders::emptyShapeInfoWithShape(const DataType dataType, std::vector<LongType>& shape,
                                                 memory::Workspace* workspace) {
  auto shapeInfo = createShapeInfo(dataType, 'c', shape, workspace);
  ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
  return shapeInfo;
}

LongType* ShapeBuilders::emptyShapeInfo(const DataType dataType, memory::Workspace* workspace) {
  auto shapeInfo = createScalarShapeInfo(dataType, workspace);
  ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
  return shapeInfo;
}

LongType* ShapeBuilders::emptyShapeInfo(const DataType dataType, const char order,
                                        const std::vector<LongType>& shape, memory::Workspace* workspace) {
  auto shapeInfo = createShapeInfo(dataType, order, shape.size(), shape.data(), workspace, true);
  return shapeInfo;
}

LongType* ShapeBuilders::emptyShapeInfo(const DataType dataType, const char order, int rank,
                                        const LongType* shapeOnly, memory::Workspace* workspace) {
  auto shapeInfo2 = new LongType[shape::shapeInfoLength(rank) + SHAPE_ALLOC_PADDING];
  // Initialize to zero to avoid uninitialized memory issues
  memset(shapeInfo2, 0, shape::shapeInfoLength(rank) * sizeof(LongType));
  shapeInfo2[0] = rank;

  for(int i = 0; i < rank; i++) {
    shapeInfo2[i + 1] = shapeOnly[i];
    //all empty strides are zero
    shapeInfo2[i + 1 + rank] = 0;
  }

  shapeInfo2[2 * rank + 2] = -1;  // EWS unknown for empty arrays
  shape::setOrder(shapeInfo2, order);

  ArrayOptions::setPropertyBits(shapeInfo2, {ARRAY_EMPTY,ArrayOptions::flagForDataType(dataType)});
  return shapeInfo2;
}

////////////////////////////////////////////////////////////////////////////////
LongType* ShapeBuilders::createShapeInfo(const DataType dataType, const char order,
                                         const std::vector<LongType>& shapeOnly, memory::Workspace* workspace) {
  bool isEmpty = false;
  // Check if any dimension is 0 (which makes the array empty)
  for(size_t i = 0; i < shapeOnly.size(); i++) {
    if(shapeOnly[i] == 0) {
      isEmpty = true;
      break;
    }
  }
  auto ret = createShapeInfo(dataType, order, shapeOnly.size(), shapeOnly.data(), workspace, isEmpty);
  if(isEmpty && !ArrayOptions::hasPropertyBitSet(ret, ARRAY_EMPTY)) {
    THROW_EXCEPTION("Shape builders: empty was specified was true but shape info returned false");
  } else if(!isEmpty && ArrayOptions::hasPropertyBitSet(ret, ARRAY_EMPTY)) {
    THROW_EXCEPTION("Shape builders: empty was specified was false but shape info returned true");
  }
  return ret;
}

////////////////////////////////////////////////////////////////////////////////
LongType* ShapeBuilders::createShapeInfo(const DataType dataType, const char order,
                                         const std::initializer_list<LongType>& shapeOnly,
                                         memory::Workspace* workspace) {
  return createShapeInfo(dataType, order, std::vector<LongType>(shapeOnly), workspace);
}

////////////////////////////////////////////////////////////////////////////////
LongType* ShapeBuilders::copyShapeInfo(const LongType* inShapeInfo, const bool copyStrides,
                                       memory::Workspace* workspace) {
  if (inShapeInfo == nullptr) {
    THROW_EXCEPTION("copyShapeInfo: inShapeInfo is nullptr");
  }

  LongType rank = shape::rank(inShapeInfo);
  if (rank < 0 || rank > SD_MAX_RANK) {
    THROW_EXCEPTION("copyShapeInfo: inShapeInfo has invalid rank");
  }

  LongType outLen = shape::shapeInfoLength(rank) + SHAPE_ALLOC_PADDING;
  if (outLen <= 0 || outLen > 10000) {
    THROW_EXCEPTION("copyShapeInfo: unreasonable output length");
  }

  LongType* outShapeInfo = new LongType[outLen]();  // zero-initialize

  // Validate allocation succeeded
  if (outShapeInfo == nullptr) {
    THROW_EXCEPTION("copyShapeInfo: new[] returned nullptr");
  }

  memcpy(outShapeInfo, inShapeInfo, shape::shapeInfoByteLength(inShapeInfo));

  // Set guard bytes in the padding region for corruption detection
  LongType shapeLen = shape::shapeInfoLength(rank);
  uint8_t* guardBytes = reinterpret_cast<uint8_t*>(outShapeInfo) + (shapeLen * sizeof(LongType));
  memset(guardBytes, 0xAB, SHAPE_ALLOC_PADDING);

  if (!copyStrides) shape::updateStrides(outShapeInfo, shape::order(outShapeInfo), false);

  return outShapeInfo;
}


LongType* ShapeBuilders::setAsView(const LongType* inShapeInfo) {
  LongType* outShapeInfo = copyShapeInfo(inShapeInfo, true, nullptr);
  ArrayOptions::toggleIsView(outShapeInfo);
  return outShapeInfo;
}

////////////////////////////////////////////////////////////////////////////////
LongType* ShapeBuilders::copyShapeInfoAndType(const LongType* inShapeInfo, const DataType dtype,
                                              const bool copyStrides, memory::Workspace* workspace) {
  LongType* outShapeInfo = copyShapeInfo(inShapeInfo, copyStrides, workspace);
  ArrayOptions::setExtra(outShapeInfo, ArrayOptions::propertyWithoutDataTypeValue(ArrayOptions::extra(inShapeInfo)));  // set extra value to 0 (like in DataTypeEx::TypeEx
  ArrayOptions::setDataType(outShapeInfo, dtype);
  return outShapeInfo;
}

////////////////////////////////////////////////////////////////////////////////
LongType* ShapeBuilders::copyShapeInfoAndType(const LongType* inShapeInfo,
                                              const LongType* shapeInfoToGetTypeFrom, const bool copyStrides,
                                              memory::Workspace* workspace) {
  return copyShapeInfoAndType(inShapeInfo, ArrayOptions::dataType(shapeInfoToGetTypeFrom), copyStrides,
                              workspace);
}

////////////////////////////////////////////////////////////////////////////////
LongType* ShapeBuilders::createSubArrShapeInfo(const LongType* inShapeInfo, const LongType* dims, const int dimsSize,
                                               memory::Workspace* workspace) {
  LongType* subArrShapeInfo = nullptr;
  ALLOCATE(subArrShapeInfo, workspace, shape::shapeInfoLength(dimsSize), LongType);

  subArrShapeInfo[0] = dimsSize;  // rank
  subArrShapeInfo[2 * dimsSize + 1] = 0;                          // extra flags
  subArrShapeInfo[2 * dimsSize + 2] = -1;                         // EWS unknown for sub-arrays
  ArrayOptions::copyDataType(subArrShapeInfo, inShapeInfo);        // type
  subArrShapeInfo[2 * dimsSize + 3] = shape::order(inShapeInfo);  // order

  LongType* shape = shape::shapeOf(subArrShapeInfo);
  LongType* strides = shape::stride(subArrShapeInfo);

  bool isEmpty = false;
  for (int i = 0; i < dimsSize; ++i) {

    shape[i] = shape::sizeAt(inShapeInfo, dims[i]);
    if(shape[i] == 0) {
      isEmpty = true;
    }
    strides[i] = shape::strideAt(inShapeInfo, dims[i]);
  }



  // Note: checkStridesEwsAndOrder removed - EWS is deprecated and the order
  // is already correctly set from the input shape at line 292. That function
  // was incorrectly overriding the order based on stride contiguity patterns.
  if(isEmpty)
    ArrayOptions::togglePropertyBit(subArrShapeInfo, ARRAY_EMPTY);
  return subArrShapeInfo;
}

}  // namespace sd
