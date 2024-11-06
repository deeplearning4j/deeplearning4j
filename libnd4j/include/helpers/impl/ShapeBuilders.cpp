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

#include "array/ShapeDescriptor.h"

namespace sd {

LongType* ShapeBuilders::createShapeInfoFrom(ShapeDescriptor* descriptor) {
  LongType bufferLen = shape::shapeInfoLength(descriptor->rank());
  auto ret = new LongType[bufferLen];
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

  shape::setElementWiseStride(ret, descriptor->ews());
  shape::setExtra(ret, descriptor->extra());
  return ret;
}

LongType* ShapeBuilders::createScalarShapeInfo(const DataType dataType, memory::Workspace* workspace) {
  // there is no reason for shape info to use workspaces. we have constant shape helper for this
  // workspaces with shapebuffers also appears to cause issues when reused elsewhere.
  LongType lenOfShapeInfo = 6;
  auto newShape = new LongType[lenOfShapeInfo];
  newShape[0] = 0;
  newShape[1] = 0;
  newShape[2] = 1;
  newShape[3] = ArrayOptions::setDataTypeValue(ArrayOptions::defaultFlag(), dataType);
  newShape[4] = 1;
  newShape[5] = 99;
  return newShape;
}
LongType* ShapeBuilders::createVectorShapeInfo(const DataType dataType, const LongType length,
                                               memory::Workspace* workspace) {
  //there is no reason for shape info to use workspaces. we have constant shape helper for this
  // workspaces with shapebuffers also appears to cause issues when reused elsewhere.
  LongType* newShape = new LongType[shape::shapeInfoLength(static_cast<LongType>(1))];

  newShape[0] = 1;
  newShape[1] = length;
  newShape[2] = 1;
  newShape[3] =  ArrayOptions::setDataTypeValue(ArrayOptions::defaultFlag(), dataType);
  newShape[4] = 1;
  newShape[5] = 99;
  return newShape;
}

////////////////////////////////////////////////////////////////////////////////
 LongType  * ShapeBuilders::createShapeInfo(const DataType dataType, const char order, int rank, const LongType* shapeOnly,
                                    memory::Workspace* workspace, bool empty)  {
  LongType* shapeInfo = nullptr;

  if (rank == 0) {  // scalar case
    shapeInfo = createScalarShapeInfo(dataType, workspace);
  } else {
    shapeInfo = new LongType[shape::shapeInfoLength(rank)];
    shapeInfo[0] = rank;
    for (int i = 0; i < rank; i++) {
      shapeInfo[i + 1] = shapeOnly[i];
    }

    ArrayOptions::resetFlags(shapeInfo);
    shape::updateStrides(shapeInfo, order);
  }

  ArrayOptions::setDataType(shapeInfo, dataType);

  if (empty) {
    ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
  }

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
  auto shapeInfo2 = new LongType[shape::shapeInfoLength(rank)];
  shapeInfo2[0] = rank;

  for(int i = 0; i < rank; i++) {
    shapeInfo2[i + 1] = shapeOnly[i];
    //all empty strides are zero
    shapeInfo2[i + 1 + rank] = 0;
  }

  shape::setOrder(shapeInfo2, order);


  ArrayOptions::setPropertyBits(shapeInfo2, {ARRAY_EMPTY,ArrayOptions::flagForDataType(dataType)});
  return shapeInfo2;
}

////////////////////////////////////////////////////////////////////////////////
LongType* ShapeBuilders::createShapeInfo(const DataType dataType, const char order,
                                         const std::vector<LongType>& shapeOnly, memory::Workspace* workspace) {
  bool isEmpty = false;
  //shape size 1 but 0 can be scalar
  if(shapeOnly.size() > 1)
    for(int i = 0; i < shapeOnly.size(); i++) {
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
  LongType* outShapeInfo = nullptr;
  ALLOCATE(outShapeInfo, workspace, shape::shapeInfoLength(inShapeInfo), sd::LongType);

  memcpy(outShapeInfo, inShapeInfo, shape::shapeInfoByteLength(inShapeInfo));

  if (!copyStrides) shape::updateStrides(outShapeInfo, shape::order(outShapeInfo));

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
  subArrShapeInfo[2 * dimsSize + 1] = 0;
  ArrayOptions::copyDataType(subArrShapeInfo, inShapeInfo);   // type
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



  shape::checkStridesEwsAndOrder(subArrShapeInfo);
  if(isEmpty)
    ArrayOptions::togglePropertyBit(subArrShapeInfo, ARRAY_EMPTY);
  return subArrShapeInfo;
}

}  // namespace sd
