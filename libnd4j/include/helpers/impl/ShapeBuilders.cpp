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

LongType* ShapeBuilders::createShapeInfoFrom(ShapeDescriptor *descriptor) {
  sd::LongType bufferLen = shape::shapeInfoLength(descriptor->rank());
  sd::LongType  *ret = new sd::LongType[bufferLen];
  ret[0] = descriptor->rank();
  shape::setOrder(ret,descriptor->order());
  shape::setOffset(ret,0);
  shape::setElementWiseStride(ret,descriptor->ews());
  shape::setShape(ret,descriptor->shape_strides().data());
  shape::setStride(ret,(descriptor->shape_strides().data() + descriptor->rank()));
  shape::setExtra(ret,descriptor->extra());
  return ret;
}

sd::LongType* ShapeBuilders::createScalarShapeInfo(const sd::DataType dataType, sd::memory::Workspace* workspace) {
  // there is no reason for shape info to use workspaces. we have constant shape helper for this
  // workspaces with shapebuffers also appears to cause issues when reused elsewhere.
  sd::LongType lenOfShapeInfo = 6;
  sd::LongType* newShape = new sd::LongType[lenOfShapeInfo];
  newShape[0] = 0;
  newShape[1] = 0;
  newShape[2] = 1;
  newShape[3] = ArrayOptions::setDataTypeValue(ArrayOptions::defaultFlag(), dataType);
  newShape[4] = 1;
  newShape[5] = 99;
  return newShape;
}
sd::LongType* ShapeBuilders::createVectorShapeInfo(const sd::DataType dataType, const sd::LongType length,
                                                   sd::memory::Workspace* workspace) {
  //there is no reason for shape info to use workspaces. we have constant shape helper for this
  //workspaces with shapebuffers also appears to cause issues when reused elsewhere.
  sd::LongType* newShape = new sd::LongType[shape::shapeInfoLength(static_cast<sd::LongType>(1))];

  newShape[0] = 1;
  newShape[1] = length;
  newShape[2] = 1;
  newShape[3] =  ArrayOptions::setDataTypeValue(ArrayOptions::defaultFlag(), dataType);
  newShape[4] = 1;
  newShape[5] = 99;
  return newShape;
}

////////////////////////////////////////////////////////////////////////////////
LongType* ShapeBuilders::createShapeInfo(const sd::DataType dataType, const char order, int rank,
                                         const sd::LongType* shapeOnly, memory::Workspace* workspace, bool empty) {
  sd::LongType* shapeInfo = nullptr;


  if (rank == 0) {  // scalar case
    shapeInfo = ShapeBuilders::createScalarShapeInfo(dataType, workspace);
  } else {
    shapeInfo = new sd::LongType [shape::shapeInfoLength(rank)];
    shapeInfo[0] = rank;
    for (int i = 0; i < rank; i++) {
      shapeInfo[i + 1] = shapeOnly[i];
    }

    ArrayOptions::resetFlags(shapeInfo);
    shape::updateStrides(shapeInfo, order);


  }

  sd::ArrayOptions::setDataType(shapeInfo, dataType);

  if(empty) {
    ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
  }


  return shapeInfo;
}

sd::LongType* ShapeBuilders::emptyShapeInfoWithShape(const sd::DataType dataType,std::vector<sd::LongType> &shape, memory::Workspace* workspace) {
  auto shapeInfo = createShapeInfo(dataType,'c',shape,workspace);
  ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
  return shapeInfo;
}

sd::LongType* ShapeBuilders::emptyShapeInfo(const sd::DataType dataType, memory::Workspace* workspace) {
  auto shapeInfo = createScalarShapeInfo(dataType, workspace);
  ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
  return shapeInfo;
}

sd::LongType* ShapeBuilders::emptyShapeInfo(const sd::DataType dataType, const char order,
                                            const std::vector<sd::LongType>& shape, memory::Workspace* workspace) {
  auto shapeInfo = createShapeInfo(dataType, order, shape.size(),shape.data(), workspace,true);
  return shapeInfo;
}

sd::LongType* ShapeBuilders::emptyShapeInfo(const sd::DataType dataType, const char order, int rank,
                                            const sd::LongType* shapeOnly, memory::Workspace* workspace) {

  sd::LongType  *shapeInfo2 = new sd::LongType[shape::shapeInfoLength(rank)];
  shapeInfo2[0] = rank;

  for(int i = 0; i < rank; i++) {
    shapeInfo2[i + 1] = shapeOnly[i];
    //all empty strides are zero
    shapeInfo2[i + 1 + rank] = 0;
  }

  shape::setOffset(shapeInfo2, 0);
  shape::setOrder(shapeInfo2, order);


  ArrayOptions::setPropertyBits(shapeInfo2, {ARRAY_EMPTY,ArrayOptions::flagForDataType(dataType)});
  return shapeInfo2;
}

////////////////////////////////////////////////////////////////////////////////
sd::LongType* ShapeBuilders::createShapeInfo(const sd::DataType dataType, const char order,
                                             const std::vector<sd::LongType>& shapeOnly, memory::Workspace* workspace) {
  bool isEmpty = false;
  //shape size 1 but 0 can be scalar
  if(shapeOnly.size() > 1)
    for(int i = 0; i < shapeOnly.size(); i++) {
      if(shapeOnly[i] == 0) {
        isEmpty = true;
        break;
      }
    }
  auto ret =  ShapeBuilders::createShapeInfo(dataType, order, shapeOnly.size(), shapeOnly.data(), workspace, isEmpty);
  if(isEmpty && !ArrayOptions::hasPropertyBitSet(ret, ARRAY_EMPTY)) {
    THROW_EXCEPTION("Shape builders: empty was specified was true but shape info returned false");
  } else if(!isEmpty && ArrayOptions::hasPropertyBitSet(ret, ARRAY_EMPTY)) {
    //TODO: this was triggering.
    THROW_EXCEPTION("Shape builders: empty was specified was false but shape info returned true");
  }
  return ret;
}

////////////////////////////////////////////////////////////////////////////////
sd::LongType* ShapeBuilders::createShapeInfo(const sd::DataType dataType, const char order,
                                             const std::initializer_list<sd::LongType>& shapeOnly,
                                             memory::Workspace* workspace) {
  return ShapeBuilders::createShapeInfo(dataType, order, std::vector<sd::LongType>(shapeOnly), workspace);
}

////////////////////////////////////////////////////////////////////////////////
sd::LongType* ShapeBuilders::copyShapeInfo(const sd::LongType* inShapeInfo, const bool copyStrides,
                                           memory::Workspace* workspace) {
  sd::LongType* outShapeInfo = nullptr;
  ALLOCATE(outShapeInfo, workspace, shape::shapeInfoLength(inShapeInfo), sd::LongType);

  memcpy(outShapeInfo, inShapeInfo, shape::shapeInfoByteLength(inShapeInfo));

  if (!copyStrides) shape::updateStrides(outShapeInfo, shape::order(outShapeInfo));

  return outShapeInfo;
}

////////////////////////////////////////////////////////////////////////////////
sd::LongType* ShapeBuilders::copyShapeInfoAndType(const sd::LongType* inShapeInfo, const DataType dtype,
                                                  const bool copyStrides, memory::Workspace* workspace) {
  sd::LongType* outShapeInfo = ShapeBuilders::copyShapeInfo(inShapeInfo, copyStrides, workspace);
  ArrayOptions::setExtra(outShapeInfo, ArrayOptions::propertyWithoutDataTypeValue(ArrayOptions::extra(inShapeInfo)));  // set extra value to 0 (like in DataTypeEx::TypeEx
  ArrayOptions::setDataType(outShapeInfo, dtype);
  return outShapeInfo;
}

////////////////////////////////////////////////////////////////////////////////
sd::LongType* ShapeBuilders::copyShapeInfoAndType(const sd::LongType* inShapeInfo,
                                                  const sd::LongType* shapeInfoToGetTypeFrom, const bool copyStrides,
                                                  memory::Workspace* workspace) {
  return ShapeBuilders::copyShapeInfoAndType(inShapeInfo, ArrayOptions::dataType(shapeInfoToGetTypeFrom), copyStrides,
                                             workspace);
}

////////////////////////////////////////////////////////////////////////////////
sd::LongType* ShapeBuilders::createSubArrShapeInfo(const sd::LongType* inShapeInfo, const LongType* dims, const int dimsSize,
                                                   memory::Workspace* workspace) {
  sd::LongType* subArrShapeInfo = nullptr;
  ALLOCATE(subArrShapeInfo, workspace, shape::shapeInfoLength(dimsSize), sd::LongType);

  subArrShapeInfo[0] = dimsSize;  // rank
  subArrShapeInfo[2 * dimsSize + 1] = 0;
  sd::ArrayOptions::copyDataType(subArrShapeInfo, inShapeInfo);   // type
  subArrShapeInfo[2 * dimsSize + 3] = shape::order(inShapeInfo);  // order

  sd::LongType* shape = shape::shapeOf(subArrShapeInfo);
  sd::LongType* strides = shape::stride(subArrShapeInfo);

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
