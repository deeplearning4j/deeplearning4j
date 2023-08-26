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
  int bufferLen = shape::shapeInfoLength(descriptor->rank());
  sd::LongType  *ret;
  printf("Executing createShapeInfoFrom...\n");
  if(descriptor->_dataType == sd::DataType::ANY) {
    ret = new sd::LongType[bufferLen];
    memset(ret, 0, bufferLen * sizeof(sd::LongType));
    return ret;
  }
  //don't access to early if vector is actually empty due to scalar case
  auto _shape = descriptor->_shape_strides.data();
  auto _strides = descriptor->_shape_strides.data() + descriptor->_rank;
  switch (descriptor->_rank) {
    case 0: {
      ret = ShapeBuilders::createScalarShapeInfo(descriptor->_dataType);
      ret[2] = descriptor->_ews;
    } break;
    case 1: {
      ret = ShapeBuilders::createVectorShapeInfo(descriptor->_dataType, _shape[0]);
      ret[2 + descriptor->_rank * 2] = descriptor->_ews;
      ret[2] = _strides[0];
      ret[2 + descriptor->_rank * 2 + 1] = descriptor->_order;
    } break;
    default: {
      ret = ShapeBuilders::createShapeInfo(descriptor->_dataType, descriptor->_order, descriptor->_rank, _shape);
      for (int e = 0; e < descriptor->_rank; e++) ret[e + 1 + descriptor->_rank] = _strides[e];
      ret[2 + descriptor->_rank * 2] = descriptor->_ews;
    }
  }


  ArrayOptions::setPropertyBit(ret, descriptor->_extraProperties);
  return ret;
}

sd::LongType* ShapeBuilders::createScalarShapeInfo(const sd::DataType dataType, sd::memory::Workspace* workspace) {
  // there is no reason for shape info to use workspaces. we have constant shape helper for this
  // workspaces with shapebuffers also appears to cause issues when reused elsewhere.
  sd::LongType lenOfShapeInfo = shape::shapeInfoLength(static_cast<sd::LongType>(0));
  sd_printf("Scalar shape info shape info length is %d\n", lenOfShapeInfo);
  sd::LongType* newShape = new sd::LongType[lenOfShapeInfo];
  sd_print("Created new shape\n");
  newShape[0] = 0;
  newShape[1] = 0;
  newShape[2] = 1;
  newShape[3] = 0;
  newShape[4] = 1;
  newShape[5] = 99;
  sd_print("Set all values about to set data type\n");
  sd::ArrayOptions::setDataType(newShape, dataType);
  sd_print("Finished createScalarShapeInfo\n");
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
  newShape[3] = 1;
  newShape[4] = 1;
  newShape[5] = 99;

  sd::ArrayOptions::setDataType(newShape, dataType);

  return newShape;
}

////////////////////////////////////////////////////////////////////////////////
sd::LongType* ShapeBuilders::createShapeInfo(const sd::DataType dataType, const char order, int rank,
                                             const sd::LongType* shapeOnly, memory::Workspace* workspace) {
  sd::LongType* shapeInfo = nullptr;

  if (rank == 0) {  // scalar case
    shapeInfo = ShapeBuilders::createScalarShapeInfo(dataType, workspace);
  } else {
    ALLOCATE(shapeInfo, workspace, shape::shapeInfoLength(rank), sd::LongType);
    shapeInfo[0] = rank;
    bool isEmpty = false;
    for (int i = 0; i < rank; ++i) {
      shapeInfo[i + 1] = shapeOnly[i];

      if (shapeOnly[i] == 0) isEmpty = true;
    }

    if (!isEmpty) {
      shape::updateStrides(shapeInfo, order);
    } else {
      shapeInfo[shape::shapeInfoLength(rank) - 1] = order;
      memset(shape::stride(shapeInfo), 0, rank * sizeof(sd::LongType));
      ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
    }

    sd::ArrayOptions::setDataType(shapeInfo, dataType);
  }

  return shapeInfo;
}

sd::LongType* ShapeBuilders::emptyShapeInfo(const sd::DataType dataType, memory::Workspace* workspace) {
  auto shapeInfo = createScalarShapeInfo(dataType, workspace);
  ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
  return shapeInfo;
}

sd::LongType* ShapeBuilders::emptyShapeInfo(const sd::DataType dataType, const char order,
                                            const std::vector<sd::LongType>& shape, memory::Workspace* workspace) {
  auto shapeInfo = createShapeInfo(dataType, order, shape, workspace);
  memset(shape::stride(shapeInfo), 0, shape.size() * sizeof(sd::LongType));
  ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
  return shapeInfo;
}

sd::LongType* ShapeBuilders::emptyShapeInfo(const sd::DataType dataType, const char order, int rank,
                                            const sd::LongType* shapeOnly, memory::Workspace* workspace){

  auto shapeInfo = createShapeInfo(dataType, order, rank, shapeOnly, workspace);
  memset(shape::stride(shapeInfo), 0, rank * sizeof(sd::LongType));
  ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
  return shapeInfo;
}

////////////////////////////////////////////////////////////////////////////////
sd::LongType* ShapeBuilders::createShapeInfo(const sd::DataType dataType, const char order,
                                             const std::vector<sd::LongType>& shapeOnly, memory::Workspace* workspace) {
  return ShapeBuilders::createShapeInfo(dataType, order, shapeOnly.size(), shapeOnly.data(), workspace);
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

  for (int i = 0; i < dimsSize; ++i) {
    shape[i] = shape::sizeAt(inShapeInfo, dims[i]);
    strides[i] = shape::strideAt(inShapeInfo, dims[i]);
  }

  shape::checkStridesEwsAndOrder(subArrShapeInfo);

  return subArrShapeInfo;
}

}  // namespace sd
