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

namespace sd {

sd::LongType* ShapeBuilders::createScalarShapeInfo(const sd::DataType dataType, sd::memory::Workspace* workspace) {
  sd::LongType* newShape;
  ALLOCATE(newShape, workspace, shape::shapeInfoLength(0), sd::LongType);
  newShape[0] = 0;
  newShape[1] = 0;
  newShape[2] = 1;
  newShape[3] = 99;

  sd::ArrayOptions::setDataType(newShape, dataType);

  return newShape;
}

sd::LongType* ShapeBuilders::createVectorShapeInfo(const sd::DataType dataType, const sd::LongType length,
                                                   sd::memory::Workspace* workspace) {
  sd::LongType* newShape;
  ALLOCATE(newShape, workspace, shape::shapeInfoLength(1), sd::LongType);

  newShape[0] = 1;
  newShape[1] = length;
  newShape[2] = 1;
  newShape[3] = 0;
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
sd::LongType* ShapeBuilders::createSubArrShapeInfo(const sd::LongType* inShapeInfo, const int* dims, const int dimsSize,
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
