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
//  @author raver119@gmail.com
//

#ifndef __CUDABLAS__
#include <array/PrimaryPointerDeallocator.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ShapeBuilders.h>
#include <helpers/ShapeUtils.h>
#include <helpers/logger.h>

namespace sd {



ConstantShapeHelper::ConstantShapeHelper() {
  _cache.resize(1);
  for (int e = 0; e < 1; e++) {
    SD_MAP_IMPL<ShapeDescriptor , ConstantShapeBuffer *> cache;
    _cache[e] = cache;
  }
}

ConstantShapeHelper& ConstantShapeHelper::getInstance() {
  static ConstantShapeHelper instance;
  return instance;
}

ConstantShapeBuffer* ConstantShapeHelper::bufferForShapeInfo(sd::DataType dataType, char order,
                                                             const std::vector<sd::LongType>& shape) {
  auto descriptor = new ShapeDescriptor(dataType, order, shape);
  auto ret =  bufferForShapeInfo(descriptor);
  delete descriptor;
  return ret;
}

ConstantShapeBuffer* ConstantShapeHelper::bufferForShapeInfo(const sd::DataType dataType, const char order,
                                                             const int rank, const sd::LongType* shape) {
  auto descriptor = new ShapeDescriptor(dataType, order, shape, rank);
  auto ret =  bufferForShapeInfo(descriptor);
  delete descriptor;
  return ret;
}

ConstantShapeBuffer * ConstantShapeHelper::bufferForShapeInfo(ShapeDescriptor *descriptor) {
  int deviceId = 0;
  std::lock_guard<std::mutex> lock(_mutex);
  if(_cache.empty()) {
    THROW_EXCEPTION("Cache is empty!");
  }

  if (_cache[deviceId].count(*descriptor) == 0) {
    auto hPtr =
        std::make_shared<PointerWrapper>(descriptor->toShapeInfo(), std::make_shared<PrimaryPointerDeallocator>());
    ConstantShapeBuffer *constantShapeBuffer2 = new ConstantShapeBuffer(hPtr);
    _cache[deviceId][*descriptor] = constantShapeBuffer2;
    return constantShapeBuffer2;
  } else {
    return _cache[deviceId].at(*descriptor);
  }
}

ConstantShapeBuffer* ConstantShapeHelper::bufferForShapeInfo(const sd::LongType* shapeInfo) {
  auto descriptor = new ShapeDescriptor(shapeInfo);
  auto ret =  bufferForShapeInfo(descriptor);
  delete descriptor;
  return ret;
}

bool ConstantShapeHelper::checkBufferExistenceForShapeInfo(ShapeDescriptor *descriptor) {
  int deviceId = 0;
  std::lock_guard<std::mutex> lock(_mutex);

  return _cache[deviceId].count(*descriptor) != 0;
}

const sd::LongType* ConstantShapeHelper::createShapeInfo(const sd::DataType dataType, const char order, const int rank,
                                                         const sd::LongType* shape) {
  auto descriptor = new ShapeDescriptor(dataType, order, shape, rank);
  auto ret =  bufferForShapeInfo(descriptor)->primary();
  delete descriptor;
  return ret;
}

const sd::LongType* ConstantShapeHelper::createShapeInfo(const sd::DataType dataType, const sd::LongType* shapeInfo) {
  return ConstantShapeHelper::createShapeInfo(dataType, shape::order(shapeInfo), shape::rank(shapeInfo),
                                              shape::shapeOf(const_cast<sd::LongType*>(shapeInfo)));
}

const sd::LongType* ConstantShapeHelper::emptyShapeInfo(const sd::DataType dataType) {
  auto descriptor = ShapeDescriptor::emptyDescriptor(dataType);
  auto ret = bufferForShapeInfo(descriptor)->primary();
  delete descriptor;
  return ret;
}

const sd::LongType* ConstantShapeHelper::scalarShapeInfo(const sd::DataType dataType) {
  auto descriptor = ShapeDescriptor::scalarDescriptor(dataType);
  auto ret =  bufferForShapeInfo(descriptor)->primary();
  delete descriptor;
  return ret;
}

const sd::LongType* ConstantShapeHelper::vectorShapeInfo(const sd::LongType length, const sd::DataType dataType) {
  auto descriptor = ShapeDescriptor::vectorDescriptor(length, dataType);
  auto ret = bufferForShapeInfo(descriptor)->primary();
  delete descriptor;
  return ret;
}

const sd::LongType* ConstantShapeHelper::createShapeInfo(const sd::DataType dataType, const char order,
                                                         const std::vector<sd::LongType>& shape) {
  ShapeDescriptor * descriptor = new ShapeDescriptor(dataType, order, shape);
  auto ret =  bufferForShapeInfo(descriptor)->primary();
  delete descriptor;
  return ret;
}

const sd::LongType* ConstantShapeHelper::createShapeInfo(ShapeDescriptor* descriptor) {
  return bufferForShapeInfo(descriptor)->primary();
}

const sd::LongType* ConstantShapeHelper::createFromExisting(sd::LongType* shapeInfo, bool destroyOriginal) {
  ShapeDescriptor *descriptor = new ShapeDescriptor(shapeInfo);
  auto result = createShapeInfo(descriptor);
  if (destroyOriginal) RELEASE(shapeInfo, nullptr)
  delete descriptor;
  return result;
}

const sd::LongType* ConstantShapeHelper::createFromExisting(sd::LongType* shapeInfo, sd::memory::Workspace* workspace) {
  ShapeDescriptor *descriptor = new ShapeDescriptor(shapeInfo);
  auto result = createShapeInfo(descriptor);

  RELEASE(shapeInfo, workspace);
  delete descriptor;
  return result;
}


const sd::LongType * ConstantShapeHelper::createFromExisting(const sd::LongType* shapeInfo, bool destroyOriginal) {
  ShapeDescriptor *descriptor = new ShapeDescriptor(shapeInfo);
  auto result = createShapeInfo(descriptor);
  delete descriptor;
  return result;
}

const sd::LongType * ConstantShapeHelper::createFromExisting(const sd::LongType* shapeInfo, sd::memory::Workspace* workspace) {
  ShapeDescriptor *descriptor = new ShapeDescriptor(shapeInfo);
  auto result = createShapeInfo(descriptor);
  delete descriptor;
  return result;
}

////////////////////////////////////////////////////////////////////////
ConstantShapeBuffer* ConstantShapeHelper::createShapeInfoWithUnitiesForBroadcast(const sd::LongType* maxShapeInfo,
                                                                                 const sd::LongType* minShapeInfo,
                                                                                 sd::memory::Workspace* workspace,
    const std::vector<LongType>& dimensions) {
  sd::LongType* newShapeInfo = nullptr;
  ALLOCATE(newShapeInfo, workspace, shape::shapeInfoLength(shape::rank(maxShapeInfo)), sd::LongType);

  newShapeInfo[0] = shape::rank(maxShapeInfo);
  newShapeInfo[2 * shape::rank(maxShapeInfo) + 1] = 0;
  sd::ArrayOptions::copyDataType(newShapeInfo, minShapeInfo);                      // type
  newShapeInfo[2 * newShapeInfo[0] + 2] = shape::elementWiseStride(minShapeInfo);  // ews
  newShapeInfo[2 * newShapeInfo[0] + 3] = shape::order(minShapeInfo);              // order

  if (!dimensions.empty()) {
    for (sd::LongType k = 0, j = 0, i = 0; i < shape::rank(maxShapeInfo); ++i) {
      if (j < dimensions.size() && dimensions[j] == i) {
        shape::shapeOf(newShapeInfo)[i] = shape::shapeOf(minShapeInfo)[k];
        shape::stride(newShapeInfo)[i] = shape::stride(minShapeInfo)[k++];
        ++j;
      } else {
        shape::shapeOf(newShapeInfo)[i] = 1;
        shape::stride(newShapeInfo)[i] = 0;
        if (shape::sizeAt(minShapeInfo, k) == 1 && dimensions.size() != shape::rank(minShapeInfo)) ++k;
      }
    }
  } else {
    for (int j = shape::rank(minShapeInfo) - 1, i = shape::rank(maxShapeInfo) - 1; i >= 0; --i) {
      if (j >= 0) {
        shape::shapeOf(newShapeInfo)[i] = shape::shapeOf(minShapeInfo)[j];
        shape::stride(newShapeInfo)[i] = shape::shapeOf(minShapeInfo)[j] == 1 ? 0 : shape::stride(minShapeInfo)[j];
        --j;
      } else {
        shape::shapeOf(newShapeInfo)[i] = 1;
        shape::stride(newShapeInfo)[i] = 0;
      }
    }
  }

  ShapeDescriptor *descriptor = new ShapeDescriptor(newShapeInfo);

  RELEASE(newShapeInfo, workspace);

  auto ret = bufferForShapeInfo(descriptor);
  delete descriptor;
  return ret;
}

////////////////////////////////////////////////////////////////////////
ConstantShapeBuffer* ConstantShapeHelper::createShapeInfoWithNoUnitiesForReduce(const sd::LongType* maxShapeInfo, const std::vector<LongType>* dimsWithUnities,
                                                                                sd::memory::Workspace* workspace) {
  sd::LongType* newShapeInfo = nullptr;
  ALLOCATE(newShapeInfo, workspace, shape::shapeInfoLength(shape::rank(maxShapeInfo) - dimsWithUnities->size()),
           sd::LongType);

  sd::LongType temp;
  if (dimsWithUnities->size() == 1 && shape::isCommonVector(maxShapeInfo, temp) && temp == dimsWithUnities->at(0)) {
    auto dims = ShapeUtils::evalDimsToExclude(shape::rank(maxShapeInfo), 1,&temp);
    shape::excludeUnitiesFromShapeInfo(maxShapeInfo, dims->data(), dims->size(), newShapeInfo);
  } else {
    shape::excludeUnitiesFromShapeInfo(maxShapeInfo, dimsWithUnities->data(), dimsWithUnities->size(), newShapeInfo);
  }

  ShapeDescriptor *descriptor = new ShapeDescriptor(newShapeInfo);

  RELEASE(newShapeInfo, workspace);

  auto ret =  bufferForShapeInfo(descriptor);
  delete descriptor;
  return ret;
}

////////////////////////////////////////////////////////////////////////
ConstantShapeBuffer* ConstantShapeHelper::createSubArrShapeInfo(const sd::LongType* inShapeInfo, const LongType* dims,
                                                                const sd::LongType dimsSize, sd::memory::Workspace* workspace) {
  sd::LongType* newShapeInfo = ShapeBuilders::createSubArrShapeInfo(inShapeInfo, dims, dimsSize, workspace);

  ShapeDescriptor *descriptor = new ShapeDescriptor(newShapeInfo);

  RELEASE(newShapeInfo, workspace);

  auto ret = bufferForShapeInfo(descriptor);
  delete descriptor;
  return ret;
}

}  // namespace sd

#endif
