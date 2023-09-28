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
#include <array/CudaPointerDeallocator.h>
#include <array/PrimaryPointerDeallocator.h>
#include <array/ShapeDescriptor.h>
#include <exceptions/cuda_exception.h>
#include <execution/AffinityManager.h>
#include <helpers/ConstantHelper.h>
#include <helpers/ShapeBuilders.h>
#include <helpers/ShapeUtils.h>

#include "../ConstantShapeHelper.h"

namespace sd {


ConstantShapeHelper::ConstantShapeHelper() {
  auto numDevices = AffinityManager::numberOfDevices();

  _cache.resize(numDevices);
  for (int e = 0; e < numDevices; e++) {
    SD_MAP_IMPL<ShapeDescriptor, ConstantShapeBuffer *> cache;
    _cache[e] = cache;
  }
}


ConstantShapeHelper& ConstantShapeHelper::getInstance() {
  static ConstantShapeHelper instance;
  return instance;
}

ConstantShapeBuffer* ConstantShapeHelper::bufferForShapeInfo(const sd::DataType dataType, const char order,
                                                             const int rank, const sd::LongType* shape) {
  ShapeDescriptor *descriptor = new ShapeDescriptor(dataType, order, shape, rank);
  auto ret = bufferForShapeInfo(descriptor);
  delete descriptor;
  return ret;
}

ConstantShapeBuffer* ConstantShapeHelper::storeAndWrapBuffer(LongType* buffer, ShapeDescriptor* descriptor) {
  int deviceId = AffinityManager::currentDeviceId();

  std::lock_guard<std::mutex> lock(_mutex);

  if(descriptor == nullptr)
    descriptor = new ShapeDescriptor(buffer);

  if(descriptor->dataType() == sd::DataType::UNKNOWN) {
    THROW_EXCEPTION("Unable to create array with unknown data type.");
  }

  if(buffer == nullptr) {
    THROW_EXCEPTION("Unable to create and store a shape buffer with null buffer.");
  }


  if(ArrayOptions::dataType(buffer) == sd::DataType::UNKNOWN) {
    THROW_EXCEPTION("Unable to create and store a shape buffer with unknown data type.");
  }


  if (_cache[deviceId].count(*descriptor) == 0) {
    auto hPtr =
        std::make_shared<PointerWrapper>(buffer, std::make_shared<PrimaryPointerDeallocator>());

    auto hPtrPointer = hPtr->pointer();
    auto byteLength = shape::shapeInfoByteLength(hPtr->pointerAsT<sd::LongType>());
    auto dealloc = std::make_shared<CudaPointerDeallocator>();
    auto replicated =  ConstantHelper::getInstance().replicatePointer(hPtrPointer,
                                                                      byteLength);
    auto dPtr = std::make_shared<PointerWrapper>(
        replicated,
        dealloc);

    ConstantShapeBuffer *buffer =  new ConstantShapeBuffer(hPtr, dPtr);
    _cache[deviceId][*descriptor] = buffer;
    return buffer;
  } else {
    return _cache[deviceId].at(*descriptor);
  }
}


ConstantShapeBuffer* ConstantShapeHelper::bufferForShapeInfo(ShapeDescriptor *descriptor) {
  return storeAndWrapBuffer(descriptor->toShapeInfo(), descriptor);
}

ConstantShapeBuffer* ConstantShapeHelper::bufferForShapeInfo(const sd::LongType* shapeInfo) {
  ShapeDescriptor *descriptor = new ShapeDescriptor(shapeInfo);
  auto ret = bufferForShapeInfo(descriptor);
  delete descriptor;
  return ret;
}

bool ConstantShapeHelper::checkBufferExistenceForShapeInfo(ShapeDescriptor *descriptor) {
  auto deviceId = AffinityManager::currentDeviceId();
  std::lock_guard<std::mutex> lock(_mutex);

  return _cache[deviceId].count(*descriptor) != 0;
}

const sd::LongType* ConstantShapeHelper::createShapeInfo(const sd::DataType dataType, const char order, const int rank,
                                                         const sd::LongType* shape, LongType extraProperties = -1) {

  if(extraProperties < 0) {
    extraProperties = ArrayOptions::flagForDataType(dataType);
  }




  ShapeDescriptor *descriptor =
      new ShapeDescriptor(dataType, order, shape, (sd::LongType*)nullptr, rank, extraProperties);
  auto ret = bufferForShapeInfo(descriptor)->primary();
  //delete descriptor;
  return ret;
}

const sd::LongType * ConstantShapeHelper::createShapeInfo(const sd::DataType dataType, const sd::LongType* shapeInfo) {
  return ConstantShapeHelper::createShapeInfo(dataType, shape::order(shapeInfo), shape::rank(shapeInfo),
                                              shape::shapeOf(const_cast<sd::LongType*>(shapeInfo)), -1);
}

const sd::LongType * ConstantShapeHelper::emptyShapeInfoWithShape(const sd::DataType dataType,std::vector<sd::LongType> &shape) {
  auto descriptor = ShapeBuilders::createShapeInfo(dataType,'c', shape, nullptr);
  ArrayOptions::setPropertyBit(descriptor, ARRAY_EMPTY);
  auto existing = createFromExisting(descriptor);
  //delete descriptor;
  return existing;
}

const sd::LongType * ConstantShapeHelper::emptyShapeInfo(const sd::DataType dataType) {
  auto descriptor = ShapeBuilders::emptyShapeInfo(dataType,nullptr);
  auto existing = createFromExisting(descriptor);
  //delete descriptor;
  return existing;
}

const sd::LongType * ConstantShapeHelper::scalarShapeInfo(const sd::DataType dataType) {
  auto descriptor = ShapeBuilders::createScalarShapeInfo(dataType);
  auto ret = createFromExisting(descriptor);
  // delete descriptor;
  return ret;
}

const sd::LongType * ConstantShapeHelper::vectorShapeInfo(const sd::LongType length, const sd::DataType dataType) {
  auto descriptor = ShapeBuilders::createVectorShapeInfo(dataType, length);
  auto ret = createFromExisting(descriptor);
  //delete descriptor;
  return ret;
}

const sd::LongType * ConstantShapeHelper::createShapeInfo(const sd::DataType dataType, const char order,
                                                          const std::vector<sd::LongType>& shape) {
  auto ret = ShapeBuilders::createShapeInfo(dataType, order, shape, nullptr);
  auto existing = createFromExisting(ret);
  return existing;
}

const sd::LongType * ConstantShapeHelper::createShapeInfo(ShapeDescriptor *descriptor) {
  return bufferForShapeInfo(descriptor)->primary();
}


const sd::LongType * ConstantShapeHelper::createFromExisting(const sd::LongType* shapeInfo, bool destroyOriginal) {
  ShapeDescriptor *descriptor = new ShapeDescriptor(shapeInfo);
  auto result = createShapeInfo(descriptor);
  // delete descriptor;
  return result;
}

const sd::LongType * ConstantShapeHelper::createFromExisting(const sd::LongType* shapeInfo, sd::memory::Workspace* workspace) {
  ShapeDescriptor *descriptor = new ShapeDescriptor(shapeInfo);
  auto result = createShapeInfo(descriptor);
  delete descriptor;
  return result;
}


const sd::LongType * ConstantShapeHelper::createFromExisting(sd::LongType* shapeInfo, bool destroyOriginal) {
  ShapeDescriptor *descriptor = new ShapeDescriptor(shapeInfo);
  auto result = createShapeInfo(descriptor);
  delete descriptor;
  if (destroyOriginal) RELEASE(shapeInfo, nullptr);

  return result;
}

const sd::LongType * ConstantShapeHelper::createFromExisting(sd::LongType* shapeInfo, sd::memory::Workspace* workspace) {
  ShapeDescriptor *descriptor = new ShapeDescriptor(shapeInfo);
  auto result = createShapeInfo(descriptor);
  delete descriptor;
  //RELEASE(shapeInfo, workspace);
  return result;
}

////////////////////////////////////////////////////////////////////////
ConstantShapeBuffer * ConstantShapeHelper::createShapeInfoWithUnitiesForBroadcast(const sd::LongType* maxShapeInfo,
                                                                                  const sd::LongType* minShapeInfo,
                                                                                  sd::memory::Workspace* workspace,
                                                                                  const std::vector<sd::LongType>& dimensions) {
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

  ArrayOptions::setDataType(newShapeInfo, ArrayOptions::dataType(maxShapeInfo));

  ShapeDescriptor *descriptor = new ShapeDescriptor(newShapeInfo);
  //RELEASE(newShapeInfo, workspace);

  auto ret = bufferForShapeInfo(descriptor);
  delete descriptor;
  return ret;
}

////////////////////////////////////////////////////////////////////////
ConstantShapeBuffer * ConstantShapeHelper::createShapeInfoWithNoUnitiesForReduce(const sd::LongType* inShapeInfo, const std::vector<LongType> *dimsWithUnities,
                                                                                 sd::memory::Workspace* workspace) {
  sd::LongType* newShapeInfo = nullptr;
  ALLOCATE(newShapeInfo, workspace, shape::shapeInfoLength(shape::rank(inShapeInfo) - dimsWithUnities->size()),
           sd::LongType);

  sd::LongType temp;
  if (dimsWithUnities->size() == 1 && shape::isCommonVector(inShapeInfo, temp) && temp == dimsWithUnities->at(0)) {
    auto dims = ShapeUtils::evalDimsToExclude(shape::rank(inShapeInfo), 1,&temp);
    shape::excludeUnitiesFromShapeInfo(inShapeInfo, dims->data(), dims->size(), newShapeInfo);
    delete dims;
  } else {
    shape::excludeUnitiesFromShapeInfo(inShapeInfo, dimsWithUnities->data(), dimsWithUnities->size(), newShapeInfo);
  }

  ShapeDescriptor *descriptor = new ShapeDescriptor(newShapeInfo);

  //RELEASE(newShapeInfo, workspace);

  auto ret = bufferForShapeInfo(descriptor);
  delete descriptor;

  return ret;
}

////////////////////////////////////////////////////////////////////////
ConstantShapeBuffer *ConstantShapeHelper::createSubArrShapeInfo(const sd::LongType* inShapeInfo, const LongType* dims,
                                                                const LongType dimsSize, sd::memory::Workspace* workspace) {
  sd::LongType* newShapeInfo = ShapeBuilders::createSubArrShapeInfo(inShapeInfo, dims, dimsSize, workspace);

  ShapeDescriptor *descriptor = new ShapeDescriptor(newShapeInfo);

  //RELEASE(newShapeInfo, workspace);

  auto ret =  bufferForShapeInfo(descriptor);
  delete descriptor;
  return ret;
}

}  // namespace sd
