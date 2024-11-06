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
    SD_MAP_IMPL<ShapeDescriptor, ConstantShapeBuffer*> cache;
    _cache[e] = cache;
  }
}

ConstantShapeHelper& ConstantShapeHelper::getInstance() {
  static ConstantShapeHelper instance;
  return instance;
}

ConstantShapeBuffer* ConstantShapeHelper::bufferForShapeInfo(const DataType dataType, const char order, const int rank,
                                                             const LongType* shape) {
  ShapeDescriptor* descriptor = new ShapeDescriptor(dataType, order, shape, rank);
  auto ret = bufferForShapeInfo(descriptor);
  // note we used to delete descriptors here. Some end up being keys in the
  // constant shape helper. We should avoid deleting these.
  return ret;
}


ConstantShapeBuffer* ConstantShapeHelper::storeAndWrapBuffer(ShapeDescriptor* descriptor) {
  int deviceId = AffinityManager::currentDeviceId();
  std::lock_guard<std::mutex> lock(_mutex);
  if(descriptor == nullptr)
    THROW_EXCEPTION("Unable to create and store a shape buffer with null descriptor.");

  auto buffer = descriptor->toShapeInfo();



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
    auto byteLength = shape::shapeInfoByteLength(hPtr->pointerAsT<LongType>());
    auto dealloc = std::make_shared<CudaPointerDeallocator>();
    auto replicated = ConstantHelper::getInstance().replicatePointer(hPtrPointer, byteLength);
    auto dPtr = std::make_shared<PointerWrapper>(replicated, dealloc);

    ConstantShapeBuffer *constantShapeBuffer2 = new ConstantShapeBuffer(hPtr,dPtr);
    //validate
    if(Environment::getInstance().isVerbose() || Environment::getInstance().isDebug()) {
      auto descBuffer = descriptor->toShapeInfo();
      auto constBuffer = constantShapeBuffer2->primary();
      if(!shape::haveSameShapeAndStrides(descBuffer, constBuffer)) {
        std::string errorMessage;
        errorMessage += "Attempting to store Shape info and cache buffer shape info that do not match: \n";
        errorMessage += "Shape info:\n";
        errorMessage += shape::shapeToString(descBuffer,"\n");
        errorMessage += "\nCache buffer shape info:\n";
        errorMessage += shape::shapeToString(constBuffer,"\n");
        THROW_EXCEPTION(errorMessage.c_str());
      }
    }
    _cache[deviceId][*descriptor] = constantShapeBuffer2;
    return constantShapeBuffer2;
  } else {
    auto cacheBuff = _cache[deviceId].at(*descriptor);
    auto cacheBuffPrim = _cache[deviceId].at(*descriptor)->primary();
    if(Environment::getInstance().isDebug() || Environment::getInstance().isVerbose()) {
      //ensure cache values aren't inconsistent when we debug
      if(!shape::haveSameShapeAndStrides(buffer, cacheBuffPrim)) {
        std::string errorMessage;
        errorMessage += "Shape info and cache hit shape info do not match.\n";
        errorMessage += "Shape info:\n";
        errorMessage += shape::shapeToString(buffer,"\n");
        errorMessage += "\nCache hit shape info:\n";
        errorMessage += shape::shapeToString(cacheBuffPrim,"\n");
#if defined(SD_GCC_FUNCTRACE)
        Printer p;
        std::ostringstream oss;
        p.print(cacheBuff->st, oss);
        errorMessage += "=======================================================Stack trace when written.============================\n";
        errorMessage += oss.str();
        errorMessage += "=======================================================End of stack trace when written.============================\n";
#endif
        THROW_EXCEPTION(errorMessage.c_str());
      }

    }
    auto ret =  _cache[deviceId].at(*descriptor);
    return ret;
  }
}

ShapeDescriptor* ConstantShapeHelper::findBufferForShapeInfo(ShapeDescriptor *descriptor) {
  for (const auto& cache : _cache) {
    auto it = cache.find(*descriptor);
    if (it != cache.end()) {
      // Key found in the map
      auto ret = it->first;
      return new ShapeDescriptor(ret);
    }
  }

  // Key not found in any map
  return nullptr;
}
ConstantShapeBuffer* ConstantShapeHelper::bufferForShapeInfo(ShapeDescriptor* descriptor) {
  return storeAndWrapBuffer(descriptor);
}

ConstantShapeBuffer* ConstantShapeHelper::bufferForShapeInfo(const LongType* shapeInfo) {
  ShapeDescriptor* descriptor = new ShapeDescriptor(shapeInfo);
  auto ret = bufferForShapeInfo(descriptor);
  return ret;
}

bool ConstantShapeHelper::checkBufferExistenceForShapeInfo(ShapeDescriptor* descriptor) {
  auto deviceId = AffinityManager::currentDeviceId();
  std::lock_guard<std::mutex> lock(_mutex);

  return _cache[deviceId].count(*descriptor) != 0;
}

const LongType* ConstantShapeHelper::createShapeInfo(const DataType dataType, const char order, const int rank,
                                                     const LongType* shape, LongType extraProperties = -1) {
  if (extraProperties < 0) {
    extraProperties = ArrayOptions::flagForDataType(dataType);
  }

  ShapeDescriptor* descriptor = new ShapeDescriptor(dataType, order, shape, (LongType*)nullptr, rank, extraProperties);
  auto ret = bufferForShapeInfo(descriptor)->primary();
  ArrayOptions::validateSingleDataType(ArrayOptions::dataType(ret));

  // note we used to delete descriptors here. Some end up being keys in the
  // constant shape helper. We should avoid deleting these.
  return ret;
}

const LongType* ConstantShapeHelper::createShapeInfo(const DataType dataType, const LongType* shapeInfo) {
  return createShapeInfo(dataType, shape::order(shapeInfo), shape::rank(shapeInfo),
                         shape::shapeOf(const_cast<LongType*>(shapeInfo)), -1);
}

const LongType* ConstantShapeHelper::emptyShapeInfoWithShape(const DataType dataType, std::vector<LongType>& shape) {
  auto descriptor = ShapeBuilders::createShapeInfo(dataType, 'c', shape, nullptr);
  ArrayOptions::setPropertyBit(descriptor, ARRAY_EMPTY);
  auto existing = createFromExisting(descriptor);
  // note we used to delete descriptors here. Some end up being keys in the
  // constant shape helper. We should avoid deleting these.
  return existing;
}

const LongType* ConstantShapeHelper::emptyShapeInfo(const DataType dataType) {
  auto descriptor = ShapeBuilders::emptyShapeInfo(dataType, nullptr);
  auto existing = createFromExisting(descriptor);
  if (ArrayOptions::dataType(descriptor) != dataType) {
    std::string errorMessage;
    errorMessage += "ConstantShapeHelper::emptyShapeInfo: DataType mismatch. Expected ";
    errorMessage += DataTypeUtils::asString(dataType);
    errorMessage += " but got ";
    errorMessage += DataTypeUtils::asString(ArrayOptions::dataType(descriptor));
    THROW_EXCEPTION(errorMessage.c_str());
  }
  // note we used to delete descriptors here. Some end up being keys in the
  // constant shape helper. We should avoid deleting these.
  return existing;
}

const LongType* ConstantShapeHelper::scalarShapeInfo(const DataType dataType) {
  auto descriptor = ShapeBuilders::createScalarShapeInfo(dataType);
  auto ret = createFromExisting(descriptor);
  // note we used to delete descriptors here. Some end up being keys in the
  // constant shape helper. We should avoid deleting these.
  return ret;
}

const LongType* ConstantShapeHelper::vectorShapeInfo(const LongType length, const DataType dataType) {
  auto descriptor = ShapeBuilders::createVectorShapeInfo(dataType, length);
  auto ret = createFromExisting(descriptor);
  // note we used to delete descriptors here. Some end up being keys in the
  // constant shape helper. We should avoid deleting these.
  return ret;
}

const LongType* ConstantShapeHelper::createShapeInfo(const DataType dataType, const char order,
                                                     const std::vector<LongType>& shape) {
  auto ret = ShapeBuilders::createShapeInfo(dataType, order, shape, nullptr);
  auto existing = createFromExisting(ret);
  return existing;
}

const LongType* ConstantShapeHelper::createShapeInfo(ShapeDescriptor* descriptor) {
  return bufferForShapeInfo(descriptor)->primary();
}

const LongType* ConstantShapeHelper::createFromExisting(const LongType* shapeInfo, bool destroyOriginal) {
  ShapeDescriptor* descriptor = new ShapeDescriptor(shapeInfo);
  auto result = createShapeInfo(descriptor);
  // note we used to delete descriptors here. Some end up being keys in the
  // constant shape helper. We should avoid deleting these.
  return result;
}

const LongType* ConstantShapeHelper::createFromExisting(const LongType* shapeInfo, memory::Workspace* workspace) {
  ShapeDescriptor* descriptor = new ShapeDescriptor(shapeInfo);
  auto result = createShapeInfo(descriptor);
  if (Environment::getInstance().isDeleteShapeInfo()) delete descriptor;
  return result;
}

const LongType* ConstantShapeHelper::createFromExisting(LongType* shapeInfo, bool destroyOriginal) {
  ShapeDescriptor* descriptor = new ShapeDescriptor(shapeInfo);
  auto result = createShapeInfo(descriptor);
  return result;
}

const LongType* ConstantShapeHelper::createFromExisting(LongType* shapeInfo, memory::Workspace* workspace) {
  ShapeDescriptor* descriptor = new ShapeDescriptor(shapeInfo);
  auto result = createShapeInfo(descriptor);
  return result;
}

////////////////////////////////////////////////////////////////////////
ConstantShapeBuffer* ConstantShapeHelper::createShapeInfoWithUnitiesForBroadcast(
    const LongType* maxShapeInfo, const LongType* minShapeInfo, memory::Workspace* workspace,
    const std::vector<LongType>& dimensions) {
  LongType* newShapeInfo = nullptr;
  ALLOCATE(newShapeInfo, workspace, shape::shapeInfoLength(shape::rank(maxShapeInfo)), sd::LongType);

  newShapeInfo[0] = shape::rank(maxShapeInfo);
  newShapeInfo[2 * shape::rank(maxShapeInfo) + 1] = 0;
  ArrayOptions::copyDataType(newShapeInfo, minShapeInfo);                          // type
  newShapeInfo[2 * newShapeInfo[0] + 2] = shape::elementWiseStride(minShapeInfo);  // ews
  newShapeInfo[2 * newShapeInfo[0] + 3] = shape::order(minShapeInfo);              // order

  if (!dimensions.empty()) {
    for (LongType k = 0, j = 0, i = 0; i < shape::rank(maxShapeInfo); ++i) {
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

  ShapeDescriptor* descriptor = new ShapeDescriptor(newShapeInfo);
  //RELEASE(newShapeInfo, workspace);

  auto ret = bufferForShapeInfo(descriptor);
  if (Environment::getInstance().isDeleteShapeInfo())
    RELEASE(descriptor, workspace);
  return ret;
}

////////////////////////////////////////////////////////////////////////
ConstantShapeBuffer* ConstantShapeHelper::createShapeInfoWithNoUnitiesForReduce(
    const LongType* inShapeInfo, const std::vector<LongType>* dimsWithUnities, memory::Workspace* workspace) {
  LongType* newShapeInfo = nullptr;
  ALLOCATE(newShapeInfo, workspace, shape::shapeInfoLength(shape::rank(inShapeInfo) - dimsWithUnities->size()),
           sd::LongType);

  LongType temp;
  if (dimsWithUnities->size() == 1 && shape::isCommonVector(inShapeInfo, temp) && temp == dimsWithUnities->at(0)) {
    auto dims = ShapeUtils::evalDimsToExclude(shape::rank(inShapeInfo), 1, &temp);
    shape::excludeUnitiesFromShapeInfo(inShapeInfo, dims->data(), dims->size(), newShapeInfo);
    delete dims;
  } else {
    shape::excludeUnitiesFromShapeInfo(inShapeInfo, dimsWithUnities->data(), dimsWithUnities->size(), newShapeInfo);
  }

  ShapeDescriptor* descriptor = new ShapeDescriptor(newShapeInfo);

  RELEASE(newShapeInfo, workspace);

  auto ret = bufferForShapeInfo(descriptor);
 //note we used to delete descriptors here. Some end up being used
  // in the constant shape helper and should not be deleted.
  return ret;
}

////////////////////////////////////////////////////////////////////////
ConstantShapeBuffer* ConstantShapeHelper::createSubArrShapeInfo(const LongType* inShapeInfo, const LongType* dims,
                                                                const LongType dimsSize, memory::Workspace* workspace) {
  LongType* newShapeInfo = ShapeBuilders::createSubArrShapeInfo(inShapeInfo, dims, dimsSize, workspace);

  ShapeDescriptor* descriptor = new ShapeDescriptor(newShapeInfo);

  RELEASE(newShapeInfo, workspace);

  auto ret = bufferForShapeInfo(descriptor);
  return ret;
}

}  // namespace sd
