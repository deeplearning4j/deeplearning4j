/* ******************************************************************************
*
* Copyright (c) 2024 Konduit K.K.
* This program and the accompanying materials are made available under the
* terms of the Apache License, Version 2.0 which is available at
* https://www.apache.org/licenses/LICENSE-2.0.
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

#ifndef __CUDABLAS__
#include <array/PrimaryPointerDeallocator.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ShapeBuilders.h>
#include <helpers/ShapeUtils.h>
#include <helpers/shape.h>
#include <system/Environment.h>

namespace sd {

ConstantShapeHelper::~ConstantShapeHelper() {
}

ConstantShapeHelper::ConstantShapeHelper() {
}

ConstantShapeHelper& ConstantShapeHelper::getInstance() {
 static ConstantShapeHelper instance;
 return instance;
}

ConstantShapeBuffer* ConstantShapeHelper::createConstBuffFromExisting(sd::LongType* shapeInfo, sd::memory::Workspace* workspace) {
 auto result = bufferForShapeInfo(shapeInfo);
 return result;
}

ConstantShapeBuffer* ConstantShapeHelper::bufferForShapeInfo(LongType* shapeInfo) {
 auto buffer = _shapeTrie.getOrCreate(shapeInfo);
 if (!buffer || !buffer->primary()) {
   THROW_EXCEPTION("Failed to get/create shape buffer");
 }
 return buffer;
}

ConstantShapeBuffer* ConstantShapeHelper::bufferForShapeInfo(DataType dataType, char order,
                                                            const std::vector<LongType>& shape) {
 auto descriptor = ShapeBuilders::createShapeInfo(dataType, order, shape);
 return bufferForShapeInfo(descriptor);
}

ConstantShapeBuffer* ConstantShapeHelper::bufferForShapeInfo(DataType dataType, char order,
                                                            int rank,  LongType* shape) {
 auto descriptor = ShapeBuilders::createShapeInfo(dataType, order, rank, shape, nullptr, false);
 return bufferForShapeInfo(descriptor);
}

LongType* ConstantShapeHelper::emptyShapeInfoWithShape(DataType dataType, std::vector<LongType>& shape) {
 auto descriptor = ShapeBuilders::createShapeInfo(dataType, 'c', shape, nullptr);
 ArrayOptions::setPropertyBit(descriptor, ARRAY_EMPTY);
 auto existing = createFromExisting(descriptor, false);
 return existing;
}

LongType* ConstantShapeHelper::createShapeInfo(DataType dataType, char order,
                                              const std::vector<LongType>& shape) {
 auto descriptor = ShapeBuilders::createShapeInfo(dataType, order, shape);
 return bufferForShapeInfo(descriptor)->primary();
}

LongType* ConstantShapeHelper::createShapeInfo(DataType dataType, char order, int rank,
                                              LongType* shape, LongType extraProperties) {
 if (extraProperties < 0) {
   extraProperties = ArrayOptions::flagForDataType(dataType);
 }

 std::unique_ptr<LongType[]> strides(order == 'c' ? shape::calcStrides(shape, rank)
                                                  : shape::calcStridesFortran(shape, rank));

 auto descriptor = ShapeBuilders::createShapeInfo(dataType, order, rank, shape, strides.get(),
                                                  nullptr, extraProperties);
 auto ret = bufferForShapeInfo(descriptor)->primary();
 ArrayOptions::validateSingleDataType(ArrayOptions::dataType(ret));

 return ret;
}

LongType* ConstantShapeHelper::createShapeInfo(DataType dataType, LongType* shapeInfo) {
 return createShapeInfo(dataType, shape::order(shapeInfo), shape::rank(shapeInfo),
                        shape::shapeOf(const_cast<LongType*>(shapeInfo)), -1);
}

LongType* ConstantShapeHelper::createShapeInfo(ShapeDescriptor* descriptor) {
 return bufferForShapeInfo(descriptor->toShapeInfo())->primary();
}

LongType* ConstantShapeHelper::emptyShapeInfo(DataType dataType) {
 auto descriptor = ShapeBuilders::emptyShapeInfo(dataType);
 return bufferForShapeInfo(descriptor)->primary();
}

LongType* ConstantShapeHelper::scalarShapeInfo(DataType dataType) {
 auto descriptor = ShapeBuilders::createScalarShapeInfo(dataType);
 return bufferForShapeInfo(descriptor)->primary();
}

LongType* ConstantShapeHelper::vectorShapeInfo(LongType length, DataType dataType) {
 auto descriptor = ShapeBuilders::createVectorShapeInfo(dataType, length);
 return bufferForShapeInfo(descriptor)->primary();
}

LongType* ConstantShapeHelper::createFromExisting(LongType* shapeInfo, bool destroyOriginal) {
 if (!shapeInfo) {
   THROW_EXCEPTION("Null shape info");
 }
 auto buffer = bufferForShapeInfo(shapeInfo);
 return buffer->primary();
}

LongType* ConstantShapeHelper::createFromExisting(LongType* shapeInfo, sd::memory::Workspace* workspace) {
 auto result = bufferForShapeInfo(shapeInfo);
 return result->primary();
}

LongType* ConstantShapeHelper::castToDataType(LongType* shapeInfo, DataType newType) {
 if (!shapeInfo) {
   THROW_EXCEPTION("Null shape info");
 }
 if (ArrayOptions::dataType(shapeInfo) == newType) {
   return shapeInfo;
 }

 auto tempShapeInfo = ShapeBuilders::copyShapeInfoWithNewType(shapeInfo, newType);
 if (!tempShapeInfo) {
   THROW_EXCEPTION("Failed to create temp shape info");
 }

 auto buffer = bufferForShapeInfo(tempShapeInfo);
 if(ArrayOptions::dataType(buffer->primary()) != newType) {
   std::string errorMessage;
   errorMessage += "castToDataType: new data type is ";
   errorMessage += DataTypeUtils::asString(newType);
   errorMessage += " data type from new constant created data type ";
   errorMessage += DataTypeUtils::asString(ArrayOptions::dataType(buffer->primary()));
   errorMessage += "\n";
   THROW_EXCEPTION(errorMessage.c_str());
 }
 return buffer->primary();
}

bool ConstantShapeHelper::checkBufferExistenceForShapeInfo(ShapeDescriptor* descriptor) {
 return _shapeTrie.exists(descriptor->toShapeInfo());
}

ConstantShapeBuffer* ConstantShapeHelper::createShapeInfoWithUnitiesForBroadcast(sd::LongType* maxShapeInfo,
                                                                                sd::LongType* minShapeInfo,
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
     if (j < static_cast<sd::LongType>(dimensions.size()) && dimensions[j] == i) {
       shape::shapeOf(newShapeInfo)[i] = shape::shapeOf(minShapeInfo)[k];
       shape::stride(newShapeInfo)[i] = shape::stride(minShapeInfo)[k++];
       ++j;
     } else {
       shape::shapeOf(newShapeInfo)[i] = 1;
       shape::stride(newShapeInfo)[i] = 0;
       if (shape::sizeAt(minShapeInfo, k) == 1 && static_cast<sd::LongType>(dimensions.size()) != shape::rank(minShapeInfo)) ++k;
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

 auto ret = bufferForShapeInfo(newShapeInfo);
 return ret;
}

ConstantShapeBuffer* ConstantShapeHelper::createShapeInfoWithNoUnitiesForReduce(sd::LongType* maxShapeInfo,
                                                                               const std::vector<LongType>* dimsWithUnities,
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

 auto ret = bufferForShapeInfo(newShapeInfo);
 return ret;
}



} // namespace sd

#endif