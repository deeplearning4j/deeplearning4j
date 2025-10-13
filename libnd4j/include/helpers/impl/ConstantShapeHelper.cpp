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

#include "helpers/ConstantShapeHelper.h"

#include "array/ConstantShapeBuffer.h"
#include "system/common.h"
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

ConstantShapeBuffer* ConstantShapeHelper::createConstBuffFromExisting(sd::LongType* shapeInfo) {
 auto result = bufferForShapeInfo(shapeInfo);
 return result;
}

ConstantShapeBuffer* ConstantShapeHelper::bufferForShapeInfo(LongType* shapeInfo) {
 if(shapeInfo == nullptr) {
   THROW_EXCEPTION("shapeInfo is nullptr");
 }
 if(shape::rank(shapeInfo) < 0 || shape::rank(shapeInfo) > SD_MAX_RANK) {
   THROW_EXCEPTION("shapeInfo is not a valid rank.");
 }

 auto buffer = _shapeTrie.getOrCreate(shapeInfo);
 if (buffer == nullptr || buffer->primary() == nullptr) {
   THROW_EXCEPTION("Failed to get/create shape buffer");
 }
 return buffer;
}
ConstantShapeBuffer* ConstantShapeHelper::createSubArrShapeInfo( sd::LongType* inShapeInfo,  LongType* dims,
                                                                 sd::LongType dimsSize) {
 sd::LongType* newShapeInfo = ShapeBuilders::createSubArrShapeInfo(inShapeInfo, dims, dimsSize, nullptr);
 auto ret = bufferForShapeInfo(newShapeInfo);
 delete[] newShapeInfo;
 return ret;
}

ConstantShapeBuffer* ConstantShapeHelper::bufferForShapeInfo(DataType dataType, char order,
                                                            const std::vector<LongType>& shape) {
 auto descriptor = ShapeBuilders::createShapeInfo(dataType, order, shape);
 auto result = bufferForShapeInfo(descriptor);
 delete[] descriptor;
 return result;
}

ConstantShapeBuffer* ConstantShapeHelper::bufferForShapeInfo(DataType dataType, char order,
                                                            int rank,  LongType* shape) {
 auto descriptor = ShapeBuilders::createShapeInfo(dataType, order, rank, shape, nullptr, false);
 auto result = bufferForShapeInfo(descriptor);
 delete[] descriptor;
 return result;
}

LongType* ConstantShapeHelper::emptyShapeInfoWithShape(DataType dataType, std::vector<LongType>& shape) {
 auto descriptor = ShapeBuilders::createShapeInfo(dataType, 'c', shape, nullptr);
 ArrayOptions::setPropertyBit(descriptor, ARRAY_EMPTY);
 auto existing = createFromExisting(descriptor);
 delete[] descriptor;
 return existing;
}

LongType* ConstantShapeHelper::createShapeInfo(DataType dataType, char order,
                                              const std::vector<LongType>& shape) {
 auto descriptor = ShapeBuilders::createShapeInfo(dataType, order, shape);
 auto result = bufferForShapeInfo(descriptor)->primary();
 delete[] descriptor;
 return result;
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

 delete[] descriptor;
 return ret;
}

LongType* ConstantShapeHelper::createShapeInfo(DataType dataType, LongType* shapeInfo) {
 auto result = createShapeInfo(dataType, shape::order(shapeInfo), shape::rank(shapeInfo),
                        shape::shapeOf(const_cast<LongType*>(shapeInfo)), -1);
 return result;
}

LongType* ConstantShapeHelper::emptyShapeInfo(DataType dataType) {
 auto descriptor = ShapeBuilders::emptyShapeInfo(dataType);
 auto result = bufferForShapeInfo(descriptor)->primary();
 delete[] descriptor;
 return result;
}


LongType* ConstantShapeHelper::scalarShapeInfo(DataType dataType) {
 auto descriptor = ShapeBuilders::createScalarShapeInfo(dataType);
 return bufferForShapeInfo(descriptor)->primary();
}

LongType* ConstantShapeHelper::vectorShapeInfo(LongType length, DataType dataType) {
 auto descriptor = ShapeBuilders::createVectorShapeInfo(dataType, length);
 auto result = bufferForShapeInfo(descriptor)->primary();
 delete[] descriptor;
 return result;
}


LongType* ConstantShapeHelper::createShapeInfo(ShapeDescriptor* descriptor) {
 auto shapeInfo = descriptor->toShapeInfo();
 auto result = bufferForShapeInfo(shapeInfo)->primary();
 delete[] shapeInfo;
 return result;
}


ConstantShapeBuffer* ConstantShapeHelper::bufferForShapeInfoWithView(LongType* shapeInfo) {
 if (shapeInfo == nullptr) {
   THROW_EXCEPTION("shapeInfo is nullptr");
 }

 LongType* newShapeInfo = ShapeBuilders::copyShapeInfo(shapeInfo, false, nullptr);



 ArrayOptions::setPropertyBit(newShapeInfo, ARRAY_IS_VIEW);

 auto buffer = bufferForShapeInfo(newShapeInfo);

 delete[] newShapeInfo;

 return buffer;
}

ConstantShapeBuffer* ConstantShapeHelper::bufferForShapeInfoWithoutView(LongType* shapeInfo) {
 if (shapeInfo == nullptr) {
   THROW_EXCEPTION("shapeInfo is nullptr");
 }

 LongType* newShapeInfo = ShapeBuilders::copyShapeInfo(shapeInfo, false, nullptr);
 ArrayOptions::unsetPropertyBit(newShapeInfo, ARRAY_IS_VIEW);

 auto buffer = bufferForShapeInfo(newShapeInfo);
 delete[] newShapeInfo;
 return buffer;
}

ConstantShapeBuffer* ConstantShapeHelper::bufferForShapeInfoWithNeedsCopy(LongType* shapeInfo) {
 if (shapeInfo == nullptr) {
   THROW_EXCEPTION("shapeInfo is nullptr");
 }

 LongType* newShapeInfo = ShapeBuilders::copyShapeInfo(shapeInfo, false, nullptr);
 ArrayOptions::setPropertyBit(newShapeInfo, ARRAY_NEEDS_COPY);

 auto buffer = bufferForShapeInfo(newShapeInfo);
 delete[] newShapeInfo;
 return buffer;
}

ConstantShapeBuffer* ConstantShapeHelper::bufferForShapeInfoWithoutNeedsCopy(LongType* shapeInfo) {
 if (shapeInfo == nullptr) {
   THROW_EXCEPTION("shapeInfo is nullptr");
 }

 LongType* newShapeInfo = ShapeBuilders::copyShapeInfo(shapeInfo, false, nullptr);
 ArrayOptions::unsetPropertyBit(newShapeInfo, ARRAY_NEEDS_COPY);

 auto buffer = bufferForShapeInfo(newShapeInfo);
 delete[] newShapeInfo;
 return buffer;
}

ConstantShapeBuffer* ConstantShapeHelper::bufferForShapeInfoWithCopyOffset(LongType* shapeInfo, int inputIndex) {
 if (shapeInfo == nullptr) {
   THROW_EXCEPTION("shapeInfo is nullptr");
 }

 if (inputIndex < 0 || inputIndex > 10) {
   THROW_EXCEPTION("Input index out of range [0-10]");
 }

 LongType* newShapeInfo = ShapeBuilders::copyShapeInfo(shapeInfo, false, nullptr);
 LongType flag = ArrayOptions::copyOffsetFlagForInput(inputIndex);
 ArrayOptions::setPropertyBit(newShapeInfo, flag);

 auto buffer = bufferForShapeInfo(newShapeInfo);
 delete[] newShapeInfo;
 return buffer;
}

ConstantShapeBuffer* ConstantShapeHelper::bufferForShapeInfoWithoutCopyOffset(LongType* shapeInfo, int inputIndex) {
 if (shapeInfo == nullptr) {
   THROW_EXCEPTION("shapeInfo is nullptr");
 }

 if (inputIndex < 0 || inputIndex > 10) {
   THROW_EXCEPTION("Input index out of range [0-10]");
 }

 LongType* newShapeInfo = ShapeBuilders::copyShapeInfo(shapeInfo, false, nullptr);
 LongType flag = ArrayOptions::copyOffsetFlagForInput(inputIndex);
 ArrayOptions::unsetPropertyBit(newShapeInfo, flag);

 auto buffer = bufferForShapeInfo(newShapeInfo);
 delete[] newShapeInfo;
 return buffer;
}

ConstantShapeBuffer* ConstantShapeHelper::bufferForShapeInfoWithoutAllCopyOffsets(LongType* shapeInfo) {
 if (shapeInfo == nullptr) {
   THROW_EXCEPTION("shapeInfo is nullptr");
 }

 LongType* newShapeInfo = ShapeBuilders::copyShapeInfo(shapeInfo, false, nullptr);
 ArrayOptions::clearAllCopyOffsets(newShapeInfo);

 auto buffer = bufferForShapeInfo(newShapeInfo);
 delete[] newShapeInfo;
 return buffer;
}

ConstantShapeBuffer* ConstantShapeHelper::bufferForShapeInfoWithFlags(LongType* shapeInfo,
                                                                      LongType flagsToSet,
                                                                      LongType flagsToUnset) {
 if (shapeInfo == nullptr) {
   THROW_EXCEPTION("shapeInfo is nullptr");
 }

 LongType* newShapeInfo = ShapeBuilders::copyShapeInfo(shapeInfo, false, nullptr);

 // Unset flags first
 if (flagsToUnset != 0) {
   LongType extraIdx = ArrayOptions::extraIndex(newShapeInfo);
   newShapeInfo[extraIdx] = newShapeInfo[extraIdx] & ~flagsToUnset;
 }

 // Then set flags
 if (flagsToSet != 0) {
   LongType extraIdx = ArrayOptions::extraIndex(newShapeInfo);
   newShapeInfo[extraIdx] = newShapeInfo[extraIdx] | flagsToSet;
 }

 auto buffer = bufferForShapeInfo(newShapeInfo);
 delete[] newShapeInfo;
 return buffer;
}

ConstantShapeBuffer* ConstantShapeHelper::bufferForShapeInfoAsViewWithOffset(LongType* shapeInfo,
                                                                             int inputIndex) {
 if (shapeInfo == nullptr) {
   THROW_EXCEPTION("shapeInfo is nullptr");
 }

 if (inputIndex < 0 || inputIndex > 10) {
   THROW_EXCEPTION("Input index out of range [0-10]");
 }

 LongType* newShapeInfo = ShapeBuilders::copyShapeInfo(shapeInfo, false, nullptr);

 // Set view flag
 ArrayOptions::setPropertyBit(newShapeInfo, ARRAY_IS_VIEW);

 // Set copy offset flag for specified input
 LongType flag = ArrayOptions::copyOffsetFlagForInput(inputIndex);
 ArrayOptions::setPropertyBit(newShapeInfo, flag);

 auto buffer = bufferForShapeInfo(newShapeInfo);
 delete[] newShapeInfo;
 return buffer;
}

LongType* ConstantShapeHelper::createFromExisting(LongType* shapeInfo) {
 if (!shapeInfo) {
   THROW_EXCEPTION("Null shape info");
 }
 auto buffer = bufferForShapeInfo(shapeInfo);
 return buffer->primary();
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
 auto result = buffer->primary();
 delete[] tempShapeInfo;
 if(ArrayOptions::dataType(result) != newType) {
   std::string errorMessage;
   errorMessage += "castToDataType: new data type is ";
   errorMessage += DataTypeUtils::asString(newType);
   errorMessage += " data type from new constant created data type ";
   errorMessage += DataTypeUtils::asString(ArrayOptions::dataType(result));
   errorMessage += "\n";
   THROW_EXCEPTION(errorMessage.c_str());
 }
 return result;
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
 RELEASE(newShapeInfo, workspace);
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
   delete dims;
 } else {
   shape::excludeUnitiesFromShapeInfo(maxShapeInfo, dimsWithUnities->data(), dimsWithUnities->size(), newShapeInfo);
 }

 auto ret = bufferForShapeInfo(newShapeInfo);
 RELEASE(newShapeInfo, workspace);
 return ret;
}



} // namespace sd


