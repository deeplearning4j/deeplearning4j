/* ******************************************************************************
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


ConstantShapeHelper::~ConstantShapeHelper() {
}

ConstantShapeHelper::ConstantShapeHelper() {
}

ConstantShapeHelper& ConstantShapeHelper::getInstance() {
 static ConstantShapeHelper instance;
 return instance;
}

const sd::LongType * ConstantShapeHelper::emptyShapeInfoWithShape(const sd::DataType dataType, std::vector<sd::LongType> &shape) {
 auto descriptor = ShapeBuilders::createShapeInfo(dataType,'c', shape, nullptr);
 ArrayOptions::setPropertyBit(descriptor, ARRAY_EMPTY);
 auto existing = createFromExisting(descriptor);
 return existing;
}




ConstantShapeBuffer* ConstantShapeHelper::bufferForShapeInfo(sd::DataType dataType, char order,
                                                            const std::vector<sd::LongType>& shape) {
 auto descriptor = ShapeBuilders::createShapeInfo(dataType, order, shape);
 auto ret = bufferForShapeInfo(descriptor);
 return ret;
}

ConstantShapeBuffer* ConstantShapeHelper::bufferForShapeInfo(const sd::DataType dataType, const char order,
                                                            const int rank, const sd::LongType* shape) {
 auto descriptor = ShapeBuilders::createShapeInfo(dataType,order,rank,shape,nullptr,false);
 auto ret = bufferForShapeInfo(descriptor);
 return ret;
}

ConstantShapeBuffer* ConstantShapeHelper::storeAndWrapBuffer(const LongType* shapeInfo) {
 if (shapeInfo == nullptr) {
   THROW_EXCEPTION("Unable to create and store a shape buffer with null shape info.");
 }

 // Create early copy for validation
 const int shapeInfoLength = shape::shapeInfoLength(shape::rank(shapeInfo));
 LongType* buffer = new LongType[shapeInfoLength];
 std::memcpy(buffer, shapeInfo, shapeInfoLength * sizeof(LongType));

 if (ArrayOptions::dataType(buffer) == DataType::UNKNOWN) {
   delete[] buffer;
   THROW_EXCEPTION("Unable to create and store a shape buffer with unknown data type.");
 }

 // Validate shape info
 if(Environment::getInstance().isDebug() || Environment::getInstance().isVerbose()) {
   if(!shape::haveSameShapeAndStrides(shapeInfo, buffer)) {
     std::string errorMessage;
     errorMessage += "Shape info validation failed:\n";
     errorMessage += "Original shape info:\n";
     errorMessage += shape::shapeToString(shapeInfo,"\n");
     errorMessage += "\nBuffer shape info:\n";
     errorMessage += shape::shapeToString(buffer,"\n");
     delete[] buffer;
     THROW_EXCEPTION(errorMessage.c_str());
   }
 }

 delete[] buffer;  // Clean up temporary buffer

 // Use DirectShapeTrie to handle storage and caching
 return _shapeTrie.getOrCreate(shapeInfo);
}

ConstantShapeBuffer* ConstantShapeHelper::bufferForShapeInfo(ShapeDescriptor *descriptor) {
 return storeAndWrapBuffer(descriptor->toShapeInfo());
}

ConstantShapeBuffer* ConstantShapeHelper::bufferForShapeInfo(const sd::LongType* shapeInfo) {
 if (shapeInfo == nullptr) {
   THROW_EXCEPTION("Unable to create shape buffer with null shape info.");
 }

 if (ArrayOptions::dataType(shapeInfo) == DataType::UNKNOWN) {
   THROW_EXCEPTION("Unable to create array with unknown data type.");
 }

 auto ret = _shapeTrie.getOrCreate(shapeInfo);
 auto retTest = ret->primary();

 if(!shape::haveSameShapeAndStrides(shapeInfo, retTest)) {
   std::string errorMessage;
   errorMessage += "Attempting to store Shape info and cache buffer shape info that do not match: \n";
   errorMessage += "Shape info:\n";
   errorMessage += shape::shapeToString(shapeInfo,"\n");
   errorMessage += "\nCache buffer shape info:\n";
   errorMessage += shape::shapeToString(retTest,"\n");
   THROW_EXCEPTION(errorMessage.c_str());
 }

 return ret;
}

bool ConstantShapeHelper::checkBufferExistenceForShapeInfo(ShapeDescriptor *descriptor) {
 return _shapeTrie.exists(descriptor->toShapeInfo());
}





const sd::LongType* ConstantShapeHelper::createShapeInfo(const sd::DataType dataType, const char order, const int rank,
                                                        const sd::LongType* shape, LongType extraProperties) {
 if(extraProperties < 0) {
   extraProperties = ArrayOptions::flagForDataType(dataType);
 }

 sd::LongType  *strides = order == 'c' ? shape::calcStrides(shape,rank) : shape::calcStridesFortran(shape,rank);
 sd::LongType  *descriptor =
     ShapeBuilders::createShapeInfo(dataType, order,rank,shape,strides,nullptr,extraProperties);
 delete[] strides;
 auto ret = bufferForShapeInfo(descriptor)->primary();
 ArrayOptions::validateSingleDataType(ArrayOptions::dataType(ret));

 return ret;
}

const sd::LongType* ConstantShapeHelper::createShapeInfo(const sd::DataType dataType, const sd::LongType* shapeInfo) {
 return ConstantShapeHelper::createShapeInfo(dataType, shape::order(shapeInfo), shape::rank(shapeInfo),
                                             shape::shapeOf(const_cast<sd::LongType*>(shapeInfo)), -1);
}

const sd::LongType* ConstantShapeHelper::emptyShapeInfo(const sd::DataType dataType) {
 auto descriptor = ShapeBuilders::emptyShapeInfo(dataType);
 auto ret = bufferForShapeInfo(descriptor)->primary();
 return ret;
}

const sd::LongType* ConstantShapeHelper::scalarShapeInfo(const sd::DataType dataType) {
 auto descriptor = ShapeBuilders::createScalarShapeInfo(dataType);
 auto ret = bufferForShapeInfo(descriptor)->primary();
 return ret;
}

const sd::LongType* ConstantShapeHelper::vectorShapeInfo(const sd::LongType length, const sd::DataType dataType) {
 auto descriptor = ShapeBuilders::createVectorShapeInfo(dataType,length);
 auto ret = bufferForShapeInfo(descriptor)->primary();
 return ret;
}

const sd::LongType* ConstantShapeHelper::createShapeInfo(const sd::DataType dataType, const char order,
                                                        const std::vector<sd::LongType>& shape) {
 sd::LongType *descriptor = ShapeBuilders::createShapeInfo(dataType,order,shape);
 auto ret = bufferForShapeInfo(descriptor)->primary();
 return ret;
}

const sd::LongType* ConstantShapeHelper::createShapeInfo(ShapeDescriptor* descriptor) {
 return bufferForShapeInfo(descriptor->toShapeInfo())->primary();
}

const LongType* ConstantShapeHelper::createFromExisting(const sd::LongType* shapeInfo, bool destroyOriginal) {
 auto result = bufferForShapeInfo(shapeInfo)->primary();
 return result;
}

const LongType* ConstantShapeHelper::createFromExisting(const sd::LongType* shapeInfo, sd::memory::Workspace* workspace) {
 auto result = bufferForShapeInfo(shapeInfo);
 return result->primary();
}

const LongType* ConstantShapeHelper::createFromExisting(sd::LongType* shapeInfo, bool destroyOriginal) {
 ShapeDescriptor *descriptor = new ShapeDescriptor(shapeInfo, false);
 auto result = createShapeInfo(descriptor);
 if(destroyOriginal) {
   RELEASE(const_cast<sd::LongType*>(shapeInfo), nullptr);
 }
 return result;
}

const LongType* ConstantShapeHelper::createFromExisting(sd::LongType* shapeInfo, sd::memory::Workspace* workspace) {
 ShapeDescriptor *descriptor = new ShapeDescriptor(shapeInfo, false);
 auto result = createShapeInfo(descriptor);
 return result;
}


const LongType* ConstantShapeHelper::castToDataType(const LongType* shapeInfo, const DataType newType) {
 if (shapeInfo == nullptr) {
   THROW_EXCEPTION("Unable to cast data type of null shape info.");
 }

 // If the current type is the same as the target type, return the original
 if (ArrayOptions::dataType(shapeInfo) == newType) {
   return shapeInfo;
 }

 // Create a temporary shape info with the new type to search in the trie
 LongType* tempShapeInfo = ShapeBuilders::copyShapeInfoWithNewType(shapeInfo, newType);

 // Check if we already have this shape info in the trie
 if (_shapeTrie.exists(tempShapeInfo)) {
   // Found in trie, clean up temp and return existing
   auto buffer = _shapeTrie.getOrCreate(tempShapeInfo);
   delete[] tempShapeInfo;
   return buffer->primary();
 }

 // Not found in trie, store the temporary shape info
 auto buffer = _shapeTrie.getOrCreate(tempShapeInfo);
 delete[] tempShapeInfo;
 return buffer->primary();
}

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

 auto ret = bufferForShapeInfo(newShapeInfo);
 return ret;
}

ConstantShapeBuffer* ConstantShapeHelper::createShapeInfoWithNoUnitiesForReduce(const sd::LongType* maxShapeInfo,
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

ConstantShapeBuffer* ConstantShapeHelper::createSubArrShapeInfo(const sd::LongType* inShapeInfo, const LongType* dims,
                                                               const sd::LongType dimsSize, sd::memory::Workspace* workspace) {
 sd::LongType* newShapeInfo = ShapeBuilders::createSubArrShapeInfo(inShapeInfo, dims, dimsSize, workspace);
 auto ret = bufferForShapeInfo(newShapeInfo);
 return ret;
}

ConstantShapeBuffer* ConstantShapeHelper::createConstBuffFromExisting(const sd::LongType* shapeInfo, sd::memory::Workspace* workspace) {
 auto result = bufferForShapeInfo(shapeInfo);
 return result;
}

} // namespace sd

#endif