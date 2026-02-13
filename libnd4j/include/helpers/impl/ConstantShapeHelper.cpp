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
#include <string>

#ifdef SD_CUDA
#include <cuda_runtime.h>
#endif

namespace sd {

ConstantShapeHelper::~ConstantShapeHelper() {

}

ConstantShapeHelper::ConstantShapeHelper() {
}

ConstantShapeHelper& ConstantShapeHelper::getInstance() {
 static ConstantShapeHelper instance;
 return instance;
}

void ConstantShapeHelper::initializeEarly() {
  ConstantShapeHelper& instance = getInstance();
  instance._shapeTrie.waitForInitialization();
}

ConstantShapeBuffer* ConstantShapeHelper::createConstBuffFromExisting(sd::LongType* shapeInfo) {
 auto result = bufferForShapeInfo(shapeInfo);
 return result;
}

ConstantShapeBuffer* ConstantShapeHelper::bufferForShapeInfo(LongType* shapeInfo) {
 if(shapeInfo == nullptr) {
   THROW_EXCEPTION("shapeInfo is nullptr");
 }

 LongType inputRank = shape::rank(shapeInfo);
 if(inputRank < 0 || inputRank > SD_MAX_RANK) {
   std::string errorMessage = "bufferForShapeInfo: input shapeInfo has invalid rank: ";
   errorMessage += std::to_string(inputRank);
   errorMessage += " (ptr: 0x";

   // Format address as hex for easier debugging
   char addrBuf[32];
   snprintf(addrBuf, sizeof(addrBuf), "%lx", reinterpret_cast<unsigned long>(shapeInfo));
   errorMessage += addrBuf;

   // Print first few bytes in hex to help diagnose what kind of data this is
   errorMessage += ", first 8 bytes as hex: ";
   for (int i = 0; i < 8 && i < 64; i++) {
     char byteBuf[8];
     snprintf(byteBuf, sizeof(byteBuf), "%02x ", (unsigned char)(reinterpret_cast<char*>(shapeInfo)[i]));
     errorMessage += byteBuf;
   }

   errorMessage += "). This could indicate: 1) Use-after-free, 2) Memory corruption, ";
   errorMessage += "3) GPU pointer passed to CPU code, or 4) Uninitialized memory.";
   THROW_EXCEPTION(errorMessage.c_str());
 }

 const LongType MAX_REASONABLE_DIM = 1000000000LL;  // 1 billion - generous limit
 const LongType* shapeValues = shape::shapeOf(shapeInfo);
 for (int i = 0; i < inputRank; i++) {
   LongType dimValue = shapeValues[i];
   if (dimValue < 0 || dimValue > MAX_REASONABLE_DIM) {
     std::string errorMessage = "bufferForShapeInfo: SHAPE CORRUPTION DETECTED! ";
     errorMessage += "Dimension " + std::to_string(i) + " has value " + std::to_string(dimValue);
     errorMessage += " (0x";
     char hexBuf[32];
     snprintf(hexBuf, sizeof(hexBuf), "%lx", static_cast<unsigned long>(dimValue));
     errorMessage += hexBuf;
     errorMessage += ") which exceeds reasonable limit of " + std::to_string(MAX_REASONABLE_DIM);
     errorMessage += ". Full shape: [";
     for (int j = 0; j < inputRank; j++) {
       if (j > 0) errorMessage += ", ";
       errorMessage += std::to_string(shapeValues[j]);
     }
     errorMessage += "]. This indicates memory corruption - the shape buffer was overwritten ";
     errorMessage += "by garbage data (possibly a pointer value being interpreted as shape data).";
     THROW_EXCEPTION(errorMessage.c_str());
   }
 }

 auto buffer = _shapeTrie.getOrCreate(shapeInfo);
 if (buffer == nullptr) {
   THROW_EXCEPTION("bufferForShapeInfo: getOrCreate returned nullptr");
 }
 if (!buffer->isValid()) {
   std::string errorMessage = "bufferForShapeInfo: getOrCreate returned invalid ConstantShapeBuffer (magic number check failed). ";
   errorMessage += "ConstantShapeBuffer ptr: ";
   errorMessage += std::to_string(reinterpret_cast<uintptr_t>(buffer));
   THROW_EXCEPTION(errorMessage.c_str());
 }
 if (buffer->primary() == nullptr) {
   THROW_EXCEPTION("bufferForShapeInfo: getOrCreate returned buffer with nullptr primary()");
 }

 LongType* returnedShapeInfo = buffer->primary();
 LongType returnedRank = returnedShapeInfo[0];
 if (returnedRank < 0 || returnedRank > SD_MAX_RANK) {
   std::string errorMessage = "bufferForShapeInfo: RETURNED buffer contains invalid rank: ";
   errorMessage += std::to_string(returnedRank);
   errorMessage += " (input rank was: ";
   errorMessage += std::to_string(inputRank);
   errorMessage += ", input ptr: ";
   errorMessage += std::to_string(reinterpret_cast<uintptr_t>(shapeInfo));
   errorMessage += ", returned ptr: ";
   errorMessage += std::to_string(reinterpret_cast<uintptr_t>(returnedShapeInfo));
   errorMessage += ", ConstantShapeBuffer ptr: ";
   errorMessage += std::to_string(reinterpret_cast<uintptr_t>(buffer));
   errorMessage += ")";
   THROW_EXCEPTION(errorMessage.c_str());
 }

 // Verify ranks match
 if (returnedRank != inputRank) {
   std::string errorMessage = "bufferForShapeInfo: RANK MISMATCH! Input rank: ";
   errorMessage += std::to_string(inputRank);
   errorMessage += ", returned rank: ";
   errorMessage += std::to_string(returnedRank);
   errorMessage += ". This indicates cache corruption or hash collision.";
   THROW_EXCEPTION(errorMessage.c_str());
 }

 return buffer;
}
ConstantShapeBuffer* ConstantShapeHelper::createSubArrShapeInfo( sd::LongType* inShapeInfo,  LongType* dims,
                                                                 sd::LongType dimsSize) {
 sd::LongType* newShapeInfo = ShapeBuilders::createSubArrShapeInfo(inShapeInfo, dims, dimsSize, nullptr);
 auto ret = bufferForShapeInfo(newShapeInfo);
 RELEASE(newShapeInfo, nullptr);
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
 auto result = bufferForShapeInfo(descriptor)->primary();
 delete[] descriptor;  // Fix memory leak - descriptor was never freed
 return result;
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

  // BUGFIX: Must pass true to preserve strides (e.g., for transposed/permuted views)
  // Previously passed false which caused strides to be reset to contiguous
  LongType* newShapeInfo = ShapeBuilders::copyShapeInfo(shapeInfo, true, nullptr);

  ArrayOptions::setPropertyBit(newShapeInfo, ARRAY_IS_VIEW);

  auto buffer = bufferForShapeInfo(newShapeInfo);

  // Check guard bytes before freeing — detect if anything wrote past the shape info
  if (sd::Environment::getInstance().isDebug()) {
    LongType rank = newShapeInfo[0];
    if (rank >= 0 && rank <= SD_MAX_RANK) {
      LongType shapeLen = shape::shapeInfoLength(rank);
      auto guardBytes = reinterpret_cast<uint8_t*>(newShapeInfo) + (shapeLen * sizeof(LongType));
      bool guardCorrupted = false;
      size_t firstCorruptedOffset = 0;
      for (size_t i = 0; i < 64; i++) {  // check first 64 guard bytes
        if (guardBytes[i] != 0xAB) {
          guardCorrupted = true;
          firstCorruptedOffset = i;
          break;
        }
      }
      if (guardCorrupted) {
        fprintf(stderr, "\n!!! SHAPE INFO GUARD BYTES CORRUPTED in bufferForShapeInfoWithView !!!\n");
        fprintf(stderr, "  shapeInfo=%p, rank=%lld, shapeLen=%lld\n",
                newShapeInfo, static_cast<long long>(rank), static_cast<long long>(shapeLen));
        fprintf(stderr, "  First corrupted guard byte at offset %zu from shape info end\n", firstCorruptedOffset);
        fprintf(stderr, "  Guard bytes: ");
        for (size_t j = 0; j < 16; j++) {
          fprintf(stderr, "%02x ", guardBytes[j]);
        }
        fprintf(stderr, "\n");
        fflush(stderr);
      }
    }
  }

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
   delete dims;
 } else {
   shape::excludeUnitiesFromShapeInfo(maxShapeInfo, dimsWithUnities->data(), dimsWithUnities->size(), newShapeInfo);
 }

 auto ret = bufferForShapeInfo(newShapeInfo);
 RELEASE(newShapeInfo, workspace);
 return ret;
}

void ConstantShapeHelper::clearCache() {
 std::lock_guard<std::mutex> lock(_mutex);
 _shapeTrie.clearCache();
}

LongType ConstantShapeHelper::getCachedEntries() const {
 return _shapeTrie.getCachedEntries();
}

LongType ConstantShapeHelper::getCachedBytes() const {
 return _shapeTrie.getCachedBytes();
}

LongType ConstantShapeHelper::getPeakCachedEntries() const {
 return _shapeTrie.getPeakCachedEntries();
}

LongType ConstantShapeHelper::getPeakCachedBytes() const {
 return _shapeTrie.getPeakCachedBytes();
}

std::string ConstantShapeHelper::toString(int maxDepth, int maxEntries) const {
 return _shapeTrie.toString(maxDepth, maxEntries);
}

void ConstantShapeHelper::getCachedPointers(std::unordered_set<void*>& out_pointers) const {
 _shapeTrie.getCachedPointers(out_pointers);
}

} // namespace sd

