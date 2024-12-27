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

#include "../ConstantTadHelper.h"

#include <array/ConstantOffsetsBuffer.h>
#include <array/PrimaryPointerDeallocator.h>
#include <helpers/ShapeUtils.h>


#ifndef __CUDABLAS__

namespace sd {

/* Old implementation (commented out)
ConstantTadHelper::ConstantTadHelper() {
  SD_MAP_IMPL<TadDescriptor *, TadPack *> pack;
  _cache.emplace_back(pack);
}
*/

// New trie-based implementation
ConstantTadHelper::ConstantTadHelper() {} // Default constructor is fine now

ConstantTadHelper &ConstantTadHelper::getInstance() {
  static ConstantTadHelper instance;
  return instance;
}

TadPack *ConstantTadHelper::tadForDimensions(const sd::LongType *originalShape, LongType dimension,
                                           const bool keepUnitiesInShape) {
  return tadForDimensions(originalShape, &dimension, 1, keepUnitiesInShape);
}

TadPack *ConstantTadHelper::tadForDimensions(const sd::LongType *originalShape, const std::vector<LongType> *dimensions,
                                           const bool keepUnitiesInShape) {
  return tadForDimensions(originalShape, const_cast<sd::LongType *>(dimensions->data()), dimensions->size(), keepUnitiesInShape);
}

TadPack *ConstantTadHelper::tadForDimensions(const sd::LongType *originalShape, LongType *dimensions, LongType dimLength,
                                           const bool keepUnitiesInShape) {
  std::vector<LongType> dims(dimensions, dimensions + dimLength);
  std::lock_guard<std::mutex> lock(_mutex);
  
  /* Old implementation (commented out)
  TadDescriptor *tadDescriptor = new TadDescriptor(originalShape, dimensions, dimLength, keepUnitiesInShape);
  if(tadDescriptor == nullptr)
    THROW_EXCEPTION("ConstantTadHelper::tadForDimensions: descriptor is nullptr!");
  
  if (_cache[deviceId].count(tadDescriptor) == 0) {
    // if there's no TadPack matching this descriptor - create one
    const auto shapeInfo = descriptor->originalShape().toShapeInfo();
    // ... rest of the old implementation
  }
  return _cache[deviceId][tadDescriptor];
  */

  // New trie-based implementation
  TadPack* existingPack = _trie.getOrCreate(dims);
  if (existingPack != nullptr) {
    return existingPack;
  }
  
  // If no existing pack, create a new one
  const sd::LongType rank = shape::rank(originalShape);
  const std::vector<sd::LongType> *dimsToExclude = ShapeUtils::evalDimsToExclude(rank, dimLength, dimensions);

  const sd::LongType numOfSubArrs = ShapeUtils::getNumOfSubArrs(originalShape, *dimsToExclude);
  const sd::LongType subArrRank = (rank == dimsToExclude->size() || keepUnitiesInShape) ? rank : rank - dimsToExclude->size();

  auto sPtr = std::make_shared<PointerWrapper>(
      new sd::LongType[shape::shapeInfoLength(subArrRank)],
      std::make_shared<PrimaryPointerDeallocator>());

  std::shared_ptr<PointerWrapper> oPtr;
  if (numOfSubArrs > 0)
    oPtr = std::make_shared<PointerWrapper>(new sd::LongType[numOfSubArrs], std::make_shared<PrimaryPointerDeallocator>());
  else {
    oPtr = std::make_shared<PointerWrapper>(new sd::LongType[1], std::make_shared<PrimaryPointerDeallocator>());
    oPtr->pointerAsT<sd::LongType>()[0] = 0;
  }

  if (numOfSubArrs > 0) {
    shape::calcSubArrsShapeInfoAndOffsets(originalShape, numOfSubArrs, dimsToExclude->size(), dimsToExclude->data(),
                                        sPtr->pointerAsT<sd::LongType>(), oPtr->pointerAsT<sd::LongType>(),
                                        keepUnitiesInShape);
  } else {
    sd::LongType *shapeInfoCopy = new sd::LongType[shape::shapeInfoLength(rank)];
    memcpy(shapeInfoCopy, originalShape, shape::shapeInfoByteLength(rank));
    sd::LongType *sPtrInfo = sPtr->pointerAsT<sd::LongType>();
    shape::copyTo(shape::shapeInfoLength(rank), shapeInfoCopy, sPtrInfo);
    delete[] shapeInfoCopy;
  }

  const ConstantShapeBuffer shapeBuffer(sPtr);
  const ConstantOffsetsBuffer offsetsBuffer(oPtr);
  TadPack *t = new TadPack(shapeBuffer, offsetsBuffer, numOfSubArrs, dimensions, dimLength);

  // Store in trie
  auto* node = _trie.getOrCreateNode(dims);
  node->_tadPack.store(t, std::memory_order_release);
  _trie.incrementStripeCount(_trie.computeStripeIndex(dims));
  return t;
}

TadPack *ConstantTadHelper::tadForDimensions(ShapeDescriptor &descriptor, std::vector<LongType> &dimensions,
                                           const bool keepUnitiesInShape) {
  return tadForDimensions(descriptor.toShapeInfo(), dimensions.data(), dimensions.size(), keepUnitiesInShape);
}

TadPack *ConstantTadHelper::tadForDimensions(TadDescriptor *descriptor) {
  return tadForDimensions(descriptor->originalShape().toShapeInfo(), 
                         descriptor->axis().data(), 
                         descriptor->axis().size(),
                         descriptor->areUnitiesinShape());
}

}  // namespace sd

#endif