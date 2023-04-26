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
#include "../ConstantTadHelper.h"

#include <array/ConstantOffsetsBuffer.h>
#include <array/PrimaryPointerDeallocator.h>
#include <helpers/ShapeUtils.h>
#include <helpers/TAD.h>

#ifndef __CUDABLAS__

namespace sd {

ConstantTadHelper::ConstantTadHelper() {
  SD_MAP_IMPL<TadDescriptor *, TadPack *> pack;
  _cache.emplace_back(pack);
}

ConstantTadHelper &ConstantTadHelper::getInstance() {
  static ConstantTadHelper instance;
  return instance;
}

TadPack *ConstantTadHelper::tadForDimensions(const sd::LongType *originalShape, LongType dimension,
                                             const bool keepUnitiesInShape) {
  return tadForDimensions(originalShape, &dimension, 1, keepUnitiesInShape);
}

TadPack *ConstantTadHelper::tadForDimensions(const sd::LongType *originalShape, const std::vector<LongType> &dimensions,
                                             const bool keepUnitiesInShape) {
  return tadForDimensions(originalShape, const_cast<sd::LongType *>(dimensions.data()), dimensions.size(), keepUnitiesInShape);
}

TadPack *ConstantTadHelper::tadForDimensions(const sd::LongType *originalShape, LongType *dimensions, LongType dimLength,
                                             const bool keepUnitiesInShape) {
  TadDescriptor *tadDescriptor = new TadDescriptor(originalShape, dimensions, dimLength, keepUnitiesInShape);
  return tadForDimensions(tadDescriptor);
}

TadPack *ConstantTadHelper::tadForDimensions(ShapeDescriptor &descriptor, std::vector<LongType> &dimensions,
                                             const bool keepUnitiesInShape) {
  TadDescriptor *tadDescriptor = new TadDescriptor(descriptor, dimensions, keepUnitiesInShape);
  return tadForDimensions(tadDescriptor);
}

TadPack *ConstantTadHelper::tadForDimensions(TadDescriptor *descriptor) {
/*  const int deviceId = 0;
  if(descriptor == nullptr)
    throw std::runtime_error("ConstantTadHelper::tadForDimensions: descriptor is nullptr!");
  std::lock_guard<std::mutex> lock(_mutex);
  if (_cache[deviceId].count(descriptor) == 0) {
    sd_printf("Creating new tad\n",0);
    // if there's no TadPack matching this descriptor - create one
    const auto shapeInfo = descriptor->originalShape().toShapeInfo();
    const int rank = shape::rank(shapeInfo);
    const std::vector<sd::LongType> dimsToExclude = ShapeUtils::evalDimsToExclude(rank, descriptor->axis());
    const sd::LongType numOfSubArrs = ShapeUtils::getNumOfSubArrs(shapeInfo, dimsToExclude);
    const int subArrRank =
        (rank == dimsToExclude.size() || descriptor->areUnitiesinShape()) ? rank : rank - dimsToExclude.size();

    auto sPtr = std::make_shared<PointerWrapper>(
        new sd::LongType[shape::shapeInfoLength(subArrRank)],
        std::make_shared<PrimaryPointerDeallocator>());  // shape of sub-arrays (same for all for them)
    auto oPtr =
        std::make_shared<PointerWrapper>(new sd::LongType[numOfSubArrs], std::make_shared<PrimaryPointerDeallocator>());

    if (numOfSubArrs > 0)
      shape::calcSubArrsShapeInfoAndOffsets(shapeInfo, numOfSubArrs, dimsToExclude.size(), dimsToExclude.data(),
                                            sPtr->pointerAsT<sd::LongType>(), oPtr->pointerAsT<sd::LongType>(),
                                            descriptor->areUnitiesinShape());

    ConstantShapeBuffer shapeBuffer(sPtr);
    ConstantOffsetsBuffer offsetsBuffer(oPtr);
    TadPack *t = new TadPack(shapeBuffer, offsetsBuffer, numOfSubArrs);

    _cache[deviceId][descriptor] = t;
    sd_printf("Created new tad\n",0);

  }
  sd_printf("Returning tad\n",0);
  return _cache[deviceId][descriptor];*/

  sd_printf("Creating new tad\n",0);
// if there's no TadPack matching this descriptor - create one
  const auto shapeInfo = descriptor->originalShape().toShapeInfo();
  sd_printf("After shape info creation\n",0);
  const int rank = shape::rank(shapeInfo);
  sd_printf("After rank %d\n",rank);
  const std::vector<sd::LongType> dimsToExclude = ShapeUtils::evalDimsToExclude(rank, descriptor->axis());
  sd_printf("After dims to exclude %d\n",rank);

  const sd::LongType numOfSubArrs = ShapeUtils::getNumOfSubArrs(shapeInfo, dimsToExclude);
  sd_printf("After num sub arrs %d\n",numOfSubArrs);

  const int subArrRank =
      (rank == dimsToExclude.size() || descriptor->areUnitiesinShape()) ? rank : rank - dimsToExclude.size();
  sd_printf("After sub rank %d\n",subArrRank);

  auto sPtr = std::make_shared<PointerWrapper>(
      new sd::LongType[shape::shapeInfoLength(subArrRank)],
      std::make_shared<PrimaryPointerDeallocator>());  // shape of sub-arrays (same for all for them)
  sd_printf("After sub array pointer %d\n",subArrRank);

  auto oPtr =
      std::make_shared<PointerWrapper>(new sd::LongType[numOfSubArrs], std::make_shared<PrimaryPointerDeallocator>());
  sd_printf("After o ptr %d\n",subArrRank);

  if (numOfSubArrs > 0) {
    shape::calcSubArrsShapeInfoAndOffsets(shapeInfo, numOfSubArrs, dimsToExclude.size(), dimsToExclude.data(),
                                          sPtr->pointerAsT<sd::LongType>(), oPtr->pointerAsT<sd::LongType>(),
                                          descriptor->areUnitiesinShape());
    sd_printf("After  num sub arrs%d\n",subArrRank);

  }
  ConstantShapeBuffer shapeBuffer(sPtr);
  ConstantOffsetsBuffer offsetsBuffer(oPtr);
  TadPack *t = new TadPack(shapeBuffer, offsetsBuffer, numOfSubArrs);
  return t;
}
}  // namespace sd

#endif
