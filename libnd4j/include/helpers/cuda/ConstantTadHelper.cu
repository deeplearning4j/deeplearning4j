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
#include <array/CudaPointerDeallocator.h>
#include <array/PrimaryPointerDeallocator.h>
#include <exceptions/cuda_exception.h>
#include <execution/AffinityManager.h>
#include <execution/LaunchContext.h>
#include <helpers/ConstantHelper.h>
#include <helpers/ShapeUtils.h>
#include <helpers/TAD.h>

#include "../ConstantTadHelper.h"

namespace sd {
ConstantTadHelper::ConstantTadHelper() {
  auto numDevices = AffinityManager::numberOfDevices();

  for (int e = 0; e < numDevices; e++) {
    SD_MAP_IMPL<TadDescriptor *, TadPack *> pack;
    _cache.emplace_back(pack);
  }
}

ConstantTadHelper &ConstantTadHelper::getInstance() {
  static ConstantTadHelper instance;
  return instance;
}

TadPack * ConstantTadHelper::tadForDimensions(const LongType *originalShape, LongType dimension,
                                              const bool keepUnitiesInShape) {
  return tadForDimensions(originalShape, &dimension, 1, keepUnitiesInShape);
}

TadPack * ConstantTadHelper::tadForDimensions(const LongType *originalShape, const std::vector<LongType> *dimensions,
                                              const bool keepUnitiesInShape) {
  return tadForDimensions(originalShape, const_cast<LongType *>(dimensions->data()), dimensions->size(), keepUnitiesInShape);
}

TadPack * ConstantTadHelper::tadForDimensions(const LongType *originalShape, LongType *dimensions, LongType dimLength,
                                              const bool keepUnitiesInShape) {
  TadDescriptor *tadDescriptor = new TadDescriptor(originalShape, dimensions, dimLength, keepUnitiesInShape);
  return tadForDimensions(tadDescriptor);
}

TadPack * ConstantTadHelper::tadForDimensions(ShapeDescriptor &descriptor, std::vector<LongType> &dimensions,
                                              const bool keepUnitiesInShape) {

  TadDescriptor *tadDescriptor = new TadDescriptor(descriptor, dimensions, keepUnitiesInShape);
  return tadForDimensions(tadDescriptor);
}

TadPack *ConstantTadHelper::tadForDimensions(TadDescriptor *descriptor) {
  const int deviceId = AffinityManager::currentDeviceId();
  if(descriptor == nullptr)
    THROW_EXCEPTION("ConstantTadHelper::tadForDimensions: descriptor is nullptr!");
  std::lock_guard<std::mutex> lock(_mutex);
  if (_cache[deviceId].count(descriptor) == 0) {
    // if there's no TadPack matching this descriptor - create one
    const auto shapeInfo = ConstantShapeHelper::getInstance().createFromExisting(descriptor->originalShape().toShapeInfo());
    const LongType rank = shape::rank(shapeInfo);
    const std::vector<LongType> *dimsToExclude = ShapeUtils::evalDimsToExclude(rank, descriptor->axis().size(),descriptor->axis().data());

    const LongType numOfSubArrs = ShapeUtils::getNumOfSubArrs(shapeInfo, *dimsToExclude);
    if(numOfSubArrs > 0) {
      const LongType subArrRank =
          (rank == dimsToExclude->size() || descriptor->areUnitiesinShape()) ? rank : rank - dimsToExclude->size();

      auto sPtr = std::make_shared<PointerWrapper>(
          new LongType[shape::shapeInfoLength(subArrRank)]);  // shape of sub-arrays (same for all for them)
      auto oPtr =
          std::make_shared<PointerWrapper>(new LongType[numOfSubArrs]);

      if (numOfSubArrs > 0)
        shape::calcSubArrsShapeInfoAndOffsets(shapeInfo, numOfSubArrs, dimsToExclude->size(), dimsToExclude->data(),
                                              sPtr->pointerAsT<LongType>(), oPtr->pointerAsT<LongType>(),
                                              descriptor->areUnitiesinShape());

      Pointer soPtr;
      auto res = cudaMalloc(reinterpret_cast<void **>(&soPtr), numOfSubArrs * sizeof(LongType));
      if (res != 0) throw cuda_exception::build("Memory allocation for tadOffsets failed", res);

      res = cudaMemcpy(soPtr, oPtr->pointer(), numOfSubArrs * sizeof(LongType), cudaMemcpyHostToDevice);
      if (res != 0) throw cuda_exception::build("tadOffsets copy failed", res);

      // TODO: add deallocator here?
      auto ssPtr = std::make_shared<PointerWrapper>(
          ConstantHelper::getInstance().replicatePointer(sPtr->pointer(), shape::shapeInfoByteLength(subArrRank)));
      ConstantOffsetsBuffer *offsetsBuffer = new ConstantOffsetsBuffer(
          oPtr, std::make_shared<PointerWrapper>(soPtr, std::make_shared<CudaPointerDeallocator>()));

      auto shapesBuffer = ConstantShapeHelper::getInstance().bufferForShapeInfo(sPtr->pointerAsT<LongType>());
      //note that we pass in .data() here because tad pack is a copy constructor.
      TadPack *t = new TadPack(*shapesBuffer, *offsetsBuffer, numOfSubArrs, descriptor->axis().data(), descriptor->axis().size());
      _cache[deviceId][descriptor] = t;
    } else {
      //base case: number of sub arrays is zero. just return the original shape.
       auto shapeInfo =
          ConstantShapeHelper::getInstance().createFromExisting(descriptor->originalShape().toShapeInfo());
       LongType rank = shape::rank(shapeInfo);
      const LongType subArrRank = rank;

      auto sPtr = std::make_shared<PointerWrapper>(
          new LongType[shape::shapeInfoLength(subArrRank)]);  // shape of sub-arrays (same for all for them)

      sd::LongType * shapeInfo2 = sPtr->pointerAsT<LongType>();
      auto nonConstant = const_cast<LongType *>(shapeInfo);
      auto nonConst2 = const_cast<LongType *>(shapeInfo2);
      shape::copyTo<LongType>(shape::shapeInfoLength(subArrRank), nonConstant, nonConst2);
      LongType *baseOffset = new LongType[numOfSubArrs];
      baseOffset[0] = 0;
      auto oPtr = std::make_shared<PointerWrapper>(baseOffset);

      Pointer soPtr;
      auto res = cudaMalloc(reinterpret_cast<void **>(&soPtr), numOfSubArrs * sizeof(LongType));
      if (res != 0) throw cuda_exception::build("Memory allocation for tadOffsets failed", res);

      res = cudaMemcpy(soPtr, oPtr->pointer(), numOfSubArrs * sizeof(LongType), cudaMemcpyHostToDevice);
      if (res != 0) throw cuda_exception::build("tadOffsets copy failed", res);

      // TODO: add deallocator here?
      auto ssPtr = std::make_shared<PointerWrapper>(
          ConstantHelper::getInstance().replicatePointer(sPtr->pointer(), shape::shapeInfoByteLength(subArrRank)));
      ConstantOffsetsBuffer *offsetsBuffer = new ConstantOffsetsBuffer(
          oPtr, std::make_shared<PointerWrapper>(soPtr, std::make_shared<CudaPointerDeallocator>()));

      auto shapesBuffer = ConstantShapeHelper::getInstance().bufferForShapeInfo(sPtr->pointerAsT<LongType>());
      // note that we pass in .data() here because tad pack is a copy constructor.
      TadPack *t = new TadPack(*shapesBuffer, *offsetsBuffer, numOfSubArrs, descriptor->axis().data(),
                               descriptor->axis().size());
      _cache[deviceId][descriptor] = t;



    }

    delete[] dimsToExclude;


  }


  return _cache[deviceId][descriptor];

  // if there's no TadPack matching this descriptor - create one

}
}  // namespace sd
