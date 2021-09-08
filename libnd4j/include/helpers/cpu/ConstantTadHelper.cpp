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
#include <helpers/TAD.h>
#include <helpers/ShapeUtils.h>
#include <array/ConstantOffsetsBuffer.h>
#include <array/PrimaryPointerDeallocator.h>

#ifndef __CUDABLAS__


namespace sd {

    ConstantTadHelper::ConstantTadHelper() {
        MAP_IMPL<TadDescriptor, TadPack> pack;
        _cache.emplace_back(pack);
    }

    ConstantTadHelper& ConstantTadHelper::getInstance() {
      static ConstantTadHelper instance;
      return instance;
    }

    TadPack ConstantTadHelper::tadForDimensions(const Nd4jLong *originalShape, int dimension, const bool keepUnitiesInShape) {
        return tadForDimensions(originalShape, &dimension, 1, keepUnitiesInShape);
    }

    TadPack ConstantTadHelper::tadForDimensions(const Nd4jLong *originalShape, const std::vector<int> &dimensions, const bool keepUnitiesInShape) {
        return tadForDimensions(originalShape, const_cast<int *>(dimensions.data()), dimensions.size(), keepUnitiesInShape);
    }

    TadPack ConstantTadHelper::tadForDimensions(const Nd4jLong *originalShape, int* dimensions, int dimLength, const bool keepUnitiesInShape) {
        TadDescriptor tadDescriptor(originalShape, dimensions, dimLength, keepUnitiesInShape);
        return tadForDimensions(tadDescriptor);
    }

    TadPack ConstantTadHelper::tadForDimensions(ShapeDescriptor &descriptor, std::vector<int> &dimensions, const bool keepUnitiesInShape) {
        TadDescriptor tadDescriptor(descriptor, dimensions, keepUnitiesInShape);
        return tadForDimensions(tadDescriptor);
    }

    TadPack ConstantTadHelper::tadForDimensions(TadDescriptor &descriptor) {
        const int deviceId = 0;

        std::lock_guard<std::mutex> lock(_mutex);
        if (_cache[deviceId].count(descriptor) == 0) {
          // if there's no TadPack matching this descriptor - create one
            const auto shapeInfo = descriptor.originalShape().toShapeInfo();
            const int rank = shape::rank(shapeInfo);
            const std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(rank, descriptor.axis());
            const Nd4jLong numOfSubArrs = ShapeUtils::getNumOfSubArrs(shapeInfo, dimsToExclude);
            const int subArrRank = (rank == dimsToExclude.size() || descriptor.areUnitiesinShape()) ? rank : rank - dimsToExclude.size();

            auto sPtr = std::make_shared<PointerWrapper>(new Nd4jLong[shape::shapeInfoLength(subArrRank)], std::make_shared<PrimaryPointerDeallocator>());   // shape of sub-arrays (same for all for them)
            auto oPtr = std::make_shared<PointerWrapper>(new Nd4jLong[numOfSubArrs], std::make_shared<PrimaryPointerDeallocator>());

            if (numOfSubArrs > 0)
                shape::calcSubArrsShapeInfoAndOffsets(shapeInfo, numOfSubArrs, dimsToExclude.size(), dimsToExclude.data(), sPtr->pointerAsT<Nd4jLong>(), oPtr->pointerAsT<Nd4jLong>(), descriptor.areUnitiesinShape());

            ConstantShapeBuffer shapeBuffer(sPtr);
            ConstantOffsetsBuffer offsetsBuffer(oPtr);
            TadPack t(shapeBuffer, offsetsBuffer, numOfSubArrs);
            _cache[deviceId][descriptor] = t;

            delete[] shapeInfo;
        }

        return _cache[deviceId][descriptor];
    }
}

#endif