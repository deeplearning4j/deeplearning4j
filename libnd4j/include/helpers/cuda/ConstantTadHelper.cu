/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
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

//
//  @author raver119@gmail.com
//

#include "../ConstantTadHelper.h"
#include <helpers/TAD.h>
#include <helpers/ConstantHelper.h>
#include <execution/AffinityManager.h>
#include <exceptions/cuda_exception.h>
#include <execution/LaunchContext.h>
#include <helpers/ShapeUtils.h>
#include <array/PrimaryPointerDeallocator.h>
#include <array/CudaPointerDeallocator.h>

namespace sd {
    ConstantTadHelper::ConstantTadHelper() {
        auto numDevices = AffinityManager::numberOfDevices();

        for (int e = 0; e < numDevices; e++) {
            MAP_IMPL<TadDescriptor, TadPack> pack;
            _cache.emplace_back(pack);
        }
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
        const int deviceId = AffinityManager::currentDeviceId();

        std::lock_guard<std::mutex> lock(_mutex);

        if (_cache[deviceId].count(descriptor) == 0) {
            const auto shapeInfo = descriptor.originalShape().toShapeInfo();
            const int rank = shape::rank(shapeInfo);
            const std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(rank, descriptor.axis());
            const Nd4jLong numOfSubArrs = ShapeUtils::getNumOfSubArrs(shapeInfo, dimsToExclude);
            const int subArrRank = (rank == dimsToExclude.size() || descriptor.areUnitiesinShape()) ? rank : rank - dimsToExclude.size();

            auto sPtr = std::make_shared<PointerWrapper>(new Nd4jLong[shape::shapeInfoLength(subArrRank)], std::make_shared<PrimaryPointerDeallocator>());
            auto oPtr = std::make_shared<PointerWrapper>(new Nd4jLong[numOfSubArrs], std::make_shared<PrimaryPointerDeallocator>());

            if (numOfSubArrs > 0)
                shape::calcSubArrsShapeInfoAndOffsets(shapeInfo, numOfSubArrs, dimsToExclude.size(), dimsToExclude.data(), sPtr->pointerAsT<Nd4jLong>(), oPtr->pointerAsT<Nd4jLong>(), descriptor.areUnitiesinShape());

            Nd4jPointer soPtr;
            auto res = cudaMalloc(reinterpret_cast<void**>(&soPtr),  numOfSubArrs * sizeof(Nd4jLong));
            if (res != 0)
                throw cuda_exception::build("Memory allocation for tadOffsets failed", res);

            res = cudaMemcpy(soPtr, oPtr->pointer(), numOfSubArrs * sizeof(Nd4jLong), cudaMemcpyHostToDevice);
            if (res != 0)
                throw cuda_exception::build("tadOffsets copy failed", res);

            // TODO: add deallocator here?
            auto ssPtr = std::make_shared<PointerWrapper>(ConstantHelper::getInstance().replicatePointer(sPtr->pointer(), shape::shapeInfoByteLength(subArrRank)));



            ConstantShapeBuffer shapesBuffer(sPtr, ssPtr);
            ConstantOffsetsBuffer offsetsBuffer(oPtr, std::make_shared<PointerWrapper>(soPtr, std::make_shared<CudaPointerDeallocator>()));

            TadPack t(shapesBuffer, offsetsBuffer, numOfSubArrs);
            _cache[deviceId][descriptor] = t;

            TadPack r = _cache[deviceId][descriptor];

            delete[] shapeInfo;

            return r;
        } else {
            TadPack r = _cache[deviceId][descriptor];

            return r;
        }
    }
}