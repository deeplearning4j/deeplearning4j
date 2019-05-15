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
#include <TAD.h>
#include <ConstantHelper.h>
#include <exceptions/cuda_exception.h>
#include <execution/LaunchContext.h>
#include <ShapeUtils.h>

namespace nd4j {
    ConstantTadHelper::ConstantTadHelper() {
        std::map<TadDescriptor, TadPack> pack;
        _cache.emplace_back(pack);
    }

    ConstantTadHelper* ConstantTadHelper::getInstance() {
        if (!_INSTANCE)
            _INSTANCE = new ConstantTadHelper();

        return _INSTANCE;
    }

    TadPack& ConstantTadHelper::tadForDimensions(Nd4jLong *originalShape, int dimension, const bool keepUnitiesInShape) {
        return tadForDimensions(originalShape, &dimension, 1, keepUnitiesInShape);
    }

    TadPack& ConstantTadHelper::tadForDimensions(Nd4jLong *originalShape, const std::vector<int> &dimensions, const bool keepUnitiesInShape) {
        return tadForDimensions(originalShape, const_cast<int *>(dimensions.data()), dimensions.size(), keepUnitiesInShape);
    }

    TadPack& ConstantTadHelper::tadForDimensions(Nd4jLong *originalShape, int* dimensions, int dimLength, const bool keepUnitiesInShape) {
        TadDescriptor tadDescriptor(originalShape, dimensions, dimLength, keepUnitiesInShape);
        return tadForDimensions(tadDescriptor);
    }

    TadPack& ConstantTadHelper::tadForDimensions(ShapeDescriptor &descriptor, std::vector<int> &dimensions, const bool keepUnitiesInShape) {
        TadDescriptor tadDescriptor(descriptor, dimensions, keepUnitiesInShape);
        return tadForDimensions(tadDescriptor);
    }

    TadPack& ConstantTadHelper::tadForDimensions(TadDescriptor &descriptor) {
        const int deviceId = 0;

        _mutex.lock();

        if (_cache[deviceId].count(descriptor) == 0) {
            const auto shapeInfo = descriptor.originalShape().toShapeInfo();
            const int rank = shape::rank(shapeInfo);
            const std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(rank, descriptor.axis());
            const Nd4jLong numOfSubArrs = ShapeUtils::getNumOfSubArrs(shapeInfo, dimsToExclude);
            const int subArrRank = (rank == dimsToExclude.size() || descriptor.areUnitiesinShape()) ? rank : rank - dimsToExclude.size();

            auto sPtr = new Nd4jLong[shape::shapeInfoLength(subArrRank)];
            auto oPtr = new Nd4jLong[numOfSubArrs];

            shape::calcSubArrShapeAndOffsets(shapeInfo, numOfSubArrs, dimsToExclude.size(), dimsToExclude.data(), sPtr, oPtr, descriptor.areUnitiesinShape());

            auto ssPtr = ConstantHelper::getInstance()->replicatePointer(sPtr, shape::shapeInfoByteLength(subArrRank));
            Nd4jPointer soPtr;
            auto res = cudaMalloc(reinterpret_cast<void**>(&soPtr),  numOfSubArrs * sizeof(Nd4jLong));
            if (res != 0)
                throw cuda_exception::build("Memory allocation for tadOffsets failed", res);

            cudaMemcpyAsync(soPtr, oPtr, numOfSubArrs * sizeof(Nd4jLong), cudaMemcpyHostToDevice, *nd4j::LaunchContext ::defaultContext()->getCudaSpecialStream());
            res = cudaStreamSynchronize(*nd4j::LaunchContext ::defaultContext()->getCudaSpecialStream());
            if (res != 0)
                throw cuda_exception::build("tadOffsets copy failed", res);

            DataBuffer shapesBuffer(sPtr, ssPtr, shape::shapeInfoLength(subArrRank), sizeof(Nd4jLong));
            DataBuffer offsetsBuffer(oPtr, soPtr, numOfSubArrs, sizeof(Nd4jLong));

            TadPack t(shapesBuffer, offsetsBuffer, numOfSubArrs);
            _cache[deviceId][descriptor] = t;

            TadPack &r = _cache[deviceId][descriptor];
            _mutex.unlock();

            delete[] shapeInfo;

            return r;
        } else {
            TadPack &r = _cache[deviceId][descriptor];
            _mutex.unlock();

            return r;
        }
    }

    nd4j::ConstantTadHelper* nd4j::ConstantTadHelper::_INSTANCE = 0;
}