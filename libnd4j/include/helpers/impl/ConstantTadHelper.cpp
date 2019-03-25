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

    TadPack& ConstantTadHelper::tadForDimensions(Nd4jLong *originalShape, int dimensions) {
        return tadForDimensions(originalShape, &dimensions, 1);
    }

    TadPack& ConstantTadHelper::tadForDimensions(Nd4jLong *originalShape, const std::vector<int> &dimensions) {
        return tadForDimensions(originalShape, const_cast<int *>(dimensions.data()), dimensions.size());
    }

    TadPack& ConstantTadHelper::tadForDimensions(Nd4jLong *originalShape, int* dimensions, int dimLength) {
        TadDescriptor tadDescriptor(originalShape, dimensions, dimLength);
        return tadForDimensions(tadDescriptor);
    }

    TadPack& ConstantTadHelper::tadForDimensions(ShapeDescriptor &descriptor, std::vector<int> &dimensions) {
        TadDescriptor tadDescriptor(descriptor, dimensions);
        return tadForDimensions(tadDescriptor);
    }

    TadPack& ConstantTadHelper::tadForDimensions(TadDescriptor &descriptor) {
        const int deviceId = 0;

        _mutex.lock();
        if (_cache[deviceId].count(descriptor) == 0) {
            auto shapeInfo = descriptor.originalShape().toShapeInfo();
            shape::TAD tad;
            tad.init(shapeInfo, descriptor.axis().data(), descriptor.axis().size());
            tad.createTadOnlyShapeInfo();
            tad.createOffsets();

            auto sPtr = new Nd4jLong[shape::shapeInfoLength(tad.tadOnlyShapeInfo)];
            auto oPtr = new Nd4jLong[tad.numTads];

            memcpy(sPtr, tad.tadOnlyShapeInfo, shape::shapeInfoByteLength(tad.tadOnlyShapeInfo));
            memcpy(oPtr, tad.tadOffsets, tad.numTads * sizeof(Nd4jLong));

            DataBuffer shapesBuffer(sPtr, nullptr);
            DataBuffer offsetsBuffer(oPtr, nullptr);
            TadPack t(shapesBuffer, offsetsBuffer, tad.numTads);
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