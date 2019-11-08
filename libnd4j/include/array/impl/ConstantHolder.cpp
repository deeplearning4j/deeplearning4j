/**
* Copyright (c) 2019 Konduit K.K.
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
// Created by raver on 5/17/2019.
//

#include <DataTypeUtils.h>
#include <array/ConstantHolder.h>
#include <shape.h>

namespace nd4j {
    ConstantHolder::ConstantHolder(const ConstantHolder& other) {
        _buffers = other._buffers;
        _deviceId = other._deviceId;
    }

    bool ConstantHolder::hasBuffer(nd4j::DataType dataType) {
        return _buffers.count(dataType) > 0;
    }

    std::mutex* ConstantHolder::mutex() {
        return &_mutex;
    }

    template <typename T>
    bool ConstantHolder::hasBuffer() {
        return hasBuffer(DataTypeUtils::fromT<T>());
    }
    BUILD_SINGLE_TEMPLATE(template ND4J_EXPORT bool ConstantHolder::hasBuffer, (void), LIBND4J_TYPES);

    void ConstantHolder::addBuffer(ConstantDataBuffer &pointer, nd4j::DataType dataType) {
        _buffers[dataType] = pointer;
    }

    template <typename T>
    void ConstantHolder::addBuffer(ConstantDataBuffer &pointer) {
        addBuffer(pointer, DataTypeUtils::fromT<T>());
    }
    BUILD_SINGLE_TEMPLATE(template ND4J_EXPORT void ConstantHolder::addBuffer, (ConstantDataBuffer& cb), LIBND4J_TYPES);

    ConstantDataBuffer* ConstantHolder::getConstantDataBuffer(nd4j::DataType dataType) {
        if (!hasBuffer(dataType))
            throw std::runtime_error("Requested dataType is absent in storage");

        return &_buffers[dataType];
    }

    template <typename T>
    ConstantDataBuffer* ConstantHolder::getConstantDataBuffer() {
        return getConstantDataBuffer(DataTypeUtils::fromT<T>());
    }
    BUILD_SINGLE_TEMPLATE(template ND4J_EXPORT ConstantDataBuffer* ConstantHolder::getConstantDataBuffer, (), LIBND4J_TYPES);
}