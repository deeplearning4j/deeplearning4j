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
// @author raver119@gmail.com
//

#include <array/InteropDataBuffer.h>
#include <array/DataTypeUtils.h>
#include <execution/AffinityManager.h>
#include <helpers/logger.h>

namespace sd {
    InteropDataBuffer::InteropDataBuffer(InteropDataBuffer &dataBuffer, uint64_t length, uint64_t offset) {
        _dataBuffer = dataBuffer.getDataBuffer();

        // offset is always absolute to the original buffer
        _offset = offset;

        if (_offset + length > _dataBuffer->getLenInBytes()) {
            throw std::runtime_error("offset + length is higher than original length");
        }
    }

    InteropDataBuffer::InteropDataBuffer(std::shared_ptr<DataBuffer> databuffer) {
        _dataBuffer = databuffer;
    }

    InteropDataBuffer::InteropDataBuffer(size_t elements, sd::DataType dtype, bool allocateBoth) {
        if (elements == 0) {
            _dataBuffer = std::make_shared<DataBuffer>();
            _dataBuffer->setDataType(dtype);
        } else {
            _dataBuffer = std::make_shared<DataBuffer>(elements, dtype, nullptr, allocateBoth);
        }
    }

    std::shared_ptr<DataBuffer> InteropDataBuffer::getDataBuffer() const {
        return _dataBuffer;
    }

    std::shared_ptr<DataBuffer> InteropDataBuffer::dataBuffer() {
        return _dataBuffer;
    }

    void* InteropDataBuffer::primary() const {
        return reinterpret_cast<int8_t *>(_dataBuffer->primary()) + _offset;
    }

    void* InteropDataBuffer::special() const {
        return reinterpret_cast<int8_t *>(_dataBuffer->special()) + _offset;
    }

    void InteropDataBuffer::setPrimary(void* ptr, size_t length) {
        _dataBuffer->setPrimaryBuffer(ptr, length);
    }

    void InteropDataBuffer::setSpecial(void* ptr, size_t length) {
        _dataBuffer->setSpecialBuffer(ptr, length);
    }

    uint64_t InteropDataBuffer::offset() const {
        return _offset;
    }

    void InteropDataBuffer::setOffset(uint64_t offset) {
        _offset = offset;
    }

    int InteropDataBuffer::deviceId() const {
        return _dataBuffer->deviceId();
    }


    void InteropDataBuffer::registerSpecialUse(const std::vector<const InteropDataBuffer*>& writeList, const std::vector<const InteropDataBuffer*>& readList) {
        for (const auto &v:writeList) {
            if (v == nullptr)
                continue;

            v->getDataBuffer()->writeSpecial();
        }
    }

    void InteropDataBuffer::prepareSpecialUse(const std::vector<const InteropDataBuffer*>& writeList, const std::vector<const InteropDataBuffer*>& readList, bool synchronizeWritables) {
        auto currentDeviceId = sd::AffinityManager::currentDeviceId();
        for (const auto &v:readList) {
            if (v == nullptr)
                continue;

            if (v->getDataBuffer()->deviceId() != currentDeviceId)
                v->getDataBuffer()->migrate();

            v->getDataBuffer()->syncToSpecial();
        }

        // we don't tick write list, only ensure the same device affinity
        for (const auto &v:writeList) {
            if (v == nullptr)
                continue;

            // special case for legacy ops - views can be updated on host side, thus original array can be not updated
            if (!v->getDataBuffer()->isSpecialActual())
                v->getDataBuffer()->syncToSpecial();

            if (v->getDataBuffer()->deviceId() != currentDeviceId)
                v->getDataBuffer()->migrate();
        }
    }

    void InteropDataBuffer::registerPrimaryUse(const std::vector<const InteropDataBuffer*>& writeList, const std::vector<const InteropDataBuffer*>& readList) {
        for (const auto &v:writeList) {
            if (v == nullptr)
                continue;
        }
    }

    void InteropDataBuffer::preparePrimaryUse(const std::vector<const InteropDataBuffer*>& writeList, const std::vector<const InteropDataBuffer*>& readList, bool synchronizeWritables) {
        for (const auto &v:readList) {
            if (v == nullptr)
                continue;

            v->getDataBuffer()->syncToPrimary(LaunchContext::defaultContext());
        }
    }

    void InteropDataBuffer::expand(size_t newlength) {
        _dataBuffer->expand(newlength * DataTypeUtils::sizeOf(_dataBuffer->getDataType()));
    }

    void InteropDataBuffer::setDeviceId(int deviceId) {
        _dataBuffer->setDeviceId(deviceId);
    }
}
