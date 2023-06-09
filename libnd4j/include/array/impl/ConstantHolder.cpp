/**
* Copyright (c) 2019 Konduit K.K.
 *
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
// Created by raver on 5/17/2019.
//
#include <array/ConstantHolder.h>
#include <array/DataTypeUtils.h>
#include <helpers/shape.h>

namespace sd {
ConstantHolder::ConstantHolder(const ConstantHolder& other) {
  _buffers = other._buffers;
  _deviceId = other._deviceId;
}

bool ConstantHolder::hasBuffer(sd::DataType dataType) { return _buffers.count(dataType) > 0; }

std::mutex* ConstantHolder::mutex() { return &_mutex; }

template <typename T>
bool ConstantHolder::hasBuffer() {
  return hasBuffer(DataTypeUtils::fromT<T>());
}
BUILD_SINGLE_TEMPLATE(template SD_LIB_EXPORT bool ConstantHolder::hasBuffer, (void), SD_COMMON_TYPES);

void ConstantHolder::addBuffer(ConstantDataBuffer& pointer, sd::DataType dataType) { _buffers[dataType] = pointer; }

template <typename T>
void ConstantHolder::addBuffer(ConstantDataBuffer& pointer) {
  addBuffer(pointer, DataTypeUtils::fromT<T>());
}
BUILD_SINGLE_TEMPLATE(template SD_LIB_EXPORT void ConstantHolder::addBuffer, (ConstantDataBuffer & cb),
                      SD_COMMON_TYPES);

ConstantDataBuffer* ConstantHolder::getConstantDataBuffer(sd::DataType dataType) {
  if (!hasBuffer(dataType)) THROW_EXCEPTION("Requested dataType is absent in storage");

  return &_buffers[dataType];
}

template <typename T>
ConstantDataBuffer* ConstantHolder::getConstantDataBuffer() {
  return getConstantDataBuffer(DataTypeUtils::fromT<T>());
}
BUILD_SINGLE_TEMPLATE(template SD_LIB_EXPORT ConstantDataBuffer* ConstantHolder::getConstantDataBuffer, (),
                      SD_COMMON_TYPES);
}  // namespace sd
