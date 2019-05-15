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

#include <array/ConstantDescriptor.h>
#include <DataTypeUtils.h>
#include <stdexcept>

namespace nd4j {
    ConstantDescriptor::ConstantDescriptor(std::initializer_list<Nd4jLong> &values) {
        _integerValues = values;
    }

    ConstantDescriptor::ConstantDescriptor(std::initializer_list<double> &values) {
        _floatValues = values;
    }

    ConstantDescriptor::ConstantDescriptor(std::vector<Nd4jLong> &values) {
        _integerValues = values;
    }

    ConstantDescriptor::ConstantDescriptor(std::vector<double> &values) {
        _floatValues = values;
    }

    // equal to operator
    bool ConstantDescriptor::operator==(const ConstantDescriptor &other) const {
        return std::tie(_floatValues, _integerValues) == std::tie(other._floatValues, other._integerValues);
    }

    // less than operator
    bool ConstantDescriptor::operator<(const ConstantDescriptor &other) const {
        return std::tie(_floatValues, _integerValues) < std::tie(other._floatValues, other._integerValues);
    }

    bool ConstantDescriptor::hasPointer(nd4j::DataType dataType) {
        return _references.count(dataType) > 0;
    }

    template <typename T>
    bool ConstantDescriptor::hasPointer() {
        return hasPointer(DataTypeUtils::fromT<T>());
    }
    BUILD_SINGLE_TEMPLATE(template ND4J_EXPORT bool ConstantDescriptor::hasPointer, (), LIBND4J_TYPES);

    void ConstantDescriptor::addPointer(Nd4jPointer pointer, nd4j::DataType dataType) {
        _references[dataType] = pointer;
    }

    template <typename T>
    void ConstantDescriptor::addPointer(Nd4jPointer pointer) {
        addPointer(pointer, DataTypeUtils::fromT<T>());
    }
    BUILD_SINGLE_TEMPLATE(template ND4J_EXPORT void ConstantDescriptor::addPointer, (Nd4jPointer), LIBND4J_TYPES);

    Nd4jPointer ConstantDescriptor::getPointer(nd4j::DataType dataType) {
        if (!hasPointer(dataType))
            throw std::runtime_error("Requested dataType is absent in storage");

        return _references[dataType];
    }

    template <typename T>
    void* ConstantDescriptor::getPointer() {
        return reinterpret_cast<T*>(getPointer(DataTypeUtils::fromT<T>()));
    }
    BUILD_SINGLE_TEMPLATE(template ND4J_EXPORT void* ConstantDescriptor::getPointer, (), LIBND4J_TYPES);
}
