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

#include "../ShapeDescriptor.h"
#include <shape.h>

using namespace nd4j;

//////////////////////////////////////////////////////////////////////////
// equal to operator
bool ShapeDescriptor::operator==(const ShapeDescriptor& other) const {

    if(_empty != other._empty)
        return false;
    if(_rank != other._rank)
        return false;
    if(_order != other._order)
        return false;
    if(_dataType != other._dataType)
        return false;
    if(_ews != other._ews)
        return false;

    if(_shape != other._shape)
        return false;

    if(_strides != other._strides)
        return false;

    return true;
}

//////////////////////////////////////////////////////////////////////////
// less than operator
bool ShapeDescriptor::operator<(const ShapeDescriptor& other) const {
    
    if(_empty == true && other._empty == true)
        return false;
    if(_empty == false && other._empty == true)
        return false;
    if(_rank > other._rank)
        return false;
    if (_dataType > other._dataType)
        return false;
    
    if(_rank == other._rank) {
                
        if(_shape > other._shape)
            return false;
                
        if(_strides > other._strides)
            return false;
    }

    if(_ews > other._ews)
        return false;
    if(_order > other._order)
        return false;

    return !(*this == other);
}

//////////////////////////////////////////////////////////////////////////
ShapeDescriptor::ShapeDescriptor(const DataType type, const char order, const std::vector<Nd4jLong> &shape): _dataType(type), _order(order), _shape(shape) {
    _rank = shape.size();
    _ews = 1;
    _strides.resize(shape.size());

    if (order == 'c')
        shape::calcStrides(_shape.data(), shape.size(), _strides.data());
    else
        shape::calcStridesFortran(_shape.data(), shape.size(), _strides.data());

    if (_shape.empty())
        _empty = true;
    else {
        for (auto v:_shape) {
            if (v == 0) {
                _empty = true;
                break;
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
ShapeDescriptor::ShapeDescriptor(const DataType type, const char order, const std::initializer_list<Nd4jLong> &shape): _dataType(type), _order(order), _shape(shape) {
    _rank = shape.size();
    _ews = 1;

    if (order == 'c')
        shape::calcStrides(_shape.data(), shape.size(), _strides.data());
    else
        shape::calcStridesFortran(_shape.data(), shape.size(), _strides.data());

    if (_shape.empty())
        _empty = true;
    else {
        for (auto v:_shape) {
            if (v == 0) {
                _empty = true;
                break;
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
ShapeDescriptor::ShapeDescriptor(const DataType type, const char order, const std::vector<Nd4jLong> &shape, const std::vector<Nd4jLong> &strides, const Nd4jLong ews): ShapeDescriptor(type, order, shape, strides) { 
    _ews = ews;
}

ShapeDescriptor::ShapeDescriptor(const Nd4jLong *shapeInfo) {
    _order = shape::order(shapeInfo);
    _ews = shape::elementWiseStride(shapeInfo);
    _rank = shape::rank(shapeInfo);
    _dataType = ArrayOptions::dataType(shapeInfo);

    _empty = shape::isEmpty(shapeInfo);

    for (int e = 0; e < _rank; e++)
        _shape.emplace_back(shapeInfo[e + 1]);

    for (int e = 0; e < _rank; e++)
        _strides.emplace_back(shapeInfo[e + 1 + _rank]);
}

int ShapeDescriptor::rank() const {
    return _rank;
}

Nd4jLong ShapeDescriptor::ews() const {
    return _ews;
}

char ShapeDescriptor::order() const {
    return _order;
}

DataType ShapeDescriptor::dataType() const {
    return _dataType;
}

bool ShapeDescriptor::isEmpty() const {
    return _empty;
}
std::vector<Nd4jLong>& ShapeDescriptor::shape() {
    return _shape;
}

std::vector<Nd4jLong>& ShapeDescriptor::strides() {
    return _strides;
}

ShapeDescriptor::ShapeDescriptor(const ShapeDescriptor &other) {
    _rank = other._rank;
    _ews = other._ews;
    _empty = other._empty;
    _dataType = other._dataType;
    _order = other._order;
    _shape = other._shape;
    _strides = other._strides;
}

//////////////////////////////////////////////////////////////////////////
ShapeDescriptor::ShapeDescriptor(const DataType type, const char order, const std::vector<Nd4jLong> &shape, const std::vector<Nd4jLong> &strides): _dataType(type), _order(order), _shape(shape) {

    if (strides.empty() && !shape.empty()) {
        if (order == 'c')
            shape::calcStrides(_shape.data(), shape.size(), _strides.data());
        else
            shape::calcStridesFortran(_shape.data(), shape.size(), _strides.data());
    } 
    else {
        _strides = strides;
    }

    if (_shape.empty())
        _empty = true;
    else {
        for (auto v:_shape) {
            if (v == 0) {
                _empty = true;
                break;
            }
        }
    }
}
