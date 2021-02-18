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


#include <algorithm>
#include "../TadDescriptor.h"

namespace sd {
    TadDescriptor::TadDescriptor(const TadDescriptor &other) {
        _originalShape = other._originalShape;
        _axis = other._axis;
        _unitiesInShape = other._unitiesInShape;
    }

    TadDescriptor::TadDescriptor(const Nd4jLong *originalShape, const int *dimensions, const int length, const bool keepUnitiesInShape) {
        ShapeDescriptor descriptor(originalShape);

        _axis.resize(length);
        for (int e = 0; e < length; e++)
            _axis[e] = dimensions[e];

        if (length > 1)
            std::sort(_axis.begin(), _axis.end());

        _originalShape = descriptor;
        _unitiesInShape = keepUnitiesInShape;
    }

    TadDescriptor::TadDescriptor(const ShapeDescriptor &descriptor, const std::vector<int> &dimensions, const bool keepUnitiesInShape) {
        _originalShape = descriptor;
        _axis = dimensions;
        _unitiesInShape = keepUnitiesInShape;

        if (_axis.size() > 1)
            std::sort(_axis.begin(), _axis.end());
    }

    bool TadDescriptor::operator==(const TadDescriptor &other) const {
        return std::tie(_originalShape, _axis, _unitiesInShape) == std::tie(other._originalShape, other._axis, other._unitiesInShape);
    }


    bool TadDescriptor::operator<(const TadDescriptor &other) const {
        return std::tie(_originalShape, _axis, _unitiesInShape) < std::tie(other._originalShape, other._axis, other._unitiesInShape);
    }

    std::vector<int>& TadDescriptor::axis() {
        return _axis;
    }

    ShapeDescriptor& TadDescriptor::originalShape(){
        return _originalShape;
    }

    ShapeDescriptor const& TadDescriptor::originalShapeConst() const{
        return _originalShape;
    }

    bool TadDescriptor::areUnitiesinShape() const {
        return _unitiesInShape;
    }
}

namespace std {
    size_t hash<sd::TadDescriptor>::operator()(const sd::TadDescriptor &k) const {
        // Compute individual hash values for first,
        // second and third and combine them using XOR
        // and bit shifting:
        auto res = std::hash<int>()((int)k.areUnitiesinShape());
        res ^= std::hash<sd::ShapeDescriptor>()(k.originalShapeConst())  + 0x9e3779b9 + (res << 6) + (res >> 2);
        auto axes = const_cast<sd::TadDescriptor&>(k).axis();
        for (auto a: axes) {
            res ^= std::hash<int>()(a) + 0x9e3779b9 + (res << 6) + (res >> 2);
        }
        return res;
    }
}