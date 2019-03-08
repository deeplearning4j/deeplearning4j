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


#include <algorithm>
#include "../TadDescriptor.h"

namespace nd4j {
    TadDescriptor::TadDescriptor(const TadDescriptor &other) {
        _originalShape = other._originalShape;
        _axis = other._axis;
    }

    TadDescriptor::TadDescriptor(const Nd4jLong *originalShape, const int *dimensions, const int length) {
        ShapeDescriptor descriptor(originalShape);

        _axis.resize(length);
        for (int e = 0; e < length; e++)
            _axis[e] = dimensions[e];

        if (length > 1)
            std::sort(_axis.begin(), _axis.end());

        _originalShape = descriptor;
    }

    TadDescriptor::TadDescriptor(const ShapeDescriptor &descriptor, const std::vector<int> &dimensions) {
        _originalShape = descriptor;
        _axis = dimensions;

        if (_axis.size() > 1)
            std::sort(_axis.begin(), _axis.end());
    }

    bool TadDescriptor::operator==(const TadDescriptor &other) const {
        return std::tie(_originalShape, _axis) == std::tie(other._originalShape, other._axis);
    }


    bool TadDescriptor::operator<(const TadDescriptor &other) const {
        return std::tie(_originalShape, _axis) < std::tie(other._originalShape, other._axis);
    }

    std::vector<int>& TadDescriptor::axis() {
        return _axis;
    }

    ShapeDescriptor& TadDescriptor::originalShape() {
        return _originalShape;
    }
}