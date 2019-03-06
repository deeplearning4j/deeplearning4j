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

namespace nd4j {
    ShapeDescriptor::ShapeDescriptor(DataType type, char order, std::vector<Nd4jLong> &shape) {
        _dataType = type;
        _order = order;
        _shape = shape;


        // TODO:: calculate strides here

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

    ShapeDescriptor::ShapeDescriptor(DataType type, char order, std::vector<Nd4jLong> &shape, std::vector<Nd4jLong> &strides) {
        _dataType = type;
        _order = order;
        _shape = shape;

        if (strides.empty()) {
            // TODO:: calculate strides here
        } else {
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
}