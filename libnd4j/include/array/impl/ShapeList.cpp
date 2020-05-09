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

#include <system/pointercast.h>
#include <array/ShapeList.h>

namespace sd {
    //ShapeList::ShapeList(bool autoRemovable) {
//        _autoremovable = autoRemovable;
//    }

    ShapeList::ShapeList(const Nd4jLong* shape) {
        if (shape != nullptr)
            _shapes.push_back(shape);
    }

    ShapeList::~ShapeList() {
        if (_autoremovable)
            destroy();
    }

    ShapeList::ShapeList(const std::vector<const Nd4jLong*> &shapes, bool isWorkspace) : ShapeList(shapes){
        _workspace = isWorkspace;
    }

    ShapeList::ShapeList(const std::vector<const Nd4jLong*>& shapes) {
        _shapes = shapes;
    }

    std::vector<const Nd4jLong*>* ShapeList::asVector() {
        return &_shapes;
    }

    void ShapeList::destroy() {
        if (_destroyed)
            return;

        if (!_workspace)
            for (auto v:_shapes)
                if(v != nullptr)
                    delete[] v;

        _destroyed = true;
    }

    int ShapeList::size() const {
        return (int) _shapes.size();
    }

    const Nd4jLong* ShapeList::at(int idx) {
        if (_shapes.size() <= idx)
            throw std::runtime_error("Can't find requested variable by index");

        return _shapes.at(idx);
    }

    void ShapeList::push_back(const Nd4jLong *shape) {
        _shapes.push_back(shape);
    }

    void ShapeList::detach() {
        for (int e = 0; e < _shapes.size(); e++) {
            _shapes[e] = shape::detachShape(_shapes[e]);
        }

        _autoremovable = true;
        _workspace = false;
    }
}