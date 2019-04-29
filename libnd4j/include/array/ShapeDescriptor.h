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

#ifndef DEV_TESTS_SHAPEDESCRIPTOR_H
#define DEV_TESTS_SHAPEDESCRIPTOR_H

#include <map>
#include <vector>
#include <dll.h>
#include <pointercast.h>
#include <DataType.h>
#include <initializer_list>

namespace nd4j {

class ND4J_EXPORT ShapeDescriptor {

    private:
        int _rank = 0;
        std::vector<Nd4jLong> _shape;
        std::vector<Nd4jLong> _strides;
        Nd4jLong _ews = 1;
        char _order = 'c';
        DataType _dataType;
        bool _empty = false;

    public:
        explicit ShapeDescriptor(const ShapeDescriptor &other);
        explicit ShapeDescriptor(const Nd4jLong *shapeInfo);
        explicit ShapeDescriptor(const DataType type, const Nd4jLong length);
        explicit ShapeDescriptor(const DataType type, const char order, const Nd4jLong *shape, const int rank);
        explicit ShapeDescriptor(const DataType type, const char order, const std::initializer_list<Nd4jLong> &shape);
        explicit ShapeDescriptor(const DataType type, const char order, const std::vector<Nd4jLong> &shape);
        explicit ShapeDescriptor(const DataType type, const char order, const std::vector<Nd4jLong> &shape, const std::vector<Nd4jLong> &strides);
        explicit ShapeDescriptor(const DataType type, const char order, const std::vector<Nd4jLong> &shape, const std::vector<Nd4jLong> &strides, const Nd4jLong ews);
        ShapeDescriptor() = default;
        ~ShapeDescriptor() = default;

        int rank() const;
        Nd4jLong ews() const;
        char order() const;
        DataType dataType() const;
        bool isEmpty() const;
        std::vector<Nd4jLong>& shape();
        std::vector<Nd4jLong>& strides();

        // we use default copy assignment operator
        ShapeDescriptor& operator=(const ShapeDescriptor& other) = default;

        // we use default move assignment operator
        ShapeDescriptor& operator=(ShapeDescriptor&& other) noexcept = default;

        // equal to operator
        bool operator==(const ShapeDescriptor &other) const;

        // less than operator
        bool operator<(const ShapeDescriptor &other) const;

        Nd4jLong* toShapeInfo();
    };


}


#endif //DEV_TESTS_SHAPEDESCRIPTOR_H
