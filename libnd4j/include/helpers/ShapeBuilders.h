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

#ifndef DEV_TESTS_SHAPEBUILDERS_H
#define DEV_TESTS_SHAPEBUILDERS_H

#include <vector>
#include <helpers/shape.h>
#include <pointercast.h>
#include <memory/Workspace.h>
#include <array/DataType.h>
#include <array/ArrayOptions.h>

namespace nd4j {
    class ShapeBuilders {
    public:        
        static Nd4jLong* createScalarShapeInfo(nd4j::DataType dataType, nd4j::memory::Workspace* workspace = nullptr);
        
        static Nd4jLong* createVectorShapeInfo(const nd4j::DataType dataType, const Nd4jLong length, nd4j::memory::Workspace* workspace = nullptr);

        /**
        *   create shapeInfo for given order basing on shape stored in shapeOnly vector
        *   memory allocation for shapeInfo is on given workspace
        */        
        static Nd4jLong* createShapeInfo(const nd4j::DataType dataType, const char order, int rank, const Nd4jLong* shapeOnly, memory::Workspace* workspace = nullptr);
        static Nd4jLong* createShapeInfo(const nd4j::DataType dataType, const char order, const std::vector<Nd4jLong>& shapeOnly, memory::Workspace* workspace = nullptr);
        static Nd4jLong* createShapeInfo(const nd4j::DataType dataType, const char order, const std::initializer_list<Nd4jLong>& shapeOnly, memory::Workspace* workspace = nullptr);

        /**
        *   allocates memory for new shapeInfo and copy all information from inShapeInfo to new shapeInfo
        *   if copyStrides is false then strides for new shapeInfo are recalculated
        */
        static Nd4jLong* copyShapeInfo(const Nd4jLong* inShapeInfo, const bool copyStrides, memory::Workspace* workspace = nullptr);
        static Nd4jLong* copyShapeInfoAndType(const Nd4jLong* inShapeInfo, const DataType dtype, const bool copyStrides, memory::Workspace* workspace = nullptr);
        static Nd4jLong* copyShapeInfoAndType(const Nd4jLong* inShapeInfo, const Nd4jLong* shapeInfoToGetTypeFrom, const bool copyStrides, memory::Workspace* workspace = nullptr);

        static Nd4jLong* emptyShapeInfo(const nd4j::DataType dataType, memory::Workspace* workspace = nullptr);

    };
}


#endif //DEV_TESTS_SHAPEBUILDERS_H
