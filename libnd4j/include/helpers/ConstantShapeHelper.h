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

#ifndef DEV_TESTS_CONSTANTSHAPEHELPER_H
#define DEV_TESTS_CONSTANTSHAPEHELPER_H

#include <dll.h>
#include <pointercast.h>
#include <map>
#include <mutex>
#include <vector>
#include <ShapeDescriptor.h>
#include <DataBuffer.h>

namespace nd4j {

    class ND4J_EXPORT ConstantShapeHelper {
    private:
        static ConstantShapeHelper *_INSTANCE;

        std::mutex _mutex;
        std::vector<std::map<ShapeDescriptor, DataBuffer>> _cache;


        ConstantShapeHelper();
    public:
        ~ConstantShapeHelper() = default;

        static ConstantShapeHelper* getInstance();


        DataBuffer& bufferForShapeInfo(nd4j::DataType dataType, char order, const std::vector<Nd4jLong> &shape);
        DataBuffer& bufferForShapeInfo(ShapeDescriptor &descriptor);
        DataBuffer& bufferForShapeInfo(const Nd4jLong *shapeInfo);


        Nd4jLong* emptyShapeInfo(const nd4j::DataType dataType);
        Nd4jLong* scalarShapeInfo(const nd4j::DataType dataType);
        Nd4jLong* vectorShapeInfo(const Nd4jLong length, const nd4j::DataType dataType);
        Nd4jLong* createShapeInfo(const nd4j::DataType dataType, const char order, const std::vector<Nd4jLong> &shape);
        Nd4jLong* createShapeInfo(const nd4j::DataType dataType, const char order, const int rank, const Nd4jLong* shape);

        bool checkBufferExistenceForShapeInfo(ShapeDescriptor &descriptor);
    };
}

#endif //DEV_TESTS_CONSTANTSHAPEHELPER_H
