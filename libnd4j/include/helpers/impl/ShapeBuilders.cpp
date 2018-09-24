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

#include <helpers/ShapeBuilders.h>

namespace nd4j {


    Nd4jLong* ShapeBuilders::createScalarShapeInfo(nd4j::DataType dataType, nd4j::memory::Workspace* workspace) {
        auto res = createScalarShapeInfo(workspace);
        nd4j::ArrayOptions::setDataType(res, dataType);
        return res;
    }

    Nd4jLong* ShapeBuilders::createScalarShapeInfo(nd4j::memory::Workspace* workspace) {
        Nd4jLong *newShape;
        ALLOCATE(newShape, workspace, shape::shapeInfoLength(0), Nd4jLong);
        newShape[0] = 0;
        newShape[1] = 0;
        newShape[2] = 1;
        newShape[3] = 99;

        return newShape;
    }

    Nd4jLong* ShapeBuilders::createVectorShapeInfo(nd4j::DataType dataType, Nd4jLong length, nd4j::memory::Workspace *workspace) {
        auto res = createVectorShapeInfo(length, workspace);
        nd4j::ArrayOptions::setDataType(res, dataType);
        return res;
    }

    Nd4jLong* ShapeBuilders::createVectorShapeInfo(Nd4jLong length, nd4j::memory::Workspace* workspace) {
        Nd4jLong *newShape;
        ALLOCATE(newShape, workspace, shape::shapeInfoLength(1), Nd4jLong);

        newShape[0] = 1;
        newShape[1] = length;
        newShape[2] = 1;
        newShape[3] = 0;
        newShape[4] = 1;
        newShape[5] = 99;

        return newShape;
    }

    Nd4jLong* ShapeBuilders::createShapeInfo(const nd4j::DataType dataType, const char order, const std::vector<Nd4jLong>& shapeOnly, memory::Workspace* workspace) {
        auto res = createShapeInfo(order, shapeOnly, workspace);
        nd4j::ArrayOptions::setDataType(res, dataType);
        return res;
    }

    ////////////////////////////////////////////////////////////////////////////////
    Nd4jLong* ShapeBuilders::createShapeInfo(const char order, const std::vector<Nd4jLong>& shapeOnly, memory::Workspace* workspace) {
        int rank = shapeOnly.size();

        if(shapeOnly[0] == 0) // scalar case
            rank = 0;

        Nd4jLong* shapeInfo = nullptr;

        if(rank == 0) {    // scalar case
            shapeInfo = ShapeBuilders::createScalarShapeInfo(workspace);
        }
        else {
            ALLOCATE(shapeInfo, workspace, shape::shapeInfoLength(rank), Nd4jLong);
            shapeInfo[0] = rank;
            for(int i = 0; i < rank; ++i)
                shapeInfo[i + 1] = shapeOnly[i];

            shape::updateStrides(shapeInfo, order);
        }

        return shapeInfo;
    }

////////////////////////////////////////////////////////////////////////////////
Nd4jLong* ShapeBuilders::createShapeInfo(const nd4j::DataType dataType, const char order, const std::initializer_list<Nd4jLong>& shapeOnly, memory::Workspace* workspace) {

    return ShapeBuilders::createShapeInfo(dataType, order, std::vector<Nd4jLong>(shapeOnly), workspace);
}

////////////////////////////////////////////////////////////////////////////////
Nd4jLong* ShapeBuilders::copyShapeInfo(const Nd4jLong* inShapeInfo, const bool copyStrides, memory::Workspace* workspace) {

    Nd4jLong *outShapeInfo = nullptr;
    ALLOCATE(outShapeInfo, workspace, shape::shapeInfoLength(inShapeInfo), Nd4jLong);
    
    memcpy(outShapeInfo, inShapeInfo, shape::shapeInfoByteLength(inShapeInfo));

    if(!copyStrides)
        shape::updateStrides(outShapeInfo, shape::order(outShapeInfo));

    return outShapeInfo;
}


}