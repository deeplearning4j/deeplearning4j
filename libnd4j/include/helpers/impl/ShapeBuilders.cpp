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

namespace sd {


    Nd4jLong* ShapeBuilders::createScalarShapeInfo(const sd::DataType dataType, sd::memory::Workspace* workspace) {
        Nd4jLong *newShape;
        ALLOCATE(newShape, workspace, shape::shapeInfoLength(0), Nd4jLong);
        newShape[0] = 0;
        newShape[1] = 0;
        newShape[2] = 1;
        newShape[3] = 99;

        sd::ArrayOptions::setDataType(newShape, dataType);

        return newShape;
    }

    Nd4jLong* ShapeBuilders::createVectorShapeInfo(const sd::DataType dataType, const Nd4jLong length, sd::memory::Workspace* workspace) {
        Nd4jLong *newShape;
        ALLOCATE(newShape, workspace, shape::shapeInfoLength(1), Nd4jLong);

        newShape[0] = 1;
        newShape[1] = length;
        newShape[2] = 1;
        newShape[3] = 0;
        newShape[4] = 1;
        newShape[5] = 99;

        sd::ArrayOptions::setDataType(newShape, dataType);

        return newShape;
    }

    ////////////////////////////////////////////////////////////////////////////////
    Nd4jLong* ShapeBuilders::createShapeInfo(const sd::DataType dataType, const char order, int rank, const Nd4jLong* shapeOnly, memory::Workspace* workspace) {
        Nd4jLong* shapeInfo = nullptr;

        if(rank == 0) {    // scalar case
            shapeInfo = ShapeBuilders::createScalarShapeInfo(dataType, workspace);
        }
        else {
            ALLOCATE(shapeInfo, workspace, shape::shapeInfoLength(rank), Nd4jLong);
            shapeInfo[0] = rank;
            bool isEmpty = false;
            for(int i = 0; i < rank; ++i) {
                shapeInfo[i + 1] = shapeOnly[i];

                if (shapeOnly[i] == 0)
                    isEmpty = true;
            }

            if (!isEmpty) {
                shape::updateStrides(shapeInfo, order);
            }
            else {
                shapeInfo[shape::shapeInfoLength(rank) - 1] = order;
                memset(shape::stride(shapeInfo), 0, rank * sizeof(Nd4jLong));
                ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
            }

            sd::ArrayOptions::setDataType(shapeInfo, dataType);
        }

        return shapeInfo;
    }

    Nd4jLong* ShapeBuilders::emptyShapeInfo(const sd::DataType dataType, memory::Workspace* workspace) {
        auto shapeInfo = createScalarShapeInfo(dataType, workspace);
        ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
        return shapeInfo;
    }

    Nd4jLong* ShapeBuilders::emptyShapeInfo(const sd::DataType dataType, const char order, const std::vector<Nd4jLong> &shape, memory::Workspace* workspace) {
        auto shapeInfo = createShapeInfo(dataType, order, shape, workspace);
        memset(shape::stride(shapeInfo), 0, shape.size() * sizeof(Nd4jLong));
        ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
        return shapeInfo;
    }

////////////////////////////////////////////////////////////////////////////////
    Nd4jLong* ShapeBuilders::createShapeInfo(const sd::DataType dataType, const char order, const std::vector<Nd4jLong>& shapeOnly, memory::Workspace* workspace) {

        return ShapeBuilders::createShapeInfo(dataType, order, shapeOnly.size(), shapeOnly.data(), workspace);
    }

////////////////////////////////////////////////////////////////////////////////
    Nd4jLong* ShapeBuilders::createShapeInfo(const sd::DataType dataType, const char order, const std::initializer_list<Nd4jLong>& shapeOnly, memory::Workspace* workspace) {

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

////////////////////////////////////////////////////////////////////////////////
    Nd4jLong* ShapeBuilders::copyShapeInfoAndType(const Nd4jLong* inShapeInfo, const DataType dtype, const bool copyStrides, memory::Workspace* workspace) {

        Nd4jLong* outShapeInfo = ShapeBuilders::copyShapeInfo(inShapeInfo, copyStrides, workspace);
        ArrayOptions::setDataType(outShapeInfo, dtype);

        return outShapeInfo;
    }

////////////////////////////////////////////////////////////////////////////////
    Nd4jLong* ShapeBuilders::copyShapeInfoAndType(const Nd4jLong* inShapeInfo, const Nd4jLong* shapeInfoToGetTypeFrom, const bool copyStrides, memory::Workspace* workspace) {

        return ShapeBuilders::copyShapeInfoAndType(inShapeInfo, ArrayOptions::dataType(shapeInfoToGetTypeFrom), copyStrides, workspace);
    }

////////////////////////////////////////////////////////////////////////////////
Nd4jLong* ShapeBuilders::createSubArrShapeInfo(const Nd4jLong* inShapeInfo, const int* dims, const int dimsSize, memory::Workspace* workspace) {

    Nd4jLong *subArrShapeInfo = nullptr;
    ALLOCATE(subArrShapeInfo, workspace, shape::shapeInfoLength(dimsSize), Nd4jLong);

    subArrShapeInfo[0] = dimsSize;                                 // rank
    sd::ArrayOptions::copyDataType(subArrShapeInfo, inShapeInfo);  // type
    subArrShapeInfo[2*dimsSize + 3] = shape::order(inShapeInfo);   // order

    Nd4jLong* shape = shape::shapeOf(subArrShapeInfo);
    Nd4jLong* strides = shape::stride(subArrShapeInfo);

    for(int i = 0; i < dimsSize; ++i) {
        shape[i]   = shape::sizeAt(inShapeInfo, dims[i]);
        strides[i] = shape::strideAt(inShapeInfo, dims[i]);
    }

    shape::checkStridesEwsAndOrder(subArrShapeInfo);

    return subArrShapeInfo;
}

}