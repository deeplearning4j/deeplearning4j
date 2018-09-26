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
// Created by agibsonccc on 2/20/18.
//

#include "testinclude.h"
#include <reduce3.h>
#include <ShapeUtils.h>
#include <vector>

class EuclideanTest : public testing::Test {
public:
    Nd4jLong yShape[4] = {4,4};
    Nd4jLong xShape[2] = {1,4};
    float y[16] ={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    float x[4] = {1,2,3,4};
    int dimension[1] = {1};
    int dimensionLength = 1;
    int opNum = 1;
    float extraVals[1] = {0};
    float result[4] = {0.0,0.0,0.0,0.0};

    std::vector<int> dim = {1};
};

TEST_F(EuclideanTest,Test1) {
    auto shapeBuffer = shape::shapeBuffer(2, nd4j::DataType::FLOAT32, yShape);
    auto xShapeBuffer = shape::shapeBuffer(2, nd4j::DataType::FLOAT32, xShape);

    //int *tadShapeBuffer = shape::computeResultShape(shapeBuffer,dimension,dimensionLength);
    auto tadShapeBuffer = nd4j::ShapeUtils::evalReduceShapeInfo('c', dim, shapeBuffer, false, true, nullptr);
            functions::reduce3::Reduce3<float, float>::exec(opNum,
                                             x,
                                             xShapeBuffer,
                                             extraVals,
                                             y,
                                             shapeBuffer,
                                             result,
                                             tadShapeBuffer,
                                             dimension,
                                             dimensionLength);

    float distancesAssertion[4] = {0.0,8.0,16.0,24.0};
    for(int i = 0; i < 4; i++) {
        ASSERT_EQ(distancesAssertion[i],result[i]);
    }

    //ASSERT_EQ(result[1],result[0]);
    delete[] shapeBuffer;
    delete[] tadShapeBuffer;
    delete[] xShapeBuffer;
}




