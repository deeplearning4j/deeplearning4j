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
// Created by agibsonccc on 1/17/17.
//
#include "testinclude.h"
#include <reduce3.h>

class EqualsTest : public testing::Test {
public:
    Nd4jLong firstShapeBuffer[8] = {2,1,2,1,1,0,1,102};
    float data[2] = {1.0,7.0};
    Nd4jLong secondShapeBuffer[8] = {2,2,1,6,1,0,6,99};
    float dataSecond[12] = {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0};
    int opNum = 4;
    float extraArgs[1] = {1e-6};
    int dimension[1] = {2147483647};
    int dimensionLength = 1;
};


TEST_F(EqualsTest,Eps) {
    auto val = nd4j::NDArrayFactory::create(0.0f);
    functions::reduce3::Reduce3<float, float>::execScalar(opNum,
                                                               data,
                                                               firstShapeBuffer,
                                                               extraArgs,
                                                               dataSecond,
                                                               secondShapeBuffer,
                                                               val.buffer(),
                                                               val.shapeInfo());
    ASSERT_TRUE(val.e<float>(0) < 0.5);
}
