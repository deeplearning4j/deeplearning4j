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
#include <loops/reduce3.h>

class EqualsTest : public testing::Test {
public:
    const Nd4jLong firstShapeBuffer[8] = {2,1,2,1,1,0,1,102};
    float data[2] = {1.0f, 7.0f};
    const Nd4jLong secondShapeBuffer[8] = {2,2,1,6,1,0,6,99};
    float dataSecond[12] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    int opNum = 4;
    float extraArgs[1] = {1e-6f};
    int dimension[1] = {2147483647};
    int dimensionLength = 1;
};

#ifndef __CUDABLAS__

TEST_F(EqualsTest,Eps) {
    auto val = sd::NDArrayFactory::create(0.0f);
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

#endif
