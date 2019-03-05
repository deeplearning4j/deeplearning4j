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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 05.03.2019
//

#ifndef LIBND4J_CONVOLUTIONTESTS_H
#define LIBND4J_CONVOLUTIONTESTS_H

#include "testlayers.h"
#include <NDArray.h>
#include <Context.h>
#include <Node.h>
#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/convolutions.h>
#include <ops/declarable/helpers/col2im.h>
#include <PointersManager.h>

using namespace nd4j;
using namespace nd4j::graph;

class ConvolutionTests2 : public testing::Test {
public:

};

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests2, im2col_1) {

    int bS=2, iH=4,iW=3,  iC=4,oC=3,  kH=3,kW=2,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
    int       oH=2,oW=2;
    int paddingMode = 0;             // 1-SAME, 0-VALID;    

    NDArray image('c', {bS, iC, iH, iW}, nd4j::DataType::DOUBLE);
    NDArray expected('c', {bS, iC, kH, kW, oH, oW}, {1,  2, 4,  5, 2,  3, 5,  6, 4,  5, 7,  8, 5,  6, 8,  9, 7,  8, 10, 11, 8,  9, 11, 12, 13, 14, 16, 17, 14, 
                                                    15, 17, 18, 16, 17, 19, 20, 17, 18, 20, 21, 19, 20, 22, 23, 20, 21, 23, 24, 25, 26, 28, 29, 26, 27, 29, 30, 
                                                    28, 29, 31, 32, 29, 30, 32, 33, 31, 32, 34, 35, 32, 33, 35, 36, 37, 38, 40, 41, 38, 39, 41, 42, 40, 41, 43, 
                                                    44, 41, 42, 44, 45, 43, 44, 46, 47, 44, 45, 47, 48, 49, 50, 52, 53, 50, 51, 53, 54, 52, 53, 55, 56, 53, 54, 
                                                    56, 57, 55, 56, 58, 59, 56, 57, 59, 60, 61, 62, 64, 65, 62, 63, 65, 66, 64, 65, 67, 68, 65, 66, 68, 69, 67, 
                                                    68, 70, 71, 68, 69, 71, 72, 73, 74, 76, 77, 74, 75, 77, 78, 76, 77, 79, 80, 77, 78, 80, 81, 79, 80, 82, 83, 
                                                    80, 81, 83, 84, 85, 86, 88, 89, 86, 87, 89, 90, 88, 89, 91, 92, 89, 90, 92, 93, 91, 92, 94, 95, 92, 93, 95, 96});
    
    image.linspace(1, 1);
    
    nd4j::ops::im2col op;
    auto results = op.execute({&image}, {}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode});
    auto column = results->at(0);

    // column->printIndexedBuffer();

    ASSERT_EQ(Status::OK(), results->status());

    ASSERT_TRUE(expected.isSameShape(column));
    ASSERT_TRUE(expected.equalsTo(column));    
    
    delete results;
}


#endif //LIBND4J_CONVOLUTIONTESTS_H

