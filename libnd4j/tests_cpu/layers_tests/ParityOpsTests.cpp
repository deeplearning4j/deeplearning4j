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
// Created by raver119 on 12.10.2017.
//

#include "testlayers.h"
#include <NDArray.h>
#include <ops/declarable/CustomOperations.h>


using namespace nd4j;
using namespace nd4j::ops;

class ParityOpsTests : public testing::Test {
public:

};


TEST_F(ParityOpsTests, TestZeroAs1) {
    NDArray<float> x('c', {10, 10});
    x.assign(1.0);

    NDArray<float> exp('c', {10, 10});
    exp.assign(0.0f);

    nd4j::ops::zeros_as<float> op;

    auto result = op.execute({&x}, {}, {});

    auto z = result->at(0);

    ASSERT_TRUE(z->isSameShape(&x));
    ASSERT_TRUE(z->equalsTo(&exp));

    delete result;
}

TEST_F(ParityOpsTests, TestMaximum1) {
    NDArray<float> x('c', {10, 10});
    x.assign(1.0);

    NDArray<float> y('c', {10, 10});
    y.assign(2.0);

    nd4j::ops::maximum<float> op;

    auto result = op.execute({&x, &y}, {}, {});

    auto z = result->at(0);

    ASSERT_TRUE(y.equalsTo(z));

    delete result;
}


TEST_F(ParityOpsTests, TestMinimum1) {
    NDArray<float> x('c', {10, 10});
    x.assign(1.0f);

    NDArray<float> y('c', {10, 10});
    y.assign(-2.0f);


    nd4j::ops::minimum<float> op;

    auto result = op.execute({&x, &y}, {}, {});

    auto z = result->at(0);

    ASSERT_TRUE(y.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, TestTear1) {
    NDArray<float> input('c', {10, 5});
    auto tads = input.allTensorsAlongDimension({1});
    for (int e = 0; e < tads->size(); e++) {
        ASSERT_EQ(5, tads->at(e)->lengthOf());
        tads->at(e)->assign((float) e + 1);
    }

    nd4j::ops::tear<float> op;

    auto result = op.execute({&input}, {}, {1});

    ASSERT_EQ(10, result->size());

    for (int e = 0; e < result->size(); e++)
        ASSERT_TRUE(tads->at(e)->equalsTo(result->at(e)));

    delete result;
    delete tads;
}

TEST_F(ParityOpsTests, TestUnstack1) {
    NDArray<float> input('c', {10, 5});
    auto tads = input.allTensorsAlongDimension({1});
    for (int e = 0; e < tads->size(); e++) {
        ASSERT_EQ(5, tads->at(e)->lengthOf());
        tads->at(e)->assign((float) e + 1);
    }

    nd4j::ops::unstack<float> op;

    auto result = op.execute({&input}, {}, {0});

    ASSERT_EQ(10, result->size());

    // result->at(0)->printShapeInfo("rz");
    // tads->at(0)->printShapeInfo("re");

    for (int e = 0; e < result->size(); e++)
        ASSERT_TRUE(tads->at(e)->equalsTo(result->at(e)));

    delete result;
    delete tads;
}



TEST_F(ParityOpsTests, TestUnstack2) {
    NDArray<float> input('c', {5,2,6});
    auto tads = input.allTensorsAlongDimension({0,1});
    for (int e = 0; e < tads->size(); e++) {
        ASSERT_EQ(10, tads->at(e)->lengthOf());
        tads->at(e)->assign((float) e + 1);
    }

    nd4j::ops::unstack<float> op;

    auto result = op.execute({&input}, {}, {2});

    ASSERT_EQ(6, result->size());

    for (int e = 0; e < result->size(); e++)
        ASSERT_TRUE(tads->at(e)->equalsTo(result->at(e)));

    delete result;
    delete tads;
}

TEST_F(ParityOpsTests, TestUnstack3) { 
    NDArray<float> input('c', {3,2,3});
    NDArray<float> exp('c', {3, 2}, {1.f, 4., 7., 10.f, 13.f,  16.f});
    input.linspace(1);

    nd4j::ops::unstack<float> op;

    auto result = op.execute({&input}, {}, {2});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(ParityOpsTests, TestUnstack4) { 
    NDArray<float> input('c', {3,2,3});
    NDArray<float> exp('c', {3, 3}, { 1, 2, 3, 7, 8, 9, 13, 14, 15.});
    input.linspace(1);

    nd4j::ops::unstack<float> op;

    auto result = op.execute({&input}, {}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, TestUnstack5) { 
    NDArray<float> input('c', {3,2,3});
    NDArray<float> exp('c', {2, 3}, { 1, 2, 3, 4, 5, 6});
    input.linspace(1);

    nd4j::ops::unstack<float> op;

    auto result = op.execute({&input}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, TestUnstack6) { 
    NDArray<float> input('c', {1, 1, 1});
    NDArray<float> exp('c', {1, 1}, {1});
    input.linspace(1);

    nd4j::ops::unstack<float> op;

    auto result = op.execute({&input}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, TestUnstack7) { 
    NDArray<float> input('c', {1, 1, 1});
    NDArray<float> exp('c', {1, 1}, {1});
    input.linspace(1);

    nd4j::ops::unstack<float> op;

    auto result = op.execute({&input}, {}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, TestUnstack8) { 
    NDArray<float> input('c', {1, 1});
    NDArray<float> exp('c', {1}, {1});
    input.linspace(1);

    nd4j::ops::unstack<float> op;

    auto result = op.execute({&input}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, TestUnstack9) {
    NDArray<float> input('c', {1, 1});
    NDArray<float> exp('c', {1}, {1});
    input.linspace(1);

    nd4j::ops::unstack<float> op;

    auto result = op.execute({&input}, {}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(ParityOpsTests, ExpandDimsTest1) {
    NDArray<float> input('c', {5, 5});
    input.linspace(1);
    auto reshaped = input.reshape('c', {5, 1, 5});

    nd4j::ops::expand_dims<float> op;
    auto result = op.execute({&input}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(reshaped->isSameShape(z));
    ASSERT_TRUE(reshaped->equalsTo(z));

    delete result;
    delete reshaped;

}


TEST_F(ParityOpsTests, ExpandDimsTest2) {
    NDArray<float> input('c', {3, 4});
    input.linspace(1);
    auto reshaped = input.reshape('c', {1, 3, 4});

    nd4j::ops::expand_dims<float> op;
    auto result = op.execute({&input}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(reshaped->isSameShape(z));
    ASSERT_TRUE(reshaped->equalsTo(z));

    delete result;
    delete reshaped;

}


TEST_F(ParityOpsTests, ExpandDimsTest3) {
    NDArray<float> input('c', {3, 4});
    input.linspace(1);
    auto reshaped = input.reshape('c', {3, 1, 4});

    nd4j::ops::expand_dims<float> op;
    auto result = op.execute({&input}, {}, {-2});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(reshaped->isSameShape(z));
    ASSERT_TRUE(reshaped->equalsTo(z));

    delete result;
    delete reshaped;

}

TEST_F(ParityOpsTests, ExpandDimsTest4) {
    NDArray<float> input('c', {3, 4});
    input.linspace(1);
    auto reshaped = input.reshape('c', {1, 3, 4});

    nd4j::ops::expand_dims<float> op;
    auto result = op.execute({&input}, {}, {-3});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(reshaped->isSameShape(z));
    ASSERT_TRUE(reshaped->equalsTo(z));

    delete result;
    delete reshaped;

}


TEST_F(ParityOpsTests, Test_Shape_1) {
    NDArray<float> x('c', {3, 4, 5, 6});
    NDArray<float> exp('c', {4}, {3, 4, 5, 6});

    nd4j::ops::shape_of<float> op;
    auto result = op.execute({&x}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(ParityOpsTests, Test_Equals_1) {
    NDArray<float> x('c', {1, 5}, {1, 2, 3, 4, 5});
    NDArray<float> y('c', {1, 5}, {1, 0, 3, 0, 5});
    NDArray<float> exp('c', {1, 5}, {1, 0, 1, 0, 1});

    nd4j::ops::equals<float> op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(ParityOpsTests, Test_NotEquals_1) {
    NDArray<float> x('c', {1, 5}, {1, 2, 3, 4, 5});
    NDArray<float> y('c', {1, 5}, {1, 0, 3, 0, 5});
    NDArray<float> exp('c', {1, 5}, {0, 1, 0, 1, 0});

    nd4j::ops::not_equals<float> op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, Test_Less_1) {
    NDArray<float> x('c', {1, 5}, {1, 2, 3, 4, 5});
    NDArray<float> y('c', {1, 5}, {5, 4, 3, 2, 1});
    NDArray<float> exp('c', {1, 5}, {1, 1, 0, 0, 0});

    nd4j::ops::less<float> op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, Test_LessEquals_1) {
    NDArray<float> x('c', {1, 5}, {1, 2, 3, 4, 5});
    NDArray<float> y('c', {1, 5}, {5, 4, 3, 2, 1});
    NDArray<float> exp('c', {1, 5}, {1, 1, 1, 0, 0});

    nd4j::ops::less_equal<float> op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, Test_GreaterEquals_1) {
    NDArray<float> x('c', {1, 5}, {1, 2, 3, 4, 5});
    NDArray<float> y('c', {1, 5}, {5, 4, 3, 2, 1});
    NDArray<float> exp('c', {1, 5}, {0, 0, 1, 1, 1});

    nd4j::ops::greater_equal<float> op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, Test_Greater_1) {
    NDArray<float> x('c', {1, 5}, {1, 2, 3, 4, 5});
    NDArray<float> y('c', {1, 5}, {5, 4, 3, 2, 1});
    NDArray<float> exp('c', {1, 5}, {0, 0, 0, 1, 1});

    nd4j::ops::greater<float> op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, Test_Where_1) {
    NDArray<float> mask('c', {3, 3}, {1, 1, 1,  0, 0, 0,  1, 1, 1});
    NDArray<float> x('c', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    NDArray<float> y('c', {3, 3}, {9, 8, 7, 6, 5, 4, 3, 2, 1});
    NDArray<float> exp('c', {3, 3}, {1, 2, 3, 6, 5, 4, 7, 8, 9});

    nd4j::ops::Where<float> op;
    auto result = op.execute({&mask, &x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    // z->printIndexedBuffer("result");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, Test_Where_2) {
    NDArray<float> mask('c', {1, 3}, {1, 0, 0});
    NDArray<float> x('c', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    NDArray<float> y('c', {3, 3}, {9, 8, 7, 6, 5, 4, 3, 2, 1});
    NDArray<float> exp('c', {3, 3}, {1, 2, 3, 6, 5, 4, 3, 2, 1});

    nd4j::ops::Where<float> op;
    auto result = op.execute({&mask, &x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(ParityOpsTests, Test_Where_3) {
    NDArray<float> mask('c', {2, 2, 3}, {0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0});
    NDArray<float> exp('c', {5, 3}, {0, 0, 1, 0, 0, 2, 0, 1, 1, 1, 0, 0, 1, 1, 2});

    nd4j::ops::Where<float> op;
    auto result = op.execute({&mask}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    // z->printShapeInfo("z");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, Test_Select_1) {
    NDArray<float> mask('c', {1, 3}, {1, 0, 0});
    NDArray<float> x('c', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    NDArray<float> y('c', {3, 3}, {9, 8, 7, 6, 5, 4, 3, 2, 1});
    NDArray<float> exp('c', {3, 3}, {1, 2, 3, 6, 5, 4, 3, 2, 1});

    nd4j::ops::select<float> op;
    auto result = op.execute({&mask, &x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, Test_Select_2) {
    NDArray<float> mask('c', {2, 2}, {1, 0, 1, 0});
    NDArray<float> x('c', {2, 2}, {1, 2, 3, 4 });
    NDArray<float> y('c', {2, 2}, {9, 8, 7, 6});
    NDArray<float> exp('c', {2, 2}, {1, 8, 3, 6});

    nd4j::ops::select<float> op;
    auto result = op.execute({&mask, &x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, Test_Select_3) {
    NDArray<float> mask('c', {1, 1}, {0});
    NDArray<float> x('c', {1, 1}, {1});
    NDArray<float> y('c', {1, 1}, {2});
    NDArray<float> exp('c', {1, 1}, {2});

    nd4j::ops::select<float> op;
    auto result = op.execute({&mask, &x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, Test_Reshape_TF_1) {
    NDArray<float> x('c', {2, 2}, {1, 2, 3, 4});
    NDArray<float> shape('c', {1, 3}, {1, 2, 2});

    NDArray<float> exp('c', {1, 2, 2}, {1, 2, 3, 4});
    
    nd4j::ops::reshape<float> op;

    auto result = op.execute({&x, &shape}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, Test_Bias_Add_1) {
    NDArray<float> x('c', {10, 5});
    x.assign(0.0);
    NDArray<float> bias('c', {1, 5}, {1, 2, 3, 4, 5});
    nd4j::ops::biasadd<float> op;

    auto result = op.execute({&x, &bias}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);


    auto tads = z->allTensorsAlongDimension({1});
    for (int e = 0; e < tads->size(); e++) {
        ASSERT_TRUE(bias.equalsTo(tads->at(e)));
    }

    delete tads;
    delete result;
}

TEST_F(ParityOpsTests, Test_Scatter_Add_1) {
    NDArray<float> matrix('c', {2, 2}, {1, 2, 3, 4});
    NDArray<float> idc('c', {1}, {0});
    NDArray<float> updates('c', {1, 2}, {1, 1});
    NDArray<float> exp('c', {2, 2}, {2, 3, 3, 4});

    nd4j::ops::scatter_add<float> op;
    auto result = op.execute({&matrix, &idc, &updates}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0); 

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, Test_Scatter_Add_2) {
    NDArray<float> vec('c', {4}, {1, 2, 3, 4});
    NDArray<float> idc('c', {1, 4}, {0, 1, 2, 3});
    NDArray<float> updates('c', {1, 4}, {1, 1, 1, 1});
    NDArray<float> exp('c', {1, 4}, {2, 3, 4, 5});

    nd4j::ops::scatter_add<float> op;
    auto result = op.execute({&vec, &idc, &updates}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0); 

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, Test_Scatter_Add_3) {
    NDArray<float> matrix('c', {2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    NDArray<float> idc('c', {1}, {0});
    NDArray<float> updates('c', {1, 2, 2}, {1, 1, 1, 1});
    NDArray<float> exp('c', {2, 2, 2}, {2, 3, 4, 5, 5, 6, 7, 8});

    nd4j::ops::scatter_add<float> op;
    auto result = op.execute({&matrix, &idc, &updates}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0); 

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, Test_Scatter_Add_4) {
    NDArray<float> matrix('c', {2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    NDArray<float> idc('c', {1, 2}, {0, 0});
    NDArray<float> updates('c', {1, 2, 2, 2}, {1, 1, 1, 1, 1, 1, 1, 1});
    NDArray<float> exp('c', {2, 2, 2}, {3, 4, 5, 6, 5, 6, 7, 8});

    nd4j::ops::scatter_add<float> op;
    auto result = op.execute({&matrix, &idc, &updates}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0); 

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, Test_Scatter_Add_5) {
    NDArray<float> matrix('c', {2, 2, 3}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    NDArray<float> idc('c', {2, 2}, {1, 1, 0, 0});
    NDArray<float> updates('c', {2, 2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    NDArray<float> exp('c', {2, 2, 3}, {9., 11., 13.,15., 17., 19., 9., 11., 13.,15., 17., 19.});

    nd4j::ops::scatter_add<float> op;
    auto result = op.execute({&matrix, &idc, &updates}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0); 

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, Test_Scatter_Add_6) {
    NDArray<float> matrix('c', {2, 2, 2}, {1, 1, 1, 1, 1, 1, 1, 1});
    NDArray<float> idc('c', {2, 2}, {1, 1, 0, 0});
    NDArray<float> updates('c', {2, 2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8});
    NDArray<float> exp('c', {2, 2, 2}, {7, 9, 11, 13, 7, 9, 11, 13});

    nd4j::ops::scatter_add<float> op;
    auto result = op.execute({&matrix, &idc, &updates}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0); 

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, Test_Scatter_Add_7) {
    NDArray<float> matrix('c', {10, 3}, {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f,9.f,10.f,11.f,12.f,13.f,14.f,15.f,16.f,17.f,18.f,19.f,20.f,21.f,22.f,23.f,24.f,25.f,26.f,27.f,28.f,29.f,30.f});
    NDArray<float> idc(5.f);
    NDArray<float> updates('c', {3}, {10.f, 20.f, 30.f});
    NDArray<float> exp('c', {10, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f,11.f,12.f, 13.f,14.f,15.f, 26.f,37.f,48.f, 19.f,20.f,21.f, 22.f,23.f,24.f, 25.f,26.f,27.f, 28.f,29.f,30.f});

    nd4j::ops::scatter_add<float> op;
    auto result = op.execute({&matrix, &idc, &updates}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0); 

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, scatterMax_test1) {
    NDArray<float> matrix('c', {2, 2}, {1, 2, 3, 4});
    NDArray<float> idc('c', {1}, {0});
    NDArray<float> updates('c', {1, 2}, {10, 1});
    NDArray<float> exp('c', {2, 2}, {10, 2, 3, 4});

    nd4j::ops::scatter_max<float> op;
    auto result = op.execute({&matrix, &idc, &updates}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0); 

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, scatterMax_test2) {
    NDArray<float> vec('c', {4}, {1, 2, 3, 4});
    NDArray<float> idc('c', {1, 4}, {0, 1, 2, 3});
    NDArray<float> updates('c', {1, 4}, {10, 1, 30, 1});
    NDArray<float> exp('c', {1, 4}, {10, 2, 30, 4});

    nd4j::ops::scatter_max<float> op;
    auto result = op.execute({&vec, &idc, &updates}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0); 

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, scatterMax_test3) {
    NDArray<float> matrix('c', {2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    NDArray<float> idc('c', {1}, {0});
    NDArray<float> updates('c', {1, 2, 2}, {10, 1, 30, 1});
    NDArray<float> exp('c', {2, 2, 2}, {10, 2, 30, 4, 5, 6, 7, 8});

    nd4j::ops::scatter_max<float> op;
    auto result = op.execute({&matrix, &idc, &updates}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0); 

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, scatterMax_test4) {
    NDArray<float> matrix('c', {2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    NDArray<float> idc('c', {1, 2}, {0, 0});
    NDArray<float> updates('c', {1, 2, 2, 2}, {1,10,1,10, 1,1,10,1.});
    NDArray<float> exp('c', {2, 2, 2}, {1, 10, 10, 10, 5, 6, 7, 8});

    nd4j::ops::scatter_max<float> op;
    auto result = op.execute({&matrix, &idc, &updates}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);     

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, scatterMax_test5) {
    NDArray<float> matrix('c', {2, 2, 3}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    NDArray<float> idc('c', {2, 2}, {1, 1, 0, 0});
    NDArray<float> updates('c', {2, 2, 2, 3}, {2,10,1,10, 2,10,1,10, 2,10,1,10,  10,2,10,1, 10,2,10,1, 10,2,10,1.});
    NDArray<float> exp('c', {2, 2, 3}, {10, 2, 10,   2, 10, 2,   2, 10, 2,   10, 2, 10});

    nd4j::ops::scatter_max<float> op;
    auto result = op.execute({&matrix, &idc, &updates}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0); 

   ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, scatterMax_test6) {
    NDArray<float> matrix('c', {2, 2, 2}, {1, 1, 1, 1, 1, 1, 1, 1});
    NDArray<float> idc('c', {2, 2}, {1, 1, 0, 0});
    NDArray<float> updates('c', {2, 2, 2, 2}, {0,2,0,2, 0,2,0,2, 2,0,2,0.,  2,0,2,0});
    NDArray<float> exp('c', {2, 2, 2}, {2, 1, 2, 1, 1, 2, 1, 2});

    nd4j::ops::scatter_max<float> op;
    auto result = op.execute({&matrix, &idc, &updates}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0); 

   ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(ParityOpsTests, scatterMin_test1) {
    NDArray<float> matrix('c', {2, 2}, {1, 2, 3, 4});
    NDArray<float> idc('c', {1}, {0});
    NDArray<float> updates('c', {1, 2}, {-1, 1});
    NDArray<float> exp('c', {2, 2}, {-1, 1, 3, 4});

    nd4j::ops::scatter_min<float> op;
    auto result = op.execute({&matrix, &idc, &updates}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0); 

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, scatterMin_test2) {
    NDArray<float> vec('c', {4}, {1, 2, 3, 4});
    NDArray<float> idc('c', {1, 4}, {0, 1, 2, 3});
    NDArray<float> updates('c', {1, 4}, {10, 1, 30, 1});
    NDArray<float> exp('c', {1, 4}, {1, 1, 3, 1});

    nd4j::ops::scatter_min<float> op;
    auto result = op.execute({&vec, &idc, &updates}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0); 

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, scatterMin_test3) {
    NDArray<float> matrix('c', {2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    NDArray<float> idc('c', {1}, {0});
    NDArray<float> updates('c', {1, 2, 2}, {10, 1, 30, 2});
    NDArray<float> exp('c', {2, 2, 2}, {1, 1, 3, 2, 5, 6, 7, 8});

    nd4j::ops::scatter_min<float> op;
    auto result = op.execute({&matrix, &idc, &updates}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0); 

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, scatterMin_test4) {
    NDArray<float> matrix('c', {2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    NDArray<float> idc('c', {1, 2}, {0, 0});
    NDArray<float> updates('c', {1, 2, 2, 2}, {1,10,1,10, 1,1,10,1.});
    NDArray<float> exp('c', {2, 2, 2}, {1, 1, 1, 1, 5, 6, 7, 8});

    nd4j::ops::scatter_min<float> op;
    auto result = op.execute({&matrix, &idc, &updates}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);     

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////
TEST_F(ParityOpsTests, scatterND_test1) {    
    
    NDArray<float> indices('c', {2, 1}, {1.f, 0.f});
    NDArray<float> updates('c', {2, 4}, {10.f, 20.f, 30.f, 40.f, 50.f, 60.f, 70.f, 80.f});
    NDArray<float> shape('c', {2}, {3, 4});
    NDArray<float> exp('c', {3, 4}, {50.f, 60.f, 70.f, 80.f, 10.f, 20.f, 30.f, 40.f, 0.f,  0.f,  0.f,  0.f});

    nd4j::ops::scatter_nd<float> op;
    auto result = op.execute({&indices, &updates, &shape}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0); 

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////
TEST_F(ParityOpsTests, scatterND_test2) {    
    
    NDArray<float> indices('c', {3, 1}, {4.f, 2.f, 0.f});
    NDArray<float> updates('c', {3, 4});
    NDArray<float> shape('c', {2}, {5.f, 4.f});
    NDArray<float> exp('c', {5, 4}, {9.f,10.f,11.f,12.f, 0.f, 0.f, 0.f, 0.f, 5.f, 6.f, 7.f, 8.f, 0.f, 0.f, 0.f, 0.f, 1.f, 2.f, 3.f, 4.f});
    updates.linspace(1.f);

    nd4j::ops::scatter_nd<float> op;
    auto result = op.execute({&indices, &updates, &shape}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0); 

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////
TEST_F(ParityOpsTests, scatterND_test3) {    
    
    NDArray<float> indices('c', {2, 3, 1}, {0.f, 2.f, 7.f, 3.f, 6.f, 9.f});
    NDArray<float> updates('c', {2,3, 3,4});
    NDArray<float> shape('c', {3}, {10.f, 3.f, 4.f});
    NDArray<float> exp('c', {10, 3, 4}, {1.f,  2.f,  3.f,  4., 5.f,  6.f,  7.f,  8., 9.f, 10.f, 11.f, 12., 0.f,  0.f,  0.f,  0., 0.f,  0.f,  0.f,  0., 0.f,  0.f,  0.f,  0.,
                                        13.f, 14.f, 15.f, 16.,17.f, 18.f, 19.f, 20.,21.f, 22.f, 23.f, 24.,37.f, 38.f, 39.f, 40.,41.f, 42.f, 43.f, 44.,45.f, 46.f, 47.f, 48.,
                                         0.f,  0.f,  0.f,  0., 0.f,  0.f,  0.f,  0., 0.f,  0.f,  0.f,  0., 0.f,  0.f,  0.f,  0., 0.f,  0.f,  0.f,  0., 0.f,  0.f,  0.f,  0.,
                                        49.f, 50.f, 51.f, 52.,53.f, 54.f, 55.f, 56.,57.f, 58.f, 59.f, 60.,25.f, 26.f, 27.f, 28.,29.f, 30.f, 31.f, 32.,33.f, 34.f, 35.f, 36.,
                                         0.f,  0.f,  0.f,  0., 0.f,  0.f,  0.f,  0., 0.f,  0.f,  0.f,  0.,61.f, 62.f, 63.f, 64.,65.f, 66.f, 67.f, 68.,69.f, 70.f, 71.f, 72.,});
    updates.linspace(1.f);

    nd4j::ops::scatter_nd<float> op;
    auto result = op.execute({&indices, &updates, &shape}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0); 

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

 