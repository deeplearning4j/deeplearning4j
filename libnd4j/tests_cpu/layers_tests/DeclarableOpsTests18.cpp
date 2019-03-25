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
// Created by raver119 on 09.02.18.
//


#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <helpers/helper_hash.h>
#include <NDArray.h>
#include <array/NDArrayList.h>


using namespace nd4j;
using namespace nd4j::graph;


class DeclarableOpsTests18 : public testing::Test {
public:

    DeclarableOpsTests18() {
        printf("\n");
        fflush(stdout);
    }
};

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Min_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {4}, {1.f, 2.f, 3.f, 4.f});
    x.linspace(1);

    nd4j::ops::reduce_min op;
    auto result = op.execute({&x}, {}, {0, 1});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Min_2) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1,1,4}, {1.f, 2.f, 3.f, 4.f});
    x.linspace(1);

    nd4j::ops::reduce_min op;
    auto result = op.execute({&x}, {1.}, {0, 1});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Min_3) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {3}, {1.f, 5.f, 9.f});
    x.linspace(1);

    nd4j::ops::reduce_min op;
    auto result = op.execute({&x}, {}, {0, 2});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Min_4) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1,3,1}, {1.f, 5.f, 9.f});
    x.linspace(1);

    nd4j::ops::reduce_min op;
    auto result = op.execute({&x}, {1.}, {0, 2});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Min_5) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(1.f);
    x.linspace(1);

    nd4j::ops::reduce_min op;
    auto result = op.execute({&x}, {}, {});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Min_6) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(1.f);
    x.linspace(1);

    nd4j::ops::reduce_min op;
    auto result = op.execute({&x}, {}, {0,1,2});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Min_7) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1, 1, 1}, {1.f});
    x.linspace(1);
//    x.printIndexedBuffer("Input with shape (2, 3, 4) is");
    nd4j::ops::reduce_min op;
    auto result = op.execute({&x}, {1.}, {0,1,2});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Max_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {4}, {21.f, 22.f, 23.f, 24.f});
    x.linspace(1);

    nd4j::ops::reduce_max op;
    auto result = op.execute({&x}, {}, {0,1});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
//    output->printShapeInfo("Output shape");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Max_2) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>('c', {1,1,4}, {21.f, 22.f, 23.f, 24.f});
    x.linspace(1);

    nd4j::ops::reduce_max op;
    auto result = op.execute({&x}, {1.}, {0, 1});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Max_3) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {3}, {16.f, 20.f, 24.f});
    x.linspace(1);

    nd4j::ops::reduce_max op;
    auto result = op.execute({&x}, {}, {0, 2});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Max_4) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1,3,1}, {16.f, 20.f, 24.f});
    x.linspace(1);

    nd4j::ops::reduce_max op;
    auto result = op.execute({&x}, {1.}, {0, 2});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Max_5) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(24.f);
    x.linspace(1);

    nd4j::ops::reduce_max op;
    auto result = op.execute({&x}, {}, {});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Max_6) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(24.f);
    x.linspace(1);

    nd4j::ops::reduce_max op;
    auto result = op.execute({&x}, {}, {0,1,2});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Max_7) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1, 1, 1}, {24.f});
    x.linspace(1);
//    x.printIndexedBuffer("Input with shape (2, 3, 4) is");
    nd4j::ops::reduce_max op;
    auto result = op.execute({&x}, {1.}, {0,1,2});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Norm1_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {4}, {66.f, 72.f, 78.f, 84.f});
    x.linspace(1);

    nd4j::ops::reduce_norm1 op;
    auto result = op.execute({&x}, {}, {0,1});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Norm1_2) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>('c', {1,1,4}, {66.f, 72.f, 78.f, 84.f});
    x.linspace(1);

    nd4j::ops::reduce_norm1 op;
    auto result = op.execute({&x}, {1.}, {0, 1});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Norm1_3) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {3}, {68.f, 100.f, 132.f});
    x.linspace(1);

    nd4j::ops::reduce_norm1 op;
    auto result = op.execute({&x}, {}, {0, 2});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Norm1_4) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1,3,1}, {68.f, 100.f, 132.f});
    x.linspace(1);

    nd4j::ops::reduce_norm1 op;
    auto result = op.execute({&x}, {1.}, {0, 2});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Norm1_5) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(300.f);
    x.linspace(1);

    nd4j::ops::reduce_norm1 op;
    auto result = op.execute({&x}, {}, {});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Norm1_6) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(300.f);
    x.linspace(1);

    nd4j::ops::reduce_norm1 op;
    auto result = op.execute({&x}, {}, {0,1,2});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Norm1_7) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1, 1, 1}, {300.f});
    x.linspace(1);
//    x.printIndexedBuffer("Input with shape (2, 3, 4) is");
    nd4j::ops::reduce_norm1 op;
    auto result = op.execute({&x}, {1.}, {0,1,2});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

TEST_F(DeclarableOpsTests18, Test_Reduce_Norm2_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {4}, {31.7175f, 33.823071f, 35.97221f, 38.15757f});
    x.linspace(1);

    nd4j::ops::reduce_norm2 op;
    auto result = op.execute({&x}, {}, {0,1});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Norm2_2) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>('c', {1,1,4}, {31.7175f, 33.823071f, 35.97221f, 38.15757f});
    x.linspace(1);

    nd4j::ops::reduce_norm2 op;
    auto result = op.execute({&x}, {1.}, {0, 1});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Norm2_3) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {3}, {29.597298f, 39.344631f, 49.759422f});
    x.linspace(1);

    nd4j::ops::reduce_norm2 op;
    auto result = op.execute({&x}, {}, {0, 2});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Norm2_4) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1,3,1}, {29.597298f, 39.344631f, 49.759422f});
    x.linspace(1);

    nd4j::ops::reduce_norm2 op;
    auto result = op.execute({&x}, {1.}, {0, 2});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Norm2_5) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(70.f);
    x.linspace(1);

    nd4j::ops::reduce_norm2 op;
    auto result = op.execute({&x}, {}, {});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Norm2_6) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(70.f);
    x.linspace(1);

    nd4j::ops::reduce_norm2 op;
    auto result = op.execute({&x}, {}, {0,1,2});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Norm2_7) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1, 1, 1}, {70.f});
    x.linspace(1);
//    x.printIndexedBuffer("Input with shape (2, 3, 4) is");
    nd4j::ops::reduce_norm2 op;
    auto result = op.execute({&x}, {1.}, {0,1,2});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_NormMax_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {4}, {21.f, 22.f, 23.f, 24.f});
    x.linspace(1);

    nd4j::ops::reduce_norm_max op;
    auto result = op.execute({&x}, {}, {0,1});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_NormMax_2) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1,1,4}, {21.f, 22.f, 23.f, 24.f});
    x.linspace(1);

    nd4j::ops::reduce_norm_max op;
    auto result = op.execute({&x}, {1.f}, {0,1});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_NormMax_3) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {3}, {16.f, 20.f, 24.f});
    x.linspace(1);

    nd4j::ops::reduce_norm_max op;
    auto result = op.execute({&x}, {}, {0,2});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_NormMax_4) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1, 3, 1}, {16.f, 20.f, 24.f});
    x.linspace(1);

    nd4j::ops::reduce_norm_max op;
    auto result = op.execute({&x}, {1.f}, {0,2});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_NormMax_5) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(24.f);
    x.linspace(1);

    nd4j::ops::reduce_norm_max op;
    auto result = op.execute({&x}, {}, {});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_NormMax_6) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(24.f);
    x.linspace(1);

    nd4j::ops::reduce_norm_max op;
    auto result = op.execute({&x}, {}, {0, 1, 2});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_NormMax_7) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1, 1, 1}, {24.f});
    x.linspace(1);

    nd4j::ops::reduce_norm_max op;
    auto result = op.execute({&x}, {1.f}, {});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_SquaredNorm_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {4}, {1006.f, 1144.f, 1294.f, 1456.f});
    x.linspace(1);

    nd4j::ops::reduce_sqnorm op;
    auto result = op.execute({&x}, {}, {0,1});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_SquaredNorm_2) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1,1,4}, {1006.f, 1144.f, 1294.f, 1456.f});
    x.linspace(1);

    nd4j::ops::reduce_sqnorm op;
    auto result = op.execute({&x}, {1.f}, {0,1});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_SquaredNorm_3) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {3}, {876.f, 1548.f, 2476.f});
    x.linspace(1);

    nd4j::ops::reduce_sqnorm op;
    auto result = op.execute({&x}, {}, {0,2});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_SquaredNorm_4) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1, 3, 1}, {876.f, 1548.f, 2476.f});
    x.linspace(1);

    nd4j::ops::reduce_sqnorm op;
    auto result = op.execute({&x}, {1.f}, {0,2});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_SquaredNorm_5) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(4900.f);
    x.linspace(1);

    nd4j::ops::reduce_sqnorm op;
    auto result = op.execute({&x}, {}, {});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_SquaredNorm_6) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(4900.f);
    x.linspace(1);

    nd4j::ops::reduce_sqnorm op;
    auto result = op.execute({&x}, {}, {0, 1, 2});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_SquaredNorm_7) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1, 1, 1}, {4900.f});
    x.linspace(1);

    nd4j::ops::reduce_sqnorm op;
    auto result = op.execute({&x}, {1.f}, {});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Sum_BP_1) {

    auto input = NDArrayFactory::create<double>('c', {3, 4},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.});
    auto eps = NDArrayFactory::create<double>(0.5f);
    auto exp = NDArrayFactory::create<double>('c', {3, 4}, {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,0.5f});
    //************************************//

    nd4j::ops::reduce_sum_bp op;
    auto result = op.execute({&input, &eps}, {}, {});

    ASSERT_EQ(Status::OK(), result->status());
    auto z = result->at(0);
//    z->printIndexedBuffer("Result is ");
//    z->printShapeInfo();
    ASSERT_TRUE(exp.equalsTo(z));
    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Sum_BP_2) {

    auto input = NDArrayFactory::create<double>('c', {3, 4},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.});
    auto eps = NDArrayFactory::create<double>('c', {1, 1}, {0.5f});
    auto exp = NDArrayFactory::create<double>('c', {3, 4}, {0.5f, 0.5f, 0.5f, 0.5f,
                                     0.5f, 0.5f, 0.5f, 0.5f,
                                     0.5f, 0.5f, 0.5f,0.5f});
    //************************************//

    nd4j::ops::reduce_sum_bp op;
    auto result = op.execute({&input, &eps}, {1.f}, {});

    ASSERT_EQ(Status::OK(), result->status());
    auto z = result->at(0);
//  z->printIndexedBuffer("Result is ");
//  z->printShapeInfo();
    ASSERT_TRUE(exp.equalsTo(z));
    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Sum_BP_3) {

    auto input = NDArrayFactory::create<double>('c', {3, 4},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.});
    auto eps = NDArrayFactory::create<double>('c', {4}, {1.f, 2.f, 3.f, 4.f});
    auto exp = NDArrayFactory::create<double>('c', {3, 4}, {1.f, 2.f, 3.f, 4.f,
                                     1.f, 2.f, 3.f, 4.f,
                                     1.f, 2.f, 3.f, 4.f});
    //************************************//

    nd4j::ops::reduce_sum_bp op;
    auto result = op.execute({&input, &eps}, {}, {0});

    ASSERT_EQ(Status::OK(), result->status());
    auto z = result->at(0);
//    z->printIndexedBuffer("Result is ");
//    z->printShapeInfo();
    ASSERT_TRUE(exp.equalsTo(z));
    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Sum_BP_4) {

    auto input = NDArrayFactory::create<double>('c', {3, 4},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.});
    auto eps = NDArrayFactory::create<double>('c', {1, 4}, {1.f, 2.f, 3.f, 4.f});
    auto exp = NDArrayFactory::create<double>('c', {3, 4}, {1.f, 2.f, 3.f, 4.f,
                                     1.f, 2.f, 3.f, 4.f,
                                     1.f, 2.f, 3.f, 4.f});
    //************************************//

    nd4j::ops::reduce_sum_bp op;
    auto result = op.execute({&input, &eps}, {1.f}, {0});

    ASSERT_EQ(Status::OK(), result->status());
    auto z = result->at(0);
//    z->printIndexedBuffer("Result is ");
//    z->printShapeInfo();
    ASSERT_TRUE(exp.equalsTo(z));
    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Prod_BP_1) {

    auto input = NDArrayFactory::create<double>('c', {3, 5},   {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f});
    auto eps = NDArrayFactory::create<double>(1307674368000.f);
    //************************************//
//    auto exp = NDArrayFactory::create<double>('c', {3, 4}, {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,0.5f});
    //************************************//
    auto exp = NDArrayFactory::create<double>('c', {3, 5},   {1710012166826558903812096.f, 855006083413279451906048.f, 570004067618451974258688.f,
                                       427503041706639725953024.f, 342002454982589992140800.f, 285002033809225987129344.f,
                                       244287457550765131825152.f, 213751520853319862976512.f, 190001355872817324752896.f,
                                       171001227491294996070400.f, 155455648254341989531648.f, 142501016904612993564672.f,
                                       131539399526781282156544.f, 122143728775382565912576.f, 114000815325130245799936.f});

    nd4j::ops::reduce_prod_bp op;
    auto result = op.execute({&input, &eps}, {}, {});

    ASSERT_EQ(Status::OK(), result->status());
    auto z = result->at(0);
//    z->printIndexedBuffer("Result is ");
//    z->printShapeInfo();
    ASSERT_TRUE(exp.equalsTo(z));
    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Prod_BP_2) {

    auto input = NDArrayFactory::create<double>('c', {3, 4},   {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f});
    auto eps = NDArrayFactory::create<double>(0.5f);
    //************************************//
//    auto exp = NDArrayFactory::create<double>('c', {3, 4}, {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,0.5f});
    //************************************//
    auto exp = NDArrayFactory::create<double>('c', {3, 4});

    nd4j::ops::reduce_prod_bp op;
    nd4j::ops::reduce_prod op_exp;
    auto res = op_exp.execute({&input}, {}, {});
    auto result = op.execute({&input, &eps}, {}, {});
    exp.assign(res->at(0)->e<double>(0));
    exp /= input;
    exp *= eps.e<double>(0);
    ASSERT_EQ(Status::OK(), result->status());
    auto z = result->at(0);
    //z->printIndexedBuffer("Result is ");
    //exp.printIndexedBuffer("Expected");
//    z->printShapeInfo();
    ASSERT_TRUE(exp.equalsTo(z));
    delete result;
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Prod_BP_3) {

    auto input = NDArrayFactory::create<double>('c', {3, 4},   {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f});
    auto eps = NDArrayFactory::create<double>('c', {1, 4}, {1.f, 2.f, 3.f, 4.f});
    //************************************//
    auto exp = NDArrayFactory::create<double>('c', {3, 4}, {45.f, 120.f, 231.f, 384.f, 9.f, 40.f, 99.f, 192.f, 5.f, 24.f, 63.f, 128.f});

    nd4j::ops::reduce_prod_bp op;
    //nd4j::ops::reduce_prod op_exp;
    auto result = op.execute({&input, &eps}, {1.f}, {0});

    ASSERT_EQ(Status::OK(), result->status());
    auto z = result->at(0);
//    z->printIndexedBuffer("Result is ");
//    exp.printIndexedBuffer("Expected");
//    z->printShapeInfo();
    ASSERT_TRUE(exp.equalsTo(z));
    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Prod_BP_03) {

    auto input = NDArrayFactory::create<double>('c', {3, 4},   {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f});
    auto eps = NDArrayFactory::create<double>('c', {1, 4}, {1.f, 2.f, 3.f, 4.f});
    //************************************//
    auto exp = NDArrayFactory::create<double>('c', {3, 4}, {45.f, 120.f, 231.f, 384.f, 9.f, 40.f, 99.f, 192.f, 5.f, 24.f, 63.f, 128.f});
    auto axis = NDArrayFactory::create<int>('c', {1}, 0);
    nd4j::ops::reduce_prod_bp op;
    //nd4j::ops::reduce_prod op_exp;
    auto result = op.execute({&input, &eps, &axis}, {}, {}, {true});

    ASSERT_EQ(Status::OK(), result->status());
    auto z = result->at(0);
//    z->printIndexedBuffer("Result is ");
//    exp.printIndexedBuffer("Expected");
//    z->printShapeInfo();
    ASSERT_TRUE(exp.equalsTo(z));
    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Prod_BP_4) {

    auto input = NDArrayFactory::create<double>('c', {3, 4},   {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f});
    auto eps = NDArrayFactory::create<double>('c', {4}, {1.f, 2.f, 3.f, 4.f});
    //************************************//
    auto exp = NDArrayFactory::create<double>('c', {3, 4}, {45.f, 120.f, 231.f, 384.f, 9.f, 40.f, 99.f, 192.f, 5.f, 24.f, 63.f, 128.f});

    nd4j::ops::reduce_prod_bp op;
    nd4j::ops::reduce_prod op_exp;
//    auto res = op_exp.execute({&input}, {}, {});
    auto result = op.execute({&input, &eps}, {0.f}, {0});

    ASSERT_EQ(Status::OK(), result->status());
    auto z = result->at(0);
//    z->printIndexedBuffer("Result is ");
//    exp.printIndexedBuffer("Expected");
//    z->printShapeInfo();
    ASSERT_TRUE(exp.equalsTo(z));
    delete result;
//    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Prod_BP_5) {

    auto input = NDArrayFactory::create<double>('c', {3, 4},   {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f});
    auto eps = NDArrayFactory::create<double>('c', {3}, {1.f, 2.f, 3.f});
    //************************************//
    auto exp = NDArrayFactory::create<double>('c', {3, 4}, {24.f, 12.f, 8.f, 6.f, 672.f, 560.f, 480.f, 420.f, 3960.f, 3564.f, 3240.f, 2970.f});

    nd4j::ops::reduce_prod_bp op;
    nd4j::ops::reduce_prod op_exp;
//    auto res = op_exp.execute({&input}, {}, {});
    auto result = op.execute({&input, &eps}, {0.f}, {1});

    ASSERT_EQ(Status::OK(), result->status());
    auto z = result->at(0);
//    z->printIndexedBuffer("Result is ");
//    exp.printIndexedBuffer("Expected");
//    z->printShapeInfo();
    ASSERT_TRUE(exp.equalsTo(z));
    delete result;
//    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Min_BP_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto eps = NDArrayFactory::create<double>('c', {4}, {1.f, 2.f, 3.f, 4.f});
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4});
    exp.p(0, eps.e<double>(0));
    exp.p(1, eps.e<double>(1));
    exp.p(2, eps.e<double>(2));
    exp.p(3, eps.e<double>(3));
    x.linspace(1);
//    x.printIndexedBuffer("Input is");
//    exp.printIndexedBuffer("Expected ");
    nd4j::ops::reduce_min_bp op;
    auto result = op.execute({&x, &eps}, {}, {0, 1});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Min_BP_2) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto eps = NDArrayFactory::create<double>('c', {1, 1, 4}, {1.f, 2.f, 3.f, 4.f});
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4});
    exp.p(0, eps.e<double>(0));
    exp.p(1, eps.e<double>(1));
    exp.p(2, eps.e<double>(2));
    exp.p(3, eps.e<double>(3));
    x.linspace(1);
//    x.printIndexedBuffer("Input is");
//    exp.printIndexedBuffer("Expected ");
    nd4j::ops::reduce_min_bp op;
    auto result = op.execute({&x, &eps}, {1.f}, {0, 1});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}
////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Min_BP_02) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto eps = NDArrayFactory::create<double>('c', {1, 1, 4}, {1.f, 2.f, 3.f, 4.f});
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4});
    exp.p(0, eps.e<double>(0));
    exp.p(1, eps.e<double>(1));
    exp.p(2, eps.e<double>(2));
    exp.p(3, eps.e<double>(3));
    auto axes = NDArrayFactory::create<int>({0,1});
    x.linspace(1);
//    x.printIndexedBuffer("Input is");
//    exp.printIndexedBuffer("Expected ");
    nd4j::ops::reduce_min_bp op;
    auto result = op.execute({&x, &eps, &axes}, {}, {}, {true});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Min_BP_3) {

    auto x = NDArrayFactory::create<double>('c', {3, 4});
    auto eps = NDArrayFactory::create<double>('c', {1, 1}, {0.5f});
    auto exp = NDArrayFactory::create<double>('c', {3, 4});
    x.linspace(1);
    x.p(2,2, -1.f);
    exp.p(2,2, 0.5f);
//    x.printIndexedBuffer("Input is");
//    exp.printIndexedBuffer("Expected ");
    nd4j::ops::reduce_min_bp op;
    auto result = op.execute({&x, &eps}, {1.f}, {});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Min_BP_4) {

    auto x = NDArrayFactory::create<double>('c', {3, 4});
    auto eps = NDArrayFactory::create<double>(0.5f);
    auto exp = NDArrayFactory::create<double>('c', {3, 4});
    x.linspace(1);
    x.p(2,2, -1.f);
    exp.p(2,2, 0.5f);
//    x.printIndexedBuffer("Input is");
//    exp.printIndexedBuffer("Expected ");
    nd4j::ops::reduce_min_bp op;
    auto result = op.execute({&x, &eps}, {}, {});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Min_BP_5) {

    auto x = NDArrayFactory::create<double>('c', {4, 4});
    auto eps = NDArrayFactory::create<double>('c', {4}, {1.f, 2.f, 3.f, 4.f});
    auto exp = NDArrayFactory::create<double>('c', {4, 4});
    x.linspace(1);
    x.p(0,0, -1.f);
    x.p(1,1, -2.f);
    x.p(2,2, -3.f);
    x.p(3,3, -4.f);
    exp.p(0,0, 1.f);
    exp.p(1,1, 2.f);
    exp.p(2,2, 3.f);
    exp.p(3,3, 4.f);
//    exp(2,2) = 0.5f;
//    x.printIndexedBuffer("Input is");
//    exp.printIndexedBuffer("Expected ");
    nd4j::ops::reduce_min_bp op;
    auto result = op.execute({&x, &eps}, {}, {0});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Min_BP_6) {

    auto x = NDArrayFactory::create<double>('c', {4, 4});
    auto eps = NDArrayFactory::create<double>('c', {1,4}, {1.f, 2.f, 3.f, 4.f});
    auto exp = NDArrayFactory::create<double>('c', {4, 4});
    x.linspace(1);
    x.p(0,0, -1.f);
    x.p(1,1, -2.f);
    x.p(2,2, -3.f);
    x.p(3,3, -4.f);
    exp.p(0,0, 1.f);
    exp.p(1,1, 2.f);
    exp.p(2,2, 3.f);
    exp.p(3,3, 4.f);
//    exp(2,2) = 0.5f;
//    x.printIndexedBuffer("Input is");
//    exp.printIndexedBuffer("Expected ");
    nd4j::ops::reduce_min_bp op;
    auto result = op.execute({&x, &eps}, {1.f}, {0});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Max_BP_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto eps = NDArrayFactory::create<double>('c', {4}, {21.f, 22.f, 23.f, 24.f});
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4});
    exp.p(20, eps.e<double>(0));
    exp.p(21, eps.e<double>(1));
    exp.p(22, eps.e<double>(2));
    exp.p(23, eps.e<double>(3));
    x.linspace(1);
//    x.printIndexedBuffer("Input is");
//    exp.printIndexedBuffer("Expected ");
    nd4j::ops::reduce_max_bp op;
    auto result = op.execute({&x, &eps}, {}, {0, 1});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Max_BP_2) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto eps = NDArrayFactory::create<double>('c', {1, 1, 4}, {21.f, 22.f, 23.f, 24.f});
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4});
    exp.p(20, eps.e<double>(0));
    exp.p(21, eps.e<double>(1));
    exp.p(22, eps.e<double>(2));
    exp.p(23, eps.e<double>(3));
    x.linspace(1);
//    x.printIndexedBuffer("Input is");
//    exp.printIndexedBuffer("Expected ");
    nd4j::ops::reduce_max_bp op;
    auto result = op.execute({&x, &eps}, {1.f}, {0, 1});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Max_BP_02) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto eps = NDArrayFactory::create<double>('c', {1, 1, 4}, {21.f, 22.f, 23.f, 24.f});
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4});
    exp.p(20, eps.e<double>(0));
    exp.p(21, eps.e<double>(1));
    exp.p(22, eps.e<double>(2));
    exp.p(23, eps.e<double>(3));
    auto axes = NDArrayFactory::create<int>({0, 1});
    x.linspace(1);
//    x.printIndexedBuffer("Input is");
//    exp.printIndexedBuffer("Expected ");
    nd4j::ops::reduce_max_bp op;
    auto result = op.execute({&x, &eps, &axes}, {}, {}, {true});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Max_BP_3) {

    auto x = NDArrayFactory::create<double>('c', {4, 4});
    auto eps = NDArrayFactory::create<double>('c', {4}, {1.f, 2.f, 3.f, 4.f});
    auto exp = NDArrayFactory::create<double>('c', {4, 4});
    x.linspace(1);
    x.p(0,0, 21.f);
    x.p(1,1, 22.f);
    x.p(2,2, 23.f);
    x.p(3,3, 24.f);
    exp.p(0,0, 1.f);
    exp.p(1,1, 2.f);
    exp.p(2,2, 3.f);
    exp.p(3,3, 4.f);
//    x.printIndexedBuffer("Input is");
//    exp.printIndexedBuffer("Expected ");
    nd4j::ops::reduce_max_bp op;
    auto result = op.execute({&x, &eps}, {}, {0});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Max_BP_4) {

    auto x = NDArrayFactory::create<double>('c', {4, 4});
    auto eps = NDArrayFactory::create<double>('c', {1,4}, {1.f, 2.f, 3.f, 4.f});
    auto exp = NDArrayFactory::create<double>('c', {4, 4});
    x.linspace(1);
    x.p(0,0, 21.f);
    x.p(1,1, 22.f);
    x.p(2,2, 23.f);
    x.p(3,3, 24.f);
    exp.p(0,0, 1.f);
    exp.p(1,1, 2.f);
    exp.p(2,2, 3.f);
    exp.p(3,3, 4.f);

//    x.printIndexedBuffer("Input is");
//    exp.printIndexedBuffer("Expected ");
    nd4j::ops::reduce_max_bp op;
    auto result = op.execute({&x, &eps}, {1.f}, {0});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Norm1_BP_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto eps = NDArrayFactory::create<double>(5.f);
    x.linspace(1);
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4});
    x.p(12, -2.f);
    x.p(20, -3.f);
    exp.assign(5.f);
    exp.p(12, -exp.e<double>(12));
    exp.p(20, -exp.e<double>(20));
    nd4j::ops::reduce_norm1_bp op;
    auto result = op.execute({&x, &eps}, {}, {});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Norm1_BP_2) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto eps = NDArrayFactory::create<double>({1.f, 2.f, 3.f, 4.f});
    x.linspace(1);
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f,1.f, 2.f, 3.f, 4.f,1.f, 2.f, 3.f, 4.f});
    nd4j::ops::reduce_norm1_bp op;
    auto result = op.execute({&x, &eps}, {}, {0,1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);
    output->printIndexedBuffer("Result is");
    exp.printIndexedBuffer("Expect is");
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Norm1_BP_02) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto eps = NDArrayFactory::create<double>({1.f, 2.f, 3.f, 4.f});
    x.linspace(1);
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f,1.f, 2.f, 3.f, 4.f,1.f, 2.f, 3.f, 4.f});
    auto axes = NDArrayFactory::create<int>({0,1});
    nd4j::ops::reduce_norm1_bp op;
    auto result = op.execute({&x, &eps, &axes}, {}, {}, {false});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);
    output->printIndexedBuffer("Result is");
    exp.printIndexedBuffer("Expect is");
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Norm1_BP_3) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto eps = NDArrayFactory::create<double>('c', {1, 1, 4}, {1.f, 2.f, 3.f, 4.f});
    x.linspace(1);
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f,1.f, 2.f, 3.f, 4.f,1.f, 2.f, 3.f, 4.f});
    nd4j::ops::reduce_norm1_bp op;
    auto result = op.execute({&x, &eps}, {1.f}, {0,1});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Norm2_BP_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto eps = NDArrayFactory::create<double>('c', {4}, {31.7175f, 33.823071f, 35.97221f, 38.15757f});
    x.linspace(1);

    nd4j::ops::reduce_norm2_bp op;
    auto result = op.execute({&x, &eps}, {}, {0,1});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(x.isSameShape(output));
    ASSERT_TRUE(x.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Norm2_BP_2) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto eps = NDArrayFactory::create<double>('c', {1, 1, 4}, {31.7175f, 33.823071f, 35.97221f, 38.15757f});
    x.linspace(1);

    nd4j::ops::reduce_norm2_bp op;
    auto result = op.execute({&x, &eps}, {1.f}, {0,1});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(x.isSameShape(output));
    ASSERT_TRUE(x.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Norm2_BP_02) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto eps = NDArrayFactory::create<double>('c', {1, 1, 4}, {31.7175f, 33.823071f, 35.97221f, 38.15757f});
    auto axes = NDArrayFactory::create<int>({0, 1});
    x.linspace(1);

    nd4j::ops::reduce_norm2_bp op;
    auto result = op.execute({&x, &eps, &axes}, {}, {}, {true});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(x.isSameShape(output));
    ASSERT_TRUE(x.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Norm2_BP_3) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto eps = NDArrayFactory::create<double>('c', {3}, {29.597298f, 39.344631f, 49.759422f});
    x.linspace(1);

    nd4j::ops::reduce_norm2_bp op;
    auto result = op.execute({&x, &eps}, {}, {0, 2});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(x.isSameShape(output));
    ASSERT_TRUE(x.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Norm2_BP_4) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto eps = NDArrayFactory::create<double>('c', {1,3,1}, {29.597298f, 39.344631f, 49.759422f});
    x.linspace(1);

    nd4j::ops::reduce_norm2_bp op;
    auto result = op.execute({&x, &eps}, {1.f}, {0, 2});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(x.isSameShape(output));
    ASSERT_TRUE(x.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_SquaredNorm_BP_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto eps = NDArrayFactory::create<double>('c', {4}, {1.f, 2.f, 3.f, 4.f});
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4}, { 2.f,  8.f, 18.f, 32.f,
                                        10.f, 24.f, 42.f, 64.f,
                                        18.f, 40.f, 66.f, 96.f,
                                            26.f, 56.f, 90.f, 128.f,
                                            34.f, 72.f, 114.f, 160.f,
                                            42.f, 88.f, 138.f, 192.f});
    x.linspace(1);

    nd4j::ops::reduce_sqnorm_bp op;
    auto result = op.execute({&x, &eps}, {}, {0,1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_SquaredNorm_BP_01) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto eps = NDArrayFactory::create<double>('c', {4}, {1.f, 2.f, 3.f, 4.f});
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4}, { 2.f,  8.f, 18.f, 32.f,
                                                                10.f, 24.f, 42.f, 64.f,
                                                                18.f, 40.f, 66.f, 96.f,
                                                                26.f, 56.f, 90.f, 128.f,
                                                                34.f, 72.f, 114.f, 160.f,
                                                                42.f, 88.f, 138.f, 192.f});
    auto axes = NDArrayFactory::create<int>({0, 1});
    x.linspace(1);

    nd4j::ops::reduce_sqnorm_bp op;
    auto result = op.execute({&x, &eps, &axes}, {}, {}, {false});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_NormMax_BP_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto eps = NDArrayFactory::create<double>('c', {4}, {1.f, 2.f, 3.f, 4.f});
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4});
    x.linspace(1);
    exp.p(20, 1.f);
    exp.p(21, 2.f);
    exp.p(22, 3.f);
    exp.p(23, 4.f);

    nd4j::ops::reduce_norm_max_bp op;
    auto result = op.execute({&x, &eps}, {}, {0,1});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_NormMax_BP_2) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto eps = NDArrayFactory::create<double>('c', {1,1,4}, {1.f, 2.f, 3.f, 4.f});
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4});
    x.linspace(1);
    exp.p(20, 1.f);
    exp.p(21, 2.f);
    exp.p(22, 3.f);
    exp.p(23, 4.f);

    nd4j::ops::reduce_norm_max_bp op;
    auto result = op.execute({&x, &eps}, {1.f}, {0,1});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_NormMax_BP_02) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto eps = NDArrayFactory::create<double>('c', {1,1,4}, {1.f, 2.f, 3.f, 4.f});
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto axes = NDArrayFactory::create<int>({0,1});
    x.linspace(1);
    exp.p(20, 1.f);
    exp.p(21, 2.f);
    exp.p(22, 3.f);
    exp.p(23, 4.f);

    nd4j::ops::reduce_norm_max_bp op;
    auto result = op.execute({&x, &eps, &axes}, {}, {}, {true});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}
////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_NormMax_BP_3) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto eps = NDArrayFactory::create<double>('c', {3}, {1.f, 2.f, 3.f});
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4});
    x.linspace(1);

    exp.p(15, 1.f);
    exp.p(19, 2.f);
    exp.p(23, 3.f);

    nd4j::ops::reduce_norm_max_bp op;
    auto result = op.execute({&x, &eps}, {}, {0,2});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_NormMax_BP_4) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto eps = NDArrayFactory::create<double>('c', {1, 3, 1}, {1.f, 2.f, 3.f});
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4});
    x.linspace(1);
    exp.p(15, 1.f);
    exp.p(19, 2.f);
    exp.p(23, 3.f);
    nd4j::ops::reduce_norm_max_bp op;
    auto result = op.execute({&x, &eps}, {1.f}, {0,2});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_NormMax_BP_5) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto eps = NDArrayFactory::create<double>(1.f);
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4});
    x.linspace(1);
    exp.p(23, 1.f);
    nd4j::ops::reduce_norm_max_bp op;
    auto result = op.execute({&x, &eps}, {}, {});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_NormMax_BP_6) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto eps = NDArrayFactory::create<double>(1.f);
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4});
    x.linspace(1);
    exp.p(23, 1.f);

    nd4j::ops::reduce_norm_max_bp op;
    auto result = op.execute({&x, &eps}, {}, {0, 1, 2});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_NormMax_BP_7) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto eps = NDArrayFactory::create<double>('c', {1, 1, 1}, {1.f});
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4});
    x.linspace(1);
    exp.p(23, 1.f);
    nd4j::ops::reduce_norm_max_bp op;
    auto result = op.execute({&x, &eps}, {1.f}, {});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Dot_BP_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto y = NDArrayFactory::create<double>('c', {2, 3, 4});
    NDArray* z; // = NDArrayFactory::create<double>('c', {4});
    auto eps = NDArrayFactory::create<double>(1.f);
//    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4});
    x.linspace(1);
    y.linspace(2);


    nd4j::ops::reduce_dot_bp op;
    auto result = op.execute({&x, &y, &eps}, {}, {});
    auto output = result->at(0);
    auto outputX = result->at(1);
    //tput->printIndexedBuffer("Result is");

//    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(x.equalsTo(outputX));
    ASSERT_TRUE(y.equalsTo(output));

    delete result;
//    delete z;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Dot_BP_2) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto y = NDArrayFactory::create<double>('c', {2, 3, 4});
//    auto z; // = NDArrayFactory::create<double>('c', {4});
    auto eps = NDArrayFactory::create<double>('c', {2, 4});
    auto expX = NDArrayFactory::create<double>('c', {2, 3, 4}, {2.f, 4.f, 6.f, 8.f, 2.f, 4.f, 6.f, 8.f, 2.f, 4.f, 6.f, 8.f,
                                        10.f, 12.f, 14.f, 16.f, 10.f, 12.f, 14.f, 16.f, 10.f, 12.f, 14.f, 16.f
                                        });
    auto expY = NDArrayFactory::create<double>('c', {2, 3, 4}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f,
                                         5.f, 6.f, 7.f, 8.f, 5.f, 6.f, 7.f, 8.f, 5.f, 6.f, 7.f, 8.f
    });
    x.assign(1.f);
    eps.linspace(1);
    y.assign(2.f);
    nd4j::ops::reduce_dot_bp op;
    auto result = op.execute({&x, &y, &eps}, {}, {1});
    ASSERT_EQ(result->status(), ND4J_STATUS_OK);
    ASSERT_EQ(result->size(), 2);
    auto outputX = result->at(0);
    auto outputY = result->at(1);
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(expX.equalsTo(outputX));
    ASSERT_TRUE(expY.equalsTo(outputY));

    delete result;
//    delete z;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Dot_BP_02) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto y = NDArrayFactory::create<double>('c', {2, 3, 4});
//    auto z; // = NDArrayFactory::create<double>('c', {4});
    auto eps = NDArrayFactory::create<double>('c', {2, 4});
    auto expX = NDArrayFactory::create<double>('c', {2, 3, 4}, {2.f, 4.f, 6.f, 8.f, 2.f, 4.f, 6.f, 8.f, 2.f, 4.f, 6.f, 8.f,
                                                                10.f, 12.f, 14.f, 16.f, 10.f, 12.f, 14.f, 16.f, 10.f, 12.f, 14.f, 16.f
    });
    auto expY = NDArrayFactory::create<double>('c', {2, 3, 4}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f,
                                                                5.f, 6.f, 7.f, 8.f, 5.f, 6.f, 7.f, 8.f, 5.f, 6.f, 7.f, 8.f
    });
    auto axis = NDArrayFactory::create<int>('c', {1}, {1});
    x.assign(1.f);
    eps.linspace(1);
    y.assign(2.f);
    nd4j::ops::reduce_dot_bp op;
    auto result = op.execute({&x, &y, &eps, &axis}, {}, {}, {false});
    ASSERT_EQ(result->status(), ND4J_STATUS_OK);
    ASSERT_EQ(result->size(), 2);
    auto outputX = result->at(0);
    auto outputY = result->at(1);
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(expX.equalsTo(outputX));
    ASSERT_TRUE(expY.equalsTo(outputY));

    delete result;
//    delete z;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_Dot_BP_3) {

    auto x = NDArrayFactory::create<double>('c', {3, 4});
    auto y = NDArrayFactory::create<double>('c', {3, 4});
    auto eps = NDArrayFactory::create<double>('c', {3});
    auto expX = NDArrayFactory::create<double>('c', {3, 4}, {2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f, 6.f, 6.f, 6.f, 6.f});
    auto expY = NDArrayFactory::create<double>('c', {3, 4}, {1.f, 2.f, 3.f, 4.f, 10.f, 12.f, 14.f, 16.f, 27.f, 30.f, 33.f, 36.f});
    x.linspace(1);
    eps.linspace(1);
    y.assign(2.f);

    nd4j::ops::reduce_dot_bp op;
    auto result = op.execute({&x,&y, &eps}, {}, {1});
    auto outputX = result->at(0);
    auto outputY = result->at(1);

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_TRUE(expX.equalsTo(outputX));
    ASSERT_TRUE(expY.equalsTo(outputY));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_CumSum_BP_1) {

    auto x = NDArrayFactory::create<double>('c', {3, 4});
//    auto y = NDArrayFactory::create<double>('c', {3, 4});
//    auto z; // = NDArrayFactory::create<double>('c', {4});
    auto eps = NDArrayFactory::create<double>('c', {3, 4});
    auto exp = NDArrayFactory::create<double>('c', {3, 4}, {12.f, 11.f, 10.f, 9.f, 8.f, 7.f,
                                      6.f,  5.f,  4.f, 3.f, 2.f, 1.f});
    x.linspace(1);
    eps.assign(1.f);

//    z = x.applyReduce3<simdOps::Dot<float>>(&y, {0}, nullptr);
    nd4j::ops::cumsum_bp op;
    auto result = op.execute({&x, &eps}, {}, {0,0});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
//    output->printShapeInfo("Result shape is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
//    delete z;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_CumSum_BP_2) {
    auto x = NDArrayFactory::create<double>('c', {3, 4});
//    auto y = NDArrayFactory::create<double>('c', {3, 4});
//    auto z; // = NDArrayFactory::create<double>('c', {4});
    auto eps = NDArrayFactory::create<double>('c', {3, 4});
    auto exp = NDArrayFactory::create<double>('c', {3, 4}, { 11.f, 10.f, 9.f, 8.f, 7.f, 6.f,
                                      5.f,  4.f, 3.f, 2.f, 1.f, 0.f});
    x.linspace(1);
//    exp.linspace(1);
    eps.assign(1.f);

//    z = x.applyReduce3<simdOps::Dot<float>>(&y, {0}, nullptr);
    nd4j::ops::cumsum_bp op;
    auto result = op.execute({&x, &eps}, {}, {1,0});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
//    output->printShapeInfo("Result shape is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests18, Test_Reduce_CumSum_BP_3) {
    auto x = NDArrayFactory::create<double>('c', {3, 4});
//    auto y = NDArrayFactory::create<double>('c', {3, 4});
//    auto z; // = NDArrayFactory::create<double>('c', {4});
    auto eps = NDArrayFactory::create<double>('c', {3, 4});
    auto exp = NDArrayFactory::create<double>('c', {3, 4});

    x.linspace(1);
    exp.linspace(0);
    eps.assign(1.f);

//    z = x.applyReduce3<simdOps::Dot<float>>(&y, {0}, nullptr);
    nd4j::ops::cumsum_bp op;
    auto result = op.execute({&x, &eps}, {}, {1,1});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
//    output->printShapeInfo("Result shape is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

//    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;

}

