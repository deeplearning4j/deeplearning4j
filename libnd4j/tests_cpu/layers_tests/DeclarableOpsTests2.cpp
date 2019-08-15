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

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <helpers/helper_hash.h>
#include <NDArray.h>
#include <array/NDArrayList.h>


using namespace nd4j;
using namespace nd4j::graph;

class DeclarableOpsTests2 : public testing::Test {
public:

    DeclarableOpsTests2() {
        printf("\n");
    }
};

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, gather_1) {

    NDArray input('c', {2,3,4}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}, nd4j::DataType::FLOAT32);
    NDArray indices('c', {1,6},   {0,1, 2,2, 1,2}, nd4j::DataType::INT32);
    NDArray expected('c', {2,1,6,4}, {1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12, 9,10,11,12, 5, 6, 7, 8, 9,10,11,12, 13,14,15,16, 17,18,19,20, 21,22,23,24, 21,22,23,24, 17,18,19,20, 21,22,23,24}, nd4j::DataType::FLOAT32);

    nd4j::ops::gather op;

    auto result = op.execute({&input, &indices}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto* output = result->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete result;
}


TEST_F(DeclarableOpsTests2, gather_2) {

    NDArray input('c', {2,3,4}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24});
    //auto indices ('c', {1,6},   {0,1, 2,2, 1,2});
    NDArray expected('c', {2,6,4}, {1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12, 9,10,11,12, 5, 6, 7, 8, 9,10,11,12, 13,14,15,16, 17,18,19,20, 21,22,23,24, 21,22,23,24, 17,18,19,20, 21,22,23,24});

    nd4j::ops::gather op;

    auto result = op.execute({&input}, {}, {1, 0,1, 2,2, 1,2});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto* output = result->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete result;
}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, gather_3) {

    NDArray input   ('c', {2,3,4}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24});
    NDArray indices ('c', {1,1},   {2}, nd4j::DataType::INT32);
    NDArray expected('c', {2,1,1,4}, {9,10,11,12,21,22,23,24});

    nd4j::ops::gather op;

    auto result = op.execute({&input, &indices}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto* output = result->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete result;
}

TEST_F(DeclarableOpsTests2, gather_4) {

    NDArray input('c', {2,3,4}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24});
    //auto indices ('c', {1,1},   {2});
    NDArray expected('c', {2,4}, {9,10,11,12,21,22,23,24});

    nd4j::ops::gather op;

    auto result = op.execute({&input}, {}, {1, 2});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto* output = result->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, gather_5) {

    NDArray input   ('c', {2,3,4},   {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24});
    NDArray indices ('c', {2,3},     {0, 1, 2, 2, 1,2}, nd4j::DataType::INT32);
    NDArray expected('c', {2,2,3,4}, {1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,  9,10,11,12, 5, 6, 7, 8, 9,10,11,12, 13,14,15,16,17,18,19,20,21,22,23,24, 21,22,23,24,17,18,19,20,21,22,23,24});

    nd4j::ops::gather op;

    auto result = op.execute({&input, &indices}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto* output = result->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete result;
}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, gather_6) {

    NDArray input   ('c', {3,3,4},   {1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12, 13,14,15,16,17,18,19,20,21,22,23,24, 25,26,27,28,29,30,31,32,33,34,35,36});
    NDArray indices ('c', {2,3},     {0, 1, 2, 2, 1,2}, nd4j::DataType::INT32);
    NDArray expected('c', {2,3,3,4}, {1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12, 13,14,15,16,17,18,19,20,21,22,23,24, 25,26,27,28,29,30,31,32,33,34,35,36, 25,26,27,28,29,30,31,32,33,34,35,36, 13,14,15,16,17,18,19,20,21,22,23,24, 25,26,27,28,29,30,31,32,33,34,35,36});

    nd4j::ops::gather op;

    auto result = op.execute({&input, &indices}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto* output = result->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete result;
}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, gather_7) {

    NDArray input   ('c', {2,3,4},   {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24});
    NDArray indices ('c', {2,3},     {0, 1, 2, 2, 1,2}, nd4j::DataType::INT64);
    NDArray expected('c', {2,3,2,3}, {1, 2, 3, 3, 2, 3, 5, 6, 7, 7, 6, 7, 9,10,11,11,10,11, 13,14,15,15,14,15, 17,18,19,19,18,19, 21,22,23,23,22,23});

    nd4j::ops::gather op;

    auto result = op.execute({&input, &indices}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto* output = result->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete result;
}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, gather_8) {

    NDArray input('c', {3,5}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}, nd4j::DataType::FLOAT32);
    NDArray indices('c', {1}, {2}, nd4j::DataType::INT32);
    NDArray expected('c', {1,5}, {11, 12, 13, 14, 15.}, nd4j::DataType::FLOAT32);

    nd4j::ops::gather op;

    auto result = op.execute({&input, &indices}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto* output = result->at(0);
    // output->printShapeInfo();
    // output->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, gather_9) {
    NDArray x('c', {2, 4, 3, 2}, nd4j::DataType::FLOAT32);
    NDArray indices('c', {2}, {1, 0}, nd4j::DataType::INT32);

    nd4j::ops::gather op;
    auto result = op.execute({&x, &indices}, {}, {-2});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, gather_10) {
    NDArray x('c', {2, 2}, {1, 2, 3, 4});
    NDArray e('c', {2, 2}, {3, 4, 1, 2});

    nd4j::ops::gather op;
    auto result = op.execute({&x}, {}, {0, 1, 0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(e.isSameShape(z));
    ASSERT_TRUE(e.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, gather_11) {

    NDArray x('c', {2, 2}, {1, 2, 3, 4});
    NDArray indices('c', {2}, {1, 0}, nd4j::DataType::INT64);
    NDArray e('c', {2, 2}, {3, 4, 1, 2});

    nd4j::ops::gather op;
    auto result = op.execute({&x, &indices}, {}, {0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(e.isSameShape(z));
    ASSERT_TRUE(e.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, gather_12) {

    NDArray input('c', {4}, {2.f, 3.f, 4.f, 5.f});
    NDArray indices('c', {2}, {0, 2}, nd4j::DataType::INT32);
    NDArray exp('c', {2}, {2.f, 4.f});

    nd4j::ops::gather op;
    auto result = op.execute({&input, &indices}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, gather_13) {

    NDArray input   ('c', {2,3,4,5}, nd4j::DataType::DOUBLE);
    NDArray indices ('c', {2,3,4}, {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3}, nd4j::DataType::INT32);
    NDArray expected('c', {2,3,  2,3,4,  5}, {0,  1,  2,  3,  4, 5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0,  1,  2,  3,  4, 5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0,  1,  2,  3,  4, 5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                            0,  1,  2,  3,  4, 5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0,  1,  2,  3,  4, 5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0,  1,  2,  3,  4, 5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                            40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                            40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                            60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                            60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                            80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                            80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                            100,101,102,103,104, 105,106,107,108,109, 110,111,112,113,114, 115,116,117,118,119, 100,101,102,103,104, 105,106,107,108,109, 110,111,112,113,114, 115,116,117,118,119, 100,101,102,103,104, 105,106,107,108,109, 110,111,112,113,114, 115,116,117,118,119,
                            100,101,102,103,104, 105,106,107,108,109, 110,111,112,113,114, 115,116,117,118,119, 100,101,102,103,104, 105,106,107,108,109, 110,111,112,113,114, 115,116,117,118,119, 100,101,102,103,104, 105,106,107,108,109, 110,111,112,113,114, 115,116,117,118,119});

    input.linspace(0);

    nd4j::ops::gather op;

    auto result = op.execute({&input, &indices}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto* output = result->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete result;
}

TEST_F(DeclarableOpsTests2, BroadcastGradientArgs_1) {

    NDArray input   ('c', {3,3,4},   {1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12, 13,14,15,16,17,18,19,20,21,22,23,24, 25,26,27,28,29,30,31,32,33,34,35,36}, nd4j::DataType::INT32);
    NDArray indices ('c', {2,3},     {0, 1, 2, 2, 1,2}, nd4j::DataType::INT32);

    nd4j::ops::broadcastgradientargs op;

    auto result = op.execute({&input, &indices}, {}, {});

    ASSERT_EQ(ND4J_STATUS_KERNEL_FAILURE, result->status());

    delete result;
}

TEST_F(DeclarableOpsTests2, NLP_Cbow_Test_1) {
    auto exp0 = NDArrayFactory::create<double>('c', {1, 10});
    auto exp1 = NDArrayFactory::create<double>('c', {1, 10});
    auto exp2 = NDArrayFactory::create<double>('c', {1, 10});

    exp0.assign(0.0095);
    exp1.assign(0.019875);
    exp2.assign(0.02);

    auto target = NDArrayFactory::create<int>(0);
    auto ngStarter = NDArrayFactory::empty<int>();
    auto context = NDArrayFactory::create<int>('c', {3}, {0, 1, 2});
    auto locked = NDArrayFactory::create<int>('c', {3});
    auto indices = NDArrayFactory::create<int>('c', {2}, {4, 5});
    auto codes = NDArrayFactory::create<int8_t>('c', {2}, {1, 1});
    auto syn0 = NDArrayFactory::create<double>('c', {100, 10});
    auto syn1 = NDArrayFactory::create<double>('c', {100, 10});
    auto syn1Neg = NDArrayFactory::empty<double>();
    auto expTable = NDArrayFactory::create<double>('c', {10000});
    auto negTable = NDArrayFactory::empty<double>();
    auto numWords = NDArrayFactory::create<int>('c', {1}, {1});

    syn0.assign(0.01);
    syn1.assign(0.02);
    expTable.assign(0.5);

    auto alpha = NDArrayFactory::create<double>(0.025);
    auto randomValue = NDArrayFactory::create<Nd4jLong>(2L);
    auto inferenceVector = NDArrayFactory::empty<double>();

    nd4j::ops::cbow op;
    auto result = op.execute({&target, &ngStarter, &context, &indices, &codes, &syn0, &syn1, &syn1Neg, &expTable, &negTable, &alpha, &randomValue, &numWords, &locked, &inferenceVector}, {}, {}, {true}, true);
    ASSERT_EQ(Status::OK(), result->status());

    auto row_s0_0 = syn0({0,1, 0,0}, true);
    auto row_s0_1 = syn0({1,2, 0,0}, true);
    auto row_s0_2 = syn0({2,3, 0,0}, true);

    auto row_s1_4 = syn1({4,5, 0,0}, true);
    auto row_s1_5 = syn1({5,6, 0,0}, true);
    auto row_s1_6 = syn1({6,7, 0,0}, true);

    ASSERT_EQ(exp0, row_s0_0);
    ASSERT_EQ(exp0, row_s0_1);
    ASSERT_EQ(exp0, row_s0_2);

    ASSERT_EQ(exp1, row_s1_4);
    ASSERT_EQ(exp1, row_s1_5);
    ASSERT_EQ(exp2, row_s1_6);

    delete result;
}

TEST_F(DeclarableOpsTests2, Test_Concat_3D_1) {
    auto x0 = NDArrayFactory::create<double>('c', {1, 100, 150});
    auto x1 = NDArrayFactory::create<double>('c', {1, 100, 150});
    auto x2 = NDArrayFactory::create<double>('c', {1, 100, 150});
    auto x3 = NDArrayFactory::create<double>('c', {1, 100, 150});

    x0.assign(1.0);
    x1.assign(2.0);
    x2.assign(3.0);
    x3.assign(4.0);

    nd4j::ops::concat op;
    auto result = op.execute({&x0, &x1, &x2, &x3}, {}, {0}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    Nd4jLong numOfTads= ShapeUtils::getNumOfSubArrs(z->getShapeInfo(), {0});
    ASSERT_TRUE(4 == numOfTads);

    for (int e = 0; e < numOfTads; e++) {
        NDArray tad  = (*z)(e, {0});
        auto mean = tad.meanNumber().e<double>(0);
        ASSERT_NEAR((double) e+1, mean, 1e-5);
    }

    delete result;
}

TEST_F(DeclarableOpsTests2, YetAnotherMatmulTest_1) {
    auto A = NDArrayFactory::create<float>('c', {3, 3});
    auto B = NDArrayFactory::create<float>('c', {3, 1});
    auto exp = NDArrayFactory::create<float>('c', {3, 1}, {14.00,  32.00,  50.00});

    A.linspace(1);
    B.linspace(1);

    nd4j::ops::matmul op;

    auto result = op.execute({&A, &B}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests2, Test_Squeeze_1) {
    auto x = NDArrayFactory::create<float>('c', {2, 1, 3, 1, 1, 1, 4});
    x.linspace(1);
    auto exp = x.reshape('c', {2, 3, 4});

    nd4j::ops::squeeze op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests2, Test_Squeeze_2) {
    auto x = NDArrayFactory::create<float>('c', {2, 3, 4});
    x.linspace(1);
    auto exp = x.dup();

    nd4j::ops::squeeze op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp->isSameShape(z));
    ASSERT_TRUE(exp->equalsTo(z));

    delete result;
    delete exp;
}

TEST_F(DeclarableOpsTests2, Test_FloorMod_1) {
    auto x = NDArrayFactory::create<float>('c', {1, 3}, {2.0, 6.0, -3.0});
    auto y = NDArrayFactory::create<float>('c', {1, 3}, {-3.0, 2.0, -2.0});
    auto exp = NDArrayFactory::create<float>('c', {1, 3}, {-1.,  0., -1.,});

    nd4j::ops::floormod op;

    auto result = op.execute({&x, &y}, {}, {});

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests2, Test_FloorDiv_1) {
    auto x = NDArrayFactory::create<float>('c', {1, 3}, {3.0, 6.0, -3.0});
    auto y = NDArrayFactory::create<float>('c', {1, 3}, {-2.0, 2.0, -2.0});
    auto exp = NDArrayFactory::create<float>('c', {1, 3}, {-2.,  3., 1.,});

    nd4j::ops::floordiv op;

    auto result = op.execute({&x, &y}, {}, {});

    auto z = result->at(0);
//    z->printShapeInfo("FloorDiv1 shape");
//    z->printIndexedBuffer("FloorDiv1");
    ASSERT_TRUE(exp.isSameShape(z));

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests2, Test_FloorDiv_2) {
    auto x = NDArrayFactory::create<float>('c', {1, 3}, {3.0, 6.0, -3.0});
    auto y = NDArrayFactory::create<float>('c', {1, 3}, {-2.0, 2.0, -2.0});
    auto eps = NDArrayFactory::create<float>('c', {1, 3}, {1, 2, 3});

    auto exp1 = NDArrayFactory::create<float>('c', {1, 3}, {0.f,  0.f, 0.f});
    auto exp2 = NDArrayFactory::create<float>('c', {1, 3}, {0.f, 0.f, 0.f});

    nd4j::ops::floordiv_bp op;

    auto result = op.execute({&x, &y, &eps}, {}, {});
    ASSERT_EQ(result->status(), Status::OK());
    auto z1 = result->at(0);
    auto z2 = result->at(1);
//    z->printShapeInfo("FloorDiv1 shape");
//    z1->printIndexedBuffer("FloorDiv2_1");
//    z2->printIndexedBuffer("FloorDiv2_2");

    ASSERT_TRUE(exp1.equalsTo(z1));
    ASSERT_TRUE(exp2.equalsTo(z2));

    delete result;
}

TEST_F(DeclarableOpsTests2, Test_CRelu_1) {
    auto x = NDArrayFactory::create<float>('c', {2, 2}, {1.0, 2.0, 3.0, 4.0});
    auto exp = NDArrayFactory::create<float>('c', {2, 4}, {1.0, 2.0, 0, 0, 3.0, 4.0, 0, 0});

    nd4j::ops::crelu op;

    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests2, Test_CRelu_BP_2) {
    auto x = NDArrayFactory::create<float>('c', {2, 2}, {1.0, 2.0, -3.0, 4.0});
    auto eps = NDArrayFactory::create<float>('c', {2, 4}, {1.0, 2.0, 4, 3, 3.0, 4.0, 2, 1});
    auto exp = NDArrayFactory::create<float>('c', {2, 2}, {1, 2, -2, 4});

    nd4j::ops::crelu_bp op;
    auto result = op.execute({&x, &eps}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(1, result->size());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests2, Test_Concat_BP_1) {
    auto x = NDArrayFactory::create<float>('c', {2, 2});
    auto y = NDArrayFactory::create<float>('c', {2, 2});
    auto eps = NDArrayFactory::create<float>('c', {2, 4}, {1.0, 2.0, 0, 1, 3.0, 4.0, 0, 1});
    auto expEX = NDArrayFactory::create<float>('c', {2, 2}, {1, 2, 3, 4});
    auto expEY = NDArrayFactory::create<float>('c', {2, 2}, {0, 1, 0, 1});

    nd4j::ops::concat_bp op;
    auto result = op.execute({&x, &y, &eps}, {}, {-1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(2, result->size());

    auto epsX = result->at(0);
    auto epsY = result->at(1);

    ASSERT_TRUE(expEX.isSameShape(epsX));
    ASSERT_TRUE(expEX.equalsTo(epsX));

    ASSERT_TRUE(expEY.isSameShape(epsY));
    ASSERT_TRUE(expEY.equalsTo(epsY));

    delete result;
}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, TestTensorDot5) {

    auto x = NDArrayFactory::create<float>('c', {2,3,4}, {1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15});
    auto y = NDArrayFactory::create<float>('c', {2,4,3}, {2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16});
    auto expected = NDArrayFactory::create<float>('c', {2,4,2,4}, {44,110,160, 66,132, 38, 88,154, 68,170,224,102,204, 82,136,238, 92,230,288,138,276,126,184,322, 116,290,352,174,348,170,232,406, 76,190,160,114,228,182,152,266, 100,250,224,150,300,226,200,350, 124,310,288,186,372,270,248,434, 148,370,352,222,444,314,296,518});

    nd4j::ops::tensormmul op;
    auto results = op.execute({&x, &y}, {}, {1,1,1,2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, TestTensorDot6) {

    auto x = NDArrayFactory::create<float>('c', {2,3,4}, {1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15});
    auto y = NDArrayFactory::create<float>('f', {2,4,3}, {2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16});
    auto expected = NDArrayFactory::create<float>('c', {2,4,2,4}, {22, 66,110,154, 44, 88,132,176, 34,102,170,238, 68,136,204,272, 46,138,230,322, 92,184,276,368, 58,174,290,406,116,232,348,464, 38,114,190,266, 76,152,228,304, 50,150,250,350,100,200,300,400, 62,186,310,434,124,248,372,496, 74,222,370,518,148,296,444,592});

    nd4j::ops::tensormmul op;
    auto results = op.execute({&x, &y}, {}, {1,1,1,2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, TestTensorDot7) {

    auto x = NDArrayFactory::create<float>('f', {2,3,4}, {1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15});
    auto y = NDArrayFactory::create<float>('c', {2,4,3}, {2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16});
    auto expected = NDArrayFactory::create<float>('c', {2,4,2,4}, {76,166,112,106,196, 62,136,226, 60,174,208, 98,212,230,136,250, 76,214,336,122,260,174,168,306, 124,286,240,178,340,150,232,394, 100,226,176,142,268,106,184,310, 84,234,272,134,284,274,184,334, 100,274,400,158,332,218,216,390, 148,346,304,214,412,194,280,478});

    nd4j::ops::tensormmul op;
    auto results = op.execute({&x, &y}, {}, {1,1,1,2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, TestTensorDot8) {

    auto x = NDArrayFactory::create<float>('f', {2,3,4}, {1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15});
    auto y = NDArrayFactory::create<float>('f', {2,4,3}, {2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16});
    auto expected = NDArrayFactory::create<float>('c', {2,4,2,4}, {30, 90,150,210, 60,120,180,240, 38,114,190,266, 76,152,228,304, 46,138,230,322, 92,184,276,368, 54,162,270,378,108,216,324,432, 42,126,210,294, 84,168,252,336, 50,150,250,350,100,200,300,400, 58,174,290,406,116,232,348,464, 66,198,330,462,132,264,396,528});

    nd4j::ops::tensormmul op;
    auto results = op.execute({&x, &y}, {}, {1,1,1,2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, TestTensorDot9) {

    // NDArray z('f',{2,2,3}, nd4j::DataType::DOUBLE);
    // z.linspace(1);
    // z.printShapeInfo();
    // z.printIndexedBuffer();
    // z.reshapei('c', {4,3});
    // z.printShapeInfo();
    // z.printIndexedBuffer();

    auto x = NDArrayFactory::create<float>('f', {2,3,4}, {1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15});
    auto y = NDArrayFactory::create<float>('f', {2,4,3}, {2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16});
    auto expected = NDArrayFactory::create<float>('c', {3,4,4,3}, {14, 14, 14, 30, 30, 30, 46, 46, 46, 62, 62, 62, 86, 86, 86,198,198,198,310,310,310,422,422,422, 62, 62, 62,142,142,142,222,222,222,302,302,302, 38, 38, 38, 86, 86, 86,134,134,134,182,182,182, 38, 38, 38, 86, 86, 86,134,134,134,182,182,182, 14, 14, 14, 30, 30, 30, 46, 46, 46, 62, 62, 62, 86, 86, 86,198,198,198,310,310,310,422,422,422, 62, 62, 62,142,142,142,222,222,222,302,302,302, 62, 62, 62,142,142,142,222,222,222,302,302,302, 38, 38, 38, 86, 86, 86,134,134,134,182,182,182, 14, 14, 14, 30, 30, 30, 46, 46, 46, 62, 62, 62, 86, 86, 86,198,198,198,310,310,310,422,422,422});

    nd4j::ops::tensormmul op;
    auto results = op.execute({&x, &y}, {}, {1,0,1,0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, TestTensorDot10) {

    auto x = NDArrayFactory::create<float>('f', {2,3,4}, {1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15});
    auto y = NDArrayFactory::create<float>('f', {2,4,3}, {2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16});
    auto expected = NDArrayFactory::create<float>('c', {4,4}, {114,258,402,546, 138,314,490,666, 162,370,578,786, 186,426,666,906});

    nd4j::ops::tensormmul op;
    auto results = op.execute({&x, &y}, {}, {2,0,1, 2,0,2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, TestTensorDot11) {

    auto x = NDArrayFactory::create<float>('c', {2,3,4}, {1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15});
    auto y = NDArrayFactory::create<float>('f', {2,4,3}, {2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16});
    auto expected = NDArrayFactory::create<float>('c', {4,4}, {98,218,338,458, 134,302,470,638, 170,386,602,818, 206,470,734,998});

    nd4j::ops::tensormmul op;
    auto results = op.execute({&x, &y}, {}, {2,0,1, 2,0,2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, TestTensorDot12) {

    auto x = NDArrayFactory::create<float>('c', {2,3,4}, {1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15});
    auto y = NDArrayFactory::create<float>('c', {2,4,3}, {2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16});
    auto expected = NDArrayFactory::create<float>('c', {4,4}, {272,292,312,332, 368,396,424,452, 464,500,536,572, 560,604,648,692});

    nd4j::ops::tensormmul op;
    auto results = op.execute({&x, &y}, {}, {2,0,1, 2,0,2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, TestTensorDot13) {

    auto x = NDArrayFactory::create<float>('c', {2,3,4}, {1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15});
    auto y = NDArrayFactory::create<float>('c', {4,2,3}, {2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16});
    auto expected = NDArrayFactory::create<float>('c', {3,3}, {640,560,640, 576,624,576, 640,560,640});

    nd4j::ops::tensormmul op;
    auto results = op.execute({&x, &y}, {}, {2,0,2, 2,1,0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, TestTensorDot14) {

    auto x = NDArrayFactory::create<float>('f', {2,3,4}, {1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15});
    auto y = NDArrayFactory::create<float>('c', {4,2,3}, {2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16});
    auto expected = NDArrayFactory::create<float>('c', {3,3}, {648,600,520, 648,536,648, 520,600,648});

    nd4j::ops::tensormmul op;
    auto results = op.execute({&x, &y}, {}, {2,0,2, 2,1,0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, TestTensorDot15) {

    auto x = NDArrayFactory::create<float>('f', {2,3,4}, {1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15});
    auto y = NDArrayFactory::create<float>('f', {4,2,3}, {2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16});
    auto expected = NDArrayFactory::create<float>('c', {3,3}, {624,624,624, 656,656,656, 624,624,624});

    nd4j::ops::tensormmul op;
    auto results = op.execute({&x, &y}, {}, {2,0,2, 2,1,0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absolute_difference_loss_test_1) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto predictions = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto weights = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto expected = NDArrayFactory::create<float>('c', {2,3,4,5});

    labels.linspace(1);
    predictions.linspace(2);
    weights.assign(0.5f);
    expected.assign(0.5f);

    nd4j::ops::absolute_difference_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absolute_difference_loss_test_2) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto predictions = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto weights = NDArrayFactory::create<float>('c', {1,1,4,5});
    auto expected = NDArrayFactory::create<float>('c', {2,3,4,5});

    labels.linspace(1);
    predictions.linspace(2);
    weights.assign(0.5f);
    expected.assign(0.5f);

    nd4j::ops::absolute_difference_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);
    // result->printIndexedBuffer("ADL test2");
    // expected.printIndexedBuffer("ADL expec");
    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absolute_difference_loss_test_3) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto predictions = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto weights = NDArrayFactory::create<float>('c', {1,1,1,5});
    auto expected = NDArrayFactory::create<float>('c', {2,3,4,5});

    labels.linspace(1);
    predictions.linspace(2);
    weights.assign(0.5f);
    expected.assign(0.5f);

    nd4j::ops::absolute_difference_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absolute_difference_loss_test_4) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto predictions = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto weights = NDArrayFactory::create<float>('c', {2,1,1,5});
    auto expected = NDArrayFactory::create<float>('c', {2,3,4,5});

    labels.linspace(1);
    predictions.linspace(2);
    weights.assign(0.5f);
    expected.assign(0.5f);

    nd4j::ops::absolute_difference_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absolute_difference_loss_test_5) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto predictions = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto weights = NDArrayFactory::create<float>('c', {1,1});
    auto expected = NDArrayFactory::create<float>('c', {2,3,4,5});

    labels.linspace(1);
    predictions.linspace(2);
    weights.assign(0.5f);
    expected.assign(0.5f);

    nd4j::ops::absolute_difference_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absolute_difference_loss_test_6) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto predictions = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto weights = NDArrayFactory::create<float>('c', {1,1});
    auto expected = NDArrayFactory::create<float>('c', {2,3,4,5});

    labels.linspace(1);
    predictions.linspace(2);
    weights.assign(0.f);
    expected.assign(0.f);

    nd4j::ops::absolute_difference_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absolute_difference_loss_test_7) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto predictions = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto weights = NDArrayFactory::create<float>('c', {2,3,4,5});

    labels.linspace(1);
    predictions.linspace(2);
    weights.assign(0.5f);

    nd4j::ops::absolute_difference_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE(result->e<float>(0) == 60.f);

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absolute_difference_loss_test_8) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto predictions = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto weights = NDArrayFactory::create<float>('c', {2,3,4,5});

    labels.linspace(1);
    predictions.linspace(2);
    weights.assign(0.f);

    nd4j::ops::absolute_difference_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE(result->e<float>(0) == 0.f);

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absolute_difference_loss_test_9) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto predictions = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto weights = NDArrayFactory::create<float>('c', {2,1,4,1});

    labels.linspace(1);
    predictions.linspace(2);
    weights.assign(0.5f);

    nd4j::ops::absolute_difference_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE(result->e<float>(0) == 60.);

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absolute_difference_loss_test_10) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto predictions = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto weights = NDArrayFactory::create<float>('c', {1,1});

    labels.linspace(1);
    predictions.linspace(2);
    weights.assign(0.5f);

    nd4j::ops::absolute_difference_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE(result->e<float>(0) == 60.f);

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absolute_difference_loss_test_11) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto predictions = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto weights = NDArrayFactory::create<float>('c', {1,1});

    labels.linspace(1);
    predictions.linspace(2);
    weights.assign(0.5f);

    nd4j::ops::absolute_difference_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE(result->e<float>(0) == 1.f);

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absolute_difference_loss_test_12) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto predictions = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto weights = NDArrayFactory::create<float>('c', {1,1});

    labels.linspace(1);
    predictions.linspace(2);
    weights.assign(0.f);

    nd4j::ops::absolute_difference_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE(result->e<float>(0) == 0.f);

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absolute_difference_loss_test_13) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto predictions = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto weights = NDArrayFactory::create<float>('c', {2,3,4,5});

    labels.linspace(1);
    predictions.linspace(2);
    weights.assign(0.5f);

    nd4j::ops::absolute_difference_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE(result->e<float>(0) == 1.f);

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absolute_difference_loss_test_14) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto predictions = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto weights = NDArrayFactory::create<float>('c', {2,3,4,5});

    labels.linspace(1);
    predictions.linspace(2);
    weights.assign(0.5);
    weights.p(1, 0.f);
    weights.p(2, 0.f);

    nd4j::ops::absolute_difference_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE(result->e<float>(0) == 1.f);

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absolute_difference_loss_test_15) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto predictions = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto weights = NDArrayFactory::create<float>('c', {2,3,4,5});

    labels.linspace(1);
    predictions.linspace(3);
    weights.assign(0.5f);

    nd4j::ops::absolute_difference_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE(result->e<float>(0) == 2.f);

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absolute_difference_loss_test_16) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto predictions = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto weights = NDArrayFactory::create<float>('c', {2,3,4,5});

    labels.linspace(1);
    predictions.linspace(3);
    weights.assign(0.5f);
    predictions.p(0, 0.f);
    predictions.p(1, 0.f);
    predictions.p(2, 0.f);
    predictions.p(3, 0.f);

    nd4j::ops::absolute_difference_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 2.01667, 1e-5);

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absolute_difference_loss_test_17) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto predictions = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto weights = NDArrayFactory::create<float>('c', {2,3,4,5});

    labels.linspace(1);
    predictions.linspace(3);
    weights.assign(0.5f);
    predictions.p(0, 0.f);
    predictions.p(1, 0.f);
    predictions.p(2, 0.f);
    predictions.p(3, 0.f);
    labels.p(0, 0.f);
    labels.p(1, 0.f);
    labels.p(2, 0.f);
    labels.p(3, 0.f);

    nd4j::ops::absolute_difference_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<float>(0), 1.93333, 1e-5);

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absolute_difference_loss_test_18) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto predictions = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto weights = NDArrayFactory::create<float>('c', {2,1,1,5});

    labels.linspace(1);
    predictions.linspace(3);
    weights.assign(0.5f);
    predictions.p(0, 0.f);
    predictions.p(1, 0.f);
    predictions.p(2, 0.f);
    predictions.p(3, 0.);
    labels.p(0, 0.f);
    labels.p(1, 0.f);
    labels.p(2, 0.f);
    labels.p(3, 0.f);

    nd4j::ops::absolute_difference_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<float>(0), 1.93333f, 1e-5);

    delete results;

}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absolute_difference_loss_test_19) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto predictions = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto weights = NDArrayFactory::create<float>('c', {1,1});

    labels.linspace(1);
    predictions.linspace(3);
    weights.assign(0.5);

    nd4j::ops::absolute_difference_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE(result->e<double>(0) == 1.);

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absolute_difference_loss_test_20) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto predictions = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto weights = NDArrayFactory::create<float>('c', {2,3,4,5});

    labels.linspace(1);
    predictions.linspace(3);
    weights.assign(0.5);

    nd4j::ops::absolute_difference_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE(result->e<double>(0) == 1.);

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absolute_difference_loss_test_21) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto predictions = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto weights = NDArrayFactory::create<float>('c', {2,3,1,1});

    labels.linspace(1);
    predictions.linspace(3);
    weights.assign(0.5);

    nd4j::ops::absolute_difference_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE(result->e<float>(0) == 1.f);

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absolute_difference_loss_test_22) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto predictions = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto weights = NDArrayFactory::create<float>('c', {1,1});

    labels.linspace(1);
    predictions.linspace(3);
    weights.assign(0.);

    nd4j::ops::absolute_difference_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE(result->e<float>(0) == 0.);

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absolute_difference_loss_test_23) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto predictions = NDArrayFactory::create<float>('c', {2,3,4,5});
    auto weights = NDArrayFactory::create<float>('c', {2,3,4,5});

    labels.linspace(1);
    predictions.linspace(3);
    weights.assign(0.5);
    predictions.p(0, 0.);
    predictions.p(1, 0.);
    predictions.p(2, 0.);
    predictions.p(3, 0.);
    labels.p(0, 0.);
    labels.p(1, 0.);
    labels.p(2, 0.);
    labels.p(3, 0.);
    weights.p(40+0, 0.);
    weights.p(40+1, 0.);
    weights.p(40+2, 0.);
    weights.p(40+3, 0.);

    nd4j::ops::absolute_difference_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 0.965517, 1e-5);

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, cosine_distance_loss_test1) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<float>('c', {2,3,4});
    auto weights = NDArrayFactory::create<float>('c', {1,3,4});
    auto expected = NDArrayFactory::create<float>('c', {1,3,4}, {-91.5,-107.5,-125.5,-145.5, -167.5,-191.5,-217.5,-245.5, -275.5,-307.5,-341.5,-377.5});

    labels.linspace(1);
    predictions.linspace(2);
    weights.assign(0.5);

    nd4j::ops::cosine_distance_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {0,0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, cosine_distance_loss_test2) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<float>('c', {2,3,4});
    auto weights = NDArrayFactory::create<float>('c', {2,1,4});
    auto expected = NDArrayFactory::create<float>('c', {2,1,4}, {-3.25, -4., -4.75, -5.5,-12.25,-13.,-13.75,-14.5});

    labels.linspace(1);
    weights.assign(0.5);
    predictions.assign(0.5);

    nd4j::ops::cosine_distance_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {0,1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, cosine_distance_loss_test3) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<float>('c', {2,3,4});
    auto weights = NDArrayFactory::create<float>('c', {2,3,1});
    auto expected = NDArrayFactory::create<float>('c', {2,3,1}, {-2., -6.,-10.,-14.,-18.,-22.});

    labels.linspace(1);
    weights.assign(0.5);
    predictions.assign(0.5);

    nd4j::ops::cosine_distance_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {0,2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, cosine_distance_loss_test4) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<float>('c', {2,3,4});
    auto weights = NDArrayFactory::create<float>('c', {1,1});
    auto expected = NDArrayFactory::create<float>('c', {2,3,1}, {-2., -6.,-10.,-14.,-18.,-22.});

    labels.linspace(1);
    weights.assign(0.5);
    predictions.assign(0.5);

    nd4j::ops::cosine_distance_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {0,2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}


///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, cosine_distance_loss_test5) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<float>('c', {2,3,4});
    auto weights = NDArrayFactory::create<float>('c', {2,1,4});

    labels.linspace(1);
    weights.assign(0.5);
    predictions.assign(0.5);

    nd4j::ops::cosine_distance_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {1,1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE(result->e<double>(0) == -71.);

    delete results;

}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, cosine_distance_loss_test6) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<float>('c', {2,3,4});
    auto weights = NDArrayFactory::create<float>('c', {1,1});

    labels.linspace(1);
    weights.assign(0.5);
    predictions.assign(0.5);

    nd4j::ops::cosine_distance_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {1,1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE(result->e<float>(0) == -71.);

    delete results;

}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, cosine_distance_loss_test7) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<float>('c', {2,3,4});
    auto weights = NDArrayFactory::create<float>('c', {1,1,4});

    labels.linspace(1);
    weights.assign(0.5);
    predictions.assign(0.5);

    nd4j::ops::cosine_distance_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {1,0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE(result->e<float>(0) == -69.);

    delete results;

}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, cosine_distance_loss_test8) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<float>('c', {2,3,4});
    auto weights = NDArrayFactory::create<float>('c', {2,3,1});

    labels.linspace(1);
    weights.assign(0.5);
    predictions.assign(0.5);

    nd4j::ops::cosine_distance_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {2,2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE(result->e<float>(0) == -24.);

    delete results;

}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, cosine_distance_loss_test9) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<float>('c', {2,3,4});
    auto weights = NDArrayFactory::create<float>('c', {1,1});

    labels.linspace(1);
    weights.assign(0.5);
    predictions.assign(0.5);

    nd4j::ops::cosine_distance_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {2,2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE(result->e<double>(0) == -24.);

    delete results;

}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, cosine_distance_loss_test10) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<float>('c', {2,3,4});
    auto weights = NDArrayFactory::create<float>('c', {2,3,1});

    labels.linspace(1);
    weights.assign(0.5);
    predictions.assign(0.5);
    weights.p(0, 0.);
    weights.p(1, 0.);

    nd4j::ops::cosine_distance_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {2,2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE(result->e<double>(0) == -32.);

    delete results;

}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, hinge_loss_test1) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<float>('c', {2,3,4});
    auto weights = NDArrayFactory::create<float>('c', {2,3,4});
    auto expected = NDArrayFactory::create<float>('c', {2,3,4}, {1., 0. , 0., 2.5,0., 3.5, 0., 4.5,0., 5.5, 0., 6.5, 0., 7.5, 0., 8.5,0., 9.5,10., 0. ,0.,11.5, 0.,12.5});

    logits.linspace(1);
    weights.assign(0.5);

    nd4j::ops::hinge_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);
    // result->printBuffer();

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, hinge_loss_test2) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<float>('c', {2,3,4});
    auto weights = NDArrayFactory::create<float>('c', {1,1});
    auto expected = NDArrayFactory::create<float>('c', {2,3,4}, {1., 0. , 0., 2.5,0., 3.5, 0., 4.5,0., 5.5, 0., 6.5, 0., 7.5, 0., 8.5,0., 9.5,10., 0. ,0.,11.5, 0.,12.5});

    logits.linspace(1);
    weights.assign(0.5);

    nd4j::ops::hinge_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);
    // result->printBuffer();

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, hinge_loss_test3) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<float>('c', {2,3,4});
    auto weights = NDArrayFactory::create<float>('c', {1,3,1});
    auto expected = NDArrayFactory::create<float>('c', {2,3,4}, {1., 0. , 0., 2.5,0., 3.5, 0., 4.5,0., 5.5, 0., 6.5, 0., 7.5, 0., 8.5,0., 9.5,10., 0. ,0.,11.5, 0.,12.5});

    logits.linspace(1);
    weights.assign(0.5);

    nd4j::ops::hinge_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);
    // result->printBuffer();

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, hinge_loss_test4) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<float>('c', {2,3,4});
    auto weights = NDArrayFactory::create<float>('c', {2,3,4});

    logits.linspace(1);
    weights.assign(0.5);


    nd4j::ops::hinge_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE(result->e<double>(0) == 83.);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, hinge_loss_test5) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<float>('c', {2,3,4});
    auto weights = NDArrayFactory::create<float>('c', {1,1});

    logits.linspace(1);
    weights.assign(0.5);


    nd4j::ops::hinge_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE(result->e<double>(0) == 83.);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, hinge_loss_test6) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<float>('c', {2,3,4});
    auto weights = NDArrayFactory::create<float>('c', {2,1,1});

    logits.linspace(1);
    weights.assign(0.5);


    nd4j::ops::hinge_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE(result->e<double>(0) == 83.);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, hinge_loss_test7) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<float>('c', {2,3,4});
    auto weights = NDArrayFactory::create<float>('c', {2,3,4});

    logits.linspace(1);
    weights.assign(0.5);


    nd4j::ops::hinge_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 6.91667, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, hinge_loss_test8) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<float>('c', {2,3,4});
    auto weights = NDArrayFactory::create<float>('c', {1,1});

    logits.linspace(1);
    weights.assign(0.5);


    nd4j::ops::hinge_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 6.91667, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, hinge_loss_test9) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<float>('c', {2,3,4});
    auto weights = NDArrayFactory::create<float>('c', {1,1,4});

    logits.linspace(1);
    weights.assign(0.5);


    nd4j::ops::hinge_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 6.91667, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, hinge_loss_test10) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<float>('c', {2,3,4});
    auto weights = NDArrayFactory::create<float>('c', {2,3,4});

    logits.linspace(1);
    weights.assign(0.5);


    nd4j::ops::hinge_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 3.45833, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, hinge_loss_test11) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<float>('c', {2,3,4});
    auto weights = NDArrayFactory::create<float>('c', {2,1,4});

    logits.linspace(1);
    weights.assign(0.5);


    nd4j::ops::hinge_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 3.45833, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, hinge_loss_test12) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<float>('c', {2,3,4});
    auto weights = NDArrayFactory::create<float>('c', {2,3,4});

    logits.linspace(1);
    weights.assign(0.5);
    weights.p(0, 0.);
    weights.p(1, 0.);
    weights.p(2, 0.);
    weights.p(3, 0.);


    nd4j::ops::hinge_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 3.975, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, hinge_loss_test13) {

    auto labels = NDArrayFactory::create<float>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<float>('c', {2,3,4});
    auto weights = NDArrayFactory::create<float>('c', {1,1});

    logits.linspace(1);
    weights.assign(0.);


    nd4j::ops::hinge_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE(result->e<double>(0) == 0.);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, huber_loss_test1) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3,4});
    auto expected = NDArrayFactory::create<double>('c', {2,3,4}, {0.0425 ,0.0875 ,0.13250001,0.17749999,0.22250001,0.26750001,0.31250003,0.35749999,0.4025    ,0.44749999,0.49249998,0.53750002, 0.58249998,0.6275    ,0.67250001,0.71749997,0.76249999,0.8075    ,0.85250002,0.89749998,0.9425    ,0.98749995,1.03250015,1.0775001});

    labels.linspace(0.1, 0.1);
    predictions.linspace(1);
    weights.assign(0.5);

    nd4j::ops::huber_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {0.1}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, huber_loss_test2) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3,1});
    auto expected = NDArrayFactory::create<double>('c', {2,3,4}, {0.0425 ,0.0875 ,0.13250001,0.17749999,0.22250001,0.26750001,0.31250003,0.35749999,0.4025    ,0.44749999,0.49249998,0.53750002, 0.58249998,0.6275    ,0.67250001,0.71749997,0.76249999,0.8075    ,0.85250002,0.89749998,0.9425    ,0.98749995,1.03250015,1.0775001});

    labels.linspace(0.1, 0.1);
    predictions.linspace(1);
    weights.assign(0.5);

    nd4j::ops::huber_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {0.1}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, huber_loss_test3) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {1,1});
    auto expected = NDArrayFactory::create<double>('c', {2,3,4}, {0.0425 ,0.0875 ,0.13250001,0.17749999,0.22250001,0.26750001,0.31250003,0.35749999,0.4025    ,0.44749999,0.49249998,0.53750002, 0.58249998,0.6275    ,0.67250001,0.71749997,0.76249999,0.8075    ,0.85250002,0.89749998,0.9425    ,0.98749995,1.03250015,1.0775001});

    labels.linspace(0.1, 0.1);
    predictions.linspace(1);
    weights.assign(0.5);

    nd4j::ops::huber_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {0.1}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, huber_loss_test4) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3,4});

    labels.linspace(0.1, 0.1);
    predictions.linspace(1);
    weights.assign(0.5);

    nd4j::ops::huber_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {0.1}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 13.44, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, huber_loss_test5) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {1,1});

    labels.linspace(0.1, 0.1);
    predictions.linspace(1);
    weights.assign(0.5);

    nd4j::ops::huber_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {0.1}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 13.44, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, huber_loss_test6) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3,4});

    labels.linspace(0.1, 0.1);
    predictions.linspace(1);
    weights.assign(0.5);

    nd4j::ops::huber_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {0.1}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 1.12, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, huber_loss_test7) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,1,1});

    labels.linspace(0.1, 0.1);
    predictions.linspace(1);
    weights.assign(0.5);

    nd4j::ops::huber_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {0.1}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 1.12, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, huber_loss_test8) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3,4});

    labels.linspace(0.1, 0.1);
    predictions.linspace(1);
    weights.assign(0.5);
    weights.p(0, 0.);
    weights.p(1, 0.);
    weights.p(2, 0.);
    weights.p(3, 0.);

    nd4j::ops::huber_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {0.1}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 1.3, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, huber_loss_test9) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3,4});

    labels.linspace(0.1, 0.1);
    predictions.linspace(1);
    weights.assign(0.5);

    nd4j::ops::huber_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {0.1}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 0.56, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, huber_loss_test10) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {1,1});

    labels.linspace(0.1, 0.1);
    predictions.linspace(1);
    weights.assign(0.5);

    nd4j::ops::huber_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {0.1}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 0.56, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, huber_loss_test11) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3,4});

    labels.linspace(0.1, 0.1);
    predictions.linspace(1);
    weights.assign(0.5);
    weights.p(0, 0.);
    weights.p(1, 0.);
    weights.p(2, 0.);
    weights.p(3, 0.);

    nd4j::ops::huber_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {0.1}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 0.65, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, log_loss_test1) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3,4});
    auto expected = NDArrayFactory::create<double>('c', {2,3,4}, {1.60943663,  2.48403668,  3.05256081,  3.40363169,  3.57730675,  3.59525585,  3.46986699,  3.20791793,  2.81228209,  2.28273821,  1.61630058,  0.80721998, -0.15329313, -1.27764463, -2.5828433 , -4.09208679, -5.83734226, -7.8636713 ,-10.23689461,-13.05822182,-16.49509811,-20.85659218,-26.82411766,-36.52717209});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::log_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {1e-7}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, log_loss_test2) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,1,4});
    auto expected = NDArrayFactory::create<double>('c', {2,3,4}, {1.60943663,  2.48403668,  3.05256081,  3.40363169,  3.57730675,  3.59525585,  3.46986699,  3.20791793,  2.81228209,  2.28273821,  1.61630058,  0.80721998, -0.15329313, -1.27764463, -2.5828433 , -4.09208679, -5.83734226, -7.8636713 ,-10.23689461,-13.05822182,-16.49509811,-20.85659218,-26.82411766,-36.52717209});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::log_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {1e-7}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, log_loss_test3) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    NDArray weights(nd4j::DataType::DOUBLE);
    auto expected = NDArrayFactory::create<double>('c', {2,3,4}, {1.60943663,  2.48403668,  3.05256081,  3.40363169,  3.57730675,  3.59525585,  3.46986699,  3.20791793,  2.81228209,  2.28273821,  1.61630058,  0.80721998, -0.15329313, -1.27764463, -2.5828433 , -4.09208679, -5.83734226, -7.8636713 ,-10.23689461,-13.05822182,-16.49509811,-20.85659218,-26.82411766,-36.52717209});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::log_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {1e-7}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, log_loss_test4) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3,4});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::log_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {1e-7}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), -113.886429, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, log_loss_test5) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {1,3,1});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::log_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {1e-7}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), -113.886429, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, log_loss_test6) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    NDArray weights(nd4j::DataType::DOUBLE);

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::log_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {1e-7}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), -113.886429, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, log_loss_test7) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3,4});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::log_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {1e-7}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), -9.490536, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, log_loss_test8) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {1,3,1});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::log_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {1e-7}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), -9.490536, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, log_loss_test9) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    NDArray weights(nd4j::DataType::DOUBLE);

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::log_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {1e-7}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), -9.490536, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, log_loss_test10) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3,4});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);
    weights.p(0, 0.);
    weights.p(1, 0.);
    weights.p(2, 0.);
    weights.p(3, 0.);

    nd4j::ops::log_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {1e-7}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), -12.443609, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, log_loss_test11) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3,4});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::log_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {1e-7}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), -4.745268, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, log_loss_test12) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {1,1});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::log_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {1e-7}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), -4.745268, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, log_loss_test13) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3,4});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);
    weights.p(0, 0.);
    weights.p(1, 0.);
    weights.p(2, 0.);
    weights.p(3, 0.);

    nd4j::ops::log_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {1e-7}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), -6.221805, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, mean_pairwssqerr_loss_test1) {
    auto labels = NDArrayFactory::create<double>('c', {1,3}, {0., 0.5, 1.});
    auto predictions = NDArrayFactory::create<double>('c', {1,3}, {1., 1., 1.});
    auto weights = NDArrayFactory::create<double>('c', {1,1}, {1});
    auto expected = NDArrayFactory::create<double>('c', {1,1}, {1.});

    nd4j::ops::mean_pairwssqerr_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, mean_pairwssqerr_loss_test2) {
    auto labels = NDArrayFactory::create<double>('c', {10,4}, {-0.5533444483384939, -0.4045807428083095, -0.38990808632111873, -1.3367815555936828, 2.2110825342567204, -0.3322538938773163, 0.5683588435736076, 1.401524673423209, -0.2216208609234102, -0.23645194877057543, -1.9319189398422172, 0.6106128799796062, 1.6973842275926025, -2.8306371397325553E-4, -1.1550401544465256, -0.08357706614294765, -0.27784822018757077, 0.8290894318337857, 1.6484476009013025, -0.7752524785358668, -0.9700596207063842, 3.0809371469543207, -0.23684959888998405, 0.22403535560739518, 0.6146150452128438, -1.1250088686147994, -0.5915314787415693, -0.0944090155356556, 0.7995514825959854, -1.2290496239142903, -1.8329592004926936, -0.1694821152623061, -1.7614978090471403, 0.07929168376086736, 0.4086255139492943, 2.045562727396195, -0.48701853719962834, 0.10304152395720723, -0.8993147347502636, -0.49078404206110715});
    auto predictions = NDArrayFactory::create<double>('c', {10,4}, {-0.5982871220907984, 1.2010665656903237, 0.30243355682445544, -0.2070857400459659, 0.6962389393180044, -0.5878034128580758, 0.8325626284025988, -0.3555823702782838, -0.7099759151434476, 1.7971905051128672, -1.1018498592680859, 0.008705918349147959, -1.713038986676157, 0.5029671900704719, 0.7491261275031563, -0.34800067781360444, -1.3529065441284513, -0.6075230577852321, -0.6153583973120907, 1.6014780660677996, 0.6444219215516616, 0.7925830851904783, -0.5006063079380708, 1.7812300901376552, 0.4736193941708224, 1.411502849640833, 0.9555142545037492, -0.03936687661890644, 1.31661624967917, 0.7344531724786305, 0.8388550872918745, 0.7010030219905558, -0.5442944240155373, 0.4437344837841118, -1.7502823958671712, -1.9271369730241665, 0.9256612923554498, 1.9065401403827893, 0.42450175148842717, -0.11783183865542822});
    auto weights = NDArrayFactory::create<double>('c', {1,1}, {1});
    auto expected = NDArrayFactory::create<double>('c', {10,1}, {1.9665822560405073, 3.806679563402927, 6.185624212589066, 20.237895345263905, 16.739700814450472, 13.655430201400929, 6.473256392322658, 3.9337379694106325, 22.509455553531062, 1.4741234749089487});

    nd4j::ops::mean_pairwssqerr_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, mean_pairwssqerr_loss_test3) {
    auto labels = NDArrayFactory::create<double>('c', {10,4}, {0.9165069946629816, 0.166426191704143, 0.13873357227527264, -0.5986162145785378, 0.4763504550662989, 1.2259816058633732, -0.4653205175596491, -1.7447031523970766, 1.349525448316014, 2.433089865629357, -2.54858150221601, -0.6060282162911894, 0.2625377104613349, -0.5007107584102752, 0.9576065700956302, -0.35787770401703584, -0.2608532564720665, 0.65688909921908, -0.1705876431948587, 1.2052884124800949, -0.976783296084278, 1.1163504624016534, -0.10545986164581109, -1.0632271027867568, 0.26460250034147065, -0.2299030354616135, -0.418989869909565, 0.7954060747536896, 0.37934127200736545, 0.8550487997440007, 0.2984909806904042, 0.1329065864221682, 1.478600294413247, 0.05421279873635542, -1.0552978360622536, -0.743808639782604, -1.3371851696151362, 2.7752972493355963, -1.6107187893743549, 1.5030902829432997});
    auto predictions = NDArrayFactory::create<double>('c', {10,4}, {-3.398114657004427, 0.40587455906092945, 1.587706448479039, 0.27394335709083156, 1.0463122023764637, -0.6552570653663903, -0.26929204111727345, -2.710461824817806, 0.9141296064806023, -0.7632270851454939, -0.4077235519855459, 0.5555107559107472, -0.6776140976423888, 1.2422270521180823, 0.2372445100636733, 0.08522757123963924, -2.708523129389936, 0.09738215252575103, -0.8797837670498875, 0.8714091607391934, -0.628958978867591, 0.49380147969660415, -0.6663578349373824, 0.14570184758600965, -0.4710388511314244, 0.7708214742640788, 0.06836525442683238, -1.2786368797129386, -0.5077556003990912, 0.45383439418987664, 1.1686877788409553, -0.3078567969393852, -2.2375730522738198, 1.0108200459611192, 0.21955367964983963, 1.2268011099696847, 0.48061693077695455, -0.5306373077054981, 1.5005367299570744, -2.1005486985463966});
    auto weights = NDArrayFactory::create<double>('c', {10,1}, {0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0});
    auto expected = NDArrayFactory::create<double>('c', {10,1}, {0.0, 0.0, 21.748459867092496, 6.090581568657439, 7.51315897553838, 5.999534225166869, 22.58050883748054, 6.8600435676788605, 107.5976928688877, 191.56864939172544});

    nd4j::ops::mean_pairwssqerr_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, mean_pairwssqerr_loss_test4) {
    auto labels = NDArrayFactory::create<double>('c', {10,4}, {-1.9540657282602247, -0.37099621218123746, 0.24959541842365968, 0.4125896396216978, -0.8661959659606203, 0.3651479206362867, -1.7475031047706964, -1.0962133982440159, 0.8451229874730279, 0.6876932162478913, 1.2598782790596628, 0.9372328828104118, 1.383555504464105, -0.816048166961237, 0.009041816630426176, -0.004376554457540983, -0.2386352931506252, -0.6494407817111416, 1.7888273635934742, -1.2157303560822368, -0.2446697859467434, -0.3040881765177774, -0.25843499040765916, -0.16479617511053568, 1.8063435075905592, 0.36002291874022285, -0.43317974028771883, 1.070086390817373, -1.0788479808458253, -0.3364318348487324, -0.859106579072977, 0.43984270049845064, -0.23662331183489546, -1.263417124724063, -0.3123732566483939, -0.125249623799724, -1.951308433393268, -0.4925779190927575, -1.081735149025745, -1.9910331435034687});
    auto predictions = NDArrayFactory::create<double>('c', {10,4}, {-1.7053977111021588, 1.7704125629388408, -0.0876171627499475, 0.9428762101237441, 0.9080108618240852, -0.478732892339118, -0.8189639230649537, 1.3359668242925342, -0.07499867017894829, 0.6169780756804321, -1.1891117691972148, -0.319354110980483, -1.4287263424900434, -0.3556443786879834, 0.6389682186473912, 0.3161742985911756, 0.9047447733840537, -1.9974117226910393, 2.1067775658502326, 0.17035521714679938, -1.1393894489992826, 1.4570837278971687, 0.6312249731754015, -0.42793125692777634, -1.0685964336386844, -0.3590636581851568, -0.19147354841437528, -0.10128937266756889, -0.5714869078294972, 0.2682604831358205, 0.6608524575561853, 0.35658907103040305, -0.7053263272861181, -0.6318441042427088, 2.131292677079184, -0.3624048087249232, 1.6008209804575328, 0.1245980660014825, 1.0685424462364297, -0.5672594432046791});
    auto weights = NDArrayFactory::create<double>('c', {1,1}, {1});

    nd4j::ops::mean_pairwssqerr_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 60.74394998193965, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, mean_pairwssqerr_loss_test5) {
    auto labels = NDArrayFactory::create<double>('c', {10,4}, {0.9165069946629816, 0.166426191704143, 0.13873357227527264, -0.5986162145785378, 0.4763504550662989, 1.2259816058633732, -0.4653205175596491, -1.7447031523970766, 1.349525448316014, 2.433089865629357, -2.54858150221601, -0.6060282162911894, 0.2625377104613349, -0.5007107584102752, 0.9576065700956302, -0.35787770401703584, -0.2608532564720665, 0.65688909921908, -0.1705876431948587, 1.2052884124800949, -0.976783296084278, 1.1163504624016534, -0.10545986164581109, -1.0632271027867568, 0.26460250034147065, -0.2299030354616135, -0.418989869909565, 0.7954060747536896, 0.37934127200736545, 0.8550487997440007, 0.2984909806904042, 0.1329065864221682, 1.478600294413247, 0.05421279873635542, -1.0552978360622536, -0.743808639782604, -1.3371851696151362, 2.7752972493355963, -1.6107187893743549, 1.5030902829432997});
    auto predictions = NDArrayFactory::create<double>('c', {10,4}, {-3.398114657004427, 0.40587455906092945, 1.587706448479039, 0.27394335709083156, 1.0463122023764637, -0.6552570653663903, -0.26929204111727345, -2.710461824817806, 0.9141296064806023, -0.7632270851454939, -0.4077235519855459, 0.5555107559107472, -0.6776140976423888, 1.2422270521180823, 0.2372445100636733, 0.08522757123963924, -2.708523129389936, 0.09738215252575103, -0.8797837670498875, 0.8714091607391934, -0.628958978867591, 0.49380147969660415, -0.6663578349373824, 0.14570184758600965, -0.4710388511314244, 0.7708214742640788, 0.06836525442683238, -1.2786368797129386, -0.5077556003990912, 0.45383439418987664, 1.1686877788409553, -0.3078567969393852, -2.2375730522738198, 1.0108200459611192, 0.21955367964983963, 1.2268011099696847, 0.48061693077695455, -0.5306373077054981, 1.5005367299570744, -2.1005486985463966});
    auto weights = NDArrayFactory::create<double>('c', {1,1}, {1});

    nd4j::ops::mean_pairwssqerr_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 15.189082270182983, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, mean_pairwssqerr_loss_test6) {
    auto labels = NDArrayFactory::create<double>('c', {10,4}, {0.7712557146220891, 0.37344724586647443, -1.465944048516541, 0.3226845250222374, 0.3153238532645865, -0.6453963287132424, -1.7695663855309438, -0.31350813714835285, 0.6209850696184357, -1.0632582557661083, 0.8971205782356552, -0.7361143357044725, 0.4349813432397299, 1.1012674501462072, -1.846028584047857, -0.04711049067212126, 0.3511384383511822, -1.5908669452488973, 0.6271232025632083, -0.5370025878354387, 0.09775855957778733, 0.8465118033582384, -0.5118005514773271, -0.8215749768059044, -0.5154271246850248, -0.6614138367887438, -2.721743038982485, -0.20634785234624944, 1.074134378795222, -0.515671736473577, 0.33574452224656587, -0.4258992514621533, -1.6946210614398756, 2.0853105493575246, -0.23223717047374226, -1.3145231337861756, -0.307739072607248, -0.13713627422120406, -0.05615471338688221, -0.7031780205843188});
    auto predictions = NDArrayFactory::create<double>('c', {10,4}, {-0.8253096544930751, 0.81324545672996, 1.2530858908292535, 0.6881658781201572, 0.11626814971230247, 0.810096847233213, -0.41726775033902014, -0.07246036077805246, -0.3491325803119671, -0.7381717490678714, -1.258884944199858, 2.6195012275145992, 0.3241066697239042, -1.3306435333372646, -0.3413119919683999, 0.13167356361127197, -0.3992424507051653, 0.14454163796541403, -2.4931643208872316, 1.8740911656038526, -2.3404306490682956, -0.8036392545918644, -1.9726177395274997, -0.20128619801149433, -1.0680828820641624, -0.6228179015361869, 1.0785520122486962, -0.26148573195062036, -0.9154287856620913, 0.6612224269248097, -0.21735407368781667, 0.5584864652543093, 1.0208212201167435, -0.7560947201084579, -0.9092906572495081, 0.47525819203475833, 1.2215678456801444, -0.39319465979983964, 1.9435677135606038, 1.4540100039010526});
    auto weights = NDArrayFactory::create<double>('c', {1,1}, {1});

    nd4j::ops::mean_pairwssqerr_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 13.568564090650312, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, mean_pairwssqerr_loss_test7) {
    auto labels = NDArrayFactory::create<double>('c', {10,4}, {-0.06125002348040258, 0.5143643450377119, 2.6790723358660036, -0.8032552006036418, -2.4374371040644163, -0.1562964773317163, -1.3957988654288038, 1.2791626503391635, -1.433421873294552, -1.1819478586737284, 0.05162930965054662, -0.538650473505593, -0.548171720093084, -0.3103900587344872, -2.3955103171953342, 0.7127238680062526, 0.7182079438418053, 1.1842662402382182, 0.09585189676958715, 0.9276146067349225, 0.7856673461867428, 0.41368195133354113, -0.2939280190178078, -2.400566355562181, -1.1841519118039245, -1.066170501847581, -0.9274507409610022, 1.7671863041813334, -1.2849985781031494, -1.275990164491566, -0.8866824403466698, -0.6074077385015517, 0.7647344603897107, -1.048099070426831, 0.9433828938345293, -0.5591415819237762, 1.7962773615541947, -0.42365710367758247, -0.0385518907389571, -1.109959713481321});
    auto predictions = NDArrayFactory::create<double>('c', {10,4}, {-0.7445687252538243, 0.2293875300325241, -1.0231630280206505, -0.18532545069458992, -0.07797403344353356, -0.9132035669873787, 0.9352296415512886, -1.7406458535354787, 0.8578334648119594, -0.6186274065269556, 0.4874824473654153, -0.9285817343788997, 0.1654680500853023, -0.6371334533926012, 1.3115245864160707, -2.072558735678832, 0.660795731844733, -0.34942292767044864, 0.05787182311194333, -0.12939210444705632, -0.6457028552461069, -0.6048992126598505, -0.17179604529778109, 1.292989642826032, -0.28867767615688045, 0.7635565516046265, -1.5464151753137487, -1.273368390129285, -1.074046012825826, -0.3534580692302915, 0.5757285568118223, 1.823271242883469, 0.31618576929075215, 0.5422847605415213, -0.7836698021860683, -0.6292022623165172, 2.1114596721927508, 0.4634986528550097, 0.08922001427846013, 1.5767749644913223});
    auto weights = NDArrayFactory::create<double>('c', {10,1}, {0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0});

    nd4j::ops::mean_pairwssqerr_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 198.318201904499, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, mean_pairwssqerr_loss_test8) {
    auto labels = NDArrayFactory::create<double>('c', {10,4}, {1.2003157672694111, -1.0738078620687983, 1.4513396266923826, 0.5753935722952708, -0.5424028602429585, 0.9816221437385002, -1.0566397385428794, 1.503481308203513, -0.6543147953583112, 1.7453669976827346, -0.1557689124924227, 0.3387794658137257, -1.2306868494328145, -0.3299042398395769, 0.026464968146954395, -1.5077479623528403, -0.27514168845621795, 0.18739335150879793, 1.7319910646645431, 1.5228099405663476, 0.8522684742808536, 0.2362049362675063, 0.2610756525241469, 0.457998065505686, -2.7342179885912623, -0.10968795695808314, 0.581598742956297, -1.9309885922934567, -1.5775788440607954, -0.04254899350225641, -0.3125858556254039, -1.1328154327730207, 0.00566243314780096, 0.8492052576274621, 0.05945202212214481, 1.4976918834497108, 0.8869512918387292, 0.4014181932175132, -0.015512552855187248, -1.3609667909108454});
    auto predictions = NDArrayFactory::create<double>('c', {10,4}, {-1.1088399463364795, 0.09302972835006071, 0.033839927431215555, -0.39567507675572494, 0.8269497207597863, 1.111162272517752, 0.4930937252630912, -1.4561668998323452, 0.9417715392862969, -1.0553855492735509, 0.05848285303876081, 0.8852337518047972, -0.7472824481835305, 0.404906922583895, -0.2198309547562547, 1.9536515925189717, 0.8165036568007779, -0.19524282774410398, -0.09111693087754393, 1.1604245932512238, -0.6243762858131077, 1.4297003275591034, -0.17220079411538428, -2.3139504326793032, 0.3839796486999712, 2.0287791964679234, 0.1534441713632995, -0.6062103319229825, -0.4965880982906036, -0.373907747810053, -1.6566345746154432, 0.17534987728494222, -1.6713458890334796, 1.254628987947714, 1.914596591838086, -1.0816010467183583, 0.25033738231939673, -1.605752685708275, 1.1029112741353981, 0.3237822320282494});
    auto weights = NDArrayFactory::create<double>('c', {10,1}, {0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0});

    nd4j::ops::mean_pairwssqerr_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 10.709003499121707, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, mean_pairwssqerr_loss_test9) {
    auto labels = NDArrayFactory::create<double>('c', {10,4}, {0.054445708809271035, 2.107634671009908, -0.7906421810578572, -1.075840781788665, 0.11881403008710377, 0.8444812915085994, -0.305754504070933, 1.6429935026781464, 0.8155105031719394, 0.04900134907242568, 0.6847004530975871, 0.23315535615893132, 0.17011663306483038, -1.1865513655938285, 1.5931597087896407, -1.7937514075547496, -0.036695307704292295, -1.6416280650778925, 1.130578912176608, -1.1267224667674058, -0.8690453889645526, 0.6717944721406133, 0.0850200492927782, 1.1294419289013125, 0.2154793028698133, 0.4557382556428947, -0.7343674069166273, -0.20013117860162175, -0.6096905108192562, 0.42022878041905926, -0.7446306649741321, 0.01724811509597817, 1.843091605690758, 1.008879504632424, 1.198292190689489, -0.4474144618813475, 0.25202981742888664, 0.07036737843407408, 1.2400630276444486, -1.1072825235557615});
    auto predictions = NDArrayFactory::create<double>('c', {10,4}, {-1.6788168943811437, 1.1823653279081687, -0.3580541857004183, -0.4449970504370699, -1.3031645333940127, 0.5755013195969282, -0.7997343141774744, -0.8806735270004084, 0.9705277499376251, -1.6360067944580943, 0.12579369136710156, 1.0525902242414313, -1.625751312422252, -0.03900152587147075, 0.4112500942756277, 0.6589999986358094, 0.6144107111689617, 2.8561269030217264, 1.5299963640392247, -0.314093051147705, 1.6523278218751989, -0.5504653447714114, 0.53395260877978, 0.409795577698306, 0.4466825218051794, 1.2382059301630401, 0.4834869732526594, -0.635409128905636, -1.9343816841697272, -0.4192523056060229, -1.0662979055059818, 0.4270901960618144, -0.7391311480757151, -0.8268168961897452, -1.0855715553457785, -9.410401291588706E-4, -0.7721838774717349, 0.4784019579457375, -0.6979798841469268, -0.319729737118584});
    auto weights = NDArrayFactory::create<double>('c', {10,1}, {0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0});

    nd4j::ops::mean_pairwssqerr_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 17.686067864414472, 1e-5);

    delete results;
}
///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, mean_sqerr_loss_test1) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3,4});
    auto expected = NDArrayFactory::create<double>('c', {2,3,4}, {0.125, 0.5, 1.125, 2., 3.125, 4.5, 6.125, 8.,10.125,12.5,15.125,18.,21.125,24.5,28.125,32.,36.125,40.5,45.125,50.,55.125,60.5,66.125,72.});

    predictions.linspace(0.5, 0.5);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::mean_sqerr_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, mean_sqerr_loss_test2) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,1,4});
    auto expected = NDArrayFactory::create<double>('c', {2,3,4}, {0.125, 0.5, 1.125, 2., 3.125, 4.5, 6.125, 8.,10.125,12.5,15.125,18.,21.125,24.5,28.125,32.,36.125,40.5,45.125,50.,55.125,60.5,66.125,72.});

    predictions.linspace(0.5, 0.5);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::mean_sqerr_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, mean_sqerr_loss_test3) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,1,1});
    auto expected = NDArrayFactory::create<double>('c', {2,3,4}, {0.125, 0.5, 1.125, 2., 3.125, 4.5, 6.125, 8.,10.125,12.5,15.125,18.,21.125,24.5,28.125,32.,36.125,40.5,45.125,50.,55.125,60.5,66.125,72.});

    predictions.linspace(0.5, 0.5);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::mean_sqerr_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, mean_sqerr_loss_test4) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3,4});
    auto expected = NDArrayFactory::create<double>('c', {2,3,4}, {0., 0., 0., 0., 3.125, 4.5, 6.125, 8.,10.125,12.5,15.125,18.,21.125,24.5,28.125,32.,36.125,40.5,45.125,50.,55.125,60.5,66.125,72.});

    predictions.linspace(0.5, 0.5);
    labels.linspace(1);
    weights.assign(0.5);
    weights.p(0, 0.);
    weights.p(1, 0.);
    weights.p(2, 0.);
    weights.p(3, 0.);

    nd4j::ops::mean_sqerr_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, mean_sqerr_loss_test5) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3,4});

    predictions.linspace(0.5, 0.5);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::mean_sqerr_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 612.5, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, mean_sqerr_loss_test6) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {1,1,4});

    predictions.linspace(0.5, 0.5);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::mean_sqerr_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 612.5, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, mean_sqerr_loss_test7) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {1,1});

    predictions.linspace(0.5, 0.5);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::mean_sqerr_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 612.5, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, mean_sqerr_loss_test8) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3,4});

    predictions.linspace(0.5, 0.5);
    labels.linspace(1);
    weights.assign(0.5);
    weights.p(0, 0.);
    weights.p(1, 0.);
    weights.p(2, 0.);
    weights.p(3, 0.);

    nd4j::ops::mean_sqerr_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 608.75, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, mean_sqerr_loss_test9) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3,4});

    predictions.linspace(0.5, 0.5);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::mean_sqerr_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 51.041668, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, mean_sqerr_loss_test10) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {1,3,1});

    predictions.linspace(0.5, 0.5);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::mean_sqerr_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 51.041668, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, mean_sqerr_loss_test11) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {1,1});

    predictions.linspace(0.5, 0.5);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::mean_sqerr_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 51.041668, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, mean_sqerr_loss_test12) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3,1});

    predictions.linspace(0.5, 0.5);
    labels.linspace(1);
    weights.assign(0.5);
    weights.p(0, 0.);
    weights.p(1, 0.);
    weights.p(2, 0.);

    nd4j::ops::mean_sqerr_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 88.541664, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, mean_sqerr_loss_test13) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3,4});

    predictions.linspace(0.5, 0.5);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::mean_sqerr_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 25.520834, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, mean_sqerr_loss_test14) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,1,4});

    predictions.linspace(0.5, 0.5);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::mean_sqerr_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 25.520834, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, mean_sqerr_loss_test15) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {1,1});

    predictions.linspace(0.5, 0.5);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::mean_sqerr_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 25.520834, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, mean_sqerr_loss_test16) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4});
    auto predictions = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3,1});

    predictions.linspace(0.5, 0.5);
    labels.linspace(1);
    weights.assign(0.5);
    weights.p(0, 0.);
    weights.p(1, 0.);
    weights.p(2, 0.);

    nd4j::ops::mean_sqerr_loss op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 44.270832, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, sigm_cross_entropy_loss_test1) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3,4});
    auto expected = NDArrayFactory::create<double>('c', {2,3,4}, {0.37219834,0.29906943,0.27717763,0.45650762,0.23703849,0.51874399,0.20159303,0.58555031,0.17057693,0.65663081,0.14366767,0.73164123,0.12050423,0.81020868,0.10070664,0.89195037,0.08389302,0.97648883,1.01969337,0.06346401,0.05775976,1.15254164,0.04777273,1.2434181 });

    logits.linspace(0.1, 0.1);
    weights.assign(0.5);

    nd4j::ops::sigm_cross_entropy_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {0.}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, sigm_cross_entropy_loss_test2) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,1,1});
    auto expected = NDArrayFactory::create<double>('c', {2,3,4}, {0.37219834,0.29906943,0.27717763,0.45650762,0.23703849,0.51874399,0.20159303,0.58555031,0.17057693,0.65663081,0.14366767,0.73164123,0.12050423,0.81020868,0.10070664,0.89195037,0.08389302,0.97648883,1.01969337,0.06346401,0.05775976,1.15254164,0.04777273,1.2434181 });

    logits.linspace(0.1, 0.1);
    weights.assign(0.5);

    nd4j::ops::sigm_cross_entropy_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {0.}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, sigm_cross_entropy_loss_test3) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {1,1});
    auto expected = NDArrayFactory::create<double>('c', {2,3,4}, {0.37219834,0.29906943,0.27717763,0.45650762,0.23703849,0.51874399,0.20159303,0.58555031,0.17057693,0.65663081,0.14366767,0.73164123,0.12050423,0.81020868,0.10070664,0.89195037,0.08389302,0.97648883,1.01969337,0.06346401,0.05775976,1.15254164,0.04777273,1.2434181 });

    logits.linspace(0.1, 0.1);
    weights.assign(0.5);

    nd4j::ops::sigm_cross_entropy_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {0.}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, sigm_cross_entropy_loss_test4) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3,4});
    auto expected = NDArrayFactory::create<double>('c', {2,3,4}, {0.24719833, 0.54906946, 0.65217763,-0.04349237,0.86203849,-0.23125602, 1.07659304,-0.41444966,1.29557693,-0.59336919, 1.5186677 ,-0.76835877,1.74550426,-0.93979132, 1.9757067 ,-1.10804963,2.20889306,-1.27351117,-1.35530663, 2.56346393,2.68275976,-1.59745836, 2.92277265,-1.7565819 });

    logits.linspace(0.1, 0.1);
    weights.assign(0.5);

    nd4j::ops::sigm_cross_entropy_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {5.}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, sigm_cross_entropy_loss_test5) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3,4});

    logits.linspace(0.1, 0.1);
    weights.assign(0.5);

    nd4j::ops::sigm_cross_entropy_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {0.}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 11.2187976837, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, sigm_cross_entropy_loss_test6) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3,1});

    logits.linspace(0.1, 0.1);
    weights.assign(0.5);

    nd4j::ops::sigm_cross_entropy_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {0.}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 11.2187976837, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, sigm_cross_entropy_loss_test7) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {1,1});

    logits.linspace(0.1, 0.1);
    weights.assign(0.5);

    nd4j::ops::sigm_cross_entropy_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {0.}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 11.2187976837, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, sigm_cross_entropy_loss_test8) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3,4});

    logits.linspace(0.1, 0.1);
    weights.assign(0.5);

    nd4j::ops::sigm_cross_entropy_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {5.}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 10.2187976837, 1e-5);

    delete results;
}


///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, sigm_cross_entropy_loss_test9) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3,1});

    logits.linspace(0.1, 0.1);
    weights.assign(0.5);
    weights.p(0, 0.);
    weights.p(1, 0.);
    weights.p(2, 0.);

    nd4j::ops::sigm_cross_entropy_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {5.}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 6.06840181351, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, sigm_cross_entropy_loss_test10) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3,4});

    logits.linspace(0.1, 0.1);
    weights.assign(0.5);

    nd4j::ops::sigm_cross_entropy_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {0.}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 0.934899806976, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, sigm_cross_entropy_loss_test11) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {1,1,4});

    logits.linspace(0.1, 0.1);
    weights.assign(0.5);

    nd4j::ops::sigm_cross_entropy_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {0.}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 0.934899806976, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, sigm_cross_entropy_loss_test12) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {1,1});

    logits.linspace(0.1, 0.1);
    weights.assign(0.5);

    nd4j::ops::sigm_cross_entropy_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {5.}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 0.851566493511, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, sigm_cross_entropy_loss_test13) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3,1});

    logits.linspace(0.1, 0.1);
    weights.assign(0.5);
    weights.p(0, 0.);
    weights.p(1, 0.);
    weights.p(2, 0.);

    nd4j::ops::sigm_cross_entropy_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {5.}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 1.01140034199, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, sigm_cross_entropy_loss_test14) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3,4});

    logits.linspace(0.1, 0.1);
    weights.assign(0.5);

    nd4j::ops::sigm_cross_entropy_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {0.}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 0.467449903488, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, sigm_cross_entropy_loss_test15) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {1,3,1});

    logits.linspace(0.1, 0.1);
    weights.assign(0.5);

    nd4j::ops::sigm_cross_entropy_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {0.}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 0.467449903488, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, sigm_cross_entropy_loss_test16) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {1,1});

    logits.linspace(0.1, 0.1);
    weights.assign(0.5);

    nd4j::ops::sigm_cross_entropy_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {5.}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 0.425783246756, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, sigm_cross_entropy_loss_test17) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3,1});

    logits.linspace(0.1, 0.1);
    weights.assign(0.5);
    weights.p(0, 0.);
    weights.p(1, 0.);
    weights.p(2, 0.);

    nd4j::ops::sigm_cross_entropy_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {5.}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 0.505700170994, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, softmax_cross_entropy_loss_test1) {

    auto labels = NDArrayFactory::create<int>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3});
    auto expected = NDArrayFactory::create<double>('c', {2,3}, {1.39253557,1.44253552,1.44253552,1.44253552,1.39253557,1.44253552});

    logits.linspace(0.1, 0.1);
    weights.assign(0.5);

    nd4j::ops::softmax_cross_entropy_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {0.}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);
    result->printIndexedBuffer("SCEL Output");
    expected.printIndexedBuffer("SCEL Expect");
    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, softmax_cross_entropy_loss_test2) {

    auto labels = NDArrayFactory::create<int>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3});
    auto expected = NDArrayFactory::create<double>('c', {2,3}, {-0.92835701,-1.12835705,-1.12835705,-1.12835705,-0.92835701,-1.12835705});

    logits.linspace(0.1, 0.1);
    weights.assign(0.5);

    nd4j::ops::softmax_cross_entropy_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {5.}, {0}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, softmax_cross_entropy_loss_test3) {

    auto labels = NDArrayFactory::create<int>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,1});
    auto expected = NDArrayFactory::create<double>('c', {2,3}, {-0.92835701,-1.12835705,-1.12835705,-1.12835705,-0.92835701,-1.12835705});

    logits.linspace(0.1, 0.1);
    weights.assign(0.5);

    nd4j::ops::softmax_cross_entropy_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {5.}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, softmax_cross_entropy_loss_test4) {

    auto labels = NDArrayFactory::create<int>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {1,3});
    auto expected = NDArrayFactory::create<double>('c', {2,3}, {-0.92835701,-1.12835705,-1.12835705,-1.12835705,-0.92835701,-1.12835705});

    logits.linspace(0.1, 0.1);
    weights.assign(0.5);

    nd4j::ops::softmax_cross_entropy_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {5.}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, softmax_cross_entropy_loss_test5) {

    auto labels = NDArrayFactory::create<int>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {1,1});
    auto expected = NDArrayFactory::create<double>('c', {2,3}, {-0.92835701,-1.12835705,-1.12835705,-1.12835705,-0.92835701,-1.12835705});

    logits.linspace(0.1, 0.1);
    weights.assign(0.5);

    nd4j::ops::softmax_cross_entropy_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {5.}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, softmax_cross_entropy_loss_test6) {

    auto labels = NDArrayFactory::create<int>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3});

    logits.linspace(0.1, 0.1);
    weights.assign(0.5);

    nd4j::ops::softmax_cross_entropy_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {0.}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), 8.55521392822, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, softmax_cross_entropy_loss_test7) {

    auto labels = NDArrayFactory::create<int>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {2,3});

    logits.linspace(0.1, 0.1);
    weights.assign(0.5);

    nd4j::ops::softmax_cross_entropy_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {5.}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), -6.37014198303, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, softmax_cross_entropy_loss_test8) {

    auto labels = NDArrayFactory::create<int>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {1,1});

    logits.linspace(0.1, 0.1);
    weights.assign(0.5);

    nd4j::ops::softmax_cross_entropy_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {5.}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), -6.37014198303, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, softmax_cross_entropy_loss_test9) {

    auto labels = NDArrayFactory::create<int>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {1,3});

    logits.linspace(0.1, 0.1);
    weights.assign(0.5);

    nd4j::ops::softmax_cross_entropy_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {5.}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), -6.37014198303, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, softmax_cross_entropy_loss_test10) {

    auto labels = NDArrayFactory::create<int>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {1,3});

    logits.linspace(0.1, 0.1);
    weights.assign(0.5);

    nd4j::ops::softmax_cross_entropy_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {5.}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), -2.12338066101, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, softmax_cross_entropy_loss_test11) {

    auto labels = NDArrayFactory::create<int>('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto weights = NDArrayFactory::create<double>('c', {1,3});

    logits.linspace(0.1, 0.1);
    weights.assign(0.5);

    nd4j::ops::softmax_cross_entropy_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {5.}, {3}, {}, false, nd4j::DataType::DOUBLE);

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), -1.06169033051, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, softmax_cross_entropy_loss_test12) {

    auto labels = NDArrayFactory::create<int>('c', {2,4},{0,1,1,0,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,4});
    auto weights = NDArrayFactory::create<double>('c', {2,1});

    logits.linspace(0.1, 0.1);
    weights.assign(0.5);

    nd4j::ops::softmax_cross_entropy_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {5.}, {3}, {}, false, nd4j::DataType::DOUBLE);

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(result->isScalar());
    ASSERT_NEAR(result->e<double>(0), -2.18880319595, 1e-5);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, softmax_cross_entropy_loss_test13) {

    auto labels = NDArrayFactory::create<int>('c', {2,4},{0,1,1,0,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,4});
    auto weights = NDArrayFactory::create<double>('c', {2,1});
    auto expected = NDArrayFactory::create<double>('c', {2,1}, {1.39253557,1.44253552});

    logits.linspace(0.1, 0.1);
    weights.assign(0.5);

    nd4j::ops::softmax_cross_entropy_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {0.}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}



///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, softmax_cross_entropy_loss_test14) {

    auto labels = NDArrayFactory::create<int>('c', {2,4},{0,1,1,0,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,4});
    auto weights = NDArrayFactory::create<double>('c', {2,1});
    auto expected = NDArrayFactory::create<double>('c', {2,1}, {-2.08880329, -2.28880334});

    logits.linspace(0.1, 0.1);
    weights.assign(0.5);

    nd4j::ops::softmax_cross_entropy_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {5.}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, softmax_cross_entropy_loss_test15) {

    auto labels = NDArrayFactory::create<int>('c', {2,4},{0,1,1,0,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,4});
    auto weights = NDArrayFactory::create<double>('c', {1,1});
    auto expected = NDArrayFactory::create<double>('c', {2,1}, {-2.08880329, -2.28880334});

    logits.linspace(0.1, 0.1);
    weights.assign(0.5);

    nd4j::ops::softmax_cross_entropy_loss op;
    auto results = op.execute({&logits, &weights, &labels}, {5.}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, lstmCell_test1) {

    const int batchSize = 2;
    const int inSize    = 10;
    const int numProj   = 4;
    const int numUnits  = 4;

    auto xt   = NDArrayFactory::create<double>('c', {batchSize, inSize});
    auto ht_1 = NDArrayFactory::create<double>('c', {batchSize, numProj});
    auto ct_1 = NDArrayFactory::create<double>('c', {batchSize, numUnits});
    auto Wx   = NDArrayFactory::create<double>('c', {inSize, 4*numUnits});
    auto Wh   = NDArrayFactory::create<double>('c', {numProj, 4*numUnits});
    auto Wc   = NDArrayFactory::create<double>('c', {3*numUnits});
    auto Wp   = NDArrayFactory::create<double>('c', {numUnits, numProj});
    auto b    = NDArrayFactory::create<double>('c', {4*numUnits});

    xt.assign(1.);
    ht_1.assign(2.);
    ct_1.assign(3.);
    Wx.assign(0.5);
    Wh.assign(0.5);
    Wc.assign(0.5);
    Wp.assign(0.5);
    b.assign(0.7);

    auto expHt = NDArrayFactory::create<double>('c', {batchSize, numProj}, {0.99926789,0.99926789,0.99926789,0.99926789,0.99926789,0.99926789,0.99926789,0.99926789});
    auto expCt = NDArrayFactory::create<double>('c', {batchSize, numUnits},{3.99987108,3.99987108,3.99987108,3.99987108,3.99987108,3.99987108,3.99987108,3.99987108});

    nd4j::ops::lstmCell op;
    auto results = op.execute({&xt, &ht_1, &ct_1, &Wx, &Wh, &Wc, &Wp, &b}, {0., 0., 1.}, {0, 0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *ht = results->at(0);
    auto *ct = results->at(1);

    ASSERT_TRUE(expHt.isSameShape(ht));
    ASSERT_TRUE(expHt.equalsTo(ht));
    ASSERT_TRUE(expCt.isSameShape(ct));
    ASSERT_TRUE(expCt.equalsTo(ct));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, lstmCell_test2) {

    const int batchSize = 2;
    const int inSize    = 10;
    const int numProj   = 4;
    const int numUnits  = 4;

    auto xt   = NDArrayFactory::create<double>('c', {batchSize, inSize});
    auto ht_1 = NDArrayFactory::create<double>('c', {batchSize, numProj});
    auto ct_1 = NDArrayFactory::create<double>('c', {batchSize, numUnits});
    auto Wx   = NDArrayFactory::create<double>('c', {inSize, 4*numUnits});
    auto Wh   = NDArrayFactory::create<double>('c', {numProj, 4*numUnits});
    auto Wc   = NDArrayFactory::create<double>('c', {3*numUnits});
    auto Wp   = NDArrayFactory::create<double>('c', {numUnits, numProj});
    auto b    = NDArrayFactory::create<double>('c', {4*numUnits});

    xt.assign(1.);
    ht_1.assign(2.);
    ct_1.assign(3.);
    Wx.assign(0.5);
    Wh.assign(0.5);
    Wc.assign(0.5);
    Wp.assign(0.5);
    b.assign(0.7);

    auto expHt = NDArrayFactory::create<double>('c', {batchSize, numProj}, {0.95867589,0.95867589,0.95867589,0.95867589,0.95867589,0.95867589,0.95867589,0.95867589});
    auto expCt = NDArrayFactory::create<double>('c', {batchSize, numUnits},{1.93001527,1.93001527,1.93001527,1.93001527, 1.93001527,1.93001527,1.93001527,1.93001527});

    nd4j::ops::lstmCell op;
    auto results = op.execute({&xt, &ht_1, &ct_1, &Wx, &Wh, &Wc, &Wp, &b}, {0., 0., -10.5}, {0, 0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *ht = results->at(0);
    auto *ct = results->at(1);

    ASSERT_TRUE(expHt.isSameShape(ht));
    ASSERT_TRUE(expHt.equalsTo(ht));
    ASSERT_TRUE(expCt.isSameShape(ct));
    ASSERT_TRUE(expCt.equalsTo(ct));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, lstmCell_test3) {

    const int batchSize = 2;
    const int inSize    = 10;
    const int numProj   = 4;
    const int numUnits  = 4;

    auto xt   = NDArrayFactory::create<double>('c', {batchSize, inSize});
    auto ht_1 = NDArrayFactory::create<double>('c', {batchSize, numProj});
    auto ct_1 = NDArrayFactory::create<double>('c', {batchSize, numUnits});
    auto Wx   = NDArrayFactory::create<double>('c', {inSize, 4*numUnits});
    auto Wh   = NDArrayFactory::create<double>('c', {numProj, 4*numUnits});
    auto Wc   = NDArrayFactory::create<double>('c', {3*numUnits});
    auto Wp   = NDArrayFactory::create<double>('c', {numUnits, numProj});
    auto b    = NDArrayFactory::create<double>('c', {4*numUnits});

    xt.assign(1.);
    ht_1.assign(2.);
    ct_1.assign(3.);
    Wx.assign(0.5);
    Wh.assign(0.5);
    Wc.assign(0.5);
    Wp.assign(0.5);
    b.assign(0.7);

    auto expHt = NDArrayFactory::create<double>('c', {batchSize, numProj}, {0.37992568,0.37992568,0.37992568,0.37992568,0.37992568,0.37992568,0.37992568,0.37992568});
    auto expCt = NDArrayFactory::create<double>('c', {batchSize, numUnits},{0.4,  0.4,  0.4,  0.4,   0.4,  0.4,  0.4,  0.4});

    nd4j::ops::lstmCell op;
    auto results = op.execute({&xt, &ht_1, &ct_1, &Wx, &Wh, &Wc, &Wp, &b}, {0.4, 0., 1.5}, {0, 0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *ht = results->at(0);
    auto *ct = results->at(1);

    ASSERT_TRUE(expHt.isSameShape(ht));
    ASSERT_TRUE(expHt.equalsTo(ht));
    ASSERT_TRUE(expCt.isSameShape(ct));
    ASSERT_TRUE(expCt.equalsTo(ct));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, lstmCell_test4) {

    const int batchSize = 2;
    const int inSize    = 10;
    const int numProj   = 4;
    const int numUnits  = 4;

    auto xt   = NDArrayFactory::create<double>('c', {batchSize, inSize});
    auto ht_1 = NDArrayFactory::create<double>('c', {batchSize, numProj});
    auto ct_1 = NDArrayFactory::create<double>('c', {batchSize, numUnits});
    auto Wx   = NDArrayFactory::create<double>('c', {inSize, 4*numUnits});
    auto Wh   = NDArrayFactory::create<double>('c', {numProj, 4*numUnits});
    auto Wc   = NDArrayFactory::create<double>('c', {3*numUnits});
    auto Wp   = NDArrayFactory::create<double>('c', {numUnits, numProj});
    auto b    = NDArrayFactory::create<double>('c', {4*numUnits});

    xt.assign(1.);
    ht_1.assign(2.);
    ct_1.assign(3.);
    Wx.assign(0.5);
    Wh.assign(0.5);
    Wc.assign(0.5);
    Wp.assign(0.5);
    b.assign(0.7);

    auto expHt = NDArrayFactory::create<double>('c', {batchSize, numProj}, {0.37992568,0.37992568,0.37992568,0.37992568,0.37992568,0.37992568,0.37992568,0.37992568});
    auto expCt = NDArrayFactory::create<double>('c', {batchSize, numUnits},{0.4,  0.4,  0.4,  0.4,   0.4,  0.4,  0.4,  0.4});

    nd4j::ops::lstmCell op;
    auto results = op.execute({&xt, &ht_1, &ct_1, &Wx, &Wh, &Wc, &Wp, &b}, {0.4, 0.3, 1.5}, {0, 0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *ht = results->at(0);
    auto *ct = results->at(1);

    ASSERT_TRUE(expHt.isSameShape(ht));
    ASSERT_TRUE(expHt.equalsTo(ht));
    ASSERT_TRUE(expCt.isSameShape(ct));
    ASSERT_TRUE(expCt.equalsTo(ct));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, lstmCell_test5) {

    const int batchSize = 2;
    const int inSize    = 10;
    const int numProj   = 3;
    const int numUnits  = 4;

    auto xt   = NDArrayFactory::create<double>('c', {batchSize, inSize});
    auto ht_1 = NDArrayFactory::create<double>('c', {batchSize, numProj});
    auto ct_1 = NDArrayFactory::create<double>('c', {batchSize, numUnits});
    auto Wx   = NDArrayFactory::create<double>('c', {inSize, 4*numUnits});
    auto Wh   = NDArrayFactory::create<double>('c', {numProj, 4*numUnits});
    auto Wc   = NDArrayFactory::create<double>('c', {3*numUnits});
    auto Wp   = NDArrayFactory::create<double>('c', {numUnits, numProj});
    auto b    = NDArrayFactory::create<double>('c', {4*numUnits});

    xt.assign(1.);
    ht_1.assign(2.);
    ct_1.assign(3.);
    Wx.assign(0.5);
    Wh.assign(0.5);
    Wc.assign(0.5);
    Wp.assign(0.5);
    b.assign(0.7);

    auto expHt = NDArrayFactory::create<double>('c', {batchSize, numProj}, {0.3,0.3,0.3,0.3,0.3,0.3});
    auto expCt = NDArrayFactory::create<double>('c', {batchSize, numUnits},{0.4,  0.4,  0.4,  0.4,   0.4,  0.4,  0.4,  0.4});

    nd4j::ops::lstmCell op;
    auto results = op.execute({&xt, &ht_1, &ct_1, &Wx, &Wh, &Wc, &Wp, &b}, {0.4, 0.3, 1.5}, {0, 1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *ht = results->at(0);
    auto *ct = results->at(1);

    ASSERT_TRUE(expHt.isSameShape(ht));
    ASSERT_TRUE(expHt.equalsTo(ht));
    ASSERT_TRUE(expCt.isSameShape(ct));
    ASSERT_TRUE(expCt.equalsTo(ct));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, lstmCell_test6) {

    const int batchSize = 2;
    const int inSize    = 10;
    const int numProj   = 3;
    const int numUnits  = 4;

    auto xt   = NDArrayFactory::create<double>('c', {batchSize, inSize});
    auto ht_1 = NDArrayFactory::create<double>('c', {batchSize, numProj});
    auto ct_1 = NDArrayFactory::create<double>('c', {batchSize, numUnits});
    auto Wx   = NDArrayFactory::create<double>('c', {inSize, 4*numUnits});
    auto Wh   = NDArrayFactory::create<double>('c', {numProj, 4*numUnits});
    auto Wc   = NDArrayFactory::create<double>('c', {3*numUnits});
    auto Wp   = NDArrayFactory::create<double>('c', {numUnits, numProj});
    auto b    = NDArrayFactory::create<double>('c', {4*numUnits});

    xt.assign(1.);
    ht_1.assign(2.);
    ct_1.assign(3.);
    Wx.assign(0.5);
    Wh.assign(0.5);
    Wc.assign(0.5);
    Wp.assign(0.5);
    b.assign(0.7);

    auto expHt = NDArrayFactory::create<double>('c', {batchSize, numProj}, {1.99832496,1.99832496,1.99832496,1.99832496,1.99832496,1.99832496});
    auto expCt = NDArrayFactory::create<double>('c', {batchSize, numUnits},{3.99972188,3.99972188,3.99972188,3.99972188,3.99972188,3.99972188,3.99972188,3.99972188});

    nd4j::ops::lstmCell op;
    auto results = op.execute({&xt, &ht_1, &ct_1, &Wx, &Wh, &Wc, &Wp, &b}, {0., 0., 1.5}, {0, 1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *ht = results->at(0);
    auto *ct = results->at(1);

    ASSERT_TRUE(expHt.isSameShape(ht));
    ASSERT_TRUE(expHt.equalsTo(ht));
    ASSERT_TRUE(expCt.isSameShape(ct));
    ASSERT_TRUE(expCt.equalsTo(ct));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, lstmCell_test7) {

    const int batchSize = 2;
    const int inSize    = 10;
    const int numProj   = 3;
    const int numUnits  = 4;

    auto xt   = NDArrayFactory::create<double>('c', {batchSize, inSize});
    auto ht_1 = NDArrayFactory::create<double>('c', {batchSize, numProj});
    auto ct_1 = NDArrayFactory::create<double>('c', {batchSize, numUnits});
    auto Wx   = NDArrayFactory::create<double>('c', {inSize, 4*numUnits});
    auto Wh   = NDArrayFactory::create<double>('c', {numProj, 4*numUnits});
    auto Wc   = NDArrayFactory::create<double>('c', {3*numUnits});
    auto Wp   = NDArrayFactory::create<double>('c', {numUnits, numProj});
    auto b    = NDArrayFactory::create<double>('c', {4*numUnits});

    xt.assign(1.);
    ht_1.assign(2.);
    ct_1.assign(3.);
    Wx.assign(0.5);
    Wh.assign(0.5);
    Wc.assign(0.5);
    Wp.assign(0.5);
    b.assign(0.7);

    auto expHt = NDArrayFactory::create<double>('c', {batchSize, numProj}, {0.75977136,0.75977136,0.75977136,0.75977136,0.75977136,0.75977136});
    auto expCt = NDArrayFactory::create<double>('c', {batchSize, numUnits},{0.4,  0.4,  0.4,  0.4,   0.4,  0.4,  0.4,  0.4});

    nd4j::ops::lstmCell op;
    auto results = op.execute({&xt, &ht_1, &ct_1, &Wx, &Wh, &Wc, &Wp, &b}, {0.4, 0., 1.5}, {0, 1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *ht = results->at(0);
    auto *ct = results->at(1);

    ASSERT_TRUE(expHt.isSameShape(ht));
    ASSERT_TRUE(expHt.equalsTo(ht));
    ASSERT_TRUE(expCt.isSameShape(ct));
    ASSERT_TRUE(expCt.equalsTo(ct));

    delete results;
}


///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, lstmCell_test8) {

    const int batchSize = 2;
    const int inSize    = 10;
    const int numProj   = 4;
    const int numUnits  = 4;

    auto xt   = NDArrayFactory::create<double>('c', {batchSize, inSize});
    auto ht_1 = NDArrayFactory::create<double>('c', {batchSize, numProj});
    auto ct_1 = NDArrayFactory::create<double>('c', {batchSize, numUnits});
    auto Wx   = NDArrayFactory::create<double>('c', {inSize, 4*numUnits});
    auto Wh   = NDArrayFactory::create<double>('c', {numProj, 4*numUnits});
    auto Wc   = NDArrayFactory::create<double>('c', {3*numUnits});
    auto Wp   = NDArrayFactory::create<double>('c', {numUnits, numProj});
    auto b    = NDArrayFactory::create<double>('c', {4*numUnits});

    xt.assign(1.);
    ht_1.assign(2.);
    ct_1.assign(3.);
    Wx.assign(0.5);
    Wh.assign(0.5);
    Wc.assign(0.5);
    Wp.assign(0.5);
    b.assign(0.7);

    auto expHt = NDArrayFactory::create<double>('c', {batchSize, numProj}, {0.99930672,0.99930672,0.99930672,0.99930672, 0.99930672,0.99930672,0.99930672,0.99930672});
    auto expCt = NDArrayFactory::create<double>('c', {batchSize, numUnits},{3.99996277,3.99996277,3.99996277,3.99996277,3.99996277,3.99996277,3.99996277,3.99996277});

    nd4j::ops::lstmCell op;
    auto results = op.execute({&xt, &ht_1, &ct_1, &Wx, &Wh, &Wc, &Wp, &b}, {0., 0., 10.5}, {1, 0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *ht = results->at(0);
    auto *ct = results->at(1);

    ASSERT_TRUE(expHt.isSameShape(ht));
    ASSERT_TRUE(expHt.equalsTo(ht,1e-4));
    ASSERT_TRUE(expCt.isSameShape(ct));
    ASSERT_TRUE(expCt.equalsTo(ct,1e-4));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, lstmCell_test9) {

    const int batchSize = 2;
    const int inSize    = 10;
    const int numProj   = 4;
    const int numUnits  = 4;

    auto xt   = NDArrayFactory::create<double>('c', {batchSize, inSize});
    auto ht_1 = NDArrayFactory::create<double>('c', {batchSize, numProj});
    auto ct_1 = NDArrayFactory::create<double>('c', {batchSize, numUnits});
    auto Wx   = NDArrayFactory::create<double>('c', {inSize, 4*numUnits});
    auto Wh   = NDArrayFactory::create<double>('c', {numProj, 4*numUnits});
    auto Wc   = NDArrayFactory::create<double>('c', {3*numUnits});
    auto Wp   = NDArrayFactory::create<double>('c', {numUnits, numProj});
    auto b    = NDArrayFactory::create<double>('c', {4*numUnits});

    xt.assign(1.);
    ht_1.assign(2.);
    ct_1.assign(3.);
    Wx.assign(0.5);
    Wh.assign(0.5);
    Wc.assign(0.5);
    Wp.assign(0.5);
    b.assign(0.7);

    auto expHt = NDArrayFactory::create<double>('c', {batchSize, numProj}, {0.99501777,0.99501777,0.99501777,0.99501777,0.99501777,0.99501777,0.99501777,0.99501777});
    auto expCt = NDArrayFactory::create<double>('c', {batchSize, numUnits},{3.,3.,3.,3.,3.,3.,3.,3.});

    nd4j::ops::lstmCell op;
    auto results = op.execute({&xt, &ht_1, &ct_1, &Wx, &Wh, &Wc, &Wp, &b}, {3., 0., 10.5}, {1, 0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *ht = results->at(0);
    auto *ct = results->at(1);

    ASSERT_TRUE(expHt.isSameShape(ht));
    ASSERT_TRUE(expHt.equalsTo(ht,1e-4));
    ASSERT_TRUE(expCt.isSameShape(ct));
    ASSERT_TRUE(expCt.equalsTo(ct));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, lstmCell_test10) {

    const int batchSize = 2;
    const int inSize    = 10;
    const int numProj   = 3;
    const int numUnits  = 4;

    auto xt   = NDArrayFactory::create<double>('c', {batchSize, inSize});
    auto ht_1 = NDArrayFactory::create<double>('c', {batchSize, numProj});
    auto ct_1 = NDArrayFactory::create<double>('c', {batchSize, numUnits});
    auto Wx   = NDArrayFactory::create<double>('c', {inSize, 4*numUnits});
    auto Wh   = NDArrayFactory::create<double>('c', {numProj, 4*numUnits});
    auto Wc   = NDArrayFactory::create<double>('c', {3*numUnits});
    auto Wp   = NDArrayFactory::create<double>('c', {numUnits, numProj});
    auto b    = NDArrayFactory::create<double>('c', {4*numUnits});

    xt.assign(1.);
    ht_1.assign(2.);
    ct_1.assign(3.);
    Wx.assign(0.5);
    Wh.assign(0.5);
    Wc.assign(0.5);
    Wp.assign(0.5);
    b.assign(0.7);

    auto expHt = NDArrayFactory::create<double>('c', {batchSize, numProj}, {1.99861344,1.99861344,1.99861344,1.99861344,1.99861344,1.99861344});
    auto expCt = NDArrayFactory::create<double>('c', {batchSize, numUnits},{3.99996277,  3.99996277,  3.99996277,  3.99996277,3.99996277,  3.99996277,  3.99996277,  3.99996277});

    nd4j::ops::lstmCell op;
    auto results = op.execute({&xt, &ht_1, &ct_1, &Wx, &Wh, &Wc, &Wp, &b}, {0., 0., 10.5}, {1, 1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *ht = results->at(0);
    auto *ct = results->at(1);

    ASSERT_TRUE(expHt.isSameShape(ht));
    ASSERT_TRUE(expHt.equalsTo(ht));
    ASSERT_TRUE(expCt.isSameShape(ct));
    ASSERT_TRUE(expCt.equalsTo(ct));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, lstmCell_test11) {

    const int batchSize = 2;
    const int inSize    = 10;
    const int numProj   = 3;
    const int numUnits  = 4;

    auto xt   = NDArrayFactory::create<double>('c', {batchSize, inSize});
    auto ht_1 = NDArrayFactory::create<double>('c', {batchSize, numProj});
    auto ct_1 = NDArrayFactory::create<double>('c', {batchSize, numUnits});
    auto Wx   = NDArrayFactory::create<double>('c', {inSize, 4*numUnits});
    auto Wh   = NDArrayFactory::create<double>('c', {numProj, 4*numUnits});
    auto Wc   = NDArrayFactory::create<double>('c', {3*numUnits});
    auto Wp   = NDArrayFactory::create<double>('c', {numUnits, numProj});
    auto b    = NDArrayFactory::create<double>('c', {4*numUnits});

    xt.assign(1.);
    ht_1.assign(2.);
    ct_1.assign(3.);
    Wx.assign(0.5);
    Wh.assign(0.5);
    Wc.assign(0.5);
    Wp.assign(0.5);
    b.assign(0.7);

    auto expHt = NDArrayFactory::create<double>('c', {batchSize, numProj}, {1.99003554,1.99003554,1.99003554,1.99003554,1.99003554,1.99003554});
    auto expCt = NDArrayFactory::create<double>('c', {batchSize, numUnits},{3.,3.,3.,3.,3.,3.,3.,3.});

    nd4j::ops::lstmCell op;
    auto results = op.execute({&xt, &ht_1, &ct_1, &Wx, &Wh, &Wc, &Wp, &b}, {3., 0., 10.5}, {1, 1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *ht = results->at(0);
    auto *ct = results->at(1);

    ASSERT_TRUE(expHt.isSameShape(ht));
    ASSERT_TRUE(expHt.equalsTo(ht));
    ASSERT_TRUE(expCt.isSameShape(ct));
    ASSERT_TRUE(expCt.equalsTo(ct));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, lstmCell_test12) {

    const int batchSize = 2;
    const int inSize    = 10;
    const int numProj   = 3;
    const int numUnits  = 4;

    auto xt   = NDArrayFactory::create<double>('c', {batchSize, inSize});
    auto ht_1 = NDArrayFactory::create<double>('c', {batchSize, numProj});
    auto ct_1 = NDArrayFactory::create<double>('c', {batchSize, numUnits});
    auto Wx   = NDArrayFactory::create<double>('c', {inSize, 4*numUnits});
    auto Wh   = NDArrayFactory::create<double>('c', {numProj, 4*numUnits});
    auto Wc   = NDArrayFactory::create<double>('c', {3*numUnits});
    auto Wp   = NDArrayFactory::create<double>('c', {numUnits, numProj});
    auto b    = NDArrayFactory::create<double>('c', {4*numUnits});

    xt.assign(1.);
    ht_1.assign(2.);
    ct_1.assign(3.);
    Wx.assign(0.5);
    Wh.assign(0.5);
    Wc.assign(0.5);
    Wp.assign(0.5);
    b.assign(0.7);

    auto expHt = NDArrayFactory::create<double>('c', {batchSize, numProj}, {1.,1.,1.,1.,1.,1.});
    auto expCt = NDArrayFactory::create<double>('c', {batchSize, numUnits},{3.,3.,3.,3.,3.,3.,3.,3.});

    nd4j::ops::lstmCell op;
    auto results = op.execute({&xt, &ht_1, &ct_1, &Wx, &Wh, &Wc, &Wp, &b}, {3., 1.,-5.}, {1, 1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *ht = results->at(0);
    auto *ct = results->at(1);

    ASSERT_TRUE(expHt.isSameShape(ht));
    ASSERT_TRUE(expHt.equalsTo(ht));
    ASSERT_TRUE(expCt.isSameShape(ct));
    ASSERT_TRUE(expCt.equalsTo(ct));

    delete results;
}
