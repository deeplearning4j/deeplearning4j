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
// Created by raver119 on 31.10.2017.
//

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <NDArray.h>
#include <NativeOps.h>

using namespace nd4j;
using namespace nd4j::graph;

class IndexingTests : public testing::Test {
public:

};

TEST_F(IndexingTests, StridedSlice_1) {
    auto x = NDArrayFactory::create<float>('c', {3, 3, 3});
    auto exp = NDArrayFactory::create<float>('c', {1, 1, 3});
    exp.p(0, 25.f);
    exp.p(1, 26.f);
    exp.p(2, 27.f);

    x.linspace(1);

    //nd4j_debug("print x->rankOf(): %i", x.rankOf());

    /*
    auto tads = x.allTensorsAlongDimension({0});
    nd4j_debug("numTads: %i\n", tads->size());
    for (int e = 0; e < tads->size(); e++)
        tads->at(e)->assign((float) e);
    */

    nd4j::ops::strided_slice op;

    auto result = op.execute({&x}, {}, {0,0,0,0,0, 2,2,0,  3,3,3,  1,1,1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(IndexingTests, StridedSlice_2) {
    auto x = NDArrayFactory::create<float>('c', {5, 5, 5});

    float _expB[] = {86.f,   87.f,   88.f,  91.f,   92.f,   93.f,  96.f,   97.f,   98.f,   111.f,  112.f,  113.f,  116.f,  117.f,  118.f,  121.f,  122.f,  123.f,};
    auto exp = NDArrayFactory::create<float>('c', {2, 3, 3});
    exp.setBuffer(_expB);

    x.linspace(1);

    nd4j::ops::strided_slice op;

    auto result = op.execute({&x}, {}, {0,0,0,0,0, 3,2,0,  5,5,3,  1,1,1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(IndexingTests, StridedSlice_3) {
    auto x = NDArrayFactory::create<float>('c', {5, 5, 5});

    float _expB[] = {86.f, 88.f,  91.f, 93.f, 96.f, 98.f, 111.f,  113.f,  116.f, 118.f,  121.f,  123.f,};
    auto exp = NDArrayFactory::create<float>('c', {2, 3, 2});
    exp.setBuffer(_expB);

    x.linspace(1);

    nd4j::ops::strided_slice op;

    auto result = op.execute({&x}, {}, {0,0,0,0,0, 3,2,0,  5,5,3,  1,1,2});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(IndexingTests, SimpleSlice_1) {
    float _inB[] = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6};
    auto input = NDArrayFactory::create<float>('c', {3, 2, 3});
    input.setBuffer(_inB);

    auto exp = NDArrayFactory::create<float>('c', {1, 1, 3});
    exp.p(0, 3.0f);
    exp.p(1, 3.0f);
    exp.p(2, 3.0f);

    nd4j::ops::slice op;

    auto result = op.execute({&input}, {}, {1,0,0, 1,1,3});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(IndexingTests, SimpleSlice_2) {
    float _inB[] = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6};
    auto input = NDArrayFactory::create<float>('c', {3, 2, 3});
    input.setBuffer(_inB);

    auto exp = NDArrayFactory::create<float>('c', {1, 2, 3});
    exp.p(0, 3.0f);
    exp.p(1, 3.0f);
    exp.p(2, 3.0f);
    exp.p(3, 4.0f);
    exp.p(4, 4.0f);
    exp.p(5, 4.0f);

    nd4j::ops::slice op;

    auto result = op.execute({&input}, {}, {1,0,0, 1,2,3});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(IndexingTests, SimpleSlice_3) {
    float _inB[] = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6};
    auto input = NDArrayFactory::create<float>('c', {3, 2, 3});
    input.setBuffer(_inB);

    auto exp = NDArrayFactory::create<float>('c', {2, 1, 3});
    exp.p(0, 3.0f);
    exp.p(1, 3.0f);
    exp.p(2, 3.0f);
    exp.p(3, 5.0f);
    exp.p(4, 5.0f);
    exp.p(5, 5.0f);

    nd4j::ops::slice op;

    auto result = op.execute({&input}, {}, {1,0,0, 2,1,3});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(IndexingTests, SimpleSlice_4) {
    auto input = NDArrayFactory::create<double>('c', {3, 2, 3}, {1.0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
    auto start = NDArrayFactory::create<double>('c', {3}, {1.0, 0.0, 0.0});
    auto stop = NDArrayFactory::create<double>('c', {3}, {2.0, 1.0, 3.0});
    auto exp = NDArrayFactory::create<double>('c', {2, 1, 3}, {3.0, 3.0, 3.0, 5.0, 5.0, 5.0});

    nd4j::ops::slice op;

    auto result = op.execute({&input, &start, &stop}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(IndexingTests, MaskedSlice_0) {
    auto matrix = NDArrayFactory::create<float>('c', {3, 5});
    auto tads = matrix.allTensorsAlongDimension({1});
    for (int e = 0; e < tads->size(); e++) {
        tads->at(e)->assign((float) (e+1));
    }

    auto exp = NDArrayFactory::create<float>('c', {1, 5});
    exp.assign(2.0f);

    nd4j::ops::strided_slice op;
    auto result = op.execute({&matrix}, {}, {0,0,0,0,0,   1, 2, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    // z->printShapeInfo("z");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
    delete tads;
}


TEST_F(IndexingTests, MaskedSlice_00) {
    auto matrix = NDArrayFactory::create<float>('c', {3, 5});
    auto tads = matrix.allTensorsAlongDimension({1});
    for (int e = 0; e < tads->size(); e++) {
        tads->at(e)->assign((float) (e+1));
    }

    auto exp = NDArrayFactory::create<float>('c', {1, 2}, {2, 2});


    nd4j::ops::strided_slice op;
    auto result = op.execute({&matrix}, {}, {0,0,0,0,0,   1, 1, 2, 3, 1, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    // z->printShapeInfo("z");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
    delete tads;
}


TEST_F(IndexingTests, MaskedSlice_1) {
    auto matrix = NDArrayFactory::create<float>('c', {3, 5});
    auto tads = matrix.allTensorsAlongDimension({1});
    for (int e = 0; e < tads->size(); e++) {
        tads->at(e)->assign((float) (e+1));
    }

    auto exp = NDArrayFactory::create<float>('c', {5});
    exp.assign(2.0f);

    nd4j::ops::strided_slice op;
    auto result = op.execute({&matrix}, {}, {0,0,0,0,1,   1, 2, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    // z->printShapeInfo("z");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
    delete tads;
}

TEST_F(IndexingTests, MaskedSlice_2) {
    float _buff[] = {1.f, 1.2f, 1.3f, 2.f, 2.2f, 2.3f, 3.f, 3.2f, 3.3f, 4.f, 4.2f, 4.3f, 5.f,  5.2f, 5.3f, 6.f,   6.2f,  6.3f,  7.f,   7.2f,  7.3f,  8.f,   8.2f,  8.3f,  9.f,   9.2f,  9.3f};
    auto matrix = NDArrayFactory::create<float>('c', {3, 3, 3});
    matrix.setBuffer(_buff);

    float _expB[] = {4.000000f, 4.200000f, 4.300000f, 5.000000f, 5.200000f, 5.300000f, 6.000000f, 6.200000f, 6.300000f};
    auto exp = NDArrayFactory::create<float>('c', {3, 3});
    exp.setBuffer(_expB);

    // output = tf.strided_slice(a, [1, 0, 0], [3, 3, 3], shrink_axis_mask=5)
    nd4j::ops::strided_slice op;
    auto result = op.execute({&matrix}, {}, {0,0,0,0,1,   1, 0, 0,  3, 3, 3,  1, 1, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(IndexingTests, MaskedSlice_3) {
    float _buff[] = {1.f, 1.2f, 1.3f, 2.f, 2.2f, 2.3f, 3.f, 3.2f, 3.3f, 4.f, 4.2f, 4.3f, 5.f,  5.2f, 5.3f, 6.f,   6.2f,  6.3f,  7.f,   7.2f,  7.3f,  8.f,   8.2f,  8.3f,  9.f,   9.2f,  9.3f};
    auto matrix = NDArrayFactory::create<float>('c', {3, 3, 3});
    matrix.setBuffer(_buff);

    float _expB[] = { 4.f,   4.2f,  4.3f, 7.f, 7.2f,  7.3f};
    auto exp = NDArrayFactory::create<float>('c', {2, 3});
    exp.setBuffer(_expB);

    // output = tf.strided_slice(a, [1, 0, 0], [3, 3, 3], shrink_axis_mask=5)
    nd4j::ops::strided_slice op;
    auto result = op.execute({&matrix}, {}, {0,0,0,0,2,   1, 0, 0,  3, 3, 3,  1, 1, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(IndexingTests, MaskedSlice_4) {
    float _buff[] = {1.f, 1.2f, 1.3f, 2.f, 2.2f, 2.3f, 3.f, 3.2f, 3.3f, 4.f, 4.2f, 4.3f, 5.f,  5.2f, 5.3f, 6.f,   6.2f,  6.3f,  7.f,   7.2f,  7.3f,  8.f,   8.2f,  8.3f,  9.f,   9.2f,  9.3f};
    auto matrix = NDArrayFactory::create<float>('c', {3, 3, 3});
    matrix.setBuffer(_buff);

    float _expB[] = { 4.f,   4.2f,  4.3f};
    auto exp = NDArrayFactory::create<float>('c', {3});
    exp.setBuffer(_expB);

    // output = tf.strided_slice(a, [1, 0, 0], [3, 3, 3], shrink_axis_mask=5)
    nd4j::ops::strided_slice op;
    auto result = op.execute({&matrix}, {}, {0,0,0,0, 3,   1, 0, 0,  3, 3, 3,  1, 1, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);


    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(IndexingTests, Live_Slice_1) {
    float _buff[] = {1.f, 1.2f, 1.3f, 2.f, 2.2f, 2.3f, 3.f, 3.2f, 3.3f, 4.f, 4.2f, 4.3f, 5.f,  5.2f, 5.3f, 6.f,   6.2f,  6.3f,  7.f,   7.2f,  7.3f,  8.f,   8.2f,  8.3f,  9.f,   9.2f,  9.3f};
    auto matrix = NDArrayFactory::create<float>('c', {3, 3, 3});
    matrix.setBuffer(_buff);

    float _expB[] = { 4.f,   4.2f,  4.3f};
    auto exp = NDArrayFactory::create<float>('c', {3});
    exp.setBuffer(_expB);

    auto begin = NDArrayFactory::create<float>('c', {1, 3}, {1.0f, 0.0f, 0.0f});
    auto end = NDArrayFactory::create<float>('c', {1, 3}, {3.0f, 3.0f, 3.0f});
    auto stride = NDArrayFactory::create<float>('c', {1, 3}, {1.0f, 1.0f, 1.0f});

    // output = tf.strided_slice(a, [1, 0, 0], [3, 3, 3], shrink_axis_mask=5)
    nd4j::ops::strided_slice op;
    auto result = op.execute({&matrix, &begin, &end, &stride}, {}, {0,0,0,0,3});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    // z->printShapeInfo("z shape");
    
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(IndexingTests, Test_StridedSlice_1) {
    auto x = NDArrayFactory::create<float>('c', {1, 2}, {5.f, 2.f});
    auto a = NDArrayFactory::create<float>('c', {1, 1}, {0.f});
    auto b = NDArrayFactory::create<float>('c', {1, 1}, {1.f});
    auto c = NDArrayFactory::create<float>('c', {1, 1}, {1.f});
    auto exp = NDArrayFactory::create<float>(5.0f);

    nd4j::ops::strided_slice op;
    auto result = op.execute({&x, &a, &b, &c}, {}, {0, 0, 0, 0, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    // z->printIndexedBuffer("Z");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(IndexingTests, Test_StridedSlice_2) {
    auto x = NDArrayFactory::create<float>('c', {2, 3}, {1, 2, 3, 4, 5, 6});
    auto a = NDArrayFactory::create<float>('c', {1, 2}, {1, 1});
    auto b = NDArrayFactory::create<float>('c', {1, 2}, {2, 2});
    auto c = NDArrayFactory::create<float>('c', {1, 2}, {1, 1});
    auto exp = NDArrayFactory::create<float>('c', {1}, {5.0});

    nd4j::ops::strided_slice op;
    auto result = op.execute({&x, &a, &b, &c}, {}, {0, 0, 0, 0, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    // z->printIndexedBuffer("Z");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(IndexingTests, Test_StridedSlice_3) {
    auto x = NDArrayFactory::create<float>('c', {2, 3}, {1, 2, 3, 4, 5, 6});
    auto a = NDArrayFactory::create<float>('c', {1, 2}, {1, 2});
    auto b = NDArrayFactory::create<float>('c', {1, 2}, {2, 3});
    auto c = NDArrayFactory::create<float>('c', {1, 2}, {1, 1});
    auto exp = NDArrayFactory::create<float>('c', {1}, {6.0});

    nd4j::ops::strided_slice op;
    auto result = op.execute({&x, &a, &b, &c}, {}, {0, 0, 0, 0, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    //z->printIndexedBuffer("Z");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(IndexingTests, Test_StridedSlice_4) {
    auto x = NDArrayFactory::create<float>('c', {1, 2}, {5, 2});
    auto a = NDArrayFactory::create<float>('c', {1, 1}, {0});
    auto b = NDArrayFactory::create<float>('c', {1, 1}, {1});
    auto c = NDArrayFactory::create<float>('c', {1, 1}, {1});
    auto exp = NDArrayFactory::create<float>(5.0f);

    nd4j::ops::strided_slice op;
    auto result = op.execute({&x, &a, &b, &c}, {}, {0, 0, 0, 0, 1, 0, 1, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    //z->printIndexedBuffer("Z");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(IndexingTests, Test_Subarray_Strided_1) {
    auto x = NDArrayFactory::create<float>('c', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto exp = NDArrayFactory::create<float>('c', {3, 2}, {1, 3, 4, 6, 7, 9});
    IndicesList indices({NDIndex::all(), NDIndex::interval(0, 3, 2)});
    auto sub = x.subarray(indices);

    //sub->printShapeInfo("sub shape");
    //sub->printIndexedBuffer("sub buffr");

    ASSERT_TRUE(exp.isSameShape(sub));
    ASSERT_TRUE(exp.equalsTo(sub));

    delete sub;
}


/*
TEST_F(IndexingTests, MaskedSlice_5) {
    float _buff[] = {1.f, 1.2f, 1.3f, 2.f, 2.2f, 2.3f, 3.f, 3.2f, 3.3f, 4.f, 4.2f, 4.3f, 5.f,  5.2f, 5.3f, 6.f,   6.2f,  6.3f,  7.f,   7.2f,  7.3f,  8.f,   8.2f,  8.3f,  9.f,   9.2f,  9.3f};
    auto matrix('c', {3, 3, 3});
    matrix.setBuffer(_buff);

    float _expB[] = { 4.f,   4.2f,  4.3f, 7.f, 7.2f,  7.3f};
    auto exp('c', {2, 3});
    exp.setBuffer(_expB);

    // output = tf.strided_slice(a, [1, 0, 0], [3, 3, 3], shrink_axis_mask=5)
    nd4j::ops::strided_slice<float> op;
    auto result = op.execute({&matrix}, {}, {0,0,0,0,2,   1, 0, 0,  3, 3, 3});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}
*/