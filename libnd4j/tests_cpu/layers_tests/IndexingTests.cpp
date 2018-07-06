//
// Created by raver119 on 31.10.2017.
//

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <NDArray.h>
#include <NDArrayFactory.h>
#include <NativeOps.h>

using namespace nd4j;
using namespace nd4j::graph;

class IndexingTests : public testing::Test {
public:

};

TEST_F(IndexingTests, StridedSlice_1) {
    NDArray<float> x('c', {3, 3, 3});
    NDArray<float> exp('c', {1, 1, 3});
    exp.putScalar(0, 25.f);
    exp.putScalar(1, 26.f);
    exp.putScalar(2, 27.f);

    x.linspace(1);

    //nd4j_debug("print x->rankOf(): %i", x.rankOf());

    /*
    auto tads = NDArrayFactory<float>::allTensorsAlongDimension(&x, {0});
    nd4j_debug("numTads: %i\n", tads->size());
    for (int e = 0; e < tads->size(); e++)
        tads->at(e)->assign((float) e);
    */

    nd4j::ops::strided_slice<float> op;

    auto result = op.execute({&x}, {}, {0,0,0,0,0, 2,2,0,  3,3,3,  1,1,1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(IndexingTests, StridedSlice_2) {
    NDArray<float> x('c', {5, 5, 5});

    float _expB[] = {86.f,   87.f,   88.f,  91.f,   92.f,   93.f,  96.f,   97.f,   98.f,   111.f,  112.f,  113.f,  116.f,  117.f,  118.f,  121.f,  122.f,  123.f,};
    NDArray<float> exp('c', {2, 3, 3});
    exp.setBuffer(_expB);

    x.linspace(1);

    nd4j::ops::strided_slice<float> op;

    auto result = op.execute({&x}, {}, {0,0,0,0,0, 3,2,0,  5,5,3,  1,1,1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(IndexingTests, StridedSlice_3) {
    NDArray<float> x('c', {5, 5, 5});

    float _expB[] = {86.f, 88.f,  91.f, 93.f, 96.f, 98.f, 111.f,  113.f,  116.f, 118.f,  121.f,  123.f,};
    NDArray<float> exp('c', {2, 3, 2});
    exp.setBuffer(_expB);

    x.linspace(1);

    nd4j::ops::strided_slice<float> op;

    auto result = op.execute({&x}, {}, {0,0,0,0,0, 3,2,0,  5,5,3,  1,1,2});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(IndexingTests, SimpleSlice_1) {
    float _inB[] = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6};
    NDArray<float> input('c', {3, 2, 3});
    input.setBuffer(_inB);

    NDArray<float> exp('c', {1, 1, 3});
    exp.putScalar(0, 3.0f);
    exp.putScalar(1, 3.0f);
    exp.putScalar(2, 3.0f);

    nd4j::ops::slice<float> op;

    auto result = op.execute({&input}, {}, {1,0,0, 1,1,3});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(IndexingTests, SimpleSlice_2) {
    float _inB[] = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6};
    NDArray<float> input('c', {3, 2, 3});
    input.setBuffer(_inB);

    NDArray<float> exp('c', {1, 2, 3});
    exp.putScalar(0, 3.0f);
    exp.putScalar(1, 3.0f);
    exp.putScalar(2, 3.0f);
    exp.putScalar(3, 4.0f);
    exp.putScalar(4, 4.0f);
    exp.putScalar(5, 4.0f);

    nd4j::ops::slice<float> op;

    auto result = op.execute({&input}, {}, {1,0,0, 1,2,3});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(IndexingTests, SimpleSlice_3) {
    float _inB[] = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6};
    NDArray<float> input('c', {3, 2, 3});
    input.setBuffer(_inB);

    NDArray<float> exp('c', {2, 1, 3});
    exp.putScalar(0, 3.0f);
    exp.putScalar(1, 3.0f);
    exp.putScalar(2, 3.0f);
    exp.putScalar(3, 5.0f);
    exp.putScalar(4, 5.0f);
    exp.putScalar(5, 5.0f);

    nd4j::ops::slice<float> op;

    auto result = op.execute({&input}, {}, {1,0,0, 2,1,3});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(IndexingTests, SimpleSlice_4) {
    NDArray<double> input('c', {3, 2, 3}, {1.0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
    NDArray<double> start('c', {3}, {1.0, 0.0, 0.0});
    NDArray<double> stop('c', {3}, {2.0, 1.0, 3.0});
    NDArray<double> exp('c', {2, 1, 3}, {3.0, 3.0, 3.0, 5.0, 5.0, 5.0});

    nd4j::ops::slice<double> op;

    auto result = op.execute({&input, &start, &stop}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(IndexingTests, MaskedSlice_0) {
    NDArray<float> matrix('c', {3, 5});
    auto tads = NDArrayFactory<float>::allTensorsAlongDimension(&matrix, {1});
    for (int e = 0; e < tads->size(); e++) {
        tads->at(e)->assign((float) (e+1));
    }

    NDArray<float> exp('c', {1, 5});
    exp.assign(2.0f);

    nd4j::ops::strided_slice<float> op;
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
    NDArray<float> matrix('c', {3, 5});
    auto tads = NDArrayFactory<float>::allTensorsAlongDimension(&matrix, {1});
    for (int e = 0; e < tads->size(); e++) {
        tads->at(e)->assign((float) (e+1));
    }

    NDArray<float> exp('c', {1, 2}, {2, 2});


    nd4j::ops::strided_slice<float> op;
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
    NDArray<float> matrix('c', {3, 5});
    auto tads = NDArrayFactory<float>::allTensorsAlongDimension(&matrix, {1});
    for (int e = 0; e < tads->size(); e++) {
        tads->at(e)->assign((float) (e+1));
    }

    NDArray<float> exp('c', {5});
    exp.assign(2.0f);

    nd4j::ops::strided_slice<float> op;
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
    NDArray<float> matrix('c', {3, 3, 3});
    matrix.setBuffer(_buff);

    float _expB[] = {4.000000f, 4.200000f, 4.300000f, 5.000000f, 5.200000f, 5.300000f, 6.000000f, 6.200000f, 6.300000f};
    NDArray<float> exp('c', {3, 3});
    exp.setBuffer(_expB);

    // output = tf.strided_slice(a, [1, 0, 0], [3, 3, 3], shrink_axis_mask=5)
    nd4j::ops::strided_slice<float> op;
    auto result = op.execute({&matrix}, {}, {0,0,0,0,1,   1, 0, 0,  3, 3, 3,  1, 1, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(IndexingTests, MaskedSlice_3) {
    float _buff[] = {1.f, 1.2f, 1.3f, 2.f, 2.2f, 2.3f, 3.f, 3.2f, 3.3f, 4.f, 4.2f, 4.3f, 5.f,  5.2f, 5.3f, 6.f,   6.2f,  6.3f,  7.f,   7.2f,  7.3f,  8.f,   8.2f,  8.3f,  9.f,   9.2f,  9.3f};
    NDArray<float> matrix('c', {3, 3, 3});
    matrix.setBuffer(_buff);

    float _expB[] = { 4.f,   4.2f,  4.3f, 7.f, 7.2f,  7.3f};
    NDArray<float> exp('c', {2, 3});
    exp.setBuffer(_expB);

    // output = tf.strided_slice(a, [1, 0, 0], [3, 3, 3], shrink_axis_mask=5)
    nd4j::ops::strided_slice<float> op;
    auto result = op.execute({&matrix}, {}, {0,0,0,0,2,   1, 0, 0,  3, 3, 3,  1, 1, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(IndexingTests, MaskedSlice_4) {
    float _buff[] = {1.f, 1.2f, 1.3f, 2.f, 2.2f, 2.3f, 3.f, 3.2f, 3.3f, 4.f, 4.2f, 4.3f, 5.f,  5.2f, 5.3f, 6.f,   6.2f,  6.3f,  7.f,   7.2f,  7.3f,  8.f,   8.2f,  8.3f,  9.f,   9.2f,  9.3f};
    NDArray<float> matrix('c', {3, 3, 3});
    matrix.setBuffer(_buff);

    float _expB[] = { 4.f,   4.2f,  4.3f};
    NDArray<float> exp('c', {3});
    exp.setBuffer(_expB);

    // output = tf.strided_slice(a, [1, 0, 0], [3, 3, 3], shrink_axis_mask=5)
    nd4j::ops::strided_slice<float> op;
    auto result = op.execute({&matrix}, {}, {0,0,0,0, 3,   1, 0, 0,  3, 3, 3,  1, 1, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);


    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(IndexingTests, Live_Slice_1) {
    float _buff[] = {1.f, 1.2f, 1.3f, 2.f, 2.2f, 2.3f, 3.f, 3.2f, 3.3f, 4.f, 4.2f, 4.3f, 5.f,  5.2f, 5.3f, 6.f,   6.2f,  6.3f,  7.f,   7.2f,  7.3f,  8.f,   8.2f,  8.3f,  9.f,   9.2f,  9.3f};
    NDArray<float> matrix('c', {3, 3, 3});
    matrix.setBuffer(_buff);

    float _expB[] = { 4.f,   4.2f,  4.3f};
    NDArray<float> exp('c', {3});
    exp.setBuffer(_expB);

    NDArray<float> begin('c', {1, 3}, {1.0f, 0.0f, 0.0f});
    NDArray<float> end('c', {1, 3}, {3.0f, 3.0f, 3.0f});
    NDArray<float> stride('c', {1, 3}, {1.0f, 1.0f, 1.0f});

    // output = tf.strided_slice(a, [1, 0, 0], [3, 3, 3], shrink_axis_mask=5)
    nd4j::ops::strided_slice<float> op;
    auto result = op.execute({&matrix, &begin, &end, &stride}, {}, {0,0,0,0,3});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    // z->printShapeInfo("z shape");
    
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(IndexingTests, Test_StridedSlice_1) {
    NDArray<float> x('c', {1, 2}, {5.f, 2.f});
    NDArray<float> a('c', {1, 1}, {0.f});
    NDArray<float> b('c', {1, 1}, {1.f});
    NDArray<float> c('c', {1, 1}, {1.f});
    NDArray<float> exp(5.0f);

    nd4j::ops::strided_slice<float> op;
    auto result = op.execute({&x, &a, &b, &c}, {}, {0, 0, 0, 0, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    // z->printIndexedBuffer("Z");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(IndexingTests, Test_StridedSlice_2) {
    NDArray<float> x('c', {2, 3}, {1, 2, 3, 4, 5, 6});
    NDArray<float> a('c', {1, 2}, {1, 1});
    NDArray<float> b('c', {1, 2}, {2, 2});
    NDArray<float> c('c', {1, 2}, {1, 1});
    NDArray<float> exp('c', {1}, {5.0});

    nd4j::ops::strided_slice<float> op;
    auto result = op.execute({&x, &a, &b, &c}, {}, {0, 0, 0, 0, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    // z->printIndexedBuffer("Z");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(IndexingTests, Test_StridedSlice_3) {
    NDArray<float> x('c', {2, 3}, {1, 2, 3, 4, 5, 6});
    NDArray<float> a('c', {1, 2}, {1, 2});
    NDArray<float> b('c', {1, 2}, {2, 3});
    NDArray<float> c('c', {1, 2}, {1, 1});
    NDArray<float> exp('c', {1}, {6.0});

    nd4j::ops::strided_slice<float> op;
    auto result = op.execute({&x, &a, &b, &c}, {}, {0, 0, 0, 0, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    //z->printIndexedBuffer("Z");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(IndexingTests, Test_StridedSlice_4) {
    NDArray<float> x('c', {1, 2}, {5, 2});
    NDArray<float> a('c', {1, 1}, {0});
    NDArray<float> b('c', {1, 1}, {1});
    NDArray<float> c('c', {1, 1}, {1});
    NDArray<float> exp(5.0f);

    nd4j::ops::strided_slice<float> op;
    auto result = op.execute({&x, &a, &b, &c}, {}, {0, 0, 0, 0, 1, 0, 1, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    //z->printIndexedBuffer("Z");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(IndexingTests, Test_Subarray_Strided_1) {
    NDArray<float> x('c', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    NDArray<float> exp('c', {3, 2}, {1, 3, 4, 6, 7, 9});
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
    NDArray<float> matrix('c', {3, 3, 3});
    matrix.setBuffer(_buff);

    float _expB[] = { 4.f,   4.2f,  4.3f, 7.f, 7.2f,  7.3f};
    NDArray<float> exp('c', {2, 3});
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