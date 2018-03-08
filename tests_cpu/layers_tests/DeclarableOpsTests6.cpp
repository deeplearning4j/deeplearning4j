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

class DeclarableOpsTests6 : public testing::Test {
public:

    DeclarableOpsTests6() {
        printf("\n");
        fflush(stdout);
    }
};


TEST_F(DeclarableOpsTests6, Test_Dilation2D_Again_1) {
    NDArray<float> x('c', {4, 128, 128, 4});
    NDArray<float> w('c', {4, 5, 4});
    NDArray<float> exp('c', {4, 64, 43, 4});


    nd4j::ops::dilation2d<float> op;
    auto result = op.execute({&x, &w}, {}, {1, 1,5,7,1, 1,2,3,1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));

    delete result;
}


TEST_F(DeclarableOpsTests6, Test_Dilation2D_Again_2) {
    NDArray<float> x('c', {4, 26, 19, 4});
    NDArray<float> w('c', {11, 7, 4});

    nd4j::ops::dilation2d<float> op;
    auto result = op.execute({&x, &w}, {}, {0, 1,2,3,1, 1,3,2,1});
    ASSERT_EQ(Status::OK(), result->status());

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_Simple_Scalar_1) {
    NDArray<float> x('c', {1, 1}, {2.0f});
    NDArray<float> exp('c', {1, 1}, {4.0f});

    nd4j::ops::test_scalar<float> op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_Conv3D_NDHWC_11) {
    NDArray<float> x('c', {4, 2, 28, 28, 3});
    NDArray<float> w('c', {2, 5, 5, 3, 4});
    NDArray<float> exp('c', {4, 1, 7, 10, 4});

    nd4j::ops::conv3dnew<float> op;
    auto result = op.execute({&x, &w}, {}, {2,5,5, 5,4,3, 0,0,0, 1,1,1, 1,1});
    ASSERT_EQ(Status::OK(), result->status());

    ShapeList shapeList({x.shapeInfo(), w.shapeInfo()});
    ContextPrototype<float> proto;
    Context<float> ctx(1);
    ctx.getIArguments()->push_back(2);
    ctx.getIArguments()->push_back(5);
    ctx.getIArguments()->push_back(5);

    ctx.getIArguments()->push_back(5);
    ctx.getIArguments()->push_back(4);
    ctx.getIArguments()->push_back(3);

    ctx.getIArguments()->push_back(0);
    ctx.getIArguments()->push_back(0);
    ctx.getIArguments()->push_back(0);

    ctx.getIArguments()->push_back(1);
    ctx.getIArguments()->push_back(1);
    ctx.getIArguments()->push_back(1);

    ctx.getIArguments()->push_back(0);
    ctx.getIArguments()->push_back(0);

    auto shapes = op.calculateOutputShape(&shapeList, ctx);
    ASSERT_EQ(1, shapes->size());

    auto s = shapes->at(0);

    shape::printShapeInfoLinear("calculated shape", s);

    auto z = result->at(0);
    z->printShapeInfo("z shape");

    ASSERT_TRUE(exp.isSameShape(z));

    delete result;

    shapes->destroy();
    delete shapes;
}

TEST_F(DeclarableOpsTests6, Test_gather_Edge_1) {
    NDArray<float> x('c', {2, 4, 3, 2});
    NDArray<float> indices('c', {2}, {1.f, 0.f});

    nd4j::ops::gather<float> op;
    auto result = op.execute({&x, &indices}, {}, {-2});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_gatherNd_Edge_1) {
    NDArray<float> x('c', {2, 4, 2, 2});
    NDArray<float> indices('c', {3, 3}, {0,2,1, 0,1,0, 1,3,1});
    NDArray<float> exp('c', {3,2}, {11.f, 12.f, 5.f, 6.f, 31.f, 32.f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::gather_nd<float> op;
    auto result = op.execute({&x, &indices}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    z->printIndexedBuffer();
    z->printShapeInfo("z shape");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}



TEST_F(DeclarableOpsTests6, Test_StB_1) {
    NDArray<float> x('c', {4, 64, 64, 4});
    NDArray<float> blocks('c', {2}, {8, 8});
    NDArray<float> paddings('c', {2, 2}, {12, 12, 16, 16});

    x.assign(1.0f);

    nd4j::ops::space_to_batch<float> op;
    auto result = op.execute({&x, &blocks, &paddings}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    //nd4j_printf("Mean: %f\n", z->meanNumber());

    delete result;

}

TEST_F(DeclarableOpsTests6, Test_StB_2) {
    NDArray<float> x('c', {2, 6, 6, 2});
    NDArray<float> blocks('c', {2}, {2, 2});
    NDArray<float> paddings('c', {2, 2}, {2, 2, 2, 2});

    x.assign(1.0f);

    nd4j::ops::space_to_batch<float> op;
    auto result = op.execute({&x, &blocks, &paddings}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    delete result;

}

TEST_F(DeclarableOpsTests6, Test_BtS_1) {
    NDArray<float> x('f', {256, 8, 8, 2});
    NDArray<float> blocks('c',{2}, {8, 8});
    NDArray<float> crops('c', {2, 2});

    nd4j::ops::batch_to_space<float> op;
    auto result = op.execute({&x, &blocks, &crops}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_Order_1) {
    NDArray<float> x('f', {2, 3});
    NDArray<float> exp('c', {2, 3}, {1, 2, 3, 4, 5, 6});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::order<float> op;
    auto result = op.execute({&x}, {}, {0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));
    ASSERT_NE(x.ordering(), z->ordering());

    delete result;
}


TEST_F(DeclarableOpsTests6, Test_CumSum_Inclusive_Reverse_1) {
    NDArray<float> x('c', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    NDArray<float> exp('c', {3, 3}, {12.f, 15.f, 18.f, 11.f, 13.f, 15.f, 7.f, 8.f, 9.f});

    nd4j::ops::cumsum<float> op;
    auto result = op.execute({&x}, {}, {0, 1, 0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_CumSum_Inclusive_Reverse_2) {
    NDArray<float> x('c', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    NDArray<float> exp('c', {3, 3}, {6.f, 5.f, 3.f, 15.f, 11.f, 6.f, 24.f, 17.f, 9.f,});

    nd4j::ops::cumsum<float> op;
    auto result = op.execute({&x}, {}, {0, 1, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_CumSum_Exclusive_Reverse_1) {
    NDArray<float> x('c', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    NDArray<float> exp('c', {3, 3}, {11.f, 13.f, 15.f, 7.f, 8.f, 9.f, 0.f, 0.f, 0.f});

    nd4j::ops::cumsum<float> op;
    auto result = op.execute({&x}, {}, {1, 1, 0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_CumSum_Exclusive_Reverse_2) {
    NDArray<float> x('c', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    NDArray<float> exp('c', {3, 3}, {5.f, 3.f, 0.f, 11.f, 6.f, 0.f, 17.f, 9.f, 0.f});

    nd4j::ops::cumsum<float> op;
    auto result = op.execute({&x}, {}, {1, 1, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_CumSum_Exclusive_Reverse_2_1) {
    NDArray<float> x('c', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    NDArray<float> axis('c', {1}, {1});
    NDArray<float> exp('c', {3, 3}, {5.f, 3.f, 0.f, 11.f, 6.f, 0.f, 17.f, 9.f, 0.f});

    nd4j::ops::cumsum<float> op;
    auto result = op.execute({&x, &axis}, {}, {1, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, TestDropout_1) {

    NDArray<float> x('c', {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    NDArray<float> shape({2.f, 2.f});
    nd4j::ops::dropout<float> op;

    auto ress = op.execute({&x, &shape}, {0.2f}, {113});

    ASSERT_EQ(ND4J_STATUS_OK, ress->status());
    ress->at(0)->printIndexedBuffer("Result is ");
    x.printIndexedBuffer("Input is");

    delete ress;
}


TEST_F(DeclarableOpsTests6, TestDropout_2) {
//    NDArray<float> x0('c', {10, 10});
//    NDArray<float> x1('c', {10, 10});
    NDArray<float> x('c', {3, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f});

    nd4j::ops::dropout<float> op;

    auto ress = op.execute({&x}, {0.4f}, {113});

    ASSERT_EQ(ND4J_STATUS_OK, ress->status());
    x.printIndexedBuffer("Input is");
    ress->at(0)->printIndexedBuffer("Result is ");

    delete ress;
}

TEST_F(DeclarableOpsTests6, TestDropout_3) {
//    NDArray<float> x0('c', {10, 10});
//    NDArray<float> x1('c', {10, 10});
    NDArray<float> x('c', {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    NDArray<float> shape({1.f, 2.f});

    nd4j::ops::dropout<float> op;

    auto ress = op.execute({&x, &shape}, {0.4f}, {113});

    ASSERT_EQ(ND4J_STATUS_OK, ress->status());
    //x.printIndexedBuffer("Input is");
    //ress->at(0)->printIndexedBuffer("Result is ");

    delete ress;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, MaxPoolWithArgmax_1) {

    NDArray<double> x('c', {2, 2, 2, 4}, {
             5.5, 0.,   0.3,  5.5,
             1.5, 0.,   1.3,  6.5,
             8.6, 0.,    0.,  0.4,
             2.5, 1.,   0.3,  4.5,
             1.5, 1.,   1.3,  1.5,
             3.5, 0.,   1.3,  2.5,
             2.6, 2.,    3.,  1.4,
             4.5, 1.,   0.3,  0.5}
    );       
    NDArray<double> expI('c', {2, 2, 2, 4}, {
             0,  1,  2,  3,
             4,  5,  6,  7,
             8,  9, 10, 11,
            12, 13, 14, 15,
             0,  1,  2,  3,
             4,  5,  6,  7,
             8,  9, 10, 11,
            12, 13, 14, 15}
    );

    nd4j::ops::max_pool_with_argmax<double> op;

    auto ress = op.execute({&x}, {}, {1,1,1,1,1,1,1,1,1});

    
    ASSERT_EQ(ND4J_STATUS_OK, ress->status());
    ASSERT_TRUE(expI.isSameShape(ress->at(0)));
    ASSERT_TRUE(expI.isSameShape(ress->at(1)));
    ASSERT_TRUE(x.equalsTo(ress->at(0)));
    ASSERT_TRUE(expI.equalsTo(ress->at(1)));
    //x.printIndexedBuffer("Input is");
    //ress->at(0)->printIndexedBuffer("Result is ");
    ASSERT_TRUE(expI.equalsTo(ress->at(1)));
    
    delete ress;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, SufficientStatistics_1) {
//    NDArray<float> x0('c', {10, 10});
//    NDArray<float> x1('c', {10, 10});
    NDArray<double> x('c', {2, 2, 2, 4}, {
        5.5, 0.,  0.3, 5.5,
        1.5, 0.,  1.3, 6.5,
        8.6, 0.,   0., 0.4,
        2.5, 1.,  0.3, 4.5,
        1.5, 1.,  1.3, 1.5,
        3.5, 0.,  1.3, 2.5,
        2.6, 2.,   3., 1.4,
        4.5, 1.,  0.3, 0.5}
    );
// ------------------------------------
    double count = 8.0;
    NDArray<double> sumExp({30.2, 5., 7.8, 22.8});
    NDArray<double> sqrExp({154.22,   7.,    14.34, 103.62});

    NDArray<double> axis({0.0, 1.0, 2.0});

    nd4j::ops::sufficient_statistics<double> op;

    auto ress = op.execute({&x, &axis}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, ress->status());
    ASSERT_EQ(ress->at(0)->getScalar(0), count);
    ASSERT_TRUE(sumExp.equalsTo(ress->at(1)));
    ASSERT_TRUE(sqrExp.equalsTo(ress->at(2)));

    delete ress;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, SufficientStatistics_2) {
//    NDArray<float> x0('c', {10, 10});
//    NDArray<float> x1('c', {10, 10});
    NDArray<double> x('c', {2, 2, 2, 4}, {
        5.5, 0.,  0.3, 5.5,
        1.5, 0.,  1.3, 6.5,
        8.6, 0.,   0., 0.4,
        2.5, 1.,  0.3, 4.5,
        1.5, 1.,  1.3, 1.5,
        3.5, 0.,  1.3, 2.5,
        2.6, 2.,   3., 1.4,
        4.5, 1.,  0.3, 0.5}
    );
// ------------------------------------
    double count = 4.0;
    NDArray<double> sumExp('c', {2, 4}, {
        18.2,        3.,         4.6,        8.8,
        12.,         2.,         3.2,        14.}
    );

    NDArray<double> sqrExp('c', {2, 4}, {
        113.22, 5., 10.78, 34.62,
           41., 2.,  3.56, 69.}
    );

    NDArray<double> axis({0.0, 1.0});

    nd4j::ops::sufficient_statistics<double> op;

    auto ress = op.execute({&x, &axis}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, ress->status());
    ASSERT_EQ(ress->at(0)->getScalar(0), count);
    ASSERT_TRUE(sumExp.equalsTo(ress->at(1)));
    ASSERT_TRUE(sqrExp.equalsTo(ress->at(2)));

    delete ress;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, BinCount_1) {

    NDArray<double> x('c', {2, 2, 2}, {
        1, 2, 0, 1, 2, 2, 1, 2}
    );
// ------------------------------------

    NDArray<double> exp({1., 3., 4.});

    nd4j::ops::bincount<double> op;

    auto res = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, res->status());
    ASSERT_TRUE(exp.equalsTo(res->at(0)));

    delete res;
}

/////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, BinCount_2) {

    NDArray<double> x('c', {2, 2, 2}, {
        1, 2, 0, 1, 2, 2, 1, 2}
    );

    NDArray<double> weights('c', {2, 2, 2}, {
        2, 1, 3, 1, 5, 1, 1, 6}
    );

// ------------------------------------

    NDArray<double> exp({3., 4., 13.});

    nd4j::ops::bincount<double> op;

    auto res = op.execute({&x, &weights}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, res->status());
    ASSERT_TRUE(exp.equalsTo(res->at(0)));

    delete res;
}

/////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, BinCount_3) {

    NDArray<double> x('c', {2, 2, 2}, {
        1, 2, 0, 1, 2, 2, 1, 2}
    );

    NDArray<double> weights('c', {2, 2, 2}, {
        2, 1, 3, 1, 5, 1, 1, 6}
    );

// ------------------------------------

    NDArray<double> exp({3., 4.});

    nd4j::ops::bincount<double> op;

    auto res = op.execute({&x, &weights}, {}, {0, 2});

    ASSERT_EQ(ND4J_STATUS_OK, res->status());
    ASSERT_TRUE(exp.equalsTo(res->at(0)));

    delete res;
}

/////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, BroadcastDynamicShape_1) {

    NDArray<double> x( {2., 2., 2.} );

    NDArray<double> y({ 2., 1., 2.});

// ------------------------------------

    NDArray<double> exp({2., 2., 2.});

    nd4j::ops::broadcast_dynamic_shape<double> op;

    auto res = op.execute({&x, &y}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, res->status());
    ASSERT_TRUE(exp.equalsTo(res->at(0)));

    delete res;
}

/////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, BroadcastDynamicShape_2) {

    NDArray<double> x( {2., 2.} );

    NDArray<double> y({2.0, 1.0, 2.0});

// ------------------------------------
    NDArray<double> exp({2., 2., 2.});

    nd4j::ops::broadcast_dynamic_shape<double> op;

    auto res = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, res->status());
    ASSERT_TRUE(exp.equalsTo(res->at(0)));

    delete res;
}

/////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, BroadcastDynamicShape_3) {

    NDArray<double> x( {2., 2., 2.} );

    NDArray<double> y({ 2.0, 1.0});

// ------------------------------------

    NDArray<double> exp({2., 2., 2.});

    nd4j::ops::broadcast_dynamic_shape<double> op;

    auto res = op.execute({&x, &y}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, res->status());
    ASSERT_TRUE(exp.equalsTo(res->at(0)));

    delete res;
}

/////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, BroadcastDynamicShape_4) {

    NDArray<double> x({2., 2., 2.});

    NDArray<double> y({2., 2.});

// ------------------------------------

    NDArray<double> exp({2., 2., 2.});

    nd4j::ops::broadcast_dynamic_shape<double> op;

    auto res = op.execute({&x, &y}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, res->status());
    ASSERT_TRUE(exp.equalsTo(res->at(0)));

    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, ClipByGlobalNorm_1) {

    NDArray<double> x('c', {2, 3, 3}, {-3.0, 0.0, 0.0, 4.0, 0.0, 0.0,
                                      -3.0, 0.0, 0.0, 4.0, 0.0, 0.0,
                                      -3.0, 0.0, 0.0, 4.0, 0.0, 0.0}
    );

    NDArray<double> exp('c', {2, 3, 3}, {
            -0.2771281,  0.,          0.,
            0.36950415,  0.,          0.,
            -0.2771281,  0.,          0.,
            0.36950415,  0.,          0.,
            -0.2771281,  0.,          0.,
            0.36950415,  0.,          0.}
    );
//    8.660254
//    NDArray<double> expNorm(8.660254);

    nd4j::ops::clip_by_global_norm<double> op;
    auto result = op.execute({&x}, {0.8}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    auto norm = result->at(1);
    z->printIndexedBuffer("Output");
    exp.printIndexedBuffer("Expected");
    norm->printIndexedBuffer("Norm");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));
//    ASSERT_TRUE(expNorm.equalsTo(norm));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, ClipByGlobalNorm_2) {

    NDArray<double> x('c', {2, 3, 3}, {-3.0, 0.0, 0.0, 4.0, 0.0, 0.0,
                                      -3.0, 0.0, 0.0, 4.0, 0.0, 0.0,
                                      -3.0, 0.0, 0.0, 4.0, 0.0, 0.0}
    );

    NDArray<double> a('c', {2, 3, 3}, {-3.0, 0.0, 0.0, 4.0, 0.0, 0.0,
                                      -3.0, 0.0, 0.0, 4.0, 0.0, 0.0,
                                      -3.0, 0.0, 0.0, 4.0, 0.0, 0.0}
    );

    NDArray<double> exp('c', {2, 3, 3}, {
                                    -0.44090813,   0.,          0.,
                                      0.5878775,   0.,          0.,
                                    -0.44090813,   0.,          0.,
                                      0.5878775,   0.,          0.,
                                    -0.44090813,   0.,          0.,
                                      0.5878775,   0.,          0.}
//12.247449

    );

    nd4j::ops::clip_by_global_norm<double> op;
    auto result = op.execute({&x, &a}, {1.8}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    auto y = result->at(1);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.isSameShape(y));
    ASSERT_TRUE(exp.equalsTo(z));
    ASSERT_TRUE(exp.equalsTo(y));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, ClipByGlobalNorm_3) {

    NDArray<double> x('c', {2, 3, 3}, {-3.0, 0.0, 0.0, 4.0, 0.0, 0.0, -3.0, 0.0, 0.0, 4.0, 0.0, 0.0, -3.0, 0.0, 0.0, 4.0, 0.0, 0.0});
    NDArray<double> a('c', {2, 3, 3}, {-3.0, 0.0, 0.0, 4.0, 0.0, 0.0, -3.0, 0.0, 0.0, 4.0, 0.0, 0.0, -3.0, 0.0, 0.0, 4.0, 0.0, 0.0});
    NDArray<double> exp('c', {2, 3, 3}, {
            -0.19595918,  0.,          0.,
              0.2612789,  0.,          0.,
            -0.19595918,  0.,          0.,
              0.2612789,  0.,          0.,
            -0.19595918,  0.,          0.,
              0.2612789,   0.,          0.}
    );

    nd4j::ops::clip_by_global_norm<double> op;
    auto result = op.execute({&x, &a}, {0.8}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    auto y = result->at(1);
    z->printIndexedBuffer("Output 1");
    y->printIndexedBuffer("Output 2");
    result->at(2)->printIndexedBuffer("Global norm is");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.isSameShape(y));
    ASSERT_TRUE(result->at(2)->isScalar());
    ASSERT_TRUE(exp.equalsTo(z));
    ASSERT_TRUE(exp.equalsTo(y));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, MatrixDeterminant_1) {

    NDArray<double> x('c', {2, 3, 3}, {-3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, -3.0, 4.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 4.0});
    NDArray<double> exp({36.0, -48.0});

    nd4j::ops::matrix_determinant<double> op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    z->printIndexedBuffer("Output ");
    exp.printIndexedBuffer("Expected ");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, MatrixInverse_1) {

    NDArray<double> x('c', {2, 5, 5}, {
                    2.,  4., 60.,  8., 10.,
                    0.,  1.,  2.,  3.,  4.,
                    0.,  0.,  2.,  4.,  6.,
                    0.,  0.,  0.,  1.,  2.,
                    0.,  0.,  0.,  0.,  4.,

                     1.,  0.,  0.,  0.,  0.,
                     2.,  1.,  0.,  0.,  0.,
                    30.,  2.,  1.,  0.,  0.,
                     4.,  3.,  2.,  1.,  0.,
                     5.,  4.,  3.,  2.,  1.,
    });

    NDArray<double> exp('c', {2, 5, 5}, {
                    0.5, -2.0, -13.0, 54.0, -6.75,
                    0.0,  1.0,  -1.0,  1.0,   0.0,
                      0,    0,   0.5, -2.0,  0.25,
                      0,    0,     0,  1.0,  -0.5,
                      0,    0,     0,    0,  0.25,
    
                    1.0,  0.0,  0.0,  0.0, 0.,
                   -2.0,  1.0,   0.,   0., 0.,
                  -26.0, -2.0,    1,    0, 0.,
                   54.0,  1.0, -2.0,    1, 0.,
                  -27.0,  0.0,  1.0, -2.0, 1.
    });

    nd4j::ops::matrix_inverse<double> op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
//    z->printIndexedBuffer("Output ");
//    exp.printIndexedBuffer("Expected ");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


////////////////////////////////////////////////////////////////////////////////
/*
TEST_F(DeclarableOpsTests6, MatrixInverse_2) {

    NDArray<double> x('c', {2, 5, 5}, {
                    1.,  2., 30.,  4.,  5.,
                    0.,  1.,  2.,  3.,  4.,
                    0.,  0.,  1.,  2.,  3.,
                    0.,  0.,  0.,  1.,  2.,
                    0.,  0.,  0.,  0.,  1.,

                     4.,   0.,  0.,  0.,  0.,
                     4.,   2.,  0.,  0.,  0.,
                    30.,   2.,  1.,  0.,  0.,
                     8.,   6.,  4.,  2.,  0.,
                    15.,  12.,  9.,  6.,  3.,
    });

    NDArray<double> exp('c', {2, 5, 5}, {
     1.0,  -2.0,  -26.0,  54.0, -27.0,
     0.0,   1.0,  -2.0,    1.0,   0.0,
     0.0,   0.0,   1.0,   -2.0,   1.0, 
     0.0,   0.0,   0.0,    1.0,  -2.0, 
     0.0,   0.0,   0.0,    0.0,   1.0, 

     0.25,  0.0,    0.0,   0.0,   0.0,
    -0.50,  0.5,    0.0,   0.0,   0.0,
    -6.50, -1.0,    1.0,   0.0,   0.0,
    13.50,  0.5,   -2.0,   0.5,   0.0,
    -6.75,  0.0,    1.0,  -1.0,   0.33333333
    });

    nd4j::ops::matrix_inverse<double> op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    z->printIndexedBuffer("Output ");
    exp.printIndexedBuffer("Expected ");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}
*/
////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, MatrixInverse_3) {

    NDArray<double> x('c', {5, 5}, {
                     4.,   0.,  0.,  0.,  0.,
                     4.,   2.,  0.,  0.,  0.,
                    30.,   2.,  1.,  0.,  0.,
                     8.,   6.,  4.,  2.,  0.,
                    15.,  12.,  9.,  6.,  3.,
    });

    NDArray<double> exp('c', {5, 5}, {
     0.25,  0.0,    0.0,   0.0,   0.0,
    -0.50,  0.5,    0.0,   0.0,   0.0,
    -6.50, -1.0,    1.0,   0.0,   0.0,
    13.50,  0.5,   -2.0,   0.5,   0.0,
    -6.75,  0.0,    1.0,  -1.0,   0.33333333
    });

    nd4j::ops::matrix_inverse<double> op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    //z->printIndexedBuffer("Output ");
    //exp.printIndexedBuffer("Expected ");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, MatrixInverse_4) {

    NDArray<double> x('c', {5, 5}, {
                    1.,  2., 30.,  4.,  5.,
                    0.,  1.,  2.,  3.,  4.,
                    0.,  0.,  1.,  2.,  3.,
                    0.,  0.,  0.,  1.,  2.,
                    0.,  0.,  0.,  0.,  1.
    });

    NDArray<double> exp('c', {5, 5}, {
     1.0,  -2.0,  -26.0,  54.0, -27.0,
     0.0,   1.0,  -2.0,    1.0,   0.0,
     0.0,   0.0,   1.0,   -2.0,   1.0, 
     0.0,   0.0,   0.0,    1.0,  -2.0, 
     0.0,   0.0,   0.0,    0.0,   1.0 
    });

    nd4j::ops::matrix_inverse<double> op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    z->printIndexedBuffer("Output ");
    exp.printIndexedBuffer("Expected ");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}
