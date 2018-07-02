//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 22.06.2018
//


#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <NDArray.h>
#include <ops/ops.h>


using namespace nd4j;


class DeclarableOpsTests9 : public testing::Test {
public:

    DeclarableOpsTests9() {
        printf("\n");
        fflush(stdout);
    }
};

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, reduceStDevBP_test3) {

    NDArray<float> x('c', {3,4});
    NDArray<float> gradO1('c', {3,1}, {1.,2.,3.});
    NDArray<float> gradO2('c', {3}, {1.,2.,3.});
    NDArray<float> exp('c', {3,4}, {-0.335410, -0.111803, 0.111803, 0.335410, -0.670820, -0.223607, 0.223607, 0.670820, -1.006231, -0.335410, 0.335410, 1.006231});         

    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_stdev_bp<float> op;

    auto result = op.execute({&x, &gradO2}, {0,0}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    
    auto output = result->at(0);
    // output->printIndexedBuffer();
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO1}, {1,0}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    
    output = result->at(0);
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
    delete result;

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, exponentialDistributionInv_test1) {
    
    const int N = 50000;
    const double lambda = 2.;
    const double mean   = 1. / lambda; 
    const double std    = mean;

    NDArray<double> x('c', {N});
    double extraParams[] = {lambda};

    Nd4jLong *buffer = new Nd4jLong[N];
    NativeOps nativeOps;
    nd4j::random::RandomBuffer* rng = (nd4j::random::RandomBuffer *) nativeOps.initRandom(nullptr, 123, N, (Nd4jPointer) buffer);    
    if (rng == nullptr)
        throw std::runtime_error("DeclarableOpsTests9.exponentialDistributionInv_test1: RNG initialization failed !");
    
    functions::random::RandomFunction<double>::template execTransform<randomOps::ExponentialDistributionInv<double>>(rng, x.getBuffer(), x.getShapeInfo(), extraParams);
    const double actualMean = x.meanNumber();
    const double actualStd  = x.template varianceNumber<simdOps::SummaryStatsStandardDeviation<double>>(true);
 
    ASSERT_NEAR(mean, actualMean, 0.01);
    ASSERT_NEAR(std,  actualStd, 0.01);    

    nativeOps.destroyRandom((Nd4jPointer) rng);
    delete[] buffer;
        
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, exponentialDistributionInv_test2) {
    
    const int N = 50000;
    const double lambda = 2.;
    const double mean   = 1. / lambda; 
    const double std    = mean;
    double extraParams[] = {lambda};

    NDArray<double> x('c', {N});
    NDArray<double> y('c', {N});
    NDArrayFactory<double>::linspace(0., y, 1./N);  // [0, 1)


    Nd4jLong *buffer = new Nd4jLong[N];
    NativeOps nativeOps;
    nd4j::random::RandomBuffer* rng = (nd4j::random::RandomBuffer *) nativeOps.initRandom(nullptr, 123, N, (Nd4jPointer) buffer);    
    if (rng == nullptr)
        throw std::runtime_error("DeclarableOpsTests9.exponentialDistributionInv_test2: RNG initialization failed !");
    
    functions::random::RandomFunction<double>::template execTransform<randomOps::ExponentialDistributionInv<double>>(rng, y.getBuffer(), y.getShapeInfo(), x.getBuffer(), x.getShapeInfo(), extraParams);

    const double actualMean = x.meanNumber();
    const double actualStd  = x.template varianceNumber<simdOps::SummaryStatsStandardDeviation<double>>(true);

    ASSERT_NEAR(mean, actualMean, 0.01);
    ASSERT_NEAR(std,  actualStd, 0.01);    

    nativeOps.destroyRandom((Nd4jPointer) rng);
    delete[] buffer;

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, exponentialDistribution_test1) {
    
    const int N = 50000;
    const double lambda = 2.;
    const double mean   = 1. / lambda; 
    const double std    = mean;

    NDArray<double> x('c', {N});
    double extraParams[] = {lambda};

    Nd4jLong *buffer = new Nd4jLong[N];
    NativeOps nativeOps;
    nd4j::random::RandomBuffer* rng = (nd4j::random::RandomBuffer *) nativeOps.initRandom(nullptr, 123, N, (Nd4jPointer) buffer);    
    if (rng == nullptr)
        throw std::runtime_error("DeclarableOpsTests9.exponentialDistribution_test1: RNG initialization failed !");
    
    functions::random::RandomFunction<double>::template execTransform<randomOps::ExponentialDistribution<double>>(rng, x.getBuffer(), x.getShapeInfo(), extraParams);
    const double actualMean = x.meanNumber();
    const double actualStd  = x.template varianceNumber<simdOps::SummaryStatsStandardDeviation<double>>(true);
 
    ASSERT_NEAR(mean, actualMean, 0.01);
    ASSERT_NEAR(std,  actualStd, 0.01);    

    nativeOps.destroyRandom((Nd4jPointer) rng);
    delete[] buffer;       
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, exponentialDistribution_test2) {
    
    const int N = 50000;
    const double lambda = 2.;
    const double mean   = 1. / lambda; 
    const double std    = mean;
    double extraParams[] = {lambda};

    NDArray<double> x('c', {N});
    NDArray<double> y('c', {N});
    NDArrayFactory<double>::linspace(-N/2., y);  // [-25000, 25000)


    Nd4jLong *buffer = new Nd4jLong[N];
    NativeOps nativeOps;
    nd4j::random::RandomBuffer* rng = (nd4j::random::RandomBuffer *) nativeOps.initRandom(nullptr, 123, N, (Nd4jPointer) buffer);    
    if (rng == nullptr)
        throw std::runtime_error("DeclarableOpsTests9.exponentialDistribution_test2: RNG initialization failed !");
    
    functions::random::RandomFunction<double>::template execTransform<randomOps::ExponentialDistribution<double>>(rng, y.getBuffer(), y.getShapeInfo(), x.getBuffer(), x.getShapeInfo(), extraParams);

    const double actualMean = x.meanNumber();
    const double actualStd  = x.template varianceNumber<simdOps::SummaryStatsStandardDeviation<double>>(true);

    ASSERT_NEAR(mean, actualMean, 0.01);
    ASSERT_NEAR(std,  actualStd, 0.01);    

    nativeOps.destroyRandom((Nd4jPointer) rng);
    delete[] buffer;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, tile_bp_test1) {

    NDArray<double> input   ('c', {2, 3}, {1.,2.,3.,4.,5.,6.});
    NDArray<double> gradO   ('c', {4, 9});
    NDArray<double> gradIExp('c', {2, 3}, {0.78, 0.84, 0.9,1.32, 1.38, 1.44});

    NDArrayFactory<double>::linspace(0.01, gradO, 0.01);

    nd4j::ops::tile_bp<double> op;
    ResultSet<double>* results = op.execute({&input, &gradO}, {}, {2, 3});
    NDArray<double>* gradI = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(gradIExp.isSameShape(gradI));
    ASSERT_TRUE(gradIExp.equalsTo(gradI));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, tile_bp_test2) {

    NDArray<double> input   ('c', {2, 3}, {1.,2.,3.,4.,5.,6.});
    NDArray<double> gradO   ('c', {2, 9});
    NDArray<double> gradIExp('c', {2, 3}, {0.12, 0.15, 0.18, 0.39, 0.42, 0.45});

    NDArrayFactory<double>::linspace(0.01, gradO, 0.01);

    nd4j::ops::tile_bp<double> op;
    ResultSet<double>* results = op.execute({&input, &gradO}, {}, {1, 3});
    NDArray<double>* gradI = results->at(0);
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(gradIExp.isSameShape(gradI));
    ASSERT_TRUE(gradIExp.equalsTo(gradI));

    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test1) {

    NDArray<float> x0('c', {2,3,4});
    NDArray<float> x1('c', {2,2,4});
    NDArray<float> x2('c', {2,1,4});
    NDArray<float> exp('c', {2,6,4}, {1.f,  2.f,  3.f,  4.f, 5.f,  6.f,  7.f,  8.f, 9.f, 10.f, 11.f, 12.f, 1.f,  2.f,  3.f,  4.f, 5.f,  6.f,  7.f,  8.f, 1.f,  2.f,  3.f,  4.f,
                                     13.f, 14.f, 15.f, 16.f,17.f, 18.f, 19.f, 20.f,21.f, 22.f, 23.f, 24.f, 9.f, 10.f, 11.f, 12.f,13.f, 14.f, 15.f, 16.f, 5.f,  6.f,  7.f,  8.});

    NDArrayFactory<float>::linspace(1, x0);
    NDArrayFactory<float>::linspace(1, x1);
    NDArrayFactory<float>::linspace(1, x2);

    nd4j::ops::concat<float> op;

    auto result = op.execute({&x0, &x1, &x2}, {}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test2) {

    NDArray<float> x0('c', {1,3,1});
    NDArray<float> x1('c', {1,2,1});
    NDArray<float> x2('c', {1,1,1});
    NDArray<float> exp('c', {1,6,1}, {1.f, 2.f, 3.f, 1.f, 2.f, 1.f});

    NDArrayFactory<float>::linspace(1, x0);
    NDArrayFactory<float>::linspace(1, x1);
    NDArrayFactory<float>::linspace(1, x2);

    nd4j::ops::concat<float> op;

    auto result = op.execute({&x0, &x1, &x2}, {}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test3) {

    NDArray<float> x0('c', {3});
    NDArray<float> x1('c', {2});
    NDArray<float> x2('c', {1});
    NDArray<float> exp('c', {6}, {1.f, 2.f, 3.f, 1.f, 2.f, 1.f});

    NDArrayFactory<float>::linspace(1, x0);
    NDArrayFactory<float>::linspace(1, x1);
    NDArrayFactory<float>::linspace(1, x2);

    nd4j::ops::concat<float> op;

    auto result = op.execute({&x0, &x1, &x2}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test4) {

    NDArray<float> x0('c', {1,1,1}, {1.f});
    NDArray<float> x1('c', {1,1,1}, {2.f});
    NDArray<float> x2('c', {1,1,1}, {3.f});
    NDArray<float> exp('c', {1,3,1}, {1.f, 2.f, 3.f});

    nd4j::ops::concat<float> op;

    auto result = op.execute({&x0, &x1, &x2}, {}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test5) {

    NDArray<float> x0(1.f);
    NDArray<float> x1('c', {1}, {2.f});
    NDArray<float> x2(3.f);
    NDArray<float> exp('c', {3}, {1.f, 2.f, 3.f});

    nd4j::ops::concat<float> op;

    auto result = op.execute({&x0, &x1, &x2}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test6) {

    NDArray<float> x0(1.f);
    NDArray<float> x1('c', {2}, {2.f, 20.f});
    NDArray<float> x2(3.f);
    NDArray<float> exp('c', {4}, {1.f, 2.f, 20.f, 3.f});

    nd4j::ops::concat<float> op;

    auto result = op.execute({&x0, &x1, &x2}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test7) {

    NDArray<float> x0(1.f);
    NDArray<float> x1(2.f);
    NDArray<float> x2(3.f);
    NDArray<float> exp('c', {3}, {1.f, 2.f, 3.f});

    nd4j::ops::concat<float> op;

    auto result = op.execute({&x0, &x1, &x2}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test8) {

    NDArray<float> x0(1.f);
    NDArray<float> exp('c', {1}, {1.f});

    nd4j::ops::concat<float> op;

    auto result = op.execute({&x0}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test9) {

    NDArray<float> x0('c', {1}, {1.f});
    NDArray<float> exp('c', {1}, {1.f});

    nd4j::ops::concat<float> op;

    auto result = op.execute({&x0}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}




//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, tile_bp_test3) {

    NDArray<double> input   ('c', {2, 3}, {1.,2.,3.,4.,5.,6.});
    NDArray<double> gradO   ('c', {2, 3});
    NDArray<double> gradIExp('c', {2, 3}, {0.01, 0.02, 0.03,0.04, 0.05, 0.06});

    NDArrayFactory<double>::linspace(0.01, gradO, 0.01);

    nd4j::ops::tile_bp<double> op;
    ResultSet<double>* results = op.execute({&input, &gradO}, {}, {1, 1});
    NDArray<double>* gradI = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(gradIExp.isSameShape(gradI));
    ASSERT_TRUE(gradIExp.equalsTo(gradI));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, tile_bp_test4) {

    NDArray<double> input   ('c', {6}, {1.,2.,3.,4.,5.,6.});
    NDArray<double> gradO   ('c', {12});
    NDArray<double> gradIExp('c', {6}, {0.08, 0.1 , 0.12, 0.14, 0.16, 0.18});

    NDArrayFactory<double>::linspace(0.01, gradO, 0.01);

    nd4j::ops::tile_bp<double> op;
    ResultSet<double>* results = op.execute({&input, &gradO}, {}, {2});
    NDArray<double>* gradI = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(gradIExp.isSameShape(gradI));
    ASSERT_TRUE(gradIExp.equalsTo(gradI));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, tile_bp_test5) {

    NDArray<double> input   ('c', {1}, {1.});
    NDArray<double> gradO   ('c', {1});
    NDArray<double> gradIExp('c', {1}, {0.01});

    NDArrayFactory<double>::linspace(0.01, gradO, 0.01);

    nd4j::ops::tile_bp<double> op;
    ResultSet<double>* results = op.execute({&input, &gradO}, {}, {1});
    NDArray<double>* gradI = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(gradIExp.isSameShape(gradI));
    ASSERT_TRUE(gradIExp.equalsTo(gradI));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, tile_bp_test6) {

    NDArray<double> input   ('c', {2, 1, 3}, {1.,2.,3.,4.,5.,6.});
    NDArray<double> gradO   ('c', {2, 3, 6});
    NDArray<double> gradIExp('c', {2, 1, 3}, {0.51, 0.57, 0.63, 1.59, 1.65, 1.71});

    NDArrayFactory<double>::linspace(0.01, gradO, 0.01);

    nd4j::ops::tile_bp<double> op;
    ResultSet<double>* results = op.execute({&input, &gradO}, {}, {1, 3, 2});
    NDArray<double>* gradI = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(gradIExp.isSameShape(gradI));
    ASSERT_TRUE(gradIExp.equalsTo(gradI));

    delete results;
}
////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, TestDropout_BP_1) {

    NDArray<float> x('c', {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    NDArray<float> errs('c', {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    NDArray<float> shape({2.f, 2.f});
    nd4j::ops::dropout_bp<float> op;

    auto ress = op.execute({&x, &errs, &shape}, {0.2f}, {113});

    ASSERT_EQ(ND4J_STATUS_OK, ress->status());
    ress->at(0)->printIndexedBuffer("Result is ");
    //x.printIndexedBuffer("Input is");

    delete ress;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, TestDropout_1) {

    NDArray<float> x('c', {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
//    NDArray<float> errs('c', {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    NDArray<float> shape({2.f, 2.f});
    nd4j::ops::dropout<float> op;

    auto ress = op.execute({&x, &shape}, {0.2f}, {113});

    ASSERT_EQ(ND4J_STATUS_OK, ress->status());
    ress->at(0)->printIndexedBuffer("Result is ");
    //x.printIndexedBuffer("Input is");

    delete ress;
}

TEST_F(DeclarableOpsTests9, Test_Dropout_01) {
    NDArray<float> x0('c', {10, 10});
    NDArray<float> x1('c', {10, 10});

    NDArrayFactory<float>::linspace(1, x0);
    NDArrayFactory<float>::linspace(1, x1);
    NativeOps nativeOps;

    Nd4jLong* _bufferA = new Nd4jLong[100000];
    long _seed = 119L;
    auto _rngA = (nd4j::random::RandomBuffer *) nativeOps.initRandom(nullptr, _seed, 100000, (Nd4jPointer) _bufferA);

    float prob[] = {0.5f};

    x0.template applyRandom<randomOps::DropOut<float>>(_rngA, nullptr, &x0, prob);
//    x1.template applyRandom<randomOps::DropOut<float>>(_rngB, nullptr, &x1, prob);
    x0.printIndexedBuffer("Result1");
//    ASSERT_TRUE(x0.equalsTo(&x1));

    // this check is required to ensure we're calling wrong signature
//    ASSERT_FALSE(x0.equalsTo(nexp0));
//    ASSERT_FALSE(x0.equalsTo(nexp1));
//    ASSERT_FALSE(x0.equalsTo(nexp2));
    nativeOps.destroyRandom(_rngA);
    delete [] _bufferA;
}

TEST_F(DeclarableOpsTests9, Test_DropoutInverted_01) {
    NDArray<float> x0('c', {10, 10});
    NDArray<float> x1('c', {10, 10});

    NDArrayFactory<float>::linspace(1, x0);
    NDArrayFactory<float>::linspace(1, x1);
    NativeOps nativeOps;

    float prob[] = {0.5f};
    Nd4jLong* _bufferA = new Nd4jLong[100000];
    long _seed = 119L;
    auto _rngA = (nd4j::random::RandomBuffer *) nativeOps.initRandom(nullptr, _seed, 100000, (Nd4jPointer) _bufferA);

    x0.template applyRandom<randomOps::DropOutInverted<float>>(_rngA, nullptr, &x0, prob);
//    x1.template applyRandom<randomOps::DropOutInverted<float>>(_rngB, nullptr, &x1, prob);
    x0.printIndexedBuffer("Result1");
    int count = 0;
    for (int e = 0; e < x0.lengthOf(); e++)
        if (x0.getScalar(e) != 0.f)
            count++;
    nd4j_printf("X0 count %i\n", count);
//    ASSERT_TRUE(x0.equalsTo(&x1));

    // this check is required to ensure we're calling wrong signature
//    ASSERT_FALSE(x0.equalsTo(nexp0));
//    ASSERT_FALSE(x0.equalsTo(nexp1));
//    ASSERT_FALSE(x0.equalsTo(nexp2));
    nativeOps.destroyRandom(_rngA);
    delete [] _bufferA;

    nd4j::ops::dropout<float> op;

    auto ress = op.execute({&x0}, {0.98f}, {119});

    ASSERT_EQ(ND4J_STATUS_OK, ress->status());
    ress->at(0)->printIndexedBuffer("Dropout result is ");
    count = 0;
    for (int e = 0; e < ress->at(0)->lengthOf(); e++)
        if (ress->at(0)->getScalar(e) != 0.f)
            count++;
    nd4j_printf("Dropout count %i\n", count);

    delete ress;
}
