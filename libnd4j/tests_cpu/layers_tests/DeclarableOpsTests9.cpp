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

    x.linspace(1);

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
    y.linspace(0., 1./N);  // [0, 1)


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
    y.linspace(-N/2.);  // [-25000, 25000)


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

    gradO.linspace(0.01, 0.01);

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

    gradO.linspace(0.01, 0.01);

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

    x0.linspace(1);
    x1.linspace(1);
    x2.linspace(1);

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

    x0.linspace(1);
    x1.linspace(1);
    x2.linspace(1);

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

    x0.linspace(1);
    x1.linspace(1);
    x2.linspace(1);

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

    gradO.linspace(0.01, 0.01);

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

    gradO.linspace(0.01, 0.01);

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

    gradO.linspace(0.01, 0.01);

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

    gradO.linspace(0.01, 0.01);

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
//    ress->at(0)->printIndexedBuffer("Result is ");
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
//    ress->at(0)->printIndexedBuffer("Result is ");
    //x.printIndexedBuffer("Input is");

    delete ress;
}

TEST_F(DeclarableOpsTests9, Test_Dropout_01) {
    NDArray<float> x0('c', {10, 10});
    NDArray<float> x1('c', {10, 10});

    x0.linspace(1);
    x1.linspace(1);
    NativeOps nativeOps;

    Nd4jLong* _bufferA = new Nd4jLong[100000];
    long _seed = 119L;
    auto _rngA = (nd4j::random::RandomBuffer *) nativeOps.initRandom(nullptr, _seed, 100000, (Nd4jPointer) _bufferA);

    float prob[] = {0.5f};

    x0.template applyRandom<randomOps::DropOut<float>>(_rngA, nullptr, &x0, prob);
//    x1.template applyRandom<randomOps::DropOut<float>>(_rngB, nullptr, &x1, prob);
//    x0.printIndexedBuffer("Result1");
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

    x0.linspace(1);
    x1.linspace(1);
    NativeOps nativeOps;

    float prob[] = {0.5f};
    Nd4jLong* _bufferA = new Nd4jLong[100000];
    long _seed = 119L;
    auto _rngA = (nd4j::random::RandomBuffer *) nativeOps.initRandom(nullptr, _seed, 100000, (Nd4jPointer) _bufferA);

    x0.template applyRandom<randomOps::DropOutInverted<float>>(_rngA, nullptr, &x0, prob);
//    x1.template applyRandom<randomOps::DropOutInverted<float>>(_rngB, nullptr, &x1, prob);
//    x0.printIndexedBuffer("01Result1");
    int count = 0;
    for (int e = 0; e < x0.lengthOf(); e++)
        if (x0.getScalar(e) != 0.f)
            count++;
//    nd4j_printf("\nX0 count %i\n", count);
//    ASSERT_TRUE(x0.equalsTo(&x1));

    // this check is required to ensure we're calling wrong signature
//    ASSERT_FALSE(x0.equalsTo(nexp0));
//    ASSERT_FALSE(x0.equalsTo(nexp1));
//    ASSERT_FALSE(x0.equalsTo(nexp2));
    nativeOps.destroyRandom(_rngA);
    delete [] _bufferA;

    nd4j::ops::dropout<float> op;

    auto ress = op.execute({&x1}, {0.5f}, {119});

    ASSERT_EQ(ND4J_STATUS_OK, ress->status());
//    ress->at(0)->printIndexedBuffer("01Dropout result is ");
    count = 0;
    for (int e = 0; e < ress->at(0)->lengthOf(); e++)
        if (ress->at(0)->getScalar(e) != 0.f)
            count++;
//    nd4j_printf("\n01Dropout count %i\n\n", count);

    nd4j::ops::dropout_bp<float> op2;
    //NDArray<float> exp('c', {10,10}, {4.f, 0.f, 12.f, 0.f, 20.f, 24.f, 0.f, 32.f, 0.f, 0.f, 0.f, 0.f, 52.f, 56.f, 60.f, 0.f, 0.f, 0.f, 0.f, 0.f, 84.f, 88.f, 0.f, 0.f, 0.f, 0.f, 108.f, 0.f, 0.f, 120.f, 0.f, 0.f, 132.f, 0.f, 0.f, 0.f, 0.f, 0.f, 156.f, 0.f, 164.f, 168.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 200.f, 204.f, 0.f, 0.f, 0.f, 220.f, 0.f, 0.f, 232.f, 236.f, 240.f, 0.f, 248.f, 0.f, 0.f, 260.f, 0.f, 0.f, 0.f, 276.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 316.f, 0.f, 324.f, 0.f, 0.f, 336.f, 0.f, 0.f, 0.f, 0.f, 356.f, 0.f, 0.f, 368.f, 0.f, 0.f, 0.f, 384.f, 388.f, 0.f, 0.f, 400.f});
    //02Dropout result is  [4.000000, 0.000000, 12.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 36.000000, 0.000000, 0.000000, 0.000000, 0.000000, 56.000000, 60.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 88.000000, 0.000000, 96.000000, 0.000000, 0.000000, 108.000000, 0.000000, 0.000000, 120.000000, 0.000000, 128.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 156.000000, 0.000000, 164.000000, 0.000000, 0.000000, 0.000000, 0.000000, 184.000000, 0.000000, 0.000000, 0.000000, 200.000000, 0.000000, 0.000000, 0.000000, 216.000000, 0.000000, 0.000000, 0.000000, 232.000000, 0.000000, 240.000000, 0.000000, 248.000000, 0.000000, 0.000000, 260.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 308.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 348.000000, 0.000000, 356.000000, 0.000000, 0.000000, 0.000000, 0.000000, 376.000000, 0.000000, 384.000000, 0.000000, 0.000000, 0.000000, 400.000000]

    auto ressX = op2.execute({&x1, &x0}, {0.5f}, {119});

    ASSERT_EQ(ND4J_STATUS_OK, ressX->status());
//    ressX->at(0)->printIndexedBuffer("02Dropout result is ");
//    ASSERT_TRUE(exp.equalsTo(ressX->at(0)));
    delete ressX;
    delete ress;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, Test_AlphaDropout_01) {
    NDArray<float> x0('c', {10, 10});
    NDArray<float> x1('c', {10, 10});

    x0.linspace(1);
//    NDArrayFactory<float>::linspace(1, x1);
    NativeOps nativeOps;

    Nd4jLong* _bufferA = new Nd4jLong[100000];
    long _seed = 119L;
    auto _rngA = (nd4j::random::RandomBuffer *) nativeOps.initRandom(nullptr, _seed, 100000, (Nd4jPointer) _bufferA);

    float prob[] = {0.5f, 0.5f, 1.5f, 1.6f};
    
    x0.template applyRandom<randomOps::AlphaDropOut<float>>(_rngA, nullptr, &x0, prob);
//    x1.template applyRandom<randomOps::DropOut<float>>(_rngB, nullptr, &x1, prob);
//    x0.printIndexedBuffer("Result1Alpha");
//    ASSERT_TRUE(x0.equalsTo(&x1));

    // this check is required to ensure we're calling wrong signature
//    ASSERT_FALSE(x0.equalsTo(nexp0));
//    ASSERT_FALSE(x0.equalsTo(nexp1));
//    ASSERT_FALSE(x0.equalsTo(nexp2));
    nativeOps.destroyRandom(_rngA);
    delete [] _bufferA;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, Test_AlphaDropout_02) {
    NDArray<float> x0('c', {10, 10});
//    NDArray<float> x1('c', {10, 10});

    x0.linspace(1);
//    NDArrayFactory<float>::linspace(1, x1);
    NativeOps nativeOps;

    Nd4jLong* _bufferA = new Nd4jLong[100000];
    long _seed = 119L;
    auto _rngA = (nd4j::random::RandomBuffer *) nativeOps.initRandom(nullptr, _seed, 100000, (Nd4jPointer) _bufferA);

    float prob[] = {0.5f, 1.0f, 0.f, 0.f};
    
    x0.template applyRandom<randomOps::AlphaDropOut<float>>(_rngA, nullptr, &x0, prob);
//    x1.template applyRandom<randomOps::DropOut<float>>(_rngB, nullptr, &x1, prob);
//    x0.printIndexedBuffer("Result1Alpha");
//    ASSERT_TRUE(x0.equalsTo(&x1));

    // this check is required to ensure we're calling wrong signature
//    ASSERT_FALSE(x0.equalsTo(nexp0));
//    ASSERT_FALSE(x0.equalsTo(nexp1));
//    ASSERT_FALSE(x0.equalsTo(nexp2));
    nativeOps.destroyRandom(_rngA);
    delete [] _bufferA;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, Test_AlphaDropout_BP_1) {
    NDArray<float> x('c', {10, 10});
    NDArray<float> eps('c', {10, 10});

    x.linspace(1);
    eps.linspace(1);

    nd4j::ops::alpha_dropout_bp<float> op;

    auto ress = op.execute({&x, &eps}, {0.5f, 0.5f, 1.5f, 1.6f}, {119});

    ASSERT_EQ(ND4J_STATUS_OK, ress->status());

    ress->at(0)->printIndexedBuffer("Result1AlphaBP");
    delete ress;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test1) {

    NDArray<double> x ('c', {3, 4});
    NDArray<double> y ('c', {4, 3});
    NDArray<double> exp('f', {3, 3}, {35.,  79., 123., 40.,  92., 144., 45., 105., 165.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test2) {

    NDArray<double> x ('c', {3, 4});
    NDArray<double> y ('f', {4, 3});
    NDArray<double> exp('f', {3, 3}, {35., 79., 123.,40., 92., 144.,45.,105., 165.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test3) {

    NDArray<double> x ('f', {3, 4});
    NDArray<double> y ('c', {4, 3});
    NDArray<double> exp('f', {3, 3}, {35., 79., 123.,40., 92., 144.,45.,105., 165.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test4) {

    NDArray<double> x ('f', {3, 4});
    NDArray<double> y ('f', {4, 3});
    NDArray<double> exp('f', {3, 3}, {35., 79., 123.,40., 92., 144.,45.,105., 165.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test5) {

    NDArray<double> x ('c', {4, 3});
    NDArray<double> y ('c', {4, 3});
    NDArray<double> exp('f', {3, 3}, {83.,  94., 105., 94., 107., 120., 105., 120., 135.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test6) {

    NDArray<double> x ('c', {4, 3});
    NDArray<double> y ('f', {3, 4});
    NDArray<double> exp('f', {3, 3}, {35.,  40.,  45., 79.,  92., 105., 123., 144., 165.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {1, 1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test7) {

    NDArray<double> x ('c', {5,  3,4});
    NDArray<double> y ('f', {5,  3,4});
    NDArray<double> exp('f',{5,  3,3}, {3. ,  84.6, 281.4, 593.4, 1020.6, 7. , 107.8, 323.8, 655. , 1101.4,11. , 131. , 366.2, 716.6, 1182.2,
                                        7. , 107.8, 323.8, 655. , 1101.4,17.4, 137.4, 372.6, 723. , 1188.6,27.8, 167. , 421.4, 791. , 1275.8,
                                       11. , 131. , 366.2, 716.6, 1182.2,27.8, 167. , 421.4, 791. , 1275.8,44.6, 203. , 476.6, 865.4, 1369.4,});

    x.linspace(1.);
    y.linspace(0.1, 0.1);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {0, 1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test8) {

    NDArray<double> x ('c', {2,5,  3,4});
    NDArray<double> y ('f', {2,5,  3,4});
    NDArray<double> exp('f',{2,5,  3,3}, {3. , 1563. ,  84.6, 2220.6, 281.4, 2993.4, 593.4, 3881.4,1020.6, 4884.6,   7. , 1663. , 107.8, 2339.8, 323.8, 3131.8, 655. , 4039. ,1101.4, 5061.4,
                                          11. , 1763. , 131. , 2459. , 366.2, 3270.2, 716.6, 4196.6,1182.2, 5238.2,   7. , 1663. , 107.8, 2339.8, 323.8, 3131.8, 655. , 4039. ,1101.4, 5061.4,
                                          17.4, 1769.4, 137.4, 2465.4, 372.6, 3276.6, 723. , 4203. ,1188.6, 5244.6,  27.8, 1875.8, 167. , 2591. , 421.4, 3421.4, 791. , 4367. ,1275.8, 5427.8,
                                          11. , 1763. , 131. , 2459. , 366.2, 3270.2, 716.6, 4196.6,1182.2, 5238.2,  27.8, 1875.8, 167. , 2591. , 421.4, 3421.4, 791. , 4367. ,1275.8, 5427.8,
                                          44.6, 1988.6, 203. , 2723. , 476.6, 3572.6, 865.4, 4537.4,1369.4, 5617.4});

    x.linspace(1.);
    y.linspace(0.1, 0.1);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {0, 1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}
 
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test9) {

    NDArray<double> x ('c', {2,5,  4,3});
    NDArray<double> y ('f', {2,5,  3,4});
    NDArray<double> exp('f',{2,5,  3,3}, {7. , 1639. , 103. , 2311. , 314.2, 3098.2, 640.6, 4000.6,1082.2, 5018.2,   8. , 1664. , 108.8, 2340.8, 324.8, 3132.8, 656. , 4040. ,1102.4, 5062.4,
                                          9. , 1689. , 114.6, 2370.6, 335.4, 3167.4, 671.4, 4079.4,1122.6, 5106.6,  15.8, 1743.8, 131. , 2435. , 361.4, 3241.4, 707. , 4163. ,1167.8, 5199.8,
                                          18.4, 1770.4, 138.4, 2466.4, 373.6, 3277.6, 724. , 4204. ,1189.6, 5245.6,  21. , 1797. , 145.8, 2497.8, 385.8, 3313.8, 741. , 4245. ,1211.4, 5291.4,
                                          24.6, 1848.6, 159. , 2559. , 408.6, 3384.6, 773.4, 4325.4,1253.4, 5381.4,  28.8, 1876.8, 168. , 2592. , 422.4, 3422.4, 792. , 4368. ,1276.8, 5428.8,
                                          33. , 1905. , 177. , 2625. , 436.2, 3460.2, 810.6, 4410.6,1300.2, 5476.2});

    x.linspace(1.);
    y.linspace(0.1, 0.1);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {1, 1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}
   
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test10) {

    NDArray<double> x ('c', {1, 4, 3});
    NDArray<double> y ('f', {1, 3, 4});
    NDArray<double> exp('f', {1, 3, 3}, {35.,  40.,  45., 79.,  92., 105., 123., 144., 165.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {1, 1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test11) {

    NDArray<double> x ('c', {4, 1});
    NDArray<double> y ('f', {1, 4});
    NDArray<double> exp('f', {1, 1}, {15});

    x.linspace(1.);
    y.linspace(0.5, 0.5);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {1, 1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test12) {

    NDArray<double> x ('c', {1, 4, 1});
    NDArray<double> y ('f', {1, 1, 4});
    NDArray<double> exp('f', {1, 1, 1}, {15});

    x.linspace(1.);
    y.linspace(0.5, 0.5);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {1, 1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test13) {

    NDArray<double> x ('c', {2, 3});
    NDArray<double> y ('c', {3, 5});
    NDArray<double> exp('f', {5, 2}, {23. , 26. , 29. , 32. , 35., 50. , 57.5, 65. , 72.5, 80.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {0, 0, 1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test14) {

    NDArray<double> x ('c', {3, 2});
    NDArray<double> y ('c', {3, 5});
    NDArray<double> exp('f', {5, 2}, {37. , 41.5, 46. , 50.5, 55., 46. , 52. , 58. , 64. , 70.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {1, 0, 1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test15) {

    NDArray<double> x ('c', {3, 2});
    NDArray<double> y ('c', {3, 5});
    NDArray<double> exp('f', {5, 2}, {37. , 41.5, 46. , 50.5, 55., 46. , 52. , 58. , 64. , 70.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {1, 0, 1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test16) {

    NDArray<double> x ('c', {2,2,  3,5});
    NDArray<double> y ('c', {2,2,  4,3});
    NDArray<double> exp('f',{2,2,  4,5}, {4.6, 281.8, 89.2, 582.4, 10. , 314.2,108.1, 628.3, 15.4, 346.6,127. , 674.2, 20.8, 379. ,145.9, 720.1,  5.2, 289.6, 93.4, 593.8,
                                          11.5, 322.9,113.2, 640.6, 17.8, 356.2,133. , 687.4, 24.1, 389.5,152.8, 734.2,  5.8, 297.4, 97.6, 605.2, 13. , 331.6,118.3, 652.9,
                                          20.2, 365.8,139. , 700.6, 27.4, 400. ,159.7, 748.3,  6.4, 305.2,101.8, 616.6, 14.5, 340.3,123.4, 665.2, 22.6, 375.4,145. , 713.8,
                                          30.7, 410.5,166.6, 762.4,  7. , 313. ,106. , 628. , 16. , 349. ,128.5, 677.5, 25. , 385. ,151. , 727. , 34. , 421. ,173.5, 776.5});

    x.linspace(1.);
    y.linspace(0.1, 0.1);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {1, 1, 1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test17) {

    NDArray<double> x ('f', {4, 3});
    NDArray<double> y ('c', {4});
    NDArray<double> exp('f',{3}, {7., 8., 9.});

    x.linspace(1.);
    y.linspace(0.1, 0.1);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {1, 0});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test18) {

    NDArray<double> x ('f', {3});
    NDArray<double> y ('c', {4, 3});
    NDArray<double> exp('f',{4}, {1.4, 3.2, 5., 6.8});

    x.linspace(1.);
    y.linspace(0.1, 0.1);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {0, 1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test19) {

    NDArray<double> x ('f', {1, 1});
    NDArray<double> y ('c', {1, 1});
    NDArray<double> exp('f',{1, 1}, {0.2});

    x.linspace(2.);
    y.linspace(0.1, 0.1);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test20) {

    NDArray<double> x ('f', {1, 1});
    NDArray<double> y ('c', {1, 1});
    NDArray<double> exp('f',{1, 1}, {0.2});

    x.linspace(2.);
    y.linspace(0.1, 0.1);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {1,1,1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test21) {

    NDArray<double> x ('f', {1});
    NDArray<double> y ('c', {1, 1});
    NDArray<double> exp('f',{1}, {0.2});

    x.linspace(2.);
    y.linspace(0.1, 0.1);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test22) {

    NDArray<double> x ('f', {1,1});
    NDArray<double> y ('c', {1});
    NDArray<double> exp('f',{1}, {0.2});

    x.linspace(2.);
    y.linspace(0.1, 0.1);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test23) {

    NDArray<double> x ('f', {4});   
    NDArray<double> y ('c', {4});
    NDArray<double> exp(3.);

    x.linspace(1.);
    y.linspace(0.1, 0.1);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {1, 1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test24) {

    NDArray<double> x ('f', {1}, {2.});   
    NDArray<double> y ('c', {1}, {3.});
    NDArray<double> exp(6.);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {1, 1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}


////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test10) {

    NDArray<float> x0('c', {2,3,4});
    NDArray<float> x1('f', {2,2,4});
    NDArray<float> x2('c', {2,1,4});
    NDArray<float> exp('c', {2,6,4}, { 1.f,  2.f,  3.f,  4.f, 5.f,  6.f,  7.f,  8.f, 9.f, 10.f, 11.f, 12.f, 1.f,  2.f,  3.f,  4.f, 5.f,  6.f,  7.f,  8.f, 1.f,  2.f,  3.f,  4.f,
                                      13.f, 14.f, 15.f, 16.f,17.f, 18.f, 19.f, 20.f,21.f, 22.f, 23.f, 24.f, 9.f, 10.f, 11.f, 12.f,13.f, 14.f, 15.f, 16.f, 5.f,  6.f,  7.f,  8.f});

    x0.linspace(1);
    x1.linspace(1);
    x2.linspace(1);

    nd4j::ops::concat<float> op;

    auto result = op.execute({&x0, &x1, &x2}, {}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}


////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test11) {

    NDArray<float> x0('c', {2,3,4});
    NDArray<float> x1('f', {2,2,4});
    NDArray<float> x2('f', {2,1,4});
    NDArray<float> exp('c', {2,6,4}, { 1.f,  2.f,  3.f,  4.f, 5.f,  6.f,  7.f,  8.f, 9.f, 10.f, 11.f, 12.f, 1.f,  2.f,  3.f,  4.f, 5.f,  6.f,  7.f,  8.f, 1.f,  2.f,  3.f,  4.f,
                                      13.f, 14.f, 15.f, 16.f,17.f, 18.f, 19.f, 20.f,21.f, 22.f, 23.f, 24.f, 9.f, 10.f, 11.f, 12.f,13.f, 14.f, 15.f, 16.f, 5.f,  6.f,  7.f,  8.f});

    x0.linspace(1);
    x1.linspace(1);
    x2.linspace(1);

    nd4j::ops::concat<float> op;

    auto result = op.execute({&x0, &x1, &x2}, {}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test12) {

    NDArray<float> x0('c', {2,3,4});
    NDArray<float> x1('f', {2,2,4});
    NDArray<float> x2('f', {2,1,4});
    NDArray<float> exp('c', {2,6,4}, { 1.f,  2.f,  3.f,  4.f, 5.f,  6.f,  7.f,  8.f, 9.f, 10.f, 11.f, 12.f, 1.f,  2.f,  3.f,  4.f, 5.f,  6.f,  7.f,  8.f, 1.f,  2.f,  3.f,  4.f,
                                      13.f, 14.f, 15.f, 16.f,17.f, 18.f, 19.f, 20.f,21.f, 22.f, 23.f, 24.f, 9.f, 10.f, 11.f, 12.f,13.f, 14.f, 15.f, 16.f, 5.f,  6.f,  7.f,  8.f});

    x0.linspace(1);
    x1.linspace(1);
    x2.linspace(1);

    nd4j::ops::concat<float> op;

    auto result = op.execute({&x0, &x1, &x2}, {}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test13) {

    NDArray<float> x0('f', {2,3,4});
    NDArray<float> x1('f', {2,2,4});
    NDArray<float> x2('f', {2,1,4});
    NDArray<float> exp('f', {2,6,4}, { 1.f, 13.f, 5.f, 17.f, 9.f, 21.f, 1.f,  9.f, 5.f, 13.f, 1.f,  5.f, 2.f, 14.f, 6.f, 18.f,10.f, 22.f, 2.f, 10.f, 6.f, 14.f, 2.f,  6.f,
                                       3.f, 15.f, 7.f, 19.f,11.f, 23.f, 3.f, 11.f, 7.f, 15.f, 3.f,  7.f, 4.f, 16.f, 8.f, 20.f,12.f, 24.f, 4.f, 12.f, 8.f, 16.f, 4.f,  8.f});

    x0.linspace(1);
    x1.linspace(1);
    x2.linspace(1);

    nd4j::ops::concat<float> op;

    auto result = op.execute({&x0, &x1, &x2}, {}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);


    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}


 
