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


