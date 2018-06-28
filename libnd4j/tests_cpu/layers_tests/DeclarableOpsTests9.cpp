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
