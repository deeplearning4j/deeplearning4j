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
TEST_F(DeclarableOpsTests9, random_exponential_test3) {
    
    const double lambda = 2.;
    const double mean   = 1. / lambda; 
    const double std    = mean;
    NDArray<double> shapeArr('c', {1}, {100000});

    // NDArray<double> x('c', {100000});
    // const double min = 0.;
    // const double max = 1.;
    // const int len = static_cast<int>(shapeArr(0));
    // for(int i=0; i<len; ++i) {
    //     double f = (double)rand() / RAND_MAX;
    //     double elem = min + f * (max - min);
    //     // x(i) = 1. - nd4j::math::nd4j_pow<double>((double) M_E, -(lambda * elem));
    //     x(i) = -nd4j::math::nd4j_log<double>(1. - elem) / lambda;
    // }

    // double actualMean = 0.;
    // for(int i=0; i<len; ++i)
    //     actualMean += x(i);
    // actualMean /= len;

    // double actualStd = 0.;
    // for(int i=0; i<len; ++i)
    //     actualStd += (x(i) - actualMean)*(x(i) - actualMean);
    // actualStd = nd4j::math::nd4j_sqrt<double>(actualStd/(len-1));

    nd4j::ops::random_exponential<double> op;
    ResultSet<double>* results = op.execute({&shapeArr}, {lambda}, {});
    ASSERT_EQ(Status::OK(), results->status());

    NDArray<double>* output = results->at(0);

    const double actualMean = output->meanNumber();
    const double actualStd  = nd4j::math::nd4j_sqrt<double>(output->template varianceNumber<simdOps::SummaryStatsStandardDeviation<double>>(true));
    // const double actualMean = x.meanNumber();
    // const double actualStd  = nd4j::math::nd4j_sqrt<double>(x.template varianceNumber<simdOps::SummaryStatsStandardDeviation<double>>(true));


    printf("%f\n", actualMean);
    printf("%f\n", actualStd);

    ASSERT_EQ(mean, actualMean);
    ASSERT_EQ(std,  actualStd);
    ASSERT_EQ(1,1);
    
    delete results;
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, random_exponential_test4) {
    
    const double lambda = 2.;
    const double mean   = 1. / lambda; 
    const double std    = mean;
    NDArray<double> shapeArr('c', {1}, {100000});
    NDArray<double> x('c', {100000});
    NDArrayFactory<double>::linspace(-100., x);

    nd4j::ops::random_exponential<double> op;
    ResultSet<double>* results = op.execute({&shapeArr, &x}, {lambda}, {});
    ASSERT_EQ(Status::OK(), results->status());

    NDArray<double>* output = results->at(0);

    const double actualMean = output->meanNumber();
    const double actualStd  = nd4j::math::nd4j_sqrt<double>(output->template varianceNumber<simdOps::SummaryStatsStandardDeviation<double>>(true));

    printf("%f\n", actualMean);
    printf("%f\n", actualStd);

    ASSERT_EQ(mean, actualMean);
    ASSERT_EQ(std,  actualStd);
    
    delete results;
}
