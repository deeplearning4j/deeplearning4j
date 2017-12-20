#include "testlayers.h"
#include <ops/declarable/helpers/householder.h>

using namespace nd4j;

class HelpersTests1 : public testing::Test {
public:
    
    HelpersTests1() {
        
        std::cout<<std::endl<<std::flush;
    }

};


///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, evalHouseholderData_test1) {
            
    NDArray<double> x('c', {1,9}, {14,17,3,1,9,1,2,5,11});            
    NDArray<double> tailExpexted('c', {1,8}, {0.415009, 0.0732369, 0.0244123,  0.219711, 0.0244123, 0.0488246,  0.122062,  0.268535});
    double coeffExpected = 1.51923;
    double normXExpected = -26.96294;

    double coeff, normX;
    NDArray<double> tail('c', {1,8});
    ops::helpers::evalHouseholderData(x, tail, normX, coeff);
    // tail.printBuffer();

    ASSERT_NEAR(normX, normXExpected, 1e-5);
    ASSERT_NEAR(coeff, coeffExpected, 1e-5);
    ASSERT_TRUE(tail.equalsTo(&tailExpexted));

}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, evalHouseholderData_test2) {
            
    NDArray<double> x('c', {1,9}, {14,17,3,1,9,1,2,5,11});            
    NDArray<double> w('c', {1,9});
    NDArray<double> identity('c', {9,9});            
    identity.setIdentity();    
    
    double coeff, normX;
    NDArray<double> tail('c', {1,8});
    ops::helpers::evalHouseholderData(x, tail, normX, coeff);
    NDArray<double> expected('c', {1,9}, {normX,  0.00000,  -0.00000,   0.00000,   0.00000,   0.00000,   0.00000,  -0.00000,   0.00000});

    w({{},{1,-1}}).assign(&tail);
    w(0) = 1.;
    NDArray<double>* wT = w.transpose();
    NDArray<double> P = identity - mmul(*wT, w) * coeff;    
    NDArray<double> result = mmul(x,P);
    
    ASSERT_TRUE(result.equalsTo(&expected));
    delete wT;

}
