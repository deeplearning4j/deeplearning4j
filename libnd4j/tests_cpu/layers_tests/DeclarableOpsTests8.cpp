//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 10.06.2018
//


#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <NDArray.h>
// #include <array/NDArrayList.h>

using namespace nd4j;


class DeclarableOpsTests8 : public testing::Test {
public:

    DeclarableOpsTests8() {
        printf("\n");
        fflush(stdout);
    }
};


////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceVariance_test1) {

    NDArray<float> x('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.f});
    NDArray<float> exp('c', {4}, {602.2222f, 727.13885f, 993.5555f, 755.8889f});
        
    nd4j::ops::reduce_variance<float> op;
    auto result = op.execute({&x}, {}, {0,1});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceVariance_test2) {

    NDArray<float> x('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.f});
    NDArray<float> exp('c', {1,1,4}, {602.2222f, 727.13885f, 993.5555f, 755.8889f});
    
    nd4j::ops::reduce_variance<float> op;
    auto result = op.execute({&x}, {1.}, {0,1});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceVariance_test3) {

    NDArray<float> x('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.f});
    NDArray<float> exp('c', {3}, {900.9375f, 969.8594f, 424.1875f});
        
    nd4j::ops::reduce_variance<float> op;
    auto result = op.execute({&x}, {}, {0,2});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceVariance_test4) {

    NDArray<float> x('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.f});
    NDArray<float> exp('c', {1,3,1}, {900.9375f, 969.8594f, 424.1875f});  
        
    nd4j::ops::reduce_variance<float> op;
    auto result = op.execute({&x}, {1.}, {0,2});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceVariance_test5) {

    NDArray<float> x('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.f});
    NDArray<float> exp(788.6927f);
        
    nd4j::ops::reduce_variance<float> op;
    auto result = op.execute({&x}, {}, {});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceVariance_test6) {

    NDArray<float> x('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    NDArray<float> exp(788.6927f);
           
    nd4j::ops::reduce_variance<float> op;
    auto result = op.execute({&x}, {}, {0,1,2});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceVariance_test7) {

    NDArray<float> x('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    NDArray<float> exp('c', {1,1,1}, {788.6927f});
           
    nd4j::ops::reduce_variance<float> op;
    auto result = op.execute({&x}, {1.}, {0,1,2});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDev_test1) {

    NDArray<float> x('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    NDArray<float> exp('c', {4}, {24.54022f, 26.96551f, 31.52072f, 27.49343f});
        
    nd4j::ops::reduce_stdev<float> op;
    auto result = op.execute({&x}, {}, {0,1});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDev_test2) {

    NDArray<float> x('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    NDArray<float> exp('c', {1,1,4}, {24.54022f, 26.96551f, 31.52072f, 27.49343f});
    
    nd4j::ops::reduce_stdev<float> op;
    auto result = op.execute({&x}, {1.}, {0,1});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDev_test3) {

    NDArray<float> x('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    NDArray<float> exp('c', {3}, {30.01562f, 31.14257f, 20.59581f});
        
    nd4j::ops::reduce_stdev<float> op;
    auto result = op.execute({&x}, {}, {0,2});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDev_test4) {

    NDArray<float> x('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    NDArray<float> exp('c', {1,3,1}, {30.01562f, 31.14257f, 20.59581f});  
        
    nd4j::ops::reduce_stdev<float> op;
    auto result = op.execute({&x}, {1.}, {0,2});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDev_test5) {

    NDArray<float> x('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    NDArray<float> exp(28.08367f);
        
    nd4j::ops::reduce_stdev<float> op;
    auto result = op.execute({&x}, {}, {});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDev_test6) {

    NDArray<float> x('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    NDArray<float> exp(28.08367f);
           
    nd4j::ops::reduce_stdev<float> op;
    auto result = op.execute({&x}, {}, {0,1,2});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDev_test7) {

    NDArray<float> x('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    NDArray<float> exp('c', {1,1,1}, {28.08367f});
           
    nd4j::ops::reduce_stdev<float> op;
    auto result = op.execute({&x}, {1.f}, {0,1,2});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDev_test8) {

    NDArray<float> x('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    NDArray<float> exp('c', {4}, {26.88246f, 29.53924f, 34.52921f, 30.11755f});
        
    nd4j::ops::reduce_stdev<float> op;
    auto result = op.execute({&x}, {0.f,1.f}, {0,1});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceVarianceBP_test1) {

    NDArray<float> x('c', {3,4});
    NDArray<float> gradO1('c', {1,1}, {0.5f});
    NDArray<float> gradO2(0.5f);
    NDArray<float> exp12('c', {3,4}, {-0.5f, -0.4090909f, -0.3181818f, -0.22727273f, -0.13636364f, -0.045454547f, 0.045454547f, 0.13636364f, 0.22727273f, 0.3181818f, 0.4090909f, 0.5f});
    NDArray<float> exp34('c', {3,4}, {-0.45833334f, -0.375f, -0.29166666f, -0.20833333f, -0.125f, -0.041666668f, 0.041666668f, 0.125f, 0.20833333f, 0.29166666f, 0.375f, 0.45833334f});

    NDArrayFactory<float>::linspace(1, x);
            
    nd4j::ops::reduce_variance_bp<float> op;

    auto result = op.execute({&x, &gradO2}, {0,1}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    
    auto output = result->at(0);
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO1}, {1,1}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    
    output = result->at(0);
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));
    delete result;    

    result = op.execute({&x, &gradO2}, {0,0}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    
    output = result->at(0);
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));
    delete result;    

    result = op.execute({&x, &gradO1}, {1,0}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    
    output = result->at(0);
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));
    delete result;   

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceVarianceBP_test2) {

    NDArray<float> x('c', {3,4});
    NDArray<float> gradO1('c', {1,4}, {1.f,2.f,3.f,4.f});
    NDArray<float> gradO2('c', {4}, {1.f,2.f,3.f,4.f});
    NDArray<float> exp12('c', {3,4}, {-2.666667f, -5.333333f, -8.000000f,  -10.666667f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 2.666667f, 5.333333f,  8.000000f, 10.666667f});
    NDArray<float> exp34('c', {3,4}, {-4.000000f, -8.000000f, -12.000000f, -16.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 4.000000f, 8.000000f, 12.000000f, 16.000000f});

    NDArrayFactory<float>::linspace(1, x);
            
    nd4j::ops::reduce_variance_bp<float> op;

    auto result = op.execute({&x, &gradO2}, {0,0}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    
    auto output = result->at(0);    
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO1}, {1,0}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    
    output = result->at(0);        
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output)); 
    delete result;    

    result = op.execute({&x, &gradO2}, {0,1}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    
    output = result->at(0);    
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));
    delete result;    

    result = op.execute({&x, &gradO1}, {1,1}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    
    output = result->at(0);
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));
    delete result;   

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceVarianceBP_test3) {

    NDArray<float> x('c', {3,4});
    NDArray<float> gradO1('c', {3,1}, {1.f,2.f,3.f});
    NDArray<float> gradO2('c', {3}, {1.f,2.f,3.f});
    NDArray<float> exp12('c', {3,4}, {-0.750000f, -0.250000f, 0.250000f, 0.750000f, -1.500000f, -0.500000f, 0.500000f, 1.500000f, -2.250000f, -0.750000f, 0.750000f, 2.250000f});
    NDArray<float> exp34('c', {3,4}, {-1.000000f, -0.333333f, 0.333333f, 1.000000f, -2.000000f, -0.666667f, 0.666667f, 2.000000f, -3.000000f, -1.000000f, 1.000000f, 3.000000f});

    NDArrayFactory<float>::linspace(1, x);
            
    nd4j::ops::reduce_variance_bp<float> op;

    auto result = op.execute({&x, &gradO2}, {0,0}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    
    auto output = result->at(0);    
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO1}, {1,0}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    
    output = result->at(0);        
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output)); 
    delete result;    

    result = op.execute({&x, &gradO2}, {0,1}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    
    output = result->at(0);    
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));
    delete result;    

    result = op.execute({&x, &gradO1}, {1,1}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    
    output = result->at(0);
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));
    delete result;   

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDevBP_test1) {

    NDArray<float> x('c', {3,4});
    NDArray<float> gradO1('c', {1,1}, {0.5f});
    NDArray<float> gradO2(0.5f);
    NDArray<float> exp12('c', {3,4}, {-0.069337524f, -0.056730703f, -0.04412388f, -0.031517055f, -0.018910235f, -0.0063034114f, 0.0063034114f, 0.018910235f, 0.031517055f, 0.04412388f, 0.056730703f, 0.069337524f});     
    NDArray<float> exp34('c', {3,4}, {-0.06638563f, -0.05431551f, -0.0422454f, -0.030175284f, -0.01810517f, -0.006035057f, 0.006035057f, 0.01810517f, 0.030175284f, 0.0422454f, 0.05431551f, 0.06638563f});

    NDArrayFactory<float>::linspace(1, x);
            
    nd4j::ops::reduce_stdev_bp<float> op;

    auto result = op.execute({&x, &gradO2}, {0,1}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    
    auto output = result->at(0);
    // output->printIndexedBuffer();
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO1}, {1,1}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    
    output = result->at(0);
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO2}, {0,0}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    
    output = result->at(0);
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO1}, {1,0}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    
    output = result->at(0);
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));
    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDevBP_test2) {

    NDArray<float> x('c', {3,4});
    NDArray<float> gradO1('c', {1,4}, {1.f,2.f,3.f,4.f});
    NDArray<float> gradO2('c', {4}, {1.f,2.f,3.f,4.f});
    NDArray<float> exp12('c', {3,4}, {-0.4082483f, -0.8164966f, -1.2247449f, -1.6329932f, 0.0, 0.0, 0.0, 0.0, 0.4082483f, 0.8164966f, 1.2247449f, 1.6329932f});
    NDArray<float> exp34('c', {3,4}, {-0.5f, -1.0f, -1.5f, -2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f});

    NDArrayFactory<float>::linspace(1, x);
            
    nd4j::ops::reduce_stdev_bp<float> op;

    auto result = op.execute({&x, &gradO2}, {0,0}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    
    auto output = result->at(0);        
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO1}, {1,0}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    
    output = result->at(0);        
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output)); 
    delete result;    

    result = op.execute({&x, &gradO2}, {0,1}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    
    output = result->at(0);    
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));
    delete result;    

    result = op.execute({&x, &gradO1}, {1,1}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    
    output = result->at(0);
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));
    delete result;   
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDevBP_test3) {

    NDArray<float> x('c', {3,4});
    NDArray<float> gradO1('c', {3,1}, {1.f,2.f,3.f});
    NDArray<float> gradO2('c', {3}, {1.f,2.f,3.f});
    NDArray<float> exp12('c', {3,4}, {-0.3354102f, -0.1118034f, 0.1118034f, 0.3354102f, -0.6708204f, -0.2236068f, 0.2236068f, 0.6708204f, -1.0062306f, -0.3354102f, 0.3354102f, 1.0062306f});
    NDArray<float> exp34('c', {3,4}, {-0.38729835f, -0.12909944f, 0.12909944f, 0.38729835f, -0.7745967f, -0.2581989f, 0.2581989f, 0.7745967f, -1.161895f, -0.38729835f, 0.38729835f, 1.161895f});

    NDArrayFactory<float>::linspace(1, x);
            
    nd4j::ops::reduce_stdev_bp<float> op;

    auto result = op.execute({&x, &gradO2}, {0,0}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    
    auto output = result->at(0);    
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO1}, {1,0}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    
    output = result->at(0);        
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output)); 
    delete result;    

    result = op.execute({&x, &gradO2}, {0,1}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    
    output = result->at(0);    
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));
    delete result;    

    result = op.execute({&x, &gradO1}, {1,1}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    
    output = result->at(0);
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));
    delete result;   

}