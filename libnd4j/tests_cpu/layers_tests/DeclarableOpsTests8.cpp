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


////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Sum_1) {
    
    NDArray<float> input('c', {3, 5},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.});    
    NDArray<float> exp(120.f);
    //************************************//

    nd4j::ops::reduce_sum<float> op;
    auto result = op.execute({&input}, {}, {});

    ASSERT_EQ(Status::OK(), result->status());
    auto z = result->at(0);    
    //z->printIndexedBuffer("Result is ");
    ASSERT_TRUE(exp.equalsTo(z));
    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Sum_2) {
    
    NDArray<float> input('c', {3, 5},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.});    
    NDArray<float> exp({15.f, 40.f, 65.f});
    //************************************//

    nd4j::ops::reduce_sum<float> op;
    auto result = op.execute({&input}, {}, {1});

    ASSERT_EQ(Status::OK(), result->status());
    auto z = result->at(0);    
//    z->printIndexedBuffer("Result is ");
    ASSERT_TRUE(exp.equalsTo(z));
    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Prod_1) {
    
    NDArray<float> input('c', {3, 5},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.});    
    NDArray<float> exp(1307674368000.f);
    //************************************//

    nd4j::ops::reduce_prod<float> op;
    auto result = op.execute({&input}, {}, {});

    ASSERT_EQ(Status::OK(), result->status());
    auto z = result->at(0);    
    //z->printIndexedBuffer("Result is ");
    ASSERT_TRUE(exp.equalsTo(z));
    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Prod_2) {
    
    NDArray<float> input('c', {3, 5},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.});    
    NDArray<float> exp({120.f, 30240.f, 360360.f});
    //************************************//

    nd4j::ops::reduce_prod<float> op;
    auto result = op.execute({&input}, {}, {1});

    ASSERT_EQ(Status::OK(), result->status());
    auto z = result->at(0);    
//    z->printIndexedBuffer("Result is ");
    ASSERT_TRUE(exp.equalsTo(z));
    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Sum_01) {

    NDArray<float> x('c', {2,3,4});
    NDArray<float> exp('c', {4}, {66.f, 72.f, 78.f, 84.f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_sum<float> op;
    auto result = op.execute({&x}, {}, {0,1});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Sum_02) {

    NDArray<float> x('c', {2,3,4});
    NDArray<float> exp('c', {1,1,4}, {66.f, 72.f, 78.f, 84.f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_sum<float> op;
    auto result = op.execute({&x}, {1.}, {0, 1});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Sum_3) {

    NDArray<float> x('c', {2,3,4});
    NDArray<float> exp('c', {3}, {68.f, 100.f, 132.f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_sum<float> op;
    auto result = op.execute({&x}, {}, {0, 2});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Sum_4) {

    NDArray<float> x('c', {2,3,4});
    NDArray<float> exp('c', {1,3,1}, {68.f, 100.f, 132.f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_sum<float> op;
    auto result = op.execute({&x}, {1.}, {0, 2});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Sum_5) {

    NDArray<float> x('c', {2,3,4});
    NDArray<float> exp(300.f);
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_sum<float> op;
    auto result = op.execute({&x}, {}, {});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Sum_6) {

    NDArray<float> x('c', {2,3,4});
    NDArray<float> exp(300.f);
    NDArrayFactory<float>::linspace(1, x);
           
    nd4j::ops::reduce_sum<float> op;
    auto result = op.execute({&x}, {}, {0,1,2});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Sum_7) {

    NDArray<float> x('c', {2,3,4});
    NDArray<float> exp('c', {1,1,1}, {300.f});
    NDArrayFactory<float>::linspace(1, x);
//    x.printIndexedBuffer("Input with shape (2, 3, 4) is");       
    nd4j::ops::reduce_sum<float> op;
    auto result = op.execute({&x}, {1.}, {0,1,2});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Prod_01) {

    NDArray<float> x('c', {2,3,2});
    NDArray<float> exp('c', {2}, {10395.f, 46080.f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_prod<float> op;
    auto result = op.execute({&x}, {}, {0,1});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Prod_02) {

    NDArray<float> x('c', {2,3,2});
    NDArray<float> exp('c', {1,1,2}, {10395.f, 46080.f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_prod<float> op;
    auto result = op.execute({&x}, {1.}, {0, 1});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Prod_3) {

    NDArray<float> x('c', {2,3,2});
    NDArray<float> exp('c', {3}, {112.f, 1080.f, 3960.f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_prod<float> op;
    auto result = op.execute({&x}, {}, {0, 2});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Prod_4) {

    NDArray<float> x('c', {2,3,2});
    NDArray<float> exp('c', {1,3,1}, {112.f, 1080.f, 3960.f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_prod<float> op;
    auto result = op.execute({&x}, {1.}, {0, 2});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Prod_5) {

    NDArray<float> x('c', {2,3,2});
    NDArray<float> exp(479001600.f);
    NDArrayFactory<float>::linspace(1, x);
    
    nd4j::ops::reduce_prod<float> op;
    auto result = op.execute({&x}, {}, {});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Prod_6) {

    NDArray<float> x('c', {2,3,2});
    NDArray<float> exp(479001600.f);
    NDArrayFactory<float>::linspace(1, x);
           
    nd4j::ops::reduce_prod<float> op;
    auto result = op.execute({&x}, {}, {0,1,2});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Prod_7) {

    NDArray<float> x('c', {2,3,2});
    NDArray<float> exp('c', {1, 1, 1}, {479001600.f});
    NDArrayFactory<float>::linspace(1, x);
//    x.printIndexedBuffer("Input with shape (2, 3, 4) is");       
    nd4j::ops::reduce_prod<float> op;
    auto result = op.execute({&x}, {1.}, {0,1,2});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Min_1) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {4}, {1.f, 2.f, 3.f, 4.f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_min<float> op;
    auto result = op.execute({&x}, {}, {0, 1});
    auto output = result->at(0);    
    // output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Min_2) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {1,1,4}, {1.f, 2.f, 3.f, 4.f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_min<float> op;
    auto result = op.execute({&x}, {1.}, {0, 1});
    auto output = result->at(0);    
    // output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Min_3) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {3}, {1.f, 5.f, 9.f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_min<float> op;
    auto result = op.execute({&x}, {}, {0, 2});
    auto output = result->at(0);    
    // output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Min_4) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {1,3,1}, {1.f, 5.f, 9.f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_min<float> op;
    auto result = op.execute({&x}, {1.}, {0, 2});
    auto output = result->at(0);    
    // output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Min_5) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp(1.f);
    NDArrayFactory<float>::linspace(1, x);
    
    nd4j::ops::reduce_min<float> op;
    auto result = op.execute({&x}, {}, {});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Min_6) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp(1.f);
    NDArrayFactory<float>::linspace(1, x);
           
    nd4j::ops::reduce_min<float> op;
    auto result = op.execute({&x}, {}, {0,1,2});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Min_7) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {1, 1, 1}, {1.f});
    NDArrayFactory<float>::linspace(1, x);
    // x.printIndexedBuffer("Input with shape (2, 3, 4) is");       
    nd4j::ops::reduce_min<float> op;
    auto result = op.execute({&x}, {1.}, {0,1,2});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Max_1) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {4}, {21.f, 22.f, 23.f, 24.f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_max<float> op;
    auto result = op.execute({&x}, {}, {0,1});
    auto output = result->at(0);    
    // output->printIndexedBuffer("Result is");
    // output->printShapeInfo("Output shape");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Max_2) {

    NDArray<float> x('c', {2,3,4});
    NDArray<float> exp('c', {1,1,4}, {21.f, 22.f, 23.f, 24.f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_max<float> op;
    auto result = op.execute({&x}, {1.}, {0, 1});
    auto output = result->at(0);    
    // output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Max_3) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {3}, {16.f, 20.f, 24.f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_max<float> op;
    auto result = op.execute({&x}, {}, {0, 2});
    auto output = result->at(0);    
    // output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Max_4) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {1,3,1}, {16.f, 20.f, 24.f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_max<float> op;
    auto result = op.execute({&x}, {1.}, {0, 2});
    auto output = result->at(0);    
    // output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Max_5) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp(24.f);
    NDArrayFactory<float>::linspace(1, x);
    
    nd4j::ops::reduce_max<float> op;
    auto result = op.execute({&x}, {}, {});
    auto output = result->at(0);    
    // output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Max_6) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp(24.f);
    NDArrayFactory<float>::linspace(1, x);
           
    nd4j::ops::reduce_max<float> op;
    auto result = op.execute({&x}, {}, {0,1,2});
    auto output = result->at(0);    
    // output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Max_7) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {1, 1, 1}, {24.f});
    NDArrayFactory<float>::linspace(1, x);
//    x.printIndexedBuffer("Input with shape (2, 3, 4) is");       
    nd4j::ops::reduce_max<float> op;
    auto result = op.execute({&x}, {1.}, {0,1,2});
    auto output = result->at(0);    
    // output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}


////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Dot_1) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {4}, {1.f, 2.f, 3.f, 4.f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_dot_bp<float> op;
    auto result = op.execute({&x, &exp}, {}, {0,1});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(x.isSameShape(output));
//    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}
/*
////////////////////////////////////////////////////////////////////////////////

TEST_F(DeclarableOpsTests8, Test_Reduce_Dot_2) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {1,1,4}, {1.f, 2.f, 3.f, 4.f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_dot<float> op;
    auto result = op.execute({&x}, {1.}, {0, 1});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Dot_3) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {3}, {112.f, 1080.f, 3960.f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_dot<float> op;
    auto result = op.execute({&x}, {}, {0, 2});
    auto output = result->at(0);    
    // output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Dot_4) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {1,3,1}, {112.f, 1080.f, 3960.f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_dot<float> op;
    auto result = op.execute({&x}, {1.}, {0, 2});
    auto output = result->at(0);    
    //output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Dot_5) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp(479001600.f);
    NDArrayFactory<float>::linspace(1, x);
    
    nd4j::ops::reduce_dot<float> op;
    auto result = op.execute({&x}, {}, {});
    auto output = result->at(0);    
    //output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Dot_6) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp(479001600.f);
    NDArrayFactory<float>::linspace(1, x);
           
    nd4j::ops::reduce_dot<float> op;
    auto result = op.execute({&x}, {}, {0,1,2});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Dot_7) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {1, 1, 1}, {479001600.f});
    NDArrayFactory<float>::linspace(1, x);
//    x.printIndexedBuffer("Input with shape (2, 3, 4) is");       
    nd4j::ops::reduce_dot<float> op;
    auto result = op.execute({&x}, {1.}, {0,1,2});
    auto output = result->at(0);    
    //output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}
*/

TEST_F(DeclarableOpsTests8, Test_Reduce_Norm1_1) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {4}, {66.f, 72.f, 78.f, 84.f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_norm1<float> op;
    auto result = op.execute({&x}, {}, {0,1});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Norm1_2) {

    NDArray<float> x('c', {2,3,4});
    NDArray<float> exp('c', {1,1,4}, {66.f, 72.f, 78.f, 84.f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_norm1<float> op;
    auto result = op.execute({&x}, {1.}, {0, 1});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Norm1_3) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {3}, {68.f, 100.f, 132.f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_norm1<float> op;
    auto result = op.execute({&x}, {}, {0, 2});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Norm1_4) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {1,3,1}, {68.f, 100.f, 132.f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_norm1<float> op;
    auto result = op.execute({&x}, {1.}, {0, 2});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Norm1_5) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp(300.f);
    NDArrayFactory<float>::linspace(1, x);
    
    nd4j::ops::reduce_norm1<float> op;
    auto result = op.execute({&x}, {}, {});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Norm1_6) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp(300.f);
    NDArrayFactory<float>::linspace(1, x);
           
    nd4j::ops::reduce_norm1<float> op;
    auto result = op.execute({&x}, {}, {0,1,2});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Norm1_7) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {1, 1, 1}, {300.f});
    NDArrayFactory<float>::linspace(1, x);
//    x.printIndexedBuffer("Input with shape (2, 3, 4) is");       
    nd4j::ops::reduce_norm1<float> op;
    auto result = op.execute({&x}, {1.}, {0,1,2});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

TEST_F(DeclarableOpsTests8, Test_Reduce_Norm2_1) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {4}, {31.7175f, 33.823071f, 35.97221f, 38.15757f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_norm2<float> op;
    auto result = op.execute({&x}, {}, {0,1});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Norm2_2) {

    NDArray<float> x('c', {2,3,4});
    NDArray<float> exp('c', {1,1,4}, {31.7175f, 33.823071f, 35.97221f, 38.15757f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_norm2<float> op;
    auto result = op.execute({&x}, {1.}, {0, 1});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Norm2_3) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {3}, {29.597298f, 39.344631f, 49.759422f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_norm2<float> op;
    auto result = op.execute({&x}, {}, {0, 2});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Norm2_4) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {1,3,1}, {29.597298f, 39.344631f, 49.759422f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_norm2<float> op;
    auto result = op.execute({&x}, {1.}, {0, 2});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Norm2_5) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp(70.f);
    NDArrayFactory<float>::linspace(1, x);
    
    nd4j::ops::reduce_norm2<float> op;
    auto result = op.execute({&x}, {}, {});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Norm2_6) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp(70.f);
    NDArrayFactory<float>::linspace(1, x);
           
    nd4j::ops::reduce_norm2<float> op;
    auto result = op.execute({&x}, {}, {0,1,2});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Norm2_7) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {1, 1, 1}, {70.f});
    NDArrayFactory<float>::linspace(1, x);
//    x.printIndexedBuffer("Input with shape (2, 3, 4) is");       
    nd4j::ops::reduce_norm2<float> op;
    auto result = op.execute({&x}, {1.}, {0,1,2});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_NormMax_1) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {4}, {21.f, 22.f, 23.f, 24.f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_norm_max<float> op;
    auto result = op.execute({&x}, {}, {0,1});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_NormMax_2) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {1,1,4}, {21.f, 22.f, 23.f, 24.f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_norm_max<float> op;
    auto result = op.execute({&x}, {1.f}, {0,1});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_NormMax_3) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {3}, {16.f, 20.f, 24.f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_norm_max<float> op;
    auto result = op.execute({&x}, {}, {0,2});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_NormMax_4) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {1, 3, 1}, {16.f, 20.f, 24.f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_norm_max<float> op;
    auto result = op.execute({&x}, {1.f}, {0,2});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_NormMax_5) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp(24.f);
    NDArrayFactory<float>::linspace(1, x);
    
    nd4j::ops::reduce_norm_max<float> op;
    auto result = op.execute({&x}, {}, {});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_NormMax_6) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp(24.f);
    NDArrayFactory<float>::linspace(1, x);
    
    nd4j::ops::reduce_norm_max<float> op;
    auto result = op.execute({&x}, {}, {0, 1, 2});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_NormMax_7) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {1, 1, 1}, {24.f});
    NDArrayFactory<float>::linspace(1, x);
    
    nd4j::ops::reduce_norm_max<float> op;
    auto result = op.execute({&x}, {1.f}, {});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_SquaredNorm_1) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {4}, {1006.f, 1144.f, 1294.f, 1456.f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_sqnorm<float> op;
    auto result = op.execute({&x}, {}, {0,1});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_SquaredNorm_2) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {1,1,4}, {1006.f, 1144.f, 1294.f, 1456.f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_sqnorm<float> op;
    auto result = op.execute({&x}, {1.f}, {0,1});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_SquaredNorm_3) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {3}, {876.f, 1548.f, 2476.f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_sqnorm<float> op;
    auto result = op.execute({&x}, {}, {0,2});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_SquaredNorm_4) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {1, 3, 1}, {876.f, 1548.f, 2476.f});
    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::reduce_sqnorm<float> op;
    auto result = op.execute({&x}, {1.f}, {0,2});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_SquaredNorm_5) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp(4900.f);
    NDArrayFactory<float>::linspace(1, x);
    
    nd4j::ops::reduce_sqnorm<float> op;
    auto result = op.execute({&x}, {}, {});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_SquaredNorm_6) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp(4900.f);
    NDArrayFactory<float>::linspace(1, x);
    
    nd4j::ops::reduce_sqnorm<float> op;
    auto result = op.execute({&x}, {}, {0, 1, 2});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_SquaredNorm_7) {

    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {1, 1, 1}, {4900.f});
    NDArrayFactory<float>::linspace(1, x);
    
    nd4j::ops::reduce_sqnorm<float> op;
    auto result = op.execute({&x}, {1.f}, {});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Sum_BP_1) {
    
    NDArray<float> input('c', {3, 4},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.});    
    NDArray<float> eps(0.5f);
    NDArray<float> exp('c', {3, 4}, {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,0.5f});
    //************************************//

    nd4j::ops::reduce_sum_bp<float> op;
    auto result = op.execute({&input, &eps}, {}, {});

    ASSERT_EQ(Status::OK(), result->status());
    auto z = result->at(0);    
//    z->printIndexedBuffer("Result is ");
//    z->printShapeInfo();
    ASSERT_TRUE(exp.equalsTo(z));
    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Sum_BP_2) {
    
    NDArray<float> input('c', {3, 4},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.});    
    NDArray<float> eps('c', {1, 1}, {0.5f});
    NDArray<float> exp('c', {3, 4}, {0.5f, 0.5f, 0.5f, 0.5f, 
                                     0.5f, 0.5f, 0.5f, 0.5f, 
                                     0.5f, 0.5f, 0.5f,0.5f});
    //************************************//

    nd4j::ops::reduce_sum_bp<float> op;
    auto result = op.execute({&input, &eps}, {1.f}, {});

    ASSERT_EQ(Status::OK(), result->status());
    auto z = result->at(0);    
//  z->printIndexedBuffer("Result is ");
//  z->printShapeInfo();
    ASSERT_TRUE(exp.equalsTo(z));
    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Sum_BP_3) {
    
    NDArray<float> input('c', {3, 4},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.});    
    NDArray<float> eps('c', {4}, {1.f, 2.f, 3.f, 4.f});
    NDArray<float> exp('c', {3, 4}, {1.f, 2.f, 3.f, 4.f, 
                                     1.f, 2.f, 3.f, 4.f, 
                                     1.f, 2.f, 3.f, 4.f});
    //************************************//

    nd4j::ops::reduce_sum_bp<float> op;
    auto result = op.execute({&input, &eps}, {}, {0});

    ASSERT_EQ(Status::OK(), result->status());
    auto z = result->at(0);    
//    z->printIndexedBuffer("Result is ");
//    z->printShapeInfo();
    ASSERT_TRUE(exp.equalsTo(z));
    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Sum_BP_4) {
    
    NDArray<float> input('c', {3, 4},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.});    
    NDArray<float> eps('c', {1, 4}, {1.f, 2.f, 3.f, 4.f});
    NDArray<float> exp('c', {3, 4}, {1.f, 2.f, 3.f, 4.f, 
                                     1.f, 2.f, 3.f, 4.f, 
                                     1.f, 2.f, 3.f, 4.f});
    //************************************//

    nd4j::ops::reduce_sum_bp<float> op;
    auto result = op.execute({&input, &eps}, {1.f}, {0});

    ASSERT_EQ(Status::OK(), result->status());
    auto z = result->at(0);    
//    z->printIndexedBuffer("Result is ");
//    z->printShapeInfo();
    ASSERT_TRUE(exp.equalsTo(z));
    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Prod_BP_1) {
    
    NDArray<float> input('c', {3, 5},   {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f});    
    NDArray<float> eps(1307674368000.f);
    //************************************//
//    NDArray<float> exp('c', {3, 4}, {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,0.5f});
    //************************************//
    NDArray<float> exp('c', {3, 5},   {1710012166826558903812096.f, 855006083413279451906048.f, 570004067618451974258688.f, 
                                       427503041706639725953024.f, 342002454982589992140800.f, 285002033809225987129344.f, 
                                       244287457550765131825152.f, 213751520853319862976512.f, 190001355872817324752896.f, 
                                       171001227491294996070400.f, 155455648254341989531648.f, 142501016904612993564672.f, 
                                       131539399526781282156544.f, 122143728775382565912576.f, 114000815325130245799936.f});    

    nd4j::ops::reduce_prod_bp<float> op;
    auto result = op.execute({&input, &eps}, {}, {});

    ASSERT_EQ(Status::OK(), result->status());
    auto z = result->at(0);    
//    z->printIndexedBuffer("Result is ");
//    z->printShapeInfo();
    ASSERT_TRUE(exp.equalsTo(z));
    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceMean_test1) {

    NDArray<float> x('c', {2,3,4});
    NDArray<float> exp('c', {4}, {11.f, 12.f, 13.f, 14.f});
    NDArrayFactory<float>::linspace(1, x);
    
        
    nd4j::ops::reduce_mean<float> op;
    auto result = op.execute({&x}, {}, {0,1});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceMean_test2) {

    NDArray<float> x('c', {2,3,4});
    NDArray<float> exp('c', {1,1,4}, {11.f, 12.f, 13.f, 14.f});
    NDArrayFactory<float>::linspace(1, x);
    
        
    nd4j::ops::reduce_mean<float> op;
    auto result = op.execute({&x}, {1.}, {0,1});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceMean_test3) {

    NDArray<float> x('c', {2,3,4});
    NDArray<float> exp('c', {3}, {8.5f, 12.5f, 16.5f});
    NDArrayFactory<float>::linspace(1, x);
    
        
    nd4j::ops::reduce_mean<float> op;
    auto result = op.execute({&x}, {}, {0,2});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceMean_test4) {

    NDArray<float> x('c', {2,3,4});
    NDArray<float> exp('c', {1,3,1}, {8.5f, 12.5f, 16.5f});
    NDArrayFactory<float>::linspace(1, x);
    
        
    nd4j::ops::reduce_mean<float> op;
    auto result = op.execute({&x}, {1.f}, {0,2});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceMean_test5) {

    NDArray<float> x('c', {2,3,4});
    NDArray<float> exp(12.5f);
    NDArrayFactory<float>::linspace(1, x);
    
        
    nd4j::ops::reduce_mean<float> op;
    auto result = op.execute({&x}, {}, {});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceMean_test6) {

    NDArray<float> x('c', {2,3,4});
    NDArray<float> exp(12.5f);
    NDArrayFactory<float>::linspace(1, x);
           
    nd4j::ops::reduce_mean<float> op;
    auto result = op.execute({&x}, {}, {0,1,2});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceMean_test7) {

    NDArray<float> x('c', {2,3,4});
    NDArray<float> exp('c', {1,1,1}, {12.5f});
    NDArrayFactory<float>::linspace(1, x);
           
    nd4j::ops::reduce_mean<float> op;
    auto result = op.execute({&x}, {1.}, {0,1,2});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceMeanBP_test1) {

    NDArray<float> x('c', {3,4});
    NDArray<float> gradO1(0.5f);
    NDArray<float> gradO2('c', {1,1}, {0.5f});
    NDArray<float> exp('c', {3,4}, {1./24, 1./24, 1./24, 1./24, 1./24, 1./24, 1./24, 1./24, 1./24, 1./24, 1./24, 1./24});

    NDArrayFactory<float>::linspace(1, x);
            
    nd4j::ops::reduce_mean_bp<float> op;

    auto result = op.execute({&x, &gradO1}, {0}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    
    auto output = result->at(0);
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO2}, {1}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    
    output = result->at(0);
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
    delete result;
}


////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceMeanBP_test3) {

    NDArray<float> x('c', {3,4});
    NDArray<float> gradO1('c', {4},  {1.f, 2.f, 3.f, 4.f});
    NDArray<float> gradO2('c', {1,4}, {1.f, 2.f, 3.f, 4.f});
    NDArray<float> exp('c', {3,4}, {1.f/3.f, 2.f/3.f, 1.f, 4.f/3.f, 1.f/3.f, 2.f/3.f, 1.f, 4.f/3.f, 1.f/3.f, 2.f/3.f, 1.f, 4.f/3.f});

    NDArrayFactory<float>::linspace(1, x);
            
    nd4j::ops::reduce_mean_bp<float> op;

    auto result = op.execute({&x, &gradO1}, {0}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    
    auto output = result->at(0);
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO2}, {1}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    
    output = result->at(0);
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceMeanBP_test5) {

    NDArray<float> x('c', {3,4});
    NDArray<float> gradO1('c', {3}, {1.f, 2.f, 3.f});
    NDArray<float> gradO2('c', {3,1}, {1.f, 2.f, 3.f});
    NDArray<float> exp('c', {3,4}, {0.25f, 0.25f, 0.25f, 0.25f, 0.5f, 0.5f, 0.5f, 0.5f, 0.75f, 0.75f, 0.75f, 0.75f});

    NDArrayFactory<float>::linspace(1, x);
            
    nd4j::ops::reduce_mean_bp<float> op;

    auto result = op.execute({&x, &gradO1}, {0}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    
    auto output = result->at(0);
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO2}, {1}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    
    output = result->at(0);
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDevBP_test4) {

    NDArray<float> x('c', {3}, {2.f, 3.f, 4.f});
    NDArray<float> gradO(0.5f);    
    NDArray<float> exp('c', {3}, {-0.25f, 0.f, 0.25f});    
            
    nd4j::ops::reduce_stdev_bp<float> op;

    auto result = op.execute({&x, &gradO}, {0,1}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());        
    auto output = result->at(0);        

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
    delete result;

}
