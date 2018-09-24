/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

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

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.f});
    auto exp = NDArrayFactory::create<double>('c', {4}, {602.2222f, 727.13885f, 993.5555f, 755.8889f});
        
    nd4j::ops::reduce_variance op;
    auto result = op.execute({&x}, {}, {0,1});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceVariance_test2) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.f});
    auto exp = NDArrayFactory::create<double>('c', {1,1,4}, {602.2222f, 727.13885f, 993.5555f, 755.8889f});
    
    nd4j::ops::reduce_variance op;
    auto result = op.execute({&x}, {1.}, {0,1});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceVariance_test3) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.f});
    auto exp = NDArrayFactory::create<double>('c', {3}, {900.9375f, 969.8594f, 424.1875f});
        
    nd4j::ops::reduce_variance op;
    auto result = op.execute({&x}, {}, {0,2});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceVariance_test4) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.f});
    auto exp = NDArrayFactory::create<double>('c', {1,3,1}, {900.9375f, 969.8594f, 424.1875f});  
        
    nd4j::ops::reduce_variance op;
    auto result = op.execute({&x}, {1.}, {0,2});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceVariance_test5) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.f});
    auto exp = NDArrayFactory::create<double>(788.6927f);
        
    nd4j::ops::reduce_variance op;
    auto result = op.execute({&x}, {}, {});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceVariance_test6) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    auto exp = NDArrayFactory::create<double>(788.6927f);
           
    nd4j::ops::reduce_variance op;
    auto result = op.execute({&x}, {}, {0,1,2});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceVariance_test7) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    auto exp = NDArrayFactory::create<double>('c', {1,1,1}, {788.6927f});
           
    nd4j::ops::reduce_variance op;
    auto result = op.execute({&x}, {1.}, {0,1,2});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDev_test1) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    auto exp = NDArrayFactory::create<double>('c', {4}, {24.54022f, 26.96551f, 31.52072f, 27.49343f});
        
    nd4j::ops::reduce_stdev op;
    auto result = op.execute({&x}, {}, {0,1});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDev_test2) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    auto exp = NDArrayFactory::create<double>('c', {1,1,4}, {24.54022f, 26.96551f, 31.52072f, 27.49343f});
    
    nd4j::ops::reduce_stdev op;
    auto result = op.execute({&x}, {1.}, {0,1});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDev_test3) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    auto exp = NDArrayFactory::create<double>('c', {3}, {30.01562f, 31.14257f, 20.59581f});
        
    nd4j::ops::reduce_stdev op;
    auto result = op.execute({&x}, {}, {0,2});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDev_test4) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    auto exp = NDArrayFactory::create<double>('c', {1,3,1}, {30.01562f, 31.14257f, 20.59581f});  
        
    nd4j::ops::reduce_stdev op;
    auto result = op.execute({&x}, {1.}, {0,2});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDev_test5) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    auto exp = NDArrayFactory::create<double>(28.08367f);
        
    nd4j::ops::reduce_stdev op;
    auto result = op.execute({&x}, {}, {});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDev_test6) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    auto exp = NDArrayFactory::create<double>(28.08367f);
           
    nd4j::ops::reduce_stdev op;
    auto result = op.execute({&x}, {}, {0,1,2});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDev_test7) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    auto exp = NDArrayFactory::create<double>('c', {1,1,1}, {28.08367f});
           
    nd4j::ops::reduce_stdev op;
    auto result = op.execute({&x}, {1.f}, {0,1,2});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDev_test8) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    auto exp = NDArrayFactory::create<double>('c', {4}, {26.88246f, 29.53924f, 34.52921f, 30.11755f});
        
    nd4j::ops::reduce_stdev op;
    auto result = op.execute({&x}, {0.f,1.f}, {0,1});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceVarianceBP_test1) {

    auto x = NDArrayFactory::create<double>('c', {3,4});
    auto gradO1 = NDArrayFactory::create<double>('c', {1,1}, {0.5f});
    auto gradO2 = NDArrayFactory::create<double>(0.5f);
    auto exp12 = NDArrayFactory::create<double>('c', {3,4}, {-0.5f, -0.4090909f, -0.3181818f, -0.22727273f, -0.13636364f, -0.045454547f, 0.045454547f, 0.13636364f, 0.22727273f, 0.3181818f, 0.4090909f, 0.5f});
    auto exp34 = NDArrayFactory::create<double>('c', {3,4}, {-0.45833334f, -0.375f, -0.29166666f, -0.20833333f, -0.125f, -0.041666668f, 0.041666668f, 0.125f, 0.20833333f, 0.29166666f, 0.375f, 0.45833334f});

    x.linspace(1);
            
    nd4j::ops::reduce_variance_bp op;

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

    auto x = NDArrayFactory::create<double>('c', {3,4});
    auto gradO1 = NDArrayFactory::create<double>('c', {1,4}, {1.f,2.f,3.f,4.f});
    auto gradO2 = NDArrayFactory::create<double>('c', {4}, {1.f,2.f,3.f,4.f});
    auto exp12 = NDArrayFactory::create<double>('c', {3,4}, {-2.666667f, -5.333333f, -8.000000f,  -10.666667f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 2.666667f, 5.333333f,  8.000000f, 10.666667f});
    auto exp34 = NDArrayFactory::create<double>('c', {3,4}, {-4.000000f, -8.000000f, -12.000000f, -16.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 4.000000f, 8.000000f, 12.000000f, 16.000000f});

    x.linspace(1);
            
    nd4j::ops::reduce_variance_bp op;

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

    auto x = NDArrayFactory::create<double>('c', {3,4});
    auto gradO1 = NDArrayFactory::create<double>('c', {3,1}, {1.f,2.f,3.f});
    auto gradO2 = NDArrayFactory::create<double>('c', {3}, {1.f,2.f,3.f});
    auto exp12 = NDArrayFactory::create<double>('c', {3,4}, {-0.750000f, -0.250000f, 0.250000f, 0.750000f, -1.500000f, -0.500000f, 0.500000f, 1.500000f, -2.250000f, -0.750000f, 0.750000f, 2.250000f});
    auto exp34 = NDArrayFactory::create<double>('c', {3,4}, {-1.000000f, -0.333333f, 0.333333f, 1.000000f, -2.000000f, -0.666667f, 0.666667f, 2.000000f, -3.000000f, -1.000000f, 1.000000f, 3.000000f});

    x.linspace(1);
            
    nd4j::ops::reduce_variance_bp op;

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

    auto x = NDArrayFactory::create<double>('c', {3,4});
    auto gradO1 = NDArrayFactory::create<double>('c', {1,1}, {0.5f});
    auto gradO2 = NDArrayFactory::create<double>(0.5f);
    auto exp12 = NDArrayFactory::create<double>('c', {3,4}, {-0.069337524f, -0.056730703f, -0.04412388f, -0.031517055f, -0.018910235f, -0.0063034114f, 0.0063034114f, 0.018910235f, 0.031517055f, 0.04412388f, 0.056730703f, 0.069337524f});     
    auto exp34 = NDArrayFactory::create<double>('c', {3,4}, {-0.06638563f, -0.05431551f, -0.0422454f, -0.030175284f, -0.01810517f, -0.006035057f, 0.006035057f, 0.01810517f, 0.030175284f, 0.0422454f, 0.05431551f, 0.06638563f});

    x.linspace(1);
            
    nd4j::ops::reduce_stdev_bp op;

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

    auto x = NDArrayFactory::create<double>('c', {3,4});
    auto gradO1 = NDArrayFactory::create<double>('c', {1,4}, {1.f,2.f,3.f,4.f});
    auto gradO2 = NDArrayFactory::create<double>('c', {4}, {1.f,2.f,3.f,4.f});
    auto exp12 = NDArrayFactory::create<double>('c', {3,4}, {-0.4082483f, -0.8164966f, -1.2247449f, -1.6329932f, 0.0, 0.0, 0.0, 0.0, 0.4082483f, 0.8164966f, 1.2247449f, 1.6329932f});
    auto exp34 = NDArrayFactory::create<double>('c', {3,4}, {-0.5f, -1.0f, -1.5f, -2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f});

    x.linspace(1);
            
    nd4j::ops::reduce_stdev_bp op;

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

    auto x = NDArrayFactory::create<double>('c', {3,4});
    auto gradO1 = NDArrayFactory::create<double>('c', {3,1}, {1.f,2.f,3.f});
    auto gradO2 = NDArrayFactory::create<double>('c', {3}, {1.f,2.f,3.f});
    auto exp12 = NDArrayFactory::create<double>('c', {3,4}, {-0.3354102f, -0.1118034f, 0.1118034f, 0.3354102f, -0.6708204f, -0.2236068f, 0.2236068f, 0.6708204f, -1.0062306f, -0.3354102f, 0.3354102f, 1.0062306f});
    auto exp34 = NDArrayFactory::create<double>('c', {3,4}, {-0.38729835f, -0.12909944f, 0.12909944f, 0.38729835f, -0.7745967f, -0.2581989f, 0.2581989f, 0.7745967f, -1.161895f, -0.38729835f, 0.38729835f, 1.161895f});

    x.linspace(1);
            
    nd4j::ops::reduce_stdev_bp op;

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
    
    auto input = NDArrayFactory::create<double>('c', {3, 5},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.});    
    auto exp = NDArrayFactory::create<double>(120.f);
    //************************************//

    nd4j::ops::reduce_sum op;
    auto result = op.execute({&input}, {}, {});

    ASSERT_EQ(Status::OK(), result->status());
    auto z = result->at(0);    
    //z->printIndexedBuffer("Result is ");
    ASSERT_TRUE(exp.equalsTo(z));
    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Sum_2) {
    
    auto input = NDArrayFactory::create<double>('c', {3, 5},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.});    
    auto exp = NDArrayFactory::create<double>({15.f, 40.f, 65.f});
    //************************************//

    nd4j::ops::reduce_sum op;
    auto result = op.execute({&input}, {}, {1});

    ASSERT_EQ(Status::OK(), result->status());
    auto z = result->at(0);    
//    z->printIndexedBuffer("Result is ");
    ASSERT_TRUE(exp.equalsTo(z));
    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Prod_1) {
    
    auto input = NDArrayFactory::create<double>('c', {3, 5},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.});    
    auto exp = NDArrayFactory::create<double>(1307674368000.f);
    //************************************//

    nd4j::ops::reduce_prod op;
    auto result = op.execute({&input}, {}, {});

    ASSERT_EQ(Status::OK(), result->status());
    auto z = result->at(0);    
    //z->printIndexedBuffer("Result is ");
    ASSERT_TRUE(exp.equalsTo(z));
    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Prod_2) {
    
    auto input = NDArrayFactory::create<double>('c', {3, 5},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.});    
    auto exp = NDArrayFactory::create<double>({120.f, 30240.f, 360360.f});
    //************************************//

    nd4j::ops::reduce_prod op;
    auto result = op.execute({&input}, {}, {1});

    ASSERT_EQ(Status::OK(), result->status());
    auto z = result->at(0);    
//    z->printIndexedBuffer("Result is ");
    ASSERT_TRUE(exp.equalsTo(z));
    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Sum_01) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>('c', {4}, {66.f, 72.f, 78.f, 84.f});
    x.linspace(1);

    nd4j::ops::reduce_sum op;
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

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>('c', {1,1,4}, {66.f, 72.f, 78.f, 84.f});
    x.linspace(1);

    nd4j::ops::reduce_sum op;
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

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>('c', {3}, {68.f, 100.f, 132.f});
    x.linspace(1);

    nd4j::ops::reduce_sum op;
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

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>('c', {1,3,1}, {68.f, 100.f, 132.f});
    x.linspace(1);

    nd4j::ops::reduce_sum op;
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

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>(300.f);
    x.linspace(1);

    nd4j::ops::reduce_sum op;
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

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>(300.f);
    x.linspace(1);
           
    nd4j::ops::reduce_sum op;
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

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>('c', {1,1,1}, {300.f});
    x.linspace(1);
//    x.printIndexedBuffer("Input with shape (2, 3, 4) is");       
    nd4j::ops::reduce_sum op;
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

    auto x = NDArrayFactory::create<double>('c', {2,3,2});
    auto exp = NDArrayFactory::create<double>('c', {2}, {10395.f, 46080.f});
    x.linspace(1);

    nd4j::ops::reduce_prod op;
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

    auto x = NDArrayFactory::create<double>('c', {2,3,2});
    auto exp = NDArrayFactory::create<double>('c', {1,1,2}, {10395.f, 46080.f});
    x.linspace(1);

    nd4j::ops::reduce_prod op;
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

    auto x = NDArrayFactory::create<double>('c', {2,3,2});
    auto exp = NDArrayFactory::create<double>('c', {3}, {112.f, 1080.f, 3960.f});
    x.linspace(1);

    nd4j::ops::reduce_prod op;
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

    auto x = NDArrayFactory::create<double>('c', {2,3,2});
    auto exp = NDArrayFactory::create<double>('c', {1,3,1}, {112.f, 1080.f, 3960.f});
    x.linspace(1);

    nd4j::ops::reduce_prod op;
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

    auto x = NDArrayFactory::create<double>('c', {2,3,2});
    auto exp = NDArrayFactory::create<double>(479001600.f);
    x.linspace(1);
    
    nd4j::ops::reduce_prod op;
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

    auto x = NDArrayFactory::create<double>('c', {2,3,2});
    auto exp = NDArrayFactory::create<double>(479001600.f);
    x.linspace(1);
           
    nd4j::ops::reduce_prod op;
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

    auto x = NDArrayFactory::create<double>('c', {2,3,2});
    auto exp = NDArrayFactory::create<double>('c', {1, 1, 1}, {479001600.f});
    x.linspace(1);
//    x.printIndexedBuffer("Input with shape (2, 3, 4) is");       
    nd4j::ops::reduce_prod op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {4}, {1.f, 2.f, 3.f, 4.f});
    x.linspace(1);

    nd4j::ops::reduce_min op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1,1,4}, {1.f, 2.f, 3.f, 4.f});
    x.linspace(1);

    nd4j::ops::reduce_min op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {3}, {1.f, 5.f, 9.f});
    x.linspace(1);

    nd4j::ops::reduce_min op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1,3,1}, {1.f, 5.f, 9.f});
    x.linspace(1);

    nd4j::ops::reduce_min op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(1.f);
    x.linspace(1);
    
    nd4j::ops::reduce_min op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(1.f);
    x.linspace(1);
           
    nd4j::ops::reduce_min op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1, 1, 1}, {1.f});
    x.linspace(1);
    // x.printIndexedBuffer("Input with shape (2, 3, 4) is");       
    nd4j::ops::reduce_min op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {4}, {21.f, 22.f, 23.f, 24.f});
    x.linspace(1);

    nd4j::ops::reduce_max op;
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

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>('c', {1,1,4}, {21.f, 22.f, 23.f, 24.f});
    x.linspace(1);

    nd4j::ops::reduce_max op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {3}, {16.f, 20.f, 24.f});
    x.linspace(1);

    nd4j::ops::reduce_max op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1,3,1}, {16.f, 20.f, 24.f});
    x.linspace(1);

    nd4j::ops::reduce_max op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(24.f);
    x.linspace(1);
    
    nd4j::ops::reduce_max op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(24.f);
    x.linspace(1);
           
    nd4j::ops::reduce_max op;
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

	auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1, 1, 1}, {24.f});
    x.linspace(1);
//    x.printIndexedBuffer("Input with shape (2, 3, 4) is");       
    nd4j::ops::reduce_max op;
    auto result = op.execute({&x}, {1.}, {0,1,2});
    auto output = result->at(0);    
    // output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

TEST_F(DeclarableOpsTests8, Test_Reduce_Norm1_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {4}, {66.f, 72.f, 78.f, 84.f});
    x.linspace(1);

    nd4j::ops::reduce_norm1 op;
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

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>('c', {1,1,4}, {66.f, 72.f, 78.f, 84.f});
    x.linspace(1);

    nd4j::ops::reduce_norm1 op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {3}, {68.f, 100.f, 132.f});
    x.linspace(1);

    nd4j::ops::reduce_norm1 op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1,3,1}, {68.f, 100.f, 132.f});
    x.linspace(1);

    nd4j::ops::reduce_norm1 op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(300.f);
    x.linspace(1);
    
    nd4j::ops::reduce_norm1 op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(300.f);
    x.linspace(1);
           
    nd4j::ops::reduce_norm1 op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1, 1, 1}, {300.f});
    x.linspace(1);
//    x.printIndexedBuffer("Input with shape (2, 3, 4) is");       
    nd4j::ops::reduce_norm1 op;
    auto result = op.execute({&x}, {1.}, {0,1,2});
    auto output = result->at(0);    
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

TEST_F(DeclarableOpsTests8, Test_Reduce_Norm2_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {4}, {31.7175f, 33.823071f, 35.97221f, 38.15757f});
    x.linspace(1);

    nd4j::ops::reduce_norm2 op;
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

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>('c', {1,1,4}, {31.7175f, 33.823071f, 35.97221f, 38.15757f});
    x.linspace(1);

    nd4j::ops::reduce_norm2 op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {3}, {29.597298f, 39.344631f, 49.759422f});
    x.linspace(1);

    nd4j::ops::reduce_norm2 op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1,3,1}, {29.597298f, 39.344631f, 49.759422f});
    x.linspace(1);

    nd4j::ops::reduce_norm2 op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(70.f);
    x.linspace(1);
    
    nd4j::ops::reduce_norm2 op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(70.f);
    x.linspace(1);
           
    nd4j::ops::reduce_norm2 op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1, 1, 1}, {70.f});
    x.linspace(1);
//    x.printIndexedBuffer("Input with shape (2, 3, 4) is");       
    nd4j::ops::reduce_norm2 op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {4}, {21.f, 22.f, 23.f, 24.f});
    x.linspace(1);

    nd4j::ops::reduce_norm_max op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1,1,4}, {21.f, 22.f, 23.f, 24.f});
    x.linspace(1);

    nd4j::ops::reduce_norm_max op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {3}, {16.f, 20.f, 24.f});
    x.linspace(1);

    nd4j::ops::reduce_norm_max op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1, 3, 1}, {16.f, 20.f, 24.f});
    x.linspace(1);

    nd4j::ops::reduce_norm_max op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(24.f);
    x.linspace(1);
    
    nd4j::ops::reduce_norm_max op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(24.f);
    x.linspace(1);
    
    nd4j::ops::reduce_norm_max op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1, 1, 1}, {24.f});
    x.linspace(1);
    
    nd4j::ops::reduce_norm_max op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {4}, {1006.f, 1144.f, 1294.f, 1456.f});
    x.linspace(1);

    nd4j::ops::reduce_sqnorm op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1,1,4}, {1006.f, 1144.f, 1294.f, 1456.f});
    x.linspace(1);

    nd4j::ops::reduce_sqnorm op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {3}, {876.f, 1548.f, 2476.f});
    x.linspace(1);

    nd4j::ops::reduce_sqnorm op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1, 3, 1}, {876.f, 1548.f, 2476.f});
    x.linspace(1);

    nd4j::ops::reduce_sqnorm op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(4900.f);
    x.linspace(1);
    
    nd4j::ops::reduce_sqnorm op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(4900.f);
    x.linspace(1);
    
    nd4j::ops::reduce_sqnorm op;
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

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1, 1, 1}, {4900.f});
    x.linspace(1);
    
    nd4j::ops::reduce_sqnorm op;
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
    
    auto input = NDArrayFactory::create<double>('c', {3, 4},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.});    
    auto eps = NDArrayFactory::create<double>(0.5f);
    auto exp = NDArrayFactory::create<double>('c', {3, 4}, {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,0.5f});
    //************************************//

    nd4j::ops::reduce_sum_bp op;
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
    
    auto input = NDArrayFactory::create<double>('c', {3, 4},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.});    
    auto eps = NDArrayFactory::create<double>('c', {1, 1}, {0.5f});
    auto exp = NDArrayFactory::create<double>('c', {3, 4}, {0.5f, 0.5f, 0.5f, 0.5f, 
                                     0.5f, 0.5f, 0.5f, 0.5f, 
                                     0.5f, 0.5f, 0.5f,0.5f});
    //************************************//

    nd4j::ops::reduce_sum_bp op;
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
    
    auto input = NDArrayFactory::create<double>('c', {3, 4},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.});    
    auto eps = NDArrayFactory::create<double>('c', {4}, {1.f, 2.f, 3.f, 4.f});
    auto exp = NDArrayFactory::create<double>('c', {3, 4}, {1.f, 2.f, 3.f, 4.f, 
                                     1.f, 2.f, 3.f, 4.f, 
                                     1.f, 2.f, 3.f, 4.f});
    //************************************//

    nd4j::ops::reduce_sum_bp op;
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
    
    auto input = NDArrayFactory::create<double>('c', {3, 4},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.});    
    auto eps = NDArrayFactory::create<double>('c', {1, 4}, {1.f, 2.f, 3.f, 4.f});
    auto exp = NDArrayFactory::create<double>('c', {3, 4}, {1.f, 2.f, 3.f, 4.f, 
                                     1.f, 2.f, 3.f, 4.f, 
                                     1.f, 2.f, 3.f, 4.f});
    //************************************//

    nd4j::ops::reduce_sum_bp op;
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
    
    auto input = NDArrayFactory::create<double>('c', {3, 5},   {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f});    
    auto eps = NDArrayFactory::create<double>(1307674368000.f);
    //************************************//
//    auto exp = NDArrayFactory::create<double>('c', {3, 4}, {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,0.5f});
    //************************************//
    auto exp = NDArrayFactory::create<double>('c', {3, 5},   {1710012166826558903812096.f, 855006083413279451906048.f, 570004067618451974258688.f, 
                                       427503041706639725953024.f, 342002454982589992140800.f, 285002033809225987129344.f, 
                                       244287457550765131825152.f, 213751520853319862976512.f, 190001355872817324752896.f, 
                                       171001227491294996070400.f, 155455648254341989531648.f, 142501016904612993564672.f, 
                                       131539399526781282156544.f, 122143728775382565912576.f, 114000815325130245799936.f});    

    nd4j::ops::reduce_prod_bp op;
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

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>('c', {4}, {11.f, 12.f, 13.f, 14.f});
    x.linspace(1);
    
        
    nd4j::ops::reduce_mean op;
    auto result = op.execute({&x}, {}, {0,1});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceMean_test2) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>('c', {1,1,4}, {11.f, 12.f, 13.f, 14.f});
    x.linspace(1);
    
        
    nd4j::ops::reduce_mean op;
    auto result = op.execute({&x}, {1.}, {0,1});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceMean_test3) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>('c', {3}, {8.5f, 12.5f, 16.5f});
    x.linspace(1);
    
        
    nd4j::ops::reduce_mean op;
    auto result = op.execute({&x}, {}, {0,2});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceMean_test4) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>('c', {1,3,1}, {8.5f, 12.5f, 16.5f});
    x.linspace(1);
    
        
    nd4j::ops::reduce_mean op;
    auto result = op.execute({&x}, {1.f}, {0,2});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceMean_test5) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>(12.5f);
    x.linspace(1);
    
        
    nd4j::ops::reduce_mean op;
    auto result = op.execute({&x}, {}, {});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceMean_test6) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>(12.5f);
    x.linspace(1);
           
    nd4j::ops::reduce_mean op;
    auto result = op.execute({&x}, {}, {0,1,2});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceMean_test7) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>('c', {1,1,1}, {12.5f});
    x.linspace(1);
           
    nd4j::ops::reduce_mean op;
    auto result = op.execute({&x}, {1.}, {0,1,2});
    auto output = result->at(0);    

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceMeanBP_test1) {

    auto x = NDArrayFactory::create<double>('c', {3,4});
    auto gradO1 = NDArrayFactory::create<double>(0.5f);
    auto gradO2 = NDArrayFactory::create<double>('c', {1,1}, {0.5f});
    auto exp = NDArrayFactory::create<double>('c', {3,4}, {1./24, 1./24, 1./24, 1./24, 1./24, 1./24, 1./24, 1./24, 1./24, 1./24, 1./24, 1./24});

    x.linspace(1);
            
    nd4j::ops::reduce_mean_bp op;

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
TEST_F(DeclarableOpsTests8, reduceMeanBP_test2) {

    auto x = NDArrayFactory::create<double>('c', {3,4});
    auto gradO1 = NDArrayFactory::create<double>('c', {4},  {1.f, 2.f, 3.f, 4.f});
    auto gradO2 = NDArrayFactory::create<double>('c', {1,4}, {1.f, 2.f, 3.f, 4.f});
    auto exp = NDArrayFactory::create<double>('c', {3,4}, {1.f/3.f, 2.f/3.f, 1.f, 4.f/3.f, 1.f/3.f, 2.f/3.f, 1.f, 4.f/3.f, 1.f/3.f, 2.f/3.f, 1.f, 4.f/3.f});

    x.linspace(1);
            
    nd4j::ops::reduce_mean_bp op;

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
TEST_F(DeclarableOpsTests8, reduceMeanBP_test3) {

    auto x = NDArrayFactory::create<double>('c', {3,4});
    auto gradO1 = NDArrayFactory::create<double>('c', {3}, {1.f, 2.f, 3.f});
    auto gradO2 = NDArrayFactory::create<double>('c', {3,1}, {1.f, 2.f, 3.f});
    auto exp = NDArrayFactory::create<double>('c', {3,4}, {0.25f, 0.25f, 0.25f, 0.25f, 0.5f, 0.5f, 0.5f, 0.5f, 0.75f, 0.75f, 0.75f, 0.75f});

    x.linspace(1);
            
    nd4j::ops::reduce_mean_bp op;

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

    auto x = NDArrayFactory::create<double>('c', {3}, {2.f, 3.f, 4.f});
    auto gradO = NDArrayFactory::create<double>(0.5f);    
    auto exp = NDArrayFactory::create<double>('c', {3}, {-0.25f, 0.f, 0.25f});    
            
    nd4j::ops::reduce_stdev_bp op;

    auto result = op.execute({&x, &gradO}, {0,1}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());        
    auto output = result->at(0);        

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
    delete result;

}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, avgpool2d_test13) {

    int bS=4, iH=10,iW=10,  iC=3,  kH=3,kW=3,  sH=3,sW=3,  pH=0,pW=0,  dH=1,dW=1;
    int       oH=4, oW=4;
    int paddingMode = 1;             // 1-SAME,  0-VALID
    int dataFormat  = 1;             // 1-NDHWC, 0-NCDHW

    auto input    = NDArrayFactory::create<double>('c', {bS, iH, iW, iC});
    auto expected = NDArrayFactory::create<double>('c', {bS, oH, oW, iC}, { 17.5,   18.5,   19.5,  25. ,   26. ,   27. ,  34. ,   35. ,   36. ,  41.5,   42.5,   43.5,  92.5,   93.5,   94.5, 100. ,  101. ,  102. , 109. ,  110. ,  111. , 116.5,  117.5,  118.5,
                                                      182.5,  183.5,  184.5, 190. ,  191. ,  192. , 199. ,  200. ,  201. , 206.5,  207.5,  208.5, 257.5,  258.5,  259.5, 265. ,  266. ,  267. , 274. ,  275. ,  276. , 281.5,  282.5,  283.5,
                                                      317.5,  318.5,  319.5, 325. ,  326. ,  327. , 334. ,  335. ,  336. , 341.5,  342.5,  343.5, 392.5,  393.5,  394.5, 400. ,  401. ,  402. , 409. ,  410. ,  411. , 416.5,  417.5,  418.5,
                                                      482.5,  483.5,  484.5, 490. ,  491. ,  492. , 499. ,  500. ,  501. , 506.5,  507.5,  508.5, 557.5,  558.5,  559.5, 565. ,  566. ,  567. , 574. ,  575. ,  576. , 581.5,  582.5,  583.5,
                                                      617.5,  618.5,  619.5, 625. ,  626. ,  627. , 634. ,  635. ,  636. , 641.5,  642.5,  643.5, 692.5,  693.5,  694.5, 700. ,  701. ,  702. , 709. ,  710. ,  711. , 716.5,  717.5,  718.5,
                                                      782.5,  783.5,  784.5, 790. ,  791. ,  792. , 799. ,  800. ,  801. , 806.5,  807.5,  808.5, 857.5,  858.5,  859.5, 865. ,  866. ,  867. , 874. ,  875. ,  876. , 881.5,  882.5,  883.5,
                                                      917.5,  918.5,  919.5, 925. ,  926. ,  927. , 934. ,  935. ,  936. , 941.5,  942.5,  943.5, 992.5,  993.5,  994.5,1000. , 1001. , 1002. ,1009. , 1010. , 1011. ,1016.5, 1017.5, 1018.5,
                                                     1082.5, 1083.5, 1084.5,1090. , 1091. , 1092. ,1099. , 1100. , 1101. ,1106.5, 1107.5, 1108.5,1157.5, 1158.5, 1159.5,1165. , 1166. , 1167. ,1174. , 1175. , 1176. ,1181.5, 1182.5, 1183.5});
    input.linspace(1.);    

    nd4j::ops::avgpool2d op;
    auto results = op.execute({&input}, {}, {kH,kW,  sH,sW,  pH,pW,  dH,dW,  paddingMode, 0, dataFormat});
    auto output = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
 
    delete results;
}

 
///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, softmax_cross_entropy_loss_with_logits_test1) {
    
    auto labels = NDArrayFactory::create<double>('c', {2,3,4},{0,1,1,0,0,0,1,0,1,0,1,1,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto expected = NDArrayFactory::create<double>('c', {2,3}, {2.78507, 1.34254, 4.12761, 2.88507, 2.78507, 2.88507});
                                            
    logits.linspace(0.1, 0.1);

    nd4j::ops::softmax_cross_entropy_loss_with_logits op;
    auto results = op.execute({&logits, &labels}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, softmax_cross_entropy_loss_with_logits_test2) {
    
    auto labels = NDArrayFactory::create<double>('c', {2,3,4},{0,1,1,0,0,0,1,0,1,0,1,1,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto expected = NDArrayFactory::create<double>('c', {3,4}, {0.26328, 1.46328, 1.72656, 0.     , 0.26328, 0.     , 1.46328, 0.26328, 1.72656, 0.     , 1.72656, 1.46328});
                                            
    logits.linspace(0.1, 0.1);

    nd4j::ops::softmax_cross_entropy_loss_with_logits op;
    auto results = op.execute({&logits, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, softmax_cross_entropy_loss_with_logits_test3) {
    
    auto labels = NDArrayFactory::create<double>('c', {2,3,4},{0,1,1,0,0,0,1,0,1,0,1,1,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto expected = NDArrayFactory::create<double>('c', {2,4}, {0.75125, 1.55125, 3.45375, 0.75125, 3.45375, 0.     , 2.3025 , 1.15125});
                                            
    logits.linspace(0.1, 0.1);

    nd4j::ops::softmax_cross_entropy_loss_with_logits op;
    auto results = op.execute({&logits, &labels}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, softmax_cross_entropy_loss_with_logits_test4) {
    
    auto labels = NDArrayFactory::create<double>('c', {2,3},{0,1,1,0,0,1});
    auto logits = NDArrayFactory::create<double>('c', {2,3});
    auto expected = NDArrayFactory::create<double>('c', {2}, {2.10389, 1.00194});
                                            
    logits.linspace(0.1, 0.1);

    nd4j::ops::softmax_cross_entropy_loss_with_logits op;
    auto results = op.execute({&logits, &labels}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, softmax_cross_entropy_loss_with_logits_test5) {
    
    auto labels = NDArrayFactory::create<double>('c', {2,3},{0,1,1,0,0,1});
    auto logits = NDArrayFactory::create<double>('c', {2,3});
    auto expected = NDArrayFactory::create<double>('c', {3}, {0., 0.85436, 1.40871});
                                            
    logits.linspace(0.1, 0.1);

    nd4j::ops::softmax_cross_entropy_loss_with_logits op;
    auto results = op.execute({&logits, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, softmax_cross_entropy_loss_with_logits_test6) {
    
    auto labels = NDArrayFactory::create<double>('c', {2,1}, {0,1});
    auto logits = NDArrayFactory::create<double>('c', {2,1});
    auto expected = NDArrayFactory::create<double>('c', {1}, {0.6444});
                                            
    logits.linspace(0.1, 0.1);

    nd4j::ops::softmax_cross_entropy_loss_with_logits op;
    auto results = op.execute({&logits, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, softmax_cross_entropy_loss_with_logits_test7) {
    
    auto labels = NDArrayFactory::create<double>('c', {2,1}, {0,1});
    auto logits = NDArrayFactory::create<double>('c', {2,1});
    auto expected = NDArrayFactory::create<double>('c', {2}, {0., 0.});
                                            
    logits.linspace(0.1, 0.1);

    nd4j::ops::softmax_cross_entropy_loss_with_logits op;
    auto results = op.execute({&logits, &labels}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, softmax_cross_entropy_loss_with_logits_test8) {
    
    auto labels = NDArrayFactory::create<double>('c', {2}, {0,1});
    auto logits = NDArrayFactory::create<double>('c', {2});
    auto expected = NDArrayFactory::create<double>(0.6444);
                                            
    logits.linspace(0.1, 0.1);

    nd4j::ops::softmax_cross_entropy_loss_with_logits op;
    auto results = op.execute({&logits, &labels}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, softmax_cross_entropy_loss_with_logits_test9) {
    
    auto labels = NDArrayFactory::create<double>('c', {1}, {0.});
    auto logits = NDArrayFactory::create<double>('c', {1}, {0.2});
    auto expected = NDArrayFactory::create<double>(0.);
                                               
    nd4j::ops::softmax_cross_entropy_loss_with_logits op;
    auto results = op.execute({&logits, &labels}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, softmax_cross_entropy_loss_with_logits_test10) {
    
    auto labels = NDArrayFactory::create<double>('c', {1,2}, {0,1});
    auto logits = NDArrayFactory::create<double>('c', {1,2});
    auto expected = NDArrayFactory::create<double>('c', {2}, {0., 0.});
                                            
    logits.linspace(0.1, 0.1);

    nd4j::ops::softmax_cross_entropy_loss_with_logits op;
    auto results = op.execute({&logits, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, clipbynorm_test4) {
    
    auto x = NDArrayFactory::create<double>('c', {3, 5}, {0.7044955, 0.55606544, 0.15833677, 0.001874401, 0.61595726, 0.3924779, 0.7414847, 0.4127324, 0.24026828, 0.26093036, 0.46741188, 0.01863421, 0.08528871, 0.529365, 0.5510694});
    auto exp = NDArrayFactory::create<double>('c', {3, 5}, {0.405392, 0.319980, 0.091113, 0.001079, 0.354444, 0.225846, 0.426676, 0.237501, 0.138259, 0.150149, 0.268965, 0.010723, 0.049078, 0.304615, 0.317105});    

    nd4j::ops::clipbynorm op;
    auto result = op.execute({&x}, {1.f}, {});
    auto output = result->at(0);
        
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, clipbynorm_test5) {
    
    auto x = NDArrayFactory::create<double>('c', {3, 5});
    auto exp = NDArrayFactory::create<double>('c', {3, 5}, {1.,  2.,  2.89271,  3.50524,  4.00892, 6.,  7.,  7.71389,  7.88678,  8.01784, 11., 12., 12.53507, 12.26833, 12.02676});    

    x.linspace(1);

    nd4j::ops::clipbynorm op;
    auto result = op.execute({&x}, {15.f}, {0});
    auto output = result->at(0);
        
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, clipbynorm_test6) {
    
    auto x = NDArrayFactory::create<double>('c', {3, 5});
    auto exp = NDArrayFactory::create<double>('c', {3, 5}, {1., 2., 3., 4., 5., 4.95434, 5.78006, 6.60578, 7.43151, 8.25723, 5.64288, 6.15587, 6.66886, 7.18185, 7.69484});    

    x.linspace(1);

    nd4j::ops::clipbynorm op;
    auto result = op.execute({&x}, {15.f}, {1});
    auto output = result->at(0);
        
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}
 
////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, clipbynorm_test7) {
    
    auto x = NDArrayFactory::create<double>('c', {3, 5});
    auto exp = NDArrayFactory::create<double>('c', {3, 5}, {0.42597, 0.85194, 1.27791, 1.70389, 2.12986, 2.55583, 2.9818 , 3.40777, 3.83374, 4.25971, 4.68569, 5.11166, 5.53763, 5.9636 , 6.38957});

    x.linspace(1);

    nd4j::ops::clipbynorm op;
    auto result = op.execute({&x}, {15.f}, {0,1});
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}
 
////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, clipbynorm_test8) {
    
    auto x = NDArrayFactory::create<double>('c', {3, 5});
    auto exp = NDArrayFactory::create<double>('c', {3, 5}, {0.42597, 0.85194, 1.27791, 1.70389, 2.12986, 2.55583, 2.9818 , 3.40777, 3.83374, 4.25971, 4.68569, 5.11166, 5.53763, 5.9636 , 6.38957});

    x.linspace(1);

    nd4j::ops::clipbynorm op;
    auto result = op.execute({&x}, {15.}, {});
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}
 
////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, clipbynorm_test9) {
    
    auto x = NDArrayFactory::create<double>('c', {2}, {3., 4.});
    auto exp = NDArrayFactory::create<double>('c', {2}, {2.4, 3.2});    

    nd4j::ops::clipbynorm op;
    auto result = op.execute({&x}, {4.}, {});
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}
 
////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, clipbynorm_test10) {
    
    auto x = NDArrayFactory::create<double>(6.);
    auto exp = NDArrayFactory::create<double>(5.);

    nd4j::ops::clipbynorm op;
    auto result = op.execute({&x}, {5.}, {});
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}
 
////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, clipbynorm_test11) {
    
    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4}, {1.,  2.,  3.,  4.,  4.44787,  5.33745,  6.22702,  7.1166 , 6.33046,  7.03384,  7.73723,  8.44061,
                                        13., 14., 15., 16., 15.12277, 16.01235, 16.90192, 17.7915 ,14.77107, 15.47446, 16.17784, 16.88123});

    x.linspace(1);

    nd4j::ops::clipbynorm op;
    auto result = op.execute({&x}, {35.}, {0, 2});
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}


TEST_F(DeclarableOpsTests8, clipbynorm_test_tf_119_1) {
    auto x = NDArrayFactory::create<double>('c', {3, 3}, {1, 2, 3, 4, 5,6, 7, 8, 9});
    auto e = NDArrayFactory::create<double>('c', {3, 3}, {0.03198684, 0.06397368, 0.09596053, 0.12794736, 0.15993419, 0.19192106, 0.22390789, 0.25589472, 0.28788155});

    nd4j::ops::clipbynorm op;
    auto result = op.execute({&x}, {0.54}, {});

    ASSERT_EQ(e, *result->at(0));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceMeanBP_test4) {

    auto x = NDArrayFactory::create<double>('c', {3, 4}, {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12. });
    auto gradO1 = NDArrayFactory::create<double>('c', {4}, {1., 2., 3., 4.});
    auto gradO2 = NDArrayFactory::create<double>('c', {1, 4}, {1., 2., 3., 4.});
    auto exp = NDArrayFactory::create<double>('c', {3,4}, {0.333333, 0.666667, 1.000000, 1.333333, 0.333333, 0.666667, 1.000000, 1.333333, 0.333333, 0.666667, 1.000000, 1.333333});
                                     
    nd4j::ops::reduce_mean_bp op;

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

    auto x = NDArrayFactory::create<double>('c', {3, 4}, {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12. });
    auto gradO1 = NDArrayFactory::create<double>('c', {3}, {1., 2., 3.});
    auto gradO2 = NDArrayFactory::create<double>('c', {3, 1}, {1., 2., 3.});
    auto exp = NDArrayFactory::create<double>('c', {3,4}, {0.2500,0.2500,0.2500,0.2500, 0.5000,0.5000,0.5000,0.5000, 0.7500,0.7500,0.7500,0.7500});
                                     
    nd4j::ops::reduce_mean_bp op;
    
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
TEST_F(DeclarableOpsTests8, reduceStDevBP_test5) {

    auto x = NDArrayFactory::create<double>('c', {3, 4}, {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12. });
    auto gradO1 = NDArrayFactory::create<double>('c', {4}, {1., 2., 3., 4.});
    auto gradO2 = NDArrayFactory::create<double>('c', {1, 4}, {1., 2., 3., 4.});
    auto exp = NDArrayFactory::create<double>('c', {3,4}, {-0.408248, -0.816497, -1.224745, -1.632993, 0.000000, 0.000000, 0.000000, 0.000000, 0.408248, 0.816497, 1.224745, 1.632993});
                                                                          
    nd4j::ops::reduce_stdev_bp op;

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
TEST_F(DeclarableOpsTests8, zeros_as_test1) {

    auto x = NDArrayFactory::create<double>(10.);
    auto y = NDArrayFactory::create<double>(100.);
    auto exp = NDArrayFactory::create<double>(0.);
                                                                          
    nd4j::ops::zeros_as op;

    Nd4jStatus status = op.execute({&x}, {&y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, status);    
    
    ASSERT_TRUE(y.isSameShape(exp));
    ASSERT_TRUE(y.equalsTo(exp));

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, ones_as_test1) {

    auto x = NDArrayFactory::create<double>(10.);
    auto y = NDArrayFactory::create<double>(100.);
    auto exp = NDArrayFactory::create<double>(1.);

    nd4j::ops::ones_as op;

    Nd4jStatus status = op.execute({&x}, {&y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, status);    
    
    ASSERT_TRUE(y.isSameShape(exp));
    ASSERT_TRUE(y.equalsTo(exp));
        
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, NormalizeMoments_SGO_1) {

    auto data   = NDArrayFactory::create<double>('c', {10, 10});
    data.linspace(1);
    
    auto means = data.reduceAlongDimension(reduce::Sum, {0});
    auto deviance = data.varianceAlongDimension(variance::SummaryStatsVariance, false, {0}); // = NDArrayFactory::create<double>('c', {10, 10});

    auto counts = NDArrayFactory::create<double>(10.0);

//    auto expMeans = NDArrayFactory::create<double>('c', {10, 10});

//    auto expDeviance = NDArrayFactory::create<double>('c', {10, 10});
    auto squared = NDArrayFactory::create<double>('c', {10, 10});
    data.applyTransform(transform::Square, &squared, nullptr);
    auto ssSquared = squared.reduceAlongDimension(reduce::Sum, {0});
    nd4j::ops::normalize_moments op;
    auto results = op.execute({&counts, means, ssSquared}, {0.0}, {0});
    (*means) /= counts;

//    nd4j::ops::normalize_moments op;
//    auto results = op.execute({&counts, means, deviance}, {0.0}, {});

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_EQ(results->size(), 2);

    auto outputMeans = results->at(0);    
    auto outputDeviance = results->at(1);    

//    outputMeans->printIndexedBuffer("Means");
//    outputDeviance->printIndexedBuffer("Variance");
//    deviance->printIndexedBuffer("Expected");
//    means->printIndexedBuffer("Expected means");
    ASSERT_TRUE(means->isSameShape(outputMeans));
    ASSERT_TRUE(means->equalsTo(outputMeans));    
    ASSERT_TRUE(deviance->isSameShape(outputDeviance));
    ASSERT_TRUE(deviance->equalsTo(outputDeviance));    
    delete means;
    delete deviance;
    delete ssSquared;
//    ASSERT_TRUE(expMeans.isSameShape(outputMeans));
//    ASSERT_TRUE(expMeans.equalsTo(outputMeans));    
//    ASSERT_TRUE(expMeans.isSameShape(outputDeviance));
//    ASSERT_TRUE(expDeviance.equalsTo(outputDeviance));    

    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Moments_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto expMeans = NDArrayFactory::create<double>('c', {4}, {11.f, 12.f, 13.f, 14.f});
    auto expVariance = NDArrayFactory::create<double>('c', {4}, {46.666668f, 46.666668f, 46.66666f, 46.666668f});
    x.linspace(1);

    nd4j::ops::moments op;
    auto result = op.execute({&x}, {}, {0, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    auto outputMeans = result->at(0);    
    auto outputVariance = result->at(1);    

//    outputMeans->printIndexedBuffer("Means");
//    outputVariance->printIndexedBuffer("Variance");
//    outputMeans->printShapeInfo("Result shape");


//    ASSERT_TRUE(exp.isSameShape(output));
//    ASSERT_TRUE(exp.equalsTo(output));
    ASSERT_TRUE(expMeans.isSameShape(outputMeans));
    ASSERT_TRUE(expMeans.equalsTo(outputMeans));
    ASSERT_TRUE(expVariance.isSameShape(outputVariance));
    ASSERT_TRUE(expVariance.equalsTo(outputVariance));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Moments_2) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto expMeans = NDArrayFactory::create<double>('c', {1,1,4}, {11.f, 12.f, 13.f, 14.f});
    auto expVariance = NDArrayFactory::create<double>('c', {1,1,4}, {46.666668f, 46.666668f, 46.66666f, 46.666668f});
    x.linspace(1);

    nd4j::ops::moments op;
    auto result = op.execute({&x}, {1.}, {0, 1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    auto outputMeans = result->at(0);    
    auto outputVariance = result->at(1);    

//    outputMeans->printIndexedBuffer("Means");
//    outputVariance->printIndexedBuffer("Variance");
//    outputMeans->printShapeInfo("Result shape");

//    ASSERT_TRUE(exp.isSameShape(output));
//    ASSERT_TRUE(exp.equalsTo(output));
    ASSERT_TRUE(expMeans.isSameShape(outputMeans));
    ASSERT_TRUE(expMeans.equalsTo(outputMeans));
    ASSERT_TRUE(expVariance.isSameShape(outputVariance));
    ASSERT_TRUE(expVariance.equalsTo(outputVariance));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Moments_3) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto expMeans = NDArrayFactory::create<double>('c', {3}, {8.5f, 12.5f, 16.5f});
    auto expVariance = NDArrayFactory::create<double>('c', {3}, {37.25f, 37.25f, 37.25f});
    x.linspace(1);

    nd4j::ops::moments op;
    auto result = op.execute({&x}, {}, {0, 2});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    auto outputMeans = result->at(0);    
    auto outputVariance = result->at(1);    

//    outputMeans->printIndexedBuffer("Means");
//    outputVariance->printIndexedBuffer("Variance");
//    outputMeans->printShapeInfo("Result shape");

//    ASSERT_TRUE(exp.isSameShape(output));
//    ASSERT_TRUE(exp.equalsTo(output));
    ASSERT_TRUE(expMeans.isSameShape(outputMeans));
    ASSERT_TRUE(expMeans.equalsTo(outputMeans));
    ASSERT_TRUE(expVariance.isSameShape(outputVariance));
    ASSERT_TRUE(expVariance.equalsTo(outputVariance));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Moments_4) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto expMeans = NDArrayFactory::create<double>('c', {1,3,1}, {8.5f, 12.5f, 16.5f});
    auto expVariance = NDArrayFactory::create<double>('c', {1,3,1}, {37.25f, 37.25f, 37.25f});
    x.linspace(1);

    nd4j::ops::moments op;
    auto result = op.execute({&x}, {1.}, {0, 2});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    auto outputMeans = result->at(0);    
    auto outputVariance = result->at(1);    

//    outputMeans->printIndexedBuffer("Means");
//    outputVariance->printIndexedBuffer("Variance");
//    outputMeans->printShapeInfo("Result shape");

//    ASSERT_TRUE(exp.isSameShape(output));
//    ASSERT_TRUE(exp.equalsTo(output));
    ASSERT_TRUE(expMeans.isSameShape(outputMeans));
    ASSERT_TRUE(expMeans.equalsTo(outputMeans));
    ASSERT_TRUE(expVariance.isSameShape(outputVariance));
    ASSERT_TRUE(expVariance.equalsTo(outputVariance));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Moments_6) {
    auto expMeans = NDArrayFactory::create<double>(12.5f);
    auto expVariance = NDArrayFactory::create<double>(47.916668f);

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    x.linspace(1);
           
    nd4j::ops::moments op;
    auto result = op.execute({&x}, {}, {0,1,2});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    auto outputMeans = result->at(0);    
    auto outputVariance = result->at(1);    

//    outputMeans->printIndexedBuffer("Means");
//    outputVariance->printIndexedBuffer("Variance");
//    outputMeans->printShapeInfo("Result shape");

    ASSERT_TRUE(expMeans.isSameShape(outputMeans));
    ASSERT_TRUE(expMeans.equalsTo(outputMeans));
    ASSERT_TRUE(expVariance.isSameShape(outputVariance));
    ASSERT_TRUE(expVariance.equalsTo(outputVariance));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Moments_7) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});

    auto expMeans = NDArrayFactory::create<double>('c', {1,1,1}, {12.5f});
    auto expVariance = NDArrayFactory::create<double>('c', {1,1,1}, {47.916668f});

    x.linspace(1);
    // x.printIndexedBuffer("Input with shape (2, 3, 4) is");       
    nd4j::ops::moments op;
    auto result = op.execute({&x}, {1.}, {0,1,2});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());    

    auto outputMeans = result->at(0);    
    auto outputVariance = result->at(1);    

//    outputMeans->printIndexedBuffer("Means");
//    outputVariance->printIndexedBuffer("Variance");
//    outputMeans->printShapeInfo("Result shape");
    ASSERT_TRUE(expMeans.isSameShape(outputMeans));
    ASSERT_TRUE(expMeans.equalsTo(outputMeans));
    ASSERT_TRUE(expVariance.isSameShape(outputVariance));
    ASSERT_TRUE(expVariance.equalsTo(outputVariance));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, LrnTest_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 2, 2, 2}, { 5.5, 0., 0.3, 5.5,
                                            8.6, 0.,  0., 0.4,
                                            1.5, 1., 1.3, 1.5,
                                            2.6, 2.,  3., 1.4}
    );

    auto exp = NDArrayFactory::create<double>('c', {2, 2, 2, 2}, {
                                            0.98386997,        0.,  0.05358852,  0.9824562,
                                            0.99330735,        0.,          0., 0.37139067,
                                            0.72760683, 0.4850712,   0.5848977, 0.67488194,
                                            0.7581754,  0.58321184, 0.86747235, 0.4048204}
    );

    nd4j::ops::lrn op;
    auto  results = op.execute({&x}, {1.0, 1.0, 0.5}, {2});
    auto out = results->at(0);
        
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(out));
//    out->printIndexedBuffer("LRN out");
//    exp.printIndexedBuffer("LRN exp");
    ASSERT_TRUE(exp.equalsTo(out));    
    
    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, LrnTest_2) {

    auto x = NDArrayFactory::create<double>('c', {3, 3, 5, 5});
    x.linspace(1);
    
    auto exp = NDArrayFactory::create<double>('c', {3, 3, 5, 5}, {
    0.2581989 ,0.3592106 , 0.40089184, 0.53935987, 0.70014,  
    0.4898979 ,0.46056613, 0.43971977, 0.5240002 , 0.6375767,
    0.5274096 ,0.47771242, 0.4443308 , 0.5163977 , 0.61701745,
    0.5424508 ,0.48452914, 0.44570294, 0.5123918 , 0.6068971,
    0.5505386 ,0.4881662 , 0.4462865 , 0.5099462 , 0.60088515,

    0.5555859 , 0.49042296, 0.44658744, 0.5083028 , 0.59690416,
    0.55903524, 0.4919585 , 0.44676256, 0.5071239 , 0.59407425,
    0.5615412 , 0.49307042, 0.44687328, 0.50623745, 0.5919596 ,
    0.56344414, 0.49391258, 0.4469477 , 0.5055468 , 0.59031945,
    0.56493837, 0.49457246, 0.4470002 , 0.5049936 , 0.5890103 ,

    0.56614274, 0.49510333, 0.44703856, 0.50454074, 0.5879411 ,
    0.567134  , 0.49553978, 0.4470674 , 0.504163  , 0.5870515 ,
    0.5679643 , 0.4959048 , 0.44708967, 0.5038433 , 0.5862998 ,
    0.56866974, 0.4962146 , 0.44710726, 0.5035692 , 0.58565617,
    0.56927663, 0.49648085, 0.4471213 , 0.5033315 , 0.5850988 ,


    0.56980413, 0.49671215, 0.44713274, 0.50312346, 0.58461165,
    0.57026696, 0.49691492, 0.4471422 , 0.50293994, 0.58418214,
    0.5706764 , 0.49709415, 0.44715008, 0.5027767 , 0.5838005 ,
    0.571041  , 0.4972537 , 0.44715673, 0.50263065, 0.58345926,
    0.57136786, 0.49739665, 0.44716236, 0.5024992 , 0.58315235,

    0.5716625 , 0.49752548, 0.4471672 , 0.5023803,  0.5828747 ,
    0.5719295 , 0.49764213, 0.44717142, 0.5022721,  0.5826225 ,
    0.57217246, 0.49774826, 0.44717506, 0.5021734,  0.58239233,
    0.5723947 , 0.4978453 , 0.44717824, 0.5020829,  0.58218133,
    0.57259864, 0.49793428, 0.44718108, 0.5019997,  0.5819874 ,

    0.5727864 , 0.49801624, 0.44718358, 0.5019227,  0.5818083 ,
    0.57296   , 0.49809194, 0.44718578, 0.5018515,  0.5816426 ,
    0.5731208 , 0.49816203, 0.44718775, 0.5017854,  0.58148885,
    0.57327026, 0.49822718, 0.4471895 , 0.5017239,  0.5813457 ,
    0.57340944, 0.49828786, 0.44719115, 0.5016664,  0.581212  ,


    0.57353944, 0.4983446 , 0.44719255, 0.50161266, 0.58108705,
    0.5736612 , 0.49839762, 0.4471939 , 0.50156236, 0.5809699 ,
    0.5737754 , 0.4984474 , 0.44719502, 0.501515  , 0.58085984,
    0.5738828 , 0.49849418, 0.4471962 , 0.50147045, 0.5807564 ,
    0.5739839 , 0.49853817, 0.44719717, 0.5014284 , 0.5806588 ,

    0.5740793 , 0.49857965, 0.4471981 , 0.5013887 , 0.5805666 ,
    0.5741694 , 0.49861887, 0.44719887, 0.50135124, 0.58047944,
    0.57425463, 0.49865603, 0.44719967, 0.5013157 , 0.5803969 ,
    0.5743354 , 0.4986912 , 0.44720036, 0.5012819 , 0.5803186 ,
    0.57441217, 0.49872455, 0.44720104, 0.5012499 , 0.58024424,

    0.57448506, 0.4987563 , 0.44720164, 0.5012194 , 0.58017343,
    0.57455444, 0.4987865 , 0.4472022 , 0.5011904 , 0.5801061,
    0.57462054, 0.49881527, 0.44720277, 0.5011627 , 0.5800419,
    0.57468355, 0.49884263, 0.44720328, 0.50113624, 0.5799805,
    0.57474375, 0.49886885, 0.44720373, 0.50111103, 0.5799219 }
    );
//
    nd4j::ops::lrn op;
    auto  results = op.execute({&x}, {1.0, 1.0, 0.5}, {2});
    auto out = results->at(0);
        
    ASSERT_EQ(Status::OK(), results->status());
//    ASSERT_TRUE(exp.isSameShape(out));
//    out->printIndexedBuffer("LRN out");
//    exp.printIndexedBuffer("LRN exp");
    ASSERT_TRUE(exp.equalsTo(out));    
    
    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, LrnTest_BP_1) {

    auto x = NDArrayFactory::create<double>( 'c', {3, 3, 5, 5});
    x.linspace(1);
    
    auto eps = NDArrayFactory::create<double>('c', {3, 3, 5, 5}, {
    0.2581989 ,0.3592106 , 0.40089184, 0.53935987, 0.70014,  
    0.4898979 ,0.46056613, 0.43971977, 0.5240002 , 0.6375767,
    0.5274096 ,0.47771242, 0.4443308 , 0.5163977 , 0.61701745,
    0.5424508 ,0.48452914, 0.44570294, 0.5123918 , 0.6068971,
    0.5505386 ,0.4881662 , 0.4462865 , 0.5099462 , 0.60088515,

    0.5555859 , 0.49042296, 0.44658744, 0.5083028 , 0.59690416,
    0.55903524, 0.4919585 , 0.44676256, 0.5071239 , 0.59407425,
    0.5615412 , 0.49307042, 0.44687328, 0.50623745, 0.5919596 ,
    0.56344414, 0.49391258, 0.4469477 , 0.5055468 , 0.59031945,
    0.56493837, 0.49457246, 0.4470002 , 0.5049936 , 0.5890103 ,

    0.56614274, 0.49510333, 0.44703856, 0.50454074, 0.5879411 ,
    0.567134  , 0.49553978, 0.4470674 , 0.504163  , 0.5870515 ,
    0.5679643 , 0.4959048 , 0.44708967, 0.5038433 , 0.5862998 ,
    0.56866974, 0.4962146 , 0.44710726, 0.5035692 , 0.58565617,
    0.56927663, 0.49648085, 0.4471213 , 0.5033315 , 0.5850988 ,


    0.56980413, 0.49671215, 0.44713274, 0.50312346, 0.58461165,
    0.57026696, 0.49691492, 0.4471422 , 0.50293994, 0.58418214,
    0.5706764 , 0.49709415, 0.44715008, 0.5027767 , 0.5838005 ,
    0.571041  , 0.4972537 , 0.44715673, 0.50263065, 0.58345926,
    0.57136786, 0.49739665, 0.44716236, 0.5024992 , 0.58315235,

    0.5716625 , 0.49752548, 0.4471672 , 0.5023803,  0.5828747 ,
    0.5719295 , 0.49764213, 0.44717142, 0.5022721,  0.5826225 ,
    0.57217246, 0.49774826, 0.44717506, 0.5021734,  0.58239233,
    0.5723947 , 0.4978453 , 0.44717824, 0.5020829,  0.58218133,
    0.57259864, 0.49793428, 0.44718108, 0.5019997,  0.5819874 ,

    0.5727864 , 0.49801624, 0.44718358, 0.5019227,  0.5818083 ,
    0.57296   , 0.49809194, 0.44718578, 0.5018515,  0.5816426 ,
    0.5731208 , 0.49816203, 0.44718775, 0.5017854,  0.58148885,
    0.57327026, 0.49822718, 0.4471895 , 0.5017239,  0.5813457 ,
    0.57340944, 0.49828786, 0.44719115, 0.5016664,  0.581212  ,


    0.57353944, 0.4983446 , 0.44719255, 0.50161266, 0.58108705,
    0.5736612 , 0.49839762, 0.4471939 , 0.50156236, 0.5809699 ,
    0.5737754 , 0.4984474 , 0.44719502, 0.501515  , 0.58085984,
    0.5738828 , 0.49849418, 0.4471962 , 0.50147045, 0.5807564 ,
    0.5739839 , 0.49853817, 0.44719717, 0.5014284 , 0.5806588 ,

    0.5740793 , 0.49857965, 0.4471981 , 0.5013887 , 0.5805666 ,
    0.5741694 , 0.49861887, 0.44719887, 0.50135124, 0.58047944,
    0.57425463, 0.49865603, 0.44719967, 0.5013157 , 0.5803969 ,
    0.5743354 , 0.4986912 , 0.44720036, 0.5012819 , 0.5803186 ,
    0.57441217, 0.49872455, 0.44720104, 0.5012499 , 0.58024424,

    0.57448506, 0.4987563 , 0.44720164, 0.5012194 , 0.58017343,
    0.57455444, 0.4987865 , 0.4472022 , 0.5011904 , 0.5801061,
    0.57462054, 0.49881527, 0.44720277, 0.5011627 , 0.5800419,
    0.57468355, 0.49884263, 0.44720328, 0.50113624, 0.5799805,
    0.57474375, 0.49886885, 0.44720373, 0.50111103, 0.5799219 }
    );
//
auto exp = NDArrayFactory::create<double>('c', {3,3,5,5}, {
    0.009859, 0.013075, 0.013874, 0.017893, 0.022344, 0.014551, 0.012859, 0.011511, 0.013311, 0.015834, 0.012025, 0.010047, 0.008601, 0.009920, 0.011885, 0.009505, 0.007636, 0.006299, 0.007413, 0.009095, 0.007446, 0.005743, 0.004540, 0.005533, 0.007033, 0.005821, 0.004282, 0.003209, 0.004123, 0.005491, 0.004577, 0.003198, 0.002247, 0.003097, 0.004355, 0.003652, 0.002412, 0.001565, 0.002357, 0.003517, 0.002965, 0.001844, 0.001084, 0.001821, 0.002893, 0.002451, 0.001430, 0.000741, 0.001428, 0.002422, -0.111434, -0.105946, -0.100351, -0.091868, -0.083323, -0.078775, -0.076222, -0.073291, -0.067635, -0.061692, -0.058943, -0.057832, -0.056263, -0.052198, -0.047768, -0.046002, -0.045655, -0.044839, -0.041748, -0.038271, -0.037084, -0.037161, -0.036786, -0.034331, -0.031495, 0.000077, -0.000673, -0.001181, -0.000667, 0.000079, -0.000089, -0.000802, -0.001285, -0.000793, -0.000079, -0.000228, -0.000908, -0.001368, -0.000896, -0.000212, -0.000345, -0.000996, -0.001434, -0.000981, -0.000325, -0.000444, -0.001067, -0.001487, -0.001051, -0.000421, 0.000697, 0.000188, -0.000152, 0.000210, 0.000731, 0.000650, 0.000165, -0.000161, 0.000185, 0.000683, 0.000610, 0.000145, -0.000168, 0.000164, 0.000641, 0.000574, 0.000128, -0.000172, 0.000146, 0.000604, 0.000542, 0.000113, -0.000175, 0.000131, 0.000571, -0.009490, -0.010070, -0.010409, -0.009734, -0.008834, -0.008785, -0.009351, -0.009687, -0.009054, -0.008207, -0.008167, -0.008718, -0.009050, -0.008455, -0.007654, -0.007622, -0.008159, -0.008485, -0.007924, -0.007164, -0.007138, -0.007661, -0.007981, -0.007450, -0.006728, -0.000901, -0.001327, -0.001614, -0.001310, -0.000869, -0.000913, -0.001328, -0.001607, -0.001310, -0.000882, -0.000922, -0.001326, -0.001598, -0.001309, -0.000892, -0.000930, -0.001323, -0.001588, -0.001306, -0.000900, -0.000936, -0.001319, -0.001577, -0.001302, -0.000906, 0.000339, 0.000038, -0.000164, 0.000048, 0.000355, 0.000328, 0.000035, -0.000162, 0.000045, 0.000343, 0.000318, 0.000033, -0.000159, 0.000041, 0.000332, 0.000308, 0.000030, -0.000157, 0.000039, 0.000322, 0.000299, 0.000028, -0.000155, 0.000036, 0.000312, -0.004085, -0.004479, -0.004733, -0.004396, -0.003925, -0.003925, -0.004309, -0.004558, -0.004232, -0.003775, -0.003776, -0.004151, -0.004395, -0.004079, -0.003636, -0.003637, -0.004004, -0.004242, -0.003936, -0.003505, -0.003507, -0.003866, -0.004100, -0.003802, -0.003383}
    );

    nd4j::ops::lrn_bp op;
    auto  results = op.execute({&x, &eps}, {1.0, 1.0, 0.5}, {2});
    auto out = results->at(0);
        
    ASSERT_EQ(Status::OK(), results->status());
//    ASSERT_TRUE(exp.isSameShape(out));
//    out->printIndexedBuffer("LRN BP out");
//    exp.printIndexedBuffer("LRN exp");
    ASSERT_TRUE(exp.equalsTo(out));    
    
    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, LrnTest_BP_2) {

    auto x = NDArrayFactory::create<double>( 'c', {3, 3, 5, 5});
    x.linspace(1);
    
    auto eps = NDArrayFactory::create<double>('c', {3, 3, 5, 5});
    eps.linspace(1);
//
    auto exp = NDArrayFactory::create<double>('c', {3,3,5,5}, {
    0.009859, 0.013075, 0.013874, 0.017893, 0.022344, 0.014551, 0.012859, 0.011511, 0.013311, 0.015834, 0.012025, 0.010047, 0.008601, 0.009920, 0.011885, 0.009505, 0.007636, 0.006299, 0.007413, 0.009095, 0.007446, 0.005743, 0.004540, 0.005533, 0.007033, 0.005821, 0.004282, 0.003209, 0.004123, 0.005491, 0.004577, 0.003198, 0.002247, 0.003097, 0.004355, 0.003652, 0.002412, 0.001565, 0.002357, 0.003517, 0.002965, 0.001844, 0.001084, 0.001821, 0.002893, 0.002451, 0.001430, 0.000741, 0.001428, 0.002422, -0.111434, -0.105946, -0.100351, -0.091868, -0.083323, -0.078775, -0.076222, -0.073291, -0.067635, -0.061692, -0.058943, -0.057832, -0.056263, -0.052198, -0.047768, -0.046002, -0.045655, -0.044839, -0.041748, -0.038271, -0.037084, -0.037161, -0.036786, -0.034331, -0.031495, 0.000077, -0.000673, -0.001181, -0.000667, 0.000079, -0.000089, -0.000802, -0.001285, -0.000793, -0.000079, -0.000228, -0.000908, -0.001368, -0.000896, -0.000212, -0.000345, -0.000996, -0.001434, -0.000981, -0.000325, -0.000444, -0.001067, -0.001487, -0.001051, -0.000421, 0.000697, 0.000188, -0.000152, 0.000210, 0.000731, 0.000650, 0.000165, -0.000161, 0.000185, 0.000683, 0.000610, 0.000145, -0.000168, 0.000164, 0.000641, 0.000574, 0.000128, -0.000172, 0.000146, 0.000604, 0.000542, 0.000113, -0.000175, 0.000131, 0.000571, -0.009490, -0.010070, -0.010409, -0.009734, -0.008834, -0.008785, -0.009351, -0.009687, -0.009054, -0.008207, -0.008167, -0.008718, -0.009050, -0.008455, -0.007654, -0.007622, -0.008159, -0.008485, -0.007924, -0.007164, -0.007138, -0.007661, -0.007981, -0.007450, -0.006728, -0.000901, -0.001327, -0.001614, -0.001310, -0.000869, -0.000913, -0.001328, -0.001607, -0.001310, -0.000882, -0.000922, -0.001326, -0.001598, -0.001309, -0.000892, -0.000930, -0.001323, -0.001588, -0.001306, -0.000900, -0.000936, -0.001319, -0.001577, -0.001302, -0.000906, 0.000339, 0.000038, -0.000164, 0.000048, 0.000355, 0.000328, 0.000035, -0.000162, 0.000045, 0.000343, 0.000318, 0.000033, -0.000159, 0.000041, 0.000332, 0.000308, 0.000030, -0.000157, 0.000039, 0.000322, 0.000299, 0.000028, -0.000155, 0.000036, 0.000312, -0.004085, -0.004479, -0.004733, -0.004396, -0.003925, -0.003925, -0.004309, -0.004558, -0.004232, -0.003775, -0.003776, -0.004151, -0.004395, -0.004079, -0.003636, -0.003637, -0.004004, -0.004242, -0.003936, -0.003505, -0.003507, -0.003866, -0.004100, -0.003802, -0.003383}
    );

    nd4j::ops::lrn_bp op;
    auto  results = op.execute({&x, &eps}, {1.0, 1.0, 0.5}, {2});
    auto out = results->at(0);
        
    ASSERT_EQ(Status::OK(), results->status());
//    ASSERT_TRUE(exp.isSameShape(out));
//    out->printIndexedBuffer("LRN BP out");
//    exp.printIndexedBuffer("LRN exp");
//    ASSERT_TRUE(exp.equalsTo(out));    
    
    delete results;
}


