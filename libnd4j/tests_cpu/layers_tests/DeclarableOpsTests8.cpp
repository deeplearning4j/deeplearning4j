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
#include <GradCheck.h>
// #include <array/NDArrayList.h>

using namespace nd4j;


class DeclarableOpsTests8 : public testing::Test {
public:

    DeclarableOpsTests8() {
        printf("\n");
        fflush(stdout);
    }
};

template <typename T>
class TypedDeclarableOpsTests8 : public testing::Test {
public:

    TypedDeclarableOpsTests8() {
        printf("\n");
        fflush(stdout);
    }
};

typedef ::testing::Types<double, float> TestingTypes;
TYPED_TEST_CASE(TypedDeclarableOpsTests8, TestingTypes);

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceVariance_test1) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.f});
    auto exp = NDArrayFactory::create<double>('c', {4}, {602.2222f, 727.13885f, 993.5555f, 755.8889f});
        
    nd4j::ops::reduce_variance op;
    auto result = op.execute({&x}, {}, {0,1});
    auto output = result->at(0);    

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceVariance_test8) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    auto exp = NDArrayFactory::create<double>('c', {1,1,1}, {788.6927f});
    auto axes = NDArrayFactory::create<int>({0, 1, 2});
    nd4j::ops::reduce_variance op;
    auto result = op.execute({&x, &axes}, {}, {}, {true});
    auto output = result->at(0);

    ASSERT_EQ(Status::OK(), result->status());

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    
    // output->printBuffer("Reduced STDDEV");
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDev_test08) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    auto exp = NDArrayFactory::create<double>('c', {4}, {26.88246f, 29.53924f, 34.52921f, 30.11755f});
    auto axes = NDArrayFactory::create<int>({0,1});
    nd4j::ops::reduce_stdev op;
    auto result = op.execute({&x, &axes}, {}, {}, {false, true});
    auto output = result->at(0);

    ASSERT_EQ(Status::OK(), result->status());
    // output->printBuffer("Reduced STDDEV08");
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
    ASSERT_EQ(Status::OK(), result->status());    
    auto output = result->at(0);
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO1}, {1,1}, {});
    ASSERT_EQ(Status::OK(), result->status());    
    output = result->at(0);
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));
    delete result;    

    result = op.execute({&x, &gradO2}, {0,0}, {});
    ASSERT_EQ(Status::OK(), result->status());    
    output = result->at(0);
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));
    delete result;    

    result = op.execute({&x, &gradO1}, {1,0}, {});
    ASSERT_EQ(Status::OK(), result->status());    
    output = result->at(0);
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));
    delete result;   

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceVarianceBP_test2) {

    auto x = NDArrayFactory::create<double>('c', {3,4});
    auto gradO1 = NDArrayFactory::create<double>('c', {1,4}, {1.f,2.f,3.f,4.f});
    auto gradO2 = NDArrayFactory::create<double>('c', {4}, {1.,2.,3.,4.});
    auto exp12 = NDArrayFactory::create<double>('c', {3,4}, {-2.666667f, -5.333333f, -8.000000f,  -10.666667f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 2.666667f, 5.333333f,  8.000000f, 10.666667f});
    auto exp34 = NDArrayFactory::create<double>('c', {3,4}, {-4.000000f, -8.000000f, -12.000000f, -16.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 4.000000f, 8.000000f, 12.000000f, 16.000000f});

    x.linspace(1);
            
    nd4j::ops::reduce_variance_bp op;

    auto result = op.execute({&x, &gradO2}, {0,0}, {0});
    ASSERT_EQ(Status::OK(), result->status());    
    auto output = result->at(0);    
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO1}, {1,0}, {0});
    ASSERT_EQ(Status::OK(), result->status());    
    output = result->at(0);        
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output)); 
    delete result;    

    result = op.execute({&x, &gradO2}, {0,1}, {0});
    ASSERT_EQ(Status::OK(), result->status());    
    output = result->at(0);    
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));
    delete result;    

    result = op.execute({&x, &gradO1}, {1,1}, {0});
    ASSERT_EQ(Status::OK(), result->status());    
    output = result->at(0);
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));
    delete result;   

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceVarianceBP_test02) {

    auto x = NDArrayFactory::create<double>('c', {3,4});
    auto gradO1 = NDArrayFactory::create<double>('c', {1,4}, {1.f,2.f,3.f,4.f});
    auto gradO2 = NDArrayFactory::create<double>('c', {4}, {1.f,2.f,3.f,4.f});
    auto exp12 = NDArrayFactory::create<double>('c', {3,4}, {-2.666667f, -5.333333f, -8.000000f,  -10.666667f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 2.666667f, 5.333333f,  8.000000f, 10.666667f});
    auto exp34 = NDArrayFactory::create<double>('c', {3,4}, {-4.000000f, -8.000000f, -12.000000f, -16.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 4.000000f, 8.000000f, 12.000000f, 16.000000f});
    auto axes = NDArrayFactory::create<int>({(int)0,});
    x.linspace(1);

    nd4j::ops::reduce_variance_bp op;

    auto result = op.execute({&x, &gradO2, &axes}, {}, {}, {false, false});
    ASSERT_EQ(Status::OK(), result->status());
    auto output = result->at(0);
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO1, &axes}, {}, {}, {true, false});
    ASSERT_EQ(Status::OK(), result->status());
    output = result->at(0);
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO2, &axes}, {}, {}, {false, true});
    ASSERT_EQ(Status::OK(), result->status());
    output = result->at(0);
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO1, &axes}, {}, {}, {true, true});
    ASSERT_EQ(Status::OK(), result->status());
    output = result->at(0);
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));
    delete result;

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceVarianceBP_test3) {

    auto x = NDArrayFactory::create<double>('c', {3, 4});
    auto gradO1 = NDArrayFactory::create<double>('c', {3, 1}, {1.f, 2.f, 3.f});
    auto gradO2 = NDArrayFactory::create<double>('c', {3}, {1.f, 2.f, 3.f});
    auto exp12 = NDArrayFactory::create<double>('c', {3, 4},
                                                {-0.750000f, -0.250000f, 0.250000f, 0.750000f, -1.500000f, -0.500000f,
                                                 0.500000f, 1.500000f, -2.250000f, -0.750000f, 0.750000f, 2.250000f});
    auto exp34 = NDArrayFactory::create<double>('c', {3, 4},
                                                {-1.000000f, -0.333333f, 0.333333f, 1.000000f, -2.000000f, -0.666667f,
                                                 0.666667f, 2.000000f, -3.000000f, -1.000000f, 1.000000f, 3.000000f});

    x.linspace(1);

    nd4j::ops::reduce_variance_bp op;

    auto result = op.execute({&x, &gradO2}, {0, 0}, {1});
    ASSERT_EQ(Status::OK(), result->status());
    auto output = result->at(0);
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO1}, {1, 0}, {1});
    ASSERT_EQ(Status::OK(), result->status());
    output = result->at(0);
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO2}, {0, 1}, {1});
    ASSERT_EQ(Status::OK(), result->status());
    output = result->at(0);
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO1}, {1, 1}, {1});
    ASSERT_EQ(Status::OK(), result->status());
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
    ASSERT_EQ(Status::OK(), result->status());    
    auto output = result->at(0);
    // output->printIndexedBuffer();
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO1}, {1,1}, {});
    ASSERT_EQ(Status::OK(), result->status());    
    output = result->at(0);
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO2}, {0,0}, {});
    ASSERT_EQ(Status::OK(), result->status());    
    output = result->at(0);
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO1}, {1,0}, {});
    ASSERT_EQ(Status::OK(), result->status());    
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
    ASSERT_EQ(Status::OK(), result->status());    
    auto output = result->at(0);        
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO1}, {1,0}, {0});
    ASSERT_EQ(Status::OK(), result->status());    
    output = result->at(0);        
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output)); 
    delete result;    

    result = op.execute({&x, &gradO2}, {0,1}, {0});
    ASSERT_EQ(Status::OK(), result->status());    
    output = result->at(0);    
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));
    delete result;    

    result = op.execute({&x, &gradO1}, {1,1}, {0});
    ASSERT_EQ(Status::OK(), result->status());    
    output = result->at(0);
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));
    delete result;   
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDevBP_test02) {

    int ax = 0;
    auto x = NDArrayFactory::create<double>('c', {3,4});
    auto gradO1 = NDArrayFactory::create<double>('c', {1,4}, {1.f,2.f,3.f,4.f});
    auto gradO2 = NDArrayFactory::create<double>('c', {4}, {1.f,2.f,3.f,4.f});
    auto exp12 = NDArrayFactory::create<double>('c', {3,4}, {-0.4082483f, -0.8164966f, -1.2247449f, -1.6329932f, 0.0, 0.0, 0.0, 0.0, 0.4082483f, 0.8164966f, 1.2247449f, 1.6329932f});
    auto exp34 = NDArrayFactory::create<double>('c', {3,4}, {-0.5f, -1.0f, -1.5f, -2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f});
    auto axis = NDArrayFactory::create<int>('c', {1}, {ax});
    x.linspace(1);

    nd4j::ops::reduce_stdev_bp op;

    auto result = op.execute({&x, &gradO2, &axis}, {}, {}, {false, false});
    ASSERT_EQ(Status::OK(), result->status());
    auto output = result->at(0);
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO1, &axis}, {}, {}, {true, false});
    ASSERT_EQ(Status::OK(), result->status());
    output = result->at(0);
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO2, &axis}, {}, {}, {false, true});
    ASSERT_EQ(Status::OK(), result->status());
    output = result->at(0);
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO1, &axis}, {}, {}, {true, true});
    ASSERT_EQ(Status::OK(), result->status());
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
    ASSERT_EQ(Status::OK(), result->status());    
    auto output = result->at(0);    
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO1}, {1,0}, {1});
    ASSERT_EQ(Status::OK(), result->status());    
    output = result->at(0);        
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output)); 
    delete result;    

    result = op.execute({&x, &gradO2}, {0,1}, {1});
    ASSERT_EQ(Status::OK(), result->status());    
    output = result->at(0);    
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));
    delete result;    

    result = op.execute({&x, &gradO1}, {1,1}, {1});
    ASSERT_EQ(Status::OK(), result->status());    
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
TEST_F(DeclarableOpsTests8, Test_Reduce_Sum_03) {

    auto input = NDArrayFactory::create<double>('c', {3, 5},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.});
    auto exp = NDArrayFactory::create<double>({15.f, 40.f, 65.f});
    auto axis = NDArrayFactory::create<int>('c', {1}, {1});
    //************************************//

    nd4j::ops::reduce_sum op;
    auto result = op.execute({&input, &axis}, {}, {}, {false});

    ASSERT_EQ(Status::OK(), result->status());
    auto z = result->at(0);
    // z->printIndexedBuffer("Result is ");
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
    ASSERT_EQ(Status::OK(), result->status());    

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
   // output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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
    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Prod_04) {

    auto x = NDArrayFactory::create<double>('c', {2,3,2});
    auto exp = NDArrayFactory::create<double>('c', {1,3,1}, {112.f, 1080.f, 3960.f});
    auto axes = NDArrayFactory::create<int>({0, 2});
    x.linspace(1);

    nd4j::ops::reduce_prod op;
    auto result = op.execute({&x, &axes}, {}, {}, {true});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result->status());

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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
    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Min_04) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1,3,1}, {1.f, 5.f, 9.f});
    auto axes = NDArrayFactory::create<int>({0, 2});
    x.linspace(1);

    nd4j::ops::reduce_min op;
    auto result = op.execute({&x, &axes}, {}, {}, {true});
    auto output = result->at(0);
    // output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result->status());

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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
    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Max_04) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1,3,1}, {16.f, 20.f, 24.f});
    auto axes = NDArrayFactory::create<int>({0, 2});
    x.linspace(1);

    nd4j::ops::reduce_max op;
    auto result = op.execute({&x, &axes}, {}, {}, {true});
    auto output = result->at(0);
    // output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result->status());

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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
    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Norm1_04) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1,3,1}, {68.f, 100.f, 132.f});
    auto axes = NDArrayFactory::create<int>({0, 2});
    x.linspace(1);

    nd4j::ops::reduce_norm1 op;
    auto result = op.execute({&x, &axes}, {}, {}, {true});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result->status());

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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
    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Norm2_04) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1,3,1}, {29.597298f, 39.344631f, 49.759422f});
    auto axes = NDArrayFactory::create<int>({0,2});
    x.linspace(1);

    nd4j::ops::reduce_norm2 op;
    auto result = op.execute({&x, &axes}, {}, {}, {true});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result->status());

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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
    ASSERT_EQ(Status::OK(), result->status());    

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
    ASSERT_EQ(Status::OK(), result->status());    

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
    ASSERT_EQ(Status::OK(), result->status());    

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
    ASSERT_EQ(Status::OK(), result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_NormMax_04) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1, 3, 1}, {16.f, 20.f, 24.f});
    auto axes = NDArrayFactory::create<int>({0,2});
    x.linspace(1);

    nd4j::ops::reduce_norm_max op;
    auto result = op.execute({&x, &axes}, {}, {}, {true});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(Status::OK(), result->status());

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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
    ASSERT_EQ(Status::OK(), result->status());    

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
    ASSERT_EQ(Status::OK(), result->status());    

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
    ASSERT_EQ(Status::OK(), result->status());    

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
    ASSERT_EQ(Status::OK(), result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_SquaredNorm_04) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1, 3, 1}, {876.f, 1548.f, 2476.f});
    auto axes = NDArrayFactory::create<int>({0, 2});
    x.linspace(1);

    nd4j::ops::reduce_sqnorm op;
    auto result = op.execute({&x, &axes}, {}, {}, {true});
    auto output = result->at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(Status::OK(), result->status());

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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
TEST_F(DeclarableOpsTests8, Test_Reduce_Sum_BP_04) {

    int ax = 0;
    auto input = NDArrayFactory::create<double>('c', {3, 4},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.});
    auto eps = NDArrayFactory::create<double>('c', {1, 4}, {1.f, 2.f, 3.f, 4.f});
    auto exp = NDArrayFactory::create<double>('c', {3, 4}, {1.f, 2.f, 3.f, 4.f,
                                                            1.f, 2.f, 3.f, 4.f,
                                                            1.f, 2.f, 3.f, 4.f});
    auto axis = NDArrayFactory::create<int>('c', {1}, {ax});
    //************************************//

    nd4j::ops::reduce_sum_bp op;
    auto result = op.execute({&input, &eps, &axis}, {}, {}, {true});

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

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

    ASSERT_EQ(Status::OK(), result->status());    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceMean_test8) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>('c', {1,1,1}, {12.5f});
    auto axes = NDArrayFactory::create<int>({0, 1, 2});
    x.linspace(1);

    nd4j::ops::reduce_mean op;
    auto result = op.execute({&x, &axes}, {}, {}, {true});
    auto output = result->at(0);

    ASSERT_EQ(Status::OK(), result->status());

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
    ASSERT_EQ(Status::OK(), result->status());    
    auto output = result->at(0);

    // output->printShapeInfo("o");

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO2}, {1}, {});
    ASSERT_EQ(Status::OK(), result->status());    
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
    ASSERT_EQ(Status::OK(), result->status());    
    auto output = result->at(0);
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO2}, {1}, {0});
    ASSERT_EQ(Status::OK(), result->status());    
    output = result->at(0);
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceMeanBP_test02) {

    int ax = 0;
    auto x = NDArrayFactory::create<double>('c', {3,4});
    auto gradO1 = NDArrayFactory::create<double>('c', {4},  {1.f, 2.f, 3.f, 4.f});
    auto gradO2 = NDArrayFactory::create<double>('c', {1,4}, {1.f, 2.f, 3.f, 4.f});
    auto exp = NDArrayFactory::create<double>('c', {3,4}, {1.f/3.f, 2.f/3.f, 1.f, 4.f/3.f, 1.f/3.f, 2.f/3.f, 1.f, 4.f/3.f, 1.f/3.f, 2.f/3.f, 1.f, 4.f/3.f});
    auto axis = NDArrayFactory::create<int>('c', {1}, {ax});
    x.linspace(1);

    nd4j::ops::reduce_mean_bp op;

    auto result = op.execute({&x, &gradO1, &axis}, {}, {}, {false});
    ASSERT_EQ(Status::OK(), result->status());
    auto output = result->at(0);
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO2, &axis}, {}, {}, {true});
    ASSERT_EQ(Status::OK(), result->status());
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
    ASSERT_EQ(Status::OK(), result->status());    
    auto output = result->at(0);
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output)); 
    delete result;

    result = op.execute({&x, &gradO2}, {1}, {1});
    ASSERT_EQ(Status::OK(), result->status());    
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
    ASSERT_EQ(Status::OK(), result->status());        
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
    int dataFormat  = 1;             // 1-NHWC, 0-NDHW

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
    input.syncToDevice();

    nd4j::ops::avgpool2d op;
    auto results = op.execute({&input}, {}, {kH,kW,  sH,sW,  pH,pW,  dH,dW,  paddingMode, 0, dataFormat});
    auto output = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());

    //output->printIndexedBuffer("output");
    //expected.printIndexedBuffer("expected");

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

    ASSERT_EQ(Status::OK(), results->status());

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

    ASSERT_EQ(Status::OK(), results->status());

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

    ASSERT_EQ(Status::OK(), results->status());

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

    ASSERT_EQ(Status::OK(), results->status());

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

    ASSERT_EQ(Status::OK(), results->status());

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

    ASSERT_EQ(Status::OK(), results->status());

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

    ASSERT_EQ(Status::OK(), results->status());

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

    ASSERT_EQ(Status::OK(), results->status());

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

    ASSERT_EQ(Status::OK(), results->status());

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

    ASSERT_EQ(Status::OK(), results->status());

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
    auto result = op.execute({&x}, {1.f}, {}, {}, false, nd4j::DataType::DOUBLE);
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
    auto result = op.execute({&x}, {15.f}, {0}, {}, false, nd4j::DataType::DOUBLE);
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
    auto result = op.execute({&x}, {15.f}, {1}, {}, false, nd4j::DataType::DOUBLE);
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
    auto result = op.execute({&x}, {15.f}, {0,1}, {}, false, nd4j::DataType::DOUBLE);
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
    auto result = op.execute({&x}, {15.}, {}, {}, false, nd4j::DataType::DOUBLE);
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
    auto result = op.execute({&x}, {4.}, {}, {}, false, nd4j::DataType::DOUBLE);
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
    auto result = op.execute({&x}, {5.}, {}, {}, false, nd4j::DataType::DOUBLE);
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
    auto result = op.execute({&x}, {35.}, {0, 2}, {}, false, nd4j::DataType::DOUBLE);
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}


TEST_F(DeclarableOpsTests8, clipbynorm_test_tf_119_1) {
    auto x = NDArrayFactory::create<double>('c', {3, 3}, {1, 2, 3, 4, 5,6, 7, 8, 9});
    auto e = NDArrayFactory::create<double>('c', {3, 3}, {0.03198684, 0.06397368, 0.09596053, 0.12794736, 0.15993419, 0.19192106, 0.22390789, 0.25589472, 0.28788155});

    nd4j::ops::clipbynorm op;
    auto result = op.execute({&x}, {0.54}, {}, {}, false, nd4j::DataType::DOUBLE);

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
    ASSERT_EQ(Status::OK(), result->status());    
    auto output = result->at(0);    
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO2}, {1}, {0});
    ASSERT_EQ(Status::OK(), result->status());    
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
    ASSERT_EQ(Status::OK(), result->status());    
    auto output = result->at(0);    
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO2}, {1}, {1});
    ASSERT_EQ(Status::OK(), result->status());    
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
    ASSERT_EQ(Status::OK(), result->status());    
    auto output = result->at(0);        
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO2}, {1}, {0});
    ASSERT_EQ(Status::OK(), result->status());    
    output = result->at(0);    
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, zeros_as_test1) {

    auto x = NDArrayFactory::create<double>(10.f);
    auto y = NDArrayFactory::create<double>(100.f);
    auto exp = NDArrayFactory::create<double>(0.f);
                                                                          
    nd4j::ops::zeros_as op;

    Nd4jStatus status = op.execute({&x}, {&y}, {}, {}, {});
    ASSERT_EQ(Status::OK(), status);    
    
    ASSERT_TRUE(y.isSameShape(exp));
    ASSERT_TRUE(y.equalsTo(exp));

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, zeros_as_test2) {

    auto x = NDArrayFactory::create<float>(10.f);
    //auto y = NDArrayFactory::create<float>(100.f);
    auto exp = NDArrayFactory::create<float>(0.f);

    nd4j::ops::zeros_as op;

    auto result = op.execute({&x}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());
    auto y = result->at(0);

    ASSERT_TRUE(y->isSameShape(exp));
    ASSERT_TRUE(y->equalsTo(exp));
    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, ones_as_test1) {

    auto x = NDArrayFactory::create<double>(10.);
    auto y = NDArrayFactory::create<double>(100.);
    auto exp = NDArrayFactory::create<double>(1.);

    nd4j::ops::ones_as op;

    Nd4jStatus status = op.execute({&x}, {&y}, {}, {}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(Status::OK(), status);    
    
    ASSERT_TRUE(y.isSameShape(exp));
    ASSERT_TRUE(y.equalsTo(exp));
        
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, ones_as_test2) {

    auto x = NDArrayFactory::create<double>(10.);
    //auto y = NDArrayFactory::create<double>(100.);
    auto exp = NDArrayFactory::create<double>(1.);

    nd4j::ops::ones_as op;

    auto results = op.execute({&x}, {}, {}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(Status::OK(), results->status());
    auto y = results->at(0);
    ASSERT_TRUE(y->isSameShape(exp));
    ASSERT_TRUE(y->equalsTo(exp));

    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, NormalizeMoments_SGO_1) {

    auto data   = NDArrayFactory::create<double>('c', {10, 10});
    data.linspace(1);
    
    auto means = data.reduceAlongDimension(reduce::Sum, {0});
    auto deviance = NDArrayFactory::create<double>('c', {10}, {825., 825. , 825., 825., 825., 825., 825., 825., 825., 825. }); // data.varianceAlongDimension(variance::SummaryStatsVariance, false, {0}); // = NDArrayFactory::create<double>('c', {10, 10});

    auto counts = NDArrayFactory::create<double>(10.0);

//    auto expMeans = NDArrayFactory::create<double>('c', {10, 10});

//    auto expDeviance = NDArrayFactory::create<double>('c', {10, 10});
    auto squared = NDArrayFactory::create<double>('c', {10, 10});
    data.applyTransform(transform::Square, &squared, nullptr);
    auto ssSquared = squared.reduceAlongDimension(reduce::Sum, {0});
//    ssSquared->printBuffer("Sum squared");
//    squared.printBuffer("Squared");
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
//    deviance.printIndexedBuffer("Expected");
//    means->printIndexedBuffer("Expected means");
    ASSERT_TRUE(means->isSameShape(outputMeans));
    ASSERT_TRUE(means->equalsTo(outputMeans));    
    ASSERT_TRUE(deviance.isSameShape(outputDeviance));
    ASSERT_TRUE(deviance.equalsTo(outputDeviance));
    delete means;
    //delete deviance;
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

    ASSERT_EQ(Status::OK(), result->status());    

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
    ASSERT_EQ(Status::OK(), result->status());    

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
    ASSERT_EQ(Status::OK(), result->status());    

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
    ASSERT_EQ(Status::OK(), result->status());    

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
    ASSERT_EQ(Status::OK(), result->status());    

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
    ASSERT_EQ(Status::OK(), result->status());    

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
TYPED_TEST(TypedDeclarableOpsTests8, LrnTest_01) {

    auto x = NDArrayFactory::create<TypeParam>('c', {1, 1, 2, 5}, { 1, 2., 3, 4, 5,
                                                                    6.,  7., 8, 9, 10}
    );

   auto exp = NDArrayFactory::create<TypeParam>('c', {1, 1, 2, 5}, {0.2581989 , 0.3592106 , 0.40089184, 0.53935987, 0.70014, 0.4898979 , 0.46056613, 0.43971977, 0.5240003 , 0.6375767 }//            0.72760683, 0.4850712,   0.5848977, 0.67488194,
//            0.7581754,  0.58321184, 0.86747235, 0.4048204}
   );

    nd4j::ops::lrn op;
    auto  results = op.execute({&x}, {1.0, 1.0, 0.5}, {2}, {}, false, nd4j::DataType::DOUBLE);
    auto out = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    //ASSERT_TRUE(exp.isSameShape(out));
    //out->printBuffer("LRN out");
    //exp.printBuffer("LRN exp");
    ASSERT_TRUE(exp.equalsTo(out));

    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedDeclarableOpsTests8, LrnTest_02) {

    auto x = NDArrayFactory::create<TypeParam>('c', {1, 1, 1, 6}, { 1, 2., 3, 4, 5, 6});

    auto exp = NDArrayFactory::create<TypeParam>('c', {1, 1, 1, 6}, {
        0.2581989 , 0.3592106 , 0.40089184, 0.4193139, 0.5360563, 0.67936623}
    );

    nd4j::ops::lrn op;
    auto  results = op.execute({&x}, {1.0, 1.0, 0.5}, {2}, {}, false, nd4j::DataType::DOUBLE);
    auto out = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    //ASSERT_TRUE(exp.isSameShape(out));
    //out->printIndexedBuffer("LRN out");
//    exp.printIndexedBuffer("LRN exp");
    ASSERT_TRUE(exp.equalsTo(out));

    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedDeclarableOpsTests8, LrnTest_03) {

    auto x = NDArrayFactory::create<TypeParam>('c', {1, 1, 1, 10}, { 1, 2., 3, 4, 5, 6, 7, 8, 9, 10});
    auto exp = NDArrayFactory::create<TypeParam>('c', {1, 1, 1, 10}, {0.10425719, 0.16843036, 0.2095291 , 0.23652494, 0.25449327,0.3053919 , 0.35675305, 0.4098524 , 0.46662825, 0.52999896});

    nd4j::ops::lrn op;
    auto  results = op.execute({&x}, {1.0, 1.0, 0.5}, {5}, {}, false, nd4j::DataType::DOUBLE);
    auto out = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(out));
    // out->printIndexedBuffer("LRN out");
//    exp.printIndexedBuffer("LRN exp");
    ASSERT_TRUE(exp.equalsTo(out));

    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedDeclarableOpsTests8, LrnTest_1) {

    auto x = NDArrayFactory::create<TypeParam>('c', {2, 2, 2, 2}, { 5.5, 0., 0.3, 5.5,
                                            8.6, 0.,  0., 0.4,
                                            1.5, 1., 1.3, 1.5,
                                            2.6, 2.,  3., 1.4}
    );

    auto exp = NDArrayFactory::create<TypeParam>('c', {2, 2, 2, 2}, {
                                            0.98386997,        0.,  0.05358852,  0.9824562,
                                            0.99330735,        0.,          0., 0.37139067,
                                            0.72760683, 0.4850712,   0.5848977, 0.67488194,
                                            0.7581754,  0.58321184, 0.86747235, 0.4048204}
    );

    nd4j::ops::lrn op;
    auto  results = op.execute({&x}, {1.0, 1.0, 0.5}, {2}, {}, false, nd4j::DataType::DOUBLE);
    auto out = results->at(0);
        
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(out));
//    out->printIndexedBuffer("LRN out");
//    exp.printIndexedBuffer("LRN exp");
    ASSERT_TRUE(exp.equalsTo(out));    
    
    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedDeclarableOpsTests8, LrnTest_2) {

    auto x = NDArrayFactory::create<TypeParam>('c', {3, 3, 5, 5});
    x.linspace(1);
    
    auto exp = NDArrayFactory::create<TypeParam>('c', {3, 3, 5, 5}, {
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
    auto  results = op.execute({&x}, {1.0, 1.0, 0.5}, {2}, {}, false, nd4j::DataType::DOUBLE);
    auto out = results->at(0);
        
    ASSERT_EQ(Status::OK(), results->status());
//    ASSERT_TRUE(exp.isSameShape(out));
//    out->printIndexedBuffer("LRN out");
//    exp.printIndexedBuffer("LRN exp");
    ASSERT_TRUE(exp.equalsTo(out));    
    
    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedDeclarableOpsTests8, LrnTest_3) {

    auto x = NDArrayFactory::create<TypeParam>('c', {3, 3, 5, 5});
    x.linspace(1);

    auto exp = NDArrayFactory::create<TypeParam>('c', {3, 3, 5, 5}, {
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
    auto  results = op.execute({&x}, {1.0, 1.0, 0.5}, {2}, {}, false, nd4j::DataType::FLOAT32);
    auto out = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
//    ASSERT_TRUE(exp.isSameShape(out));
//    out->printIndexedBuffer("LRN out");
//    exp.printIndexedBuffer("LRN exp");
    ASSERT_TRUE(exp.equalsTo(out));

    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedDeclarableOpsTests8, LrnTest_4) {

    // auto x = NDArrayFactory::create<TypeParam>('c', {8, 32, 64, 64});
    auto x = NDArrayFactory::create<TypeParam>('c', {2, 8, 16, 16});
    x.linspace(1);

    nd4j::ops::lrn op;
    auto  results = op.execute({&x}, {1.0, 1.0, 0.5}, {2}, {}, false, nd4j::DataType::DOUBLE);
    auto out = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
//    ASSERT_TRUE(exp.isSameShape(out));
//    out->printIndexedBuffer("LRN out");
//    exp.printIndexedBuffer("LRN exp");
//    ASSERT_TRUE(exp.equalsTo(out));

    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedDeclarableOpsTests8, LrnTest_4_119) {
    int iterations = 1000;
    // auto x = NDArrayFactory::create<TypeParam>('c', {8, 32, 64, 64});
    // auto z = NDArrayFactory::create<TypeParam>('c', {8, 32, 64, 64});
    auto x = NDArrayFactory::create<TypeParam>('c', {2, 8, 16, 16});
    auto z = NDArrayFactory::create<TypeParam>('c', {2, 8, 16, 16});
    x.linspace(1);

    nd4j::ops::lrn op;

    op.execute({&x}, {&z}, {1.0, 1.0, 0.5}, {2}, {}, false, nd4j::DataType::DOUBLE);

    auto timeStart = std::chrono::system_clock::now();

    for (int e = 0; e < iterations; e++)
        op.execute({&x}, {&z}, {1.0, 1.0, 0.5}, {2}, {}, false, nd4j::DataType::DOUBLE);

    auto timeEnd = std::chrono::system_clock::now();
    auto spanTime = std::chrono::duration_cast<std::chrono::microseconds> ((timeEnd - timeStart) / iterations).count();
    auto ttlTime = std::chrono::duration_cast<std::chrono::milliseconds> ((timeEnd - timeStart)).count();


    //ASSERT_EQ(Status::OK(), results);

    nd4j_printf("avg time: %lld ms\n", spanTime);

//    ASSERT_TRUE(exp.isSameShape(out));
//    out->printIndexedBuffer("LRN out");
//    exp.printIndexedBuffer("LRN exp");
//    ASSERT_TRUE(exp.equalsTo(out));
}

////////////////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedDeclarableOpsTests8, LrnTest_5) {

    auto x = NDArrayFactory::create<TypeParam>('f', {8, 32, 64, 64});
    x.linspace(1);

    nd4j::ops::lrn op;
    auto  results = op.execute({&x}, {1.0, 1.0, 0.5}, {2}, {}, false, nd4j::DataType::DOUBLE);
    auto out = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
//    ASSERT_TRUE(exp.isSameShape(out));
//    out->printIndexedBuffer("LRN out");
//    exp.printIndexedBuffer("LRN exp");
//    ASSERT_TRUE(exp.equalsTo(out));

    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedDeclarableOpsTests8, LrnTest_BP_01) {

    auto x = NDArrayFactory::create<double>( 'c', {1, 1, 1, 10});
    x.linspace(1);
    auto eps = NDArrayFactory::create<double>('c', {1,1,1,10});
    eps.linspace(1);
//
//    auto exp = NDArrayFactory::create<double>('c', {3,3,5,5}, {
//            0.238337, 0.309664, 0.334077, 0.376534, 0.342926, 0.370734, 0.362017, 0.354182, 0.379140, 0.376275, 0.380027, 0.368347, 0.356401, 0.378316, 0.381315, 0.382465, 0.370592, 0.357055, 0.377670, 0.382950, 0.383445, 0.371718, 0.357332, 0.377217, 0.383677, 0.383933, 0.372391, 0.357475, 0.376891, 0.384062, 0.384212, 0.372837, 0.357557, 0.376646, 0.384290, 0.384385, 0.373153, 0.357610, 0.376457, 0.384436, 0.384500, 0.373389, 0.357645, 0.376306, 0.384536, 0.384581, 0.373572, 0.357670, 0.376184, 0.384606, 0.384639, 0.373718, 0.357688, 0.376082, 0.384658, 0.384683, 0.373837, 0.357702, 0.375996, 0.384698, 0.384717, 0.373935, 0.357712, 0.375923, 0.384728, 0.384743, 0.374019, 0.357721, 0.375860, 0.384752, 0.384764, 0.374090, 0.357727, 0.375804, 0.384771, 0.384781, 0.374152, 0.357733, 0.375756, 0.384787, 0.384795, 0.374205, 0.357737, 0.375713, 0.384800, 0.384807, 0.374253, 0.357741, 0.375674, 0.384811, 0.384817, 0.374295, 0.357744, 0.375640, 0.384820, 0.384825, 0.374333, 0.357747, 0.375609, 0.384828, 0.384832, 0.374366, 0.357749, 0.375581, 0.384835, 0.384839, 0.374397, 0.357751, 0.375555, 0.384841, 0.384844, 0.374425, 0.357753, 0.375531, 0.384846, 0.384849, 0.374450, 0.357754, 0.375510, 0.384850, 0.384853, 0.374473, 0.357756, 0.375490, 0.384854, 0.384856, 0.374494, 0.357757, 0.375471, 0.384858, 0.384860, 0.374514, 0.357758, 0.375454, 0.384861, 0.384863, 0.374532, 0.357759, 0.375438, 0.384864, 0.384865, 0.374549, 0.357760, 0.375423, 0.384866, 0.384868, 0.374565, 0.357760, 0.375410, 0.384868, 0.384870, 0.374579, 0.357761, 0.375397, 0.384870, 0.384872, 0.374593, 0.357762, 0.375384, 0.384872, 0.384873, 0.374606, 0.357762, 0.375373, 0.384874, 0.384875, 0.374618, 0.357763, 0.375362, 0.384875, 0.384876, 0.374629, 0.357763, 0.375352, 0.384877, 0.384878, 0.374640, 0.357764, 0.375342, 0.384878, 0.384879, 0.374650, 0.357764, 0.375333, 0.384879, 0.384880, 0.374660, 0.357764, 0.375325, 0.384880, 0.384881, 0.374669, 0.357765, 0.375316, 0.384881, 0.384882, 0.374677, 0.357765, 0.375309, 0.384882, 0.384883, 0.374685, 0.357765, 0.375301, 0.384883, 0.384884, 0.374693, 0.357765, 0.375294, 0.384884, 0.384884, 0.374700, 0.357766, 0.375287, 0.384885, 0.384885, 0.374707, 0.357766, 0.375281, 0.384885, 0.384886, 0.374714, 0.357766, 0.375275, 0.384886}
//    );
///
    nd4j::ops::lrn_bp op;
    auto  results = op.execute({&x, &eps}, {1.0, 1.0, 0.5}, {5}, {}, false, nd4j::DataType::DOUBLE);
    auto out = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
//    ASSERT_TRUE(exp.isSameShape(out));
    //out->printBuffer("LRN BP out");
    //exp.printBuffer("LRN BP exp");
    //ASSERT_TRUE(exp.equalsTo(out));

    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedDeclarableOpsTests8, LrnTest_BP_02) {

    auto x = NDArrayFactory::create<double>( 'c', {1, 1, 1, 10});
    x.linspace(1);
    auto eps = NDArrayFactory::create<double>('c', {1,1,1,10});
    eps.linspace(1);
//
//    auto exp = NDArrayFactory::create<double>('c', {3,3,5,5}, {
//            0.238337, 0.309664, 0.334077, 0.376534, 0.342926, 0.370734, 0.362017, 0.354182, 0.379140, 0.376275, 0.380027, 0.368347, 0.356401, 0.378316, 0.381315, 0.382465, 0.370592, 0.357055, 0.377670, 0.382950, 0.383445, 0.371718, 0.357332, 0.377217, 0.383677, 0.383933, 0.372391, 0.357475, 0.376891, 0.384062, 0.384212, 0.372837, 0.357557, 0.376646, 0.384290, 0.384385, 0.373153, 0.357610, 0.376457, 0.384436, 0.384500, 0.373389, 0.357645, 0.376306, 0.384536, 0.384581, 0.373572, 0.357670, 0.376184, 0.384606, 0.384639, 0.373718, 0.357688, 0.376082, 0.384658, 0.384683, 0.373837, 0.357702, 0.375996, 0.384698, 0.384717, 0.373935, 0.357712, 0.375923, 0.384728, 0.384743, 0.374019, 0.357721, 0.375860, 0.384752, 0.384764, 0.374090, 0.357727, 0.375804, 0.384771, 0.384781, 0.374152, 0.357733, 0.375756, 0.384787, 0.384795, 0.374205, 0.357737, 0.375713, 0.384800, 0.384807, 0.374253, 0.357741, 0.375674, 0.384811, 0.384817, 0.374295, 0.357744, 0.375640, 0.384820, 0.384825, 0.374333, 0.357747, 0.375609, 0.384828, 0.384832, 0.374366, 0.357749, 0.375581, 0.384835, 0.384839, 0.374397, 0.357751, 0.375555, 0.384841, 0.384844, 0.374425, 0.357753, 0.375531, 0.384846, 0.384849, 0.374450, 0.357754, 0.375510, 0.384850, 0.384853, 0.374473, 0.357756, 0.375490, 0.384854, 0.384856, 0.374494, 0.357757, 0.375471, 0.384858, 0.384860, 0.374514, 0.357758, 0.375454, 0.384861, 0.384863, 0.374532, 0.357759, 0.375438, 0.384864, 0.384865, 0.374549, 0.357760, 0.375423, 0.384866, 0.384868, 0.374565, 0.357760, 0.375410, 0.384868, 0.384870, 0.374579, 0.357761, 0.375397, 0.384870, 0.384872, 0.374593, 0.357762, 0.375384, 0.384872, 0.384873, 0.374606, 0.357762, 0.375373, 0.384874, 0.384875, 0.374618, 0.357763, 0.375362, 0.384875, 0.384876, 0.374629, 0.357763, 0.375352, 0.384877, 0.384878, 0.374640, 0.357764, 0.375342, 0.384878, 0.384879, 0.374650, 0.357764, 0.375333, 0.384879, 0.384880, 0.374660, 0.357764, 0.375325, 0.384880, 0.384881, 0.374669, 0.357765, 0.375316, 0.384881, 0.384882, 0.374677, 0.357765, 0.375309, 0.384882, 0.384883, 0.374685, 0.357765, 0.375301, 0.384883, 0.384884, 0.374693, 0.357765, 0.375294, 0.384884, 0.384884, 0.374700, 0.357766, 0.375287, 0.384885, 0.384885, 0.374707, 0.357766, 0.375281, 0.384885, 0.384886, 0.374714, 0.357766, 0.375275, 0.384886}
//    );
///
    nd4j::ops::lrn opFF;
    nd4j::ops::lrn_bp opBP;

    const OpArgsHolder argsHolderFF({&x},         {1., 1., 0.5}, {5});
    const OpArgsHolder argsHolderBP({&x, &eps}, {1., 1., 0.5}, {5});

    bool gradOK = true; //GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);
    //auto  results = op.execute({&x, &eps}, {1.0, 1.0, 0.5}, {5}, {}, false, nd4j::DataType::DOUBLE);
    //auto out = results->at(0);

    //ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(gradOK);
    //out->printBuffer("LRN BP out");
    //exp.printBuffer("LRN BP exp");
    //ASSERT_TRUE(exp.equalsTo(out));

    //delete results;
}

////////////////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedDeclarableOpsTests8, LrnTest_BP_1) {

    auto x = NDArrayFactory::create<TypeParam>( 'c', {3, 3, 5, 5});
    x.linspace(1);
    auto eps = NDArrayFactory::create<TypeParam>('c', {3,3,5,5});
    eps.linspace(1);
//
auto exp = NDArrayFactory::create<TypeParam>('c', {3,3,5,5}, {
        0.238337, 0.309664, 0.334077, 0.376534, 0.342926, 0.370734, 0.362017, 0.354182, 0.379140, 0.376275, 0.380027, 0.368347, 0.356401, 0.378316, 0.381315, 0.382465, 0.370592, 0.357055, 0.377670, 0.382950, 0.383445, 0.371718, 0.357332, 0.377217, 0.383677, 0.383933, 0.372391, 0.357475, 0.376891, 0.384062, 0.384212, 0.372837, 0.357557, 0.376646, 0.384290, 0.384385, 0.373153, 0.357610, 0.376457, 0.384436, 0.384500, 0.373389, 0.357645, 0.376306, 0.384536, 0.384581, 0.373572, 0.357670, 0.376184, 0.384606, 0.384639, 0.373718, 0.357688, 0.376082, 0.384658, 0.384683, 0.373837, 0.357702, 0.375996, 0.384698, 0.384717, 0.373935, 0.357712, 0.375923, 0.384728, 0.384743, 0.374019, 0.357721, 0.375860, 0.384752, 0.384764, 0.374090, 0.357727, 0.375804, 0.384771, 0.384781, 0.374152, 0.357733, 0.375756, 0.384787, 0.384795, 0.374205, 0.357737, 0.375713, 0.384800, 0.384807, 0.374253, 0.357741, 0.375674, 0.384811, 0.384817, 0.374295, 0.357744, 0.375640, 0.384820, 0.384825, 0.374333, 0.357747, 0.375609, 0.384828, 0.384832, 0.374366, 0.357749, 0.375581, 0.384835, 0.384839, 0.374397, 0.357751, 0.375555, 0.384841, 0.384844, 0.374425, 0.357753, 0.375531, 0.384846, 0.384849, 0.374450, 0.357754, 0.375510, 0.384850, 0.384853, 0.374473, 0.357756, 0.375490, 0.384854, 0.384856, 0.374494, 0.357757, 0.375471, 0.384858, 0.384860, 0.374514, 0.357758, 0.375454, 0.384861, 0.384863, 0.374532, 0.357759, 0.375438, 0.384864, 0.384865, 0.374549, 0.357760, 0.375423, 0.384866, 0.384868, 0.374565, 0.357760, 0.375410, 0.384868, 0.384870, 0.374579, 0.357761, 0.375397, 0.384870, 0.384872, 0.374593, 0.357762, 0.375384, 0.384872, 0.384873, 0.374606, 0.357762, 0.375373, 0.384874, 0.384875, 0.374618, 0.357763, 0.375362, 0.384875, 0.384876, 0.374629, 0.357763, 0.375352, 0.384877, 0.384878, 0.374640, 0.357764, 0.375342, 0.384878, 0.384879, 0.374650, 0.357764, 0.375333, 0.384879, 0.384880, 0.374660, 0.357764, 0.375325, 0.384880, 0.384881, 0.374669, 0.357765, 0.375316, 0.384881, 0.384882, 0.374677, 0.357765, 0.375309, 0.384882, 0.384883, 0.374685, 0.357765, 0.375301, 0.384883, 0.384884, 0.374693, 0.357765, 0.375294, 0.384884, 0.384884, 0.374700, 0.357766, 0.375287, 0.384885, 0.384885, 0.374707, 0.357766, 0.375281, 0.384885, 0.384886, 0.374714, 0.357766, 0.375275, 0.384886}
    );
///
    nd4j::ops::lrn_bp op;
    auto  results = op.execute({&x, &eps}, {1.0, 1.0, 0.5}, {2}, {}, false, typeid(TypeParam) == typeid(float) ? nd4j::DataType::FLOAT32 : nd4j::DataType::DOUBLE);
    auto out = results->at(0);
        
    ASSERT_EQ(Status::OK(), results->status());
//    ASSERT_TRUE(exp.isSameShape(out));
    // out->printBuffer("LRN BP out");
    // exp.printBuffer("LRN BP exp");
    //ASSERT_TRUE(exp.equalsTo(out));
    
    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedDeclarableOpsTests8, LrnTest_BP_2) {

    auto x = NDArrayFactory::create<TypeParam>( 'c', {3, 3, 5, 5});
    x.linspace(1);
    
    auto eps = NDArrayFactory::create<TypeParam>('c', {3, 3, 5, 5}, {            0.2581989 ,0.3592106 , 0.40089184, 0.53935987, 0.70014,
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
                                                                                 0.57474375, 0.49886885, 0.44720373, 0.50111103, 0.5799219 });
//
    auto exp = NDArrayFactory::create<TypeParam>('c', {3,3,5,5}, {
            0.061538, 0.055617, 0.044643, 0.050772, 0.048019, 0.030270, 0.023819, 0.019468, 0.022074, 0.023990, 0.018221, 0.014664, 0.012182, 0.013954, 0.015685, 0.012967, 0.010563, 0.008841, 0.010185, 0.011621, 0.010052, 0.008248, 0.006934, 0.008015, 0.009222, 0.008204, 0.006764, 0.005702, 0.006606, 0.007642, 0.006929, 0.005732, 0.004841, 0.005618, 0.006523, 0.005996, 0.004973, 0.004205, 0.004887, 0.005689, 0.005284, 0.004391, 0.003717, 0.004324, 0.005044, 0.004723, 0.003931, 0.003331, 0.003877, 0.004531, 0.004270, 0.003558, 0.003017, 0.003514, 0.004112, 0.003896, 0.003250, 0.002757, 0.003213, 0.003764, 0.003582, 0.002991, 0.002539, 0.002959, 0.003470, 0.003315, 0.002770, 0.002352, 0.002743, 0.003219, 0.003085, 0.002580, 0.002191, 0.002556, 0.003002, 0.002885, 0.002414, 0.002051, 0.002393, 0.002812, 0.002709, 0.002268, 0.001927, 0.002250, 0.002645, 0.002553, 0.002138, 0.001818, 0.002122, 0.002496, 0.002415, 0.002023, 0.001720, 0.002009, 0.002363, 0.002290, 0.001920, 0.001632, 0.001906, 0.002244, 0.002178, 0.001826, 0.001553, 0.001814, 0.002136, 0.002076, 0.001741, 0.001481, 0.001731, 0.002038, 0.001984, 0.001664, 0.001416, 0.001654, 0.001949, 0.001899, 0.001593, 0.001356, 0.001584, 0.001867, 0.001821, 0.001528, 0.001301, 0.001520, 0.001792, 0.001750, 0.001469, 0.001250, 0.001461, 0.001722, 0.001683, 0.001413, 0.001203, 0.001406, 0.001658, 0.001622, 0.001362, 0.001159, 0.001355, 0.001599, 0.001565, 0.001314, 0.001119, 0.001308, 0.001543, 0.001512, 0.001270, 0.001081, 0.001264, 0.001491, 0.001462, 0.001228, 0.001046, 0.001223, 0.001443, 0.001415, 0.001189, 0.001013, 0.001184, 0.001397, 0.001372, 0.001153, 0.000982, 0.001148, 0.001355, 0.001331, 0.001118, 0.000952, 0.001114, 0.001315, 0.001292, 0.001086, 0.000925, 0.001082, 0.001277, 0.001255, 0.001055, 0.000899, 0.001051, 0.001241, 0.001221, 0.001026, 0.000874, 0.001023, 0.001208, 0.001188, 0.000999, 0.000851, 0.000996, 0.001176, 0.001157, 0.000973, 0.000829, 0.000970, 0.001145, 0.001128, 0.000949, 0.000808, 0.000945, 0.001117, 0.001100, 0.000925, 0.000788, 0.000922, 0.001089, 0.001073, 0.000903, 0.000769, 0.000900, 0.001063, 0.001048, 0.000882, 0.000751, 0.000879, 0.001038, 0.001024, 0.000861, 0.000734, 0.000859, 0.001015, 0.001001, 0.000842, 0.000717, 0.000840, 0.000992}
        //    0.009859, 0.013075, 0.013874, 0.017893, 0.022344, 0.014551, 0.012859, 0.011511, 0.013311, 0.015834, 0.012025, 0.010047, 0.008601, 0.009920, 0.011885, 0.009505, 0.007636, 0.006299, 0.007413, 0.009095, 0.007446, 0.005743, 0.004540, 0.005533, 0.007033, 0.005821, 0.004282, 0.003209, 0.004123, 0.005491, 0.004577, 0.003198, 0.002247, 0.003097, 0.004355, 0.003652, 0.002412, 0.001565, 0.002357, 0.003517, 0.002965, 0.001844, 0.001084, 0.001821, 0.002893, 0.002451, 0.001430, 0.000741, 0.001428, 0.002422, -0.111434, -0.105946, -0.100351, -0.091868, -0.083323, -0.078775, -0.076222, -0.073291, -0.067635, -0.061692, -0.058943, -0.057832, -0.056263, -0.052198, -0.047768, -0.046002, -0.045655, -0.044839, -0.041748, -0.038271, -0.037084, -0.037161, -0.036786, -0.034331, -0.031495, 0.000077, -0.000673, -0.001181, -0.000667, 0.000079, -0.000089, -0.000802, -0.001285, -0.000793, -0.000079, -0.000228, -0.000908, -0.001368, -0.000896, -0.000212, -0.000345, -0.000996, -0.001434, -0.000981, -0.000325, -0.000444, -0.001067, -0.001487, -0.001051, -0.000421, 0.000697, 0.000188, -0.000152, 0.000210, 0.000731, 0.000650, 0.000165, -0.000161, 0.000185, 0.000683, 0.000610, 0.000145, -0.000168, 0.000164, 0.000641, 0.000574, 0.000128, -0.000172, 0.000146, 0.000604, 0.000542, 0.000113, -0.000175, 0.000131, 0.000571, -0.009490, -0.010070, -0.010409, -0.009734, -0.008834, -0.008785, -0.009351, -0.009687, -0.009054, -0.008207, -0.008167, -0.008718, -0.009050, -0.008455, -0.007654, -0.007622, -0.008159, -0.008485, -0.007924, -0.007164, -0.007138, -0.007661, -0.007981, -0.007450, -0.006728, -0.000901, -0.001327, -0.001614, -0.001310, -0.000869, -0.000913, -0.001328, -0.001607, -0.001310, -0.000882, -0.000922, -0.001326, -0.001598, -0.001309, -0.000892, -0.000930, -0.001323, -0.001588, -0.001306, -0.000900, -0.000936, -0.001319, -0.001577, -0.001302, -0.000906, 0.000339, 0.000038, -0.000164, 0.000048, 0.000355, 0.000328, 0.000035, -0.000162, 0.000045, 0.000343, 0.000318, 0.000033, -0.000159, 0.000041, 0.000332, 0.000308, 0.000030, -0.000157, 0.000039, 0.000322, 0.000299, 0.000028, -0.000155, 0.000036, 0.000312, -0.004085, -0.004479, -0.004733, -0.004396, -0.003925, -0.003925, -0.004309, -0.004558, -0.004232, -0.003775, -0.003776, -0.004151, -0.004395, -0.004079, -0.003636, -0.003637, -0.004004, -0.004242, -0.003936, -0.003505, -0.003507, -0.003866, -0.004100, -0.003802, -0.003383}
    );

    nd4j::ops::lrn_bp op;
    auto  results = op.execute({&x, &eps}, {1.0, 1.0, 0.5}, {2}, {}, false, typeid(TypeParam) == typeid(float) ? nd4j::DataType::FLOAT32 : nd4j::DataType::DOUBLE);
    auto out = results->at(0);
        
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(out));
    //out->printBuffer("LRN BP out");
//    exp.printIndexedBuffer("LRN exp");
   // ASSERT_TRUE(exp.equalsTo(out));
    
    delete results;
}


