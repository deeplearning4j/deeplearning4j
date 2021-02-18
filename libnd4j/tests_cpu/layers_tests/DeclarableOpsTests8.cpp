/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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
#include <array/NDArray.h>
#include <helpers/GradCheck.h>
// #include <array/NDArrayList.h>

using namespace sd;


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

    sd::ops::reduce_variance op;
    auto result = op.evaluate({&x}, {}, {0,1});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceVariance_test2) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.f});
    auto exp = NDArrayFactory::create<double>('c', {1,1,4}, {602.2222f, 727.13885f, 993.5555f, 755.8889f});

    sd::ops::reduce_variance op;
    auto result = op.evaluate({&x}, {1.}, {0,1});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceVariance_test3) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.f});
    auto exp = NDArrayFactory::create<double>('c', {3}, {900.9375f, 969.8594f, 424.1875f});

    sd::ops::reduce_variance op;
    auto result = op.evaluate({&x}, {}, {0,2});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceVariance_test4) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.f});
    auto exp = NDArrayFactory::create<double>('c', {1,3,1}, {900.9375f, 969.8594f, 424.1875f});

    sd::ops::reduce_variance op;
    auto result = op.evaluate({&x}, {1.}, {0,2});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceVariance_test5) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.f});
    auto exp = NDArrayFactory::create<double>(788.6927f);

    sd::ops::reduce_variance op;
    auto result = op.evaluate({&x}, {}, {});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceVariance_test6) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    auto exp = NDArrayFactory::create<double>(788.6927f);

    sd::ops::reduce_variance op;
    auto result = op.evaluate({&x}, {}, {0,1,2});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceVariance_test7) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    auto exp = NDArrayFactory::create<double>('c', {1,1,1}, {788.6927f});

    sd::ops::reduce_variance op;
    auto result = op.evaluate({&x}, {1.}, {0,1,2});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceVariance_test8) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    auto exp = NDArrayFactory::create<double>('c', {1,1,1}, {788.6927f});
    auto axes = NDArrayFactory::create<int>({0, 1, 2});
    sd::ops::reduce_variance op;
    auto result = op.evaluate({&x, &axes}, {}, {}, {true});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDev_test1) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    auto exp = NDArrayFactory::create<double>('c', {4}, {24.54022f, 26.96551f, 31.52072f, 27.49343f});

    sd::ops::reduce_stdev op;
    auto result = op.evaluate({&x}, {}, {0,1});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDev_test2) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    auto exp = NDArrayFactory::create<double>('c', {1,1,4}, {24.54022f, 26.96551f, 31.52072f, 27.49343f});

    sd::ops::reduce_stdev op;
    auto result = op.evaluate({&x}, {1.}, {0,1});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDev_test3) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    auto exp = NDArrayFactory::create<double>('c', {3}, {30.01562f, 31.14257f, 20.59581f});

    sd::ops::reduce_stdev op;
    auto result = op.evaluate({&x}, {}, {0,2});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDev_test4) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    auto exp = NDArrayFactory::create<double>('c', {1,3,1}, {30.01562f, 31.14257f, 20.59581f});

    sd::ops::reduce_stdev op;
    auto result = op.evaluate({&x}, {1.}, {0,2});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDev_test5) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    auto exp = NDArrayFactory::create<double>(28.08367f);

    sd::ops::reduce_stdev op;
    auto result = op.evaluate({&x}, {}, {});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDev_test6) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    auto exp = NDArrayFactory::create<double>(28.08367f);

    sd::ops::reduce_stdev op;
    auto result = op.evaluate({&x}, {}, {0,1,2});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDev_test7) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    auto exp = NDArrayFactory::create<double>('c', {1,1,1}, {28.08367f});

    sd::ops::reduce_stdev op;
    auto result = op.evaluate({&x}, {1.f}, {0,1,2});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDev_test8) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    auto exp = NDArrayFactory::create<double>('c', {4}, {26.88246f, 29.53924f, 34.52921f, 30.11755f});

    sd::ops::reduce_stdev op;
    auto result = op.evaluate({&x}, {0.f,1.f}, {0,1});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());
    // output->printBuffer("Reduced STDDEV");
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDev_test08) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4}, {27.f,34.f,5.f,4.f,54.f,6.f,65.f,8.f,37.f,45.f,8.f,67.f,96.f,10.f,65.f,41.f,33.f,85.f,92.f,24.f,25.f,55.f,49.f,76.});
    auto exp = NDArrayFactory::create<double>('c', {4}, {26.88246f, 29.53924f, 34.52921f, 30.11755f});
    auto axes = NDArrayFactory::create<int>({0,1});
    sd::ops::reduce_stdev op;
    auto result = op.evaluate({&x, &axes}, {}, {}, {false, true});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());
    // output->printBuffer("Reduced STDDEV08");
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceVarianceBP_test1) {

    auto x = NDArrayFactory::create<double>('c', {3,4});
    auto gradO1 = NDArrayFactory::create<double>('c', {1,1}, {0.5f});
    auto gradO2 = NDArrayFactory::create<double>(0.5f);
    auto exp12 = NDArrayFactory::create<double>('c', {3,4}, {-0.5f, -0.4090909f, -0.3181818f, -0.22727273f, -0.13636364f, -0.045454547f, 0.045454547f, 0.13636364f, 0.22727273f, 0.3181818f, 0.4090909f, 0.5f});
    auto exp34 = NDArrayFactory::create<double>('c', {3,4}, {-0.45833334f, -0.375f, -0.29166666f, -0.20833333f, -0.125f, -0.041666668f, 0.041666668f, 0.125f, 0.20833333f, 0.29166666f, 0.375f, 0.45833334f});

    x.linspace(1);

    sd::ops::reduce_variance_bp op;

    auto result = op.evaluate({&x, &gradO2}, {0,1}, {});
    ASSERT_EQ(Status::OK(), result.status());
    auto output = result.at(0);
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));


    result = op.evaluate({&x, &gradO1}, {1,1}, {});
    ASSERT_EQ(Status::OK(), result.status());
    output = result.at(0);
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));


    result = op.evaluate({&x, &gradO2}, {0,0}, {});
    ASSERT_EQ(Status::OK(), result.status());
    output = result.at(0);
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));


    result = op.evaluate({&x, &gradO1}, {1,0}, {});
    ASSERT_EQ(Status::OK(), result.status());
    output = result.at(0);
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceVarianceBP_test2) {

    auto x = NDArrayFactory::create<double>('c', {3,4});
    auto gradO1 = NDArrayFactory::create<double>('c', {1,4}, {1.f,2.f,3.f,4.f});
    auto gradO2 = NDArrayFactory::create<double>('c', {4}, {1.,2.,3.,4.});
    auto exp12 = NDArrayFactory::create<double>('c', {3,4}, {-2.666667f, -5.333333f, -8.000000f,  -10.666667f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 2.666667f, 5.333333f,  8.000000f, 10.666667f});
    auto exp34 = NDArrayFactory::create<double>('c', {3,4}, {-4.000000f, -8.000000f, -12.000000f, -16.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 4.000000f, 8.000000f, 12.000000f, 16.000000f});

    x.linspace(1);

    sd::ops::reduce_variance_bp op;

    auto result = op.evaluate({&x, &gradO2}, {0,0}, {0});
    ASSERT_EQ(Status::OK(), result.status());
    auto output = result.at(0);
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));


    result = op.evaluate({&x, &gradO1}, {1,0}, {0});
    ASSERT_EQ(Status::OK(), result.status());
    output = result.at(0);
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));


    result = op.evaluate({&x, &gradO2}, {0,1}, {0});
    ASSERT_EQ(Status::OK(), result.status());
    output = result.at(0);
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));


    result = op.evaluate({&x, &gradO1}, {1,1}, {0});
    ASSERT_EQ(Status::OK(), result.status());
    output = result.at(0);
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));


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

    sd::ops::reduce_variance_bp op;

    auto result = op.evaluate({&x, &gradO2, &axes}, {}, {}, {false, false});
    ASSERT_EQ(Status::OK(), result.status());
    auto output = result.at(0);
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));


    result = op.evaluate({&x, &gradO1, &axes}, {}, {}, {true, false});
    ASSERT_EQ(Status::OK(), result.status());
    output = result.at(0);
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));


    result = op.evaluate({&x, &gradO2, &axes}, {}, {}, {false, true});
    ASSERT_EQ(Status::OK(), result.status());
    output = result.at(0);
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));


    result = op.evaluate({&x, &gradO1, &axes}, {}, {}, {true, true});
    ASSERT_EQ(Status::OK(), result.status());
    output = result.at(0);
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));


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

    sd::ops::reduce_variance_bp op;

    auto result = op.evaluate({&x, &gradO2}, {0, 0}, {1});
    ASSERT_EQ(Status::OK(), result.status());
    auto output = result.at(0);
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));


    result = op.evaluate({&x, &gradO1}, {1, 0}, {1});
    ASSERT_EQ(Status::OK(), result.status());
    output = result.at(0);
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));


    result = op.evaluate({&x, &gradO2}, {0, 1}, {1});
    ASSERT_EQ(Status::OK(), result.status());
    output = result.at(0);
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));


    result = op.evaluate({&x, &gradO1}, {1, 1}, {1});
    ASSERT_EQ(Status::OK(), result.status());
    output = result.at(0);
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDevBP_test1) {

    auto x = NDArrayFactory::create<double>('c', {3,4});
    auto gradO1 = NDArrayFactory::create<double>('c', {1,1}, {0.5f});
    auto gradO2 = NDArrayFactory::create<double>(0.5f);
    auto exp12 = NDArrayFactory::create<double>('c', {3,4}, {-0.069337524f, -0.056730703f, -0.04412388f, -0.031517055f, -0.018910235f, -0.0063034114f, 0.0063034114f, 0.018910235f, 0.031517055f, 0.04412388f, 0.056730703f, 0.069337524f});
    auto exp34 = NDArrayFactory::create<double>('c', {3,4}, {-0.06638563f, -0.05431551f, -0.0422454f, -0.030175284f, -0.01810517f, -0.006035057f, 0.006035057f, 0.01810517f, 0.030175284f, 0.0422454f, 0.05431551f, 0.06638563f});

    x.linspace(1);

    sd::ops::reduce_stdev_bp op;

    auto result = op.evaluate({&x, &gradO2}, {0,1}, {});
    ASSERT_EQ(Status::OK(), result.status());
    auto output = result.at(0);
    // output->printIndexedBuffer();
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));


    result = op.evaluate({&x, &gradO1}, {1,1}, {});
    ASSERT_EQ(Status::OK(), result.status());
    output = result.at(0);
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));


    result = op.evaluate({&x, &gradO2}, {0,0}, {});
    ASSERT_EQ(Status::OK(), result.status());
    output = result.at(0);
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));


    result = op.evaluate({&x, &gradO1}, {1,0}, {});
    ASSERT_EQ(Status::OK(), result.status());
    output = result.at(0);
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDevBP_test2) {

    auto x = NDArrayFactory::create<double>('c', {3,4});
    auto gradO1 = NDArrayFactory::create<double>('c', {1,4}, {1.f,2.f,3.f,4.f});
    auto gradO2 = NDArrayFactory::create<double>('c', {4}, {1.f,2.f,3.f,4.f});
    auto exp12 = NDArrayFactory::create<double>('c', {3,4}, {-0.4082483f, -0.8164966f, -1.2247449f, -1.6329932f, 0.0, 0.0, 0.0, 0.0, 0.4082483f, 0.8164966f, 1.2247449f, 1.6329932f});
    auto exp34 = NDArrayFactory::create<double>('c', {3,4}, {-0.5f, -1.0f, -1.5f, -2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f});

    x.linspace(1);

    sd::ops::reduce_stdev_bp op;

    auto result = op.evaluate({&x, &gradO2}, {0,0}, {0});
    ASSERT_EQ(Status::OK(), result.status());
    auto output = result.at(0);
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));


    result = op.evaluate({&x, &gradO1}, {1,0}, {0});
    ASSERT_EQ(Status::OK(), result.status());
    output = result.at(0);
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));


    result = op.evaluate({&x, &gradO2}, {0,1}, {0});
    ASSERT_EQ(Status::OK(), result.status());
    output = result.at(0);
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));


    result = op.evaluate({&x, &gradO1}, {1,1}, {0});
    ASSERT_EQ(Status::OK(), result.status());
    output = result.at(0);
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));

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

    sd::ops::reduce_stdev_bp op;

    auto result = op.evaluate({&x, &gradO2, &axis}, {}, {}, {false, false});
    ASSERT_EQ(Status::OK(), result.status());
    auto output = result.at(0);
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));


    result = op.evaluate({&x, &gradO1, &axis}, {}, {}, {true, false});
    ASSERT_EQ(Status::OK(), result.status());
    output = result.at(0);
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));


    result = op.evaluate({&x, &gradO2, &axis}, {}, {}, {false, true});
    ASSERT_EQ(Status::OK(), result.status());
    output = result.at(0);
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));


    result = op.evaluate({&x, &gradO1, &axis}, {}, {}, {true, true});
    ASSERT_EQ(Status::OK(), result.status());
    output = result.at(0);
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDevBP_test3) {

    auto x = NDArrayFactory::create<double>('c', {3,4});
    auto gradO1 = NDArrayFactory::create<double>('c', {3,1}, {1.f,2.f,3.f});
    auto gradO2 = NDArrayFactory::create<double>('c', {3}, {1.f,2.f,3.f});
    auto exp12 = NDArrayFactory::create<double>('c', {3,4}, {-0.3354102f, -0.1118034f, 0.1118034f, 0.3354102f, -0.6708204f, -0.2236068f, 0.2236068f, 0.6708204f, -1.0062306f, -0.3354102f, 0.3354102f, 1.0062306f});
    auto exp34 = NDArrayFactory::create<double>('c', {3,4}, {-0.38729835f, -0.12909944f, 0.12909944f, 0.38729835f, -0.7745967f, -0.2581989f, 0.2581989f, 0.7745967f, -1.161895f, -0.38729835f, 0.38729835f, 1.161895f});

    x.linspace(1);

    sd::ops::reduce_stdev_bp op;

    auto result = op.evaluate({&x, &gradO2}, {0,0}, {1});
    ASSERT_EQ(Status::OK(), result.status());
    auto output = result.at(0);
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));


    result = op.evaluate({&x, &gradO1}, {1,0}, {1});
    ASSERT_EQ(Status::OK(), result.status());
    output = result.at(0);
    ASSERT_TRUE(exp12.isSameShape(output));
    ASSERT_TRUE(exp12.equalsTo(output));


    result = op.evaluate({&x, &gradO2}, {0,1}, {1});
    ASSERT_EQ(Status::OK(), result.status());
    output = result.at(0);
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));


    result = op.evaluate({&x, &gradO1}, {1,1}, {1});
    ASSERT_EQ(Status::OK(), result.status());
    output = result.at(0);
    ASSERT_TRUE(exp34.isSameShape(output));
    ASSERT_TRUE(exp34.equalsTo(output));


}


////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Sum_1) {

    auto input = NDArrayFactory::create<double>('c', {3, 5},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.});
    auto exp = NDArrayFactory::create<double>(120.f);
    //************************************//

    sd::ops::reduce_sum op;
    auto result = op.evaluate({&input}, {}, {});

    ASSERT_EQ(Status::OK(), result.status());
    auto z = result.at(0);
    //z->printIndexedBuffer("Result is ");
    ASSERT_TRUE(exp.equalsTo(z));

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Sum_2) {

    auto input = NDArrayFactory::create<double>('c', {3, 5},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.});
    auto exp = NDArrayFactory::create<double>({15.f, 40.f, 65.f});
    //************************************//

    sd::ops::reduce_sum op;
    auto result = op.evaluate({&input}, {}, {1});

    ASSERT_EQ(Status::OK(), result.status());
    auto z = result.at(0);
//    z->printIndexedBuffer("Result is ");
    ASSERT_TRUE(exp.equalsTo(z));

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Sum_03) {

    auto input = NDArrayFactory::create<double>('c', {3, 5},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.});
    auto exp = NDArrayFactory::create<double>({15.f, 40.f, 65.f});
    auto axis = NDArrayFactory::create<int>('c', {1}, {1});
    //************************************//

    sd::ops::reduce_sum op;
    auto result = op.evaluate({&input, &axis}, {}, {}, {false});

    ASSERT_EQ(Status::OK(), result.status());
    auto z = result.at(0);
    // z->printIndexedBuffer("Result is ");
    ASSERT_TRUE(exp.equalsTo(z));

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Prod_1) {

    auto input = NDArrayFactory::create<double>('c', {3, 5},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.});
    auto exp = NDArrayFactory::create<double>(1307674368000.f);
    //************************************//

    sd::ops::reduce_prod op;
    auto result = op.evaluate({&input}, {}, {});

    ASSERT_EQ(Status::OK(), result.status());
    auto z = result.at(0);
    //z->printIndexedBuffer("Result is ");
    ASSERT_TRUE(exp.equalsTo(z));

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Prod_2) {

    auto input = NDArrayFactory::create<double>('c', {3, 5},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.});
    auto exp = NDArrayFactory::create<double>({120.f, 30240.f, 360360.f});
    //************************************//

    sd::ops::reduce_prod op;
    auto result = op.evaluate({&input}, {}, {1});

    ASSERT_EQ(Status::OK(), result.status());
    auto z = result.at(0);
//    z->printIndexedBuffer("Result is ");
    ASSERT_TRUE(exp.equalsTo(z));

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Sum_01) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>('c', {4}, {66.f, 72.f, 78.f, 84.f});
    x.linspace(1);

    sd::ops::reduce_sum op;
    auto result = op.evaluate({&x}, {}, {0,1});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Sum_02) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>('c', {1,1,4}, {66.f, 72.f, 78.f, 84.f});
    x.linspace(1);

    sd::ops::reduce_sum op;
    auto result = op.evaluate({&x}, {1.}, {0, 1});
    auto output = result.at(0);
   // output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Sum_3) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>('c', {3}, {68.f, 100.f, 132.f});
    x.linspace(1);

    sd::ops::reduce_sum op;
    auto result = op.evaluate({&x}, {}, {0, 2});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Sum_4) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>('c', {1,3,1}, {68.f, 100.f, 132.f});
    x.linspace(1);

    sd::ops::reduce_sum op;
    auto result = op.evaluate({&x}, {1.}, {0, 2});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Sum_5) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>(300.f);
    x.linspace(1);

    sd::ops::reduce_sum op;
    auto result = op.evaluate({&x}, {}, {});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Sum_6) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>(300.f);
    x.linspace(1);

    sd::ops::reduce_sum op;
    auto result = op.evaluate({&x}, {}, {0,1,2});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Sum_7) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>('c', {1,1,1}, {300.f});
    x.linspace(1);
//    x.printIndexedBuffer("Input with shape (2, 3, 4) is");
    sd::ops::reduce_sum op;
    auto result = op.evaluate({&x}, {1.}, {0,1,2});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Prod_01) {

    auto x = NDArrayFactory::create<double>('c', {2,3,2});
    auto exp = NDArrayFactory::create<double>('c', {2}, {10395.f, 46080.f});
    x.linspace(1);

    sd::ops::reduce_prod op;
    auto result = op.evaluate({&x}, {}, {0,1});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Prod_02) {

    auto x = NDArrayFactory::create<double>('c', {2,3,2});
    auto exp = NDArrayFactory::create<double>('c', {1,1,2}, {10395.f, 46080.f});
    x.linspace(1);

    sd::ops::reduce_prod op;
    auto result = op.evaluate({&x}, {1.}, {0, 1});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Prod_3) {

    auto x = NDArrayFactory::create<double>('c', {2,3,2});
    auto exp = NDArrayFactory::create<double>('c', {3}, {112.f, 1080.f, 3960.f});
    x.linspace(1);

    sd::ops::reduce_prod op;
    auto result = op.evaluate({&x}, {}, {0, 2});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Prod_4) {

    auto x = NDArrayFactory::create<double>('c', {2,3,2});
    auto exp = NDArrayFactory::create<double>('c', {1,3,1}, {112.f, 1080.f, 3960.f});
    x.linspace(1);

    sd::ops::reduce_prod op;
    auto result = op.evaluate({&x}, {1.}, {0, 2});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Prod_04) {

    auto x = NDArrayFactory::create<double>('c', {2,3,2});
    auto exp = NDArrayFactory::create<double>('c', {1,3,1}, {112.f, 1080.f, 3960.f});
    auto axes = NDArrayFactory::create<int>({0, 2});
    x.linspace(1);

    sd::ops::reduce_prod op;
    auto result = op.evaluate({&x, &axes}, {}, {}, {true});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Prod_5) {

    auto x = NDArrayFactory::create<double>('c', {2,3,2});
    auto exp = NDArrayFactory::create<double>(479001600.f);
    x.linspace(1);

    sd::ops::reduce_prod op;
    auto result = op.evaluate({&x}, {}, {});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Prod_6) {

    auto x = NDArrayFactory::create<double>('c', {2,3,2});
    auto exp = NDArrayFactory::create<double>(479001600.f);
    x.linspace(1);

    sd::ops::reduce_prod op;
    auto result = op.evaluate({&x}, {}, {0,1,2});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Prod_7) {

    auto x = NDArrayFactory::create<double>('c', {2,3,2});
    auto exp = NDArrayFactory::create<double>('c', {1, 1, 1}, {479001600.f});
    x.linspace(1);
//    x.printIndexedBuffer("Input with shape (2, 3, 4) is");
    sd::ops::reduce_prod op;
    auto result = op.evaluate({&x}, {1.}, {0,1,2});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Min_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {4}, {1.f, 2.f, 3.f, 4.f});
    x.linspace(1);

    sd::ops::reduce_min op;
    auto result = op.evaluate({&x}, {}, {0, 1});
    auto output = result.at(0);
    // output->printIndexedBuffer("Result is");
    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Min_2) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1,1,4}, {1.f, 2.f, 3.f, 4.f});
    x.linspace(1);

    sd::ops::reduce_min op;
    auto result = op.evaluate({&x}, {1.}, {0, 1});
    auto output = result.at(0);
    // output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Min_3) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {3}, {1.f, 5.f, 9.f});
    x.linspace(1);

    sd::ops::reduce_min op;
    auto result = op.evaluate({&x}, {}, {0, 2});
    auto output = result.at(0);
    // output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Min_4) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1,3,1}, {1.f, 5.f, 9.f});
    x.linspace(1);

    sd::ops::reduce_min op;
    auto result = op.evaluate({&x}, {1.}, {0, 2});
    auto output = result.at(0);
    // output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Min_04) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1,3,1}, {1.f, 5.f, 9.f});
    auto axes = NDArrayFactory::create<int>({0, 2});
    x.linspace(1);

    sd::ops::reduce_min op;
    auto result = op.evaluate({&x, &axes}, {}, {}, {true});
    auto output = result.at(0);
    // output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Min_5) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(1.f);
    x.linspace(1);

    sd::ops::reduce_min op;
    auto result = op.evaluate({&x}, {}, {});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Min_6) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(1.f);
    x.linspace(1);

    sd::ops::reduce_min op;
    auto result = op.evaluate({&x}, {}, {0,1,2});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Min_7) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1, 1, 1}, {1.f});
    x.linspace(1);
    // x.printIndexedBuffer("Input with shape (2, 3, 4) is");
    sd::ops::reduce_min op;
    auto result = op.evaluate({&x}, {1.}, {0,1,2});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Max_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {4}, {21.f, 22.f, 23.f, 24.f});
    x.linspace(1);

    sd::ops::reduce_max op;
    auto result = op.evaluate({&x}, {}, {0,1});
    auto output = result.at(0);
    // output->printIndexedBuffer("Result is");
    // output->printShapeInfo("Output shape");
    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Max_2) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>('c', {1,1,4}, {21.f, 22.f, 23.f, 24.f});
    x.linspace(1);

    sd::ops::reduce_max op;
    auto result = op.evaluate({&x}, {1.}, {0, 1});
    auto output = result.at(0);
    // output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Max_3) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {3}, {16.f, 20.f, 24.f});
    x.linspace(1);

    sd::ops::reduce_max op;
    auto result = op.evaluate({&x}, {}, {0, 2});
    auto output = result.at(0);
    // output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Max_4) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1,3,1}, {16.f, 20.f, 24.f});
    x.linspace(1);

    sd::ops::reduce_max op;
    auto result = op.evaluate({&x}, {1.}, {0, 2});
    auto output = result.at(0);
    // output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Max_04) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1,3,1}, {16.f, 20.f, 24.f});
    auto axes = NDArrayFactory::create<int>({0, 2});
    x.linspace(1);

    sd::ops::reduce_max op;
    auto result = op.evaluate({&x, &axes}, {}, {}, {true});
    auto output = result.at(0);
    // output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Max_5) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(24.f);
    x.linspace(1);

    sd::ops::reduce_max op;
    auto result = op.evaluate({&x}, {}, {});
    auto output = result.at(0);
    // output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Max_6) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(24.f);
    x.linspace(1);

    sd::ops::reduce_max op;
    auto result = op.evaluate({&x}, {}, {0,1,2});
    auto output = result.at(0);
    // output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Max_7) {

	auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1, 1, 1}, {24.f});
    x.linspace(1);
//    x.printIndexedBuffer("Input with shape (2, 3, 4) is");
    sd::ops::reduce_max op;
    auto result = op.evaluate({&x}, {1.}, {0,1,2});
    auto output = result.at(0);
    // output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

TEST_F(DeclarableOpsTests8, Test_Reduce_Norm1_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {4}, {66.f, 72.f, 78.f, 84.f});
    x.linspace(1);

    sd::ops::reduce_norm1 op;
    auto result = op.evaluate({&x}, {}, {0,1});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Norm1_2) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>('c', {1,1,4}, {66.f, 72.f, 78.f, 84.f});
    x.linspace(1);

    sd::ops::reduce_norm1 op;
    auto result = op.evaluate({&x}, {1.}, {0, 1});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Norm1_3) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {3}, {68.f, 100.f, 132.f});
    x.linspace(1);

    sd::ops::reduce_norm1 op;
    auto result = op.evaluate({&x}, {}, {0, 2});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Norm1_4) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1,3,1}, {68.f, 100.f, 132.f});
    x.linspace(1);

    sd::ops::reduce_norm1 op;
    auto result = op.evaluate({&x}, {1.}, {0, 2});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Norm1_04) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1,3,1}, {68.f, 100.f, 132.f});
    auto axes = NDArrayFactory::create<int>({0, 2});
    x.linspace(1);

    sd::ops::reduce_norm1 op;
    auto result = op.evaluate({&x, &axes}, {}, {}, {true});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Norm1_5) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(300.f);
    x.linspace(1);

    sd::ops::reduce_norm1 op;
    auto result = op.evaluate({&x}, {}, {});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Norm1_6) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(300.f);
    x.linspace(1);

    sd::ops::reduce_norm1 op;
    auto result = op.evaluate({&x}, {}, {0,1,2});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Norm1_7) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1, 1, 1}, {300.f});
    x.linspace(1);
//    x.printIndexedBuffer("Input with shape (2, 3, 4) is");
    sd::ops::reduce_norm1 op;
    auto result = op.evaluate({&x}, {1.}, {0,1,2});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

TEST_F(DeclarableOpsTests8, Test_Reduce_Norm2_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {4}, {31.7175f, 33.823071f, 35.97221f, 38.15757f});
    x.linspace(1);

    sd::ops::reduce_norm2 op;
    auto result = op.evaluate({&x}, {}, {0,1});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Norm2_2) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>('c', {1,1,4}, {31.7175f, 33.823071f, 35.97221f, 38.15757f});
    x.linspace(1);

    sd::ops::reduce_norm2 op;
    auto result = op.evaluate({&x}, {1.}, {0, 1});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Norm2_3) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {3}, {29.597298f, 39.344631f, 49.759422f});
    x.linspace(1);

    sd::ops::reduce_norm2 op;
    auto result = op.evaluate({&x}, {}, {0, 2});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Norm2_4) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1,3,1}, {29.597298f, 39.344631f, 49.759422f});
    x.linspace(1);

    sd::ops::reduce_norm2 op;
    auto result = op.evaluate({&x}, {1.}, {0, 2});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Norm2_04) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1,3,1}, {29.597298f, 39.344631f, 49.759422f});
    auto axes = NDArrayFactory::create<int>({0,2});
    x.linspace(1);

    sd::ops::reduce_norm2 op;
    auto result = op.evaluate({&x, &axes}, {}, {}, {true});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Norm2_5) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(70.f);
    x.linspace(1);

    sd::ops::reduce_norm2 op;
    auto result = op.evaluate({&x}, {}, {});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Norm2_6) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(70.f);
    x.linspace(1);

    sd::ops::reduce_norm2 op;
    auto result = op.evaluate({&x}, {}, {0,1,2});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Norm2_7) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1, 1, 1}, {70.f});
    x.linspace(1);
//    x.printIndexedBuffer("Input with shape (2, 3, 4) is");
    sd::ops::reduce_norm2 op;
    auto result = op.evaluate({&x}, {1.}, {0,1,2});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_NormMax_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {4}, {21.f, 22.f, 23.f, 24.f});
    x.linspace(1);

    sd::ops::reduce_norm_max op;
    auto result = op.evaluate({&x}, {}, {0,1});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_NormMax_2) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1,1,4}, {21.f, 22.f, 23.f, 24.f});
    x.linspace(1);

    sd::ops::reduce_norm_max op;
    auto result = op.evaluate({&x}, {1.f}, {0,1});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_NormMax_3) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {3}, {16.f, 20.f, 24.f});
    x.linspace(1);

    sd::ops::reduce_norm_max op;
    auto result = op.evaluate({&x}, {}, {0,2});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_NormMax_4) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1, 3, 1}, {16.f, 20.f, 24.f});
    x.linspace(1);

    sd::ops::reduce_norm_max op;
    auto result = op.evaluate({&x}, {1.f}, {0,2});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_NormMax_04) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1, 3, 1}, {16.f, 20.f, 24.f});
    auto axes = NDArrayFactory::create<int>({0,2});
    x.linspace(1);

    sd::ops::reduce_norm_max op;
    auto result = op.evaluate({&x, &axes}, {}, {}, {true});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_NormMax_5) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(24.f);
    x.linspace(1);

    sd::ops::reduce_norm_max op;
    auto result = op.evaluate({&x}, {}, {});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_NormMax_6) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(24.f);
    x.linspace(1);

    sd::ops::reduce_norm_max op;
    auto result = op.evaluate({&x}, {}, {0, 1, 2});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_NormMax_7) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1, 1, 1}, {24.f});
    x.linspace(1);

    sd::ops::reduce_norm_max op;
    auto result = op.evaluate({&x}, {1.f}, {});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_SquaredNorm_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {4}, {1006.f, 1144.f, 1294.f, 1456.f});
    x.linspace(1);

    sd::ops::reduce_sqnorm op;
    auto result = op.evaluate({&x}, {}, {0,1});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_SquaredNorm_2) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1,1,4}, {1006.f, 1144.f, 1294.f, 1456.f});
    x.linspace(1);

    sd::ops::reduce_sqnorm op;
    auto result = op.evaluate({&x}, {1.f}, {0,1});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_SquaredNorm_3) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {3}, {876.f, 1548.f, 2476.f});
    x.linspace(1);

    sd::ops::reduce_sqnorm op;
    auto result = op.evaluate({&x}, {}, {0,2});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_SquaredNorm_4) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1, 3, 1}, {876.f, 1548.f, 2476.f});
    x.linspace(1);

    sd::ops::reduce_sqnorm op;
    auto result = op.evaluate({&x}, {1.f}, {0,2});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_SquaredNorm_04) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1, 3, 1}, {876.f, 1548.f, 2476.f});
    auto axes = NDArrayFactory::create<int>({0, 2});
    x.linspace(1);

    sd::ops::reduce_sqnorm op;
    auto result = op.evaluate({&x, &axes}, {}, {}, {true});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");
    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_SquaredNorm_5) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(4900.f);
    x.linspace(1);

    sd::ops::reduce_sqnorm op;
    auto result = op.evaluate({&x}, {}, {});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_SquaredNorm_6) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>(4900.f);
    x.linspace(1);

    sd::ops::reduce_sqnorm op;
    auto result = op.evaluate({&x}, {}, {0, 1, 2});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_SquaredNorm_7) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1, 1, 1}, {4900.f});
    x.linspace(1);

    sd::ops::reduce_sqnorm op;
    auto result = op.evaluate({&x}, {1.f}, {});
    auto output = result.at(0);
//    output->printIndexedBuffer("Result is");

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Sum_BP_1) {

    auto input = NDArrayFactory::create<double>('c', {3, 4},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.});
    auto eps = NDArrayFactory::create<double>(0.5f);
    auto exp = NDArrayFactory::create<double>('c', {3, 4}, {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,0.5f});
    //************************************//

    sd::ops::reduce_sum_bp op;
    auto result = op.evaluate({&input, &eps}, {}, {});

    ASSERT_EQ(Status::OK(), result.status());
    auto z = result.at(0);
//    z->printIndexedBuffer("Result is ");
//    z->printShapeInfo();
    ASSERT_TRUE(exp.equalsTo(z));

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Sum_BP_2) {

    auto input = NDArrayFactory::create<double>('c', {3, 4},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.});
    auto eps = NDArrayFactory::create<double>('c', {1, 1}, {0.5f});
    auto exp = NDArrayFactory::create<double>('c', {3, 4}, {0.5f, 0.5f, 0.5f, 0.5f,
                                     0.5f, 0.5f, 0.5f, 0.5f,
                                     0.5f, 0.5f, 0.5f,0.5f});
    //************************************//

    sd::ops::reduce_sum_bp op;
    auto result = op.evaluate({&input, &eps}, {1.f}, {});

    ASSERT_EQ(Status::OK(), result.status());
    auto z = result.at(0);
//  z->printIndexedBuffer("Result is ");
//  z->printShapeInfo();
    ASSERT_TRUE(exp.equalsTo(z));

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Sum_BP_3) {

    auto input = NDArrayFactory::create<double>('c', {3, 4},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.});
    auto eps = NDArrayFactory::create<double>('c', {4}, {1.f, 2.f, 3.f, 4.f});
    auto exp = NDArrayFactory::create<double>('c', {3, 4}, {1.f, 2.f, 3.f, 4.f,
                                     1.f, 2.f, 3.f, 4.f,
                                     1.f, 2.f, 3.f, 4.f});
    //************************************//

    sd::ops::reduce_sum_bp op;
    auto result = op.evaluate({&input, &eps}, {}, {0});

    ASSERT_EQ(Status::OK(), result.status());
    auto z = result.at(0);
//    z->printIndexedBuffer("Result is ");
//    z->printShapeInfo();
    ASSERT_TRUE(exp.equalsTo(z));

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Reduce_Sum_BP_4) {

    auto input = NDArrayFactory::create<double>('c', {3, 4},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.});
    auto eps = NDArrayFactory::create<double>('c', {1, 4}, {1.f, 2.f, 3.f, 4.f});
    auto exp = NDArrayFactory::create<double>('c', {3, 4}, {1.f, 2.f, 3.f, 4.f,
                                     1.f, 2.f, 3.f, 4.f,
                                     1.f, 2.f, 3.f, 4.f});
    //************************************//

    sd::ops::reduce_sum_bp op;
    auto result = op.evaluate({&input, &eps}, {1.f}, {0});

    ASSERT_EQ(Status::OK(), result.status());
    auto z = result.at(0);
//    z->printIndexedBuffer("Result is ");
//    z->printShapeInfo();
    ASSERT_TRUE(exp.equalsTo(z));

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

    sd::ops::reduce_sum_bp op;
    auto result = op.evaluate({&input, &eps, &axis}, {}, {}, {true});

    ASSERT_EQ(Status::OK(), result.status());
    auto z = result.at(0);
//    z->printIndexedBuffer("Result is ");
//    z->printShapeInfo();
    ASSERT_TRUE(exp.equalsTo(z));

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

    sd::ops::reduce_prod_bp op;
    auto result = op.evaluate({&input, &eps}, {}, {});

    ASSERT_EQ(Status::OK(), result.status());
    auto z = result.at(0);
//    z->printIndexedBuffer("Result is ");
//    z->printShapeInfo();
    ASSERT_TRUE(exp.equalsTo(z));

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceMean_test1) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>('c', {4}, {11.f, 12.f, 13.f, 14.f});
    x.linspace(1);


    sd::ops::reduce_mean op;
    auto result = op.evaluate({&x}, {}, {0,1});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceMean_test2) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>('c', {1,1,4}, {11.f, 12.f, 13.f, 14.f});
    x.linspace(1);


    sd::ops::reduce_mean op;
    auto result = op.evaluate({&x}, {1.}, {0,1});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceMean_test3) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>('c', {3}, {8.5f, 12.5f, 16.5f});
    x.linspace(1);


    sd::ops::reduce_mean op;
    auto result = op.evaluate({&x}, {}, {0,2});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceMean_test4) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>('c', {1,3,1}, {8.5f, 12.5f, 16.5f});
    x.linspace(1);


    sd::ops::reduce_mean op;
    auto result = op.evaluate({&x}, {1.f}, {0,2});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceMean_test5) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>(12.5f);
    x.linspace(1);


    sd::ops::reduce_mean op;
    auto result = op.evaluate({&x}, {}, {});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceMean_test6) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>(12.5f);
    x.linspace(1);

    sd::ops::reduce_mean op;
    auto result = op.evaluate({&x}, {}, {0,1,2});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceMean_test7) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>('c', {1,1,1}, {12.5f});
    x.linspace(1);

    sd::ops::reduce_mean op;
    auto result = op.evaluate({&x}, {1.}, {0,1,2});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceMean_test8) {

    auto x = NDArrayFactory::create<double>('c', {2,3,4});
    auto exp = NDArrayFactory::create<double>('c', {1,1,1}, {12.5f});
    auto axes = NDArrayFactory::create<int>({0, 1, 2});
    x.linspace(1);

    sd::ops::reduce_mean op;
    auto result = op.evaluate({&x, &axes}, {}, {}, {true});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceMeanBP_test1) {

    auto x = NDArrayFactory::create<double>('c', {3,4});
    auto gradO1 = NDArrayFactory::create<double>(0.5f);
    auto gradO2 = NDArrayFactory::create<double>('c', {1,1}, {0.5f});
    auto exp = NDArrayFactory::create<double>('c', {3,4}, {1./24, 1./24, 1./24, 1./24, 1./24, 1./24, 1./24, 1./24, 1./24, 1./24, 1./24, 1./24});

    x.linspace(1);

    sd::ops::reduce_mean_bp op;

    auto result = op.evaluate({&x, &gradO1}, {0}, {});
    ASSERT_EQ(Status::OK(), result.status());
    auto output = result.at(0);

    // output->printShapeInfo("o");

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


    result = op.evaluate({&x, &gradO2}, {1}, {});
    ASSERT_EQ(Status::OK(), result.status());
    output = result.at(0);
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

}


////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceMeanBP_test2) {

    auto x = NDArrayFactory::create<double>('c', {3,4});
    auto gradO1 = NDArrayFactory::create<double>('c', {4},  {1.f, 2.f, 3.f, 4.f});
    auto gradO2 = NDArrayFactory::create<double>('c', {1,4}, {1.f, 2.f, 3.f, 4.f});
    auto exp = NDArrayFactory::create<double>('c', {3,4}, {1.f/3.f, 2.f/3.f, 1.f, 4.f/3.f, 1.f/3.f, 2.f/3.f, 1.f, 4.f/3.f, 1.f/3.f, 2.f/3.f, 1.f, 4.f/3.f});

    x.linspace(1);

    sd::ops::reduce_mean_bp op;

    auto result = op.evaluate({&x, &gradO1}, {0}, {0});
    ASSERT_EQ(Status::OK(), result.status());
    auto output = result.at(0);
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


    result = op.evaluate({&x, &gradO2}, {1}, {0});
    ASSERT_EQ(Status::OK(), result.status());
    output = result.at(0);
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

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

    sd::ops::reduce_mean_bp op;

    auto result = op.evaluate({&x, &gradO1, &axis}, {}, {}, {false});
    ASSERT_EQ(Status::OK(), result.status());
    auto output = result.at(0);
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


    result = op.evaluate({&x, &gradO2, &axis}, {}, {}, {true});
    ASSERT_EQ(Status::OK(), result.status());
    output = result.at(0);
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceMeanBP_test3) {

    auto x = NDArrayFactory::create<double>('c', {3,4});
    auto gradO1 = NDArrayFactory::create<double>('c', {3}, {1.f, 2.f, 3.f});
    auto gradO2 = NDArrayFactory::create<double>('c', {3,1}, {1.f, 2.f, 3.f});
    auto exp = NDArrayFactory::create<double>('c', {3,4}, {0.25f, 0.25f, 0.25f, 0.25f, 0.5f, 0.5f, 0.5f, 0.5f, 0.75f, 0.75f, 0.75f, 0.75f});

    x.linspace(1);

    sd::ops::reduce_mean_bp op;

    auto result = op.evaluate({&x, &gradO1}, {0}, {1});
    ASSERT_EQ(Status::OK(), result.status());
    auto output = result.at(0);
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


    result = op.evaluate({&x, &gradO2}, {1}, {1});
    ASSERT_EQ(Status::OK(), result.status());
    output = result.at(0);
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDevBP_test4) {

    auto x = NDArrayFactory::create<double>('c', {3}, {2.f, 3.f, 4.f});
    auto gradO = NDArrayFactory::create<double>(0.5f);
    auto exp = NDArrayFactory::create<double>('c', {3}, {-0.25f, 0.f, 0.25f});

    sd::ops::reduce_stdev_bp op;

    auto result = op.evaluate({&x, &gradO}, {0,1}, {});
    ASSERT_EQ(Status::OK(), result.status());
    auto output = result.at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, softmax_cross_entropy_loss_with_logits_test1) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4},{0,1,1,0,0,0,1,0,1,0,1,1,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto expected = NDArrayFactory::create<double>('c', {2,3}, {2.78507, 1.34254, 4.12761, 2.88507, 2.78507, 2.88507});

    logits.linspace(0.1, 0.1);

    sd::ops::softmax_cross_entropy_loss_with_logits op;
    auto results = op.evaluate({&logits, &labels}, {}, {});

    ASSERT_EQ(Status::OK(), results.status());

    auto *output = results.at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));


}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, softmax_cross_entropy_loss_with_logits_test2) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4},{0,1,1,0,0,0,1,0,1,0,1,1,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto expected = NDArrayFactory::create<double>('c', {3,4}, {0.26328, 1.46328, 1.72656, 0.     , 0.26328, 0.     , 1.46328, 0.26328, 1.72656, 0.     , 1.72656, 1.46328});

    logits.linspace(0.1, 0.1);

    sd::ops::softmax_cross_entropy_loss_with_logits op;
    auto results = op.evaluate({&logits, &labels}, {}, {0});

    ASSERT_EQ(Status::OK(), results.status());

    auto *output = results.at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));


}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, softmax_cross_entropy_loss_with_logits_test3) {

    auto labels = NDArrayFactory::create<double>('c', {2,3,4},{0,1,1,0,0,0,1,0,1,0,1,1,1,0,1,0,1,0,0,1,1,0,1,0});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto expected = NDArrayFactory::create<double>('c', {2,4}, {0.75125, 1.55125, 3.45375, 0.75125, 3.45375, 0.     , 2.3025 , 1.15125});

    logits.linspace(0.1, 0.1);

    sd::ops::softmax_cross_entropy_loss_with_logits op;
    auto results = op.evaluate({&logits, &labels}, {}, {1});

    ASSERT_EQ(Status::OK(), results.status());

    auto *output = results.at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));


}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, softmax_cross_entropy_loss_with_logits_test4) {

    auto labels = NDArrayFactory::create<double>('c', {2,3},{0,1,1,0,0,1});
    auto logits = NDArrayFactory::create<double>('c', {2,3});
    auto expected = NDArrayFactory::create<double>('c', {2}, {2.10389, 1.00194});

    logits.linspace(0.1, 0.1);

    sd::ops::softmax_cross_entropy_loss_with_logits op;
    auto results = op.evaluate({&logits, &labels}, {}, {});

    ASSERT_EQ(Status::OK(), results.status());

    auto output = results.at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));


}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, softmax_cross_entropy_loss_with_logits_test5) {

    auto labels = NDArrayFactory::create<double>('c', {2,3},{0,1,1,0,0,1});
    auto logits = NDArrayFactory::create<double>('c', {2,3});
    auto expected = NDArrayFactory::create<double>('c', {3}, {0., 0.85436, 1.40871});

    logits.linspace(0.1, 0.1);

    sd::ops::softmax_cross_entropy_loss_with_logits op;
    auto results = op.evaluate({&logits, &labels}, {}, {0});

    ASSERT_EQ(Status::OK(), results.status());

    auto output = results.at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));


}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, softmax_cross_entropy_loss_with_logits_test6) {

    auto labels = NDArrayFactory::create<double>('c', {2,1}, {0,1});
    auto logits = NDArrayFactory::create<double>('c', {2,1});
    auto expected = NDArrayFactory::create<double>('c', {1}, {0.6444});

    logits.linspace(0.1, 0.1);

    sd::ops::softmax_cross_entropy_loss_with_logits op;
    auto results = op.evaluate({&logits, &labels}, {}, {0});

    ASSERT_EQ(Status::OK(), results.status());

    auto output = results.at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));


}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, softmax_cross_entropy_loss_with_logits_test7) {

    auto labels = NDArrayFactory::create<double>('c', {2,1}, {0,1});
    auto logits = NDArrayFactory::create<double>('c', {2,1});
    auto expected = NDArrayFactory::create<double>('c', {2}, {0., 0.});

    logits.linspace(0.1, 0.1);

    sd::ops::softmax_cross_entropy_loss_with_logits op;
    auto results = op.evaluate({&logits, &labels}, {}, {1});

    ASSERT_EQ(Status::OK(), results.status());

    auto output = results.at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));


}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, softmax_cross_entropy_loss_with_logits_test8) {

    auto labels = NDArrayFactory::create<double>('c', {2}, {0,1});
    auto logits = NDArrayFactory::create<double>('c', {2});
    auto expected = NDArrayFactory::create<double>(0.6444);

    logits.linspace(0.1, 0.1);

    sd::ops::softmax_cross_entropy_loss_with_logits op;
    auto results = op.evaluate({&logits, &labels}, {}, {});

    ASSERT_EQ(Status::OK(), results.status());

    auto *output = results.at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));


}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, softmax_cross_entropy_loss_with_logits_test9) {

    auto labels = NDArrayFactory::create<double>('c', {1}, {0.});
    auto logits = NDArrayFactory::create<double>('c', {1}, {0.2});
    auto expected = NDArrayFactory::create<double>(0.);

    sd::ops::softmax_cross_entropy_loss_with_logits op;
    auto results = op.evaluate({&logits, &labels}, {}, {});

    ASSERT_EQ(Status::OK(), results.status());

    auto output = results.at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));


}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, softmax_cross_entropy_loss_with_logits_test10) {

    auto labels = NDArrayFactory::create<double>('c', {1,2}, {0,1});
    auto logits = NDArrayFactory::create<double>('c', {1,2});
    auto expected = NDArrayFactory::create<double>('c', {2}, {0., 0.});

    logits.linspace(0.1, 0.1);

    sd::ops::softmax_cross_entropy_loss_with_logits op;
    auto results = op.evaluate({&logits, &labels}, {}, {0});

    ASSERT_EQ(Status::OK(), results.status());

    auto output = results.at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceMeanBP_test4) {

    auto x = NDArrayFactory::create<double>('c', {3, 4}, {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12. });
    auto gradO1 = NDArrayFactory::create<double>('c', {4}, {1., 2., 3., 4.});
    auto gradO2 = NDArrayFactory::create<double>('c', {1, 4}, {1., 2., 3., 4.});
    auto exp = NDArrayFactory::create<double>('c', {3,4}, {0.333333, 0.666667, 1.000000, 1.333333, 0.333333, 0.666667, 1.000000, 1.333333, 0.333333, 0.666667, 1.000000, 1.333333});

    sd::ops::reduce_mean_bp op;

    auto result = op.evaluate({&x, &gradO1}, {0}, {0});
    ASSERT_EQ(Status::OK(), result.status());
    auto output = result.at(0);
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


    result = op.evaluate({&x, &gradO2}, {1}, {0});
    ASSERT_EQ(Status::OK(), result.status());
    output = result.at(0);
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

}


////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceMeanBP_test5) {

    auto x = NDArrayFactory::create<double>('c', {3, 4}, {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12. });
    auto gradO1 = NDArrayFactory::create<double>('c', {3}, {1., 2., 3.});
    auto gradO2 = NDArrayFactory::create<double>('c', {3, 1}, {1., 2., 3.});
    auto exp = NDArrayFactory::create<double>('c', {3,4}, {0.2500,0.2500,0.2500,0.2500, 0.5000,0.5000,0.5000,0.5000, 0.7500,0.7500,0.7500,0.7500});

    sd::ops::reduce_mean_bp op;

    auto result = op.evaluate({&x, &gradO1}, {0}, {1});
    ASSERT_EQ(Status::OK(), result.status());
    auto output = result.at(0);
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


    result = op.evaluate({&x, &gradO2}, {1}, {1});
    ASSERT_EQ(Status::OK(), result.status());
    output = result.at(0);
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

}


////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, reduceStDevBP_test5) {

    auto x = NDArrayFactory::create<double>('c', {3, 4}, {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12. });
    auto gradO1 = NDArrayFactory::create<double>('c', {4}, {1., 2., 3., 4.});
    auto gradO2 = NDArrayFactory::create<double>('c', {1, 4}, {1., 2., 3., 4.});
    auto exp = NDArrayFactory::create<double>('c', {3,4}, {-0.408248, -0.816497, -1.224745, -1.632993, 0.000000, 0.000000, 0.000000, 0.000000, 0.408248, 0.816497, 1.224745, 1.632993});

    sd::ops::reduce_stdev_bp op;

    auto result = op.evaluate({&x, &gradO1}, {0}, {0});
    ASSERT_EQ(Status::OK(), result.status());
    auto output = result.at(0);
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));


    result = op.evaluate({&x, &gradO2}, {1}, {0});
    ASSERT_EQ(Status::OK(), result.status());
    output = result.at(0);
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, zeros_as_test1) {

    auto x = NDArrayFactory::create<double>(10.f);
    auto y = NDArrayFactory::create<double>(100.f);
    auto exp = NDArrayFactory::create<double>(0.f);

    sd::ops::zeros_as op;

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

    sd::ops::zeros_as op;

    auto result = op.evaluate({&x}, {}, {});
    ASSERT_EQ(Status::OK(), result.status());
    auto y = result.at(0);

    ASSERT_TRUE(y->isSameShape(exp));
    ASSERT_TRUE(y->equalsTo(exp));

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, ones_as_test1) {

    auto x = NDArrayFactory::create<double>(10.);
    auto y = NDArrayFactory::create<double>(100.);
    auto exp = NDArrayFactory::create<double>(1.);

    sd::ops::ones_as op;

    Nd4jStatus status = op.execute({&x}, {&y});
    ASSERT_EQ(Status::OK(), status);

    ASSERT_TRUE(y.isSameShape(exp));
    ASSERT_TRUE(y.equalsTo(exp));

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, ones_as_test2) {

    auto x = NDArrayFactory::create<double>(10.);
    //auto y = NDArrayFactory::create<double>(100.);
    auto exp = NDArrayFactory::create<double>(1.);

    sd::ops::ones_as op;

    auto results = op.evaluate({&x});
    ASSERT_EQ(Status::OK(), results.status());
    auto y = results.at(0);
    ASSERT_TRUE(y->isSameShape(exp));
    ASSERT_TRUE(y->equalsTo(exp));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, ones_as_test3) {

    auto x = NDArrayFactory::create<double>(10.);
    //auto y = NDArrayFactory::create<double>(100.);
    auto exp = NDArrayFactory::create<int>(1.);

    sd::ops::ones_as op;

    auto results = op.evaluate({&x}, {}, {}, {}, {sd::DataType::INT32});
    ASSERT_EQ(Status::OK(), results.status());
    auto y = results.at(0);

    ASSERT_TRUE(y->isSameShape(exp));
    ASSERT_TRUE(y->equalsTo(exp));


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
    data.applyTransform(transform::Square, squared);
    auto ssSquared = squared.reduceAlongDimension(reduce::Sum, {0});
//    ssSquared->printBuffer("Sum squared");
//    squared.printBuffer("Squared");
    sd::ops::normalize_moments op;
    auto results = op.evaluate({&counts, &means, &ssSquared}, {0.0}, {0});
    means /= counts;
//    sd::ops::normalize_moments op;
//    auto results = op.evaluate({&counts, means, deviance}, {0.0}, {});

    ASSERT_EQ(Status::OK(), results.status());
    ASSERT_EQ(results.size(), 2);

    auto outputMeans = results.at(0);
    auto outputDeviance = results.at(1);

//    outputMeans->printIndexedBuffer("Means");
//    outputDeviance->printIndexedBuffer("Variance");
//    deviance.printIndexedBuffer("Expected");
//    means->printIndexedBuffer("Expected means");
    ASSERT_TRUE(means.isSameShape(outputMeans));
    ASSERT_TRUE(means.equalsTo(outputMeans));
    ASSERT_TRUE(deviance.isSameShape(outputDeviance));
    ASSERT_TRUE(deviance.equalsTo(outputDeviance));
    //delete deviance;
//    ASSERT_TRUE(expMeans.isSameShape(outputMeans));
//    ASSERT_TRUE(expMeans.equalsTo(outputMeans));
//    ASSERT_TRUE(expMeans.isSameShape(outputDeviance));
//    ASSERT_TRUE(expDeviance.equalsTo(outputDeviance));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Moments_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto expMeans = NDArrayFactory::create<double>('c', {4}, {11.f, 12.f, 13.f, 14.f});
    auto expVariance = NDArrayFactory::create<double>('c', {4}, {46.666668f, 46.666668f, 46.66666f, 46.666668f});
    x.linspace(1);

    sd::ops::moments op;
    auto result = op.evaluate({&x}, {}, {0, 1});

    ASSERT_EQ(Status::OK(), result.status());

    auto outputMeans = result.at(0);
    auto outputVariance = result.at(1);

//    outputMeans->printIndexedBuffer("Means");
//    outputVariance->printIndexedBuffer("Variance");
//    outputMeans->printShapeInfo("Result shape");


//    ASSERT_TRUE(exp.isSameShape(output));
//    ASSERT_TRUE(exp.equalsTo(output));
    ASSERT_TRUE(expMeans.isSameShape(outputMeans));
    ASSERT_TRUE(expMeans.equalsTo(outputMeans));
    ASSERT_TRUE(expVariance.isSameShape(outputVariance));
    ASSERT_TRUE(expVariance.equalsTo(outputVariance));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Moments_2) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto expMeans = NDArrayFactory::create<double>('c', {1,1,4}, {11.f, 12.f, 13.f, 14.f});
    auto expVariance = NDArrayFactory::create<double>('c', {1,1,4}, {46.666668f, 46.666668f, 46.66666f, 46.666668f});
    x.linspace(1);

    sd::ops::moments op;
    auto result = op.evaluate({&x}, {1.}, {0, 1});
    ASSERT_EQ(Status::OK(), result.status());

    auto outputMeans = result.at(0);
    auto outputVariance = result.at(1);

//    outputMeans->printIndexedBuffer("Means");
//    outputVariance->printIndexedBuffer("Variance");
//    outputMeans->printShapeInfo("Result shape");

//    ASSERT_TRUE(exp.isSameShape(output));
//    ASSERT_TRUE(exp.equalsTo(output));
    ASSERT_TRUE(expMeans.isSameShape(outputMeans));
    ASSERT_TRUE(expMeans.equalsTo(outputMeans));
    ASSERT_TRUE(expVariance.isSameShape(outputVariance));
    ASSERT_TRUE(expVariance.equalsTo(outputVariance));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Moments_3) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto expMeans = NDArrayFactory::create<double>('c', {3}, {8.5f, 12.5f, 16.5f});
    auto expVariance = NDArrayFactory::create<double>('c', {3}, {37.25f, 37.25f, 37.25f});
    x.linspace(1);

    sd::ops::moments op;
    auto result = op.evaluate({&x}, {}, {0, 2});
    ASSERT_EQ(Status::OK(), result.status());

    auto outputMeans = result.at(0);
    auto outputVariance = result.at(1);

//    outputMeans->printIndexedBuffer("Means");
//    outputVariance->printIndexedBuffer("Variance");
//    outputMeans->printShapeInfo("Result shape");

//    ASSERT_TRUE(exp.isSameShape(output));
//    ASSERT_TRUE(exp.equalsTo(output));
    ASSERT_TRUE(expMeans.isSameShape(outputMeans));
    ASSERT_TRUE(expMeans.equalsTo(outputMeans));
    ASSERT_TRUE(expVariance.isSameShape(outputVariance));
    ASSERT_TRUE(expVariance.equalsTo(outputVariance));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Moments_4) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto expMeans = NDArrayFactory::create<double>('c', {1,3,1}, {8.5f, 12.5f, 16.5f});
    auto expVariance = NDArrayFactory::create<double>('c', {1,3,1}, {37.25f, 37.25f, 37.25f});
    x.linspace(1);

    sd::ops::moments op;
    auto result = op.evaluate({&x}, {1.}, {0, 2});
    ASSERT_EQ(Status::OK(), result.status());

    auto outputMeans = result.at(0);
    auto outputVariance = result.at(1);

//    outputMeans->printIndexedBuffer("Means");
//    outputVariance->printIndexedBuffer("Variance");
//    outputMeans->printShapeInfo("Result shape");

//    ASSERT_TRUE(exp.isSameShape(output));
//    ASSERT_TRUE(exp.equalsTo(output));
    ASSERT_TRUE(expMeans.isSameShape(outputMeans));
    ASSERT_TRUE(expMeans.equalsTo(outputMeans));
    ASSERT_TRUE(expVariance.isSameShape(outputVariance));
    ASSERT_TRUE(expVariance.equalsTo(outputVariance));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Moments_6) {
    auto expMeans = NDArrayFactory::create<double>(12.5f);
    auto expVariance = NDArrayFactory::create<double>(47.916668f);

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    x.linspace(1);

    sd::ops::moments op;
    auto result = op.evaluate({&x}, {}, {0,1,2});
    ASSERT_EQ(Status::OK(), result.status());

    auto outputMeans = result.at(0);
    auto outputVariance = result.at(1);

//    outputMeans->printIndexedBuffer("Means");
//    outputVariance->printIndexedBuffer("Variance");
//    outputMeans->printShapeInfo("Result shape");

    ASSERT_TRUE(expMeans.isSameShape(outputMeans));
    ASSERT_TRUE(expMeans.equalsTo(outputMeans));
    ASSERT_TRUE(expVariance.isSameShape(outputVariance));
    ASSERT_TRUE(expVariance.equalsTo(outputVariance));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests8, Test_Moments_7) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});

    auto expMeans = NDArrayFactory::create<double>('c', {1,1,1}, {12.5f});
    auto expVariance = NDArrayFactory::create<double>('c', {1,1,1}, {47.916668f});

    x.linspace(1);
    // x.printIndexedBuffer("Input with shape (2, 3, 4) is");
    sd::ops::moments op;
    auto result = op.evaluate({&x}, {1.}, {0,1,2});
    ASSERT_EQ(Status::OK(), result.status());

    auto outputMeans = result.at(0);
    auto outputVariance = result.at(1);

//    outputMeans->printIndexedBuffer("Means");
//    outputVariance->printIndexedBuffer("Variance");
//    outputMeans->printShapeInfo("Result shape");
    ASSERT_TRUE(expMeans.isSameShape(outputMeans));
    ASSERT_TRUE(expMeans.equalsTo(outputMeans));
    ASSERT_TRUE(expVariance.isSameShape(outputVariance));
    ASSERT_TRUE(expVariance.equalsTo(outputVariance));


}

////////////////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedDeclarableOpsTests8, LrnTest_01) {

    auto x = NDArrayFactory::create<TypeParam>('c', {1, 1, 2, 5}, { 1.f, 2.f, 3.f, 4.f, 5.f,
                                                                    6.f, 7.f, 8.f, 9.f, 10.f}
    );

   auto exp = NDArrayFactory::create<TypeParam>('c', {1, 1, 2, 5}, {0.2581989f, 0.3592106f, 0.40089184f, 0.53935987f, 0.70014f, 0.4898979f, 0.46056613f, 0.43971977f, 0.5240003f, 0.6375767f}//            0.72760683, 0.4850712,   0.5848977, 0.67488194,
//            0.7581754,  0.58321184, 0.86747235, 0.4048204}
   );

    sd::ops::lrn op;
    auto results = op.evaluate({&x}, {1.0, 1.0, 0.5}, {2});
    auto out = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
    //ASSERT_TRUE(exp.isSameShape(out));
    //out->printBuffer("LRN out");
    //exp.printBuffer("LRN exp");
    ASSERT_TRUE(exp.equalsTo(out));


}

////////////////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedDeclarableOpsTests8, LrnTest_02) {

    auto x = NDArrayFactory::create<TypeParam>('c', {1, 1, 1, 6}, { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f});

    auto exp = NDArrayFactory::create<TypeParam>('c', {1, 1, 1, 6}, {
        0.2581989f, 0.3592106f, 0.40089184f, 0.4193139f, 0.5360563f, 0.67936623f}
    );

    sd::ops::lrn op;
    auto results = op.evaluate({&x}, {1.0, 1.0, 0.5}, {2});
    auto out = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
    //ASSERT_TRUE(exp.isSameShape(out));
    //out->printIndexedBuffer("LRN out");
//    exp.printIndexedBuffer("LRN exp");
    ASSERT_TRUE(exp.equalsTo(out));


}

////////////////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedDeclarableOpsTests8, LrnTest_03) {

    auto x = NDArrayFactory::create<TypeParam>('c', {1, 1, 1, 10}, { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f});
    auto exp = NDArrayFactory::create<TypeParam>('c', {1, 1, 1, 10}, {0.10425719f, 0.16843036f, 0.2095291f, 0.23652494f, 0.25449327f, 0.3053919f, 0.35675305f, 0.4098524f, 0.46662825f, 0.52999896f});

    sd::ops::lrn op;
    auto results = op.evaluate({&x}, {1.0, 1.0, 0.5}, {5});
    auto out = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
    ASSERT_TRUE(exp.isSameShape(out));
    // out->printIndexedBuffer("LRN out");
//    exp.printIndexedBuffer("LRN exp");
    ASSERT_TRUE(exp.equalsTo(out));


}

////////////////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedDeclarableOpsTests8, LrnTest_1) {

    auto x = NDArrayFactory::create<TypeParam>('c', {2, 2, 2, 2}, { 5.5f, 0.f, 0.3f, 5.5f,
                                            8.6f, 0.f, 0.f, 0.4f,
                                            1.5f, 1.f, 1.3f, 1.5f,
                                            2.6f, 2.f, 3.f, 1.4f}
    );

    auto exp = NDArrayFactory::create<TypeParam>('c', {2, 2, 2, 2}, {
                                            0.98386997f,        0.f,  0.05358852f,  0.9824562f,
                                            0.99330735f,        0.f,          0.f, 0.37139067f,
                                            0.72760683f, 0.4850712f,   0.5848977f, 0.67488194f,
                                            0.7581754f,  0.58321184f, 0.86747235f, 0.4048204f}
    );

    sd::ops::lrn op;
    auto results = op.evaluate({&x}, {1.0, 1.0, 0.5}, {2});
    auto out = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
    ASSERT_TRUE(exp.isSameShape(out));
//    out->printIndexedBuffer("LRN out");
//    exp.printIndexedBuffer("LRN exp");
    ASSERT_TRUE(exp.equalsTo(out));


}

////////////////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedDeclarableOpsTests8, LrnTest_2) {

    auto x = NDArrayFactory::create<TypeParam>('c', {3, 3, 5, 5});
    x.linspace(1);

    auto exp = NDArrayFactory::create<TypeParam>('c', {3, 3, 5, 5}, {
    0.2581989f, 0.3592106f,  0.40089184f,  0.53935987f,  0.70014f,
    0.4898979f, 0.46056613f,  0.43971977f,  0.5240002f,  0.6375767f,
    0.5274096f, 0.47771242f,  0.4443308f,  0.5163977f,  0.61701745f,
    0.5424508f, 0.48452914f,  0.44570294f,  0.5123918f,  0.6068971f,
    0.5505386f, 0.4881662f,  0.4462865f,  0.5099462f,  0.60088515f,

    0.5555859f,  0.49042296f,  0.44658744f,  0.5083028f,  0.59690416f,
    0.55903524f,  0.4919585f,  0.44676256f,  0.5071239f,  0.59407425f,
    0.5615412f,  0.49307042f,  0.44687328f,  0.50623745f,  0.5919596f,
    0.56344414f,  0.49391258f,  0.4469477f,  0.5055468f,  0.59031945f,
    0.56493837f,  0.49457246f,  0.4470002f,  0.5049936f,  0.5890103f,

    0.56614274f,  0.49510333f,  0.44703856f,  0.50454074f,  0.5879411f,
    0.567134f,  0.49553978f,  0.4470674f,  0.504163f,  0.5870515f,
    0.5679643f,  0.4959048f,  0.44708967f,  0.5038433f,  0.5862998f,
    0.56866974f,  0.4962146f,  0.44710726f,  0.5035692f,  0.58565617f,
    0.56927663f,  0.49648085f,  0.4471213f,  0.5033315f,  0.5850988f,


    0.56980413f,  0.49671215f,  0.44713274f,  0.50312346f,  0.58461165f,
    0.57026696f,  0.49691492f,  0.4471422f,  0.50293994f,  0.58418214f,
    0.5706764f,  0.49709415f,  0.44715008f,  0.5027767f,  0.5838005f,
    0.571041f,  0.4972537f,  0.44715673f,  0.50263065f,  0.58345926f,
    0.57136786f,  0.49739665f,  0.44716236f,  0.5024992f,  0.58315235f,

    0.5716625f,  0.49752548f,  0.4471672f,  0.5023803f,   0.5828747f,
    0.5719295f,  0.49764213f,  0.44717142f,  0.5022721f,   0.5826225f,
    0.57217246f,  0.49774826f,  0.44717506f,  0.5021734f,   0.58239233f,
    0.5723947f,  0.4978453f,  0.44717824f,  0.5020829f,   0.58218133f,
    0.57259864f,  0.49793428f,  0.44718108f,  0.5019997f,   0.5819874f,

    0.5727864f,  0.49801624f,  0.44718358f,  0.5019227f,   0.5818083f,
    0.57296f,  0.49809194f,  0.44718578f,  0.5018515f,   0.5816426f,
    0.5731208f,  0.49816203f,  0.44718775f,  0.5017854f,   0.58148885f,
    0.57327026f,  0.49822718f,  0.4471895f,  0.5017239f,   0.5813457f,
    0.57340944f,  0.49828786f,  0.44719115f,  0.5016664f,   0.581212f,


    0.57353944f,  0.4983446f,  0.44719255f,  0.50161266f,  0.58108705f,
    0.5736612f,  0.49839762f,  0.4471939f,  0.50156236f,  0.5809699f,
    0.5737754f,  0.4984474f,  0.44719502f,  0.501515f,  0.58085984f,
    0.5738828f,  0.49849418f,  0.4471962f,  0.50147045f,  0.5807564f,
    0.5739839f,  0.49853817f,  0.44719717f,  0.5014284f,  0.5806588f,

    0.5740793f,  0.49857965f,  0.4471981f,  0.5013887f,  0.5805666f,
    0.5741694f,  0.49861887f,  0.44719887f,  0.50135124f,  0.58047944f,
    0.57425463f,  0.49865603f,  0.44719967f,  0.5013157f,  0.5803969f,
    0.5743354f,  0.4986912f,  0.44720036f,  0.5012819f,  0.5803186f,
    0.57441217f,  0.49872455f,  0.44720104f,  0.5012499f,  0.58024424f,

    0.57448506f,  0.4987563f,  0.44720164f,  0.5012194f,  0.58017343f,
    0.57455444f,  0.4987865f,  0.4472022f,  0.5011904f,  0.5801061f,
    0.57462054f,  0.49881527f,  0.44720277f,  0.5011627f,  0.5800419f,
    0.57468355f,  0.49884263f,  0.44720328f,  0.50113624f,  0.5799805f,
    0.57474375f,  0.49886885f,  0.44720373f,  0.50111103f,  0.5799219f }
    );
//
    sd::ops::lrn op;
    auto results = op.evaluate({&x}, {1.0, 1.0, 0.5}, {2});
    auto out = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
//    ASSERT_TRUE(exp.isSameShape(out));
//    out->printIndexedBuffer("LRN out");
//    exp.printIndexedBuffer("LRN exp");
    ASSERT_TRUE(exp.equalsTo(out));


}

////////////////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedDeclarableOpsTests8, LrnTest_3) {

    auto x = NDArrayFactory::create<TypeParam>('c', {3, 3, 5, 5});
    x.linspace(1);

    auto exp = NDArrayFactory::create<TypeParam>('c', {3, 3, 5, 5}, {
            0.2581989f, 0.3592106f,  0.40089184f,  0.53935987f,  0.70014f,
            0.4898979f, 0.46056613f,  0.43971977f,  0.5240002f,  0.6375767f,
            0.5274096f, 0.47771242f,  0.4443308f,  0.5163977f,  0.61701745f,
            0.5424508f, 0.48452914f,  0.44570294f,  0.5123918f,  0.6068971f,
            0.5505386f, 0.4881662f,  0.4462865f,  0.5099462f,  0.60088515f,

            0.5555859f,  0.49042296f,  0.44658744f,  0.5083028f,  0.59690416f,
            0.55903524f,  0.4919585f,  0.44676256f,  0.5071239f,  0.59407425f,
            0.5615412f,  0.49307042f,  0.44687328f,  0.50623745f,  0.5919596f,
            0.56344414f,  0.49391258f,  0.4469477f,  0.5055468f,  0.59031945f,
            0.56493837f,  0.49457246f,  0.4470002f,  0.5049936f,  0.5890103f,

            0.56614274f,  0.49510333f,  0.44703856f,  0.50454074f,  0.5879411f,
            0.567134f,  0.49553978f,  0.4470674f,  0.504163f,  0.5870515f,
            0.5679643f,  0.4959048f,  0.44708967f,  0.5038433f,  0.5862998f,
            0.56866974f,  0.4962146f,  0.44710726f,  0.5035692f,  0.58565617f,
            0.56927663f,  0.49648085f,  0.4471213f,  0.5033315f,  0.5850988f,


            0.56980413f,  0.49671215f,  0.44713274f,  0.50312346f,  0.58461165f,
            0.57026696f,  0.49691492f,  0.4471422f,  0.50293994f,  0.58418214f,
            0.5706764f,  0.49709415f,  0.44715008f,  0.5027767f,  0.5838005f,
            0.571041f,  0.4972537f,  0.44715673f,  0.50263065f,  0.58345926f,
            0.57136786f,  0.49739665f,  0.44716236f,  0.5024992f,  0.58315235f,

            0.5716625f,  0.49752548f,  0.4471672f,  0.5023803f,   0.5828747f,
            0.5719295f,  0.49764213f,  0.44717142f,  0.5022721f,   0.5826225f,
            0.57217246f,  0.49774826f,  0.44717506f,  0.5021734f,   0.58239233f,
            0.5723947f,  0.4978453f,  0.44717824f,  0.5020829f,   0.58218133f,
            0.57259864f,  0.49793428f,  0.44718108f,  0.5019997f,   0.5819874f,

            0.5727864f,  0.49801624f,  0.44718358f,  0.5019227f,   0.5818083f,
            0.57296f,  0.49809194f,  0.44718578f,  0.5018515f,   0.5816426f,
            0.5731208f,  0.49816203f,  0.44718775f,  0.5017854f,   0.58148885f,
            0.57327026f,  0.49822718f,  0.4471895f,  0.5017239f,   0.5813457f,
            0.57340944f,  0.49828786f,  0.44719115f,  0.5016664f,   0.581212f,


            0.57353944f,  0.4983446f,  0.44719255f,  0.50161266f,  0.58108705f,
            0.5736612f,  0.49839762f,  0.4471939f,  0.50156236f,  0.5809699f,
            0.5737754f,  0.4984474f,  0.44719502f,  0.501515f,  0.58085984f,
            0.5738828f,  0.49849418f,  0.4471962f,  0.50147045f,  0.5807564f,
            0.5739839f,  0.49853817f,  0.44719717f,  0.5014284f,  0.5806588f,

            0.5740793f,  0.49857965f,  0.4471981f,  0.5013887f,  0.5805666f,
            0.5741694f,  0.49861887f,  0.44719887f,  0.50135124f,  0.58047944f,
            0.57425463f,  0.49865603f,  0.44719967f,  0.5013157f,  0.5803969f,
            0.5743354f,  0.4986912f,  0.44720036f,  0.5012819f,  0.5803186f,
            0.57441217f,  0.49872455f,  0.44720104f,  0.5012499f,  0.58024424f,

            0.57448506f,  0.4987563f,  0.44720164f,  0.5012194f,  0.58017343f,
            0.57455444f,  0.4987865f,  0.4472022f,  0.5011904f,  0.5801061f,
            0.57462054f,  0.49881527f,  0.44720277f,  0.5011627f,  0.5800419f,
            0.57468355f,  0.49884263f,  0.44720328f,  0.50113624f,  0.5799805f,
            0.57474375f,  0.49886885f,  0.44720373f,  0.50111103f,  0.5799219f }
    );
//
    sd::ops::lrn op;
    auto results = op.evaluate({&x}, {1.0, 1.0, 0.5}, {2});
    auto out = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
//    ASSERT_TRUE(exp.isSameShape(out));
//    out->printIndexedBuffer("LRN out");
//    exp.printIndexedBuffer("LRN exp");
    ASSERT_TRUE(exp.equalsTo(out));


}

////////////////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedDeclarableOpsTests8, LrnTest_4) {

    // auto x = NDArrayFactory::create<TypeParam>('c', {8, 32, 64, 64});
    auto x = NDArrayFactory::create<TypeParam>('c', {2, 8, 16, 16});
    x.linspace(1);

    sd::ops::lrn op;
    auto results = op.evaluate({&x}, {1.0, 1.0, 0.5}, {2});
    auto out = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
//    ASSERT_TRUE(exp.isSameShape(out));
//    out->printIndexedBuffer("LRN out");
//    exp.printIndexedBuffer("LRN exp");
//    ASSERT_TRUE(exp.equalsTo(out));


}

////////////////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedDeclarableOpsTests8, LrnTest_4_119) {
    int iterations = 1000;
    // auto x = NDArrayFactory::create<TypeParam>('c', {8, 32, 64, 64});
    // auto z = NDArrayFactory::create<TypeParam>('c', {8, 32, 64, 64});
    auto x = NDArrayFactory::create<TypeParam>('c', {2, 8, 16, 16});
    auto z = NDArrayFactory::create<TypeParam>('c', {2, 8, 16, 16});
    x.linspace(1);

    sd::ops::lrn op;

    op.execute({&x}, {&z}, {1.0, 1.0, 0.5}, {2});

    auto timeStart = std::chrono::system_clock::now();

    for (int e = 0; e < iterations; e++)
        op.execute({&x}, {&z}, {1.0, 1.0, 0.5}, {2});

    auto timeEnd = std::chrono::system_clock::now();
    auto spanTime = std::chrono::duration_cast<std::chrono::microseconds> ((timeEnd - timeStart) / iterations).count();
    auto ttlTime = std::chrono::duration_cast<std::chrono::milliseconds> ((timeEnd - timeStart)).count();


//    ASSERT_TRUE(exp.isSameShape(out));
//    ASSERT_TRUE(exp.equalsTo(out));
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
    sd::ops::lrn_bp op;
    auto results = op.evaluate({&x, &eps}, {1.0, 1.0, 0.5}, {5});
    auto out = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
//    ASSERT_TRUE(exp.isSameShape(out));
    //out->printBuffer("LRN BP out");
    //exp.printBuffer("LRN BP exp");
    //ASSERT_TRUE(exp.equalsTo(out));


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
    sd::ops::lrn opFF;
    sd::ops::lrn_bp opBP;

    const OpArgsHolder argsHolderFF({&x},         {1., 1., 0.5}, {5});
    const OpArgsHolder argsHolderBP({&x, &eps}, {1., 1., 0.5}, {5});

    bool gradOK = true; //GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);
    //auto  results = op.execute({&x, &eps}, {1.0, 1.0, 0.5}, {5}, {}, false, sd::DataType::DOUBLE);
    //auto out = results.at(0);

    //ASSERT_EQ(Status::OK(), results.status());
    ASSERT_TRUE(gradOK);
    //out->printBuffer("LRN BP out");
    //exp.printBuffer("LRN BP exp");
    //ASSERT_TRUE(exp.equalsTo(out));

    //
}

////////////////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedDeclarableOpsTests8, LrnTest_BP_1) {

    auto x = NDArrayFactory::create<TypeParam>( 'c', {3, 3, 5, 5});
    x.linspace(1);
    auto eps = NDArrayFactory::create<TypeParam>('c', {3,3,5,5});
    eps.linspace(1);
//
auto exp = NDArrayFactory::create<TypeParam>('c', {3,3,5,5}, {
        0.238337f, 0.309664f, 0.334077f, 0.376534f, 0.342926f, 0.370734f, 0.362017f, 0.354182f, 0.379140f, 0.376275f, 0.380027f, 0.368347f, 0.356401f, 0.378316f, 0.381315f, 0.382465f, 0.370592f, 0.357055f, 0.377670f, 0.382950f, 0.383445f, 0.371718f, 0.357332f, 0.377217f, 0.383677f, 0.383933f, 0.372391f, 0.357475f, 0.376891f, 0.384062f, 0.384212f, 0.372837f, 0.357557f, 0.376646f, 0.384290f, 0.384385f, 0.373153f, 0.357610f, 0.376457f, 0.384436f, 0.384500f, 0.373389f, 0.357645f, 0.376306f, 0.384536f, 0.384581f, 0.373572f, 0.357670f, 0.376184f, 0.384606f, 0.384639f, 0.373718f, 0.357688f, 0.376082f, 0.384658f, 0.384683f, 0.373837f, 0.357702f, 0.375996f, 0.384698f, 0.384717f, 0.373935f, 0.357712f, 0.375923f, 0.384728f, 0.384743f, 0.374019f, 0.357721f, 0.375860f, 0.384752f, 0.384764f, 0.374090f, 0.357727f, 0.375804f, 0.384771f, 0.384781f, 0.374152f, 0.357733f, 0.375756f, 0.384787f, 0.384795f, 0.374205f, 0.357737f, 0.375713f, 0.384800f, 0.384807f, 0.374253f, 0.357741f, 0.375674f, 0.384811f, 0.384817f, 0.374295f, 0.357744f, 0.375640f, 0.384820f, 0.384825f, 0.374333f, 0.357747f, 0.375609f, 0.384828f, 0.384832f, 0.374366f, 0.357749f, 0.375581f, 0.384835f, 0.384839f, 0.374397f, 0.357751f, 0.375555f, 0.384841f, 0.384844f, 0.374425f, 0.357753f, 0.375531f, 0.384846f, 0.384849f, 0.374450f, 0.357754f, 0.375510f, 0.384850f, 0.384853f, 0.374473f, 0.357756f, 0.375490f, 0.384854f, 0.384856f, 0.374494f, 0.357757f, 0.375471f, 0.384858f, 0.384860f, 0.374514f, 0.357758f, 0.375454f, 0.384861f, 0.384863f, 0.374532f, 0.357759f, 0.375438f, 0.384864f, 0.384865f, 0.374549f, 0.357760f, 0.375423f, 0.384866f, 0.384868f, 0.374565f, 0.357760f, 0.375410f, 0.384868f, 0.384870f, 0.374579f, 0.357761f, 0.375397f, 0.384870f, 0.384872f, 0.374593f, 0.357762f, 0.375384f, 0.384872f, 0.384873f, 0.374606f, 0.357762f, 0.375373f, 0.384874f, 0.384875f, 0.374618f, 0.357763f, 0.375362f, 0.384875f, 0.384876f, 0.374629f, 0.357763f, 0.375352f, 0.384877f, 0.384878f, 0.374640f, 0.357764f, 0.375342f, 0.384878f, 0.384879f, 0.374650f, 0.357764f, 0.375333f, 0.384879f, 0.384880f, 0.374660f, 0.357764f, 0.375325f, 0.384880f, 0.384881f, 0.374669f, 0.357765f, 0.375316f, 0.384881f, 0.384882f, 0.374677f, 0.357765f, 0.375309f, 0.384882f, 0.384883f, 0.374685f, 0.357765f, 0.375301f, 0.384883f, 0.384884f, 0.374693f, 0.357765f, 0.375294f, 0.384884f, 0.384884f, 0.374700f, 0.357766f, 0.375287f, 0.384885f, 0.384885f, 0.374707f, 0.357766f, 0.375281f, 0.384885f, 0.384886f, 0.374714f, 0.357766f, 0.375275f, 0.384886f}
    );
///
    sd::ops::lrn_bp op;
    auto  results = op.evaluate({&x, &eps}, {1.0, 1.0, 0.5}, {2}, {}, {}, false);
    auto out = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
//    ASSERT_TRUE(exp.isSameShape(out));
    // out->printBuffer("LRN BP out");
    // exp.printBuffer("LRN BP exp");
    //ASSERT_TRUE(exp.equalsTo(out));


}

////////////////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedDeclarableOpsTests8, LrnTest_BP_2) {

    auto x = NDArrayFactory::create<TypeParam>( 'c', {3, 3, 5, 5});
    x.linspace(1);

    auto eps = NDArrayFactory::create<TypeParam>('c', {3, 3, 5, 5}, {            0.2581989f, 0.3592106f, 0.40089184f, 0.53935987f, 0.70014f,
                                                                                 0.4898979f, 0.46056613f, 0.43971977f, 0.5240002f, 0.6375767f,
                                                                                 0.5274096f, 0.47771242f, 0.4443308f, 0.5163977f, 0.61701745f,
                                                                                 0.5424508f, 0.48452914f, 0.44570294f, 0.5123918f, 0.6068971f,
                                                                                 0.5505386f, 0.4881662f, 0.4462865f, 0.5099462f, 0.60088515f,

                                                                                 0.5555859f, 0.49042296f, 0.44658744f, 0.5083028f, 0.59690416f,
                                                                                 0.55903524f, 0.4919585f, 0.44676256f, 0.5071239f, 0.59407425f,
                                                                                 0.5615412f, 0.49307042f, 0.44687328f, 0.50623745f, 0.5919596f,
                                                                                 0.56344414f, 0.49391258f, 0.4469477f, 0.5055468f, 0.59031945f,
                                                                                 0.56493837f, 0.49457246f, 0.4470002f, 0.5049936f, 0.5890103f,

                                                                                 0.56614274f, 0.49510333f, 0.44703856f, 0.50454074f, 0.5879411f,
                                                                                 0.567134f, 0.49553978f, 0.4470674f, 0.504163f, 0.5870515f,
                                                                                 0.5679643f, 0.4959048f, 0.44708967f, 0.5038433f, 0.5862998f,
                                                                                 0.56866974f, 0.4962146f, 0.44710726f, 0.5035692f, 0.58565617f,
                                                                                 0.56927663f, 0.49648085f, 0.4471213f, 0.5033315f, 0.5850988f,


                                                                                 0.56980413f, 0.49671215f, 0.44713274f, 0.50312346f, 0.58461165f,
                                                                                 0.57026696f, 0.49691492f, 0.4471422f, 0.50293994f, 0.58418214f,
                                                                                 0.5706764f, 0.49709415f, 0.44715008f, 0.5027767f, 0.5838005f,
                                                                                 0.571041f, 0.4972537f, 0.44715673f, 0.50263065f, 0.58345926f,
                                                                                 0.57136786f, 0.49739665f, 0.44716236f, 0.5024992f, 0.58315235f,

                                                                                 0.5716625f, 0.49752548f, 0.4471672f, 0.5023803f, 0.5828747f,
                                                                                 0.5719295f, 0.49764213f, 0.44717142f, 0.5022721f, 0.5826225f,
                                                                                 0.57217246f, 0.49774826f, 0.44717506f, 0.5021734f, 0.58239233f,
                                                                                 0.5723947f, 0.4978453f, 0.44717824f, 0.5020829f, 0.58218133f,
                                                                                 0.57259864f, 0.49793428f, 0.44718108f, 0.5019997f, 0.5819874f,

                                                                                 0.5727864f, 0.49801624f, 0.44718358f, 0.5019227f, 0.5818083f,
                                                                                 0.57296f, 0.49809194f, 0.44718578f, 0.5018515f, 0.5816426f,
                                                                                 0.5731208f, 0.49816203f, 0.44718775f, 0.5017854f, 0.58148885f,
                                                                                 0.57327026f, 0.49822718f, 0.4471895f, 0.5017239f, 0.5813457f,
                                                                                 0.57340944f, 0.49828786f, 0.44719115f, 0.5016664f, 0.581212f,


                                                                                 0.57353944f, 0.4983446f, 0.44719255f, 0.50161266f, 0.58108705f,
                                                                                 0.5736612f, 0.49839762f, 0.4471939f, 0.50156236f, 0.5809699f,
                                                                                 0.5737754f, 0.4984474f, 0.44719502f, 0.501515f, 0.58085984f,
                                                                                 0.5738828f, 0.49849418f, 0.4471962f, 0.50147045f, 0.5807564f,
                                                                                 0.5739839f, 0.49853817f, 0.44719717f, 0.5014284f, 0.5806588f,

                                                                                 0.5740793f, 0.49857965f, 0.4471981f, 0.5013887f, 0.5805666f,
                                                                                 0.5741694f, 0.49861887f, 0.44719887f, 0.50135124f, 0.58047944f,
                                                                                 0.57425463f, 0.49865603f, 0.44719967f, 0.5013157f, 0.5803969f,
                                                                                 0.5743354f, 0.4986912f, 0.44720036f, 0.5012819f, 0.5803186f,
                                                                                 0.57441217f, 0.49872455f, 0.44720104f, 0.5012499f, 0.58024424f,

                                                                                 0.57448506f, 0.4987563f, 0.44720164f, 0.5012194f, 0.58017343f,
                                                                                 0.57455444f, 0.4987865f, 0.4472022f, 0.5011904f, 0.5801061f,
                                                                                 0.57462054f, 0.49881527f, 0.44720277f, 0.5011627f, 0.5800419f,
                                                                                 0.57468355f, 0.49884263f, 0.44720328f, 0.50113624f, 0.5799805f,
                                                                                 0.57474375f, 0.49886885f, 0.44720373f, 0.50111103f, 0.5799219f });
//
    auto exp = NDArrayFactory::create<TypeParam>('c', {3,3,5,5}, {
            0.061538f, 0.055617f, 0.044643f, 0.050772f, 0.048019f, 0.030270f, 0.023819f, 0.019468f, 0.022074f, 0.023990f, 0.018221f, 0.014664f, 0.012182f, 0.013954f, 0.015685f, 0.012967f, 0.010563f, 0.008841f, 0.010185f, 0.011621f, 0.010052f, 0.008248f, 0.006934f, 0.008015f, 0.009222f, 0.008204f, 0.006764f, 0.005702f, 0.006606f, 0.007642f, 0.006929f, 0.005732f, 0.004841f, 0.005618f, 0.006523f, 0.005996f, 0.004973f, 0.004205f, 0.004887f, 0.005689f, 0.005284f, 0.004391f, 0.003717f, 0.004324f, 0.005044f, 0.004723f, 0.003931f, 0.003331f, 0.003877f, 0.004531f, 0.004270f, 0.003558f, 0.003017f, 0.003514f, 0.004112f, 0.003896f, 0.003250f, 0.002757f, 0.003213f, 0.003764f, 0.003582f, 0.002991f, 0.002539f, 0.002959f, 0.003470f, 0.003315f, 0.002770f, 0.002352f, 0.002743f, 0.003219f, 0.003085f, 0.002580f, 0.002191f, 0.002556f, 0.003002f, 0.002885f, 0.002414f, 0.002051f, 0.002393f, 0.002812f, 0.002709f, 0.002268f, 0.001927f, 0.002250f, 0.002645f, 0.002553f, 0.002138f, 0.001818f, 0.002122f, 0.002496f, 0.002415f, 0.002023f, 0.001720f, 0.002009f, 0.002363f, 0.002290f, 0.001920f, 0.001632f, 0.001906f, 0.002244f, 0.002178f, 0.001826f, 0.001553f, 0.001814f, 0.002136f, 0.002076f, 0.001741f, 0.001481f, 0.001731f, 0.002038f, 0.001984f, 0.001664f, 0.001416f, 0.001654f, 0.001949f, 0.001899f, 0.001593f, 0.001356f, 0.001584f, 0.001867f, 0.001821f, 0.001528f, 0.001301f, 0.001520f, 0.001792f, 0.001750f, 0.001469f, 0.001250f, 0.001461f, 0.001722f, 0.001683f, 0.001413f, 0.001203f, 0.001406f, 0.001658f, 0.001622f, 0.001362f, 0.001159f, 0.001355f, 0.001599f, 0.001565f, 0.001314f, 0.001119f, 0.001308f, 0.001543f, 0.001512f, 0.001270f, 0.001081f, 0.001264f, 0.001491f, 0.001462f, 0.001228f, 0.001046f, 0.001223f, 0.001443f, 0.001415f, 0.001189f, 0.001013f, 0.001184f, 0.001397f, 0.001372f, 0.001153f, 0.000982f, 0.001148f, 0.001355f, 0.001331f, 0.001118f, 0.000952f, 0.001114f, 0.001315f, 0.001292f, 0.001086f, 0.000925f, 0.001082f, 0.001277f, 0.001255f, 0.001055f, 0.000899f, 0.001051f, 0.001241f, 0.001221f, 0.001026f, 0.000874f, 0.001023f, 0.001208f, 0.001188f, 0.000999f, 0.000851f, 0.000996f, 0.001176f, 0.001157f, 0.000973f, 0.000829f, 0.000970f, 0.001145f, 0.001128f, 0.000949f, 0.000808f, 0.000945f, 0.001117f, 0.001100f, 0.000925f, 0.000788f, 0.000922f, 0.001089f, 0.001073f, 0.000903f, 0.000769f, 0.000900f, 0.001063f, 0.001048f, 0.000882f, 0.000751f, 0.000879f, 0.001038f, 0.001024f, 0.000861f, 0.000734f, 0.000859f, 0.001015f, 0.001001f, 0.000842f, 0.000717f, 0.000840f, 0.000992f}
        //    0.009859f, 0.013075f, 0.013874f, 0.017893f, 0.022344f, 0.014551f, 0.012859f, 0.011511f, 0.013311f, 0.015834f, 0.012025f, 0.010047f, 0.008601f, 0.009920f, 0.011885f, 0.009505f, 0.007636f, 0.006299f, 0.007413f, 0.009095f, 0.007446f, 0.005743f, 0.004540f, 0.005533f, 0.007033f, 0.005821f, 0.004282f, 0.003209f, 0.004123f, 0.005491f, 0.004577f, 0.003198f, 0.002247f, 0.003097f, 0.004355f, 0.003652f, 0.002412f, 0.001565f, 0.002357f, 0.003517f, 0.002965f, 0.001844f, 0.001084f, 0.001821f, 0.002893f, 0.002451f, 0.001430f, 0.000741f, 0.001428f, 0.002422f, -0.111434f, -0.105946f, -0.100351f, -0.091868f, -0.083323f, -0.078775f, -0.076222f, -0.073291f, -0.067635f, -0.061692f, -0.058943f, -0.057832f, -0.056263f, -0.052198f, -0.047768f, -0.046002f, -0.045655f, -0.044839f, -0.041748f, -0.038271f, -0.037084f, -0.037161f, -0.036786f, -0.034331f, -0.031495f, 0.000077f, -0.000673f, -0.001181f, -0.000667f, 0.000079f, -0.000089f, -0.000802f, -0.001285f, -0.000793f, -0.000079f, -0.000228f, -0.000908f, -0.001368f, -0.000896f, -0.000212f, -0.000345f, -0.000996f, -0.001434f, -0.000981f, -0.000325f, -0.000444f, -0.001067f, -0.001487f, -0.001051f, -0.000421f, 0.000697f, 0.000188f, -0.000152f, 0.000210f, 0.000731f, 0.000650f, 0.000165f, -0.000161f, 0.000185f, 0.000683f, 0.000610f, 0.000145f, -0.000168f, 0.000164f, 0.000641f, 0.000574f, 0.000128f, -0.000172f, 0.000146f, 0.000604f, 0.000542f, 0.000113f, -0.000175f, 0.000131f, 0.000571f, -0.009490f, -0.010070f, -0.010409f, -0.009734f, -0.008834f, -0.008785f, -0.009351f, -0.009687f, -0.009054f, -0.008207f, -0.008167f, -0.008718f, -0.009050f, -0.008455f, -0.007654f, -0.007622f, -0.008159f, -0.008485f, -0.007924f, -0.007164f, -0.007138f, -0.007661f, -0.007981f, -0.007450f, -0.006728f, -0.000901f, -0.001327f, -0.001614f, -0.001310f, -0.000869f, -0.000913f, -0.001328f, -0.001607f, -0.001310f, -0.000882f, -0.000922f, -0.001326f, -0.001598f, -0.001309f, -0.000892f, -0.000930f, -0.001323f, -0.001588f, -0.001306f, -0.000900f, -0.000936f, -0.001319f, -0.001577f, -0.001302f, -0.000906f, 0.000339f, 0.000038f, -0.000164f, 0.000048f, 0.000355f, 0.000328f, 0.000035f, -0.000162f, 0.000045f, 0.000343f, 0.000318f, 0.000033f, -0.000159f, 0.000041f, 0.000332f, 0.000308f, 0.000030f, -0.000157f, 0.000039f, 0.000322f, 0.000299f, 0.000028f, -0.000155f, 0.000036f, 0.000312f, -0.004085f, -0.004479f, -0.004733f, -0.004396f, -0.003925f, -0.003925f, -0.004309f, -0.004558f, -0.004232f, -0.003775f, -0.003776f, -0.004151f, -0.004395f, -0.004079f, -0.003636f, -0.003637f, -0.004004f, -0.004242f, -0.003936f, -0.003505f, -0.003507f, -0.003866f, -0.004100f, -0.003802f, -0.003383f}
    );

    sd::ops::lrn_bp op;
    auto  results = op.evaluate({&x, &eps}, {1.0, 1.0, 0.5}, {2}, {}, {}, false);
    auto out = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
    ASSERT_TRUE(exp.isSameShape(out));
    //out->printBuffer("LRN BP out");
//    exp.printIndexedBuffer("LRN exp");
   // ASSERT_TRUE(exp.equalsTo(out));


}


