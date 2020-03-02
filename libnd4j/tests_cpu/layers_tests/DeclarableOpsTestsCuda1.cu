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
// @author raver119@gmail.com
//

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <array/NDArray.h>
#include <ops/ops.h>
#include <helpers/GradCheck.h>
#include <chrono>


using namespace sd;


class DeclarableOpsTestsCuda1 : public testing::Test {
public:

    DeclarableOpsTestsCuda1() {
        printf("\n");
        fflush(stdout);
    }
};


TEST_F(DeclarableOpsTestsCuda1, Test_CHOOSE_SCALAR_LARGE) {
    double inputData[150] = {
            0,  0.51,  0.68,  0.69,  0.86,  0.91,  0.96,  0.97,  0.97,  1.03,  1.13,  1.16,  1.16,  1.17,  1.19,  1.25,  1.25,  1.26,  1.27,  1.28,  1.29,  1.29,  1.29,  1.30,  1.31,  1.32,  1.33,  1.33,  1.35,  1.35,  1.36,  1.37,  1.38,  1.40,  1.41,  1.42,  1.43,  1.44,  1.44,  1.45,  1.45,  1.47,  1.47,  1.51,  1.51,  1.51,  1.52,  1.53,  1.56,  1.57,  1.58,  1.59,  1.61,  1.62,  1.63,  1.63,  1.64,  1.64,  1.66,  1.66,  1.67,  1.67,  1.70,  1.70,  1.70,  1.72,  1.72,  1.72,  1.72,  1.73,  1.74,  1.74,  1.76,  1.76,  1.77,  1.77,  1.80,  1.80,  1.81,  1.82,  1.83,  1.83,  1.84,  1.84,  1.84,  1.85,  1.85,  1.85,  1.86,  1.86,  1.87,  1.88,  1.89,  1.89,  1.89,  1.89,  1.89,  1.91,  1.91,  1.91,  1.92,  1.94,  1.95,  1.97,  1.98,  1.98,  1.98,  1.98,  1.98,  1.99,  2,  2,  2.01,  2.01,  2.02,  2.03,  2.03,  2.03,  2.04,  2.04,  2.05,  2.06,  2.07,  2.08,  2.08,  2.08,  2.08,  2.09,  2.09,  2.10,  2.10,  2.11,  2.11,  2.11,  2.12,  2.12,  2.13,  2.13,  2.14,  2.14,  2.14,  2.14,  2.15,  2.15,  2.16,  2.16,  2.16,  2.16,  2.16,  2.17
    };

    auto precursor = NDArrayFactory::create<double>(inputData,'c',{1,149});
    NDArray x(nullptr, precursor.specialBuffer(), precursor.shapeInfo());

    sd::ops::choose op;
    //greater than test
    auto result = op.evaluate({&x}, {0.0},{3});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(1);

    ASSERT_EQ(148,z->e<double>(0));
    //ASSERT_TRUE(exp.isSameShape(z));

    delete result;
}

/*
TEST_F(DeclarableOpsTestsCuda1, Test_Reverse_TAD_1) {
    auto x = NDArrayFactory::create<float>('c', {1, 3, 608, 608});
    auto z = x.like();
    x.linspace(1.0f);

    sd::ops::reverse op;
    auto timeStart = std::chrono::system_clock::now();
    auto status = op.execute({&x}, {&z}, {}, {1}, {});
    auto timeEnd = std::chrono::system_clock::now();
    auto outerTime = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();
    nd4j_printf("exec time: %lld us\n", outerTime);
    ASSERT_EQ(Status::OK(), status);
}
*/