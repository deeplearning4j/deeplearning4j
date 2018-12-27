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
#include <array/ExtraArguments.h>
#include <array>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace nd4j;

class CudaExtraArgumentsTests : public testing::Test {
public:

    CudaExtraArgumentsTests() {
        printf("\n");
        fflush(stdout);
    }
};

TEST_F(CudaExtraArgumentsTests, Basic_Test_1) {
    ExtraArguments args({1.0, 2.0, 3.0});

    float ef[] = {1.f, 2.f, 3.f};
    double ed[] = {1., 2., 3.};

    auto ptrFloat = reinterpret_cast<float *>(args.argumentsAsT<float>());
    auto ptrDouble = reinterpret_cast<double *>(args.argumentsAsT<double>());
    ASSERT_TRUE(ptrFloat != nullptr);
    ASSERT_TRUE(ptrDouble != nullptr);

    auto tmpFloat = new float[3];
    auto tmpDouble = new double[3];

    cudaMemcpy(tmpFloat, ptrFloat, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(tmpDouble, ptrDouble, 3 * sizeof(double), cudaMemcpyDeviceToHost);

    for (int e = 0; e < 3; e++) {
        ASSERT_NEAR(ef[e], tmpFloat[e], 1e-5f);
    }

    for (int e = 0; e < 3; e++) {
        ASSERT_NEAR(ed[e], tmpDouble[e], 1e-5);
    }

    delete[] tmpFloat;
    delete[] tmpDouble;
}


TEST_F(CudaExtraArgumentsTests, Basic_Test_2) {
    ExtraArguments args;

    auto ptrInt = args.argumentsAsT<int>();
    ASSERT_TRUE(ptrInt == nullptr);
}

