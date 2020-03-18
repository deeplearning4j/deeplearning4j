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
// Created by raver on 11/26/2018.
//

#include "testlayers.h"
#include <helpers/CudaLaunchHelper.h>

using namespace sd;
using namespace sd::graph;

class CudaLaunchHelperTests : public testing::Test {
public:

};

TEST_F(CudaLaunchHelperTests, test_reduction_blocks_1) {
    ASSERT_EQ(1, CudaLaunchHelper::getReductionBlocks(512));
}

TEST_F(CudaLaunchHelperTests, test_reduction_blocks_2) {
    ASSERT_EQ(1, CudaLaunchHelper::getReductionBlocks(121));
}

TEST_F(CudaLaunchHelperTests, test_reduction_blocks_3) {
    ASSERT_EQ(2, CudaLaunchHelper::getReductionBlocks(513));
}

TEST_F(CudaLaunchHelperTests, test_reduction_blocks_4) {
    ASSERT_EQ(3, CudaLaunchHelper::getReductionBlocks(1225));
}