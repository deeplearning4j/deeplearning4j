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
#include <array/NDArray.h>
#include <memory/Workspace.h>
#include <memory/MemoryRegistrator.h>
#include <helpers/MmulHelper.h>

using namespace sd;
using namespace sd::memory;

class CudaWorkspaceTests : public testing::Test {

};

TEST_F(CudaWorkspaceTests, Basic_Tests_1) {
    Workspace workspace(65536, 65536);

    ASSERT_EQ(0, workspace.getCurrentOffset());
    LaunchContext ctx;
    ctx.setWorkspace(&workspace);
    auto array = NDArrayFactory::create<float>('c', {5, 5}, &ctx);

    ASSERT_EQ(108, workspace.getCurrentOffset());
    ASSERT_EQ(0, workspace.getCurrentSecondaryOffset());

    array.e<int>(0);

    ASSERT_EQ(100, workspace.getCurrentSecondaryOffset());
}

TEST_F(CudaWorkspaceTests, Basic_Tests_2) {
    Workspace workspace(65536, 65536);

    ASSERT_EQ(0, workspace.getCurrentOffset());
    LaunchContext ctx;
    ctx.setWorkspace(&workspace);
    auto array = NDArrayFactory::create<float>('c', {5, 5}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, &ctx);

    ASSERT_EQ(108, workspace.getCurrentOffset());
    ASSERT_EQ(0, workspace.getCurrentSecondaryOffset());
}