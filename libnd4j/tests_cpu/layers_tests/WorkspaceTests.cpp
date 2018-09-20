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

#ifndef LIBND4J_WORKSPACETESTS_H
#define LIBND4J_WORKSPACETESTS_H

#include "testlayers.h"
#include <NDArray.h>
#include <Workspace.h>
#include <MemoryRegistrator.h>
#include <MmulHelper.h>

using namespace nd4j;
using namespace nd4j::memory;

class WorkspaceTests : public testing::Test {

};


TEST_F(WorkspaceTests, BasicInitialization1) {
    Workspace workspace(1024);

    ASSERT_EQ(1024, workspace.getCurrentSize());
    ASSERT_EQ(0, workspace.getCurrentOffset());
}

TEST_F(WorkspaceTests, BasicInitialization2) {
    Workspace workspace(65536);

    ASSERT_EQ(0, workspace.getCurrentOffset());

    auto array = NDArrayFactory::_create<float>('c', {5, 5}, &workspace);

    array.putScalar(0, 1.0f);
    array.putScalar(5, 1.0f);

    ASSERT_NEAR(2.0f, array.reduceNumber(reduce::Sum).getScalar<float>(0), 1e-5);

    ASSERT_TRUE(workspace.getCurrentOffset() > 0);
}


TEST_F(WorkspaceTests, BasicInitialization3) {
    Workspace workspace;

    ASSERT_EQ(0, workspace.getCurrentOffset());

    auto array = NDArrayFactory::_create<float>('c', {5, 5}, &workspace);

    array.putScalar(0, 1.0f);
    array.putScalar(5, 1.0f);

    ASSERT_NEAR(2.0f, array.reduceNumber(reduce::Sum).getScalar<float>(0), 1e-5);

    ASSERT_TRUE(workspace.getCurrentOffset() == 0);
}


TEST_F(WorkspaceTests, ResetTest1) {
    Workspace workspace(65536);

    auto array = NDArrayFactory::_create<float>('c', {5, 5}, &workspace);
    array.putScalar(0, 1.0f);
    array.putScalar(5, 1.0f);

    workspace.scopeOut();
    for (int e = 0; e < 5; e++) {
        workspace.scopeIn();

        auto array2 = NDArrayFactory::_create<float>('c', {5, 5}, &workspace);
        array2.putScalar(0, 1.0f);
        array2.putScalar(5, 1.0f);

        ASSERT_NEAR(2.0f, array2.reduceNumber(reduce::Sum).getScalar<float>(0), 1e-5);

        workspace.scopeOut();
    }

    ASSERT_EQ(65536, workspace.getCurrentSize());
    ASSERT_EQ(0, workspace.getCurrentOffset());
    ASSERT_EQ(0, workspace.getSpilledSize());
}


TEST_F(WorkspaceTests, StretchTest1) {
    Workspace workspace(128);
    void* ptr = workspace.allocateBytes(8);
    workspace.scopeOut();
    ASSERT_EQ(0, workspace.getSpilledSize());
    ASSERT_EQ(0, workspace.getCurrentOffset());


    workspace.scopeIn();
    for (int e = 0; e < 10; e++) {

        workspace.allocateBytes(128);

    }
    ASSERT_EQ(128 * 9, workspace.getSpilledSize());
    workspace.scopeOut();
    workspace.scopeIn();

    ASSERT_EQ(0, workspace.getCurrentOffset());

    // we should have absolutely different pointer here, due to reallocation
    void* ptr2 = workspace.allocateBytes(8);

    //ASSERT_FALSE(ptr == ptr2);


    ASSERT_EQ(1280, workspace.getCurrentSize());
    ASSERT_EQ(0, workspace.getSpilledSize());
}

TEST_F(WorkspaceTests, NewInWorkspaceTest1) {
    Workspace ws(65536);

    ASSERT_EQ(65536, ws.getCurrentSize());
    ASSERT_EQ(0, ws.getCurrentOffset());

    ASSERT_FALSE(MemoryRegistrator::getInstance()->hasWorkspaceAttached());

    MemoryRegistrator::getInstance()->attachWorkspace(&ws);

    ASSERT_TRUE(MemoryRegistrator::getInstance()->hasWorkspaceAttached());

    auto ast = NDArrayFactory::create<float>('c', {5, 5});

    ASSERT_TRUE(ws.getCurrentOffset() > 0);

    delete ast;

    MemoryRegistrator::getInstance()->forgetWorkspace();

    ASSERT_FALSE(MemoryRegistrator::getInstance()->hasWorkspaceAttached());
    ASSERT_TRUE(MemoryRegistrator::getInstance()->getWorkspace() == nullptr);
}


TEST_F(WorkspaceTests, NewInWorkspaceTest2) {
    Workspace ws(65536);

    ASSERT_EQ(65536, ws.getCurrentSize());
    ASSERT_EQ(0, ws.getCurrentOffset());

    MemoryRegistrator::getInstance()->attachWorkspace(&ws);

    auto ast = NDArrayFactory::create<float>('c', {5, 5}, &ws);

    ASSERT_TRUE(ws.getCurrentOffset() > 0);

    delete ast;

    MemoryRegistrator::getInstance()->forgetWorkspace();
}

TEST_F(WorkspaceTests, CloneTest1) {
    Workspace ws(65536);

    ws.allocateBytes(65536 * 2);

    ASSERT_EQ(65536 * 2, ws.getSpilledSize());

    auto clone = ws.clone();

    ASSERT_EQ(65536 * 2, clone->getCurrentSize());
    ASSERT_EQ(0, clone->getCurrentOffset());
    ASSERT_EQ(0, clone->getSpilledSize());

    delete clone;
}

TEST_F(WorkspaceTests, Test_Arrays_1) {
    Workspace ws(65536);
    auto x = NDArrayFactory::_create<float>('c', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9}, &ws);

    // x.printIndexedBuffer("x0");

    auto y = NDArrayFactory::_create<float>('c', {3, 3}, {-1, -2, -3, -4, -5, -6, -7, -8, -9}, &ws);

    // x.printIndexedBuffer("x2");

    auto z = NDArrayFactory::_create<float>('c', {3, 3}, {0, 0, 0, 0, 0, 0, 0, 0, 0}, &ws);

    MmulHelper::mmul(&x, &y, &z);

    y.assign(&x);


    // x.printIndexedBuffer("x3");
    // y.printIndexedBuffer("y");
    // z.printIndexedBuffer("z");
}


TEST_F(WorkspaceTests, Test_Graph_1) {
    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/ae_00.fb");
    auto workspace = graph->getVariableSpace()->workspace();

    auto status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);

    delete graph;
}

TEST_F(WorkspaceTests, Test_Externalized_1) {
    char buffer[10000];
    ExternalWorkspace pojo((Nd4jPointer) buffer, 10000, nullptr, 0);

    ASSERT_EQ(10000, pojo.sizeHost());
    ASSERT_EQ(0, pojo.sizeDevice());

    Workspace ws(&pojo);
    ASSERT_EQ(10000, ws.getCurrentSize());
    ASSERT_EQ(10000, ws.getAllocatedSize());

    auto x = NDArrayFactory::_create<float>('c', {10, 10}, &ws);

    ASSERT_EQ(64 + 400, ws.getUsedSize());
    ASSERT_EQ(64 + 400, ws.getCurrentOffset());

    x.assign(2.0);

    float m = x.meanNumber().getScalar<float>(0);
    ASSERT_NEAR(2.0f, m, 1e-5);
}

// TODO: uncomment this test once long shapes are introduced
/*
TEST_F(WorkspaceTests, Test_Big_Allocation_1) {
    Workspace ws(65536);
    NDArray<float> x('c', {256, 64, 384, 384}, &ws);
}
*/


#endif //LIBND4J_WORKSPACETESTS_H