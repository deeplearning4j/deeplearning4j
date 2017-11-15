//
// @author raver119@gmail.com
//

#ifndef LIBND4J_WORKSPACETESTS_H
#define LIBND4J_WORKSPACETESTS_H

#include "testlayers.h"
#include <NDArray.h>
#include <Workspace.h>
#include <MemoryRegistrator.h>

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

    NDArray<float> array(5, 5, 'c', &workspace);

    array.putScalar(0, 1.0f);
    array.putScalar(5, 1.0f);

    ASSERT_NEAR(2.0f, array.reduceNumber<simdOps::Sum<float>>(), 1e-5);

    ASSERT_TRUE(workspace.getCurrentOffset() > 0);
}


TEST_F(WorkspaceTests, BasicInitialization3) {
    Workspace workspace(0);

    ASSERT_EQ(0, workspace.getCurrentOffset());

    NDArray<float> array(5, 5, 'c', &workspace);

    array.putScalar(0, 1.0f);
    array.putScalar(5, 1.0f);

    ASSERT_NEAR(2.0f, array.reduceNumber<simdOps::Sum<float>>(), 1e-5);

    ASSERT_TRUE(workspace.getCurrentOffset() == 0);
}


TEST_F(WorkspaceTests, ResetTest1) {
    Workspace workspace(65536);

    NDArray<float> array(5, 5, 'c', &workspace);
    array.putScalar(0, 1.0f);
    array.putScalar(5, 1.0f);

    workspace.scopeOut();
    for (int e = 0; e < 5; e++) {
        workspace.scopeIn();

        NDArray<float> array2(5, 5, 'c', &workspace);
        array2.putScalar(0, 1.0f);
        array2.putScalar(5, 1.0f);

        ASSERT_NEAR(2.0f, array2.reduceNumber<simdOps::Sum<float>>(), 1e-5);

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

    auto ast = new NDArray<float>(5, 5, 'c');

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

    auto ast = new NDArray<float>(5, 5, 'c', &ws);

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


#endif //LIBND4J_WORKSPACETESTS_H
