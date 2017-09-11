//
// @author raver119@gmail.com
//

#ifndef LIBND4J_WORKSPACETESTS_H
#define LIBND4J_WORKSPACETESTS_H

#include "testlayers.h"
#include <NDArray.h>
#include <Workspace.h>

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


#endif //LIBND4J_WORKSPACETESTS_H
