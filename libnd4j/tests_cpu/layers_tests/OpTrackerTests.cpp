//
// Created by raver119 on 15.12.17.
//
#include "testlayers.h"
#include <Graph.h>
#include <chrono>
#include <Node.h>
#include <helpers/OpTracker.h>
#include <ops/declarable/CustomOperations.h>

using namespace nd4j;
using namespace nd4j::graph;

class OpTrackerTests : public testing::Test {
public:
    int numIterations = 10;
    int poolSize = 10;

    OpTrackerTests() {
        printf("\n");
        fflush(stdout);
    }
};

TEST_F(OpTrackerTests, Test_Existence_1) {
    nd4j::_loader loader;

    // nd4j_printf("Groups: %i; Operations: %i\n", OpTracker::getInstance()->totalGroups(), OpTracker::getInstance()->totalOperations());

    ASSERT_TRUE(OpTracker::getInstance()->totalGroups() > 0);
    ASSERT_TRUE(OpTracker::getInstance()->totalOperations() > 0);

    OpTracker::getInstance()->exportOperations();
}

TEST_F(OpTrackerTests, Test_Ops_List_1) {
    nd4j::ops::less<float> op;
    auto vec = OpRegistrator::getInstance()->getAllHashes<float>();

    // nd4j_printf("Total ops: %lld\n", vec.size());
    // nd4j_printf("Less hash: %lld\n", op.getOpHash());

    for (const auto &v: vec) {
        if (v == 5484196977525668316L) {
            auto op = OpRegistrator::getInstance()->getOperationFloat(v);
            // nd4j_printf("OpName: %s\n", op->getOpName()->c_str());
        }
    }
}



