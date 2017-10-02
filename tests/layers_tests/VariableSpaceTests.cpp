//
// @author raver119@gmail.com
//

#include "testlayers.h"
#include <flatbuffers/flatbuffers.h>
#include <graph/generated/node_generated.h>
#include <graph/generated/graph_generated.h>
#include <graph/Node.h>
#include <graph/Graph.h>
#include <NDArray.h>

using namespace nd4j::graph;

class VariableSpaceTest : public testing::Test {
public:
    int *cShape = new int[8]{2, 2, 2, 2, 1, 0, 1, 99};
    int *fShape = new int[8]{2, 2, 2, 1, 2, 0, 1, 102};
};


TEST_F(VariableSpaceTest, SettersGettersTest1) {
    auto space1 = new VariableSpace<float>();
    auto arrayA = new NDArray<float>(5, 5, 'c');
    auto arrayB = new NDArray<float>(3, 3, 'c');

    space1->putVariable(1, arrayA);
    space1->putVariable(2, arrayB);

    auto arrayRA = space1->getVariable(1);
    auto arrayRB = space1->getVariable(2);

    ASSERT_TRUE(arrayA == arrayRA->getNDArray());
    ASSERT_TRUE(arrayB == arrayRB->getNDArray());

    // we should survive this call
    delete space1;
}


TEST_F(VariableSpaceTest, SettersGettersTest2) {
    auto space1 = new VariableSpace<float>();
    auto arrayA = new NDArray<float>(5, 5, 'c');
    auto arrayB = new NDArray<float>(3, 3, 'c');

    auto varA = new Variable<float>(arrayA);
    auto varB = new Variable<float>(arrayB);

    varA->markExternal(true);

    space1->putVariable(-1, varA);
    space1->putVariable(2, varB);

    Nd4jIndex expExternal = (25 * 4) + (8 * 4);
    Nd4jIndex expInternal = (9 * 4) + (8 * 4);

    ASSERT_EQ(expExternal, space1->externalMemory());
    ASSERT_EQ(expInternal, space1->internalMemory());

    delete space1;
}

TEST_F(VariableSpaceTest, EqualityTest1) {
    VariableSpace<float> space;

    std::string name("myvar");

    auto arrayA = new NDArray<float>(3, 3, 'c');
    auto variableA = new Variable<float>(arrayA, name.c_str());

    space.putVariable(1, variableA);

    std::pair<int,int> pair(1,0);

    ASSERT_TRUE(space.hasVariable(1));
    ASSERT_TRUE(space.hasVariable(pair));
    ASSERT_TRUE(space.hasVariable(&name));

    auto rV1 = space.getVariable(1);
    auto rV2 = space.getVariable(pair);
    auto rV3 = space.getVariable(&name);

    ASSERT_TRUE(rV1 == rV2);
    ASSERT_TRUE(rV2 == rV3);
}

TEST_F(VariableSpaceTest, EqualityTest2) {
    VariableSpace<float> space;

    auto arrayA = new NDArray<float>(3, 3, 'c');

    space.putVariable(1, arrayA);

    std::pair<int,int> pair(1,0);

    ASSERT_TRUE(space.hasVariable(1));
    ASSERT_TRUE(space.hasVariable(pair));

    auto rV1 = space.getVariable(1);
    auto rV2 = space.getVariable(pair);

    ASSERT_TRUE(rV1 == rV2);
}