//
// Created by raver119 on 13/11/17.
//

#include "testlayers.h"
#include <Graph.h>
#include <Node.h>
#include <ops/declarable/CustomOperations.h>

using namespace nd4j;
using namespace nd4j::graph;

class SanityTests : public testing::Test {
public:

};


TEST_F(SanityTests, VariableSpace_1) {
    VariableSpace<float> variableSpace;
    variableSpace.putVariable(1, new Variable<float>());
    variableSpace.putVariable(1, 1, new Variable<float>());

    std::pair<int, int> pair(1, 2);
    variableSpace.putVariable(pair, new Variable<float>());
}

TEST_F(SanityTests, VariableSpace_2) {
    VariableSpace<float> variableSpace;
    variableSpace.putVariable(1, new Variable<float>(new NDArray<float>('c', {3, 3})));
    variableSpace.putVariable(1, 1, new Variable<float>(new NDArray<float>('c', {3, 3})));

    std::pair<int, int> pair(1, 2);
    variableSpace.putVariable(pair, new Variable<float>(new NDArray<float>('c', {3, 3})));
}


TEST_F(SanityTests, Graph_1) {
    Graph<float> graph;

    graph.getVariableSpace()->putVariable(1, new Variable<float>(new NDArray<float>('c', {3, 3})));
    graph.getVariableSpace()->putVariable(1, 1, new Variable<float>(new NDArray<float>('c', {3, 3})));

    std::pair<int, int> pair(1, 2);
    graph.getVariableSpace()->putVariable(pair, new Variable<float>(new NDArray<float>('c', {3, 3})));
}