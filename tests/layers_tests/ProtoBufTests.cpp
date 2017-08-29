//
// @author raver119@gmail.com
//


#include "testlayers.h"
#include <GraphExecutioner.h>

using namespace nd4j::graph;

class ProtoBufTests : public testing::Test {

};

TEST_F(ProtoBufTests, TestSimpleLoad1) {
    auto graph = GraphExecutioner<float>::importFromTensorFlow("../../../tests/resources/tensorflow_inception_graph.pb");
}