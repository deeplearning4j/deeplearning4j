//
// @author raver119@gmail.com
//


#include "testlayers.h"
#include <GraphExecutioner.h>

using namespace nd4j::graph;

class ProtoBufTests : public testing::Test {

};

TEST_F(ProtoBufTests, TestBinaryLoad1) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    auto graph = GraphExecutioner<float>::importFromTensorFlow("../../../tests/resources/tensorflow_inception_graph.pb");

    ASSERT_FALSE(graph == nullptr);
}

TEST_F(ProtoBufTests, TestTextLoad1) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    auto graph = GraphExecutioner<float>::importFromTensorFlow("../../../tests/resources/max_graph.pb.txt");

    ASSERT_FALSE(graph == nullptr);
}


TEST_F(ProtoBufTests, TestTextLoad2) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    auto graph = GraphExecutioner<float>::importFromTensorFlow("../../../tests/resources/max_add_2.pb.txt");

    ASSERT_FALSE(graph == nullptr);

    ASSERT_EQ(2, graph->getVariableSpace()->externalEntries());

    auto var0 = graph->getVariableSpace()->getVariable(new std::string("zeros"));
    auto var1 = graph->getVariableSpace()->getVariable(new std::string("ones"));

    ASSERT_TRUE(var0 != nullptr);
    ASSERT_TRUE(var1 != nullptr);
}