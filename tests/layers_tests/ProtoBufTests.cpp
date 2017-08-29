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