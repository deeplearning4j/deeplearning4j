//
// @author raver119@gmail.com
//

#include "testlayers.h"
#include <flatbuffers/flatbuffers.h>
#include <graph/generated/node_generated.h>
#include <graph/Node.h>

using namespace nd4j::graph;

class FlatBuffersTest : public testing::Test {
public:
    int alpha = 0;
};

/**
 * Simple test that creates Node & reads it
 */
TEST_F(FlatBuffersTest, BasicTests1) {
    flatbuffers::FlatBufferBuilder builder(1024);


    auto node = CreateFlatNode(builder, DataType_INHERIT, OpType_TRANSFORM, 26);

    builder.Finish(node);

    // now we have our buffer with data
    uint8_t *buf = builder.GetBufferPointer();
    int size = builder.GetSize();
    ASSERT_TRUE(size > 0);



    auto restored = GetFlatNode(buf);

    auto gA = new GNode<float, simdOps::Ones<float>>(restored);
    auto gB = new GNode<float, simdOps::Ones<float>>(restored);

    ASSERT_TRUE(gA->equals(gB));
}