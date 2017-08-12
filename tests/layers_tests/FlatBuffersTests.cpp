//
// @author raver119@gmail.com
//

#include "testlayers.h"
#include <flatbuffers/flatbuffers.h>
#include <graph/generated/node_generated.h>
#include <graph/Node.h>
#include <graph/NodeFactory.h>

using namespace nd4j::graph;

class FlatBuffersTest : public testing::Test {
public:
    int alpha = 0;

    int *cShape = new int[8]{2, 2, 2, 2, 1, 0, 1, 99};
    int *fShape = new int[8]{2, 2, 2, 1, 2, 0, 1, 102};
};

/**
 * Simple test that creates Node & reads it
 */
TEST_F(FlatBuffersTest, BasicTest1) {
    flatbuffers::FlatBufferBuilder builder(1024);


    auto node = CreateFlatNode(builder, DataType_INHERIT, OpType_TRANSFORM, 26);

    builder.Finish(node);

    // now we have our buffer with data
    uint8_t *buf = builder.GetBufferPointer();
    int size = builder.GetSize();
    ASSERT_TRUE(size > 0);



    auto restored = GetFlatNode(buf);

    auto gA = new Node<float, simdOps::Ones<float>>(restored);
    auto gB = new Node<float, simdOps::Ones<float>>(restored);

    ASSERT_TRUE(gA->equals(gB));
}


TEST_F(FlatBuffersTest, ExecutionTest1) {
    auto gA = new Node<float, simdOps::Abs<float>>(OpType_TRANSFORM);

    float *c = new float[4] {-1, -2, -3, -4};
    auto *array = new NDArray<float>(c, cShape);

    float *e = new float[4] {1, 2, 3, 4};
    auto *exp = new NDArray<float>(e, cShape);

    gA->execute(array, nullptr, array);

    ASSERT_TRUE(exp->equalsTo(array));
}