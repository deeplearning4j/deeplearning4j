//
// @author raver119@gmail.com
//

#include "testlayers.h"
#include <flatbuffers/flatbuffers.h>
#include <graph/generated/node_generated.h>
#include <graph/generated/graph_generated.h>
#include <graph/Node.h>
#include <graph/Graph.h>
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


    auto node = CreateFlatNode(builder, -1, OpType_TRANSFORM, 26, {0});

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


TEST_F(FlatBuffersTest, FlatGraphTest1) {
    flatbuffers::FlatBufferBuilder builder(4096);

    std::vector<int> outputs1, outputs2;
    outputs1.push_back(2);
    outputs2.push_back(0);

    auto vec1 = builder.CreateVector(outputs1);
    auto vec2 = builder.CreateVector(outputs2);

    auto node1 = CreateFlatNode(builder, 1, OpType_TRANSFORM, 26, {0}, DataType_INHERIT, vec1);
    auto node2 = CreateFlatNode(builder, 2, OpType_TRANSFORM, 23, {1}, DataType_INHERIT, vec2);

    std::vector<flatbuffers::Offset<FlatNode>> nodes_vector;

    nodes_vector.push_back(node1);
    nodes_vector.push_back(node2);

    auto nodes = builder.CreateVector(nodes_vector);

    FlatGraphBuilder graphBuilder(builder);

    graphBuilder.add_id(119);
    graphBuilder.add_nodes(nodes);

    auto flatGraph = graphBuilder.Finish();

    builder.Finish(flatGraph);

    uint8_t *buf = builder.GetBufferPointer();
    int size = builder.GetSize();
    ASSERT_TRUE(size > 0);


    auto restoredGraph = GetFlatGraph(buf);
    ASSERT_EQ(119, restoredGraph->id());
    ASSERT_EQ(2, restoredGraph->nodes()->size());

    ASSERT_EQ(26, restoredGraph->nodes()->Get(0)->opNum());
    ASSERT_EQ(23, restoredGraph->nodes()->Get(1)->opNum());
    ASSERT_EQ(26, restoredGraph->nodes()->Get(0)->opNum());


    Graph graph(restoredGraph);

    ASSERT_EQ(2, graph.totalNodes());
    ASSERT_EQ(1, graph.rootNodes());

    graph.execute();
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