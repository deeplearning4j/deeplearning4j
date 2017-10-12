//
// @author raver119@gmail.com
//

#include "testlayers.h"
#include <flatbuffers/flatbuffers.h>
#include <graph/generated/node_generated.h>
#include <graph/generated/graph_generated.h>
#include <graph/generated/result_generated.h>
#include <graph/Node.h>
#include <graph/Graph.h>
#include <GraphExecutioner.h>

using namespace nd4j;
using namespace nd4j::graph;

class FlatBuffersTest : public testing::Test {
public:
    int alpha = 0;

    int *cShape = new int[8]{2, 2, 2, 2, 1, 0, 1, 99};
    int *fShape = new int[8]{2, 2, 2, 1, 2, 0, 1, 102};


    ~FlatBuffersTest() {
        delete[] cShape;
        delete[] fShape;
    }
};

/**
 * Simple test that creates Node & reads it
 */
TEST_F(FlatBuffersTest, BasicTest1) {
    flatbuffers::FlatBufferBuilder builder(1024);

    auto name = builder.CreateString("wow");

    auto node = CreateFlatNode(builder, -1, name, OpType_TRANSFORM, 26, {0});

    builder.Finish(node);

    // now we have our buffer with data
    uint8_t *buf = builder.GetBufferPointer();
    int size = builder.GetSize();
    ASSERT_TRUE(size > 0);



    auto restored = GetFlatNode(buf);

    auto gA = new Node<float>(restored);
    auto gB = new Node<float>(restored);

    ASSERT_TRUE(gA->equals(gB));
}


TEST_F(FlatBuffersTest, FlatGraphTest1) {
    flatbuffers::FlatBufferBuilder builder(4096);

    auto array = new NDArray<float>(5, 5, 'c');
    array->assign(-2.0f);

    auto fShape = builder.CreateVector(array->getShapeAsVector());
    auto fBuffer = builder.CreateVector(array->getBufferAsVector());

    auto fVar = CreateFlatVariable(builder, -1, 0, fShape, fBuffer);

    std::vector<int> outputs1, outputs2, inputs1, inputs2;
    outputs1.push_back(2);
    outputs2.push_back(0);

    inputs1.push_back(-1);
    inputs2.push_back(1);


    auto vec1 = builder.CreateVector(outputs1);
    auto vec2 = builder.CreateVector(outputs2);

    auto in1 = builder.CreateVector(inputs1);
    auto in2 = builder.CreateVector(inputs2);

    auto name1 = builder.CreateString("wow1");
    auto name2 = builder.CreateString("wow2");

    auto node1 = CreateFlatNode(builder, 1, name1, OpType_TRANSFORM, 0, in1, 0, DataType_INHERIT, vec1);
    auto node2 = CreateFlatNode(builder, 2, name2, OpType_TRANSFORM, 2, in2, 0, DataType_INHERIT, vec2);

    std::vector<flatbuffers::Offset<FlatVariable>> variables_vector;
    variables_vector.push_back(fVar);

    std::vector<flatbuffers::Offset<FlatNode>> nodes_vector;

    nodes_vector.push_back(node1);
    nodes_vector.push_back(node2);

    auto nodes = builder.CreateVector(nodes_vector);

    auto variables = builder.CreateVector(variables_vector);

    FlatGraphBuilder graphBuilder(builder);

    graphBuilder.add_variables(variables);
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

    // checking op nodes
    ASSERT_EQ(0, restoredGraph->nodes()->Get(0)->opNum());
    ASSERT_EQ(2, restoredGraph->nodes()->Get(1)->opNum());
    ASSERT_EQ(0, restoredGraph->nodes()->Get(0)->opNum());

    // checking variables
    ASSERT_EQ(1, restoredGraph->variables()->size());
    ASSERT_EQ(-1, restoredGraph->variables()->Get(0)->id());

    nd4j_printf("-------------------------\n","");

    Graph<float> graph(restoredGraph);

    ASSERT_EQ(2, graph.totalNodes());
    ASSERT_EQ(1, graph.rootNodes());


    auto vs = graph.getVariableSpace();

    ASSERT_EQ(OutputMode_IMPLICIT, graph.getExecutorConfiguration()->_outputMode);

    ASSERT_EQ(3, vs->totalEntries());
    ASSERT_EQ(1, vs->externalEntries());
    ASSERT_EQ(2, vs->internalEntries());

    auto var = vs->getVariable(-1)->getNDArray();

    ASSERT_TRUE(var != nullptr);

    ASSERT_EQ(-2.0, var->reduceNumber<simdOps::Mean<float>>());

    nd4j::graph::GraphExecutioner<float>::execute(&graph);

    ASSERT_NEAR(-0.4161468, var->reduceNumber<simdOps::Mean<float>>(), 1e-5);



    auto result = (uint8_t *)nd4j::graph::GraphExecutioner<float>::executeFlatBuffer((Nd4jPointer) buf);

    auto flatResults = GetFlatResult(result);

    ASSERT_EQ(1, flatResults->variables()->size());
    ASSERT_TRUE(flatResults->variables()->Get(0)->name() != nullptr);
    ASSERT_TRUE(flatResults->variables()->Get(0)->name()->c_str() != nullptr);
    nd4j_printf("VARNAME: %s\n", flatResults->variables()->Get(0)->name()->c_str());

    auto var0 = new Variable<float>(flatResults->variables()->Get(0));
    //auto var1 = new Variable<float>(flatResults->variables()->Get(1));


    ASSERT_TRUE(var->equalsTo(var0->getNDArray()));
}

TEST_F(FlatBuffersTest, ExecutionTest1) {
    auto gA = new Node<float>(OpType_TRANSFORM);

    float *c = new float[4] {-1, -2, -3, -4};
    auto *array = new NDArray<float>(c, cShape);

    float *e = new float[4] {1, 2, 3, 4};
    auto *exp = new NDArray<float>(e, cShape);

    //gA->execute(array, nullptr, array);

    //ASSERT_TRUE(exp->equalsTo(array));
}


TEST_F(FlatBuffersTest, ExplicitOutputTest1) {
    flatbuffers::FlatBufferBuilder builder(4096);

    auto x = new NDArray<float>(5, 5, 'c');
    x->assign(-2.0f);

    auto fXShape = builder.CreateVector(x->getShapeAsVector());
    auto fXBuffer = builder.CreateVector(x->getBufferAsVector());

    auto fXVar = CreateFlatVariable(builder, -1, 0, fXShape, fXBuffer);


    auto y = new NDArray<float>(5, 5, 'c');
    y->assign(-1.0f);

    auto fYShape = builder.CreateVector(y->getShapeAsVector());
    auto fYBuffer = builder.CreateVector(y->getBufferAsVector());

    auto fYVar = CreateFlatVariable(builder, -2, 0, fYShape, fYBuffer);


    std::vector<int> inputs1, outputs1, outputs;
    inputs1.push_back(-1);
    inputs1.push_back(-2);

    outputs.push_back(-1);
    outputs.push_back(-2);

    auto out1 = builder.CreateVector(outputs1);
    auto in1 = builder.CreateVector(inputs1);
    auto o = builder.CreateVector(outputs);

    auto name1 = builder.CreateString("wow1");

    auto node1 = CreateFlatNode(builder, 1, name1, OpType_TRANSFORM, 0, in1, 0, DataType_FLOAT, out1);

    std::vector<flatbuffers::Offset<FlatVariable>> variables_vector;
    variables_vector.push_back(fXVar);
    variables_vector.push_back(fYVar);

    std::vector<flatbuffers::Offset<FlatNode>> nodes_vector;
    nodes_vector.push_back(node1);



    auto nodes = builder.CreateVector(nodes_vector);
    auto variables = builder.CreateVector(variables_vector);

    FlatGraphBuilder graphBuilder(builder);

    graphBuilder.add_variables(variables);
    graphBuilder.add_id(119);
    graphBuilder.add_nodes(nodes);
    graphBuilder.add_outputs(o);

    auto flatGraph = graphBuilder.Finish();
    builder.Finish(flatGraph);

    auto restoredGraph = new Graph<float>(GetFlatGraph(builder.GetBufferPointer()));

    GraphExecutioner<float>::execute(restoredGraph);

    auto results = restoredGraph->fetchOutputs();

    // IMPLICIT is default
    ASSERT_EQ(1, results->size());

    //ASSERT_NEAR(-2.0, results->at(0)->getNDArray()->reduceNumber<simdOps::Mean<float>>(), 1e-5);
    //ASSERT_NEAR(-1.0, results->at(1)->getNDArray()->reduceNumber<simdOps::Mean<float>>(), 1e-5);
    ASSERT_NEAR(-3.0, results->at(0)->getNDArray()->reduceNumber<simdOps::Mean<float>>(), 1e-5);

    //ASSERT_EQ(-1, results->at(0)->id());
    //ASSERT_EQ(-2, results->at(1)->id());
}


TEST_F(FlatBuffersTest, ReadFile1) {

    uint8_t* data = nd4j::graph::readFlatBuffers("../../../tests_cpu/resources/adam_sum.fb");

    auto fg = GetFlatGraph(data);
    auto restoredGraph = new Graph<float>(fg);

    ASSERT_EQ(1, restoredGraph->rootNodes());
    ASSERT_EQ(2, restoredGraph->totalNodes());

    auto ones = restoredGraph->getVariableSpace()->getVariable(-1)->getNDArray();

    ASSERT_EQ(4, ones->lengthOf());
    ASSERT_NEAR(4.0f, ones->template reduceNumber<simdOps::Sum<float>>(), 1e-5);

    Nd4jStatus status = GraphExecutioner<float>::execute(restoredGraph);
    ASSERT_EQ(ND4J_STATUS_OK, status);

    auto result = restoredGraph->getVariableSpace()->getVariable(2)->getNDArray();
    ASSERT_EQ(1, result->lengthOf());
    ASSERT_EQ(8, result->getScalar(0));
}

TEST_F(FlatBuffersTest, ReadFile2) {
    uint8_t* data = nd4j::graph::readFlatBuffers("../../../tests_cpu/resources/adam_sum.fb");
    Nd4jPointer result = GraphExecutioner<float>::executeFlatBuffer((Nd4jPointer) data);

    ArrayList<float> arrays(GetFlatResult(result));

    ASSERT_EQ(1, arrays.size());
    ASSERT_EQ(1, arrays.at(0)->lengthOf());
    ASSERT_EQ(8, arrays.at(0)->getScalar(0));
}

TEST_F(FlatBuffersTest, ReadFile3) {
    auto graph = GraphExecutioner<float>::importFromFlatBuffers("../../../tests_cpu/resources/adam_sum.fb");
    Nd4jStatus status = GraphExecutioner<float>::execute(graph);

    ASSERT_EQ(ND4J_STATUS_OK, status);

    auto z = graph->getVariableSpace()->getVariable(2)->getNDArray();

    ASSERT_EQ(1, z->lengthOf());
    ASSERT_EQ(8, z->getScalar(0));
}


TEST_F(FlatBuffersTest, ReadInception1) {
    auto graph = GraphExecutioner<float>::importFromFlatBuffers("../../../tests_cpu/resources/inception.fb");

    Nd4jStatus status = GraphExecutioner<float>::execute(graph);

    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(227));

    auto lastNode = graph->getVariableSpace()->getVariable(227)->getNDArray();

    lastNode->printShapeInfo("Result shape");

    auto argMax = lastNode->argMax();

    nd4j_printf("Predicted class: %i\n", (int) argMax);
    nd4j_printf("Probability: %f\n", lastNode->getScalar(argMax));
    nd4j_printf("Probability ipod: %f\n", lastNode->getScalar(980));
    lastNode->printBuffer("Whole output");

    ASSERT_EQ(561, (int) argMax);

    delete graph;
}