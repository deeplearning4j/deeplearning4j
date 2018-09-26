/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

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
#include <ops/declarable/CustomOperations.h>

using namespace nd4j;
using namespace nd4j::graph;

class FlatBuffersTest : public testing::Test {
public:
    int alpha = 0;

    Nd4jLong *cShape = new Nd4jLong[8]{2, 2, 2, 2, 1, 0, 1, 99};
    Nd4jLong *fShape = new Nd4jLong[8]{2, 2, 2, 1, 2, 0, 1, 102};

    FlatBuffersTest() {
        Environment::getInstance()->setDebug(false);
        Environment::getInstance()->setVerbose(false);
        Environment::getInstance()->setProfiling(false);
    }

    ~FlatBuffersTest() {
        Environment::getInstance()->setDebug(false);
        Environment::getInstance()->setVerbose(false);
        Environment::getInstance()->setProfiling(false);

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

    auto gA = new Node(restored);
    auto gB = new Node(restored);

    ASSERT_TRUE(gA->equals(gB));

    delete gA;
    delete gB;
}


TEST_F(FlatBuffersTest, FlatGraphTest1) {
    flatbuffers::FlatBufferBuilder builder(4096);

    auto array = NDArrayFactory::create_<float>('c', {5, 5});
    array->assign(-2.0f);

    auto fShape = builder.CreateVector(array->getShapeInfoAsFlatVector());
    auto fBuffer = builder.CreateVector(array->asByteVector());

    auto fArray = CreateFlatArray(builder, fShape, fBuffer, nd4j::graph::DataType::FLOAT);
    auto fVid = CreateIntPair(builder, -1);

    auto fVar = CreateFlatVariable(builder, fVid, 0, nd4j::graph::DataType::FLOAT, 0, fArray);

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

    auto node1 = CreateFlatNode(builder, 1, name1, OpType_TRANSFORM, 0, 0, in1, 0, nd4j::graph::DataType::FLOAT, vec1);
    auto node2 = CreateFlatNode(builder, 2, name2, OpType_TRANSFORM, 2, 0, in2, 0, nd4j::graph::DataType::FLOAT, vec2);

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
    ASSERT_EQ(-1, restoredGraph->variables()->Get(0)->id()->first());

    // nd4j_printf("-------------------------\n","");

    Graph graph(restoredGraph);

    // graph.printOut();

    ASSERT_EQ(2, graph.totalNodes());
    ASSERT_EQ(1, graph.rootNodes());


    auto vs = graph.getVariableSpace();

    ASSERT_EQ(OutputMode_IMPLICIT, graph.getExecutorConfiguration()->_outputMode);

    ASSERT_EQ(3, vs->totalEntries());
    ASSERT_EQ(1, vs->externalEntries());
    ASSERT_EQ(2, vs->internalEntries());

    auto var = vs->getVariable(-1)->getNDArray();

    ASSERT_TRUE(var != nullptr);

    ASSERT_EQ(-2.0, var->reduceNumber(reduce::Mean).e<float>(0));

    nd4j::graph::GraphExecutioner::execute(&graph);

    auto resultWrapper = nd4j::graph::GraphExecutioner::executeFlatBuffer((Nd4jPointer) buf);

    auto flatResults = GetFlatResult(resultWrapper->pointer());

    ASSERT_EQ(1, flatResults->variables()->size());
    ASSERT_TRUE(flatResults->variables()->Get(0)->name() != nullptr);
    ASSERT_TRUE(flatResults->variables()->Get(0)->name()->c_str() != nullptr);
    //nd4j_printf("VARNAME: %s\n", flatResults->variables()->Get(0)->name()->c_str());

    auto var0 = new Variable(flatResults->variables()->Get(0));
    //auto var1 = new Variable<float>(flatResults->variables()->Get(1));

    ASSERT_NEAR(-0.4161468, var0->getNDArray()->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

    //ASSERT_TRUE(var->equalsTo(var0->getNDArray()));

    delete array;
    delete var0;
    delete resultWrapper;
}

TEST_F(FlatBuffersTest, ExecutionTest1) {
    auto gA = new Node(OpType_TRANSFORM);

    float *c = new float[4] {-1, -2, -3, -4};
    auto array = new NDArray(c, cShape);

    float *e = new float[4] {1, 2, 3, 4};
    auto exp = new NDArray(e, cShape);

    //gA->execute(array, nullptr, array);

    //ASSERT_TRUE(exp->equalsTo(array));

    delete gA;
    delete[] c;
    delete array;
    delete[] e;
    delete exp;
}

/*
TEST_F(FlatBuffersTest, ExplicitOutputTest1) {
    flatbuffers::FlatBufferBuilder builder(4096);

    auto x = NDArrayFactory::create_<float>(5, 5, 'c');
    x->assign(-2.0f);

    auto fXShape = builder.CreateVector(x->getShapeInfoAsVector());
    auto fXBuffer = builder.CreateVector(x->asByteVector());
    auto fXArray = CreateFlatArray(builder, fXShape, fXBuffer);
    auto fXid = CreateIntPair(builder, -1);

    auto fXVar = CreateFlatVariable(builder, fXid, 0, 0, fXArray);


    auto y = NDArrayFactory::create_<float>(5, 5, 'c');
    y->assign(-1.0f);

    auto fYShape = builder.CreateVector(y->getShapeInfoAsVector());
    auto fYBuffer = builder.CreateVector(y->asByteVector());
    auto fYArray = CreateFlatArray(builder, fYShape, fYBuffer);
    auto fYid = CreateIntPair(builder, -2);

    auto fYVar = CreateFlatVariable(builder, fYid, 0, 0, fYArray);


    std::vector<flatbuffers::Offset<IntPair>> inputs1, outputs1, outputs;
    inputs1.push_back(CreateIntPair(builder, -1));
    inputs1.push_back(CreateIntPair(builder, -2));

    outputs.push_back(CreateIntPair(builder, -1));
    outputs.push_back(CreateIntPair(builder, -2));

    auto out1 = builder.CreateVector(outputs1);
    auto in1 = builder.CreateVector(inputs1);
    auto o = builder.CreateVector(outputs);

    auto name1 = builder.CreateString("wow1");

    auto node1 = CreateFlatNode(builder, 1, name1, OpType_TRANSFORM, 0, in1, 0, nd4j::graph::DataType::FLOAT);

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

    delete restoredGraph;
    delete results;
    delete x;
    delete y;
}
*/

/*
TEST_F(FlatBuffersTest, ReadFile1) {

    uint8_t* data = nd4j::graph::readFlatBuffers("./resources/adam_sum.fb");

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
    ASSERT_EQ(8, result->e(0));

    delete[] data;
    delete restoredGraph;
}

TEST_F(FlatBuffersTest, ReadFile2) {
    uint8_t* data = nd4j::graph::readFlatBuffers("./resources/adam_sum.fb");
    Nd4jPointer result = GraphExecutioner<float>::executeFlatBuffer((Nd4jPointer) data);

    ResultSet<float> arrays(GetFlatResult(result));

    ASSERT_EQ(1, arrays.size());
    ASSERT_EQ(1, arrays.at(0)->lengthOf());
    ASSERT_EQ(8, arrays.at(0)->e(0));

    delete[] data;
    delete[] (char *) result;
}

TEST_F(FlatBuffersTest, ReadFile3) {
    auto graph = GraphExecutioner<float>::importFromFlatBuffers("./resources/adam_sum.fb");
    Nd4jStatus status = GraphExecutioner<float>::execute(graph);

    ASSERT_EQ(ND4J_STATUS_OK, status);

    auto z = graph->getVariableSpace()->getVariable(2)->getNDArray();

    ASSERT_EQ(1, z->lengthOf());
    ASSERT_EQ(8, z->e(0));

    delete graph;
}


TEST_F(FlatBuffersTest, ReadInception1) {
    auto graph = GraphExecutioner<float>::importFromFlatBuffers("./resources/inception.fb");

    Nd4jStatus status = GraphExecutioner<float>::execute(graph);

    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(227));

    auto lastNode = graph->getVariableSpace()->getVariable(227)->getNDArray();

    lastNode->printShapeInfo("Result shape");

    auto argMax = lastNode->argMax();

    //nd4j_printf("Predicted class: %i\n", (int) argMax);
    //nd4j_printf("Probability: %f\n", lastNode->e(argMax));
    //nd4j_printf("Probability ipod: %f\n", lastNode->e(980));
    //lastNode->printBuffer("Whole output");

    ASSERT_EQ(561, (int) argMax);

    delete graph;
}

TEST_F(FlatBuffersTest, ReadLoops_3argsWhile_1) {
    // TF graph:
    // https://gist.github.com/raver119/b86ef727e9a094aab386e2b35e878966
    auto graph = GraphExecutioner<float>::importFromFlatBuffers("./resources/three_args_while.fb");

    ASSERT_TRUE(graph != nullptr);

    //graph->printOut();

    auto expPhi('c', {2, 2});

    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(-1));
    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(-2));

    auto phi = graph->getVariableSpace()->getVariable(-2)->getNDArray();
    auto constA = graph->getVariableSpace()->getVariable(-5)->getNDArray();
    auto lessY = graph->getVariableSpace()->getVariable(-6)->getNDArray();

    //constA->printBuffer("constA");
    //lessY->printBuffer("lessY");

    ASSERT_TRUE(expPhi.isSameShape(phi));

    Nd4jStatus status = GraphExecutioner<float>::execute(graph);

    ASSERT_EQ(ND4J_STATUS_OK, status);

    // now, we expect some values

    auto x = graph->getVariableSpace()->getVariable(20);
    auto y = graph->getVariableSpace()->getVariable(21);

    ASSERT_NEAR(110.0f, x->getNDArray()->meanNumber(), 1e-5);
    ASSERT_NEAR(33.0f, y->getNDArray()->meanNumber(), 1e-5);

    delete graph;
}



TEST_F(FlatBuffersTest, ReadTensorArrayLoop_1) {
    auto exp('c', {5, 2}, {3., 6., 9., 12., 15., 18., 21., 24., 27., 30.});
    auto graph = GraphExecutioner<float>::importFromFlatBuffers("./resources/tensor_array_loop.fb");

    ASSERT_TRUE(graph != nullptr);

    //graph->printOut();

    Nd4jStatus status = GraphExecutioner<float>::execute(graph);

    ASSERT_EQ(ND4J_STATUS_OK, status);

    auto variableSpace = graph->getVariableSpace();

    ASSERT_TRUE(variableSpace->hasVariable(23,0));

    auto z = variableSpace->getVariable(23)->getNDArray();

    //z->printShapeInfo("z shape");
    //z->printIndexedBuffer("z buffer");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete graph;
}

*/

/*
TEST_F(FlatBuffersTest, ReadLoops_NestedWhile_1) {
    // TF graph:
    // https://gist.github.com/raver119/2aa49daf7ec09ed4ddddbc6262f213a0
    nd4j::ops::assign<float> op1;

    auto graph = GraphExecutioner<float>::importFromFlatBuffers("./resources/nested_while.fb");

    ASSERT_TRUE(graph != nullptr);

    Nd4jStatus status = GraphExecutioner<float>::execute(graph);

    ASSERT_EQ(ND4J_STATUS_OK, status);

    auto x = graph->getVariableSpace()->getVariable(28);
    auto y = graph->getVariableSpace()->getVariable(29);
    auto z = graph->getVariableSpace()->getVariable(11, 2);

    ASSERT_NEAR(110.0f, x->getNDArray()->meanNumber(), 1e-5);
    ASSERT_NEAR(33.0f, y->getNDArray()->meanNumber(), 1e-5);

    // we should have only 3 cycles in nested loop
    ASSERT_NEAR(30.0f, z->getNDArray()->meanNumber(), 1e-5);

    delete graph;
}
*/
/*

TEST_F(FlatBuffersTest, ReadTensorArray_1) {
    // TF graph: https://gist.github.com/raver119/3265923eed48feecc465d17ec842b6e2
    float _expB[] = {1.000000, 1.000000, 2.000000, 2.000000, 3.000000, 3.000000};
    auto exp('c', {3, 2});
    exp.setBuffer(_expB);

    auto graph = GraphExecutioner<float>::importFromFlatBuffers("./resources/tensor_array.fb");

    ASSERT_TRUE(graph != nullptr);

    Nd4jStatus status = GraphExecutioner<float>::execute(graph);

    ASSERT_EQ(ND4J_STATUS_OK, status);

    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(14));

    auto z = graph->getVariableSpace()->getVariable(14)->getNDArray();

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete graph;
}

*/
/*
TEST_F(FlatBuffersTest, ReadStridedSlice_1) {
    // TF graph: https://gist.github.com/raver119/fc3bf2d31c91e465c635b24020fd798d
    auto graph = GraphExecutioner<float>::importFromFlatBuffers("./resources/tensor_slice.fb");

    ASSERT_TRUE(graph != nullptr);

    Nd4jStatus status = GraphExecutioner<float>::execute(graph);

    ASSERT_EQ(ND4J_STATUS_OK, status);

    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(7));

    auto z = graph->getVariableSpace()->getVariable(7)->getNDArray();

    ASSERT_NEAR(73.5f, z->e(0), 1e-5);

    delete graph;
}
*/


TEST_F(FlatBuffersTest, ReduceDim_1) {
    auto exp = NDArrayFactory::create<float>('c', {3});
    exp.assign(3.0);


    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/reduce_dim_false.fb");

    graph->printOut();

    auto variableSpace = graph->getVariableSpace();


    ASSERT_TRUE(variableSpace->hasVariable(1));
    ASSERT_TRUE(variableSpace->hasVariable(2));

    auto x = variableSpace->getVariable(1)->getNDArray();
    auto y = variableSpace->getVariable(2)->getNDArray();

    Nd4jStatus status = GraphExecutioner::execute(graph);

    ASSERT_EQ(ND4J_STATUS_OK, status);

    ASSERT_TRUE(variableSpace->hasVariable(3));

    auto result = variableSpace->getVariable(3)->getNDArray();

    ASSERT_TRUE(exp.isSameShape(result));
    ASSERT_TRUE(exp.equalsTo(result));

    delete graph;
}

TEST_F(FlatBuffersTest, ReduceDim_2) {
    auto exp = NDArrayFactory::create<float>('c', {3, 1});
    exp.assign(3.0);


    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/reduce_dim_true.fb");

    graph->printOut();

    auto variableSpace = graph->getVariableSpace();


    ASSERT_TRUE(variableSpace->hasVariable(1));
    ASSERT_TRUE(variableSpace->hasVariable(2));

    auto x = variableSpace->getVariable(1)->getNDArray();
    auto y = variableSpace->getVariable(2)->getNDArray();

    Nd4jStatus status = GraphExecutioner::execute(graph);

    ASSERT_EQ(ND4J_STATUS_OK, status);

    ASSERT_TRUE(variableSpace->hasVariable(3));

    auto result = variableSpace->getVariable(3)->getNDArray();

    ASSERT_TRUE(exp.isSameShape(result));
    ASSERT_TRUE(exp.equalsTo(result));

    delete graph;
}


TEST_F(FlatBuffersTest, Ae_00) {
    nd4j::ops::rank op1;

    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/ae_00.fb");

    auto exp = NDArrayFactory::create<float>('c', {5, 4}, {0.32454616f, -0.06604697f, 0.22593613f, 0.43166467f, -0.18320604f, 0.00102305f, -0.06963076f, 0.25266643f, 0.07568010f, -0.03009197f, 0.07805517f, 0.33180334f, -0.06220427f, 0.07249600f, -0.06726961f, -0.22998397f, -0.06343779f, 0.07384885f, -0.06891008f,  -0.23745790f});

//    graph->printOut();

    ASSERT_EQ(OutputMode_VARIABLE_SPACE, graph->getExecutorConfiguration()->_outputMode);

    auto result = GraphExecutioner::execute(graph);
    ASSERT_EQ(ND4J_STATUS_OK, result);

    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(18));

    auto z = graph->getVariableSpace()->getVariable(18)->getNDArray();

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete graph;
}

TEST_F(FlatBuffersTest, expand_dims) {
    nd4j::ops::rank op1;

    auto exp = NDArrayFactory::create<float>('c', {3, 1, 4}, {-0.95938617f, -1.20301781f, 1.22260064f, 0.50172403f, 0.59972949f, 0.78568028f, 0.31609724f, 1.51674747f, 0.68013491f, -0.05227458f, 0.25903158f, 1.13243439f});

    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/expand_dim.fb");

//    graph->printOut();

    auto result = GraphExecutioner::execute(graph);
    ASSERT_EQ(ND4J_STATUS_OK, result);
    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(5));

    auto z = graph->getVariableSpace()->getVariable(5)->getNDArray();

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete graph;
}

TEST_F(FlatBuffersTest, transpose) {
    nd4j::ops::rank op1;

    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/transpose.fb");

    //graph->printOut();

    auto result = GraphExecutioner::execute(graph);
    ASSERT_EQ(ND4J_STATUS_OK, result);

    delete graph;
}

TEST_F(FlatBuffersTest, Test_Stitches) {
    nd4j::ops::realdiv op0;

    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/partition_stitch_misc.fb");
    //graph->printOut();


    auto result = GraphExecutioner::execute(graph);
    ASSERT_EQ(ND4J_STATUS_OK, result);

    delete graph;
}

TEST_F(FlatBuffersTest, Test_GruDynamicMnist) {
    nd4j::Environment::getInstance()->setDebug(false);
    nd4j::Environment::getInstance()->setVerbose(false);

    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/gru_dynamic_mnist.fb");
    //graph->printOut();

    auto timeStart = std::chrono::system_clock::now();
    auto result = GraphExecutioner::execute(graph);
    ASSERT_EQ(ND4J_STATUS_OK, result);

    auto timeEnd = std::chrono::system_clock::now();

    auto outerTime = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();

    // nd4j_printf("GRU time 1 time %lld us\n", outerTime);

    delete graph;
}

TEST_F(FlatBuffersTest, Test_Non2D_2) {
    nd4j::Environment::getInstance()->setDebug(false);
    nd4j::Environment::getInstance()->setVerbose(false);
    nd4j::ops::realdiv op0;

    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/non2d_2.fb");
    //graph->printOut();

    auto result = GraphExecutioner::execute(graph);
    ASSERT_EQ(ND4J_STATUS_OK, result);

    delete graph;
}


TEST_F(FlatBuffersTest, Test_TensorDotMisc) {
    Environment::getInstance()->setVerbose(false);
    Environment::getInstance()->setDebug(false);

    auto e = NDArrayFactory::create<float>('c', {1, 3, 16, 20}, {4.f, 6.f, 6.f, 5.f, 6.f, 4.f, 2.f, 3.f, 5.f, 5.f, 1.f, 4.f, 6.f, 3.f, 2.f, 1.f, 5.f, 4.f, 4.f, 4.f, 4.f, 4.f, 3.f, 4.f, 2.f, 3.f, 3.f, 5.f, 3.f, 6.f, 5.f, 4.f, 4.f, 3.f, 6.f, 1.f, 2.f, 4.f, 2.f, 6.f, 4.f, 2.f, 3.f, 2.f, 3.f, 1.f, 2.f, 4.f, 3.f, 5.f, 3.f, 3.f, 5.f, 2.f, 6.f, 3.f, 4.f, 4.f, 4.f, 4.f, 6.f, 4.f, 5.f, 2.f, 5.f, 5.f, 5.f, 5.f, 2.f, 4.f, 4.f, 4.f, 5.f, 4.f, 3.f, 6.f, 3.f, 4.f, 5.f, 2.f, 5.f, 4.f, 4.f, 5.f, 4.f, 3.f, 4.f, 5.f, 5.f, 3.f, 5.f, 6.f, 6.f, 3.f, 4.f, 5.f, 7.f, 6.f, 5.f, 2.f, 4.f, 5.f, 5.f, 4.f, 5.f, 4.f, 4.f, 6.f, 3.f, 4.f, 5.f, 4.f, 6.f, 2.f, 3.f, 4.f, 3.f, 3.f, 2.f, 2.f, 3.f, 4.f, 7.f, 3.f, 5.f, 4.f, 5.f, 4.f, 4.f, 4.f, 4.f, 6.f, 2.f, 3.f, 2.f, 5.f, 5.f, 4.f, 5.f, 2.f, 2.f, 1.f, 6.f, 2.f, 2.f, 3.f, 4.f, 5.f, 5.f, 3.f, 6.f, 6.f, 4.f, 3.f, 3.f, 3.f, 3.f, 3.f, 4.f, 5.f, 4.f, 4.f, 3.f, 5.f, 2.f, 3.f, 4.f, 5.f, 3.f, 4.f, 5.f, 5.f, 8.f, 4.f, 5.f, 3.f, 3.f, 4.f, 4.f, 5.f, 4.f, 5.f, 3.f, 3.f, 7.f, 2.f, 3.f, 2.f, 6.f, 6.f, 4.f, 4.f, 3.f, 5.f, 6.f, 2.f, 4.f, 3.f, 3.f, 4.f, 5.f, 3.f, 3.f, 6.f, 5.f, 3.f, 2.f, 5.f, 4.f, 4.f, 3.f, 5.f, 5.f, 6.f, 7.f, 3.f, 4.f, 3.f, 5.f, 6.f, 7.f, 5.f, 6.f, 5.f, 7.f, 4.f, 6.f, 5.f, 5.f, 6.f, 4.f, 2.f, 5.f, 4.f, 3.f, 4.f, 1.f, 5.f, 5.f, 3.f, 2.f, 2.f, 6.f, 5.f, 5.f, 2.f, 5.f, 2.f, 4.f, 4.f, 5.f, 5.f, 4.f, 3.f, 7.f, 4.f, 5.f, 3.f, 3.f, 3.f, 2.f, 3.f, 2.f, 3.f, 3.f, 4.f, 4.f, 2.f, 4.f, 5.f, 3.f, 4.f, 5.f, 3.f, 7.f, 2.f, 1.f, 3.f, 2.f, 3.f, 2.f, 3.f, 3.f, 4.f, 3.f, 4.f, 2.f, 4.f, 4.f, 4.f, 5.f, 3.f, 5.f, 3.f, 6.f, 6.f, 5.f, 3.f, 5.f, 3.f, 4.f, 3.f, 5.f, 3.f, 5.f, 6.f, 5.f, 3.f, 4.f, 5.f, 5.f, 3.f, 3.f, 3.f, 4.f, 6.f, 4.f, 3.f, 7.f, 4.f, 4.f, 6.f, 7.f, 5.f, 5.f, 3.f, 1.f, 2.f, 5.f, 5.f, 2.f, 5.f, 7.f, 5.f, 3.f, 1.f, 4.f, 6.f, 5.f, 7.f, 5.f, 6.f, 5.f, 6.f, 4.f, 3.f, 3.f, 4.f, 3.f, 4.f, 4.f, 4.f, 4.f, 3.f, 5.f, 2.f, 4.f, 5.f, 2.f, 5.f, 5.f, 4.f, 5.f, 4.f, 5.f, 2.f, 3.f, 5.f, 3.f, 6.f, 3.f, 4.f, 5.f, 3.f, 6.f, 5.f, 5.f, 6.f, 4.f, 6.f, 7.f, 4.f, 5.f, 3.f, 5.f, 4.f, 4.f, 4.f, 2.f, 2.f, 5.f, 3.f, 5.f, 3.f, 4.f, 6.f, 3.f, 5.f, 5.f, 3.f, 5.f, 4.f, 4.f, 4.f, 5.f, 2.f, 3.f, 5.f, 4.f, 2.f, 4.f, 5.f, 4.f, 2.f, 3.f, 4.f, 4.f, 5.f, 5.f, 1.f, 4.f, 4.f, 4.f, 3.f, 4.f, 5.f, 5.f, 8.f, 4.f, 4.f, 4.f, 3.f, 6.f, 2.f, 3.f, 4.f, 4.f, 4.f, 3.f, 2.f, 3.f, 4.f, 8.f, 3.f, 5.f, 5.f, 5.f, 3.f, 3.f, 4.f, 5.f, 7.f, 3.f, 3.f, 3.f, 6.f, 6.f, 5.f, 5.f, 3.f, 4.f, 3.f, 8.f, 3.f, 4.f, 2.f, 3.f, 4.f, 4.f, 3.f, 5.f, 5.f, 3.f, 2.f, 3.f, 3.f, 3.f, 4.f, 4.f, 4.f, 6.f, 6.f, 5.f, 6.f, 4.f, 5.f, 4.f, 6.f, 4.f, 5.f, 5.f, 4.f, 7.f, 3.f, 5.f, 5.f, 3.f, 5.f, 5.f, 6.f, 4.f, 5.f, 4.f, 2.f, 7.f, 2.f, 3.f, 1.f, 4.f, 5.f, 5.f, 4.f, 4.f, 5.f, 7.f, 2.f, 3.f, 3.f, 4.f, 4.f, 5.f, 3.f, 3.f, 6.f, 6.f, 3.f, 2.f, 4.f, 3.f, 3.f, 3.f, 3.f, 4.f, 4.f, 5.f, 1.f, 2.f, 3.f, 3.f, 4.f, 5.f, 4.f, 5.f, 4.f, 5.f, 6.f, 6.f, 6.f, 6.f, 7.f, 4.f, 3.f, 4.f, 5.f, 4.f, 4.f, 2.f, 5.f, 6.f, 4.f, 2.f, 2.f, 6.f, 5.f, 5.f, 1.f, 4.f, 2.f, 3.f, 4.f, 5.f, 5.f, 4.f, 5.f, 9.f, 4.f, 6.f, 4.f, 5.f, 5.f, 3.f, 4.f, 5.f, 5.f, 5.f, 4.f, 3.f, 1.f, 3.f, 4.f, 3.f, 4.f, 4.f, 3.f, 6.f, 2.f, 3.f, 3.f, 2.f, 3.f, 3.f, 4.f, 5.f, 6.f, 5.f, 5.f, 3.f, 4.f, 5.f, 5.f, 4.f, 3.f, 4.f, 3.f, 6.f, 7.f, 6.f, 4.f, 6.f, 4.f, 3.f, 3.f, 4.f, 3.f, 5.f, 5.f, 4.f, 2.f, 3.f, 4.f, 5.f, 3.f, 4.f, 2.f, 4.f, 5.f, 3.f, 3.f, 7.f, 4.f, 2.f, 5.f, 6.f, 5.f, 5.f, 3.f, 1.f, 2.f, 4.f, 4.f, 1.f, 3.f, 6.f, 3.f, 3.f, 1.f, 4.f, 4.f, 4.f, 5.f, 3.f, 4.f, 3.f, 4.f, 2.f, 3.f, 3.f, 4.f, 3.f, 4.f, 3.f, 3.f, 4.f, 2.f, 5.f, 1.f, 3.f, 4.f, 2.f, 6.f, 4.f, 3.f, 4.f, 3.f, 3.f, 1.f, 2.f, 5.f, 2.f, 6.f, 4.f, 5.f, 6.f, 3.f, 6.f, 4.f, 4.f, 5.f, 3.f, 5.f, 6.f, 3.f, 4.f, 2.f, 4.f, 5.f, 5.f, 5.f, 2.f, 3.f, 4.f, 3.f, 5.f, 3.f, 3.f, 9.f, 6.f, 7.f, 7.f, 4.f, 4.f, 3.f, 3.f, 4.f, 4.f, 3.f, 4.f, 6.f, 5.f, 3.f, 5.f, 5.f, 5.f, 2.f, 4.f, 6.f, 7.f, 7.f, 5.f, 3.f, 4.f, 5.f, 4.f, 4.f, 5.f, 5.f, 5.f, 8.f, 4.f, 4.f, 4.f, 3.f, 5.f, 3.f, 3.f, 4.f, 4.f, 5.f, 3.f, 3.f, 2.f, 3.f, 6.f, 2.f, 5.f, 4.f, 4.f, 3.f, 3.f, 3.f, 5.f, 7.f, 2.f, 3.f, 2.f, 5.f, 5.f, 4.f, 4.f, 2.f, 2.f, 1.f, 6.f, 1.f, 2.f, 2.f, 3.f, 5.f, 4.f, 3.f, 5.f, 5.f, 3.f, 2.f, 2.f, 2.f, 2.f, 4.f, 3.f, 4.f, 4.f, 4.f, 4.f, 5.f, 2.f, 4.f, 4.f, 5.f, 2.f, 4.f, 4.f, 5.f, 9.f, 4.f, 5.f, 4.f, 3.f, 5.f, 5.f, 6.f, 4.f, 4.f, 3.f, 3.f, 6.f, 2.f, 3.f, 2.f, 5.f, 6.f, 4.f, 4.f, 3.f, 5.f, 6.f, 4.f, 5.f, 5.f, 6.f, 7.f, 4.f, 2.f, 3.f, 5.f, 4.f, 4.f, 3.f, 5.f, 5.f, 4.f, 3.f, 4.f, 5.f, 4.f, 6.f, 3.f, 4.f, 4.f, 5.f, 6.f, 6.f, 4.f, 6.f, 6.f, 6.f, 5.f, 6.f, 6.f, 7.f, 7.f, 4.f, 3.f, 4.f, 4.f, 4.f, 5.f, 2.f, 5.f, 7.f, 5.f, 2.f, 1.f, 5.f, 5.f, 4.f, 1.f, 4.f, 1.f, 3.f, 3.f, 5.f, 4.f, 4.f, 3.f, 7.f, 3.f, 6.f, 3.f, 3.f, 4.f, 1.f, 3.f, 2.f, 3.f, 3.f, 4.f, 3.f, 1.f, 3.f, 4.f, 2.f, 4.f, 4.f, 2.f, 6.f, 1.f, 2.f, 2.f, 2.f, 3.f, 2.f, 3.f, 3.f, 4.f, 4.f, 4.f, 2.f, 4.f, 4.f, 4.f, 5.f, 5.f, 5.f, 4.f, 8.f, 5.f, 5.f, 3.f, 5.f, 3.f, 3.f, 2.f, 4.f, 3.f, 5.f, 6.f, 5.f, 3.f, 4.f, 5.f, 5.f, 3.f, 4.f, 3.f, 4.f, 8.f, 6.f, 5.f, 9.f, 6.f});
    
    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/tensor_dot_misc.fb");
//    graph->printOut();

    auto result = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), result);

    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(77));

    auto z = graph->getVariableSpace()->getVariable(77,0)->getNDArray();

    ASSERT_EQ(e, *z);

    delete graph;
}


TEST_F(FlatBuffersTest, Test_MNIST_00_1) {
    auto e = NDArrayFactory::create<float>('c', {100, 10}, {0.00066107f,            0.00002358f,            0.00031518f,            0.00238039f,            0.00027216f,            0.00030300f,            0.00004659f,            0.98962247f,            0.00050380f,            0.00587174f,            0.05895791f,            0.00323104f,            0.52636790f,            0.12912551f,            0.00003951f,            0.03615341f,            0.22013727f,            0.00007333f,            0.02566659f,            0.00024759f,            0.00192367f,            0.90509874f,            0.01985082f,            0.02080356f,            0.00260053f,            0.00497826f,            0.01107823f,            0.00872595f,            0.01559795f,            0.00934229f,            0.98202229f,            0.00000150f,            0.00137381f,            0.00082931f,            0.00001806f,            0.00384426f,            0.00758274f,            0.00305049f,            0.00052152f,            0.00075617f,            0.01094264f,            0.00044708f,            0.03576852f,            0.00711267f,            0.65963465f,            0.00734364f,            0.02747800f,            0.06494589f,            0.02966754f,            0.15665947f,            0.00035806f,            0.95196360f,            0.00622721f,            0.01610696f,            0.00084180f,            0.00139947f,            0.00127350f,            0.00577912f,            0.00980321f,            0.00624705f,            0.00167418f,            0.00125611f,            0.00109477f,            0.04061511f,            0.57403159f,            0.08173440f,            0.00423709f,            0.10187119f,            0.07103974f,            0.12244581f,            0.00073566f,            0.00624759f,            0.00559816f,            0.01215601f,            0.08299568f,            0.06209232f,            0.01742392f,            0.01341172f,            0.02181461f,            0.77752429f,            0.08474547f,            0.00957346f,            0.29235491f,            0.00243696f,            0.06653537f,            0.03792902f,            0.43910959f,            0.00344940f,            0.02626713f,            0.03759870f,            0.00143713f,            0.00011047f,            0.00018820f,            0.00047970f,            0.02127167f,            0.00308758f,            0.00093357f,            0.17067374f,            0.00545499f,            0.79636300f,            0.95257199f,            0.00002157f,            0.00647615f,            0.01024892f,            0.00005942f,            0.01910058f,            0.00044579f,            0.00008416f,            0.01097712f,            0.00001441f,            0.16705236f,            0.01782482f,            0.17580827f,            0.06262068f,            0.03860324f,            0.01763505f,            0.32766294f,            0.00555595f,            0.17227779f,            0.01495883f,            0.00180449f,            0.00010494f,            0.00075124f,            0.00161161f,            0.08859238f,            0.00364861f,            0.00162414f,            0.06005199f,            0.00805061f,            0.83375996f,            0.97355360f,            0.00000305f,            0.00144336f,            0.00051544f,            0.00010043f,            0.00714774f,            0.00021183f,            0.00042562f,            0.01294680f,            0.00365222f,            0.00026871f,            0.95752406f,            0.00408361f,            0.02153200f,            0.00015639f,            0.00153930f,            0.00323335f,            0.00178700f,            0.00516464f,            0.00471107f,            0.07408376f,            0.00468759f,            0.02638813f,            0.33325842f,            0.01172767f,            0.36993489f,            0.01118315f,            0.01460529f,            0.14850292f,            0.00562817f,            0.00551083f,            0.00015134f,            0.01184739f,            0.00643833f,            0.11686873f,            0.00163741f,            0.00582776f,            0.11497385f,            0.02010887f,            0.71663547f,            0.00154932f,            0.00001290f,            0.00023825f,            0.01393047f,            0.00012438f,            0.00033184f,            0.00010033f,            0.98197538f,            0.00022847f,            0.00150876f,            0.00597587f,            0.00819661f,            0.03041674f,            0.43121871f,            0.00986523f,            0.13834484f,            0.29576671f,            0.01305170f,            0.03919542f,            0.02796829f,            0.00139392f,            0.00031466f,            0.00229704f,            0.00647669f,            0.86193180f,            0.01064646f,            0.00494287f,            0.00901443f,            0.00526376f,            0.09771839f,            0.00184158f,            0.00040986f,            0.00008309f,            0.01634205f,            0.01102151f,            0.01133229f,            0.00011603f,            0.30489817f,            0.00813993f,            0.64581543f,            0.00132390f,            0.00009014f,            0.00471620f,            0.00419161f,            0.01024686f,            0.02504917f,            0.94500881f,            0.00010234f,            0.00620976f,            0.00306121f,            0.00971363f,            0.05415262f,            0.05265132f,            0.01217585f,            0.16251956f,            0.00188165f,            0.61800343f,            0.04541704f,            0.01950107f,            0.02398386f,            0.05354780f,            0.00129718f,            0.00762409f,            0.06902183f,            0.01746517f,            0.71758413f,            0.04491642f,            0.00194128f,            0.07204670f,            0.01455537f,            0.00356139f,            0.00223315f,            0.01881612f,            0.01844147f,            0.65686893f,            0.01172961f,            0.01321550f,            0.06555344f,            0.00993031f,            0.19965005f,            0.99641657f,            0.00000005f,            0.00027076f,            0.00000523f,            0.00001288f,            0.00173779f,            0.00140848f,            0.00001787f,            0.00012701f,            0.00000342f,            0.00364264f,            0.00040242f,            0.00199880f,            0.01658181f,            0.00522031f,            0.00494563f,            0.00134627f,            0.87392259f,            0.00277323f,            0.08916643f,            0.00200165f,            0.00006030f,            0.00265544f,            0.00137030f,            0.85328883f,            0.00988892f,            0.00416652f,            0.00394441f,            0.00617034f,            0.11645336f,            0.97291315f,            0.00000182f,            0.00194084f,            0.01498440f,            0.00001028f,            0.00389095f,            0.00023297f,            0.00044887f,            0.00528154f,            0.00029516f,            0.00188889f,            0.79829764f,            0.01104437f,            0.04222726f,            0.00522182f,            0.04550264f,            0.03192228f,            0.01099020f,            0.04107348f,            0.01183154f,            0.00058263f,            0.00048307f,            0.00013920f,            0.96885711f,            0.00005209f,            0.01755359f,            0.00061751f,            0.00787173f,            0.00087605f,            0.00296709f,            0.00342248f,            0.68736714f,            0.01477064f,            0.11038199f,            0.00979373f,            0.03290173f,            0.02064420f,            0.03154078f,            0.03068676f,            0.05849051f,            0.00054699f,            0.00028973f,            0.00066918f,            0.79915440f,            0.00078404f,            0.18881910f,            0.00078736f,            0.00024780f,            0.00598373f,            0.00271761f,            0.37178108f,            0.00029151f,            0.11573081f,            0.00016159f,            0.08614764f,            0.05626433f,            0.33961067f,            0.00184490f,            0.01931754f,            0.00884999f,            0.00103338f,            0.00105793f,            0.01583840f,            0.01417849f,            0.00086645f,            0.00075313f,            0.00009471f,            0.92975640f,            0.00786521f,            0.02855594f,            0.00831110f,            0.00041050f,            0.95547730f,            0.01004958f,            0.00024040f,            0.00674337f,            0.01100292f,            0.00229303f,            0.00543977f,            0.00003204f,            0.00073861f,            0.00003656f,            0.00233217f,            0.00864751f,            0.00044351f,            0.00055325f,            0.00046273f,            0.97456056f,            0.00097461f,            0.01125053f,            0.00035382f,            0.94428235f,            0.00286066f,            0.01286138f,            0.00111129f,            0.00731637f,            0.00518610f,            0.00538214f,            0.01197775f,            0.00866815f,            0.06013579f,            0.03228600f,            0.20441757f,            0.54548728f,            0.00006484f,            0.02362618f,            0.05482962f,            0.00106437f,            0.07713205f,            0.00095635f,            0.00029120f,            0.94839782f,            0.00271641f,            0.02038633f,            0.00010249f,            0.00270848f,            0.00299053f,            0.00069419f,            0.01599395f,            0.00571855f,            0.00580072f,            0.81594771f,            0.03097420f,            0.03646614f,            0.00565077f,            0.01715674f,            0.02362122f,            0.01730293f,            0.02312471f,            0.02395495f,            0.00083797f,            0.00032276f,            0.00475549f,            0.00577861f,            0.00193654f,            0.00201117f,            0.00095864f,            0.89032167f,            0.00238766f,            0.09068950f,            0.00007685f,            0.00309113f,            0.00165920f,            0.00566203f,            0.79406202f,            0.00106585f,            0.00073159f,            0.02779965f,            0.01331810f,            0.15253356f,            0.01362522f,            0.17258310f,            0.57671696f,            0.04606603f,            0.02204953f,            0.00909986f,            0.04971812f,            0.00135137f,            0.09417208f,            0.01461779f,            0.00351132f,            0.01659229f,            0.02209206f,            0.77456558f,            0.00303461f,            0.07932901f,            0.06269170f,            0.01151956f,            0.01363456f,            0.01302921f,            0.04056359f,            0.00052574f,            0.00214679f,            0.41835260f,            0.00373941f,            0.47472891f,            0.00819933f,            0.00047488f,            0.04602791f,            0.00524084f,            0.00085833f,            0.19585223f,            0.03986045f,            0.44138056f,            0.01866945f,            0.11297230f,            0.03688592f,            0.03147812f,            0.04306961f,            0.07897298f,            0.00580970f,            0.00654101f,            0.80165571f,            0.01388136f,            0.04366852f,            0.00407737f,            0.07712067f,            0.01289223f,            0.01437380f,            0.01997955f,            0.00013239f,            0.00000585f,            0.00003676f,            0.00288744f,            0.76327205f,            0.00911173f,            0.00025323f,            0.00345270f,            0.00977252f,            0.21107534f,            0.00238540f,            0.00011487f,            0.01707160f,            0.00274678f,            0.85196322f,            0.00066304f,            0.01279381f,            0.02112481f,            0.00446795f,            0.08666852f,            0.01046857f,            0.00011744f,            0.00377885f,            0.00806424f,            0.00110093f,            0.01087467f,            0.96216726f,            0.00024677f,            0.00213707f,            0.00104427f,            0.00835356f,            0.00037980f,            0.00540865f,            0.91882282f,            0.00084274f,            0.03935680f,            0.00700863f,            0.00609934f,            0.00307425f,            0.01065346f,            0.09310398f,            0.00066428f,            0.00076882f,            0.02210450f,            0.04447530f,            0.77650899f,            0.00945148f,            0.00689890f,            0.00886871f,            0.03715509f,            0.07214937f,            0.00624633f,            0.01399398f,            0.29444799f,            0.03825752f,            0.36904955f,            0.02109544f,            0.01373637f,            0.14653027f,            0.02449317f,            0.01878268f,            0.01089148f,            0.36442387f,            0.01426089f,            0.02649262f,            0.00308395f,            0.51123023f,            0.00987128f,            0.02856500f,            0.01239803f,            0.65732223f,            0.00001665f,            0.00257388f,            0.02261361f,            0.00056261f,            0.08028404f,            0.00753943f,            0.00092872f,            0.22300763f,            0.00515121f,            0.00238470f,            0.00001802f,            0.00303019f,            0.00282769f,            0.93392336f,            0.00829813f,            0.00937593f,            0.00232166f,            0.00606702f,            0.03175319f,            0.00192149f,            0.89188498f,            0.01474108f,            0.03585867f,            0.00123343f,            0.00441551f,            0.00399710f,            0.00857630f,            0.01781271f,            0.01955875f,            0.00221238f,            0.00005268f,            0.00038176f,            0.00141851f,            0.07513693f,            0.00153898f,            0.00254140f,            0.04116146f,            0.00216117f,            0.87339473f,            0.17824675f,            0.04543359f,            0.01501061f,            0.03382575f,            0.09682461f,            0.29989448f,            0.02655865f,            0.16809541f,            0.09566309f,            0.04044705f,            0.00052125f,            0.00006512f,            0.00041621f,            0.03254773f,            0.00120942f,            0.00177929f,            0.00091721f,            0.95285058f,            0.00068729f,            0.00900588f,            0.04185560f,            0.00125587f,            0.33473280f,            0.00119652f,            0.00552071f,            0.03358750f,            0.04974457f,            0.00243473f,            0.41644078f,            0.11323092f,            0.00945223f,            0.00509389f,            0.04602458f,            0.02943204f,            0.23871920f,            0.06141117f,            0.05274383f,            0.03511769f,            0.09954999f,            0.42245534f,            0.00686926f,            0.01075546f,            0.49830484f,            0.37111449f,            0.00928881f,            0.00910977f,            0.00822666f,            0.00448587f,            0.04094843f,            0.04089646f,            0.00190534f,            0.00074783f,            0.02465805f,            0.02045769f,            0.02690129f,            0.00249506f,            0.00202899f,            0.84847659f,            0.01121813f,            0.06111111f,            0.00527403f,            0.00617689f,            0.00719898f,            0.17549324f,            0.25461593f,            0.15036304f,            0.04163047f,            0.01647436f,            0.08906800f,            0.25370511f,            0.10200825f,            0.03916828f,            0.22575049f,            0.08762794f,            0.06703069f,            0.01087492f,            0.27197123f,            0.15926389f,            0.02289790f,            0.01340644f,            0.00233572f,            0.00071111f,            0.01389953f,            0.00187034f,            0.89338356f,            0.00067592f,            0.00535080f,            0.02598928f,            0.01003115f,            0.04575264f,            0.00010197f,            0.00006095f,            0.00021980f,            0.99164659f,            0.00011408f,            0.00474983f,            0.00004892f,            0.00012496f,            0.00257160f,            0.00036128f,            0.91125363f,            0.00012225f,            0.02511939f,            0.00156989f,            0.00002669f,            0.03335980f,            0.01791442f,            0.00531134f,            0.00345027f,            0.00187230f,            0.00210833f,            0.00001888f,            0.00016036f,            0.00394190f,            0.00016232f,            0.00026980f,            0.00012382f,            0.99098623f,            0.00036967f,            0.00185874f,            0.99578768f,            0.00000018f,            0.00162244f,            0.00012927f,            0.00000136f,            0.00158810f,            0.00016544f,            0.00000476f,            0.00069853f,            0.00000226f,            0.19834445f,            0.00044551f,            0.40857196f,            0.34896207f,            0.00023418f,            0.00828141f,            0.02426279f,            0.00148875f,            0.00938030f,            0.00002860f,            0.00201644f,            0.06109568f,            0.01542680f,            0.05984236f,            0.00112191f,            0.00419699f,            0.00110061f,            0.28937989f,            0.13231210f,            0.43350723f,            0.00055382f,            0.92216444f,            0.00396460f,            0.01456171f,            0.00061405f,            0.00972675f,            0.00677260f,            0.00454273f,            0.02471014f,            0.01238921f,            0.00027888f,            0.02572848f,            0.00290584f,            0.00748292f,            0.08441166f,            0.00232722f,            0.00188305f,            0.81133318f,            0.01191756f,            0.05173124f,            0.00315098f,            0.00499059f,            0.00158580f,            0.92859417f,            0.00035086f,            0.04807130f,            0.00101955f,            0.00034313f,            0.01119398f,            0.00069962f,            0.00112821f,            0.00214349f,            0.03968662f,            0.00325992f,            0.00253143f,            0.00199443f,            0.00964058f,            0.90529889f,            0.00384289f,            0.03047365f,            0.00174196f,            0.06674320f,            0.00283191f,            0.09274873f,            0.01944309f,            0.03424436f,            0.00694406f,            0.07912937f,            0.15087396f,            0.54529935f,            0.00007096f,            0.00001000f,            0.00001498f,            0.00007066f,            0.00002792f,            0.00005677f,            0.00000490f,            0.99606401f,            0.00030978f,            0.00337013f,            0.00286575f,            0.00011636f,            0.00064778f,            0.00992065f,            0.04501861f,            0.03149971f,            0.00287679f,            0.37334359f,            0.00214695f,            0.53156382f,            0.00600238f,            0.00003215f,            0.02112119f,            0.00084685f,            0.00497269f,            0.00753993f,            0.95174772f,            0.00150877f,            0.00212018f,            0.00410815f,            0.00006566f,            0.00001179f,            0.99827027f,            0.00028396f,            0.00004237f,            0.00000550f,            0.00091406f,            0.00003423f,            0.00036640f,            0.00000567f,            0.00079063f,            0.00006855f,            0.00051338f,            0.00590454f,            0.00732460f,            0.00195139f,            0.00034534f,            0.90222436f,            0.00163695f,            0.07924022f,            0.00362202f,            0.01493629f,            0.01135249f,            0.00781013f,            0.05138498f,            0.22704794f,            0.00442778f,            0.00350683f,            0.59828150f,            0.07762999f,            0.00016529f,            0.00001219f,            0.00006521f,            0.00446292f,            0.94456083f,            0.00407963f,            0.00102245f,            0.00057420f,            0.00344479f,            0.04161252f,            0.00000981f,            0.00030270f,            0.00017082f,            0.00029943f,            0.00010159f,            0.00003605f,            0.00001875f,            0.99310946f,            0.00063157f,            0.00531995f,            0.01100852f,            0.00021492f,            0.00049603f,            0.59714299f,            0.00454595f,            0.33691072f,            0.03074775f,            0.00427598f,            0.00512297f,            0.00953417f,            0.00064403f,            0.00001687f,            0.00822414f,            0.00012918f,            0.02522905f,            0.00046274f,            0.95950085f,            0.00174588f,            0.00070707f,            0.00334025f,            0.00014754f,            0.96842438f,            0.00752080f,            0.00713038f,            0.00074491f,            0.00107368f,            0.00245372f,            0.00181830f,            0.00883226f,            0.00185409f,            0.00210863f,            0.00017522f,            0.00039881f,            0.98836052f,            0.00003650f,            0.00535216f,            0.00001887f,            0.00069545f,            0.00265663f,            0.00019714f,            0.00028919f,            0.00026057f,            0.00356666f,            0.00034738f,            0.00413719f,            0.00133701f,            0.98608136f,            0.00009625f,            0.00153734f,            0.00234698f,            0.01427079f,            0.04020482f,            0.04733688f,            0.03817881f,            0.16299380f,            0.04943828f,            0.03522370f,            0.05902825f,            0.23904003f,            0.31428465f,            0.00029359f,            0.00005619f,            0.00007707f,            0.98437482f,            0.00000957f,            0.00828004f,            0.00002787f,            0.00510217f,            0.00087425f,            0.00090444f,            0.00011413f,            0.83918202f,            0.01017746f,            0.03100164f,            0.00308035f,            0.01615586f,            0.02608237f,            0.00337026f,            0.05493741f,            0.01589854f,            0.00053240f,            0.00144792f,            0.00108170f,            0.00027300f,            0.86477506f,            0.00072790f,            0.01062538f,            0.00428096f,            0.00233054f,            0.11392505f,            0.00411633f,            0.33660546f,            0.01735369f,            0.18114267f,            0.03090077f,            0.11699959f,            0.03416851f,            0.06780743f,            0.07481573f,            0.13608985f,            0.00073468f,            0.20941530f,            0.01012138f,            0.17237675f,            0.01661461f,            0.02184150f,            0.03694551f,            0.30870155f,            0.04255475f,            0.18069389f,            0.06343270f,            0.00037455f,            0.06623310f,            0.00041474f,            0.00209181f,            0.04566626f,            0.81232506f,            0.00054500f,            0.00807252f,            0.00084416f,            0.00008067f,            0.00003926f,            0.00225794f,            0.00115743f,            0.01925980f,            0.00010427f,            0.00062067f,            0.02234522f,            0.00210706f,            0.95202768f});
    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/mnist_00.fb");
    //graph->printOut();

    auto result = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), result);

    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(6));

    auto z = graph->getVariableSpace()->getVariable(6,0)->getNDArray();

    ASSERT_EQ(e, *z);

    delete graph;
}



TEST_F(FlatBuffersTest, Test_MNIST_1) {
    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/mnist.fb");
    //graph->printOut();

    auto result = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), result);

    delete graph;
}

/*
// FIXME: uncomment this test once conv_0 fb reexported
TEST_F(FlatBuffersTest, nhwc_conv_0) {
    nd4j::ops::rank<float> op1;

    auto exp('c', {4, 2}, {2.958640f, 0.602521f, 7.571267f, 1.496686f, -2.292647f, -1.791460f, 13.055838f, 4.278642f});

    auto graph = GraphExecutioner<float>::importFromFlatBuffers("./resources/conv_0.fb");

    graph->printOut();

    auto result = GraphExecutioner<float>::execute(graph);
    ASSERT_EQ(ND4J_STATUS_OK, result);

    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(11));

    auto z = graph->getVariableSpace()->getVariable(11)->getNDArray();

    //z->printShapeInfo("z buffr");
    //z->printIndexedBuffer("z shape");

//    [[2.96,  0.60],
//    [7.57,  1.50],
//    [-2.29,  -1.79],
//    [13.06,  4.28]]

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete graph;
}

*/


/*
TEST_F(FlatBuffersTest, ReadLoops_SimpleWhile_1) {
    // TF graph:
    // https://gist.github.com/raver119/2aa49daf7ec09ed4ddddbc6262f213a0
    auto graph = GraphExecutioner<float>::importFromFlatBuffers("./resources/simple_while.fb");

    ASSERT_TRUE(graph != nullptr);

    Nd4jStatus status = GraphExecutioner<float>::execute(graph);

    ASSERT_EQ(ND4J_STATUS_OK, status);

    delete graph;
}

 */