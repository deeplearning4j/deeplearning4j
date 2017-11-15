//
// Created by raver119 on 19.09.17.
//

#ifndef LIBND4J_CYCLICTESTS_H
#define LIBND4J_CYCLICTESTS_H

#include "testlayers.h"
#include <NDArray.h>
#include <Context.h>
#include <Node.h>
#include <memory/MemoryRegistrator.h>
#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/convo/conv2d.cpp>

using namespace nd4j;
using namespace nd4j::graph;

class CyclicTests : public testing::Test {
public:
    int numLoops = 100000000;

    int extLoops = 1000;
    int intLoops = 1000;
};

// simple test for NDArray leaks
TEST_F(CyclicTests, TestNDArray1) {

    for (int i = 0; i < numLoops; i++) {
        auto arr = new NDArray<float>(1000, 1000, 'c');
        arr->assign(1.0);
        delete arr;
    }
}

TEST_F(CyclicTests, TestNDArrayWorkspace1) {

    for (int e = 0; e < extLoops; e++) {
        nd4j::memory::Workspace workspace(10000000);
        nd4j::memory::MemoryRegistrator::getInstance()->attachWorkspace(&workspace);

        for (int i = 0; i < intLoops; i++) {
            workspace.scopeIn();
            
            auto arr = new NDArray<float>(1000, 1000, 'c');
            
            delete arr;
            
            workspace.scopeOut();
        }

        if (e % 50 == 0)
            nd4j_printf("%i ext cycles passed\n", e);
    }
}


TEST_F(CyclicTests, TestGraphInstantiation1) {

    for (int e = 0; e < numLoops; e++) {
        auto graph = GraphExecutioner<float>::importFromFlatBuffers("../../../tests_cpu/resources/adam_sum.fb");

        ASSERT_FALSE(graph == nullptr);
        ASSERT_EQ(2, graph->totalNodes());
        ASSERT_EQ(1, graph->rootNodes());
        //Nd4jStatus status = GraphExecutioner<float>::execute();
        //ASSERT_EQ(ND4J_STATUS_OK, status);

        delete graph;
    }
}


TEST_F(CyclicTests, TestGraphExecution1) {

    for (int e = 0; e < numLoops; e++) {
        auto graph = GraphExecutioner<float>::importFromFlatBuffers("../../../tests_cpu/resources/adam_sum.fb");

        ASSERT_FALSE(graph == nullptr);
        ASSERT_EQ(2, graph->totalNodes());
        ASSERT_EQ(1, graph->rootNodes());
        //Nd4jStatus status = GraphExecutioner<float>::execute();
        //ASSERT_EQ(ND4J_STATUS_OK, status);

        graph->estimateRequiredMemory();

        GraphExecutioner<float>::execute(graph);

        ASSERT_EQ(1, graph->getVariableSpace()->getVariable(2)->getNDArray()->lengthOf());
        ASSERT_NEAR(8.0f, graph->getVariableSpace()->getVariable(2)->getNDArray()->getScalar(0), 1e-5);

        delete graph;
    }
}


TEST_F(CyclicTests, TestCustomOpExecution1) {


    for (int e = 0; e < numLoops; e++) {
        auto input = new NDArray<float>('c', {4, 2, 1, 11, 11});
        input->assign(451.0);

        auto output = new NDArray<float>('c', {4, 2, 1, 10, 10});


        std::pair<int, int> pair0(1,0);
        std::pair<int, int> pair1(1,1);


        VariableSpace<float>* variableSpace = new VariableSpace<float>();
        variableSpace->putVariable(-1, input);


        variableSpace->putVariable(pair0, output);


        Block<float>* block = new Block<float>(1, variableSpace, false);  // not-in-place
        block->fillInputs({-1});

        // kernel params
        block->getIArguments()->push_back(1);
        block->getIArguments()->push_back(2);
        block->getIArguments()->push_back(2);

        // stride
        block->getIArguments()->push_back(1);
        block->getIArguments()->push_back(1);
        block->getIArguments()->push_back(1);

        // padding
        block->getIArguments()->push_back(0);
        block->getIArguments()->push_back(0);
        block->getIArguments()->push_back(0);

        // ceiling
        block->getIArguments()->push_back(1);

        // padding count
        block->getIArguments()->push_back(1);


        nd4j::ops::avgpool3d<float> avgpool3d;

        Nd4jStatus result = avgpool3d.execute(block);
        ASSERT_EQ(ND4J_STATUS_OK, result);

        ASSERT_NEAR(451.0f, output->template reduceNumber<simdOps::Mean<float>>(), 1e-5);

        delete block;
        delete variableSpace;

        //delete input;
        //delete output;
    }
}


TEST_F(CyclicTests, Test_LegacyOp_1) {
    for (int e = 0; e < numLoops; e++) {
        Graph<float> graph;

        auto node = new Node<float>(OpType_TRANSFORM, 10, 1);

        graph.addNode(node);

        graph.buildGraph();
    }
}

TEST_F(CyclicTests, Test_ArrayList_1) {
    for (int e = 0; e < numLoops; e++) {

        NDArray<double> matrix('c', {3, 2});
        NDArrayFactory<double>::linspace(1, matrix);

        nd4j::ops::create_list<double> op;

        auto result = op.execute(nullptr, {&matrix}, {}, {1, 1});

        ASSERT_EQ(ND4J_STATUS_OK, result->status());

        // we return flow as well
        ASSERT_EQ(1, result->size());

        delete result;
    }
}

TEST_F(CyclicTests, Test_ArrayList_2) {

    for (int e = 0; e < numLoops; e++) {
        NDArrayList<double> list(10);
        NDArray<double> exp('c', {1, 100});
        exp.assign(4.0f);

        for (int e = 0; e < 10; e++) {
            auto row = new NDArray<double>('c', {1, 100});
            row->assign((double) e);
            list.write(e, row->dup());

            delete row;
        }

        nd4j::ops::read_list<double> op;

        auto result = op.execute(&list, {}, {}, {4});

        ASSERT_EQ(ND4J_STATUS_OK, result->status());

        auto z = result->at(0);

        ASSERT_TRUE(exp.isSameShape(z));
        ASSERT_TRUE(exp.equalsTo(z));

        delete result;
    }
}

TEST_F(CyclicTests, Test_ArrayList_9) {

    for (int e = 0; e < numLoops; e++) {

        float _expB[] = {1.000000, 1.000000, 2.000000, 2.000000, 3.000000, 3.000000};
        NDArray<float> exp('c', {3, 2});
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
}

TEST_F(CyclicTests, Test_ArrayList_10) {


    for (int e = 0; e < numLoops; e++) {

        NDArray<float> exp('c', {5, 2}, {3., 6., 9., 12., 15., 18., 21., 24., 27., 30.});
        auto graph = GraphExecutioner<float>::importFromFlatBuffers("./resources/tensor_array_loop.fb");

        ASSERT_TRUE(graph != nullptr);

        Nd4jStatus status = GraphExecutioner<float>::execute(graph);

        ASSERT_EQ(ND4J_STATUS_OK, status);

        auto variableSpace = graph->getVariableSpace();

        ASSERT_TRUE(variableSpace->hasVariable(23,0));

        auto z = variableSpace->getVariable(23)->getNDArray();

        ASSERT_TRUE(exp.isSameShape(z));
        ASSERT_TRUE(exp.equalsTo(z));

        delete graph;

    }
}


#endif //LIBND4J_CYCLICTESTS_H
