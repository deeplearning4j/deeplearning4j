//
// Created by raver119 on 19.09.17.
//

#ifndef LIBND4J_CYCLICTESTS_H
#define LIBND4J_CYCLICTESTS_H

#include "testlayers.h"
#include <NDArray.h>
#include <Block.h>
#include <Node.h>
#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <ops/declarable/DeclarableOp.h>
#include <ops/declarable/generic/convo/conv2d.cpp>

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


#endif //LIBND4J_CYCLICTESTS_H
