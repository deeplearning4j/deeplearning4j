//
// Created by raver119 on 21.02.18.
//

#include "testlayers.h"
#include <NDArray.h>
#include <graph/Variable.h>
#include <flatbuffers/flatbuffers.h>
#include <NDArrayFactory.h>
#include <ops/declarable/headers/broadcastable.h>

using namespace nd4j;
using namespace nd4j::graph;

class NodeTests : public testing::Test {
public:

};

TEST_F(NodeTests, Test_Dtype_Conversion_1) {
    auto nodeA = new Node<float>(OpType_TRANSFORM, 0, 1, {-1}, {2});

    auto nd = nodeA->asT<double>();
    auto nf = nd->asT<float>();

    ASSERT_EQ(nodeA->id(), nf->id());
    ASSERT_EQ(*nodeA->name(), *nf->name());
    ASSERT_EQ(nodeA->getOpClass(), nf->getOpClass());
    ASSERT_EQ(nodeA->opType(), nf->opType());
    ASSERT_EQ(nodeA->opNum(), nf->opNum());

    delete nodeA;
    delete nd;
    delete nf;
}


TEST_F(NodeTests, Test_Dtype_Conversion_2) {
    nd4j::ops::add<float> op;

    auto nodeA = new Node<float>(OpType_CUSTOM, 0, 1, {-1}, {2});
    nodeA->setCustomOp(&op);

    auto nd = nodeA->asT<double>();
    auto nf = nd->asT<float>();

    ASSERT_EQ(nodeA->id(), nf->id());
    ASSERT_EQ(*nodeA->name(), *nf->name());
//    ASSERT_EQ(nodeA->getOpClass(), nf->getOpClass());
    ASSERT_EQ(nodeA->opType(), nf->opType());
    ASSERT_EQ(nodeA->opNum(), nf->opNum());
    ASSERT_EQ(nodeA->getCustomOp()->getOpHash(), nf->getCustomOp()->getOpHash());

    delete nodeA;
    delete nd;
    delete nf;
}