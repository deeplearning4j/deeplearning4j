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

#include <NativeOps.h>
#include <NDArray.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/OpRegistrator.h>
#include <graph/GraphHolder.h>
#include <graph/FlatUtils.h>
#include "testlayers.h"
#include <array>

using namespace nd4j;
using namespace nd4j::ops;

class JavaInteropTests : public testing::Test {
public:

};


TEST_F(JavaInteropTests, TestShapeExposure1) {
    NDArray<float> input('c', {1, 2, 5, 4});
    NDArray<float> weights('c', {2, 2, 2, 3});
    NDArray<float> exp('c', {1, 3, 5, 4});


    NativeOps nativeOps;

    nd4j::ops::conv2d<float> op;

    std::vector<float> tArgs({});
    std::vector<Nd4jLong> iArgs({2, 2, 1, 1, 0, 0, 1, 1, 1});

    Nd4jPointer ptrs[] = {(Nd4jPointer) input.getShapeInfo(), (Nd4jPointer) weights.getShapeInfo()};

    auto shapeList = nativeOps.calculateOutputShapesFloat(nullptr, op.getOpHash(), ptrs, 2, tArgs.data(), tArgs.size(), iArgs.data(), iArgs.size());

    ASSERT_EQ(1, shapeList->size());

    ASSERT_EQ(exp.rankOf(), shape::rank((Nd4jLong *)shapeList->at(0)));
    ASSERT_EQ(exp.sizeAt(0), shape::shapeOf((Nd4jLong *)shapeList->at(0))[0]);
    ASSERT_EQ(exp.sizeAt(1), shape::shapeOf((Nd4jLong *)shapeList->at(0))[1]);
    ASSERT_EQ(exp.sizeAt(2), shape::shapeOf((Nd4jLong *)shapeList->at(0))[2]);
    ASSERT_EQ(exp.sizeAt(3), shape::shapeOf((Nd4jLong *)shapeList->at(0))[3]);

    //int *ptr = (int *) shapeList[0];
    //delete[] ptr;
    //delete shapeList;

    nativeOps.deleteShapeList((Nd4jPointer) shapeList);
}


TEST_F(JavaInteropTests, TestShapeExposure2) {
    NDArray<float> input('c', {1, 2, 5, 4});
    NDArray<float> exp('c', {4}, {1, 2, 5, 4});


    NativeOps nativeOps;

    nd4j::ops::shape_of<float> op;

    std::vector<float> tArgs({});
    std::vector<Nd4jLong> iArgs({});


    Nd4jPointer ptrs[] = {(Nd4jPointer) input.getShapeInfo()};

    auto shapeList = nativeOps.calculateOutputShapesFloat(nullptr, op.getOpHash(), ptrs, 1, tArgs.data(), tArgs.size(), iArgs.data(), iArgs.size());

    ASSERT_EQ(1, shapeList->size());

    ASSERT_EQ(exp.rankOf(), shape::rank((Nd4jLong *)shapeList->at(0)));
    ASSERT_EQ(exp.sizeAt(0), shape::shapeOf((Nd4jLong *)shapeList->at(0))[0]);

    nativeOps.deleteShapeList((Nd4jPointer) shapeList);
}

TEST_F(JavaInteropTests, TestShapeExposure3) {
    NDArray<float> x('c', {5, 30});
    NDArray<float> sizes('c', {3}, {4, 15, 11});

    IndicesList list0({NDIndex::all(), NDIndex::interval(0, 4)});
    IndicesList list1({NDIndex::all(), NDIndex::interval(4, 19)});
    IndicesList list2({NDIndex::all(), NDIndex::interval(19, 30)});

    auto sub0 = x.subarray(list0);
    auto sub1 = x.subarray(list1);
    auto sub2 = x.subarray(list2);

    sub0->assign(0.0f);
    sub1->assign(1.0f);
    sub2->assign(2.0f);

    Nd4jPointer inputBuffers[] = {x.buffer(), sizes.buffer()};
    Nd4jPointer inputShapes[] = {x.shapeInfo(), sizes.shapeInfo()};

    NativeOps nativeOps;
    nd4j::ops::split_v<float> op;
    
    Nd4jLong iArgs[] = {1};
    auto hash = op.getOpHash();

    auto shapeList = nativeOps.calculateOutputShapesFloat(nullptr, hash, inputBuffers, inputShapes, 2, nullptr, 0, iArgs, 1);

    ASSERT_EQ(3, shapeList->size());

    ASSERT_TRUE(shape::equalsSoft(sub0->shapeInfo(), shapeList->at(0)));
    ASSERT_TRUE(shape::equalsSoft(sub1->shapeInfo(), shapeList->at(1)));
    ASSERT_TRUE(shape::equalsSoft(sub2->shapeInfo(), shapeList->at(2)));

    delete sub0;
    delete sub1;
    delete sub2;

    nativeOps.deleteShapeList((Nd4jPointer) shapeList);
}


TEST_F(JavaInteropTests, TestSconv2d_1) {
    NDArray<float> input('c', {3, 3, 8, 8});
    NDArray<float> weightsD('c', {1, 3, 1, 1});
    NDArray<float> weightsP('c', {2, 3, 1, 1});
    NDArray<float> bias('c', {1, 2});
    NDArray<float> output('c', {3, 2, 8, 8});
    output.assign(0.0);

    input.linspace(1);
    weightsD.linspace(1);
    weightsP.linspace(1);
    bias.linspace(1);
    weightsD.permutei({2,3,1,0});
    weightsP.permutei({2,3,1,0});

    NDArray<float> expOutput('c', {3, 2, 8, 8});

    nd4j::ops::sconv2d<float> op;


    Nd4jPointer ptrsInBuffer[] = {(Nd4jPointer) input.getBuffer(), (Nd4jPointer) weightsD.getBuffer(), (Nd4jPointer) weightsP.getBuffer(), (Nd4jPointer) bias.getBuffer()};
    Nd4jPointer ptrsInShapes[] = {(Nd4jPointer) input.getShapeInfo(), (Nd4jPointer) weightsD.getShapeInfo(), (Nd4jPointer) weightsP.getShapeInfo(), (Nd4jPointer) bias.getShapeInfo()};


    Nd4jPointer ptrsOutBuffers[] = {(Nd4jPointer) output.getBuffer()};
    Nd4jPointer ptrsOutShapes[] = {(Nd4jPointer) output.getShapeInfo()};

    NativeOps nativeOps;

    Nd4jLong exp[] = {1, 1, 1, 1, 0, 0, 1, 1, 0, 0};

    nativeOps.execCustomOpFloat(nullptr, op.getOpHash(), ptrsInBuffer, ptrsInShapes, 4, ptrsOutBuffers, ptrsOutShapes, 1,
                                nullptr, 0, exp, 9, false);

    //output.printBuffer("output");

    ASSERT_NEAR(1423, output.getScalar(0), 1e-5);
        //nd4j_printf("Iter %i passed...\n", e);
}

TEST_F(JavaInteropTests, TestSconv2d_2) {
    NDArray<float> input('c', {3, 3, 8, 8});
    NDArray<float> weightsD('c', {1, 3, 1, 1});
    NDArray<float> output('c', {3, 3, 8, 8});
    output.assign(0.0);

    input.linspace(1);
    weightsD.linspace(1);
    weightsD.permutei({2,3,1,0});    

    NDArray<float> expOutput('c', {3, 3, 8, 8});

    nd4j::ops::sconv2d<float> op;


    Nd4jPointer ptrsInBuffer[] = {(Nd4jPointer) input.getBuffer(), (Nd4jPointer) weightsD.getBuffer()};
    Nd4jPointer ptrsInShapes[] = {(Nd4jPointer) input.getShapeInfo(), (Nd4jPointer) weightsD.getShapeInfo()};


    Nd4jPointer ptrsOutBuffers[] = {(Nd4jPointer) output.getBuffer()};
    Nd4jPointer ptrsOutShapes[] = {(Nd4jPointer) output.getShapeInfo()};

    NativeOps nativeOps;

    Nd4jLong exp[] = {1, 1, 1, 1, 0, 0, 1, 1, 0};

    nativeOps.execCustomOpFloat(nullptr, op.getOpHash(), ptrsInBuffer, ptrsInShapes, 2, ptrsOutBuffers, ptrsOutShapes, 1, nullptr, 0, exp, 9, false);

    //output.printBuffer("output");

    ASSERT_NEAR(1, output.getScalar(0), 1e-5);
}


TEST_F(JavaInteropTests, TestPooling2d_1) {
    NDArray<float> input('c', {1, 2, 4, 5});
    NDArray<float> output('c', {1, 2, 4, 5});


    Nd4jPointer ptrsInBuffer[] = {(Nd4jPointer) input.getBuffer()};
    Nd4jPointer ptrsInShapes[] = {(Nd4jPointer) input.getShapeInfo()};

    Nd4jPointer ptrsOutBuffers[] = {(Nd4jPointer) output.getBuffer()};
    Nd4jPointer ptrsOutShapes[] = {(Nd4jPointer) output.getShapeInfo()};

    std::vector<Nd4jLong> iArgs({2, 2, 1, 1, 0, 0, 1, 1, 1 , 1, 1});

    nd4j::ops::pooling2d<float> op;

    NativeOps nativeOps;

    Nd4jStatus status = nativeOps.execCustomOpFloat(nullptr, op.getOpHash(), ptrsInBuffer, ptrsInShapes, 1, ptrsOutBuffers, ptrsOutShapes, 1, nullptr, 0, iArgs.data(), 11, false);
    ASSERT_EQ(ND4J_STATUS_OK, status);

}


TEST_F(JavaInteropTests, TestMaxPooling2d_1) {
    NDArray<float> input('c', {1, 2, 4, 5});
    NDArray<float> output('c', {1, 2, 4, 5});
    input.linspace(1);


    Nd4jPointer ptrsInBuffer[] = {(Nd4jPointer) input.getBuffer()};
    Nd4jPointer ptrsInShapes[] = {(Nd4jPointer) input.getShapeInfo()};

    Nd4jPointer ptrsOutBuffers[] = {(Nd4jPointer) output.getBuffer()};
    Nd4jPointer ptrsOutShapes[] = {(Nd4jPointer) output.getShapeInfo()};

    std::vector<Nd4jLong> iArgs({2, 2, 1, 1, 0, 0, 1, 1, 1});

    nd4j::ops::maxpool2d<float> op;

    NativeOps nativeOps;

    Nd4jStatus status = nativeOps.execCustomOpFloat(nullptr, op.getOpHash(), ptrsInBuffer, ptrsInShapes, 1, ptrsOutBuffers, ptrsOutShapes, 1, nullptr, 0, iArgs.data(), 9, false);
    ASSERT_EQ(ND4J_STATUS_OK, status);

}
TEST_F(JavaInteropTests, TestCol2Im_1) {
    /*
        o.d.n.l.c.ConvolutionLayer - eps shape: [6, 1, 2, 2, 2, 4, 5, 160, 4, 2, 1, 40, 8, 0, -1, 99]
        o.d.n.l.c.ConvolutionLayer - epsNext shape: [4, 1, 2, 4, 5, 20, 20, 5, 1, 0, 1, 99]
        o.d.n.l.c.ConvolutionLayer - Strides: [1, 1]
        o.d.n.l.c.ConvolutionLayer - Padding: [0, 0]
        o.d.n.l.c.ConvolutionLayer - Input: [4,5]
        o.d.n.l.c.ConvolutionLayer - Dilation: [1, 1]
     */
    NDArray<float> input('c', {1, 2, 2, 2, 4, 5});
    NDArray<float> output('c', {1, 2, 4, 5});
    input.linspace(1);

    Nd4jPointer ptrsInBuffer[] = {(Nd4jPointer) input.getBuffer()};
    Nd4jPointer ptrsInShapes[] = {(Nd4jPointer) input.getShapeInfo()};


    Nd4jPointer ptrsOutBuffers[] = {(Nd4jPointer) output.getBuffer()};
    Nd4jPointer ptrsOutShapes[] = {(Nd4jPointer) output.getShapeInfo()};

    nd4j::ops::col2im<float> op;

    NativeOps nativeOps;

    Nd4jLong exp[] = {1, 1, 1, 1, 4, 5, 1, 1, 1};

    nativeOps.execCustomOpFloat(nullptr, op.getOpHash(), ptrsInBuffer, ptrsInShapes, 1, ptrsOutBuffers, ptrsOutShapes, 1, nullptr, 0, exp, 9, false);

    ASSERT_TRUE(output.meanNumber() > 0.0);
}

TEST_F(JavaInteropTests, TestPNorm_1) {
    /*
        o.d.n.l.c.s.SubsamplingLayer - input: [4, 1, 3, 4, 4, 16, 16, 4, 1, 0, 1, 99]
        o.d.n.l.c.s.SubsamplingLayer - output: [4, 1, 3, 3, 3, 27, 9, 3, 1, 0, 1, 99]
        o.d.n.l.c.s.SubsamplingLayer - Kernel: [2, 2]
        o.d.n.l.c.s.SubsamplingLayer - Strides: [1, 1]
        o.d.n.l.c.s.SubsamplingLayer - Pad: [0, 0]
        o.d.n.l.c.s.SubsamplingLayer - Dilation: [1, 1]
        o.d.n.l.c.s.SubsamplingLayer - Same: false
        o.d.n.l.c.s.SubsamplingLayer - pnorm: 2
     */
    NDArray<float> input('c', {1, 3, 4, 4});
    NDArray<float> output('c', {1, 3, 3, 3});
    input.linspace(1);

    NativeOps nativeOps;

    nd4j::ops::pnormpool2d<float> op;

    Nd4jLong exp[] = {2, 2, 1, 1, 0, 0, 1, 1, 0, 2, 0, 0};

    Nd4jPointer ptrsInBuffer[] = {(Nd4jPointer) input.getBuffer()};
    Nd4jPointer ptrsInShapes[] = {(Nd4jPointer) input.getShapeInfo()};

    Nd4jPointer ptrsOutBuffers[] = {(Nd4jPointer) output.getBuffer()};
    Nd4jPointer ptrsOutShapes[] = {(Nd4jPointer) output.getShapeInfo()};

    nativeOps.execCustomOpFloat(nullptr, op.getOpHash(), ptrsInBuffer, ptrsInShapes, 1, ptrsOutBuffers, ptrsOutShapes, 1, nullptr, 0, exp, 11, false);

    ASSERT_TRUE(output.meanNumber() > 0.0);
}


TEST_F(JavaInteropTests, TestInplace_1) {
    NDArray<float> input('c', {10, 10});
    //NDArray<float> exp('c', {10, 10});
    input.linspace(1);

    NativeOps nativeOps;

    nd4j::ops::clipbyvalue<float> op;

    float extras[] = {-1.0f, 1.0f};

    Nd4jPointer ptrsInBuffer[] = {(Nd4jPointer) input.getBuffer()};
    Nd4jPointer ptrsInShapes[] = {(Nd4jPointer) input.getShapeInfo()};


    Nd4jStatus result = nativeOps.execCustomOpFloat(nullptr, op.getOpHash(), ptrsInBuffer, ptrsInShapes, 1, nullptr, nullptr, 0, extras, 2, nullptr, 0, true);

    ASSERT_EQ(ND4J_STATUS_OK, result);

    ASSERT_NEAR(1.0, input.meanNumber(), 1e-5);
}

TEST_F(JavaInteropTests, Test_Synonyms_1) {
    auto op = OpRegistrator::getInstance()->getOperationHalf("RDiv");
    auto opRef = OpRegistrator::getInstance()->getOperationHalf("reversedivide");
    std::string nameExp("reversedivide");

    ASSERT_TRUE(op != nullptr);
    ASSERT_TRUE(opRef != nullptr);

    std::string name = *(op->getOpName());
    std::string nameRef = *(opRef->getOpName());

    ASSERT_EQ(nameExp, nameRef);
    ASSERT_EQ(nameRef, name);
}

TEST_F(JavaInteropTests, Test_Synonyms_2) {
    auto op = OpRegistrator::getInstance()->getOperationFloat("RDiv");
    auto opRef = OpRegistrator::getInstance()->getOperationFloat("reversedivide");
    std::string nameExp("reversedivide");

    ASSERT_TRUE(op != nullptr);
    ASSERT_TRUE(opRef != nullptr);

    std::string name = *(op->getOpName());
    std::string nameRef = *(opRef->getOpName());

    ASSERT_EQ(nameExp, nameRef);
    ASSERT_EQ(nameRef, name);
}

TEST_F(JavaInteropTests, Test_Synonyms_3) {
    auto op = OpRegistrator::getInstance()->getOperationDouble("RDiv");
    auto opRef = OpRegistrator::getInstance()->getOperationDouble("reversedivide");
    std::string nameExp("reversedivide");

    ASSERT_TRUE(op != nullptr);
    ASSERT_TRUE(opRef != nullptr);

    std::string name = *(op->getOpName());
    std::string nameRef = *(opRef->getOpName());

    ASSERT_EQ(nameExp, nameRef);
    ASSERT_EQ(nameRef, name);
}

TEST_F(JavaInteropTests, Test_GraphReuse_1) {
    NativeOps nativeOps;

    uint8_t* data = nd4j::graph::readFlatBuffers("./resources/reduce_dim_false.fb");

    nativeOps.registerGraphFloat(nullptr, 119, (Nd4jPointer) data);

    ASSERT_TRUE(GraphHolder::getInstance()->hasGraph<float>(119));

    nativeOps.unregisterGraph(nullptr, 119);

    ASSERT_FALSE(GraphHolder::getInstance()->hasGraph<float>(119));


    delete[] data;
}

TEST_F(JavaInteropTests, Test_GraphReuse_2) {
    //Environment::getInstance()->setDebug(true);
    //Environment::getInstance()->setVerbose(true);

    NDArray<float> exp0('c', {3}, {3, 3, 3});
    NDArray<float> exp1('c', {3}, {6, 6, 6});
    NDArray<float> exp2('c', {3}, {9, 9, 9});

    NativeOps nativeOps;

    // we load graph from file, because we're not in java here, and dont have buffer ready
    uint8_t* data = nd4j::graph::readFlatBuffers("./resources/reduce_dim_false.fb");

    // we ensure that there's no such a graph stored earlier
    ASSERT_FALSE(GraphHolder::getInstance()->hasGraph<float>(119));

    // register the graph, to call for it later
    nativeOps.registerGraphFloat(nullptr, 119, (Nd4jPointer) data);

    // and ensure we're ok
    ASSERT_TRUE(GraphHolder::getInstance()->hasGraph<float>(119));



    // run stuff

    NDArray<float> input_0('c', {3, 3});
    input_0.assign(1.0f);

    int idx[] = {1};

    Nd4jPointer inputs_0[] = {(Nd4jPointer) input_0.buffer()};
    Nd4jPointer shapes_0[] = {(Nd4jPointer) input_0.shapeInfo()};

    // now we're executing stored graph and providing replacement for input variable
    auto res_0 = nativeOps.executeStoredGraphFloat(nullptr, 119, inputs_0, shapes_0, idx, 1);
    ASSERT_EQ(ND4J_STATUS_OK, res_0->status());
    ASSERT_EQ(1, res_0->size());

    auto z0 = res_0->at(0)->getNDArray();
    ASSERT_TRUE(exp0.isSameShape(z0));


    NDArray<float> input_1('c', {3, 3});
    input_1.assign(2.0f);

    Nd4jPointer inputs_1[] = {(Nd4jPointer) input_1.buffer()};
    Nd4jPointer shapes_1[] = {(Nd4jPointer) input_1.shapeInfo()};

    // doing it again
    auto res_1 = nativeOps.executeStoredGraphFloat(nullptr, 119, inputs_1, shapes_1, idx, 1);
    ASSERT_EQ(ND4J_STATUS_OK, res_1->status());
    ASSERT_EQ(1, res_1->size());

    auto z1 = res_1->at(0)->getNDArray();
    ASSERT_TRUE(exp1.isSameShape(z1));


    NDArray<float> input_2('c', {3, 3});
    input_2.assign(3.0f);

    Nd4jPointer inputs_2[] = {(Nd4jPointer) input_2.buffer()};
    Nd4jPointer shapes_2[] = {(Nd4jPointer) input_2.shapeInfo()};

    // and again
    auto res_2 = nativeOps.executeStoredGraphFloat(nullptr, 119, inputs_2, shapes_2, idx, 1);
    ASSERT_EQ(ND4J_STATUS_OK, res_1->status());
    ASSERT_EQ(1, res_2->size());

    auto z2 = res_2->at(0)->getNDArray();
    ASSERT_TRUE(exp2.isSameShape(z2));


    //////// clean out
    nativeOps.unregisterGraph(nullptr, 119);

    ASSERT_FALSE(GraphHolder::getInstance()->hasGraph<float>(119));


    delete[] data;
    delete res_0;
    delete res_1;
    delete res_2;
}


TEST_F(JavaInteropTests, Test_Greater_1) {
    NDArray<float> x('c', {2, 2}, {1, 2, 1, 2});
    NDArray<float> y('c', {2, 2}, {1, 2, 0, 0});
    NDArray<float> o('c', {2, 2}, {3, 3, 3, 3});

    NDArray<float> exp('c', {2, 2}, {0, 0, 1, 1});

    nd4j::ops::greater<float> op;


    Nd4jPointer ptrsInBuffer[] = {(Nd4jPointer) x.getBuffer(), (Nd4jPointer) y.getBuffer()};
    Nd4jPointer ptrsInShapes[] = {(Nd4jPointer) x.getShapeInfo(), (Nd4jPointer) y.getShapeInfo()};


    Nd4jPointer ptrsOutBuffers[] = {(Nd4jPointer) o.getBuffer()};
    Nd4jPointer ptrsOutShapes[] = {(Nd4jPointer) o.getShapeInfo()};

    NativeOps nativeOps;

    nativeOps.execCustomOpFloat(nullptr, op.getOpHash(), ptrsInBuffer, ptrsInShapes, 2, ptrsOutBuffers, ptrsOutShapes, 1, nullptr, 0, nullptr, 0, false);

    ASSERT_TRUE(exp.equalsTo(&o));
}


TEST_F(JavaInteropTests, Test_Greater_2) {
    NDArray<double> x('c', {2, 2}, {1, 2, 1, 2});
    NDArray<double> y('c', {2, 2}, {1, 2, 0, 0});
    NDArray<double> o('c', {2, 2}, {3, 3, 3, 3});

    NDArray<double> exp('c', {2, 2}, {0, 0, 1, 1});

    nd4j::ops::greater<double> op;


    Nd4jPointer ptrsInBuffer[] = {(Nd4jPointer) x.getBuffer(), (Nd4jPointer) y.getBuffer()};
    Nd4jPointer ptrsInShapes[] = {(Nd4jPointer) x.getShapeInfo(), (Nd4jPointer) y.getShapeInfo()};


    Nd4jPointer ptrsOutBuffers[] = {(Nd4jPointer) o.getBuffer()};
    Nd4jPointer ptrsOutShapes[] = {(Nd4jPointer) o.getShapeInfo()};

    NativeOps nativeOps;

    nativeOps.execCustomOpDouble(nullptr, op.getOpHash(), ptrsInBuffer, ptrsInShapes, 2, ptrsOutBuffers, ptrsOutShapes, 1, nullptr, 0, nullptr, 0, false);

    ASSERT_TRUE(exp.equalsTo(&o));
}

TEST_F(JavaInteropTests, Test_Boolean_Op_1) {
    nd4j::ops::is_non_decreasing<float> op;

    NDArray<float> x('c', {5}, {1, 2, 3, 4, 5});
    NDArray<float> o(2.0f);
    NDArray<float> exp(1.0f);

    Nd4jPointer ptrsInBuffer[] = {(Nd4jPointer) x.getBuffer()};
    Nd4jPointer ptrsInShapes[] = {(Nd4jPointer) x.getShapeInfo()};


    Nd4jPointer ptrsOutBuffers[] = {(Nd4jPointer) o.getBuffer()};
    Nd4jPointer ptrsOutShapes[] = {(Nd4jPointer) o.getShapeInfo()};

    NativeOps nativeOps;
    auto hash = op.getOpHash();
    auto status = nativeOps.execCustomOpFloat(nullptr, hash, ptrsInBuffer, ptrsInShapes, 1, ptrsOutBuffers, ptrsOutShapes, 1, nullptr, 0, nullptr, 0, false);
    ASSERT_EQ(Status::OK(), status);

    ASSERT_TRUE(exp.equalsTo(&o));
}


TEST_F(JavaInteropTests, Test_Inplace_Outputs_1) {
    NDArray<float> x('c', {2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    NDArray<float> exp('c', {2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    NDArray<float> z('c', {2, 3});

    nd4j::ops::test_output_reshape<float> op;
    
    Nd4jPointer ptrsInBuffer[] = {(Nd4jPointer) x.getBuffer()};
    Nd4jPointer ptrsInShapes[] = {(Nd4jPointer) x.getShapeInfo()};


    Nd4jPointer ptrsOutBuffers[] = {(Nd4jPointer) z.getBuffer()};
    Nd4jPointer ptrsOutShapes[] = {(Nd4jPointer) z.getShapeInfo()};

    NativeOps nativeOps;
    auto hash = op.getOpHash();
    auto status = nativeOps.execCustomOpFloat(nullptr, hash, ptrsInBuffer, ptrsInShapes, 1, ptrsOutBuffers, ptrsOutShapes, 1, nullptr, 0, nullptr, 0, false);
    ASSERT_EQ(Status::OK(), status);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));
}


TEST_F(JavaInteropTests, Test_Inplace_Outputs_2) {
    NDArray<float> x('c', {2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    NDArray<float> y(2.0f);
    NDArray<float> z('f', {2, 3});
    NDArray<float> e('c', {2, 3}, {3.f, 4.f, 5.f, 6.f, 7.f, 8.f});


    nd4j::ops::add<float> op;

    Nd4jPointer ptrsInBuffer[] = {(Nd4jPointer) x.getBuffer(), (Nd4jPointer) y.getBuffer()};
    Nd4jPointer ptrsInShapes[] = {(Nd4jPointer) x.getShapeInfo(), (Nd4jPointer) y.getShapeInfo()};

    Nd4jPointer ptrsOutBuffers[] = {(Nd4jPointer) z.getBuffer()};
    Nd4jPointer ptrsOutShapes[] = {(Nd4jPointer) z.getShapeInfo()};

    NativeOps nativeOps;
    auto hash = op.getOpHash();
    auto status = nativeOps.execCustomOpFloat(nullptr, hash, ptrsInBuffer, ptrsInShapes, 2, ptrsOutBuffers, ptrsOutShapes, 1, nullptr, 0, nullptr, 0, false);
    ASSERT_EQ(Status::OK(), status);

    ASSERT_TRUE(e.isSameShape(z));
    ASSERT_TRUE(e.equalsTo(z));
    ASSERT_FALSE(e.ordering() == z.ordering());
}

TEST_F(JavaInteropTests, Test_Inplace_Outputs_3) {
    NDArray<float> input('c', {2, 3, 4}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24});
    NDArray<float> indices('c', {1, 6},   {0,1, 2,2, 1,2});
    NDArray<float> output('f', {2, 6, 4});
    NDArray<float> e('c', {2, 6, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12, 9,10,11,12, 5, 6, 7, 8, 9,10,11,12, 13,14,15,16, 17,18,19,20, 21,22,23,24, 21,22,23,24, 17,18,19,20, 21,22,23,24});

    nd4j::ops::gather<float> op;

     Nd4jPointer ptrsInBuffer[] = {(Nd4jPointer) input.getBuffer(), (Nd4jPointer) indices.getBuffer()};
    Nd4jPointer ptrsInShapes[] = {(Nd4jPointer) input.getShapeInfo(), (Nd4jPointer) indices.getShapeInfo()};

    Nd4jPointer ptrsOutBuffers[] = {(Nd4jPointer) output.getBuffer()};
    Nd4jPointer ptrsOutShapes[] = {(Nd4jPointer) output.getShapeInfo()};

    Nd4jLong iArgs[] = {1};

    NativeOps nativeOps;
    auto hash = op.getOpHash();
    auto status = nativeOps.execCustomOpFloat(nullptr, hash, ptrsInBuffer, ptrsInShapes, 2, ptrsOutBuffers, ptrsOutShapes, 1, nullptr, 0, iArgs, 1, false);
    ASSERT_EQ(Status::OK(), status);

    ASSERT_TRUE(e.isSameShape(output));
    ASSERT_TRUE(e.equalsTo(output));
    ASSERT_FALSE(e.ordering() == output.ordering());
}

TEST_F(JavaInteropTests, Test_Reduce3_EdgeCase) {
    NDArray<double> x('c', {3, 4, 5});
    NDArray<double> y('c', {3, 4, 5});
    NDArray<double> z('c', {5});

    std::vector<int> dims = {0, 1};

    NativeOps nativeOps;
    nativeOps.execReduce3Double(nullptr, 2, x.buffer(), x.shapeInfo(), nullptr, y.buffer(), y.shapeInfo(), z.buffer(), z.shapeInfo(), dims.data(), (int) dims.size());
}

TEST_F(JavaInteropTests, Test_SimpleIf_Output) {
    Environment::getInstance()->setDebug(true);
    Environment::getInstance()->setVerbose(false);

    NativeOps ops;

    auto pl = nd4j::graph::readFlatBuffers("./resources/simpleif_0_1.fb");
    auto ptr = ops.executeFlatGraphFloat(nullptr, pl);

    Environment::getInstance()->setDebug(false);
    Environment::getInstance()->setVerbose(false);

    delete[] pl;
    delete ptr;
}


TEST_F(JavaInteropTests, Test_AveragePooling_FF_TF_double) {

    NDArray<double> input('c', {4, 10, 10, 3}, {9.37125111, 2.20166993,             2.91434479,             5.43639755,             -2.10573769,             4.08528662,             5.86908436,             -4.46203756,             2.21057916,             5.35849190,             0.01394637,             4.40566349,             7.07982206,             -0.09633455,             2.42429352,             3.97301817,             -1.89553940,             1.99690318,             6.33141708,             0.55401880,             1.70707977,             5.55204201,             -0.03513752,             1.60011971,             2.62700319,             -2.74582434,             3.06697464,             1.06277943,             -1.16075921,             -0.78095782,             9.72352791,             -1.22686064,             1.99644792,             7.35571337,             1.40607321,             0.11390255,             9.53334427,             2.28303599,             -1.66728830,             6.16678810,             -0.04532295,             -1.97708666,             9.74906158,             1.46223176,             -1.46734393,             4.30761862,             -1.23790228,             1.24823606,             6.13938427,             -3.83689475,             -1.19625473,             7.91535568,             6.05868721,             -3.22946382,             8.81633949,             -0.19967777,             0.66053957,             2.30919123,             0.74543846,             -0.39347672,             11.11058044,             0.53720862,             1.52645731,             5.70012379,             -1.15213466,             1.16451406,             7.00526333,             1.57362783,             -2.44384766,             5.54213285,             -1.98828590,             -0.70483637,             7.88281822,             -3.59875536,             0.80745387,             13.41578484,             -1.55507684,             -0.65855008,             9.32583523,             -0.14544789,             0.73436141,             3.61176538,             -1.71268058,             -2.58490300,             9.09280205,             -3.27405524,             -2.04569697,             4.44761324,             -0.62955856,             -2.61917663,             8.04890442,             0.54579324,             0.85929775,             9.82259560,             -1.93825579,             0.77703512,             4.67090321,             -4.79267597,             -2.38906908,             9.31265545,             0.96026313,             -1.14109385,             11.54231834,             -0.01417295,             -0.39500344,             8.49191666,             0.55300158,             2.79490185,             6.92466164,             1.72254205,             2.82222271,             8.83112717,             2.95033407,             2.18054962,             6.73509789,             -2.22272944,             0.51127720,             -1.04563558,             2.15747333,             -2.30959272,             9.55441570,             1.50396204,             1.77370787,             7.38146257,             -1.79076433,             3.20961165,             7.18864202,             2.91217351,             0.43018937,             7.11078024,             -1.17386127,             -0.16817921,             6.12327290,             -2.82205725,             3.30696845,             13.51291752,             -1.30856836,             -2.38332748,             11.09487438,             -1.47190213,             -0.53050828,             4.38285351,             -5.07309771,             1.50714362,             5.72274446,             -2.85825086,             -0.89673209,             3.73791552,             -0.67708802,             -4.13149452,             -0.00671843,             -0.26566532,             0.32961160,             7.14501762,             -1.41608179,             -4.96590328,             12.26205540,             -0.65158135,             -0.88641000,             6.95777559,             -0.79058206,             -0.10260171,             7.87169170,             1.35921454,             1.11759663,             5.46187401,             -2.57214499,             2.48484039,             4.04043484,             -2.07137156,             -1.42709637,             9.25487137,             -0.12605135,             -2.66949964,             2.89412403,             0.74451172,             -2.96250391,             3.99258423,             0.27084303,             0.32213116,             5.42332172,             -0.44414216,             1.70881832,             6.69346905,             0.53058422,             -4.73146200,             4.22051668,             2.24834967,             0.66996074,             4.30173683,             0.11849818,             -4.07520294,             8.27318478,             -2.54398274,             -2.86705542,             10.11775303,             -0.99382895,             0.65881538,             7.93556786,             -1.27934420,             -1.69343162,             9.68042564,             -1.02609646,             -1.18189347,             5.75370646,             -1.67888868,             -4.48871994,             4.79537392,             -0.79212248,             -0.19855022,             6.15060997,             -0.01081491,             3.64454579,             10.82562447,             1.58859253,             -2.65847278,             8.60093212,             -1.59196103,             0.07635692,             11.76175690,             -1.17453325,             0.10122013,             6.86458445,             -2.18891335,             -2.74004745,             8.07066154,             0.71818852,             -2.03035975,             6.31053686,             0.51509416,             1.39789927,             9.43515587,             2.04256630,             0.13985133,             4.65010691,             2.40911126,             -0.36255789,             -3.06867862,             -0.45225358,             -1.56778407,             6.05917358,             -1.09891272,             1.77184200,             6.46248102,             0.96042323,             -0.24346280,             4.63436460,             -4.69907761,             1.25187206,             11.46173859,             -2.21917558,             1.28007793,             6.92173195,             2.11268163,             -3.47389889,             5.08722782,             -3.03950930,             -4.17154264,             11.30568314,             0.80361372,             2.53214502,             7.18707085,             -4.49114513,             2.85449266,             10.14906883,             -0.31974933,             -0.84472644,             -0.52459574,             0.12921631,             -1.81390119,             2.76170087,             1.03982210,             2.91744232,             -0.29048753,             5.87453508,             -1.53684759,             1.85800636,             -0.91404629,             1.28954852,             5.11354685,             -2.47475505,             -1.33179152,             2.58552408,             1.37316465,             -3.32339454,             1.54122913,             3.24953628,             -0.29758382,             2.82391763,             -1.51142192,             -1.22699404,             6.75745535,             0.65452754,             -3.29385471,             2.06008053,             2.53172946,             -4.23532820,             -1.53909743,             -0.07010663,             -1.42173731,             7.29031610,             -0.18448229,             4.59496164,             6.73027277,             0.73441899,             0.14426160,             4.14915276,             -2.97010231,             6.05851364,             4.95218086,             -2.39145470,             2.40494704,             2.10288811,             0.53503096,             1.44511235,             6.66344261,             -3.05803776,             7.21418667,             3.30303526,             -0.24163735,             3.47409391,             3.64520788,             2.15189481,             -3.11243272,             3.62310791,             0.37379482,             0.40865007,             -0.83132005,             -4.78246069,             2.07030797,             6.51765442,             3.16178989,             5.06180477,             3.78434467,             -0.96689719,             0.35965276,             5.89967585,             1.40294051,             1.11952639,             10.59778214,             0.26739889,             -1.61297631,             6.24801159,             -0.93914318,             -0.57812452,             9.92604542,             -0.73025000,             -3.38530874,             2.45646000,             -2.47949195,             0.51638460,             10.65636063,             1.97816694,             -3.00407791,             2.66914415,             -0.81951088,             -0.23316640,             2.40737987,             -2.70007610,             1.51531935,             4.08860207,             -0.27552786,             -1.31721711,             7.11568260,             -3.33498216,             -4.02545023,             7.22675610,             -0.81690705,             -2.52689576,             1.04016697,             -0.79291463,             -0.34875512,             10.00498390,             -4.24167728,             1.46162593,             11.82569408,             -1.70359993,             -0.30161047,             16.44085884,             -0.82253462,             -0.09435523,             6.13080597,             -0.20259480,             0.68308711,             6.15663004,             -6.61776876,             0.33295766,             2.55449438,             -0.17819691,             -1.14892209,             5.56776142,             1.99279118,             1.33035934,             4.45823956,             3.34916544,             -2.59905386,             6.16164446,             -2.03881931,             -2.45273542,             12.46793365,             -2.22743297,             2.83738565,             8.48628139,             -1.39347959,             -1.30867767,             11.08041477,             -4.00363779,             2.09183025,             11.30395889,             -2.20504737,             1.37426853,             8.98735619,             1.04676604,             -0.72757077,             8.28050232,             -6.70741081,             -0.65798020,             5.68592072,             -0.60760021,             0.35854483,             6.26852131,             1.94100165,             1.32112014,             0.80987954,             -1.74617672,             -0.25434083,             7.16045523,             1.58884013,             -2.64847064,             13.14820385,             1.21393633,             -2.47258949,             9.41650105,             -0.79384226,             2.48954105,             10.95629311,             0.47723705,             4.02126694,             8.02593136,             -2.20726371,             -1.18794477,             1.50836647,             0.93118095,             -1.73513174,             8.85493565,             -2.99670315,             -0.79055870,             2.39473820,             2.05046916,             -2.38055134,             11.82299423,             0.15609655,             0.68744308,             5.66401434,             -0.69281673,             2.09855556,             7.74626589,             -0.34283102,             1.00542057,             9.95838642,             0.80161905,             2.33455157,             9.80057335,             -0.93561798,             2.56991577,             8.29711342,             0.94213426,             0.44209945,             11.70259857,             0.92710167,             2.60957146,             0.24971688,             -0.86529571,             3.78628922,             6.80884457,             -0.68178189,             2.21103406,             3.18895817,             0.60283208,             -2.92716241,             6.72060776,             -1.06625068,             2.56543374,             9.97404480,             3.58080721,             -0.94936347,             10.16736984,             -1.38464379,             1.18191063,             6.66179037,             -3.56115270,             0.32329530,             10.90870762,             2.20638227,             0.19653285,             7.34650040,             -3.63859272,             -1.03027737,             5.98829985,             -3.66606474,             -3.89746714,             8.63469028,             1.22569811,             1.63240814,             3.74385309,             0.58243257,             -0.56981975,             3.69260955,             1.00979900,             -1.44030499,             8.57058144,             -1.10648811,             1.20474911,             5.43133020,             -2.14822555,             -0.07928789,             11.25825310,             0.19645604,             -5.49546146,             10.41917038,             -0.68178523,             -2.99639869,             6.50054455,             0.46488351,             -5.42328453,             9.09500027,             -2.82107449,             0.05601966,             15.34610748,             -0.06820253,             3.86699796,             10.73316956,             -3.04795432,             -0.14702171,             5.64813185,             1.44028485,             -2.47596145,             0.07280898,             -3.03187990,             -1.35183525,             9.35835648,             2.72966957,             1.88199532,             10.36187744,             -0.22834805,             -3.26738238,             6.92025137,             -2.34061313,             4.77379704,             5.28559113,             -2.96323752,             -1.76186585,             5.94436455,             0.38647744,             -5.73869514,             6.76849556,             1.40892124,             -1.19068217,             5.37919092,             -6.65328646,             3.62782669,             12.34744644,             2.44762444,             -4.19242620,             6.14906216,             0.08121119,             0.61355996,             2.69666457,             -1.88962626,             -0.55314136,             1.84937525,             1.56048691,             1.17460012,             3.75674725,             1.06198275,             -5.74625874,             5.41645575,             -1.28946674,             -1.51689398,             4.32400894,             -0.05222082,             -4.83948946,             1.80747867,             1.63144708,             -2.73887825,             1.63975775,             -2.02163982,             -0.16210437,             2.93518686,             1.14427686,             -2.83246303,             4.79283667,             2.69697428,             -3.12678456,             -1.19225168,             -2.37022972,             -3.09429741,             1.94225383,             -1.13747168,             -2.55048585,             5.40242243,             1.12777328,             3.43713188,             3.62658787,             -2.16878843,             0.30164462,             2.97407579,             -0.07275413,             -1.31149673,             4.70066261,             -2.01323795,             4.85255766,             4.59128904,             1.68084168,             1.60336494,             6.58138466,             -1.04759812,             2.69906545,             3.55769277,             -0.74327278,             2.65819693,             5.39528131,             2.11248922,             -1.06446671,             5.24546766,             -2.43146014,             4.58907509,             0.06521678,             -2.24503994,             2.45722699,             6.94863081,             0.35258654,             2.83396196,             9.92525196,             -1.12225175,             -0.34365177,             7.19116688,             -4.39813757,             0.46517885,             13.22028065,             -2.57483673,             -6.37226963,             7.58046293,             -2.74600363,             0.42231262,             8.04881668,             0.17289802,             -0.53447008,             16.55157471,             -5.63614368,             0.39288223,             3.37079263,             1.26484549,             -0.12820500,             8.46440125,             -4.39304399,             2.97676420,             0.65650189,             0.83158541,             -1.11556435,             6.32885838,             -0.36087769,             2.80724382,             9.90292645,             1.15936041,             0.20947981,             6.91249275,             -2.67404819,             2.93782163,             6.65656614,             -2.30828357,             2.98214006,             6.80611229,             -4.93821478,             -7.66555262,             7.59763002,             -0.54159302,             3.87403512,             12.42607784,             2.59284401,             -0.23375344,             8.95293331,             -0.71807784,             0.61873478,             8.66713524,             1.24289191,             -2.37835455,             2.08071637,             -0.88315344,             -3.41891551,             6.85245323,             1.73007369,             1.02169311,             7.69170332,             -2.85411978,             2.69790673,             8.12906551,             -1.19351399,             -2.26442742,             12.26104450,             -0.75579089,             -1.73274946,             10.68729019,             2.20655656,             -0.90522075,             12.42165184,             -1.67929137,             2.44851565,             9.31565762,             -0.06645700,             1.52762020,             6.18427515,             -1.68882596,             3.70261097,             3.02252960,             -3.44125366,             -1.31575799,             2.84617424,             -0.96849400,             -4.52356243,             9.95027161,             0.19966406,             -0.78874779,             8.18595028,             -4.08300209,             1.75126517,             0.96418417,             -4.04913044,             -0.95200396,             12.03637886,             -0.03041124,             0.41642749,             8.88267422,             -3.24985337,             -2.24919462,             7.32566118,             0.16964148,             -2.74123430,             7.05264473,             -3.30191112,             0.17163286,             4.81851053,             -1.64463484,             -0.85933101,             7.29276276,             2.34066939,             -2.14860010,             3.46148157,             -0.01782012,             1.51504040,             4.79304934,             1.85281146,             -1.70663762,             6.93470192,             -4.15440845,             -1.25983095,             10.52491760,             0.42930329,             -1.85146868,             11.70042324,             -0.41704914,             3.83796859,             9.21148491,             -2.79719448,             0.79470479,             6.26926661,             -5.85230207,             3.95105338,             7.84790897,             -1.38680744,             -1.78099084,             11.95235348,             -2.99841452,             -1.34507811,             6.15714645,             -1.07552516,             -2.81228638,             1.66234732,             -4.55166149,             -1.92601109,             8.64634514,             -0.48158705,             3.31595659,             7.67371941,             2.56964207,             0.12107098,             4.56467867,             -0.93541539,             1.39432955,             11.99714088,             1.05353570,             -2.13099813,             3.67617917,             3.45895386,             1.37365830,             8.74344158,             -4.17585802,             1.43908918,             6.28764772,             3.97346330,             -0.69144285,             9.07983303,             -0.41635889,             -0.14965028,             8.85469818,             1.11306190,             2.59440994,             5.38982344,             -1.07948279,             1.37252975,             10.26984596,             -0.09318046,             2.73104119,             12.45902252,             -1.55446684,             -2.76124811,             12.19395065,             -0.51846564,             1.02764034,             11.42673588,             -0.95940983,             -0.04781032,             8.78379822,             -4.88957930,             0.32534006,             11.97696400,             -3.35108662,             1.95104563,             4.46915388,             -2.32061648,             3.45230985,             8.29983711,             2.81034684,             -2.35529327,             6.07801294,             -0.98105043,             -0.05359888,             2.52291036,             -0.01986909,             -2.35321999,             10.51954269,             2.11145401,             3.53506470,             7.29093266,             0.03721160,             -1.13496494,             7.43886709,             -5.84201956,             2.50796294,             12.14647675,             2.77490377,             -2.18896222,             6.05641937,             5.32617044,             1.04221284,             10.79106712,             -2.95749092,             -2.75414610,             11.30037117,             -3.40654182,             -2.24673963,             7.49126101,             0.70811015,             -6.18003702,             13.83951187,             -1.01204085,             1.36298490,             -1.04451632,             2.42435336,             -0.02346706,             -0.85528886,             1.04731262,             0.22192979,             4.15708160,             0.34933877,             0.04814529,             2.24107265,             0.49676740,             -1.47752666,             0.45040059,             -0.70471478,             -1.19759345,             0.21711677,             0.88461423,             -2.76830935,             5.52066898,             1.97664857,             -1.75381601,             3.45877838,             1.52617192,             -1.61350942,             0.85337949,             1.97610760,             -3.40310287,             3.40319014,             -3.38691044,             -0.71319139,             1.65463758,             -0.60680127,             -1.80700517,             8.02592373,             2.59627104,             2.65895891,             5.93043184,             -4.48425817,             3.92670918,             4.19496679,             -2.28286791,             6.41634607,             5.72330523,             1.16269672,             -0.28753027,             2.46342492,             0.36693189,             0.26712441,             6.37652683,             -2.50139046,             2.43923736,             5.56310415,             0.98065847,             1.04267502,             4.16403675,             -0.04966142,             4.40897894,             3.72905660,             -3.46129870,             3.59962773,             1.34830284,             -1.76661730,             0.47943926,             5.29946661,             -1.12711561,             1.26970029,             15.17655945,             -1.50971997,             5.81345224,             8.48562050,             -4.36049604,             2.48144460,             8.23780441,             -3.46030426,             -0.84656560,             5.94946814,             1.12747943,             -2.65683913,             8.69085693,             1.31309867,             -2.79958344,             8.76840591,             -1.56444156,             1.62710834,             2.41177034,             -0.72804940,             5.70619011,             4.67169666,             -0.86167198,             -1.83803177,             2.96346045,             2.82692933,             -2.81557131,             7.11113358,             -1.90071094,             2.54244423,             11.19284058,             -0.06298946,             -1.71517313,             12.98388577,             0.84510714,             3.00816894,             2.57200313,             0.03899818,             -1.49330592,             9.60099125,             -3.59513044,             -1.30045319,             7.09241819,             -0.65233821,             -2.33627677,             8.81366920,             0.84154201,             1.03312039,             9.85289097,             0.19351870,             1.78496623,             7.34631205,             -2.16530800,             -0.65016162,             2.46842360,             0.24016285,             -1.24308395,             4.78175163,             -0.97682536,             2.20942235,             6.68382788,             3.76786447,             -1.44454038,             6.26453733,             -3.23575711,             -2.30137897,             9.53092670,             -5.55222607,             3.25999236,             9.37559509,             1.86339056,             -0.23551451,             10.23400211,             3.93031883,             -0.52629089,             7.85724449,             -2.91549587,             4.46612740,             5.66530371,             -2.70820427,             4.81359577,             10.31247330,             1.92230141,             2.53931546,             0.74986327,             1.70303428,             0.48063779,             5.31099129,             -0.78976244,             3.75864220,             4.23051405,             2.34042454,             -7.98193836,             9.83987141,             -1.46722627,             3.54497814,             10.36455154,             -4.51249075,             0.77715248,             7.78694630,             -4.59989023,             -2.49585629,             9.90296268,             1.38535416,             1.17441154,             10.10452843,             -0.98628229,             0.60194463,             9.12639141,             -3.90754628,             2.88526392,             7.24123430,             -0.15283313,             -0.75728363,             -1.15116858,             -2.53791571,             0.77229571,             6.44114161,             0.02646767,             4.95463037,             7.21066380,             1.79384065,             0.73250306,             8.04447937,             0.32576546,             -0.79447043,             10.12717724,             2.33392906,             1.30716443,             12.36073112,             -0.36694977,             -1.20438910,             7.03105593,             0.59557682,             0.69267452,             10.18113136,             2.49944925,             -0.42229167,             8.83143330,             -1.18805945,             -2.87509322,             4.53596449,             4.09732771,             -3.39088297,             -1.02536607,             0.82119560,             -3.47302604,             9.29991817,             0.21001509,             4.97036457,             9.50018406,             1.04420102,             1.96560478,             10.74769592,             -6.22709799,             3.11690164,             5.06759691,             -1.23724771,             -3.05831861,             8.12925529,             -1.93435478,             -1.10151744,             9.32263088,             -0.04249470,             -5.98547363,             10.49398136,             0.26400441,             -0.78915191,             13.28219604,             2.99276900,             0.74853164,             2.49364305,             -3.43529654,             4.05278301,             2.13498688,             -2.35444307,             -0.79900265,             4.66968822,             -0.31095147,             3.60674143,             12.37222099,             -0.07855003,             -3.30292702,             12.15215874,             0.60886210,             2.87075138,             7.75271845,             0.38044083,             3.34402204,             6.40583277,             -0.87888050,             0.67438459,             6.91080809,             1.98332930,             -0.08303714,             8.08630371,             -0.16772588,             -2.74058914,             7.17253590,             -2.69122696,             1.48173678,             8.99470139,             -1.43302310,             -0.88651133,             2.66944790,             -0.29186964,             2.00838661,             5.09587479,             -0.76676071,             -2.88322186,             8.31110573,             -0.14550979,             -1.37726915,             10.28355122,             -1.60575438,             -0.04118848,             9.97510815,             0.14440438,             -3.24632120,             9.00034523,             4.14319563,             -1.31023729,             7.16950464,             -0.70428526,             2.01559544,             7.26155043,             2.40816474,             2.09847403,             7.31264496,             -0.75401551,             2.13392544,             7.03648758,             1.04036045,             -1.15636516,             1.09634531,             -0.06340861,             -0.58107805,             -0.65623116,             1.18972754,             -0.80717683,             1.40118241,             -0.61932516,             -3.60596156,             1.59904599,             -2.23774099,             -1.13721037,             3.89620137,             -0.09115922,             -7.51356888,             2.36975193,             -1.42520905,             -2.34173775,             3.33830214,             -2.74016523,             -3.04115510,             6.00119495,             -1.36084354,             -2.45065260,             4.56992292,             -3.02825928,-3.74182844,5.11069250,-0.91531068,-2.31385994,1.83399653,3.39370203,-3.60886002});
    NDArray<double> z('c', {4, 4, 4, 3});
    NDArray<double> exp('c', {4, 4, 4, 3}, {7.97172260, 0.06878620,             2.27749538,             7.29276514,             -0.14074677,             0.65480286,             5.70313978,             -0.06546132,             0.35443667,             3.70382833,             -0.84020567,             0.63826996,             8.60301399,             -0.38236514,             1.55177069,             7.37542057,             -0.99374938,             -0.29971302,             8.84352493,             -0.67121059,             0.43132120,             4.78175592,             -1.25070143,             -1.91523600,             6.03855371,             -0.00292124,             -1.11214364,             7.90158176,             -0.57949901,             -0.96735370,             7.81192017,             -0.53255427,             -0.48009714,             3.16953635,             0.08353355,             -1.54299748,             3.74821687,             1.69396687,             0.72724354,             5.42915201,             -1.13686812,             -0.71793109,             5.78376389,             -0.72239977,             -0.60055625,             2.53636408,             0.56777251,             -2.07892323,             6.08064651,             0.68620735,             2.54017019,             5.65828180,             -0.68255502,             1.47283304,             6.10842514,             -0.39655915,             0.28380761,             1.96707797,             -1.98206317,             0.94027776,             4.71811438,             0.32104525,             -0.92409706,             8.34588146,             -1.05581069,             -0.55217457,             9.58440876,             -0.96549922,             0.45820439,             5.65453672,             -2.50953507,             -0.71441835,             8.03059578,             -0.21281289,             0.92125505,             9.26900673,             -0.35963219,             -0.70039093,             8.59924412,             -1.22358346,             0.81318003,             3.85920119,             -0.01305223,             -1.09234154,             6.33158875,             1.28094780,             -1.48926139,             4.94969177,             -0.77126902,             -1.97033751,             5.64381838,             -0.16285487,             -1.31277227,             2.39893222,             -1.32902908,             -1.39609122,             6.47572327,             -0.45267010,             1.55727172,             6.70965624,             -1.68735468,             -0.05672536,             7.25092363,             -0.64613032,             0.67050058,             3.60789680,             -2.05948973,             2.22687531,             8.15202713,             -0.70148355,             1.28314006,             8.14842319,             -1.88807654,             -1.04808438,             8.45500565,             -0.76425624,             0.94542569,             4.56179953,             -0.28786001,             -2.04502511,             8.46278095,             -0.31019822,             0.07339200,             9.34214592,             -0.61948007,             0.52481830,             8.32515621,             -1.52418160,             0.49678251,             5.11082315,             -1.09908783,             -0.52969611,             5.27806664,             0.88632923,             0.66754371,             4.75839233,             0.48928693,             -0.68036932,             6.56925392,             -0.02949905,             -2.99189186,             4.46320581,             -0.64534980,             -0.29516968,             8.60809517,             -1.13120568,             3.41720533,             5.84243155,             -1.24109328,             0.89566326,             5.99578333,             -0.42496428,             2.07076764,             3.17812920,             -0.81566459,             -0.14363396,             6.55184317,             0.39633346,             -0.43852386,             8.70214558,             -2.24613595,             0.30708700,             8.73882294,             -0.53545928,             1.54409575,             4.49452257,             -0.16509305,             0.19028664,             8.24897003,             0.44750381,             2.15448594,             8.97640514,             -0.77728152,             0.57272542,             9.03467560,             0.47173575,             -1.10807717,             3.30056310,             -0.43268481,             -0.41470885,             3.53798294,             -0.08546703,             -2.16840744,             6.18733406,             -0.17871059,             -2.59837723,             5.94218683,             -1.02990067,             -0.49760687,             3.76938033,             0.86383581,             -1.91504073});

    nd4j::ops::avgpool2d<double> op;

    Nd4jPointer ptrsInBuffer[] = {reinterpret_cast<Nd4jPointer>(input.buffer())};
    Nd4jPointer ptrsInShapes[] = {reinterpret_cast<Nd4jPointer>(input.shapeInfo())};

    Nd4jPointer ptrsOutBuffers[] = {reinterpret_cast<Nd4jPointer>(z.buffer())};
    Nd4jPointer ptrsOutShapes[] = {reinterpret_cast<Nd4jPointer>(z.shapeInfo())};
    Nd4jLong iArgs[] = {3,3,  3,3,  0,0,  1,1,1,  0,1};

    NativeOps nativeOps;
    auto hash = op.getOpHash();
    auto status = nativeOps.execCustomOpDouble(nullptr, hash, ptrsInBuffer, ptrsInShapes, 1, ptrsOutBuffers, ptrsOutShapes, 1, nullptr, 0, iArgs, 11, false);
    ASSERT_EQ(Status::OK(), status);


    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));
}


TEST_F(JavaInteropTests, Test_AveragePooling_FF_TF_float) {

    NDArray<float> input('c', {4, 10, 10, 3}, {9.37125111f, 2.20166993f,2.91434479f,5.43639755f,-2.10573769f, 4.08528662f,5.86908436f,-4.46203756f,2.21057916f,5.35849190f,0.01394637f,             4.40566349f,             7.07982206f,             -0.09633455f,             2.42429352f,             3.97301817f,             -1.89553940f,             1.99690318f,             6.33141708f,             0.55401880f,             1.70707977f,             5.55204201f,             -0.03513752f,             1.60011971f,             2.62700319f,             -2.74582434f,             3.06697464f,             1.06277943f,             -1.16075921f,             -0.78095782f,             9.72352791f,             -1.22686064f,             1.99644792f,             7.35571337f,             1.40607321f,             0.11390255f,             9.53334427f,             2.28303599f,             -1.66728830f,             6.16678810f,             -0.04532295f,             -1.97708666f,             9.74906158f,             1.46223176f,             -1.46734393f,             4.30761862f,             -1.23790228f,             1.24823606f,             6.13938427f,             -3.83689475f,             -1.19625473f,             7.91535568f,             6.05868721f,             -3.22946382f,             8.81633949f,             -0.19967777f,             0.66053957f,             2.30919123f,             0.74543846f,             -0.39347672f,             11.11058044f,             0.53720862f,             1.52645731f,             5.70012379f,             -1.15213466f,             1.16451406f,             7.00526333f,             1.57362783f,             -2.44384766f,             5.54213285f,             -1.98828590f,             -0.70483637f,             7.88281822f,             -3.59875536f,             0.80745387f,             13.41578484f,             -1.55507684f,             -0.65855008f,             9.32583523f,             -0.14544789f,             0.73436141f,             3.61176538f,             -1.71268058f,             -2.58490300f,             9.09280205f,             -3.27405524f,             -2.04569697f,             4.44761324f,             -0.62955856f,             -2.61917663f,             8.04890442f,             0.54579324f,             0.85929775f,             9.82259560f,             -1.93825579f,             0.77703512f,             4.67090321f,             -4.79267597f,             -2.38906908f,             9.31265545f,             0.96026313f,             -1.14109385f,             11.54231834f,             -0.01417295f,             -0.39500344f,             8.49191666f,             0.55300158f,             2.79490185f,             6.92466164f,             1.72254205f,             2.82222271f,             8.83112717f,             2.95033407f,             2.18054962f,             6.73509789f,             -2.22272944f,             0.51127720f,             -1.04563558f,             2.15747333f,             -2.30959272f,             9.55441570f,             1.50396204f,             1.77370787f,             7.38146257f,             -1.79076433f,             3.20961165f,             7.18864202f,             2.91217351f,             0.43018937f,             7.11078024f,             -1.17386127f,             -0.16817921f,             6.12327290f,             -2.82205725f,             3.30696845f,             13.51291752f,             -1.30856836f,             -2.38332748f,             11.09487438f,             -1.47190213f,             -0.53050828f,             4.38285351f,             -5.07309771f,             1.50714362f,             5.72274446f,             -2.85825086f,             -0.89673209f,             3.73791552f,             -0.67708802f,             -4.13149452f,             -0.00671843f,             -0.26566532f,             0.32961160f,             7.14501762f,             -1.41608179f,             -4.96590328f,             12.26205540f,             -0.65158135f,             -0.88641000f,             6.95777559f,             -0.79058206f,             -0.10260171f,             7.87169170f,             1.35921454f,             1.11759663f,             5.46187401f,             -2.57214499f,             2.48484039f,             4.04043484f,             -2.07137156f,             -1.42709637f,             9.25487137f,             -0.12605135f,             -2.66949964f,             2.89412403f,             0.74451172f,             -2.96250391f,             3.99258423f,             0.27084303f,             0.32213116f,             5.42332172f,             -0.44414216f,             1.70881832f,             6.69346905f,             0.53058422f,             -4.73146200f,             4.22051668f,             2.24834967f,             0.66996074f,             4.30173683f,             0.11849818f,             -4.07520294f,             8.27318478f,             -2.54398274f,             -2.86705542f,             10.11775303f,             -0.99382895f,             0.65881538f,             7.93556786f,             -1.27934420f,             -1.69343162f,             9.68042564f,             -1.02609646f,             -1.18189347f,             5.75370646f,             -1.67888868f,             -4.48871994f,             4.79537392f,             -0.79212248f,             -0.19855022f,             6.15060997f,             -0.01081491f,             3.64454579f,             10.82562447f,             1.58859253f,             -2.65847278f,             8.60093212f,             -1.59196103f,             0.07635692f,             11.76175690f,             -1.17453325f,             0.10122013f,             6.86458445f,             -2.18891335f,             -2.74004745f,             8.07066154f,             0.71818852f,             -2.03035975f,             6.31053686f,             0.51509416f,             1.39789927f,             9.43515587f,             2.04256630f,             0.13985133f,             4.65010691f,             2.40911126f,             -0.36255789f,             -3.06867862f,             -0.45225358f,             -1.56778407f,             6.05917358f,             -1.09891272f,             1.77184200f,             6.46248102f,             0.96042323f,             -0.24346280f,             4.63436460f,             -4.69907761f,             1.25187206f,             11.46173859f,             -2.21917558f,             1.28007793f,             6.92173195f,             2.11268163f,             -3.47389889f,             5.08722782f,             -3.03950930f,             -4.17154264f,             11.30568314f,             0.80361372f,             2.53214502f,             7.18707085f,             -4.49114513f,             2.85449266f,             10.14906883f,             -0.31974933f,             -0.84472644f,             -0.52459574f,             0.12921631f,             -1.81390119f,             2.76170087f,             1.03982210f,             2.91744232f,             -0.29048753f,             5.87453508f,             -1.53684759f,             1.85800636f,             -0.91404629f,             1.28954852f,             5.11354685f,             -2.47475505f,             -1.33179152f,             2.58552408f,             1.37316465f,             -3.32339454f,             1.54122913f,             3.24953628f,             -0.29758382f,             2.82391763f,             -1.51142192f,             -1.22699404f,             6.75745535f,             0.65452754f,             -3.29385471f,             2.06008053f,             2.53172946f,             -4.23532820f,             -1.53909743f,             -0.07010663f,             -1.42173731f,             7.29031610f,             -0.18448229f,             4.59496164f,             6.73027277f,             0.73441899f,             0.14426160f,             4.14915276f,             -2.97010231f,             6.05851364f,             4.95218086f,             -2.39145470f,             2.40494704f,             2.10288811f,             0.53503096f,             1.44511235f,             6.66344261f,             -3.05803776f,             7.21418667f,             3.30303526f,             -0.24163735f,             3.47409391f,             3.64520788f,             2.15189481f,             -3.11243272f,             3.62310791f,             0.37379482f,             0.40865007f,             -0.83132005f,             -4.78246069f,             2.07030797f,             6.51765442f,             3.16178989f,             5.06180477f,             3.78434467f,             -0.96689719f,             0.35965276f,             5.89967585f,             1.40294051f,             1.11952639f,             10.59778214f,             0.26739889f,             -1.61297631f,             6.24801159f,             -0.93914318f,             -0.57812452f,             9.92604542f,             -0.73025000f,             -3.38530874f,             2.45646000f,             -2.47949195f,             0.51638460f,             10.65636063f,             1.97816694f,             -3.00407791f,             2.66914415f,             -0.81951088f,             -0.23316640f,             2.40737987f,             -2.70007610f,             1.51531935f,             4.08860207f,             -0.27552786f,             -1.31721711f,             7.11568260f,             -3.33498216f,             -4.02545023f,             7.22675610f,             -0.81690705f,             -2.52689576f,             1.04016697f,             -0.79291463f,             -0.34875512f,             10.00498390f,             -4.24167728f,             1.46162593f,             11.82569408f,             -1.70359993f,             -0.30161047f,             16.44085884f,             -0.82253462f,             -0.09435523f,             6.13080597f,             -0.20259480f,             0.68308711f,             6.15663004f,             -6.61776876f,             0.33295766f,             2.55449438f,             -0.17819691f,             -1.14892209f,             5.56776142f,             1.99279118f,             1.33035934f,             4.45823956f,             3.34916544f,             -2.59905386f,             6.16164446f,             -2.03881931f,             -2.45273542f,             12.46793365f,             -2.22743297f,             2.83738565f,             8.48628139f,             -1.39347959f,             -1.30867767f,             11.08041477f,             -4.00363779f,             2.09183025f,             11.30395889f,             -2.20504737f,             1.37426853f,             8.98735619f,             1.04676604f,             -0.72757077f,             8.28050232f,             -6.70741081f,             -0.65798020f,             5.68592072f,             -0.60760021f,             0.35854483f,             6.26852131f,             1.94100165f,             1.32112014f,             0.80987954f,             -1.74617672f,             -0.25434083f,             7.16045523f,             1.58884013f,             -2.64847064f,             13.14820385f,             1.21393633f,             -2.47258949f,             9.41650105f,             -0.79384226f,             2.48954105f,             10.95629311f,             0.47723705f,             4.02126694f,             8.02593136f,             -2.20726371f,             -1.18794477f,             1.50836647f,             0.93118095f,             -1.73513174f,             8.85493565f,             -2.99670315f,             -0.79055870f,             2.39473820f,             2.05046916f,             -2.38055134f,             11.82299423f,             0.15609655f,             0.68744308f,             5.66401434f,             -0.69281673f,             2.09855556f,             7.74626589f,             -0.34283102f,             1.00542057f,             9.95838642f,             0.80161905f,             2.33455157f,             9.80057335f,             -0.93561798f,             2.56991577f,             8.29711342f,             0.94213426f,             0.44209945f,             11.70259857f,             0.92710167f,             2.60957146f,             0.24971688f,             -0.86529571f,             3.78628922f,             6.80884457f,             -0.68178189f,             2.21103406f,             3.18895817f,             0.60283208f,             -2.92716241f,             6.72060776f,             -1.06625068f,             2.56543374f,             9.97404480f,             3.58080721f,             -0.94936347f,             10.16736984f,             -1.38464379f,             1.18191063f,             6.66179037f,             -3.56115270f,             0.32329530f,             10.90870762f,             2.20638227f,             0.19653285f,             7.34650040f,             -3.63859272f,             -1.03027737f,             5.98829985f,             -3.66606474f,             -3.89746714f,             8.63469028f,             1.22569811f,             1.63240814f,             3.74385309f,             0.58243257f,             -0.56981975f,             3.69260955f,             1.00979900f,             -1.44030499f,             8.57058144f,             -1.10648811f,             1.20474911f,             5.43133020f,             -2.14822555f,             -0.07928789f,             11.25825310f,             0.19645604f,             -5.49546146f,             10.41917038f,             -0.68178523f,             -2.99639869f,             6.50054455f,             0.46488351f,             -5.42328453f,             9.09500027f,             -2.82107449f,             0.05601966f,             15.34610748f,             -0.06820253f,             3.86699796f,             10.73316956f,             -3.04795432f,             -0.14702171f,             5.64813185f,             1.44028485f,             -2.47596145f,             0.07280898f,             -3.03187990f,             -1.35183525f,             9.35835648f,             2.72966957f,             1.88199532f,             10.36187744f,             -0.22834805f,             -3.26738238f,             6.92025137f,             -2.34061313f,             4.77379704f,             5.28559113f,             -2.96323752f,             -1.76186585f,             5.94436455f,             0.38647744f,             -5.73869514f,             6.76849556f,             1.40892124f,             -1.19068217f,             5.37919092f,             -6.65328646f,             3.62782669f,             12.34744644f,             2.44762444f,             -4.19242620f,             6.14906216f,             0.08121119f,             0.61355996f,             2.69666457f,             -1.88962626f,             -0.55314136f,             1.84937525f,             1.56048691f,             1.17460012f,             3.75674725f,             1.06198275f,             -5.74625874f,             5.41645575f,             -1.28946674f,             -1.51689398f,             4.32400894f,             -0.05222082f,             -4.83948946f,             1.80747867f,             1.63144708f,             -2.73887825f,             1.63975775f,             -2.02163982f,             -0.16210437f,             2.93518686f,             1.14427686f,             -2.83246303f,             4.79283667f,             2.69697428f,             -3.12678456f,             -1.19225168f,             -2.37022972f,             -3.09429741f,             1.94225383f,             -1.13747168f,             -2.55048585f,             5.40242243f,             1.12777328f,             3.43713188f,             3.62658787f,             -2.16878843f,             0.30164462f,             2.97407579f,             -0.07275413f,             -1.31149673f,             4.70066261f,             -2.01323795f,             4.85255766f,             4.59128904f,             1.68084168f,             1.60336494f,             6.58138466f,             -1.04759812f,             2.69906545f,             3.55769277f,             -0.74327278f,             2.65819693f,             5.39528131f,             2.11248922f,             -1.06446671f,             5.24546766f,             -2.43146014f,             4.58907509f,             0.06521678f,             -2.24503994f,             2.45722699f,             6.94863081f,             0.35258654f,             2.83396196f,             9.92525196f,             -1.12225175f,             -0.34365177f,             7.19116688f,             -4.39813757f,             0.46517885f,             13.22028065f,             -2.57483673f,             -6.37226963f,             7.58046293f,             -2.74600363f,             0.42231262f,             8.04881668f,             0.17289802f,             -0.53447008f,             16.55157471f,             -5.63614368f,             0.39288223f,             3.37079263f,             1.26484549f,             -0.12820500f,             8.46440125f,             -4.39304399f,             2.97676420f,             0.65650189f,             0.83158541f,             -1.11556435f,             6.32885838f,             -0.36087769f,             2.80724382f,             9.90292645f,             1.15936041f,             0.20947981f,             6.91249275f,             -2.67404819f,             2.93782163f,             6.65656614f,             -2.30828357f,             2.98214006f,             6.80611229f,             -4.93821478f,             -7.66555262f,             7.59763002f,             -0.54159302f,             3.87403512f,             12.42607784f,             2.59284401f,             -0.23375344f,             8.95293331f,             -0.71807784f,             0.61873478f,             8.66713524f,             1.24289191f,             -2.37835455f,             2.08071637f,             -0.88315344f,             -3.41891551f,             6.85245323f,             1.73007369f,             1.02169311f,             7.69170332f,             -2.85411978f,             2.69790673f,             8.12906551f,             -1.19351399f,             -2.26442742f,             12.26104450f,             -0.75579089f,             -1.73274946f,             10.68729019f,             2.20655656f,             -0.90522075f,             12.42165184f,             -1.67929137f,             2.44851565f,             9.31565762f,             -0.06645700f,             1.52762020f,             6.18427515f,             -1.68882596f,             3.70261097f,             3.02252960f,             -3.44125366f,             -1.31575799f,             2.84617424f,             -0.96849400f,             -4.52356243f,             9.95027161f,             0.19966406f,             -0.78874779f,             8.18595028f,             -4.08300209f,             1.75126517f,             0.96418417f,             -4.04913044f,             -0.95200396f,             12.03637886f,             -0.03041124f,             0.41642749f,             8.88267422f,             -3.24985337f,             -2.24919462f,             7.32566118f,             0.16964148f,             -2.74123430f,             7.05264473f,             -3.30191112f,             0.17163286f,             4.81851053f,             -1.64463484f,             -0.85933101f,             7.29276276f,             2.34066939f,             -2.14860010f,             3.46148157f,             -0.01782012f,             1.51504040f,             4.79304934f,             1.85281146f,             -1.70663762f,             6.93470192f,             -4.15440845f,             -1.25983095f,             10.52491760f,             0.42930329f,             -1.85146868f,             11.70042324f,             -0.41704914f,             3.83796859f,             9.21148491f,             -2.79719448f,             0.79470479f,             6.26926661f,             -5.85230207f,             3.95105338f,             7.84790897f,             -1.38680744f,             -1.78099084f,             11.95235348f,             -2.99841452f,             -1.34507811f,             6.15714645f,             -1.07552516f,             -2.81228638f,             1.66234732f,             -4.55166149f,             -1.92601109f,             8.64634514f,             -0.48158705f,             3.31595659f,             7.67371941f,             2.56964207f,             0.12107098f,             4.56467867f,             -0.93541539f,             1.39432955f,             11.99714088f,             1.05353570f,             -2.13099813f,             3.67617917f,             3.45895386f,             1.37365830f,             8.74344158f,             -4.17585802f,             1.43908918f,             6.28764772f,             3.97346330f,             -0.69144285f,             9.07983303f,             -0.41635889f,             -0.14965028f,             8.85469818f,             1.11306190f,             2.59440994f,             5.38982344f,             -1.07948279f,             1.37252975f,             10.26984596f,             -0.09318046f,             2.73104119f,             12.45902252f,             -1.55446684f,             -2.76124811f,             12.19395065f,             -0.51846564f,             1.02764034f,             11.42673588f,             -0.95940983f,             -0.04781032f,             8.78379822f,             -4.88957930f,             0.32534006f,             11.97696400f,             -3.35108662f,             1.95104563f,             4.46915388f,             -2.32061648f,             3.45230985f,             8.29983711f,             2.81034684f,             -2.35529327f,             6.07801294f,             -0.98105043f,             -0.05359888f,             2.52291036f,             -0.01986909f,             -2.35321999f,             10.51954269f,             2.11145401f,             3.53506470f,             7.29093266f,             0.03721160f,             -1.13496494f,             7.43886709f,             -5.84201956f,             2.50796294f,             12.14647675f,             2.77490377f,             -2.18896222f,             6.05641937f,             5.32617044f,             1.04221284f,             10.79106712f,             -2.95749092f,             -2.75414610f,             11.30037117f,             -3.40654182f,             -2.24673963f,             7.49126101f,             0.70811015f,             -6.18003702f,             13.83951187f,             -1.01204085f,             1.36298490f,             -1.04451632f,             2.42435336f,             -0.02346706f,             -0.85528886f,             1.04731262f,             0.22192979f,             4.15708160f,             0.34933877f,             0.04814529f,             2.24107265f,             0.49676740f,             -1.47752666f,             0.45040059f,             -0.70471478f,             -1.19759345f,             0.21711677f,             0.88461423f,             -2.76830935f,             5.52066898f,             1.97664857f,             -1.75381601f,             3.45877838f,             1.52617192f,             -1.61350942f,             0.85337949f,             1.97610760f,             -3.40310287f,             3.40319014f,             -3.38691044f,             -0.71319139f,             1.65463758f,             -0.60680127f,             -1.80700517f,             8.02592373f,             2.59627104f,             2.65895891f,             5.93043184f,             -4.48425817f,             3.92670918f,             4.19496679f,             -2.28286791f,             6.41634607f,             5.72330523f,             1.16269672f,             -0.28753027f,             2.46342492f,             0.36693189f,             0.26712441f,             6.37652683f,             -2.50139046f,             2.43923736f,             5.56310415f,             0.98065847f,             1.04267502f,             4.16403675f,             -0.04966142f,             4.40897894f,             3.72905660f,             -3.46129870f,             3.59962773f,             1.34830284f,             -1.76661730f,             0.47943926f,             5.29946661f,             -1.12711561f,             1.26970029f,             15.17655945f,             -1.50971997f,             5.81345224f,             8.48562050f,             -4.36049604f,             2.48144460f,             8.23780441f,             -3.46030426f,             -0.84656560f,             5.94946814f,             1.12747943f,             -2.65683913f,             8.69085693f,             1.31309867f,             -2.79958344f,             8.76840591f,             -1.56444156f,             1.62710834f,             2.41177034f,             -0.72804940f,             5.70619011f,             4.67169666f,             -0.86167198f,             -1.83803177f,             2.96346045f,             2.82692933f,             -2.81557131f,             7.11113358f,             -1.90071094f,             2.54244423f,             11.19284058f,             -0.06298946f,             -1.71517313f,             12.98388577f,             0.84510714f,             3.00816894f,             2.57200313f,             0.03899818f,             -1.49330592f,             9.60099125f,             -3.59513044f,             -1.30045319f,             7.09241819f,             -0.65233821f,             -2.33627677f,             8.81366920f,             0.84154201f,             1.03312039f,             9.85289097f,             0.19351870f,             1.78496623f,             7.34631205f,             -2.16530800f,             -0.65016162f,             2.46842360f,             0.24016285f,             -1.24308395f,             4.78175163f,             -0.97682536f,             2.20942235f,             6.68382788f,             3.76786447f,             -1.44454038f,             6.26453733f,             -3.23575711f,             -2.30137897f,             9.53092670f,             -5.55222607f,             3.25999236f,             9.37559509f,             1.86339056f,             -0.23551451f,             10.23400211f,             3.93031883f,             -0.52629089f,             7.85724449f,             -2.91549587f,             4.46612740f,             5.66530371f,             -2.70820427f,             4.81359577f,             10.31247330f,             1.92230141f,             2.53931546f,             0.74986327f,             1.70303428f,             0.48063779f,             5.31099129f,             -0.78976244f,             3.75864220f,             4.23051405f,             2.34042454f,             -7.98193836f,             9.83987141f,             -1.46722627f,             3.54497814f,             10.36455154f,             -4.51249075f,             0.77715248f,             7.78694630f,             -4.59989023f,             -2.49585629f,             9.90296268f,             1.38535416f,             1.17441154f,             10.10452843f,             -0.98628229f,             0.60194463f,             9.12639141f,             -3.90754628f,             2.88526392f,             7.24123430f,             -0.15283313f,             -0.75728363f,             -1.15116858f,             -2.53791571f,             0.77229571f,             6.44114161f,             0.02646767f,             4.95463037f,             7.21066380f,             1.79384065f,             0.73250306f,             8.04447937f,             0.32576546f,             -0.79447043f,             10.12717724f,             2.33392906f,             1.30716443f,             12.36073112f,             -0.36694977f,             -1.20438910f,             7.03105593f,             0.59557682f,             0.69267452f,             10.18113136f,             2.49944925f,             -0.42229167f,             8.83143330f,             -1.18805945f,             -2.87509322f,             4.53596449f,             4.09732771f,             -3.39088297f,             -1.02536607f,             0.82119560f,             -3.47302604f,             9.29991817f,             0.21001509f,             4.97036457f,             9.50018406f,             1.04420102f,             1.96560478f,             10.74769592f,             -6.22709799f,             3.11690164f,             5.06759691f,             -1.23724771f,             -3.05831861f,             8.12925529f,             -1.93435478f,             -1.10151744f,             9.32263088f,             -0.04249470f,             -5.98547363f,             10.49398136f,             0.26400441f,             -0.78915191f,             13.28219604f,             2.99276900f,             0.74853164f,             2.49364305f,             -3.43529654f,             4.05278301f,             2.13498688f,             -2.35444307f,             -0.79900265f,             4.66968822f,             -0.31095147f,             3.60674143f,             12.37222099f,             -0.07855003f,             -3.30292702f,             12.15215874f,             0.60886210f,             2.87075138f,             7.75271845f,             0.38044083f,             3.34402204f,             6.40583277f,             -0.87888050f,             0.67438459f,             6.91080809f,             1.98332930f,             -0.08303714f,             8.08630371f,             -0.16772588f,             -2.74058914f,             7.17253590f,             -2.69122696f,             1.48173678f,             8.99470139f,             -1.43302310f,             -0.88651133f,             2.66944790f,             -0.29186964f,             2.00838661f,             5.09587479f,             -0.76676071f,             -2.88322186f,             8.31110573f,             -0.14550979f,             -1.37726915f,             10.28355122f,             -1.60575438f,             -0.04118848f,             9.97510815f,             0.14440438f,             -3.24632120f,             9.00034523f,             4.14319563f,             -1.31023729f,             7.16950464f,             -0.70428526f,             2.01559544f,             7.26155043f,             2.40816474f,             2.09847403f,             7.31264496f,             -0.75401551f,             2.13392544f,             7.03648758f,             1.04036045f,             -1.15636516f,             1.09634531f,             -0.06340861f,             -0.58107805f,             -0.65623116f,             1.18972754f,             -0.80717683f,             1.40118241f,             -0.61932516f,             -3.60596156f,             1.59904599f,             -2.23774099f,             -1.13721037f,             3.89620137f,             -0.09115922f,             -7.51356888f,             2.36975193f,             -1.42520905f,             -2.34173775f,             3.33830214f,             -2.74016523f,             -3.04115510f,             6.00119495f,             -1.36084354f,             -2.45065260f,             4.56992292f,             -3.02825928f,             -3.74182844f,             5.11069250f,             -0.91531068f,             -2.31385994f,             1.83399653f,             3.39370203f,             -3.60886002f});
    NDArray<float> z('c', {4, 4, 4, 3});
    NDArray<float> exp('c', {4, 4, 4, 3}, {7.97172260f, 0.06878620f,             2.27749538f,             7.29276514f,             -0.14074677f,             0.65480286f,             5.70313978f,             -0.06546132f,             0.35443667f,             3.70382833f,             -0.84020567f,             0.63826996f,             8.60301399f,             -0.38236514f,             1.55177069f,             7.37542057f,             -0.99374938f,             -0.29971302f,             8.84352493f,             -0.67121059f,             0.43132120f,             4.78175592f,             -1.25070143f,             -1.91523600f,             6.03855371f,             -0.00292124f,             -1.11214364f,             7.90158176f,             -0.57949901f,             -0.96735370f,             7.81192017f,             -0.53255427f,             -0.48009714f,             3.16953635f,             0.08353355f,             -1.54299748f,             3.74821687f,             1.69396687f,             0.72724354f,             5.42915201f,             -1.13686812f,             -0.71793109f,             5.78376389f,             -0.72239977f,             -0.60055625f,             2.53636408f,             0.56777251f,             -2.07892323f,             6.08064651f,             0.68620735f,             2.54017019f,             5.65828180f,             -0.68255502f,             1.47283304f,             6.10842514f,             -0.39655915f,             0.28380761f,             1.96707797f,             -1.98206317f,             0.94027776f,             4.71811438f,             0.32104525f,             -0.92409706f,             8.34588146f,             -1.05581069f,             -0.55217457f,             9.58440876f,             -0.96549922f,             0.45820439f,             5.65453672f,             -2.50953507f,             -0.71441835f,             8.03059578f,             -0.21281289f,             0.92125505f,             9.26900673f,             -0.35963219f,             -0.70039093f,             8.59924412f,             -1.22358346f,             0.81318003f,             3.85920119f,             -0.01305223f,             -1.09234154f,             6.33158875f,             1.28094780f,             -1.48926139f,             4.94969177f,             -0.77126902f,             -1.97033751f,             5.64381838f,             -0.16285487f,             -1.31277227f,             2.39893222f,             -1.32902908f,             -1.39609122f,             6.47572327f,             -0.45267010f,             1.55727172f,             6.70965624f,             -1.68735468f,             -0.05672536f,             7.25092363f,             -0.64613032f,             0.67050058f,             3.60789680f,             -2.05948973f,             2.22687531f,             8.15202713f,             -0.70148355f,             1.28314006f,             8.14842319f,             -1.88807654f,             -1.04808438f,             8.45500565f,             -0.76425624f,             0.94542569f,             4.56179953f,             -0.28786001f,             -2.04502511f,             8.46278095f,             -0.31019822f,             0.07339200f,             9.34214592f,             -0.61948007f,             0.52481830f,             8.32515621f,             -1.52418160f,             0.49678251f,             5.11082315f,             -1.09908783f,             -0.52969611f,             5.27806664f,             0.88632923f,             0.66754371f,             4.75839233f,             0.48928693f,             -0.68036932f,             6.56925392f,             -0.02949905f,             -2.99189186f,             4.46320581f,             -0.64534980f,             -0.29516968f,             8.60809517f,             -1.13120568f,             3.41720533f,             5.84243155f,             -1.24109328f,             0.89566326f,             5.99578333f,             -0.42496428f,             2.07076764f,             3.17812920f,             -0.81566459f,             -0.14363396f,             6.55184317f,             0.39633346f,             -0.43852386f,             8.70214558f,             -2.24613595f,             0.30708700f,             8.73882294f,             -0.53545928f,             1.54409575f,             4.49452257f,             -0.16509305f,             0.19028664f,             8.24897003f,             0.44750381f,             2.15448594f,             8.97640514f,             -0.77728152f,             0.57272542f,             9.03467560f,             0.47173575f,             -1.10807717f,             3.30056310f,             -0.43268481f,             -0.41470885f,             3.53798294f,             -0.08546703f,             -2.16840744f,             6.18733406f,             -0.17871059f,             -2.59837723f,             5.94218683f,             -1.02990067f,             -0.49760687f,             3.76938033f,             0.86383581f,             -1.91504073f});

    nd4j::ops::avgpool2d<float> op;

    Nd4jPointer ptrsInBuffer[] = {reinterpret_cast<Nd4jPointer>(input.buffer())};
    Nd4jPointer ptrsInShapes[] = {reinterpret_cast<Nd4jPointer>(input.shapeInfo())};

    Nd4jPointer ptrsOutBuffers[] = {reinterpret_cast<Nd4jPointer>(z.buffer())};
    Nd4jPointer ptrsOutShapes[] = {reinterpret_cast<Nd4jPointer>(z.shapeInfo())};
    Nd4jLong iArgs[] = {3,3,  3,3,  0,0,  1,1,1,  0,1};

    NativeOps nativeOps;
    auto hash = op.getOpHash();
    auto status = nativeOps.execCustomOpFloat(nullptr, hash, ptrsInBuffer, ptrsInShapes, 1, ptrsOutBuffers, ptrsOutShapes, 1, nullptr, 0, iArgs, 11, false);
    ASSERT_EQ(Status::OK(), status);


    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));
}



TEST_F(JavaInteropTests, Test_Results_Conversion_1) {
    NativeOps ops;

    auto pl = nd4j::graph::readFlatBuffers("./resources/gru_dynamic_mnist.fb");
    auto ptr = ops.executeFlatGraphFloat(nullptr, pl);

    // at this point we have FlatResults
    auto flatResult = GetFlatResult(ptr->pointer());
    auto size = flatResult->variables()->size();

    // we know exact number of outputs in this graph in given mode
    ASSERT_EQ(184, size);


    // now we're rolling through all variables and restore them one by one
    for (int e = 0; e < size; e++) {
        auto flatVar = flatResult->variables()->Get(e);
        auto flatArray = flatVar->ndarray();

        // checking var part first
        // we just want to ensure we're not experiencing overruns here
        auto name = flatVar->name()->str();

        // checking array part now
        auto shape = flatArray->shape();
        auto rank = shape->Get(0);

        ASSERT_TRUE(shape->size() > 0 && rank >= 0 &&  rank < MAX_RANK);

        // building regular NDArray out of this FlatArray
        auto ndarray = nd4j::graph::FlatUtils::fromFlatArray<float>(flatArray);

        // rank should match FlatArray
        ASSERT_EQ(rank, ndarray->rankOf());

        // array shouldn't have any NaN/Inf values
        ASSERT_TRUE(ndarray->isFinite());

        // array should be assignable
        ndarray->assign(123.f);

        // and safely removable after
        delete ndarray;
    }


    delete[] pl;
    delete ptr;

    // and we should have 0 leaks reported after this line :)
}

TEST_F(JavaInteropTests, Test_NLP_Aggregations_1) {
    NativeOps ops;

    std::array<float, 60> syn0 = {-0.022756476f, 0.0126427775f, 0.011029151f, -0.013542821f, -0.012327666f, -0.0032439455f, -0.008405109f, -0.016651405f, 0.0015980572f, -0.007442479f, 0.019937921f, -0.016222188f, -0.016541665f, 0.013372547f, 0.006625724f, 0.0058958204f, -0.01281835f, -6.2343775E-4f, 0.0019826533f, 0.010253737f, -0.010291531f, 0.0019767822f, 0.018071089f, -0.0117441565f, 0.023176769f, 0.0032820583f, 0.0061427564f, -0.01696018f, 0.0054971874f, 0.0043818625f, 0.019323621f, 0.0036080598f, 0.024376748f, -0.0024499625f, 0.019496754f, 0.010563821f, -2.0503551E-4f, -0.0146056535f, 0.009949291f, 0.017604528f, -0.0050302492f, -0.022060446f, 0.016468976f, -0.0034482107f, 0.010270384f, -0.0063356445f, -0.019934833f, -0.02325993f, 0.016109904f, -0.0031106502f, -0.0020592287f, 0.024031803f, 0.005184144f, -0.024887865f, 0.02100272f, 3.395051E-4f, 0.018432347f, 5.673498E-4f, -0.020073576f, 0.010949242f};
    std::array<float, 60> syn1;
    std::array<float, 100000> exp;

    for (int e = 0; e < syn1.size(); e++)
        syn1[e] = 0.0f;

    for (int e = 0; e < exp.size(); e++) {
        auto f = static_cast<double>(e);
        auto tmp = nd4j::math::nd4j_exp<double>((f / 100000.0 * 2.0 - 1.0) * 6.0);
        exp[e] = static_cast<float>(tmp / (tmp + 1.0));
    }

    auto maxTypes = 5;
    auto numAggregates = 1;
    auto opNum = 3;
    auto maxArgs = 6;
    auto maxShapes = 0;
    auto maxIntArrays = 2;
    auto maxIntArraySize = 40;
    auto maxIndexArguments = 10;
    auto maxRealArguments = 2;

    std::array<int, 100000> pointer;

    auto batchLimit = 512;

    int indexPos = maxTypes * batchLimit;
    int intArraysPos = indexPos + (maxIndexArguments * batchLimit);
    int realPos = (intArraysPos + (maxIntArrays * maxIntArraySize * batchLimit));
    int argsPos = (realPos + ((maxRealArguments * batchLimit))) / 2;
    int shapesPos = argsPos + (maxArgs * batchLimit);

    std::vector<int> intArray0({0, 0, 0, 0, 0});
    std::vector<int> intArray1({1, 0, 0, 0, 0});

    std::vector<int> indexingArgs0({1, 20, 5, 0, 100000, 3, 0, 0, 0});
    std::vector<int> indexingArgs1({0, 20, 5, 0, 100000, 3, 1, 0, 0});

    std::vector<float> realArgs0({0.024964055335354007f, 3.0768702268737162E18f});

    int argSize = 6;
    int shapesSize = 0;
    int indexingSize = 9;
    int realArgsSize = 2;
    int intArraysSize = 2;

    int e = 0;

        auto idx = e * maxTypes;

        // numbers of arguments
        pointer[idx] = 6; // arguments size
        pointer[idx+1] = 0; // shapes size
        pointer[idx+2] = 9; // indexing arguments size
        pointer[idx+3] = 2; // real args size
        pointer[idx+4] = 2; // intArray args size

        // indexing args
        auto idxArgs = e == 0 ? indexingArgs0 : indexingArgs1;
        for (int f = 0; f < idxArgs.size(); f++) {
            idx = indexPos + e * maxIndexArguments;
            pointer[idx + f] = idxArgs[f];
        }

        // int array values
        int bsize = maxIntArrays * maxIntArraySize;
        for (int f = 0; f < intArraysSize; f++) {
            int step = (e * bsize) + (f * maxIntArraySize);
            auto intArr = f == 0 ? intArray0 : intArray1;
            for (int x = 0; x < intArr.size(); x++) {
                idx = intArraysPos + step + x;
                pointer[idx] = intArr[x];
            }
        }

        // real args
        auto ptr = reinterpret_cast<float *>(pointer.data());
        for (int f = 0; f < realArgsSize; f++) {
            idx = realPos + (e * maxRealArguments);
            ptr[idx + f] = realArgs0[f];
        }

        //
        auto ptrptr = reinterpret_cast<void **>(pointer.data());
        idx = argsPos + e * maxArgs;
        ptrptr[idx] = reinterpret_cast<void*>(syn0.data());
        ptrptr[idx+1] = reinterpret_cast<void*>(syn1.data());
        ptrptr[idx+2] = reinterpret_cast<void*>(exp.data());


    ops.execAggregateBatchFloat(nullptr, numAggregates, opNum, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIndexArguments, maxRealArguments, pointer.data());
}