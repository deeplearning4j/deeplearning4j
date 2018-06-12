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

using namespace nd4j;
using namespace nd4j::ops;

class JavaInteropTests : public testing::Test {
public:

};


TEST_F(JavaInteropTests, TestShapeExposure1) {
    NDArray<float> input('c', {1, 2, 5, 4});
    NDArray<float> weights('c', {3, 2, 2, 2});
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

    NDArrayFactory<float>::linspace(1, input);
    NDArrayFactory<float>::linspace(1, weightsD);
    NDArrayFactory<float>::linspace(1, weightsP);
    NDArrayFactory<float>::linspace(1, bias);

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

    NDArrayFactory<float>::linspace(1, input);
    NDArrayFactory<float>::linspace(1, weightsD);

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
    NDArrayFactory<float>::linspace(1, input);


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
    NDArrayFactory<float>::linspace(1, input);

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
    NDArrayFactory<float>::linspace(1, input);

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
    NDArrayFactory<float>::linspace(1, input);

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
    Environment::getInstance()->setVerbose(true);

    NativeOps ops;

    auto pl = nd4j::graph::readFlatBuffers("./resources/simpleif_0_1.fb");
    auto ptr = ops.executeFlatGraphFloat(nullptr, pl);

    Environment::getInstance()->setDebug(false);
    Environment::getInstance()->setVerbose(false);

    delete[] pl;
    delete ptr;
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