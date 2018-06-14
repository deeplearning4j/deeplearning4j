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


TEST_F(JavaInteropTests, Test_AveragePooling_FF_TF) {
    NDArray<double> input('c', {4, 10, 10, 3}, {9.37125111, 2.20166993,             2.91434479,             5.43639755,             -2.10573769,             4.08528662,             5.86908436,             -4.46203756,             2.21057916,             5.35849190,             0.01394637,             4.40566349,             7.07982206,             -0.09633455,             2.42429352,             3.97301817,             -1.89553940,             1.99690318,             6.33141708,             0.55401880,             1.70707977,             5.55204201,             -0.03513752,             1.60011971,             2.62700319,             -2.74582434,             3.06697464,             1.06277943,             -1.16075921,             -0.78095782,             9.72352791,             -1.22686064,             1.99644792,             7.35571337,             1.40607321,             0.11390255,             9.53334427,             2.28303599,             -1.66728830,             6.16678810,             -0.04532295,             -1.97708666,             9.74906158,             1.46223176,             -1.46734393,             4.30761862,             -1.23790228,             1.24823606,             6.13938427,             -3.83689475,             -1.19625473,             7.91535568,             6.05868721,             -3.22946382,             8.81633949,             -0.19967777,             0.66053957,             2.30919123,             0.74543846,             -0.39347672,             11.11058044,             0.53720862,             1.52645731,             5.70012379,             -1.15213466,             1.16451406,             7.00526333,             1.57362783,             -2.44384766,             5.54213285,             -1.98828590,             -0.70483637,             7.88281822,             -3.59875536,             0.80745387,             13.41578484,             -1.55507684,             -0.65855008,             9.32583523,             -0.14544789,             0.73436141,             3.61176538,             -1.71268058,             -2.58490300,             9.09280205,             -3.27405524,             -2.04569697,             4.44761324,             -0.62955856,             -2.61917663,             8.04890442,             0.54579324,             0.85929775,             9.82259560,             -1.93825579,             0.77703512,             4.67090321,             -4.79267597,             -2.38906908,             9.31265545,             0.96026313,             -1.14109385,             11.54231834,             -0.01417295,             -0.39500344,             8.49191666,             0.55300158,             2.79490185,             6.92466164,             1.72254205,             2.82222271,             8.83112717,             2.95033407,             2.18054962,             6.73509789,             -2.22272944,             0.51127720,             -1.04563558,             2.15747333,             -2.30959272,             9.55441570,             1.50396204,             1.77370787,             7.38146257,             -1.79076433,             3.20961165,             7.18864202,             2.91217351,             0.43018937,             7.11078024,             -1.17386127,             -0.16817921,             6.12327290,             -2.82205725,             3.30696845,             13.51291752,             -1.30856836,             -2.38332748,             11.09487438,             -1.47190213,             -0.53050828,             4.38285351,             -5.07309771,             1.50714362,             5.72274446,             -2.85825086,             -0.89673209,             3.73791552,             -0.67708802,             -4.13149452,             -0.00671843,             -0.26566532,             0.32961160,             7.14501762,             -1.41608179,             -4.96590328,             12.26205540,             -0.65158135,             -0.88641000,             6.95777559,             -0.79058206,             -0.10260171,             7.87169170,             1.35921454,             1.11759663,             5.46187401,             -2.57214499,             2.48484039,             4.04043484,             -2.07137156,             -1.42709637,             9.25487137,             -0.12605135,             -2.66949964,             2.89412403,             0.74451172,             -2.96250391,             3.99258423,             0.27084303,             0.32213116,             5.42332172,             -0.44414216,             1.70881832,             6.69346905,             0.53058422,             -4.73146200,             4.22051668,             2.24834967,             0.66996074,             4.30173683,             0.11849818,             -4.07520294,             8.27318478,             -2.54398274,             -2.86705542,             10.11775303,             -0.99382895,             0.65881538,             7.93556786,             -1.27934420,             -1.69343162,             9.68042564,             -1.02609646,             -1.18189347,             5.75370646,             -1.67888868,             -4.48871994,             4.79537392,             -0.79212248,             -0.19855022,             6.15060997,             -0.01081491,             3.64454579,             10.82562447,             1.58859253,             -2.65847278,             8.60093212,             -1.59196103,             0.07635692,             11.76175690,             -1.17453325,             0.10122013,             6.86458445,             -2.18891335,             -2.74004745,             8.07066154,             0.71818852,             -2.03035975,             6.31053686,             0.51509416,             1.39789927,             9.43515587,             2.04256630,             0.13985133,             4.65010691,             2.40911126,             -0.36255789,             -3.06867862,             -0.45225358,             -1.56778407,             6.05917358,             -1.09891272,             1.77184200,             6.46248102,             0.96042323,             -0.24346280,             4.63436460,             -4.69907761,             1.25187206,             11.46173859,             -2.21917558,             1.28007793,             6.92173195,             2.11268163,             -3.47389889,             5.08722782,             -3.03950930,             -4.17154264,             11.30568314,             0.80361372,             2.53214502,             7.18707085,             -4.49114513,             2.85449266,             10.14906883,             -0.31974933,             -0.84472644,             -0.52459574,             0.12921631,             -1.81390119,             2.76170087,             1.03982210,             2.91744232,             -0.29048753,             5.87453508,             -1.53684759,             1.85800636,             -0.91404629,             1.28954852,             5.11354685,             -2.47475505,             -1.33179152,             2.58552408,             1.37316465,             -3.32339454,             1.54122913,             3.24953628,             -0.29758382,             2.82391763,             -1.51142192,             -1.22699404,             6.75745535,             0.65452754,             -3.29385471,             2.06008053,             2.53172946,             -4.23532820,             -1.53909743,             -0.07010663,             -1.42173731,             7.29031610,             -0.18448229,             4.59496164,             6.73027277,             0.73441899,             0.14426160,             4.14915276,             -2.97010231,             6.05851364,             4.95218086,             -2.39145470,             2.40494704,             2.10288811,             0.53503096,             1.44511235,             6.66344261,             -3.05803776,             7.21418667,             3.30303526,             -0.24163735,             3.47409391,             3.64520788,             2.15189481,             -3.11243272,             3.62310791,             0.37379482,             0.40865007,             -0.83132005,             -4.78246069,             2.07030797,             6.51765442,             3.16178989,             5.06180477,             3.78434467,             -0.96689719,             0.35965276,             5.89967585,             1.40294051,             1.11952639,             10.59778214,             0.26739889,             -1.61297631,             6.24801159,             -0.93914318,             -0.57812452,             9.92604542,             -0.73025000,             -3.38530874,             2.45646000,             -2.47949195,             0.51638460,             10.65636063,             1.97816694,             -3.00407791,             2.66914415,             -0.81951088,             -0.23316640,             2.40737987,             -2.70007610,             1.51531935,             4.08860207,             -0.27552786,             -1.31721711,             7.11568260,             -3.33498216,             -4.02545023,             7.22675610,             -0.81690705,             -2.52689576,             1.04016697,             -0.79291463,             -0.34875512,             10.00498390,             -4.24167728,             1.46162593,             11.82569408,             -1.70359993,             -0.30161047,             16.44085884,             -0.82253462,             -0.09435523,             6.13080597,             -0.20259480,             0.68308711,             6.15663004,             -6.61776876,             0.33295766,             2.55449438,             -0.17819691,             -1.14892209,             5.56776142,             1.99279118,             1.33035934,             4.45823956,             3.34916544,             -2.59905386,             6.16164446,             -2.03881931,             -2.45273542,             12.46793365,             -2.22743297,             2.83738565,             8.48628139,             -1.39347959,             -1.30867767,             11.08041477,             -4.00363779,             2.09183025,             11.30395889,             -2.20504737,             1.37426853,             8.98735619,             1.04676604,             -0.72757077,             8.28050232,             -6.70741081,             -0.65798020,             5.68592072,             -0.60760021,             0.35854483,             6.26852131,             1.94100165,             1.32112014,             0.80987954,             -1.74617672,             -0.25434083,             7.16045523,             1.58884013,             -2.64847064,             13.14820385,             1.21393633,             -2.47258949,             9.41650105,             -0.79384226,             2.48954105,             10.95629311,             0.47723705,             4.02126694,             8.02593136,             -2.20726371,             -1.18794477,             1.50836647,             0.93118095,             -1.73513174,             8.85493565,             -2.99670315,             -0.79055870,             2.39473820,             2.05046916,             -2.38055134,             11.82299423,             0.15609655,             0.68744308,             5.66401434,             -0.69281673,             2.09855556,             7.74626589,             -0.34283102,             1.00542057,             9.95838642,             0.80161905,             2.33455157,             9.80057335,             -0.93561798,             2.56991577,             8.29711342,             0.94213426,             0.44209945,             11.70259857,             0.92710167,             2.60957146,             0.24971688,             -0.86529571,             3.78628922,             6.80884457,             -0.68178189,             2.21103406,             3.18895817,             0.60283208,             -2.92716241,             6.72060776,             -1.06625068,             2.56543374,             9.97404480,             3.58080721,             -0.94936347,             10.16736984,             -1.38464379,             1.18191063,             6.66179037,             -3.56115270,             0.32329530,             10.90870762,             2.20638227,             0.19653285,             7.34650040,             -3.63859272,             -1.03027737,             5.98829985,             -3.66606474,             -3.89746714,             8.63469028,             1.22569811,             1.63240814,             3.74385309,             0.58243257,             -0.56981975,             3.69260955,             1.00979900,             -1.44030499,             8.57058144,             -1.10648811,             1.20474911,             5.43133020,             -2.14822555,             -0.07928789,             11.25825310,             0.19645604,             -5.49546146,             10.41917038,             -0.68178523,             -2.99639869,             6.50054455,             0.46488351,             -5.42328453,             9.09500027,             -2.82107449,             0.05601966,             15.34610748,             -0.06820253,             3.86699796,             10.73316956,             -3.04795432,             -0.14702171,             5.64813185,             1.44028485,             -2.47596145,             0.07280898,             -3.03187990,             -1.35183525,             9.35835648,             2.72966957,             1.88199532,             10.36187744,             -0.22834805,             -3.26738238,             6.92025137,             -2.34061313,             4.77379704,             5.28559113,             -2.96323752,             -1.76186585,             5.94436455,             0.38647744,             -5.73869514,             6.76849556,             1.40892124,             -1.19068217,             5.37919092,             -6.65328646,             3.62782669,             12.34744644,             2.44762444,             -4.19242620,             6.14906216,             0.08121119,             0.61355996,             2.69666457,             -1.88962626,             -0.55314136,             1.84937525,             1.56048691,             1.17460012,             3.75674725,             1.06198275,             -5.74625874,             5.41645575,             -1.28946674,             -1.51689398,             4.32400894,             -0.05222082,             -4.83948946,             1.80747867,             1.63144708,             -2.73887825,             1.63975775,             -2.02163982,             -0.16210437,             2.93518686,             1.14427686,             -2.83246303,             4.79283667,             2.69697428,             -3.12678456,             -1.19225168,             -2.37022972,             -3.09429741,             1.94225383,             -1.13747168,             -2.55048585,             5.40242243,             1.12777328,             3.43713188,             3.62658787,             -2.16878843,             0.30164462,             2.97407579,             -0.07275413,             -1.31149673,             4.70066261,             -2.01323795,             4.85255766,             4.59128904,             1.68084168,             1.60336494,             6.58138466,             -1.04759812,             2.69906545,             3.55769277,             -0.74327278,             2.65819693,             5.39528131,             2.11248922,             -1.06446671,             5.24546766,             -2.43146014,             4.58907509,             0.06521678,             -2.24503994,             2.45722699,             6.94863081,             0.35258654,             2.83396196,             9.92525196,             -1.12225175,             -0.34365177,             7.19116688,             -4.39813757,             0.46517885,             13.22028065,             -2.57483673,             -6.37226963,             7.58046293,             -2.74600363,             0.42231262,             8.04881668,             0.17289802,             -0.53447008,             16.55157471,             -5.63614368,             0.39288223,             3.37079263,             1.26484549,             -0.12820500,             8.46440125,             -4.39304399,             2.97676420,             0.65650189,             0.83158541,             -1.11556435,             6.32885838,             -0.36087769,             2.80724382,             9.90292645,             1.15936041,             0.20947981,             6.91249275,             -2.67404819,             2.93782163,             6.65656614,             -2.30828357,             2.98214006,             6.80611229,             -4.93821478,             -7.66555262,             7.59763002,             -0.54159302,             3.87403512,             12.42607784,             2.59284401,             -0.23375344,             8.95293331,             -0.71807784,             0.61873478,             8.66713524,             1.24289191,             -2.37835455,             2.08071637,             -0.88315344,             -3.41891551,             6.85245323,             1.73007369,             1.02169311,             7.69170332,             -2.85411978,             2.69790673,             8.12906551,             -1.19351399,             -2.26442742,             12.26104450,             -0.75579089,             -1.73274946,             10.68729019,             2.20655656,             -0.90522075,             12.42165184,             -1.67929137,             2.44851565,             9.31565762,             -0.06645700,             1.52762020,             6.18427515,             -1.68882596,             3.70261097,             3.02252960,             -3.44125366,             -1.31575799,             2.84617424,             -0.96849400,             -4.52356243,             9.95027161,             0.19966406,             -0.78874779,             8.18595028,             -4.08300209,             1.75126517,             0.96418417,             -4.04913044,             -0.95200396,             12.03637886,             -0.03041124,             0.41642749,             8.88267422,             -3.24985337,             -2.24919462,             7.32566118,             0.16964148,             -2.74123430,             7.05264473,             -3.30191112,             0.17163286,             4.81851053,             -1.64463484,             -0.85933101,             7.29276276,             2.34066939,             -2.14860010,             3.46148157,             -0.01782012,             1.51504040,             4.79304934,             1.85281146,             -1.70663762,             6.93470192,             -4.15440845,             -1.25983095,             10.52491760,             0.42930329,             -1.85146868,             11.70042324,             -0.41704914,             3.83796859,             9.21148491,             -2.79719448,             0.79470479,             6.26926661,             -5.85230207,             3.95105338,             7.84790897,             -1.38680744,             -1.78099084,             11.95235348,             -2.99841452,             -1.34507811,             6.15714645,             -1.07552516,             -2.81228638,             1.66234732,             -4.55166149,             -1.92601109,             8.64634514,             -0.48158705,             3.31595659,             7.67371941,             2.56964207,             0.12107098,             4.56467867,             -0.93541539,             1.39432955,             11.99714088,             1.05353570,             -2.13099813,             3.67617917,             3.45895386,             1.37365830,             8.74344158,             -4.17585802,             1.43908918,             6.28764772,             3.97346330,             -0.69144285,             9.07983303,             -0.41635889,             -0.14965028,             8.85469818,             1.11306190,             2.59440994,             5.38982344,             -1.07948279,             1.37252975,             10.26984596,             -0.09318046,             2.73104119,             12.45902252,             -1.55446684,             -2.76124811,             12.19395065,             -0.51846564,             1.02764034,             11.42673588,             -0.95940983,             -0.04781032,             8.78379822,             -4.88957930,             0.32534006,             11.97696400,             -3.35108662,             1.95104563,             4.46915388,             -2.32061648,             3.45230985,             8.29983711,             2.81034684,             -2.35529327,             6.07801294,             -0.98105043,             -0.05359888,             2.52291036,             -0.01986909,             -2.35321999,             10.51954269,             2.11145401,             3.53506470,             7.29093266,             0.03721160,             -1.13496494,             7.43886709,             -5.84201956,             2.50796294,             12.14647675,             2.77490377,             -2.18896222,             6.05641937,             5.32617044,             1.04221284,             10.79106712,             -2.95749092,             -2.75414610,             11.30037117,             -3.40654182,             -2.24673963,             7.49126101,             0.70811015,             -6.18003702,             13.83951187,             -1.01204085,             1.36298490,             -1.04451632,             2.42435336,             -0.02346706,             -0.85528886,             1.04731262,             0.22192979,             4.15708160,             0.34933877,             0.04814529,             2.24107265,             0.49676740,             -1.47752666,             0.45040059,             -0.70471478,             -1.19759345,             0.21711677,             0.88461423,             -2.76830935,             5.52066898,             1.97664857,             -1.75381601,             3.45877838,             1.52617192,             -1.61350942,             0.85337949,             1.97610760,             -3.40310287,             3.40319014,             -3.38691044,             -0.71319139,             1.65463758,             -0.60680127,             -1.80700517,             8.02592373,             2.59627104,             2.65895891,             5.93043184,             -4.48425817,             3.92670918,             4.19496679,             -2.28286791,             6.41634607,             5.72330523,             1.16269672,             -0.28753027,             2.46342492,             0.36693189,             0.26712441,             6.37652683,             -2.50139046,             2.43923736,             5.56310415,             0.98065847,             1.04267502,             4.16403675,             -0.04966142,             4.40897894,             3.72905660,             -3.46129870,             3.59962773,             1.34830284,             -1.76661730,             0.47943926,             5.29946661,             -1.12711561,             1.26970029,             15.17655945,             -1.50971997,             5.81345224,             8.48562050,             -4.36049604,             2.48144460,             8.23780441,             -3.46030426,             -0.84656560,             5.94946814,             1.12747943,             -2.65683913,             8.69085693,             1.31309867,             -2.79958344,             8.76840591,             -1.56444156,             1.62710834,             2.41177034,             -0.72804940,             5.70619011,             4.67169666,             -0.86167198,             -1.83803177,             2.96346045,             2.82692933,             -2.81557131,             7.11113358,             -1.90071094,             2.54244423,             11.19284058,             -0.06298946,             -1.71517313,             12.98388577,             0.84510714,             3.00816894,             2.57200313,             0.03899818,             -1.49330592,             9.60099125,             -3.59513044,             -1.30045319,             7.09241819,             -0.65233821,             -2.33627677,             8.81366920,             0.84154201,             1.03312039,             9.85289097,             0.19351870,             1.78496623,             7.34631205,             -2.16530800,             -0.65016162,             2.46842360,             0.24016285,             -1.24308395,             4.78175163,             -0.97682536,             2.20942235,             6.68382788,             3.76786447,             -1.44454038,             6.26453733,             -3.23575711,             -2.30137897,             9.53092670,             -5.55222607,             3.25999236,             9.37559509,             1.86339056,             -0.23551451,             10.23400211,             3.93031883,             -0.52629089,             7.85724449,             -2.91549587,             4.46612740,             5.66530371,             -2.70820427,             4.81359577,             10.31247330,             1.92230141,             2.53931546,             0.74986327,             1.70303428,             0.48063779,             5.31099129,             -0.78976244,             3.75864220,             4.23051405,             2.34042454,             -7.98193836,             9.83987141,             -1.46722627,             3.54497814,             10.36455154,             -4.51249075,             0.77715248,             7.78694630,             -4.59989023,             -2.49585629,             9.90296268,             1.38535416,             1.17441154,             10.10452843,             -0.98628229,             0.60194463,             9.12639141,             -3.90754628,             2.88526392,             7.24123430,             -0.15283313,             -0.75728363,             -1.15116858,             -2.53791571,             0.77229571,             6.44114161,             0.02646767,             4.95463037,             7.21066380,             1.79384065,             0.73250306,             8.04447937,             0.32576546,             -0.79447043,             10.12717724,             2.33392906,             1.30716443,             12.36073112,             -0.36694977,             -1.20438910,             7.03105593,             0.59557682,             0.69267452,             10.18113136,             2.49944925,             -0.42229167,             8.83143330,             -1.18805945,             -2.87509322,             4.53596449,             4.09732771,             -3.39088297,             -1.02536607,             0.82119560,             -3.47302604,             9.29991817,             0.21001509,             4.97036457,             9.50018406,             1.04420102,             1.96560478,             10.74769592,             -6.22709799,             3.11690164,             5.06759691,             -1.23724771,             -3.05831861,             8.12925529,             -1.93435478,             -1.10151744,             9.32263088,             -0.04249470,             -5.98547363,             10.49398136,             0.26400441,             -0.78915191,             13.28219604,             2.99276900,             0.74853164,             2.49364305,             -3.43529654,             4.05278301,             2.13498688,             -2.35444307,             -0.79900265,             4.66968822,             -0.31095147,             3.60674143,             12.37222099,             -0.07855003,             -3.30292702,             12.15215874,             0.60886210,             2.87075138,             7.75271845,             0.38044083,             3.34402204,             6.40583277,             -0.87888050,             0.67438459,             6.91080809,             1.98332930,             -0.08303714,             8.08630371,             -0.16772588,             -2.74058914,             7.17253590,             -2.69122696,             1.48173678,             8.99470139,             -1.43302310,             -0.88651133,             2.66944790,             -0.29186964,             2.00838661,             5.09587479,             -0.76676071,             -2.88322186,             8.31110573,             -0.14550979,             -1.37726915,             10.28355122,             -1.60575438,             -0.04118848,             9.97510815,             0.14440438,             -3.24632120,             9.00034523,             4.14319563,             -1.31023729,             7.16950464,             -0.70428526,             2.01559544,             7.26155043,             2.40816474,             2.09847403,             7.31264496,             -0.75401551,             2.13392544,             7.03648758,             1.04036045,             -1.15636516,             1.09634531,             -0.06340861,             -0.58107805,             -0.65623116,             1.18972754,             -0.80717683,             1.40118241,             -0.61932516,             -3.60596156,             1.59904599,             -2.23774099,             -1.13721037,             3.89620137,             -0.09115922,             -7.51356888,             2.36975193,             -1.42520905,             -2.34173775,             3.33830214,             -2.74016523,             -3.04115510,             6.00119495,             -1.36084354,             -2.45065260,             4.56992292,             -3.02825928,             -3.74182844,             5.11069250,             -0.91531068,             -2.31385994,             1.83399653,             3.39370203,             -3.60886002});
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