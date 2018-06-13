//
//  @author raver119@gmail.com
//

#include "testlayers.h"
#include <graph/GraphState.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/LegacyTransformOp.h>
#include <ops/declarable/LegacyReduceOp.h>
#include <NativeOps.h>

using namespace nd4j;
using namespace nd4j::graph;

class GraphStateTests : public testing::Test {
public:
    GraphStateTests() {
        Environment::getInstance()->setDebug(true);
        Environment::getInstance()->setVerbose(true);
    };

    ~GraphStateTests() {
        Environment::getInstance()->setDebug(false);
        Environment::getInstance()->setVerbose(false);
    }
};

/*
 * PLAN:
 * Create GraphState
 * Register Scope
 * Add few Ops to it
 * Call conditional, that refers to scopes 
 * Check results
 */

TEST_F(GraphStateTests, Basic_Tests_1) {
    NativeOps nativeOps;

    auto state = (GraphState<float> *) nativeOps.getGraphStateFloat(117L);
    ASSERT_EQ(117L, state->id());

    // this call will create scope internally
    state->registerScope(119);

    nd4j::ops::add<float> opA;
    nd4j::ops::LegacyTransformOp<float> opB(6); // simdOps::Neg

    ArgumentsList argsA;
    ArgumentsList argsB;

    state->attachOpToScope(119, 1, &opA, argsA);
    state->attachOpToScope(119, 2, &opB, argsB);

    auto scope = state->getScope(119);
    ASSERT_TRUE(scope != nullptr);
    ASSERT_EQ(2, scope->size());    

    nativeOps.deleteGraphStateFloat(state);
}

// just separate case for doubles wrapper in NativeOps, nothing else
TEST_F(GraphStateTests, Basic_Tests_2) {
    NativeOps nativeOps;

    auto state = (GraphState<double> *) nativeOps.getGraphStateDouble(117L);
    ASSERT_EQ(117L, state->id());

    // this call will create scope internally
    state->registerScope(119);

    nd4j::ops::add<double> opA;
    nd4j::ops::LegacyTransformOp<double> opB(6); // simdOps::Neg

    ArgumentsList argsA;
    ArgumentsList argsB;

    state->attachOpToScope(119, 1, &opA, argsA);
    state->attachOpToScope(119, 2, &opB, argsB);

    auto scope = state->getScope(119);
    ASSERT_TRUE(scope != nullptr);
    ASSERT_EQ(2, scope->size());    

    nativeOps.deleteGraphStateDouble(state);
}

TEST_F(GraphStateTests, Stateful_Execution_1) {
    NativeOps nativeOps;

    auto state = nativeOps.getGraphStateFloat(117L);

    Nd4jLong scopes[] = {22, 33};
    auto status = nativeOps.execCustomOpWithScopeFloat(nullptr, state, 10, scopes, 2, nullptr, nullptr, 0, nullptr, nullptr, 0);
    ASSERT_EQ(Status::THROW(), status);

    nativeOps.deleteGraphStateFloat(state);
}

TEST_F(GraphStateTests, Stateful_Execution_2) {
    NativeOps nativeOps;

    auto state = (GraphState<float> *) nativeOps.getGraphStateFloat(117L);

    state->registerScope(22);
    state->registerScope(33);

    Nd4jLong scopes[] = {22, 33};
    auto status = nativeOps.execCustomOpWithScopeFloat(nullptr, state, 10, scopes, 2, nullptr, nullptr, 0, nullptr, nullptr, 0);
    
    // it's no-op: just LogicScope
    ASSERT_EQ(Status::OK(), status);

    nativeOps.deleteGraphStateFloat(state);
}

/**
 * This test checks WHILE loop
 */
TEST_F(GraphStateTests, Stateful_Execution_3) {
    NativeOps nativeOps;

    NDArray<float> var0('c', {2, 2}, {1, 2, 3, 4});
    NDArray<float> var1(11.0f);
    NDArray<float> var2(2.0f);

    NDArray<float> res0('c', {2, 2});
    NDArray<float> res1(0.0f);
    NDArray<float> res2(0.0f);

    // registering our GraphState holder
    auto state = (GraphState<float> *) nativeOps.getGraphStateFloat(117L);

    // we're prepping pointers to input/output buffers
    Nd4jPointer ptrBuffers[] = {(Nd4jPointer) var0.buffer(), (Nd4jPointer) var1.buffer(), (Nd4jPointer)var2.buffer()};
    Nd4jPointer ptrShapes[] = {(Nd4jPointer) var0.shapeInfo(), (Nd4jPointer) var1.shapeInfo(), (Nd4jPointer)var2.shapeInfo()};

    Nd4jPointer outBuffers[] = {(Nd4jPointer) res0.buffer(), (Nd4jPointer) res1.buffer(), (Nd4jPointer) res2.buffer()};
    Nd4jPointer outShapes[] = {(Nd4jPointer) res0.shapeInfo(), (Nd4jPointer) res1.shapeInfo(), (Nd4jPointer) res2.shapeInfo()};

    // conditional scope
    state->registerScope(22);

    nd4j::ops::LegacyReduceOp<float> op1(1);
    nd4j::ops::lt_scalar<float> op2;

    // while sum(var0) < var1
    // this op takes sum
    ArgumentsList args1({{0, 0}});

    // this op compares result of sum to input variable 0:1
    ArgumentsList args2({{1, 0}, {0, 1}});

    state->attachOpToScope(22, 1, &op1, args1);
    state->attachOpToScope(22, 2, &op2, args2);

    // body scope
    state->registerScope(33);

    // var0 + var1 + var1
    // this op is var0 + var1
    ArgumentsList args3({{0, 0}, {0, 2}});

    // this op is result of previous op + 1
    ArgumentsList args4({{3, 0}, {0, 2}});

    nd4j::ops::add<float> op3;
    nd4j::ops::add<float> op4;

    state->attachOpToScope(33, 3, &op3, args3);
    state->attachOpToScope(33, 4, &op4, args4);

    // Now we define RETURN, which returns 1 modified variable, and 2 unmodified variables
    ArgumentsList args5({{4, 0}, {0, 1}, {0, 2}});

    // so, at the end of body, initial variables will be updated
    state->defineReturn(33, 5, args5);

    Nd4jLong scopes[] = {22, 33};

    // we're executing while loop
    auto status = nativeOps.execCustomOpWithScopeFloat(nullptr, state, 0, scopes, 2, ptrBuffers, ptrShapes, 3, outBuffers, outShapes, 3);
    ASSERT_EQ(Status::OK(), status);

    // now we check provided result array
    float sum = res0.template reduceNumber<simdOps::Sum<float>>();

    /*
     * Expected result is {1, 2, 3, 4} + {2} elementwise + {2} elementwise, which gives { 5, 6, 7, 8}, and sum should be 26
     *
     */
    ASSERT_NEAR(26.0f, sum, 1e-5);

    // nd4j_printf("0 ------------------\n","");

    nativeOps.deleteGraphStateFloat(state);

    // nd4j_printf("1 ------------------\n","");
}

/**
 * This test checks CONDITIONAL execution for FALSE
 */
TEST_F(GraphStateTests, Stateful_Execution_4) {
    NativeOps nativeOps;

    NDArray<float> var0('c', {2, 2}, {1, 2, 3, 4});
    NDArray<float> var1(5.0f);

    NDArray<float> res0('c', {2, 2});
    NDArray<float> res1(0.0f);

    NDArray<float> exp('c', {2, 2}, {-4, -3, -2, -1});


    // registering our GraphState holder
    auto state = (GraphState<float> *) nativeOps.getGraphStateFloat(117L);

    // we're prepping pointers to input/output buffers
    Nd4jPointer ptrBuffers[] = {(Nd4jPointer) var0.buffer(), (Nd4jPointer) var1.buffer()};
    Nd4jPointer ptrShapes[] = {(Nd4jPointer) var0.shapeInfo(), (Nd4jPointer) var1.shapeInfo()};

    Nd4jPointer outBuffers[] = {(Nd4jPointer) res0.buffer(), (Nd4jPointer) res1.buffer()};
    Nd4jPointer outShapes[] = {(Nd4jPointer) res0.shapeInfo(), (Nd4jPointer) res1.shapeInfo()};

    // conditional scope
    state->registerScope(22);

    nd4j::ops::LegacyReduceOp<float> op1(1);
    nd4j::ops::lt_scalar<float> op2;

    // if sum(var0) < var1
    // this op takes sum
    ArgumentsList args1({{0, 0}});

    // this op compares result of sum to input variable 0:1
    ArgumentsList args2({{1, 0}, {0, 1}});

    state->attachOpToScope(22, 1, &op1, args1);
    state->attachOpToScope(22, 2, &op2, args2);

    // false scope
    state->registerScope(33);

    ArgumentsList args3({{0, 0}, {0, 1}});
    nd4j::ops::subtract<float> op3;
    state->attachOpToScope(33, 3, &op3, args3);

    // return for false scope
    ArgumentsList args10({{3, 0}, {0, 1}});
    state->defineReturn(33, 10, args10);

    // true scope
    state->registerScope(44);

    ArgumentsList args4({{0, 0}, {0, 1}});
    nd4j::ops::add<float> op4;
    state->attachOpToScope(44, 4, &op4, args4);

    // return for false scope
    ArgumentsList args20({{4, 0}, {0, 1}});
    state->defineReturn(44, 20, args20);


    Nd4jLong scopes[] = {22, 33, 44};

    // we're executing conditional op
    auto status = nativeOps.execCustomOpWithScopeFloat(nullptr, state, 20, scopes, 3, ptrBuffers, ptrShapes, 2, outBuffers, outShapes, 2);
    ASSERT_EQ(Status::OK(), status);

    ASSERT_TRUE(exp.isSameShape(&res0));
    ASSERT_TRUE(exp.equalsTo(&res0));


    nativeOps.deleteGraphStateFloat(state);
}


/**
 * This test checks CONDITIONAL execution for TRUE
 */
TEST_F(GraphStateTests, Stateful_Execution_5) {
    NativeOps nativeOps;

    NDArray<float> var0('c', {2, 2}, {1, 2, 3, 4});
    NDArray<float> var1(5.0f);

    NDArray<float> res0('c', {2, 2});
    NDArray<float> res1(0.0f);

    NDArray<float> exp('c', {2, 2}, {6, 7, 8, 9});


    // registering our GraphState holder
    auto state = (GraphState<float> *) nativeOps.getGraphStateFloat(117L);

    // we're prepping pointers to input/output buffers
    Nd4jPointer ptrBuffers[] = {(Nd4jPointer) var0.buffer(), (Nd4jPointer) var1.buffer()};
    Nd4jPointer ptrShapes[] = {(Nd4jPointer) var0.shapeInfo(), (Nd4jPointer) var1.shapeInfo()};

    Nd4jPointer outBuffers[] = {(Nd4jPointer) res0.buffer(), (Nd4jPointer) res1.buffer()};
    Nd4jPointer outShapes[] = {(Nd4jPointer) res0.shapeInfo(), (Nd4jPointer) res1.shapeInfo()};

    // conditional scope
    state->registerScope(22);

    nd4j::ops::LegacyReduceOp<float> op1(1);
    nd4j::ops::gt_scalar<float> op2;

    // if sum(var0) < var1
    // this op takes sum
    ArgumentsList args1({{0, 0}});

    // this op compares result of sum to input variable 0:1
    ArgumentsList args2({{1, 0}, {0, 1}});

    state->attachOpToScope(22, 1, &op1, args1);
    state->attachOpToScope(22, 2, &op2, args2);

    // false scope
    state->registerScope(33);

    ArgumentsList args3({{0, 0}, {0, 1}});
    nd4j::ops::subtract<float> op3;
    state->attachOpToScope(33, 3, &op3, args3);

    // return for false scope
    ArgumentsList args10({{3, 0}, {0, 1}});
    state->defineReturn(33, 10, args10);

    // true scope
    state->registerScope(44);

    ArgumentsList args4({{0, 0}, {0, 1}});
    nd4j::ops::add<float> op4;
    state->attachOpToScope(44, 4, &op4, args4);

    // return for false scope
    ArgumentsList args20({{4, 0}, {0, 1}});
    state->defineReturn(44, 20, args20);


    Nd4jLong scopes[] = {22, 33, 44};

    // we're executing conditional op
    auto status = nativeOps.execCustomOpWithScopeFloat(nullptr, state, 20, scopes, 3, ptrBuffers, ptrShapes, 2, outBuffers, outShapes, 2);
    ASSERT_EQ(Status::OK(), status);

    ASSERT_TRUE(exp.isSameShape(&res0));
    ASSERT_TRUE(exp.equalsTo(&res0));


    nativeOps.deleteGraphStateFloat(state);
}