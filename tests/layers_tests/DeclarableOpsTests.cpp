//
// @author raver119@gmail.com
//

#include "testlayers.h"
#include <Block.h>
#include <Variable.h>
#include <VariableSpace.h>
#include <ops/declarable/declarable_ops.h>
#include <ops/declarable/cpu/parity_ops.h>
#include <helpers/helper_hash.h>
#include <NativeOps.h>
#include <ops/gemm.h>

using namespace nd4j::graph;

class DeclarableOpsTests : public testing::Test {
public:
    int *cShape = new int[8]{2, 2, 2, 2, 1, 0, 1, 99};
    int *fShape = new int[8]{2, 2, 2, 1, 2, 0, 1, 102};
		

};

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, BasicInitialization1) {
    auto concat = new nd4j::ops::concat<float>();
    std::string expName("concat");
    ASSERT_EQ(expName, *(concat->getOpName()));

    NDArray<float> x0(1, 5, 'c');
    NDArray<float> x1(1, 5, 'c');
    NDArray<float> x2(1, 5, 'c');
    NDArray<float> x3(1, 5, 'c');
    NDArray<float> x4(1, 5, 'c');

    x0.assign(1.0f);
    x1.assign(1.0f);
    x2.assign(1.0f);
    x3.assign(1.0f);
    x4.assign(1.0f);

    auto variableSpace = new VariableSpace<float>();

    variableSpace->putVariable(-1, &x0);
    variableSpace->putVariable(-2, &x1);
    variableSpace->putVariable(-3, &x2);
    variableSpace->putVariable(-4, &x3);
    variableSpace->putVariable(-5, &x4);

    auto nodeVar = new Variable<float>();

    variableSpace->putVariable(1, nodeVar);

    Block<float> block(1, variableSpace);

    block.fillInputs({-1, -2, -3, -4, -5});

    ASSERT_TRUE(nodeVar->getNDArray() == nullptr);

    Nd4jStatus result = concat->execute(&block);

    ASSERT_TRUE(nodeVar->getNDArray() != nullptr);

    ASSERT_EQ(25, nodeVar->getNDArray()->lengthOf());

    ASSERT_NEAR(25.0, nodeVar->getNDArray()->reduceNumber<simdOps::Sum<float>>(), 1e-5);

    ASSERT_EQ(ND4J_STATUS_OK, result);
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, BasicInitialization2) {
    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationFloat("concat");

    ASSERT_TRUE(op != nullptr);
    std::string expName("concat");
    ASSERT_EQ(expName, *(op->getOpName()));

    ASSERT_EQ(-1, op->getOpDescriptor()->getNumberOfInputs());
    ASSERT_EQ(1, op->getOpDescriptor()->getNumberOfOutputs());
}


TEST_F(DeclarableOpsTests, BasicInitialization3) {
    auto op1 = nd4j::ops::OpRegistrator::getInstance()->getOperationFloat("concat");
    std::string expName("concat");
    auto hash = nd4j::ops::HashHelper::getInstance()->getLongHash(expName);

    auto op2 = nd4j::ops::OpRegistrator::getInstance()->getOperationFloat(hash);

    ASSERT_TRUE(op1 == op2);
}


TEST_F(DeclarableOpsTests, SynonymInitialization2) {
    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationFloat("Mul");
    auto op2 = nd4j::ops::OpRegistrator::getInstance()->getOperationFloat("multiply");

    ASSERT_TRUE(op != nullptr);
    std::string expName("multiply");
    ASSERT_EQ(expName, *(op->getOpName()));
    ASSERT_TRUE(op == op2);
}


TEST_F(DeclarableOpsTests, TestTensorMmul1) {
    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> y('c', {2, 3, 4});

    for (int i = 0; i < x.lengthOf(); i++) {
        x.putScalar(i, i + 1);
        y.putScalar(i, i + 1);
    }

    NDArray<float> exp(2, 2, 'c');
    exp.putScalar(0, 650.0);
    exp.putScalar(1, 1586.0);
    exp.putScalar(2, 1586.0);
    exp.putScalar(3, 4250.0);

    VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
    variableSpace->putVariable(1, new Variable<float>());
    Block<float>* block = new Block<float>(1, variableSpace, false);
    block->fillInputs({-1, -2});
    block->getIArguments()->push_back(2);
    block->getIArguments()->push_back(1);
    block->getIArguments()->push_back(2);
    block->getIArguments()->push_back(2);
    block->getIArguments()->push_back(1);
    block->getIArguments()->push_back(2);

    nd4j::ops::tensormmul<float> tm;

    tm.execute(block);

    auto z = variableSpace->getVariable(1)->getNDArray();

    z->printBuffer("Result: ");

    ASSERT_TRUE(exp.equalsTo(z));
}

TEST_F(DeclarableOpsTests, TestTensorDot2) {
    NDArray<float> x('f', {2, 3, 4});
    NDArray<float> y('f', {2, 3, 4});

    for (int i = 0; i < x.lengthOf(); i++) {
        x.putScalar(i, i + 1);
        y.putScalar(i, i + 1);
    }

    NDArray<float> exp(2, 2, 'c');
    exp.putScalar(0, 2300.0);
    exp.putScalar(1, 2444.0);
    exp.putScalar(2, 2444.0);
    exp.putScalar(3, 2600.0);

    VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
    variableSpace->putVariable(1, new Variable<float>());
    Block<float>* block = new Block<float>(1, variableSpace, false);
    block->fillInputs({-1, -2});
    block->getIArguments()->push_back(2);
    block->getIArguments()->push_back(1);
    block->getIArguments()->push_back(2);
    block->getIArguments()->push_back(2);
    block->getIArguments()->push_back(1);
    block->getIArguments()->push_back(2);

    nd4j::ops::tensormmul<float> tm;

    tm.execute(block);

    auto z = variableSpace->getVariable(1)->getNDArray();

    z->printBuffer("Result: ");

    ASSERT_TRUE(exp.equalsTo(z));
}

TEST_F(DeclarableOpsTests, TestTensorDot3) {
    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> y('f', {2, 3, 4});

    for (int i = 0; i < x.lengthOf(); i++) {
        x.putScalar(i, i + 1);
        y.putScalar(i, i + 1);
    }

    NDArray<float> exp(2, 2, 'f');
    exp.putScalar(0, 1090.0);
    exp.putScalar(1, 2818.0);
    exp.putScalar(2, 1168.0);
    exp.putScalar(3, 3040.0);

    VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
    variableSpace->putVariable(1, new Variable<float>());
    Block<float>* block = new Block<float>(1, variableSpace, false);
    block->fillInputs({-1, -2});
    block->getIArguments()->push_back(2);
    block->getIArguments()->push_back(1);
    block->getIArguments()->push_back(2);
    block->getIArguments()->push_back(2);
    block->getIArguments()->push_back(1);
    block->getIArguments()->push_back(2);

    nd4j::ops::tensormmul<float> tm;

    tm.execute(block);

    auto z = variableSpace->getVariable(1)->getNDArray();

    z->printBuffer("Result: ");

    ASSERT_TRUE(exp.equalsTo(z));
}

TEST_F(DeclarableOpsTests, TestTensorDot4) {
    NDArray<float> x('f', {2, 3, 4});
    NDArray<float> y('c', {2, 3, 4});

    for (int i = 0; i < x.lengthOf(); i++) {
        x.putScalar(i, i + 1);
        y.putScalar(i, i + 1);
    }

    NDArray<float> exp(2, 2, 'f');
    exp.putScalar(0, 1090.0);
    exp.putScalar(1, 1168.0);
    exp.putScalar(2, 2818.0);
    exp.putScalar(3, 3040.0);

    VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
    variableSpace->putVariable(1, new Variable<float>());
    Block<float>* block = new Block<float>(1, variableSpace, false);
    block->fillInputs({-1, -2});
    block->getIArguments()->push_back(2);
    block->getIArguments()->push_back(1);
    block->getIArguments()->push_back(2);
    block->getIArguments()->push_back(2);
    block->getIArguments()->push_back(1);
    block->getIArguments()->push_back(2);

    nd4j::ops::tensormmul<float> tm;

    tm.execute(block);

    auto z = variableSpace->getVariable(1)->getNDArray();

    z->printBuffer("Result: ");

    ASSERT_TRUE(exp.equalsTo(z));
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, DivergentCheck1) {
    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationFloat("switch");

    ASSERT_TRUE(op != nullptr);
    std::string expName("Switch");
    ASSERT_EQ(expName, *(op->getOpName()));
    ASSERT_TRUE(op->getOpDescriptor()->isDivergent());
    ASSERT_EQ(2, op->getOpDescriptor()->getNumberOfOutputs());
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, AddMatrices1) {
	
	NDArray<float> x(5, 3, 'c');
	NDArray<float> y(5, 3, 'c');
	NDArray<float> exp(5, 3, 'c'); 
	x.assign(2);
	y.assign(1);
	exp.assign(3);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::add<float> addOp;
 
	addOp.execute(block);

    ASSERT_TRUE(x.equalsTo(&exp));	

}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, AddMatrixVector1) {
	
	NDArray<float> x(5, 3, 'c');
	NDArray<float> y(1, 15, 'c');
	NDArray<float> exp(5, 3, 'c'); 
	x.assign(2);
	y.assign(1);
	exp.assign(3);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::add<float> addOp;
 
	addOp.execute(block);

    ASSERT_TRUE(x.equalsTo(&exp));	
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, AddVectorVector1) {
	
	NDArray<float> x(1, 15, 'c');
	NDArray<float> y(1, 15, 'c');
	NDArray<float> exp(1, 15, 'c'); 
	x.assign(2);
	y.assign(1);
	exp.assign(3);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::add<float> addOp;
 
	addOp.execute(block);

    ASSERT_TRUE(x.equalsTo(&exp));	
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, AddMatrixScalar1) {
	
	NDArray<float> x(5, 3, 'c');
	NDArray<float> y(1, 1, 'c');
	NDArray<float> exp(5, 3, 'c'); 
	x.assign(2);
	y.assign(1);
	exp.assign(3);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::add<float> addOp;
 
	addOp.execute(block);

    ASSERT_TRUE(x.equalsTo(&exp));		
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, AddScalarScalar1) {
	
	NDArray<float> x(1, 1, 'c');
	NDArray<float> y(1, 1, 'c');
	NDArray<float> exp(1, 1, 'c'); 
	x.assign(2);
	y.assign(1);
	exp.assign(3);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::add<float> addOp;
 
	addOp.execute(block);

    ASSERT_TRUE(x.equalsTo(&exp));	
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, SubtractMatrices1) {
	
	NDArray<float> x(5, 3, 'c');
	NDArray<float> y(5, 3, 'c');
	NDArray<float> exp(5, 3, 'c'); 
	x.assign(3);
	y.assign(1);
	exp.assign(2);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::subtract<float> subOp;
 
	subOp.execute(block);

    ASSERT_TRUE(x.equalsTo(&exp));	

}

TEST_F(DeclarableOpsTests, TestRng1) {
    Nd4jIndex *buffer = new Nd4jIndex[100000];

    NativeOps nativeOps;

    nd4j::random::RandomBuffer *rng = (nd4j::random::RandomBuffer *) nativeOps.initRandom(nullptr, 123, 100000, (Nd4jPointer) buffer);

    if (rng == nullptr)
        throw "RNG initialization failed";

    NDArray<float> x(5, 3, 'c');
    VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1});
    block->setRNG(rng);
    block->getTArguments()->push_back(0.0f);
    block->getTArguments()->push_back(1.0f);

    nd4j::ops::randomuniform<float> uniform;

    Nd4jStatus status = uniform.execute(block);

    ASSERT_EQ(ND4J_STATUS_OK, status);

    ASSERT_TRUE(x.sumNumber() > 0.0);

    nativeOps.destroyRandom((Nd4jPointer) rng);
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, SubtractMatrixVector1) {
	
	NDArray<float> x(5, 3, 'c');
	NDArray<float> y(1, 15, 'c');
	NDArray<float> exp(5, 3, 'c'); 
	x.assign(3);
	y.assign(1);
	exp.assign(2);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::subtract<float> subOp;
 
	subOp.execute(block);

    ASSERT_TRUE(x.equalsTo(&exp));	

}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, MergeSumTest1) {

    NDArray<float> x(5, 5, 'c');
    NDArray<float> y(5, 5, 'c');
    NDArray<float> z(5, 5, 'c');
    NDArray<float> exp(5, 5, 'c');
    x.assign(3);
    y.assign(1);
    z.assign(2);
    exp.assign(6);

    VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
    variableSpace->putVariable(-3, &z);
    variableSpace->putVariable(1, new Variable<float>(new NDArray<float>(5,5,'c')));
    Block<float>* block = new Block<float>(1, variableSpace, false);
    block->fillInputs({-1, -2, -3});

    nd4j::ops::mergeadd<float> merge;

    merge.execute(block);

    auto res = variableSpace->getVariable(1)->getNDArray();

    res->printBuffer("Result");
    ASSERT_TRUE(res->equalsTo(&exp));

}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, ClipByValue1) {

    NDArray<float> x(5, 5, 'c');
    NDArray<float> exp(5, 5, 'c');
    x.assign(4);
    x.putScalar(0, -1);
    x.putScalar(1, 2);
    exp.assign(3);
    exp.putScalar(0, 0);
    exp.putScalar(1, 2);

    VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(1, new Variable<float>());
    Block<float>* block = new Block<float>(1, variableSpace, true);
    block->getTArguments()->push_back(0.0f);
    block->getTArguments()->push_back(3.0f);
    block->fillInputs({-1});

    nd4j::ops::clipbyvalue<float> clip;

    clip.execute(block);

    x.printBuffer("Result");
    ASSERT_TRUE(x.equalsTo(&exp));

}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, MergeMaxTest1) {

    NDArray<float> x(5, 5, 'c');
    NDArray<float> y(5, 5, 'c');
    NDArray<float> z(5, 5, 'c');
    NDArray<float> exp(5, 5, 'c');
    x.assign(3);
    y.assign(1);
    z.assign(2);
    exp.assign(3);

    VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
    variableSpace->putVariable(-3, &z);
    variableSpace->putVariable(1, new Variable<float>(new NDArray<float>(5,5,'c')));
    Block<float>* block = new Block<float>(1, variableSpace, false);
    block->fillInputs({-1, -2, -3});

    nd4j::ops::mergemax<float> merge;

    merge.execute(block);

    auto res = variableSpace->getVariable(1)->getNDArray();

    res->printBuffer("Result");
    ASSERT_TRUE(res->equalsTo(&exp));

}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, SubtractVectorVector1) {
	
	NDArray<float> x(1, 15, 'c');
	NDArray<float> y(1, 15, 'c');
	NDArray<float> exp(1, 15, 'c'); 
	x.assign(3);
	y.assign(1);
	exp.assign(2);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::subtract<float> subOp;
 
	subOp.execute(block);

    ASSERT_TRUE(x.equalsTo(&exp));	

}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, SubtractMatrixScalar1) {
	
	NDArray<float> x(5, 3, 'c');
	NDArray<float> y(1, 1, 'c');
	NDArray<float> exp(5, 3, 'c'); 
	x.assign(3);
	y.assign(1);
	exp.assign(2);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::subtract<float> subOp;
 
	subOp.execute(block);

    ASSERT_TRUE(x.equalsTo(&exp));	
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, SubtractScalarScalar1) {
	
	NDArray<float> x(1, 1, 'c');
	NDArray<float> y(1, 1, 'c');
	NDArray<float> exp(1, 1, 'c'); 
	x.assign(3);
	y.assign(1);
	exp.assign(2);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::subtract<float> subOp;
 
	subOp.execute(block);

    ASSERT_TRUE(x.equalsTo(&exp));	
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, ReverseSubtractMatrices1) {
	
	NDArray<float> x(5, 3, 'c');
	NDArray<float> y(5, 3, 'c');
	NDArray<float> exp(5, 3, 'c'); 
	x.assign(3);
	y.assign(1);
	exp.assign(-2);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::reverseSubtract<float> subOp;
 
	subOp.execute(block);

    ASSERT_TRUE(x.equalsTo(&exp));	
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, ReverseSubtractMatrixVector1) {
	
	NDArray<float> x(5, 3, 'c');
	NDArray<float> y(1, 15, 'c');
	NDArray<float> exp(5, 3, 'c'); 
	x.assign(3);
	y.assign(1);
	exp.assign(-2);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::reverseSubtract<float> subOp;
 
	subOp.execute(block);

    ASSERT_TRUE(x.equalsTo(&exp));	
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, ReverseSubtractVectorVector1) {
	
	NDArray<float> x(1, 15, 'c');
	NDArray<float> y(1, 15, 'c');
	NDArray<float> exp(1, 15, 'c'); 
	x.assign(3);
	y.assign(1);
	exp.assign(-2);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::reverseSubtract<float> subOp;
 
	subOp.execute(block);

    ASSERT_TRUE(x.equalsTo(&exp));	
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, ReverseSubtractMatrixScalar1) {
	
	NDArray<float> x(5, 3, 'c');
	NDArray<float> y(1, 1, 'c');
	NDArray<float> exp(5, 3, 'c'); 
	x.assign(3);
	y.assign(1);
	exp.assign(-2);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::reverseSubtract<float> subOp;
 
	subOp.execute(block);

    ASSERT_TRUE(x.equalsTo(&exp));	

}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, ReverseSubtractScalarScalar1) {
	
	NDArray<float> x(1, 1, 'c');
	NDArray<float> y(1, 1, 'c');
	NDArray<float> exp(1, 1, 'c'); 
	x.assign(3);
	y.assign(1);
	exp.assign(-2);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::reverseSubtract<float> subOp;
 
	subOp.execute(block);

    ASSERT_TRUE(x.equalsTo(&exp));	

}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, MultiplyMatrices1) {
	
	NDArray<float> x(5, 3, 'c');
	NDArray<float> y(5, 3, 'c');
	NDArray<float> exp(5, 3, 'c'); 
	x.assign(2);
	y.assign(3);
	exp.assign(6);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::multiply<float> mul;
 
	mul.execute(block);

    ASSERT_TRUE(x.equalsTo(&exp));	
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, MultiplyMatrixVector1) {
	
	NDArray<float> x(5, 3, 'c');
	NDArray<float> y(1, 15, 'c');
	NDArray<float> exp(5, 3, 'c'); 
	x.assign(2);
	y.assign(3);
	exp.assign(6);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::multiply<float> mul;
 
	mul.execute(block);

    ASSERT_TRUE(x.equalsTo(&exp));	
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, MultiplyVectorVector1) {
	
	NDArray<float> x(1, 15, 'c');
	NDArray<float> y(1, 15, 'c');
	NDArray<float> exp(1, 15, 'c'); 
	x.assign(2);
	y.assign(3);
	exp.assign(6);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::multiply<float> mul;
 
	mul.execute(block);

    ASSERT_TRUE(x.equalsTo(&exp));	
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, MultiplyMatrixScalar) {
	
	NDArray<float> x(5, 3, 'c');
	NDArray<float> y(1, 1, 'c');
	NDArray<float> exp(5, 3, 'c'); 
	x.assign(2);
	y.assign(3);
	exp.assign(6);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::multiply<float> mul;
 
	mul.execute(block);

    ASSERT_TRUE(x.equalsTo(&exp));	
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, MultiplyScalarScalar1) {
	
	NDArray<float> x(1, 1, 'c');
	NDArray<float> y(1, 1, 'c');
	NDArray<float> exp(1, 1, 'c'); 
	x.assign(2);
	y.assign(3);
	exp.assign(6);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::multiply<float> mul;
 
	mul.execute(block);

    ASSERT_TRUE(x.equalsTo(&exp));	
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, DivideMatrices1) {
	
	NDArray<float> x(5, 3, 'c');
	NDArray<float> y(5, 3, 'c');
	NDArray<float> exp(5, 3, 'c'); 
	x.assign(6);
	y.assign(2);
	exp.assign(3);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::divide<float> div;
 
	div.execute(block);

    ASSERT_TRUE(x.equalsTo(&exp));	
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, DivideMatrixVector1) {
	
	NDArray<float> x(5, 3, 'c');
	NDArray<float> y(1, 15, 'c');
	NDArray<float> exp(5, 3, 'c'); 
	x.assign(6);
	y.assign(2);
	exp.assign(3);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::divide<float> div;
 
	div.execute(block);

    ASSERT_TRUE(x.equalsTo(&exp));	
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, DivideVectorVector1) {
	
	NDArray<float> x(1, 15, 'c');
	NDArray<float> y(1, 15, 'c');
	NDArray<float> exp(1, 15, 'c'); 
	x.assign(6);
	y.assign(2);
	exp.assign(3);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::divide<float> div;
 
	div.execute(block);

    ASSERT_TRUE(x.equalsTo(&exp));	
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, DivideMatrixScalar1) {
	
	NDArray<float> x(5, 3, 'c');
	NDArray<float> y(1, 1, 'c');
	NDArray<float> exp(5, 3, 'c'); 
	x.assign(6);
	y.assign(2);
	exp.assign(3);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::divide<float> div;
 
	div.execute(block);

    ASSERT_TRUE(x.equalsTo(&exp));	
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, DivideScalarScalar1) {
	
	NDArray<float> x(1, 1, 'c');
	NDArray<float> y(1, 1, 'c');
	NDArray<float> exp(1, 1, 'c'); 
	x.assign(6);
	y.assign(2);
	exp.assign(3);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::divide<float> div;
 
	div.execute(block);

    ASSERT_TRUE(x.equalsTo(&exp));	
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, ReverseDivideMatrices1) {
	
	NDArray<float> x(5, 3, 'c');
	NDArray<float> y(5, 3, 'c');
	NDArray<float> exp(5, 3, 'c'); 
	x.assign(2);
	y.assign(6);
	exp.assign(3);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::reverseDivide<float> div;
 
	div.execute(block);

    ASSERT_TRUE(x.equalsTo(&exp));	
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, ReverseDivideMatrixVector1) {
	
	NDArray<float> x(5, 3, 'c');
	NDArray<float> y(1, 15, 'c');
	NDArray<float> exp(5, 3, 'c'); 
	x.assign(2);
	y.assign(6);
	exp.assign(3);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::reverseDivide<float> div;
 
	div.execute(block);

    ASSERT_TRUE(x.equalsTo(&exp));	
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, ReverseDivideVectorVector1) {
	
	NDArray<float> x(1, 15, 'c');
	NDArray<float> y(1, 15, 'c');
	NDArray<float> exp(1, 15, 'c'); 
	x.assign(2);
	y.assign(6);
	exp.assign(3);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::reverseDivide<float> div;
 
	div.execute(block);

    ASSERT_TRUE(x.equalsTo(&exp));	
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, ReverseDivideMatrixScalar1) {
	
	NDArray<float> x(5, 3, 'c');
	NDArray<float> y(1, 1, 'c');
	NDArray<float> exp(5, 3, 'c'); 
	x.assign(2);
	y.assign(6);
	exp.assign(3);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::reverseDivide<float> div;
 
	div.execute(block);

    ASSERT_TRUE(x.equalsTo(&exp));	
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, ReverseDivideScalarScalar1) {
	
	NDArray<float> x(1, 1, 'c');
	NDArray<float> y(1, 1, 'c');
	NDArray<float> exp(1, 1, 'c'); 
	x.assign(2);
	y.assign(6);
	exp.assign(3);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::reverseDivide<float> div;
 
	div.execute(block);

    ASSERT_TRUE(x.equalsTo(&exp));	
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, Reshapeas1) {
	const std::vector<int> xShape = {5,4,3};
	const std::vector<int> yShape = {3,5,4};
	
	NDArray<float> x('c', xShape);
	NDArray<float> y('f', yShape);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::reshapeas<float> reshape;
 
	reshape.execute(block);

    ASSERT_TRUE(x.isSameShape(&y));	
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, TestRegistrator1) {
    auto res = nd4j::ops::OpRegistrator::getInstance()->getAllCustomOperations();

    nd4j_printf("Ops: %s\n", res)
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, TestLegacyExecution1) {
    NativeOps nativeOps;

    auto x = new NDArray<float>(10, 10, 'c');
    x->assign(1.0f);

    auto y = new NDArray<float>(10, 10, 'c');
    y->assign(2.0f);

    auto z = new NDArray<float>(10, 10, 'c');

    auto exp = new NDArray<float>(10, 10, 'c');
    exp->assign(3.0);

    std::string opName("add");

    auto hash = nd4j::ops::HashHelper::getInstance()->getInstance()->getLongHash(opName);

    auto inputBuffers = new Nd4jPointer[2];
    auto inputShapes = new Nd4jPointer[2];

    inputBuffers[0] = (Nd4jPointer) x->_buffer;
    inputBuffers[1] = (Nd4jPointer) y->_buffer;

    inputShapes[0] = (Nd4jPointer) x->_shapeInfo;
    inputShapes[1] = (Nd4jPointer) y->_shapeInfo;

    auto outputBuffers = new Nd4jPointer[1];
    auto outputShapes = new Nd4jPointer[1];

    outputBuffers[0] = (Nd4jPointer) z->_buffer;
    outputShapes[0] = (Nd4jPointer) z->_shapeInfo;


    nativeOps.execCustomOpFloat(nullptr, hash, inputBuffers, inputShapes, 2, outputBuffers, outputShapes, 1, nullptr, 0, nullptr, 0, false);

	ASSERT_NEAR(2.0, y->meanNumber(), 1e-5);
	ASSERT_NEAR(1.0, x->meanNumber(), 1e-5);
	ASSERT_NEAR(3.0, z->meanNumber(), 1e-5);
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, TestLegacyExecution2) {
    NativeOps nativeOps;

    auto x = new NDArray<float>(10, 10, 'c');
    x->assign(1.0f);

    auto y = new NDArray<float>(10, 10, 'c');
    y->assign(2.0f);

    auto z = new NDArray<float>(10, 10, 'c');

    auto exp = new NDArray<float>(10, 10, 'c');
    exp->assign(3.0);

    std::string opName("add");

    auto hash = nd4j::ops::HashHelper::getInstance()->getInstance()->getLongHash(opName);

    auto inputBuffers = new Nd4jPointer[2];
    auto inputShapes = new Nd4jPointer[2];

    inputBuffers[0] = (Nd4jPointer) x->_buffer;
    inputBuffers[1] = (Nd4jPointer) y->_buffer;

    inputShapes[0] = (Nd4jPointer) x->_shapeInfo;
    inputShapes[1] = (Nd4jPointer) y->_shapeInfo;

    auto outputBuffers = new Nd4jPointer[1];
    auto outputShapes = new Nd4jPointer[1];

    nativeOps.execCustomOpFloat(nullptr, hash, inputBuffers, inputShapes, 2, outputBuffers, outputShapes, 1, nullptr, 0, nullptr, 0, true);

    ASSERT_NEAR(2.0, y->meanNumber(), 1e-5);
    ASSERT_NEAR(3.0, x->meanNumber(), 1e-5);
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, TestGemv1) {
	auto xBuffer = new float[15]{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f};
	auto xShape = new int[8] {2, 5, 3, 3, 1, 0, 1, 99};
	auto x = new NDArray<float>(xBuffer, xShape);

	auto yBuffer = new float[3]{2.f, 4.f, 6.f};
	auto yShape = new int[8] {2, 3, 1, 1, 1, 0, 1, 99};
	auto y = new NDArray<float>(yBuffer, yShape);

	auto z = new NDArray<float>(5, 1, 'f');

	auto expBuffer = new float[5]{28.00,  64.00,  100.00,  136.00,  172.00};
	auto exp = new NDArray<float>(expBuffer, z->_shapeInfo);

	nd4j::blas::GEMV<float>::op('f',  x->rows(), x->columns(), 1.0f, x->_buffer, y->rows(), y->_buffer, 1, 0.0, z->_buffer, 1);

	z->printBuffer();

	ASSERT_TRUE(z->equalsTo(exp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, Reshape1) {
	const std::vector<int> xShape = {5,4,3};
	const std::vector<int> yShape = {3,5,4};	
	
	NDArray<float> x('c', xShape);	
	NDArray<float> y('f', yShape);	

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1});	
	std::vector<int>* arguments = block->getIArguments();	
	*arguments = yShape;
	arguments->push_back(y.ordering());
	
	nd4j::ops::reshape<float> reshape;
	
	reshape.execute(block);

    ASSERT_TRUE(x.isSameShape(&y));	
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, Reshape2) {
	const std::vector<int> xShape = {5,4,3};
	const std::vector<int> yShape = {3,5,4};	
	
	NDArray<float> x('c', xShape);	
	NDArray<float> y('f', yShape);	

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
	variableSpace->putVariable(1, new Variable<float>());
    
	Block<float>* block = new Block<float>(1, variableSpace, false);
    block->fillInputs({-1});	
	std::vector<int>* arguments = block->getIArguments();	
	*arguments = yShape;
	arguments->push_back(y.ordering());
	
	nd4j::ops::reshape<float> reshape;
	
	Nd4jStatus status = reshape.execute(block);
	ASSERT_EQ(ND4J_STATUS_OK, status);
	NDArray<float>* result = block->getVariableSpace()->getVariable(block->getNodeId())->getNDArray();

	result->printShapeInfo();
	y.printShapeInfo();
	ASSERT_TRUE(result->isSameShape(&y));	
}

TEST_F(DeclarableOpsTests, TestScatterUpdate1) {
    NDArray<float> matrix(3, 2, 'c');
    NDArray<float> updates(2, 2, 'c');
    updates.assign(1.0);

    updates.printBuffer("Updates");

    VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &matrix);
    variableSpace->putVariable(-2, &updates);
    variableSpace->putVariable(1, new Variable<float>(&matrix));

    Block<float>* block = new Block<float>(1, variableSpace, false);
    block->fillInputs({-1, -2});

    std::vector<int>* arguments = block->getIArguments();
    arguments->push_back(0);
    arguments->push_back(1);
    arguments->push_back(1);
    arguments->push_back(2);
    arguments->push_back(1);
    arguments->push_back(2);

    nd4j::ops::scatter_update<float> op;


    Nd4jStatus result = op.execute(block);
    ASSERT_EQ(ND4J_STATUS_OK, result);

    matrix.printBuffer("Result");
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, Repeat1) {
	
	float eBuffer[8] = {1.0,2.0,1.0,2.0,3.0,4.0,3.0,4.0};
    int eShape[8] = {2, 4, 2, 2, 1, 0, 1, 99};
    NDArray<float>  x(2, 2, 'c');
    NDArray<float> exp(eBuffer, eShape);
    for (int e = 0; e < x.lengthOf(); e++)
        x.putScalar(e, e + 1);
    
	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
	variableSpace->putVariable(1, new Variable<float>());

	Block<float>* block = new Block<float>(1, variableSpace, false);
    block->fillInputs({-1});	
	std::vector<int>* arguments = block->getIArguments();	
	*arguments = {2};			// set repeats
	arguments->push_back(0);	// set dimension

	nd4j::ops::repeat<float> repeat;

	Nd4jStatus status = repeat.execute(block);
	ASSERT_EQ(ND4J_STATUS_OK, status);
	NDArray<float>* result = block->getVariableSpace()->getVariable(block->getNodeId())->getNDArray();

    ASSERT_TRUE(exp.equalsTo(result));
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, Transpose1) {

	NDArray<float> x('c', {3,5,2});
	NDArray<float> exp('f', {2,5,3});

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);

	Block<float>* block = new Block<float>(1, variableSpace, true);  // in-place
    block->fillInputs({-1});
	nd4j::ops::transpose<float> transpose;

	Nd4jStatus status = transpose.execute(block);
    ASSERT_EQ(ND4J_STATUS_OK, status);
	// ASSERT_TRUE(x.isSameShapeStrict(&exp));

	for (int e = 0; e < x.rankOf() * 2 + 2; e++) {
        ASSERT_EQ(x._shapeInfo[e], exp._shapeInfo[e]);
    }
	ASSERT_EQ(x._shapeInfo[x.rankOf() * 2 + 2],-exp._shapeInfo[x.rankOf() * 2 + 2]);
	ASSERT_EQ(x._shapeInfo[x.rankOf() * 2 + 3], exp._shapeInfo[x.rankOf() * 2 + 3]);

}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, Transpose2) {
	NDArray<float> x('c', {3,5,2});
	NDArray<float> exp('f', {2,5,3});

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
	variableSpace->putVariable(1, new Variable<float>());

	Block<float>* block = new Block<float>(1, variableSpace, false);  // not-in-place
    block->fillInputs({-1});
	nd4j::ops::transpose<float> transpose;

	Nd4jStatus  status = transpose.execute(block);
    ASSERT_EQ(ND4J_STATUS_OK, status);

	NDArray<float>* result = block->getVariableSpace()->getVariable(block->getNodeId())->getNDArray();
	// ASSERT_TRUE(result->isSameShapeStrict(&exp));
	for (int e = 0; e < result->rankOf() * 2 + 2; e++) {
        ASSERT_EQ(result->_shapeInfo[e], exp._shapeInfo[e]);
    }
	ASSERT_EQ(result->_shapeInfo[x.rankOf() * 2 + 2],-exp._shapeInfo[x.rankOf() * 2 + 2]);
	ASSERT_EQ(result->_shapeInfo[x.rankOf() * 2 + 3], exp._shapeInfo[x.rankOf() * 2 + 3]);
}


//////////////////////////////////////////////////////////////////////
// in-place
TEST_F(DeclarableOpsTests, Permute1) {

    const int shapeX[]   = {3, 5, 10, 15, 150, 15, 1, 0, 1, 99};
	const int shapeExp[] = {3, 15, 5, 10, 1, 150, 15, 0, -1, 99};    
	const std::vector<int> perm = {2, 0, 1};    
    NDArray<float>  x(shapeX);
	NDArray<float>  exp(shapeExp);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);

	Block<float>* block = new Block<float>(1, variableSpace, true);  // in-place
    block->fillInputs({-1});
	std::vector<int>* arguments = block->getIArguments();	
	*arguments = perm;		// set dimensions to be permuted
	
	nd4j::ops::permute<float> permute;
	Nd4jStatus status = permute.execute(block);	
	ASSERT_EQ(ND4J_STATUS_OK, status);
	
	ASSERT_TRUE(x.isSameShapeStrict(&exp));	
}

//////////////////////////////////////////////////////////////////////
// not-in-place
TEST_F(DeclarableOpsTests, Permute2) {		

	const int shapeX[]   = {3, 5, 10, 15, 150, 15, 1, 0, 1, 99};
	const int shapeExp[] = {3, 15, 5, 10, 1, 150, 15, 0, -1, 99};    
	const std::vector<int> perm = {2, 0, 1};    
    NDArray<float> x(shapeX);
	NDArray<float> exp(shapeExp);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
	variableSpace->putVariable(1, new Variable<float>());

	Block<float>* block = new Block<float>(1, variableSpace, false);  // not-in-place
    block->fillInputs({-1});
	std::vector<int>* arguments = block->getIArguments();	
	*arguments = perm;		// set dimensions to be permuted
	
	nd4j::ops::permute<float> permute;
	Nd4jStatus status = permute.execute(block);
	NDArray<float>* result = block->getVariableSpace()->getVariable(block->getNodeId())->getNDArray();	
	
	ASSERT_EQ(ND4J_STATUS_OK, status);	
	ASSERT_TRUE(result->isSameShapeStrict(&exp));	
}


TEST_F(DeclarableOpsTests, TestArgumentsValidation1) {
    const int shapeX[]   = {3, 5, 10, 15, 150, 15, 1, 0, 1, 99};
    const int shapeExp[] = {3, 15, 5, 10, 1, 150, 15, 0, -1, 99};
    const std::vector<int> perm = {2, 0, 1};
    NDArray<float> x(shapeX);
    NDArray<float> exp(shapeExp);

    VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(1, new Variable<float>());

    Block<float>* block = new Block<float>(1, variableSpace, false);  // not-in-place
    block->fillInputs({-1});

    nd4j::ops::permute<float> permute;
    Nd4jStatus status = permute.execute(block);

    ASSERT_TRUE(status != 0);

}


