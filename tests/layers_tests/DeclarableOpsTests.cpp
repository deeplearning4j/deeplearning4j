//
// @author raver119@gmail.com
//

#include "testlayers.h"
#include <Block.h>
#include <Variable.h>
#include <VariableSpace.h>
#include <ops/declarable/declarable_ops.h>
#include <ops/declarable/generic/parity_ops.h>
#include <ops/declarable/generic/third_party.h>
#include <helpers/helper_hash.h>
#include <NativeOps.h>
#include <ops/gemm.h>

using namespace nd4j::graph;

class DeclarableOpsTests : public testing::Test {
public:
    int *cShape = new int[8]{2, 2, 2, 2, 1, 0, 1, 99};
    int *fShape = new int[8]{2, 2, 2, 1, 2, 0, 1, 102};

	const int bS = 2;      	// batch size
	const int iD = 1;      	// input depth (number of picture channels, for example rgb=3)
	const int iH = 28;     	// picture height in pixels
	const int iW = 28;     	// picture width in pixels
	const int oD = 3;      	// output depth (= N for dense layer)
	const int kH = 5;      	// kernel height in pixels
	const int kW = 5;      	// kernel width in pixels
	const int sH = 1;      	// stride step in horizontal direction
	const int sW = 1;      	// stride step in vertical direction
	const int pH = 0;      	// padding height
	const int pW = 0; 		// padding width
	const int dH = 2;      	// dilation height
	const int dW = 2; 		// dilation width
	const int oH = (iH - kH - (kH-1)*(dH-1) + 2*pH)/sH + 1;		// output height
	const int oW = (iW - kW - (kW-1)*(dW-1) + 2*pW)/sW + 1;		// output width
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
    block.getIArguments()->push_back(1);
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

	nd4j::ops::reversesubtract<float> subOp;
 
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

	nd4j::ops::reversesubtract<float> subOp;
 
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

	nd4j::ops::reversesubtract<float> subOp;
 
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

	nd4j::ops::reversesubtract<float> subOp;
 
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

	nd4j::ops::reversesubtract<float> subOp;
 
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

TEST_F(DeclarableOpsTests, TestMatMul1) {
    auto x = new NDArray<float>(3, 5, 'c');
    for (int e = 0; e < x->lengthOf(); e++)
        x->putScalar(e, e+1);

    auto y = new NDArray<float>(5, 3, 'c');
    for (int e = 0; e < y->lengthOf(); e++)
        y->putScalar(e, e+1);

    float _expB[]{135.0, 310.0, 485.0, 150.0, 350.0, 550.0, 165.0, 390.0, 615.0};
    int _expS[] {2, 3, 3, 1, 3, 0, 1, 102};

    NDArray<float> exp(_expB, _expS);
    exp.triggerAllocationFlag(false, false);


    auto variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
    variableSpace->putVariable(1, new Variable<float>());

    auto block = new Block<float>(1, variableSpace, false);
    block->fillInputs({-1, -2});

    nd4j::ops::matmul<float> op;

    Nd4jStatus status = op.execute(block);
    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(variableSpace->hasVariable(1));

    auto result = variableSpace->getVariable(1)->getNDArray();

    ASSERT_TRUE(result->equalsTo(&exp));

    ASSERT_TRUE(block->getInnerTime() > 0);

    delete block;
    delete variableSpace;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, TestSoftMax_bp_1) {
    /*
     * INDArray input = Nd4j.linspace(1,4,4).reshape(2,2);
        INDArray inputDup = input.dup();
        Nd4j.getExecutioner().exec(new org.nd4j.linalg.api.ops.impl.transforms.gradient.SoftMaxDerivative(input,Nd4j.ones(2,2),input));
        Nd4j.getExecutioner().exec(new SoftMaxDerivative(inputDup));
        assertEquals(input,inputDup);
     */
    auto input = new NDArray<float>(2,2,'c');
    for (int e = 0; e < input->lengthOf(); e++)
        input->putScalar(e, e+1);

    auto epsilon = new NDArray<float>(2,2, 'c');
    epsilon->assign(1.0);

    auto output = new NDArray<float>(2,2,'c');

    auto exp = new NDArray<float>(2, 2, 'c');
    exp->assign(0.0);

    VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, input);
    variableSpace->putVariable(-2, epsilon);
    variableSpace->putVariable(1, output);

    Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

    nd4j::ops::softmax_bp<float> op;

    Nd4jStatus status = op.execute(block);
    ASSERT_EQ(ND4J_STATUS_OK, status);

    ASSERT_TRUE(output->equalsTo(exp));
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

	nd4j::ops::reversedivide<float> div;
 
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

	nd4j::ops::reversedivide<float> div;
 
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

	nd4j::ops::reversedivide<float> div;
 
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

	nd4j::ops::reversedivide<float> div;
 
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

	nd4j::ops::reversedivide<float> div;
 
	div.execute(block);

    ASSERT_TRUE(x.equalsTo(&exp));	
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, Reshapeas1) {
	const std::vector<int> xShape = {5,4,3};
	const std::vector<int> yShape = {3,5,4};
	
	NDArray<float> x('c', xShape);
	NDArray<float> y('f', yShape);
	NDArray<float> z;

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
	variableSpace->putVariable(1, &z);
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

    inputBuffers[0] = (Nd4jPointer) x->getBuffer();
    inputBuffers[1] = (Nd4jPointer) y->getBuffer();

    inputShapes[0] = (Nd4jPointer) x->getShapeInfo();
    inputShapes[1] = (Nd4jPointer) y->getShapeInfo();

    auto outputBuffers = new Nd4jPointer[1];
    auto outputShapes = new Nd4jPointer[1];

    outputBuffers[0] = (Nd4jPointer) z->getBuffer();
    outputShapes[0] = (Nd4jPointer) z->getShapeInfo();


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

    inputBuffers[0] = (Nd4jPointer) x->getBuffer();
    inputBuffers[1] = (Nd4jPointer) y->getBuffer();

    inputShapes[0] = (Nd4jPointer) x->getShapeInfo();
    inputShapes[1] = (Nd4jPointer) y->getShapeInfo();

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
	auto exp = new NDArray<float>(expBuffer, z->getShapeInfo());

	nd4j::blas::GEMV<float>::op('f',  x->rows(), x->columns(), 1.0f, x->getBuffer(), y->rows(), y->getBuffer(), 1, 0.0, z->getBuffer(), 1);

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
	arguments->push_back(y.ordering());
    arguments->push_back(3);
    arguments->push_back(5);
    arguments->push_back(4);
	
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
	arguments->push_back(y.ordering());
    arguments->push_back(3);
    arguments->push_back(5);
    arguments->push_back(4);
	
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

    //updates.printBuffer("Updates");

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

    //matrix.printBuffer("Result");
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
        ASSERT_EQ(x.getShapeInfo()[e], exp.getShapeInfo()[e]);
    }
	ASSERT_EQ(x.getShapeInfo()[x.rankOf() * 2 + 2],-exp.getShapeInfo()[x.rankOf() * 2 + 2]);
	ASSERT_EQ(x.getShapeInfo()[x.rankOf() * 2 + 3], exp.getShapeInfo()[x.rankOf() * 2 + 3]);

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
        ASSERT_EQ(result->getShapeInfo()[e], exp.getShapeInfo()[e]);
    }
	ASSERT_EQ(result->getShapeInfo()[x.rankOf() * 2 + 2],-exp.getShapeInfo()[x.rankOf() * 2 + 2]);
	ASSERT_EQ(result->getShapeInfo()[x.rankOf() * 2 + 3], exp.getShapeInfo()[x.rankOf() * 2 + 3]);
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

TEST_F(DeclarableOpsTests, Conv3D_ff_Test1) {
    NDArray<float> input('c', {4, 3, 3, 56, 56});
    NDArray<float> weights('f', {2, 3, 3, 5, 5});
    NDArray<float> bias('c', {1, 2});

    input.assign(1.0);
    weights.assign(2.0);
    bias.putScalar(0, 1.0f);
    bias.putScalar(1, 1.0f);

    NDArray<float> output('c', {4, 2, 1, 11, 11});

    VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &input);
    variableSpace->putVariable(-2, &weights);
    variableSpace->putVariable(-3, &bias);

    variableSpace->putVariable(1, &output);
    Block<float>* block = new Block<float>(1, variableSpace, false);  // not-in-place
    block->fillInputs({-1, -2, -3});

    block->getIArguments()->push_back(1);
    block->getIArguments()->push_back(2);
    block->getIArguments()->push_back(5);
    block->getIArguments()->push_back(5);
    block->getIArguments()->push_back(0);
    block->getIArguments()->push_back(0);
    block->getIArguments()->push_back(0);

    nd4j::ops::conv3d<float> conv3d;

    Nd4jStatus result = conv3d.execute(block);
    ASSERT_EQ(ND4J_STATUS_OK, result);

    //output.printBuffer("Result");

    ASSERT_NEAR(451.0f, output.template reduceNumber<simdOps::Mean<float>>(), 1e-5);
}

TEST_F(DeclarableOpsTests, TestReductionShape1) {
    NDArray<float> input('c', {4, 5, 5, 10, 10});

    VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &input);

    Block<float>* block = new Block<float>(1, variableSpace, false);  // not-in-place
    block->fillInputs({-1});

    // kernel params
    block->getIArguments()->push_back(1);
    block->getIArguments()->push_back(MAX_INT);

    nd4j::ops::testreduction<float> testop;

    auto shapes = testop.calculateOutputShape(new ShapeList(input.getShapeInfo()), *block);

    ASSERT_EQ(1,shapes->size());
    ASSERT_EQ(2,shapes->at(0)[0]);
    ASSERT_EQ(1,shapes->at(0)[1]);
    ASSERT_EQ(1,shapes->at(0)[2]);
}

TEST_F(DeclarableOpsTests, TestReductionShape2) {
    NDArray<float> input('c', {4, 5, 5, 10, 10});

    VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &input);

    Block<float>* block = new Block<float>(1, variableSpace, false);  // not-in-place
    block->fillInputs({-1});

    // kernel params
    block->getIArguments()->push_back(4);
    block->getIArguments()->push_back(1);
    block->getIArguments()->push_back(2);
    block->getIArguments()->push_back(3);
    block->getIArguments()->push_back(4);

    nd4j::ops::testreduction<float> testop;

    auto shapes = testop.calculateOutputShape(new ShapeList(input.getShapeInfo()), *block);

    ASSERT_EQ(1,shapes->size());
    ASSERT_EQ(2,shapes->at(0)[0]);
    ASSERT_EQ(1,shapes->at(0)[1]);
    ASSERT_EQ(4,shapes->at(0)[2]);
}

TEST_F(DeclarableOpsTests, TestCustomShape1) {
    NDArray<float> input('c', {2, 3, 4});

    VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &input);

    Block<float>* block = new Block<float>(1, variableSpace, false);  // not-in-place
    block->fillInputs({-1});

    nd4j::ops::testcustom<float> test;

    auto shapes = test.calculateOutputShape(new ShapeList(input.getShapeInfo()), *block);

    //input.printShapeInfo("input");
    //shape::printShapeInfoLinear(shape);

    ASSERT_EQ(input.getShapeInfo()[0]    , shapes->at(0)[0]);
    ASSERT_EQ(input.getShapeInfo()[1] * 2, shapes->at(0)[1]);
    ASSERT_EQ(input.getShapeInfo()[2] * 2, shapes->at(0)[2]);
    ASSERT_EQ(input.getShapeInfo()[3] * 2, shapes->at(0)[3]);
}


TEST_F(DeclarableOpsTests, DilatedMaxPool3D_ff_Test1) {
    NDArray<float> input('c', {4, 2, 1, 11, 11});

    input.assign(451.0);

    NDArray<float> output('c', {4, 2, 1, 10, 10});
    NDArray<float> indices('c', {4, 2, 1, 10, 10});


    std::pair<int, int> pair0(1,0);
    std::pair<int, int> pair1(1,1);


    VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &input);

    variableSpace->putVariable(pair0, &output);
    variableSpace->putVariable(pair1, &indices);

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

    // dilation
    block->getIArguments()->push_back(1);
    block->getIArguments()->push_back(1);
    block->getIArguments()->push_back(1);

    // ceiling
    block->getIArguments()->push_back(1);



    nd4j::ops::maxpool3d<float> maxpool3d;

    Nd4jStatus result = maxpool3d.execute(block);
    ASSERT_EQ(ND4J_STATUS_OK, result);

    //output.printBuffer("Result");

    ASSERT_NEAR(451.0f, output.template reduceNumber<simdOps::Mean<float>>(), 1e-5);


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, Sum1) {

	float xBuff[] = {1, 2, 3, 4, 5, 6, 7, 8};
	int xShape[]  = {2, 4, 2, 2, 1, 0, 1, 99};
	float expBuff[] = {16, 20};
	int expShape[]  = {2, 1, 2, 2, 1, 0, 1, 99};

	const std::vector<int> dimensions = {1,0};

	NDArray<float> x(xBuff, xShape);
	NDArray<float> z(1, 2);
	NDArray<float> exp(expBuff, expShape);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
	variableSpace->putVariable(1, &z);

	Block<float>* block = new Block<float>(1, variableSpace, false);  // not-in-place
    block->fillInputs({-1});
	std::vector<int>* arguments = block->getIArguments();
	*arguments = dimensions;

	nd4j::ops::sum<float> sum;
	Nd4jStatus status = sum.execute(block);
	NDArray<float>* result = block->getVariableSpace()->getVariable(block->getNodeId())->getNDArray();
	ASSERT_EQ(ND4J_STATUS_OK, status);
	ASSERT_TRUE(result->equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, Maxpool2d1) {

	NDArray<float> x('c', {bS,iD,iH,iW});
	NDArray<float> exp('c',{bS,iD,oH,oW});
	// NDArray<float> z('c',{bS,iD,oH,oW});

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
	// variableSpace->putVariable(1, &z);

	Block<float>* block = new Block<float>(1, variableSpace, false);
    block->fillInputs({-1});
	std::vector<int>* argI = block->getIArguments();
	*argI = { kH,kW, sH,sW, pH,pW, dW,dH, 0};  // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 - same mode;

	nd4j::ops::maxpool2d<float> pooling;
	Nd4jStatus status = pooling.execute(block);
    ASSERT_EQ(ND4J_STATUS_OK, status);
	
	NDArray<float>* result = block->getVariableSpace()->getVariable(block->getNodeId())->getNDArray();
    ASSERT_TRUE(exp.isSameShape(result));
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, Avgpool2d1) {

	NDArray<float> x('c', {bS,iD,iH,iW});
	NDArray<float> exp('c',{bS,iD,oH,oW});
	// NDArray<float> z('c',{bS,iD,oH,oW});

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
	// variableSpace->putVariable(1, &z);

	Block<float>* block = new Block<float>(1, variableSpace, false);
    block->fillInputs({-1});
	std::vector<int>* argI = block->getIArguments();
	*argI = {kH,kW, sH,sW, pH,pW, dW,dH, 0};  // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 - same mode;

	nd4j::ops::avgpool2d<float> pooling;
	Nd4jStatus status = pooling.execute(block);
    ASSERT_EQ(ND4J_STATUS_OK, status);
	
	NDArray<float>* result = block->getVariableSpace()->getVariable(block->getNodeId())->getNDArray();
    ASSERT_TRUE(exp.isSameShape(result));
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, Pnormpool2d1) {

	NDArray<float> x('c', {bS,iD,iH,iW});
	NDArray<float> exp('c',{bS,iD,oH,oW});
	// NDArray<float> z('c',{bS,iD,oH,oW});

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
	// variableSpace->putVariable(1, &z);

	Block<float>* block = new Block<float>(1, variableSpace, false);
    block->fillInputs({-1});
	std::vector<int>* argI = block->getIArguments();
	*argI = {kH,kW, sH,sW, pH,pW, dW,dH, 0, 1};  // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 - same mode; 9 - extraParam0 for pnorm case;

	nd4j::ops::pnormpool2d<float> pooling;
	Nd4jStatus status = pooling.execute(block);
    ASSERT_EQ(ND4J_STATUS_OK, status);
	
	NDArray<float>* result = block->getVariableSpace()->getVariable(block->getNodeId())->getNDArray();
    ASSERT_TRUE(exp.isSameShape(result));
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, IsMax1) {

	float xBuff[]   = {1,2,3,4,5,6,7,8,9};
	int xShape[]    = {2,3,3,3,1,0,1,99};
	float expBuff[] = {0,0,1,0,0,1,0,0,1};

	NDArray<float> x(xBuff, xShape);	
	NDArray<float> exp(expBuff, xShape);	

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);

	Block<float>* block = new Block<float>(1, variableSpace, false);
    block->fillInputs({-1});
	std::vector<int>* argI = block->getIArguments();
	*argI = {1};										// dimensions

	nd4j::ops::ismax<float> ismax;
	Nd4jStatus status = ismax.execute(block);
    ASSERT_EQ(ND4J_STATUS_OK, status);
	
	NDArray<float>* result = block->getVariableSpace()->getVariable(block->getNodeId())->getNDArray();	
    ASSERT_TRUE(exp.equalsTo(result));
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, Pooling2d1) {

	NDArray<float> x('c', {bS,iD,iH,iW});
	NDArray<float> exp('c',{bS,iD,oH,oW});
	// NDArray<float> z('c',{bS,iD,oH,oW});

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
	// variableSpace->putVariable(1, &z);

	Block<float>* block = new Block<float>(1, variableSpace, false);
    block->fillInputs({-1});
	std::vector<int>* argI = block->getIArguments();
		
	*argI = {kH,kW, sH,sW, pH,pW, dW,dH, 0, 2, 3};		// 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 - same mode; 9 - pooling mode; 10 - divisor extraParam0 for pnorm case

	nd4j::ops::pooling2d<float> pooling;
	Nd4jStatus status = pooling.execute(block);
    ASSERT_EQ(ND4J_STATUS_OK, status);
	
	NDArray<float>* result = block->getVariableSpace()->getVariable(block->getNodeId())->getNDArray();
    ASSERT_TRUE(exp.isSameShape(result));
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, MaxPool2dBP) {

	NDArray<float> input  ('c', {bS,iD,iH,iW});
	NDArray<float> epsilon('c', {bS,iD,oH,oW});
	NDArray<float> exp    ('c', {bS,iD,iH,iW});
	
	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &input);
	variableSpace->putVariable(-2, &epsilon);
	// variableSpace->putVariable(1, &z);

	Block<float>* block = new Block<float>(1, variableSpace, false);
    block->fillInputs({-1});
	block->fillInputs({-2});
	std::vector<int>* argI = block->getIArguments();
	*argI = {kH,kW, sH,sW, pH,pW, dW,dH, 0};   // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 - same mode;

	nd4j::ops::maxpool2d_bp<float> bp;
	Nd4jStatus status = bp.execute(block);
    ASSERT_EQ(ND4J_STATUS_OK, status);
	
	NDArray<float>* result = block->getVariableSpace()->getVariable(block->getNodeId())->getNDArray();
    ASSERT_TRUE(exp.isSameShape(result));

}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, AvgPool2dBP) {

	NDArray<float> input  ('c', {bS,iD,iH,iW});
	NDArray<float> epsilon('c', {bS,iD,oH,oW});
	NDArray<float> exp    ('c', {bS,iD,iH,iW});
	
	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &input);
	variableSpace->putVariable(-2, &epsilon);
	// variableSpace->putVariable(1, &z);

	Block<float>* block = new Block<float>(1, variableSpace, false);
    block->fillInputs({-1});
	block->fillInputs({-2});
	std::vector<int>* argI = block->getIArguments();
	*argI = {kH,kW, sH,sW, pH,pW, dW,dH, 0};   // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 - same mode;

	nd4j::ops::avgpool2d_bp<float> bp;
	Nd4jStatus status = bp.execute(block);
    ASSERT_EQ(ND4J_STATUS_OK, status);
	
	NDArray<float>* result = block->getVariableSpace()->getVariable(block->getNodeId())->getNDArray();
    ASSERT_TRUE(exp.isSameShape(result));

}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, PnormPool2dBP) {

	NDArray<float> input  ('c', {bS,iD,iH,iW});
	NDArray<float> epsilon('c', {bS,iD,oH,oW});
	NDArray<float> exp    ('c', {bS,iD,iH,iW});
	
	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &input);
	variableSpace->putVariable(-2, &epsilon);
	// variableSpace->putVariable(1, &z);

	Block<float>* block = new Block<float>(1, variableSpace, false);
    block->fillInputs({-1});
	block->fillInputs({-2});
	std::vector<int>* argI = block->getIArguments();
	*argI = {kH,kW, sH,sW, pH,pW, dW,dH, 0, 3};   // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 - same mode; 9 - divisor    
	std::vector<float>* argT = block->getTArguments();
	*argT = {0.000001};

	nd4j::ops::pnormpool2d_bp<float> bp;
	Nd4jStatus status = bp.execute(block);
    ASSERT_EQ(ND4J_STATUS_OK, status);
	
	NDArray<float>* result = block->getVariableSpace()->getVariable(block->getNodeId())->getNDArray();
    ASSERT_TRUE(exp.isSameShape(result));

}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, BatchNorm2D) {

    const int training = 1;
    const int isLockGammaBeta = 0;
    const int isMinibatch  = 0;
    const int K = 4;
    const float eps = 1e-5;
    const float g = 2.;
    const float b = 1.;
    const float decay = 1.;
    NDArray<float> input('c', {bS, K});
    NDArray<float> gamma('c', {1, K});
    NDArray<float> beta ('c', {1, K});
    NDArray<float> xHat ('c', {bS, K});    
    NDArray<float> globalMeanView('c', {1, K});
    NDArray<float> globalVarView('c', {1, K});
    NDArray<float> output('c', {bS, K});
    NDArray<float> outExpected('c', {bS, K});
    input(0,0)=1;input(0,1)=2;input(0,2)=3;input(0,3)=4;input(1,0)=5;input(1,1)=6;input(1,2)=7;input(1,3)=8;
    gamma.assign(1);

    VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &input);
	variableSpace->putVariable(-2, &globalMeanView);
    variableSpace->putVariable(-3, &globalVarView);
    variableSpace->putVariable(-4, &gamma);
    variableSpace->putVariable(-5, &beta);
    variableSpace->putVariable(1, &output);
    

    Block<float>* block = new Block<float>(1, variableSpace, false);
    block->fillInputs({-1,-2,-3,-4,-5});	
	std::vector<int>* argI = block->getIArguments();
    std::vector<float>* argT = block->getTArguments();
	*argI = {training, isLockGammaBeta, isMinibatch};  
    *argT = {eps, g, b, decay};  
    
    NDArray<float>* mean = input.template reduceAlongDimension<simdOps::Mean<float>>({0});        
    NDArray<float>* var  = input.template varianceAlongDimension<simdOps::SummaryStatsVariance<float>>(false, {0});
    var->template applyScalar<simdOps::Add<float>>(eps, nullptr);
    var->template applyTransform<simdOps::Sqrt<float>>(var, nullptr);            
    input.subRowVector(mean, &xHat);    
    xHat.divRowVector(var, &xHat);    
    xHat.mulRowVector(&gamma, &outExpected);
    outExpected.addRowVector(&beta, &outExpected);

    nd4j::ops::batchnorm<float> batchnorm;
	Nd4jStatus status = batchnorm.execute(block);
    ASSERT_EQ(ND4J_STATUS_OK, status);
    
	NDArray<float>* result = block->getVariableSpace()->getVariable(block->getNodeId())->getNDArray();
    ASSERT_TRUE(outExpected.equalsTo(result));

    delete mean;
    delete var;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, BatchNorm4D) {

    const int training = 1;
    const int isLockGammaBeta = 0;
    const int isMinibatch  = 0;
    const int iD = 4;
    const int iH = 3;
    const int iW = 3;
    const float eps = 1e-5;
    const float g = 2.;
    const float b = 3.;
    const float decay = 1.;
    NDArray<float> input('c', {bS, iD, iH, iW});
    NDArray<float> gamma('c', {1, iD});
    NDArray<float> beta ('c', {1, iD});
    gamma.assign(1.);
    nd4j::NDArrayFactory::linspace<float>(1.,input);   

    NDArray<float> xHat ('c', {bS, iD, iH, iW});    
    NDArray<float> globalMeanView('c', {1, iD});
    NDArray<float> globalVarView('c', {1, iD});
    NDArray<float> output('c', {bS, iD, iH, iW});
    NDArray<float> outExpected('c', {bS, iD, iH, iW});    

    VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &input);
	variableSpace->putVariable(-2, &globalMeanView);
    variableSpace->putVariable(-3, &globalVarView);
    variableSpace->putVariable(-4, &gamma);
    variableSpace->putVariable(-5, &beta);
    variableSpace->putVariable(1, &output);    

    Block<float>* block = new Block<float>(1, variableSpace, false);
    block->fillInputs({-1,-2,-3,-4,-5});	
	std::vector<int>* argI = block->getIArguments();
    std::vector<float>* argT = block->getTArguments();
	*argI = {training, isLockGammaBeta, isMinibatch};  
    *argT = {eps, g, b, decay};  
    
    NDArray<float>* mean = input.template reduceAlongDimension<simdOps::Mean<float>>({0,2,3});        
    NDArray<float>* var  = input.template varianceAlongDimension<simdOps::SummaryStatsVariance<float>>(false, {0,2,3});
    var->template applyScalar<simdOps::Add<float>>(eps, nullptr);
    var->template applyTransform<simdOps::Sqrt<float>>(var, nullptr);            
    input.template applyBroadcast<simdOps::Subtract<float>>({1}, mean, &xHat, nullptr);
    xHat.template applyBroadcast<simdOps::Divide<float>>({1}, var, &xHat, nullptr);
    xHat.template applyBroadcast<simdOps::Multiply<float>>({1}, &gamma, &outExpected, nullptr);                
    outExpected.template applyBroadcast<simdOps::Add<float>>({1}, &beta, &outExpected, nullptr);

    nd4j::ops::batchnorm<float> batchnorm;
	Nd4jStatus status = batchnorm.execute(block);
    ASSERT_EQ(ND4J_STATUS_OK, status);
    
	NDArray<float>* result = block->getVariableSpace()->getVariable(block->getNodeId())->getNDArray();
    ASSERT_TRUE(outExpected.equalsTo(result));

    delete mean;
    delete var;
}

TEST_F(DeclarableOpsTests, CompactLaunchTests1) {
    int _expS[] = {4, 2, 3, 8, 8, 192, 64, 8, 1, 0, 1, 99};
    double _expB[] = {6276.0,   12831.0,   19668.0,   26790.0,   27012.0,   20703.0,   14100.0,    7200.0,    13719.0,   28023.0,   42918.0,   58410.0,   58902.0,   45105.0,   30693.0,   15660.0,    22389.0,   45696.0,   69930.0,   95100.0,   95910.0,   73386.0,   49899.0,   25440.0,    32346.0,   65970.0,  100884.0,  137100.0,  138276.0,  105726.0,   71838.0,   36600.0,    33726.0,   68790.0,  105204.0,  142980.0,  144156.0,  110226.0,   74898.0,   38160.0,    27555.0,   56154.0,   85806.0,  116520.0,  117474.0,   89748.0,   60933.0,   31020.0,    19917.0,   40557.0,   61926.0,   84030.0,   84714.0,   64671.0,   43875.0,   22320.0,    10752.0,   21879.0,   33384.0,   45270.0,   45636.0,   34815.0,   23604.0,   12000.0,    7551.0,   15456.0,   23718.0,   32340.0,   32562.0,   24978.0,   17025.0,    8700.0,    16569.0,   33873.0,   51918.0,   70710.0,   71202.0,   54555.0,   37143.0,   18960.0,    27114.0,   55371.0,   84780.0,  115350.0,  116160.0,   88911.0,   60474.0,   30840.0,    39246.0,   80070.0,  122484.0,  166500.0,  167676.0,  128226.0,   87138.0,   44400.0,    40626.0,   82890.0,  126804.0,  172380.0,  173556.0,  132726.0,   90198.0,   45960.0,    33180.0,   67629.0,  103356.0,  140370.0,  141324.0,  107973.0,   73308.0,   37320.0,    23967.0,   48807.0,   74526.0,  101130.0,  101814.0,   77721.0,   52725.0,   26820.0,    12927.0,   26304.0,   40134.0,   54420.0,   54786.0,   41790.0,   28329.0,   14400.0,    8826.0,   18081.0,   27768.0,   37890.0,   38112.0,   29253.0,   19950.0,   10200.0,    19419.0,   39723.0,   60918.0,   83010.0,   83502.0,   64005.0,   43593.0,   22260.0,    31839.0,   65046.0,   99630.0,  135600.0,  136410.0,  104436.0,   71049.0,   36240.0,    46146.0,   94170.0,  144084.0,  195900.0,  197076.0,  150726.0,  102438.0,   52200.0,    47526.0,   96990.0,  148404.0,  201780.0,  202956.0,  155226.0,  105498.0,   53760.0,    38805.0,   79104.0,  120906.0,  164220.0,  165174.0,  126198.0,   85683.0,   43620.0,    28017.0,   57057.0,   87126.0,  118230.0,  118914.0,   90771.0,   61575.0,   31320.0,    15102.0,   30729.0,   46884.0,   63570.0,   63936.0,   48765.0,   33054.0,   16800.0,    17220.0,   34863.0,   52932.0,   71430.0,   72228.0,   54831.0,   36996.0,   18720.0,    36327.0,   73527.0,  111606.0,  150570.0,  152214.0,  115521.0,   77925.0,   39420.0,    57381.0,  116112.0,  176202.0,  237660.0,  240198.0,  182250.0,  122907.0,   62160.0,    80442.0,  162738.0,  246900.0,  332940.0,  336420.0,  255198.0,  172062.0,   87000.0,    84702.0,  171318.0,  259860.0,  350340.0,  353820.0,  268338.0,  180882.0,   91440.0,    66867.0,  135210.0,  205038.0,  276360.0,  279042.0,  211572.0,  142581.0,   72060.0,    46845.0,   94701.0,  143574.0,  193470.0,  195306.0,  148047.0,   99747.0,   50400.0,    24576.0,   49671.0,   75288.0,  101430.0,  102372.0,   77583.0,   52260.0,   26400.0,    22095.0,   44688.0,   67782.0,   91380.0,   92178.0,   69906.0,   47121.0,   23820.0,    46377.0,   93777.0,  142206.0,  191670.0,  193314.0,  146571.0,   98775.0,   49920.0,    72906.0,  147387.0,  223452.0,  301110.0,  303648.0,  230175.0,  155082.0,   78360.0,    101742.0,  205638.0,  311700.0,  419940.0,  423420.0,  320898.0,  216162.0,  109200.0,    106002.0,  214218.0,  324660.0,  437340.0,  440820.0,  334038.0,  224982.0,  113640.0,    83292.0,  168285.0,  254988.0,  343410.0,  346092.0,  262197.0,  176556.0,   89160.0,    58095.0,  117351.0,  177774.0,  239370.0,  241206.0,  182697.0,  122997.0,   62100.0,    30351.0,   61296.0,   92838.0,  124980.0,  125922.0,   95358.0,   64185.0,   32400.0,    26970.0,   54513.0,   82632.0,  111330.0,  112128.0,   84981.0,   57246.0,   28920.0,    56427.0,  114027.0,  172806.0,  232770.0,  234414.0,  177621.0,  119625.0,   60420.0,    88431.0,  178662.0,  270702.0,  364560.0,  367098.0,  278100.0,  187257.0,   94560.0,    123042.0,  248538.0,  376500.0,  506940.0,  510420.0,  386598.0,  260262.0,  131400.0,    127302.0,  257118.0,  389460.0,  524340.0,  527820.0,  399738.0,  269082.0,  135840.0,    99717.0,  201360.0,  304938.0,  410460.0,  413142.0,  312822.0,  210531.0,  106260.0,    69345.0,  140001.0,  211974.0,  285270.0,  287106.0,  217347.0,  146247.0,   73800.0,    36126.0,   72921.0,  110388.0,  148530.0,  149472.0,  113133.0,   76110.0,   38400.0,};
    NDArray<double> exp(_expB, _expS);
    exp.triggerAllocationFlag(false, false);


    NDArray<double> input('c', {2, 3, 4, 4});
    NDArray<double> weights('c', {3, 3, 5, 5});

    nd4j::NDArrayFactory::linspace<double>(1, input);
    nd4j::NDArrayFactory::linspace<double>(1, weights);

    nd4j::ops::deconv2d<double> op;
    auto result = op.execute({&input, &weights}, {}, {5, 5, 1, 1, 0, 0, 1, 1, 0});

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests, CompactLaunchTests2) {
    int _expS[] = {4, 2, 3, 8, 8, 192, 64, 8, 1, 0, 1, 99};
    double _expB[] = {6276.0,   12831.0,   19668.0,   26790.0,   27012.0,   20703.0,   14100.0,    7200.0,    13719.0,   28023.0,   42918.0,   58410.0,   58902.0,   45105.0,   30693.0,   15660.0,    22389.0,   45696.0,   69930.0,   95100.0,   95910.0,   73386.0,   49899.0,   25440.0,    32346.0,   65970.0,  100884.0,  137100.0,  138276.0,  105726.0,   71838.0,   36600.0,    33726.0,   68790.0,  105204.0,  142980.0,  144156.0,  110226.0,   74898.0,   38160.0,    27555.0,   56154.0,   85806.0,  116520.0,  117474.0,   89748.0,   60933.0,   31020.0,    19917.0,   40557.0,   61926.0,   84030.0,   84714.0,   64671.0,   43875.0,   22320.0,    10752.0,   21879.0,   33384.0,   45270.0,   45636.0,   34815.0,   23604.0,   12000.0,    7551.0,   15456.0,   23718.0,   32340.0,   32562.0,   24978.0,   17025.0,    8700.0,    16569.0,   33873.0,   51918.0,   70710.0,   71202.0,   54555.0,   37143.0,   18960.0,    27114.0,   55371.0,   84780.0,  115350.0,  116160.0,   88911.0,   60474.0,   30840.0,    39246.0,   80070.0,  122484.0,  166500.0,  167676.0,  128226.0,   87138.0,   44400.0,    40626.0,   82890.0,  126804.0,  172380.0,  173556.0,  132726.0,   90198.0,   45960.0,    33180.0,   67629.0,  103356.0,  140370.0,  141324.0,  107973.0,   73308.0,   37320.0,    23967.0,   48807.0,   74526.0,  101130.0,  101814.0,   77721.0,   52725.0,   26820.0,    12927.0,   26304.0,   40134.0,   54420.0,   54786.0,   41790.0,   28329.0,   14400.0,    8826.0,   18081.0,   27768.0,   37890.0,   38112.0,   29253.0,   19950.0,   10200.0,    19419.0,   39723.0,   60918.0,   83010.0,   83502.0,   64005.0,   43593.0,   22260.0,    31839.0,   65046.0,   99630.0,  135600.0,  136410.0,  104436.0,   71049.0,   36240.0,    46146.0,   94170.0,  144084.0,  195900.0,  197076.0,  150726.0,  102438.0,   52200.0,    47526.0,   96990.0,  148404.0,  201780.0,  202956.0,  155226.0,  105498.0,   53760.0,    38805.0,   79104.0,  120906.0,  164220.0,  165174.0,  126198.0,   85683.0,   43620.0,    28017.0,   57057.0,   87126.0,  118230.0,  118914.0,   90771.0,   61575.0,   31320.0,    15102.0,   30729.0,   46884.0,   63570.0,   63936.0,   48765.0,   33054.0,   16800.0,    17220.0,   34863.0,   52932.0,   71430.0,   72228.0,   54831.0,   36996.0,   18720.0,    36327.0,   73527.0,  111606.0,  150570.0,  152214.0,  115521.0,   77925.0,   39420.0,    57381.0,  116112.0,  176202.0,  237660.0,  240198.0,  182250.0,  122907.0,   62160.0,    80442.0,  162738.0,  246900.0,  332940.0,  336420.0,  255198.0,  172062.0,   87000.0,    84702.0,  171318.0,  259860.0,  350340.0,  353820.0,  268338.0,  180882.0,   91440.0,    66867.0,  135210.0,  205038.0,  276360.0,  279042.0,  211572.0,  142581.0,   72060.0,    46845.0,   94701.0,  143574.0,  193470.0,  195306.0,  148047.0,   99747.0,   50400.0,    24576.0,   49671.0,   75288.0,  101430.0,  102372.0,   77583.0,   52260.0,   26400.0,    22095.0,   44688.0,   67782.0,   91380.0,   92178.0,   69906.0,   47121.0,   23820.0,    46377.0,   93777.0,  142206.0,  191670.0,  193314.0,  146571.0,   98775.0,   49920.0,    72906.0,  147387.0,  223452.0,  301110.0,  303648.0,  230175.0,  155082.0,   78360.0,    101742.0,  205638.0,  311700.0,  419940.0,  423420.0,  320898.0,  216162.0,  109200.0,    106002.0,  214218.0,  324660.0,  437340.0,  440820.0,  334038.0,  224982.0,  113640.0,    83292.0,  168285.0,  254988.0,  343410.0,  346092.0,  262197.0,  176556.0,   89160.0,    58095.0,  117351.0,  177774.0,  239370.0,  241206.0,  182697.0,  122997.0,   62100.0,    30351.0,   61296.0,   92838.0,  124980.0,  125922.0,   95358.0,   64185.0,   32400.0,    26970.0,   54513.0,   82632.0,  111330.0,  112128.0,   84981.0,   57246.0,   28920.0,    56427.0,  114027.0,  172806.0,  232770.0,  234414.0,  177621.0,  119625.0,   60420.0,    88431.0,  178662.0,  270702.0,  364560.0,  367098.0,  278100.0,  187257.0,   94560.0,    123042.0,  248538.0,  376500.0,  506940.0,  510420.0,  386598.0,  260262.0,  131400.0,    127302.0,  257118.0,  389460.0,  524340.0,  527820.0,  399738.0,  269082.0,  135840.0,    99717.0,  201360.0,  304938.0,  410460.0,  413142.0,  312822.0,  210531.0,  106260.0,    69345.0,  140001.0,  211974.0,  285270.0,  287106.0,  217347.0,  146247.0,   73800.0,    36126.0,   72921.0,  110388.0,  148530.0,  149472.0,  113133.0,   76110.0,   38400.0,};
    NDArray<double> exp(_expB, _expS);
    exp.triggerAllocationFlag(false, false);


    NDArray<double> input('c', {2, 3, 4, 4});
    NDArray<double> weights('c', {3, 3, 5, 5});
    NDArray<double> z('c', {2, 3, 8, 8});

    nd4j::NDArrayFactory::linspace<double>(1, input);
    nd4j::NDArrayFactory::linspace<double>(1, weights);


    nd4j::ops::deconv2d<double> op;
    auto result = op.execute({&input, &weights}, {&z}, {}, {5, 5, 1, 1, 0, 0, 1, 1, 0});

    ASSERT_EQ(ND4J_STATUS_OK, result);

    ASSERT_TRUE(exp.isSameShape(&z));
    ASSERT_TRUE(exp.equalsTo(&z));
}


//////////////////////////////////////////////////////////////////////
// TEST_F(DeclarableOpsTests, Sum2) {

	// float xBuff[] = {1, 2, 3, 4, 5, 6, 7, 8};
	// int xShape[]  = {2, 4, 2, 2, 1, 0, 1, 99};
	// float expBuff[] = {36, 0};
	// int expShape[]  = {2, 1, 2, 2, 1, 0, 1, 99};

	// const std::vector<int> dimensions;

	// NDArray<float> x(xBuff, xShape);
	// NDArray<float> z(1, 2);
	// NDArray<float> exp(expBuff, expShape);

	// VariableSpace<float>* variableSpace = new VariableSpace<float>();
    // variableSpace->putVariable(-1, &x);
	// variableSpace->putVariable(1, &z);

	// Block<float>* block = new Block<float>(1, variableSpace, false);  // not-in-place
    // block->fillInputs({-1});
	// std::vector<int>* arguments = block->getIArguments();
	// *arguments = dimensions;

	// nd4j::ops::sum<float> sum;
	// Nd4jStatus status = sum.execute(block);
	// NDArray<float>* result = block->getVariableSpace()->getVariable(block->getNodeId())->getNDArray();
	// result->printBuffer();
	// ASSERT_EQ(ND4J_STATUS_OK, status);
	// ASSERT_TRUE(result->getScalar(0,0) == exp.getScalar(0,0));
// }

    