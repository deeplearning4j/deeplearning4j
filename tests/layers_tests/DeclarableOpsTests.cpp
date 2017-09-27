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


    delete block;
    delete variableSpace;
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
	*argI = {kH,kW, sH,sW, pH,pW, dW,dH, 0};  // 0 - number of dimensions; 1,2 - kernel Height/Width; 3,4 - stride Height/Width; 5,6 - pad Height/Width; 7,8 - dilation Height/Width; 9,10 - input Height/Width; 11 - batch size; 12 - input depth; 13 - same mode;

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
	*argI = {kH,kW, sH,sW, pH,pW, dW,dH, 0};  // 0 - number of dimensions; 1,2 - kernel Height/Width; 3,4 - stride Height/Width; 5,6 - pad Height/Width; 7,8 - dilation Height/Width; 9,10 - input Height/Width; 11 - batch size; 12 - input depth; 13 - same mode;

	nd4j::ops::avgpool2d<float> pooling;
	Nd4jStatus status = pooling.execute(block);
    ASSERT_EQ(ND4J_STATUS_OK, status);
	
	NDArray<float>* result = block->getVariableSpace()->getVariable(block->getNodeId())->getNDArray();
    ASSERT_TRUE(exp.isSameShape(result));
}


//////////////////////////////////////////////////////////////////////
/*
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
	*argI = {4, kH,kW, sH,sW, pH,pW, dW,dH, iH,iW, bS, iD, 0};  // 0 - number of dimensions; 1,2 - kernel Height/Width; 3,4 - stride Height/Width; 5,6 - pad Height/Width; 7,8 - dilation Height/Width; 9,10 - input Height/Width; 11 - batch size; 12 - input depth; 13 - same mode;

	nd4j::ops::pnormpool2d<float> pooling;
	Nd4jStatus status = pooling.execute(block);
    ASSERT_EQ(ND4J_STATUS_OK, status);
	
	NDArray<float>* result = block->getVariableSpace()->getVariable(block->getNodeId())->getNDArray();
    ASSERT_TRUE(exp.isSameShape(result));
}
*/

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
	std::vector<float>* argT = block->getTArguments();	
	*argI = {4, kH,kW, sH,sW, pH,pW, dW,dH, iH,iW, bS, iD, 0, 2};		// 0 - number of dimensions; 1,2 - kernel Height/Width; 3,4 - stride Height/Width; 5,6 - pad Height/Width; 7,8 - dilation Height/Width; 9,10 - input Height/Width; 11 - batch size; 12 - input depth; 13 - same mode; 14 - pooling mode
	*argT = {8};														// 0 - divisor for pnorm mode -> extraParam0
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
	*argI = {4, kH,kW, sH,sW, pH,pW, dW,dH, iH,iW, bS, iD, 0};   // 0 - number of dimensions; 1,2 - kernel Height/Width; 3,4 - stride Height/Width; 5,6 - pad Height/Width; 7,8 - dilation Height/Width; 9,10 - input Height/Width; 11 - batch size; 12 - input depth; 13 - same mode;

	nd4j::ops::maxpool2d_bp<float> bp;
	Nd4jStatus status = bp.execute(block);
    ASSERT_EQ(ND4J_STATUS_OK, status);
	
	NDArray<float>* result = block->getVariableSpace()->getVariable(block->getNodeId())->getNDArray();
    ASSERT_TRUE(exp.isSameShape(result));

}

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
	*argI = {4, kH,kW, sH,sW, pH,pW, dW,dH, iH,iW, bS, iD, 0};   // 0 - number of dimensions; 1,2 - kernel Height/Width; 3,4 - stride Height/Width; 5,6 - pad Height/Width; 7,8 - dilation Height/Width; 9,10 - input Height/Width; 11 - batch size; 12 - input depth; 13 - same mode;

	nd4j::ops::avgpool2d_bp<float> bp;
	Nd4jStatus status = bp.execute(block);
    ASSERT_EQ(ND4J_STATUS_OK, status);
	
	NDArray<float>* result = block->getVariableSpace()->getVariable(block->getNodeId())->getNDArray();
    ASSERT_TRUE(exp.isSameShape(result));

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
