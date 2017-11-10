//
// @author raver119@gmail.com
//

#include "testlayers.h"
#include <Block.h>
#include <iomanip>
#include <Variable.h>
#include <VariableSpace.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/CustomOperations.h>
#include <helpers/helper_hash.h>
#include <NDArray.h>
#include <NDArrayFactory.h>
#include <NativeOps.h>
#include <ops/gemm.h>

using namespace nd4j;
using namespace nd4j::graph;

class DeclarableOpsTests : public testing::Test {
public:

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

    auto x0 = new NDArray<float>(1, 5, 'c');
    auto x1 = new NDArray<float>(1, 5, 'c');
    auto x2 = new NDArray<float>(1, 5, 'c');
    auto x3 = new NDArray<float>(1, 5, 'c');
    auto x4 = new NDArray<float>(1, 5, 'c');

    x0->assign(1.0f);
    x1->assign(1.0f);
    x2->assign(1.0f);
    x3->assign(1.0f);
    x4->assign(1.0f);

    auto variableSpace = new VariableSpace<float>();

    variableSpace->putVariable(-1, x0);
    variableSpace->putVariable(-2, x1);
    variableSpace->putVariable(-3, x2);
    variableSpace->putVariable(-4, x3);
    variableSpace->putVariable(-5, x4);

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


    delete variableSpace;
    delete concat;
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
    delete[] buffer;
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

    NDArray<float> zu(5, 5, 'c');

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

	auto variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &x);
    variableSpace->putVariable(-2, &y);
	auto block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::subtract<float> subOp;
 
	subOp.execute(block);

    ASSERT_TRUE(x.equalsTo(&exp));	
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, ReverseSubtractMatrices1) {
	
	auto x = new NDArray<float>(5, 3, 'c');
	auto y = new NDArray<float>(5, 3, 'c');
	NDArray<float> exp(5, 3, 'c'); 
	x->assign(3);
	y->assign(1);
	exp.assign(-2);

	auto variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
	auto block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::reversesubtract<float> subOp;
 
	subOp.execute(block);

    ASSERT_TRUE(x->equalsTo(&exp));

    delete variableSpace;
    delete block;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, ReverseSubtractMatrixVector1) {
	
	auto x = new NDArray<float>(5, 3, 'c');
	auto y = new NDArray<float> (1, 15, 'c');
	auto exp = new NDArray<float> (5, 3, 'c');
	x->assign(3);
	y->assign(1);
	exp->assign(-2);

	auto variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
	auto block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::reversesubtract<float> subOp;
 
	subOp.execute(block);

    ASSERT_TRUE(x->equalsTo(exp));

    delete variableSpace;
    delete block;
    delete exp;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, ReverseSubtractVectorVector1) {
	
	auto x = new NDArray<float> (1, 15, 'c');
	auto y = new NDArray<float>(1, 15, 'c');
	auto exp = new NDArray<float> (1, 15, 'c');
	x->assign(3);
	y->assign(1);
	exp->assign(-2);

	auto variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
	auto block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::reversesubtract<float> subOp;
 
	subOp.execute(block);

    ASSERT_TRUE(x->equalsTo(exp));

    delete variableSpace;
    delete block;
    delete exp;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, ReverseSubtractMatrixScalar1) {
	
	auto x = new NDArray<float>(5, 3, 'c');
	auto y = new NDArray<float>(1, 1, 'c');
	auto exp = new NDArray<float>(5, 3, 'c');
	x->assign(3);
	y->assign(1);
	exp->assign(-2);

	auto variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
    auto block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::reversesubtract<float> subOp;
 
	subOp.execute(block);

    ASSERT_TRUE(x->equalsTo(exp));

    delete variableSpace;
    delete block;
    delete exp;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, ReverseSubtractScalarScalar1) {
	
	auto x = new NDArray<float>(1, 1, 'c');
	auto y = new NDArray<float>(1, 1, 'c');
	auto exp = new NDArray<float>(1, 1, 'c');
	x->assign(3);
	y->assign(1);
	exp->assign(-2);

	auto variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
	auto block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::reversesubtract<float> subOp;
 
	subOp.execute(block);

    ASSERT_TRUE(x->equalsTo(exp));

    delete variableSpace;
    delete block;
    delete exp;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, MultiplyMatrices1) {
	
	auto x = new NDArray<float>(5, 3, 'c');
	auto y = new NDArray<float>(5, 3, 'c');
	auto exp = new NDArray<float>(5, 3, 'c');
	x->assign(2);
	y->assign(3);
	exp->assign(6);

	auto variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
    auto block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::multiply<float> mul;
 
	mul.execute(block);

    ASSERT_TRUE(x->equalsTo(exp));

    delete variableSpace;
    delete block;
    delete exp;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, MultiplyMatrixVector1) {
	
	auto x = new NDArray<float>(5, 3, 'c');
	auto y = new NDArray<float>(1, 15, 'c');
	auto exp = new NDArray<float>(5, 3, 'c');
	x->assign(2);
	y->assign(3);
	exp->assign(6);

	auto variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
	auto block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::multiply<float> mul;
 
	mul.execute(block);

    ASSERT_TRUE(x->equalsTo(exp));

    delete variableSpace;
    delete block;
    delete exp;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, MultiplyVectorVector1) {
	
	auto x = new NDArray<float>(1, 15, 'c');
    auto y = new NDArray<float>(1, 15, 'c');
    auto exp = new NDArray<float>(1, 15, 'c');
	x->assign(2);
	y->assign(3);
	exp->assign(6);

	auto variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
	auto block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::multiply<float> mul;
 
	mul.execute(block);

    ASSERT_TRUE(x->equalsTo(exp));

    delete variableSpace;
    delete block;
    delete exp;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, MultiplyMatrixScalar) {
	
	auto x = new NDArray<float>(5, 3, 'c');
    auto y = new NDArray<float>(1, 1, 'c');
    auto exp = new NDArray<float>(5, 3, 'c');
	x->assign(2);
	y->assign(3);
	exp->assign(6);

	auto variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
	auto block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::multiply<float> mul;
 
	mul.execute(block);

    ASSERT_TRUE(x->equalsTo(exp));

    delete variableSpace;
    delete block;
    delete exp;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, MultiplyScalarScalar1) {
	
	auto x = new NDArray<float>(1, 1, 'c');
    auto y = new NDArray<float>(1, 1, 'c');
    auto exp = new NDArray<float>(1, 1, 'c');
	x->assign(2);
	y->assign(3);
	exp->assign(6);

	auto variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
	auto block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::multiply<float> mul;
 
	mul.execute(block);

    ASSERT_TRUE(x->equalsTo(exp));

    delete block;
    delete variableSpace;
    delete exp;
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

    auto input = new NDArray<double>(2,2,'c');
    for (int e = 0; e < input->lengthOf(); e++)
        input->putScalar(e, e+1);

    auto epsilon = new NDArray<double>(2,2, 'c');
    epsilon->putScalar(0, 0.1f);
    epsilon->putScalar(1, 0.2f);
    epsilon->putScalar(2, 0.3f);
    epsilon->putScalar(3, 0.4f);

    auto output = new NDArray<double>(2,2,'c');
    output->assign(1.0f);

    auto exp = new NDArray<double>(2, 2, 'c');
    exp->putScalar(0, -0.019661194f);
    exp->putScalar(1, 0.019661194f);
    exp->putScalar(2, -0.019661194f);
    exp->putScalar(3, 0.019661194f);

    auto variableSpace = new VariableSpace<double>();
    variableSpace->putVariable(-1, input);
    variableSpace->putVariable(-2, epsilon);
    variableSpace->putVariable(1, output);
    //variableSpace->putVariable(42, exp);

    auto block = new Block<double>(1, variableSpace, false);
    block->fillInputs({-1, -2});

    nd4j::ops::softmax_bp<double> op;

    Nd4jStatus status = op.execute(block);
    ASSERT_EQ(ND4J_STATUS_OK, status);

    ASSERT_TRUE(output->equalsTo(exp));

    delete variableSpace;
    delete block;
    delete exp;

}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, DivideMatrices1) {
	
	auto x = new NDArray<float>(5, 3, 'c');
    auto y = new NDArray<float>(5, 3, 'c');
    auto exp = new NDArray<float>(5, 3, 'c');
	x->assign(6);
	y->assign(2);
	exp->assign(3);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::divide<float> div;
 
	div.execute(block);

    ASSERT_TRUE(x->equalsTo(exp));

    delete variableSpace;
    delete block;
    delete exp;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, DivideMatrixVector1) {
	
	auto x = new NDArray<float>(5, 3, 'c');
	auto y = new NDArray<float>(1, 15, 'c');
	NDArray<float> exp(5, 3, 'c'); 
	x->assign(6);
	y->assign(2);
	exp.assign(3);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::divide<float> div;
 
	div.execute(block);

    ASSERT_TRUE(x->equalsTo(&exp));

    delete variableSpace;
    delete block;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, DivideVectorVector1) {
	
	auto x = new NDArray<float>(1, 15, 'c');
	auto y = new NDArray<float>(1, 15, 'c');
	NDArray<float> exp(1, 15, 'c'); 
	x->assign(6);
	y->assign(2);
	exp.assign(3);

	VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
	Block<float>* block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::divide<float> div;
 
	div.execute(block);

    ASSERT_TRUE(x->equalsTo(&exp));

    delete variableSpace;
    delete block;
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
	
	auto x = new NDArray<float>(5, 1, 'c');
	auto y = new NDArray<float>(5, 1, 'c');
	nd4j::NDArray<float> exp(5, 1, 'c');
	x->assign(6);
	y->assign(2);
	exp.assign(3);

	auto variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
	auto block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::divide<float> div;
 
	div.execute(block);

    x->printBuffer("x");
    ASSERT_TRUE(x->equalsTo(&exp));

    delete variableSpace;
    delete block;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, ReverseDivideMatrices1) {
	
	auto x = new NDArray<float>(5, 3, 'c');
	auto y = new NDArray<float>(5, 3, 'c');
	NDArray<float> exp(5, 3, 'c'); 
	x->assign(2);
	y->assign(6);
	exp.assign(3);

	auto variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
	auto block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::reversedivide<float> div;
 
	div.execute(block);

    ASSERT_TRUE(x->equalsTo(&exp));

    delete variableSpace;
    delete block;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, ReverseDivideMatrixVector1) {
	
	auto x = new NDArray<float>(5, 3, 'c');
	auto y = new NDArray<float>(1, 15, 'c');
	NDArray<float> exp(5, 3, 'c'); 
	x->assign(2);
	y->assign(6);
	exp.assign(3);

	auto variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
	auto block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::reversedivide<float> div;
 
	div.execute(block);

    ASSERT_TRUE(x->equalsTo(&exp));

    delete variableSpace;
    delete block;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, ReverseDivideVectorVector1) {
	
	auto x = new NDArray<float>(1, 15, 'c');
	auto y = new NDArray<float>(1, 15, 'c');
	NDArray<float> exp(1, 15, 'c'); 
	x->assign(2);
	y->assign(6);
	exp.assign(3);

	auto variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
	auto block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::reversedivide<float> div;
 
	div.execute(block);

    ASSERT_TRUE(x->equalsTo(&exp));

    delete variableSpace;
    delete block;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, ReverseDivideMatrixScalar1) {
	
	auto x = new NDArray<float>(5, 3, 'c');
	auto y = new NDArray<float>(1, 1, 'c');
	NDArray<float> exp(5, 3, 'c'); 
	x->assign(2);
	y->assign(6);
	exp.assign(3);

	auto variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
	auto block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::reversedivide<float> div;
 
	div.execute(block);

    ASSERT_TRUE(x->equalsTo(&exp));

    delete variableSpace;
    delete block;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, ReverseDivideScalarScalar1) {
	
	auto x = new NDArray<float>(1, 1, 'c');
	auto y = new NDArray<float>(1, 1, 'c');
	NDArray<float> exp(1, 1, 'c'); 
	x->assign(2);
	y->assign(6);
	exp.assign(3);

    auto variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
	auto block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::reversedivide<float> div;
 
	div.execute(block);

    ASSERT_TRUE(x->equalsTo(&exp));

    delete variableSpace;
    delete block;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, Reshapeas1) {
	const std::vector<int> xShape = {5,4,3};
	const std::vector<int> yShape = {3,5,4};
	
	auto x = new NDArray<float>('c', xShape);
	auto y = new NDArray<float>('f', yShape);


    auto variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
    auto block = new Block<float>(1, variableSpace, true);
    block->fillInputs({-1, -2});

	nd4j::ops::reshapeas<float> reshape;
 
	reshape.execute(block);

    ASSERT_TRUE(x->isSameShape(y));

    delete variableSpace;
    delete block;
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
//	ASSERT_EQ(x.getShapeInfo()[x.rankOf() * 2 + 2],-exp.getShapeInfo()[x.rankOf() * 2 + 2]);
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
	//ASSERT_EQ(result->getShapeInfo()[x.rankOf() * 2 + 2],-exp.getShapeInfo()[x.rankOf() * 2 + 2]);
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
    //block->getIArguments()->push_back(4);
    block->getIArguments()->push_back(1);
    block->getIArguments()->push_back(2);
    block->getIArguments()->push_back(3);
    block->getIArguments()->push_back(4);

    nd4j::ops::testreduction<float> testop;

    auto shapes = testop.calculateOutputShape(new ShapeList(input.getShapeInfo()), *block);

    ASSERT_EQ(1,shapes->size());
    ASSERT_EQ(2,shapes->at(0)[0]);
    ASSERT_EQ(4,shapes->at(0)[1]);
    ASSERT_EQ(1,shapes->at(0)[2]);
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
TEST_F(DeclarableOpsTests, MaxPool2d_bp1) {

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
TEST_F(DeclarableOpsTests, CompactLaunchTests1) {
    int _expS[] = {4, 2, 3, 8, 8, 192, 64, 8, 1, 0, 1, 99};
    double _expB[] = {6276.0,   12831.0,   19668.0,   26790.0,   27012.0,   20703.0,   14100.0,    7200.0,    13719.0,   28023.0,   42918.0,   58410.0,   58902.0,   45105.0,   30693.0,   15660.0,    22389.0,   45696.0,   69930.0,   95100.0,   95910.0,   73386.0,   49899.0,   25440.0,    32346.0,   65970.0,  100884.0,  137100.0,  138276.0,  105726.0,   71838.0,   36600.0,    33726.0,   68790.0,  105204.0,  142980.0,  144156.0,  110226.0,   74898.0,   38160.0,    27555.0,   56154.0,   85806.0,  116520.0,  117474.0,   89748.0,   60933.0,   31020.0,    19917.0,   40557.0,   61926.0,   84030.0,   84714.0,   64671.0,   43875.0,   22320.0,    10752.0,   21879.0,   33384.0,   45270.0,   45636.0,   34815.0,   23604.0,   12000.0,    7551.0,   15456.0,   23718.0,   32340.0,   32562.0,   24978.0,   17025.0,    8700.0,    16569.0,   33873.0,   51918.0,   70710.0,   71202.0,   54555.0,   37143.0,   18960.0,    27114.0,   55371.0,   84780.0,  115350.0,  116160.0,   88911.0,   60474.0,   30840.0,    39246.0,   80070.0,  122484.0,  166500.0,  167676.0,  128226.0,   87138.0,   44400.0,    40626.0,   82890.0,  126804.0,  172380.0,  173556.0,  132726.0,   90198.0,   45960.0,    33180.0,   67629.0,  103356.0,  140370.0,  141324.0,  107973.0,   73308.0,   37320.0,    23967.0,   48807.0,   74526.0,  101130.0,  101814.0,   77721.0,   52725.0,   26820.0,    12927.0,   26304.0,   40134.0,   54420.0,   54786.0,   41790.0,   28329.0,   14400.0,    8826.0,   18081.0,   27768.0,   37890.0,   38112.0,   29253.0,   19950.0,   10200.0,    19419.0,   39723.0,   60918.0,   83010.0,   83502.0,   64005.0,   43593.0,   22260.0,    31839.0,   65046.0,   99630.0,  135600.0,  136410.0,  104436.0,   71049.0,   36240.0,    46146.0,   94170.0,  144084.0,  195900.0,  197076.0,  150726.0,  102438.0,   52200.0,    47526.0,   96990.0,  148404.0,  201780.0,  202956.0,  155226.0,  105498.0,   53760.0,    38805.0,   79104.0,  120906.0,  164220.0,  165174.0,  126198.0,   85683.0,   43620.0,    28017.0,   57057.0,   87126.0,  118230.0,  118914.0,   90771.0,   61575.0,   31320.0,    15102.0,   30729.0,   46884.0,   63570.0,   63936.0,   48765.0,   33054.0,   16800.0,    17220.0,   34863.0,   52932.0,   71430.0,   72228.0,   54831.0,   36996.0,   18720.0,    36327.0,   73527.0,  111606.0,  150570.0,  152214.0,  115521.0,   77925.0,   39420.0,    57381.0,  116112.0,  176202.0,  237660.0,  240198.0,  182250.0,  122907.0,   62160.0,    80442.0,  162738.0,  246900.0,  332940.0,  336420.0,  255198.0,  172062.0,   87000.0,    84702.0,  171318.0,  259860.0,  350340.0,  353820.0,  268338.0,  180882.0,   91440.0,    66867.0,  135210.0,  205038.0,  276360.0,  279042.0,  211572.0,  142581.0,   72060.0,    46845.0,   94701.0,  143574.0,  193470.0,  195306.0,  148047.0,   99747.0,   50400.0,    24576.0,   49671.0,   75288.0,  101430.0,  102372.0,   77583.0,   52260.0,   26400.0,    22095.0,   44688.0,   67782.0,   91380.0,   92178.0,   69906.0,   47121.0,   23820.0,    46377.0,   93777.0,  142206.0,  191670.0,  193314.0,  146571.0,   98775.0,   49920.0,    72906.0,  147387.0,  223452.0,  301110.0,  303648.0,  230175.0,  155082.0,   78360.0,    101742.0,  205638.0,  311700.0,  419940.0,  423420.0,  320898.0,  216162.0,  109200.0,    106002.0,  214218.0,  324660.0,  437340.0,  440820.0,  334038.0,  224982.0,  113640.0,    83292.0,  168285.0,  254988.0,  343410.0,  346092.0,  262197.0,  176556.0,   89160.0,    58095.0,  117351.0,  177774.0,  239370.0,  241206.0,  182697.0,  122997.0,   62100.0,    30351.0,   61296.0,   92838.0,  124980.0,  125922.0,   95358.0,   64185.0,   32400.0,    26970.0,   54513.0,   82632.0,  111330.0,  112128.0,   84981.0,   57246.0,   28920.0,    56427.0,  114027.0,  172806.0,  232770.0,  234414.0,  177621.0,  119625.0,   60420.0,    88431.0,  178662.0,  270702.0,  364560.0,  367098.0,  278100.0,  187257.0,   94560.0,    123042.0,  248538.0,  376500.0,  506940.0,  510420.0,  386598.0,  260262.0,  131400.0,    127302.0,  257118.0,  389460.0,  524340.0,  527820.0,  399738.0,  269082.0,  135840.0,    99717.0,  201360.0,  304938.0,  410460.0,  413142.0,  312822.0,  210531.0,  106260.0,    69345.0,  140001.0,  211974.0,  285270.0,  287106.0,  217347.0,  146247.0,   73800.0,    36126.0,   72921.0,  110388.0,  148530.0,  149472.0,  113133.0,   76110.0,   38400.0,};
    NDArray<double> exp(_expB, _expS);
    exp.triggerAllocationFlag(false, false);


    NDArray<double> input('c', {2, 3, 4, 4});
    NDArray<double> weights('c', {3, 3, 5, 5});

    nd4j::NDArrayFactory<double>::linspace(1, input);
    nd4j::NDArrayFactory<double>::linspace(1, weights);

    nd4j::ops::deconv2d<double> op;
    auto result = op.execute({&input, &weights}, {}, {5, 5, 1, 1, 0, 0, 1, 1, 0});

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, CompactLaunchTests2) {
    int _expS[] = {4, 2, 3, 8, 8, 192, 64, 8, 1, 0, 1, 99};
    double _expB[] = {6276.0,   12831.0,   19668.0,   26790.0,   27012.0,   20703.0,   14100.0,    7200.0,    13719.0,   28023.0,   42918.0,   58410.0,   58902.0,   45105.0,   30693.0,   15660.0,    22389.0,   45696.0,   69930.0,   95100.0,   95910.0,   73386.0,   49899.0,   25440.0,    32346.0,   65970.0,  100884.0,  137100.0,  138276.0,  105726.0,   71838.0,   36600.0,    33726.0,   68790.0,  105204.0,  142980.0,  144156.0,  110226.0,   74898.0,   38160.0,    27555.0,   56154.0,   85806.0,  116520.0,  117474.0,   89748.0,   60933.0,   31020.0,    19917.0,   40557.0,   61926.0,   84030.0,   84714.0,   64671.0,   43875.0,   22320.0,    10752.0,   21879.0,   33384.0,   45270.0,   45636.0,   34815.0,   23604.0,   12000.0,    7551.0,   15456.0,   23718.0,   32340.0,   32562.0,   24978.0,   17025.0,    8700.0,    16569.0,   33873.0,   51918.0,   70710.0,   71202.0,   54555.0,   37143.0,   18960.0,    27114.0,   55371.0,   84780.0,  115350.0,  116160.0,   88911.0,   60474.0,   30840.0,    39246.0,   80070.0,  122484.0,  166500.0,  167676.0,  128226.0,   87138.0,   44400.0,    40626.0,   82890.0,  126804.0,  172380.0,  173556.0,  132726.0,   90198.0,   45960.0,    33180.0,   67629.0,  103356.0,  140370.0,  141324.0,  107973.0,   73308.0,   37320.0,    23967.0,   48807.0,   74526.0,  101130.0,  101814.0,   77721.0,   52725.0,   26820.0,    12927.0,   26304.0,   40134.0,   54420.0,   54786.0,   41790.0,   28329.0,   14400.0,    8826.0,   18081.0,   27768.0,   37890.0,   38112.0,   29253.0,   19950.0,   10200.0,    19419.0,   39723.0,   60918.0,   83010.0,   83502.0,   64005.0,   43593.0,   22260.0,    31839.0,   65046.0,   99630.0,  135600.0,  136410.0,  104436.0,   71049.0,   36240.0,    46146.0,   94170.0,  144084.0,  195900.0,  197076.0,  150726.0,  102438.0,   52200.0,    47526.0,   96990.0,  148404.0,  201780.0,  202956.0,  155226.0,  105498.0,   53760.0,    38805.0,   79104.0,  120906.0,  164220.0,  165174.0,  126198.0,   85683.0,   43620.0,    28017.0,   57057.0,   87126.0,  118230.0,  118914.0,   90771.0,   61575.0,   31320.0,    15102.0,   30729.0,   46884.0,   63570.0,   63936.0,   48765.0,   33054.0,   16800.0,    17220.0,   34863.0,   52932.0,   71430.0,   72228.0,   54831.0,   36996.0,   18720.0,    36327.0,   73527.0,  111606.0,  150570.0,  152214.0,  115521.0,   77925.0,   39420.0,    57381.0,  116112.0,  176202.0,  237660.0,  240198.0,  182250.0,  122907.0,   62160.0,    80442.0,  162738.0,  246900.0,  332940.0,  336420.0,  255198.0,  172062.0,   87000.0,    84702.0,  171318.0,  259860.0,  350340.0,  353820.0,  268338.0,  180882.0,   91440.0,    66867.0,  135210.0,  205038.0,  276360.0,  279042.0,  211572.0,  142581.0,   72060.0,    46845.0,   94701.0,  143574.0,  193470.0,  195306.0,  148047.0,   99747.0,   50400.0,    24576.0,   49671.0,   75288.0,  101430.0,  102372.0,   77583.0,   52260.0,   26400.0,    22095.0,   44688.0,   67782.0,   91380.0,   92178.0,   69906.0,   47121.0,   23820.0,    46377.0,   93777.0,  142206.0,  191670.0,  193314.0,  146571.0,   98775.0,   49920.0,    72906.0,  147387.0,  223452.0,  301110.0,  303648.0,  230175.0,  155082.0,   78360.0,    101742.0,  205638.0,  311700.0,  419940.0,  423420.0,  320898.0,  216162.0,  109200.0,    106002.0,  214218.0,  324660.0,  437340.0,  440820.0,  334038.0,  224982.0,  113640.0,    83292.0,  168285.0,  254988.0,  343410.0,  346092.0,  262197.0,  176556.0,   89160.0,    58095.0,  117351.0,  177774.0,  239370.0,  241206.0,  182697.0,  122997.0,   62100.0,    30351.0,   61296.0,   92838.0,  124980.0,  125922.0,   95358.0,   64185.0,   32400.0,    26970.0,   54513.0,   82632.0,  111330.0,  112128.0,   84981.0,   57246.0,   28920.0,    56427.0,  114027.0,  172806.0,  232770.0,  234414.0,  177621.0,  119625.0,   60420.0,    88431.0,  178662.0,  270702.0,  364560.0,  367098.0,  278100.0,  187257.0,   94560.0,    123042.0,  248538.0,  376500.0,  506940.0,  510420.0,  386598.0,  260262.0,  131400.0,    127302.0,  257118.0,  389460.0,  524340.0,  527820.0,  399738.0,  269082.0,  135840.0,    99717.0,  201360.0,  304938.0,  410460.0,  413142.0,  312822.0,  210531.0,  106260.0,    69345.0,  140001.0,  211974.0,  285270.0,  287106.0,  217347.0,  146247.0,   73800.0,    36126.0,   72921.0,  110388.0,  148530.0,  149472.0,  113133.0,   76110.0,   38400.0,};
    NDArray<double> exp(_expB, _expS);
    exp.triggerAllocationFlag(false, false);


    NDArray<double> input('c', {2, 3, 4, 4});
    NDArray<double> weights('c', {3, 3, 5, 5});
    NDArray<double> z('c', {2, 3, 8, 8});

    nd4j::NDArrayFactory<double>::linspace(1, input);
    nd4j::NDArrayFactory<double>::linspace(1, weights);


    nd4j::ops::deconv2d<double> op;
    auto result = op.execute({&input, &weights}, {&z}, {}, {5, 5, 1, 1, 0, 0, 1, 1, 0});

    ASSERT_EQ(ND4J_STATUS_OK, result);

    ASSERT_TRUE(exp.isSameShape(&z));
    ASSERT_TRUE(exp.equalsTo(&z));
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, BatchNorm2D) {

    const int training = 1;
    const int isLockGammaBeta = 0;
    const int isMinibatch  = 0;
    const int bS = 2;
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
    NDArray<float>* std = new NDArray<float>(var->getShapeInfo());
    var->template applyTransform<simdOps::Sqrt<float>>(std, nullptr);            
    input.subRowVector(mean, &xHat);    
    xHat.divRowVector(std, &xHat);    
    xHat.mulRowVector(&gamma, &outExpected);
    outExpected.addRowVector(&beta, &outExpected);

    nd4j::ops::batchnorm<float> batchnorm;
    Nd4jStatus status = batchnorm.execute(block);
    ASSERT_EQ(ND4J_STATUS_OK, status);
    
    NDArray<float>* result = block->getVariableSpace()->getVariable(block->getNodeId())->getNDArray();
    ASSERT_TRUE(outExpected.equalsTo(result));

    delete mean;
    delete var;
    delete std;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, BatchNorm4D) {

    const int training = 1;
    const int isLockGammaBeta = 0;
    const int isMinibatch  = 0;
    const int bS = 2;
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
    nd4j::NDArrayFactory<float>::linspace(1.,input);   

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

    
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, BatchNorm2D_BP) {
    
    const int isLockGammaBeta = 0;
    const int bS = 2; 
    const int K = 4;
    const double eps = 1e-5;
    const int effectiveBatchSize = bS;
    const std::initializer_list<int> dimensions = {0};
    NDArray<double>* input = new NDArray<double>('c', {bS, K});
    NDArray<double>* epsilon = new NDArray<double>('c', {bS, K});
    NDArray<double>* gamma = new NDArray<double>('c', {1, K});    
    NDArray<double>* dGlobalMeanView = new NDArray<double>('c', {1, K});
    NDArray<double>* dGlobalVarView = new NDArray<double>('c', {1, K});
    NDArray<double>* outEpsilon = new NDArray<double>('c', {bS, K});

    VariableSpace<double>* variableSpace = new VariableSpace<double>();
    variableSpace->putVariable(-1, input);
    variableSpace->putVariable(-2, epsilon);
    variableSpace->putVariable(-3, gamma);
    variableSpace->putVariable(-4, dGlobalMeanView);
    variableSpace->putVariable(-5, dGlobalVarView);
    variableSpace->putVariable(1, outEpsilon);
    Block<double>* block = new Block<double>(1, variableSpace, false);
    block->fillInputs({-1,-2,-3,-4,-5});    
    std::vector<int>* argI = block->getIArguments();    
    *argI = {isLockGammaBeta};       

    nd4j::NDArrayFactory<double>::linspace(10., *input);
    nd4j::NDArrayFactory<double>::linspace(1., *epsilon);    
    gamma->assign(1);
    NDArray<double>* mean = input->template reduceAlongDimension<simdOps::Mean<double>>(dimensions);        
    NDArray<double>* var  = input->template varianceAlongDimension<simdOps::SummaryStatsVariance<double>>(false, dimensions);
    var->template applyScalar<simdOps::Add<double>>(eps, nullptr);
    NDArray<double>* std = new NDArray<double>(var->getShapeInfo());
    var->template applyTransform<simdOps::Sqrt<double>>(std, nullptr);            
    NDArray<double>* xHat = new NDArray<double>(input->getShapeInfo());    
    input->subRowVector(mean, xHat);
    xHat->divRowVector(std, xHat);    

    NDArray<double>* temp1 = new NDArray<double>(input->getShapeInfo());
    NDArray<double>* temp2 = new NDArray<double>(var->getShapeInfo());

    epsilon->template applyPairwiseTransform<simdOps::Multiply<double>>(xHat, temp1, nullptr);
    NDArray<double>* dldgammaExp = temp1->sum(dimensions);
    NDArray<double>* dldbetaExp = epsilon->sum(dimensions);        
    NDArray<double>* dldxhat = new NDArray<double>(epsilon->getShapeInfo());                
    epsilon->mulRowVector(gamma, dldxhat);
        
    input->subRowVector(mean, temp1);
    dldxhat->template applyPairwiseTransform<simdOps::Multiply<double>>(temp1, temp1, nullptr);
    temp1->template applyScalar<simdOps::Multiply<double>>(-0.5);                        
    double powParams1[] = {-1.5};
    var->template applyTransform<simdOps::Pow<double>>(temp2, powParams1);
    temp1->mulRowVector(temp2, temp1);
    NDArray<double>* dldvar = temp1->sum(dimensions);

    double powParams2[] = {-0.5};
    var->template applyTransform<simdOps::Pow<double>>(temp2, powParams2);
    dldxhat->mulRowVector(temp2, temp1);
    temp1->template applyTransform<simdOps::Neg<double>>();                
    NDArray<double>* dldmu = temp1->sum(dimensions);
    input->subRowVector(mean, temp1);
    temp1->template applyScalar<simdOps::Multiply<double>>(-2.);
    NDArray<double>* temp3 = temp1->sum(dimensions);
    temp3->template applyScalar<simdOps::Divide<double>>((double)effectiveBatchSize);
    dldvar->template applyPairwiseTransform<simdOps::Multiply<double>>(temp3, temp3, nullptr);
    dldmu->template applyPairwiseTransform<simdOps::Add<double>>(temp3, dldmu, nullptr);

    NDArray<double>* dldinExp = new NDArray<double>(epsilon->getShapeInfo());
    dldxhat->mulRowVector(temp2, dldinExp);     
    input->subRowVector(mean, temp1);
    temp1->template applyScalar<simdOps::Multiply<double>>(2./effectiveBatchSize);
    temp1->mulRowVector(dldvar, temp1);
    dldinExp->template applyPairwiseTransform<simdOps::Add<double>>(temp1, nullptr);
    dldmu->template applyScalar<simdOps::Multiply<double>>(1./effectiveBatchSize, temp2, nullptr);
    dldinExp->addRowVector(temp2, dldinExp);

    nd4j::ops::batchnorm_bp<double> batchnorm_bp;
    Nd4jStatus status = batchnorm_bp.execute(block);
    ASSERT_EQ(ND4J_STATUS_OK, status);
    
    NDArray<double>* result = block->getVariableSpace()->getVariable(block->getNodeId())->getNDArray();
    ASSERT_TRUE(dldinExp->equalsTo(result));
    
    delete temp1;
    delete temp2;
    delete temp3;    
    delete dldbetaExp;
    delete dldgammaExp;
    delete dldxhat;
    delete var;
    delete mean;
    delete dldvar;
    delete dldmu;
    delete dldinExp;
    delete xHat;
    delete std;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, BatchNorm4D_BP) {
    
    const int isLockGammaBeta = 0;
    const int bS = 2; 
    const int iD = 4;
    const int iH = 3;
    const int iW = 3;
    const double eps = 1e-5;
    const std::initializer_list<int> dimensions = {0,2,3};
    const int effectiveBatchSize = bS*iH*iW;

    NDArray<double>* input = new NDArray<double>('c', {bS, iD, iH, iW});
    NDArray<double>* epsilon = new NDArray<double>('c', {bS, iD, iH, iW});
    NDArray<double>* gamma = new NDArray<double>('c', {1, iD});    
    NDArray<double>* dGlobalMeanView = new NDArray<double>('c', {1, iD});
    NDArray<double>* dGlobalVarView = new NDArray<double>('c', {1, iD});
    NDArray<double>* outEpsilon = new NDArray<double>('c', {bS, iD, iH, iW});
    
    VariableSpace<double>* variableSpace = new VariableSpace<double>();
    variableSpace->putVariable(-1, input);
    variableSpace->putVariable(-2, epsilon);
    variableSpace->putVariable(-3, gamma);
    variableSpace->putVariable(-4, dGlobalMeanView);
    variableSpace->putVariable(-5, dGlobalVarView);
    variableSpace->putVariable(1, outEpsilon);
    Block<double>* block = new Block<double>(1, variableSpace, false);
    block->fillInputs({-1,-2,-3,-4,-5});    
    std::vector<int>* argI = block->getIArguments();    
    *argI = {isLockGammaBeta};       

    nd4j::NDArrayFactory<double>::linspace(80., *input);
    nd4j::NDArrayFactory<double>::linspace(1., *epsilon);    
    gamma->assign(1);
    NDArray<double>* mean = input->template reduceAlongDimension<simdOps::Mean<double>>(dimensions);        
    NDArray<double>* var  = input->template varianceAlongDimension<simdOps::SummaryStatsVariance<double>>(false, dimensions);
    var->template applyScalar<simdOps::Add<double>>(eps, nullptr);
    NDArray<double>* std = new NDArray<double>(var->getShapeInfo());
    var->template applyTransform<simdOps::Sqrt<double>>(std, nullptr);            
    NDArray<double>* xHat = new NDArray<double>(input->getShapeInfo());    
    input->template applyBroadcast<simdOps::Subtract<double>>({1}, mean, xHat, nullptr);
    xHat->template applyBroadcast<simdOps::Divide<double>>({1}, std, xHat, nullptr);      

    NDArray<double>* temp1 = new NDArray<double>(input->getShapeInfo());
    NDArray<double>* temp2 = new NDArray<double>(var->getShapeInfo());

    epsilon->template applyPairwiseTransform<simdOps::Multiply<double>>(xHat, temp1, nullptr);
    NDArray<double>* dldgammaExp = temp1->sum(dimensions);
    NDArray<double>* dldbetaExp = epsilon->sum(dimensions);        
    NDArray<double>* dldxhat = new NDArray<double>(epsilon->getShapeInfo());                    
    epsilon->template applyBroadcast<simdOps::Multiply<double>>({1}, gamma, dldxhat, nullptr);                

    input->template applyBroadcast<simdOps::Subtract<double>>({1}, mean, temp1, nullptr);                
    dldxhat->template applyPairwiseTransform<simdOps::Multiply<double>>(temp1, temp1, nullptr);
    temp1->template applyScalar<simdOps::Multiply<double>>(-0.5);                        
    double powParams1[] = {-1.5};
    var->template applyTransform<simdOps::Pow<double>>(temp2, powParams1);
    temp1->template applyBroadcast<simdOps::Multiply<double>>({1}, temp2, temp1, nullptr);
    NDArray<double>* dldvar = temp1->sum(dimensions);

    double powParams2[] = {-0.5};
    var->template applyTransform<simdOps::Pow<double>>(temp2, powParams2);
    dldxhat->applyBroadcast<simdOps::Multiply<double>>({1}, temp2, temp1, nullptr);
    temp1->template applyTransform<simdOps::Neg<double>>();                
    NDArray<double>* dldmu = temp1->sum(dimensions);
    input->applyBroadcast<simdOps::Subtract<double>>({1}, mean, temp1, nullptr);
    temp1->template applyScalar<simdOps::Multiply<double>>(-2.);
    NDArray<double>* temp3 = temp1->sum(dimensions);
    temp3->template applyScalar<simdOps::Divide<double>>((double)effectiveBatchSize);
    dldvar->template applyPairwiseTransform<simdOps::Multiply<double>>(temp3, temp3, nullptr);
    dldmu->template applyPairwiseTransform<simdOps::Add<double>>(temp3, dldmu, nullptr);            
    
    NDArray<double>* dldinExp = new NDArray<double>(epsilon->getShapeInfo());
    dldxhat->applyBroadcast<simdOps::Multiply<double>>({1}, temp2, dldinExp, nullptr);
    input->applyBroadcast<simdOps::Subtract<double>>({1}, mean, temp1, nullptr);
    temp1->template applyScalar<simdOps::Multiply<double>>(2./effectiveBatchSize);
    temp1->applyBroadcast<simdOps::Multiply<double>>({1}, dldvar, temp1, nullptr);

    dldinExp->template applyPairwiseTransform<simdOps::Add<double>>(temp1, nullptr);
    dldmu->template applyScalar<simdOps::Multiply<double>>(1./effectiveBatchSize, temp2, nullptr);
    dldinExp->applyBroadcast<simdOps::Add<double>>({1}, temp2, dldinExp, nullptr);    

    nd4j::ops::batchnorm_bp<double> batchnorm_bp;
    Nd4jStatus status = batchnorm_bp.execute(block);
    ASSERT_EQ(ND4J_STATUS_OK, status);
    
    NDArray<double>* result = block->getVariableSpace()->getVariable(block->getNodeId())->getNDArray();
    // dldinExp->printBuffer();    
    // result->printBuffer();
    ASSERT_TRUE(dldinExp->equalsTo(result));
    
    delete temp1;
    delete temp2;
    delete temp3;    
    delete dldbetaExp;
    delete dldgammaExp;
    delete dldxhat;
    delete var;
    delete mean;
    delete dldvar;
    delete dldmu;
    delete dldinExp;
    delete xHat;
    delete std;
    delete block;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, sru1) {

    const int bS = 2;
    const int K = 3;    
    const int N = 4;
    double expStateBuff[] =  {0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715};
    double expOutputBuff[] = {1.090533, 1.174509, 1.252403, 1.324656, 1.090533, 1.174509, 1.252403, 1.324656, 1.090533, 1.174509, 1.252403, 1.324656, 1.090533, 1.174509, 1.252403, 1.324656, 1.090533, 1.174509, 1.252403, 1.324656, 1.090533, 1.174509, 1.252403, 1.324656};

    NDArray<double> input('c', {bS,K,N});
    NDArray<double> weights('c', {3*K,K});
    NDArray<double> bias('c', {1,2*K});
    NDArray<double> init('c', {bS,K});
    NDArray<double> mask('c', {bS,K});
    NDArray<double> expState('c', {bS,K,N});
    NDArray<double> expOut('c', {bS,K,N});
   
    input.assign(1.5);
    weights.assign(0.5); 
    bias.assign(0.3) ;
    init.assign(1.);
    mask.assign(1.);
    expState.setBuffer(expStateBuff);
    expOut.setBuffer(expOutputBuff);    

    nd4j::ops::sru<double> op;
    nd4j::ArrayList<double>*  results = op.execute({&input, &weights, &bias, &init, &mask}, {}, {});
    ASSERT_TRUE(results->size() == 2);    

    NDArray<double>* state  = results->at(0);
    NDArray<double>* output = results->at(1);
    // state->printBuffer();

    ASSERT_TRUE(expState.equalsTo(state));
    ASSERT_TRUE(expOut.equalsTo(output));
    
    delete results;
}

//////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, sru_logic1) {

    const int bS = 2;
    const int K = 3;    
    const int N = 4;
    double expStateBuff[] =  {0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715};
    double expOutputBuff[] = {1.090533, 1.174509, 1.252403, 1.324656, 1.090533, 1.174509, 1.252403, 1.324656, 1.090533, 1.174509, 1.252403, 1.324656, 1.090533, 1.174509, 1.252403, 1.324656, 1.090533, 1.174509, 1.252403, 1.324656, 1.090533, 1.174509, 1.252403, 1.324656};

    NDArray<double> input('c', {bS,K,N});
    NDArray<double> weights('c', {3*K,K});
    NDArray<double> bias('c', {1,2*K});
    NDArray<double> init('c', {bS,K});
    NDArray<double> mask('c', {bS,K});
    NDArray<double> expState('c', {bS,K,N});
    NDArray<double> expOut('c', {bS,K,N});
   
    input.assign(1.5);
    weights.assign(0.5); 
    bias.assign(0.3) ;
    init.assign(1.);
    mask.assign(1.);
    expState.setBuffer(expStateBuff);
    expOut.setBuffer(expOutputBuff);    

    nd4j::ops::sru_logic<double> op;
    nd4j::ArrayList<double>*  results = op.execute({&input, &weights, &bias, &init, &mask}, {}, {});
    ASSERT_TRUE(results->size() == 2);    

    NDArray<double>* state  = results->at(0);
    NDArray<double>* output = results->at(1);
    // state->printBuffer();

    ASSERT_TRUE(expState.equalsTo(state));
    ASSERT_TRUE(expOut.equalsTo(output));
    
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, sru_bp) {

    const int bS = 2;
    const int K = 3;    
    const int N = 4;
    double expGradXBuff[] = {-0.0259303, -0.03869125, -0.0302272, -0.02299165, -0.0259303, -0.03869125, -0.0302272, -0.02299165, -0.0259303, -0.03869125, -0.0302272, -0.02299165, -0.0259303, -0.03869125, -0.0302272, -0.02299165, -0.0259303, -0.03869125, -0.0302272, -0.02299165, -0.0259303, -0.03869125, -0.0302272, -0.02299165};    
    double expGradWBuff[] = {0.42526005,  0.42526005,  0.42526005, 0.42526005,  0.42526005,  0.42526005, 0.42526005,  0.42526005,  0.42526005, -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.15967215, -0.15967215, -0.15967215, -0.15967215, -0.15967215, -0.15967215, -0.15967215, -0.15967215, -0.15967215, 0.42526005,  0.42526005,  0.42526005, 0.42526005,  0.42526005,  0.42526005, 0.42526005,  0.42526005,  0.42526005, -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.15967215, -0.15967215, -0.15967215, -0.15967215, -0.15967215, -0.15967215, -0.15967215, -0.15967215, -0.15967215};
    double expGradBBuff[] = {-0.7043748, -0.7043748, -0.7043748, -0.2128962, -0.2128962, -0.2128962};
    double expGradInitBuff[] = {1.1421, 1.1421, 1.1421, 1.1421, 1.1421, 1.1421};
    double stateBuff[] = {0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715};                       

    NDArray<double> input('c', {bS,K,N});
    NDArray<double> weights('c', {3*K,K});
    NDArray<double> bias('c', {1,2*K});
    NDArray<double> init('c', {bS,K});
    NDArray<double> mask('c', {bS,K});
    NDArray<double> state('c', {bS,K,N});
    NDArray<double> inGradCt('c', {bS,K});
    NDArray<double> inGradH('c', {bS,K,N});

    NDArray<double> expGradX('c', {bS,K,N});
    expGradX.setBuffer(expGradXBuff);
    NDArray<double> expGradW('c', {bS,3*K,K});
    expGradW.setBuffer(expGradWBuff);
    NDArray<double> expGradB('c', {1,2*K});
    expGradB.setBuffer(expGradBBuff);
    NDArray<double> expGradInit('c', {bS,K});
    expGradInit.setBuffer(expGradInitBuff);

    input.assign(1.5);
    weights.assign(0.5); 
    bias.assign(0.3) ;    
    mask.assign(1.);
    init.assign(1.);
    state.setBuffer(stateBuff);
    inGradCt.assign(0.5);
    inGradH.assign(0.5);
    
    nd4j::ops::sru_bp<double> bp;
    nd4j::ArrayList<double>*  resultsBP = bp.execute({&input, &weights, &bias, &init, &state, &inGradCt, &inGradH, &mask}, {}, {});
    ASSERT_TRUE(resultsBP->size() == 4);    

    NDArray<double>* gradX    = resultsBP->at(0);
    NDArray<double>* gradW    = resultsBP->at(1);
    NDArray<double>* gradB    = resultsBP->at(2); 
    NDArray<double>* gradInit = resultsBP->at(3);

    ASSERT_TRUE(expGradX.equalsTo(gradX,1e-4)); 
    ASSERT_TRUE(expGradW.equalsTo(gradW));
    ASSERT_TRUE(expGradB.equalsTo(gradB));
    ASSERT_TRUE(expGradInit.equalsTo(gradInit));
    
    delete resultsBP;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, sru_bp_logic1) {

    const int bS = 2;
    const int K = 3;    
    const int N = 4;
    double expGradXBuff[] = {-0.0259303, -0.03869125, -0.0302272, -0.02299165, -0.0259303, -0.03869125, -0.0302272, -0.02299165, -0.0259303, -0.03869125, -0.0302272, -0.02299165, -0.0259303, -0.03869125, -0.0302272, -0.02299165, -0.0259303, -0.03869125, -0.0302272, -0.02299165, -0.0259303, -0.03869125, -0.0302272, -0.02299165};    
    double expGradWBuff[] = {0.42526005,  0.42526005,  0.42526005, 0.42526005,  0.42526005,  0.42526005, 0.42526005,  0.42526005,  0.42526005, -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.15967215, -0.15967215, -0.15967215, -0.15967215, -0.15967215, -0.15967215, -0.15967215, -0.15967215, -0.15967215, 0.42526005,  0.42526005,  0.42526005, 0.42526005,  0.42526005,  0.42526005, 0.42526005,  0.42526005,  0.42526005, -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.15967215, -0.15967215, -0.15967215, -0.15967215, -0.15967215, -0.15967215, -0.15967215, -0.15967215, -0.15967215};
    double expGradBBuff[] = {-0.7043748, -0.7043748, -0.7043748, -0.2128962, -0.2128962, -0.2128962};
    double expGradInitBuff[] = {1.1421, 1.1421, 1.1421, 1.1421, 1.1421, 1.1421};
    double stateBuff[] = {0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715};                       

    NDArray<double> input('c', {bS,K,N});
    NDArray<double> weights('c', {3*K,K});
    NDArray<double> bias('c', {1,2*K});
    NDArray<double> init('c', {bS,K});
    NDArray<double> mask('c', {bS,K});
    NDArray<double> state('c', {bS,K,N});
    NDArray<double> inGradCt('c', {bS,K});
    NDArray<double> inGradH('c', {bS,K,N});

    NDArray<double> expGradX('c', {bS,K,N});
    expGradX.setBuffer(expGradXBuff);
    NDArray<double> expGradW('c', {bS,3*K,K});
    expGradW.setBuffer(expGradWBuff);
    NDArray<double> expGradB('c', {1,2*K});
    expGradB.setBuffer(expGradBBuff);
    NDArray<double> expGradInit('c', {bS,K});
    expGradInit.setBuffer(expGradInitBuff);

    input.assign(1.5);
    weights.assign(0.5); 
    bias.assign(0.3) ;    
    mask.assign(1.);
    init.assign(1.);
    state.setBuffer(stateBuff);
    inGradCt.assign(0.5);
    inGradH.assign(0.5);
    
    nd4j::ops::sru_bp_logic<double> bp;
    nd4j::ArrayList<double>*  resultsBP = bp.execute({&input, &weights, &bias, &init, &state, &inGradCt, &inGradH, &mask}, {}, {});
    ASSERT_TRUE(resultsBP->size() == 4);    

    NDArray<double>* gradX    = resultsBP->at(0);
    NDArray<double>* gradW    = resultsBP->at(1);
    NDArray<double>* gradB    = resultsBP->at(2); 
    NDArray<double>* gradInit = resultsBP->at(3);

    ASSERT_TRUE(expGradX.equalsTo(gradX, 1e-4)); 
    ASSERT_TRUE(expGradW.equalsTo(gradW));
    ASSERT_TRUE(expGradB.equalsTo(gradB));
    ASSERT_TRUE(expGradInit.equalsTo(gradInit));
    
    delete resultsBP;
}

//////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, sru_bi_1) {

    const int bS = 2;
    const int K = 3;    
    const int N = 4;
    double expStateBuff[] =  {1.02857, 1.02857, 1.02857, 1.11288, 1.11288, 1.11288, 1.02857, 1.02857, 1.02857, 1.11288, 1.11288, 1.11288, 1.0569, 1.0569, 1.0569, 1.08501, 1.08501, 1.08501, 1.0569, 1.0569, 1.0569, 1.08501, 1.08501, 1.08501, 1.08501, 1.08501, 1.08501, 1.0569, 1.0569, 1.0569, 1.08501, 1.08501, 1.08501, 1.0569, 1.0569, 1.0569, 1.11288, 1.11288, 1.11288, 1.02857, 1.02857, 1.02857, 1.11288, 1.11288, 1.11288, 1.02857, 1.02857, 1.02857};
    double expOutputBuff[] = {0.779265, 0.779265, 0.779265, 0.810752, 0.810752, 0.810752, 0.779265, 0.779265, 0.779265, 0.810752, 0.810752, 0.810752, 0.790317, 0.790317, 0.790317, 0.800804, 0.800804, 0.800804, 0.790317, 0.790317, 0.790317, 0.800804, 0.800804, 0.800804, 0.800804, 0.800804, 0.800804, 0.790317, 0.790317, 0.790317, 0.800804, 0.800804, 0.800804, 0.790317, 0.790317, 0.790317, 0.810752, 0.810752, 0.810752, 0.779265, 0.779265, 0.779265, 0.810752, 0.810752, 0.810752, 0.779265, 0.779265, 0.779265};

    NDArray<double> input('c', {N,bS,2*K});
    NDArray<double> weights('c', {2*K,6*K});
    NDArray<double> bias('c', {1,4*K});
    NDArray<double> init('c', {bS,2*K});
    NDArray<double> mask('c', {bS,2*K});
    NDArray<double> expState('c', {N,bS,2*K});
    NDArray<double> expOut('c', {N,bS,2*K});
   
    input.assign(1.5);    
    weights.assign(0.5); 
    bias.assign(0.3) ;
    init.assign(1.);
    mask.assign(1.);
    expState.setBuffer(expStateBuff);
    expOut.setBuffer(expOutputBuff);    

    nd4j::ops::sru_bi<double> op;
    nd4j::ArrayList<double>*  results = op.execute({&input, &weights, &bias, &init, &mask}, {}, {});
    ASSERT_TRUE(results->size() == 2);    

    NDArray<double>* output = results->at(0);
    NDArray<double>* state = results->at(1);
    
    ASSERT_TRUE(expState.equalsTo(state));
    ASSERT_TRUE(expOut.equalsTo(output));
    
    delete results;
}

TEST_F(DeclarableOpsTests, sru_bi_bp_1) {

    const int bS = 2;
    const int K = 3;    
    const int N = 3;
    double expGradXBuff[] = {0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129};    
    double expGradInitBuff[] = {1.05121, 1.05121, 1.05121, 1.02676, 1.02676, 1.02676, 1.05121, 1.05121, 1.05121, 1.02676, 1.02676, 1.02676};
    double expGradWBuff[] = {0.02595354,-0.090096 ,-0.00882456,0.02595354,-0.090096 ,-0.0088245, 0.02595354,-0.090096 ,-0.00882456,0.01651665,-0.0559437,-0.0084390, 0.01651665,-0.0559437,-0.00843906,0.01651665,-0.0559437,-0.00843906, 0.02595354,-0.090096 ,-0.00882456,0.02595354,-0.090096 ,-0.0088245, 0.02595354,-0.090096 ,-0.00882456,0.01651665,-0.0559437,-0.0084390, 0.01651665,-0.0559437,-0.00843906,0.01651665,-0.0559437,-0.00843906, 0.02595354,-0.090096 ,-0.00882456,0.02595354,-0.090096 ,-0.0088245, 0.02595354,-0.090096 ,-0.00882456,0.01651665,-0.0559437,-0.0084390, 0.01651665,-0.0559437,-0.00843906,0.01651665,-0.0559437,-0.00843906, 0.02595354,-0.090096 ,-0.00882456,0.02595354,-0.090096 ,-0.0088245, 0.02595354,-0.090096 ,-0.00882456,0.01651665,-0.0559437,-0.0084390, 0.01651665,-0.0559437,-0.00843906,0.01651665,-0.0559437,-0.00843906, 0.02595354,-0.090096 ,-0.00882456,0.02595354,-0.090096 ,-0.0088245, 0.02595354,-0.090096 ,-0.00882456,0.01651665,-0.0559437,-0.0084390, 0.01651665,-0.0559437,-0.00843906,0.01651665,-0.0559437,-0.00843906, 0.02595354,-0.090096 ,-0.00882456,0.02595354,-0.090096 ,-0.0088245, 0.02595354,-0.090096 ,-0.00882456,0.01651665,-0.0559437,-0.0084390, 0.01651665,-0.0559437,-0.00843906,0.01651665,-0.0559437,-0.00843906, 0.02124567,-0.0731508,-0.00868926,0.02124567,-0.0731508,-0.0086892, 0.02124567,-0.0731508,-0.00868926,0.02084955,-0.0712011,-0.0085608, 0.02084955,-0.0712011,-0.00856086,0.02084955,-0.0712011,-0.00856086, 0.02124567,-0.0731508,-0.00868926,0.02124567,-0.0731508,-0.0086892, 0.02124567,-0.0731508,-0.00868926,0.02084955,-0.0712011,-0.0085608, 0.02084955,-0.0712011,-0.00856086,0.02084955,-0.0712011,-0.00856086, 0.02124567,-0.0731508,-0.00868926,0.02124567,-0.0731508,-0.0086892, 0.02124567,-0.0731508,-0.00868926,0.02084955,-0.0712011,-0.0085608, 0.02084955,-0.0712011,-0.00856086,0.02084955,-0.0712011,-0.00856086, 0.02124567,-0.0731508,-0.00868926,0.02124567,-0.0731508,-0.0086892, 0.02124567,-0.0731508,-0.00868926,0.02084955,-0.0712011,-0.0085608, 0.02084955,-0.0712011,-0.00856086,0.02084955,-0.0712011,-0.00856086, 0.02124567,-0.0731508,-0.00868926,0.02124567,-0.0731508,-0.0086892, 0.02124567,-0.0731508,-0.00868926,0.02084955,-0.0712011,-0.0085608, 0.02084955,-0.0712011,-0.00856086,0.02084955,-0.0712011,-0.00856086, 0.02124567,-0.0731508,-0.00868926,0.02124567,-0.0731508,-0.0086892, 0.02124567,-0.0731508,-0.00868926,0.02084955,-0.0712011,-0.0085608, 0.02084955,-0.0712011,-0.00856086,0.02084955,-0.0712011,-0.00856086, 0.01671156,-0.0570699,-0.00856086,0.01671156,-0.0570699,-0.0085608, 0.01671156,-0.0570699,-0.00856086,0.02534988,-0.0880002,-0.0086892, 0.02534988,-0.0880002,-0.00868926,0.02534988,-0.0880002,-0.00868926, 0.01671156,-0.0570699,-0.00856086,0.01671156,-0.0570699,-0.0085608, 0.01671156,-0.0570699,-0.00856086,0.02534988,-0.0880002,-0.0086892, 0.02534988,-0.0880002,-0.00868926,0.02534988,-0.0880002,-0.00868926, 0.01671156,-0.0570699,-0.00856086,0.01671156,-0.0570699,-0.0085608, 0.01671156,-0.0570699,-0.00856086,0.02534988,-0.0880002,-0.0086892, 0.02534988,-0.0880002,-0.00868926,0.02534988,-0.0880002,-0.00868926, 0.01671156,-0.0570699,-0.00856086,0.01671156,-0.0570699,-0.0085608, 0.01671156,-0.0570699,-0.00856086,0.02534988,-0.0880002,-0.0086892, 0.02534988,-0.0880002,-0.00868926,0.02534988,-0.0880002,-0.00868926, 0.01671156,-0.0570699,-0.00856086,0.01671156,-0.0570699,-0.0085608, 0.01671156,-0.0570699,-0.00856086,0.02534988,-0.0880002,-0.0086892, 0.02534988,-0.0880002,-0.00868926,0.02534988,-0.0880002,-0.00868926, 0.01671156,-0.0570699,-0.00856086,0.01671156,-0.0570699,-0.0085608, 0.01671156,-0.0570699,-0.00856086,0.02534988,-0.0880002,-0.0086892, 0.02534988,-0.0880002,-0.00868926,0.02534988,-0.0880002,-0.00868926};
    double expGradBBuff[] = {-0.0734389, -0.0734389, -0.0734389, -0.0717151, -0.0717151, -0.0717151, -0.0734389, -0.0734389, -0.0734389, -0.0717151, -0.0717151, -0.0717151, -0.00869156, -0.00869156, -0.00869156, -0.00856306, -0.00856306, -0.00856306, -0.00869156, -0.00869156, -0.00869156, -0.00856306, -0.00856306, -0.00856306};
    double stateBuff[] = {1.028569, 1.028569, 1.028569, 1.112884, 1.112884, 1.112884, 1.028569, 1.028569, 1.028569, 1.112884, 1.112884, 1.112884, 1.056905, 1.056905, 1.056905, 1.085009, 1.085009, 1.085009, 1.056905, 1.056905, 1.056905, 1.085009, 1.085009, 1.085009, 1.085009, 1.085009, 1.085009, 1.056905, 1.056905, 1.056905, 1.085009, 1.085009, 1.085009, 1.056905, 1.056905, 1.056905, 1.112884, 1.112884, 1.112884, 1.028569, 1.028569, 1.028569, 1.112884, 1.112884, 1.112884, 1.028569, 1.028569, 1.028569};
    
    NDArray<double> input('c', {N,bS,2*K});
    NDArray<double> weights('c', {2*K,6*K});
    NDArray<double> bias('c', {1,4*K});
    NDArray<double> init('c', {bS,2*K});
    NDArray<double> mask('c', {bS,2*K});
    NDArray<double> state('c', {N,bS,2*K});
    NDArray<double> inGradCt('c', {bS,2*K});
    NDArray<double> inGradH('c', {N,bS,2*K});
    
    NDArray<double> gradBias('c', {bS,4*K});
    gradBias.setBuffer(expGradBBuff);

    NDArray<double> expGradX('c', {N,bS,2*K});
    expGradX.setBuffer(expGradXBuff);
    NDArray<double> expGradW('c', {N,2*K,6*K});
    expGradW.setBuffer(expGradWBuff);
    NDArray<double> expGradB('c', {1,4*K});    
    gradBias.template reduceAlongDimension<simdOps::Sum<double>>(&expGradB, {0});    // [bS x 4K] -> [1 x 4K]    
    NDArray<double> expGradInit('c', {bS,2*K});
    expGradInit.setBuffer(expGradInitBuff);

    input.assign(1.5);
    weights.assign(0.5);
    bias.assign(0.3) ;    
    mask.assign(1.);
    init.assign(1.);
    state.setBuffer(stateBuff);
    inGradCt.assign(0.5);
    inGradH.assign(0.5);
    
    nd4j::ops::sru_bi_bp<double> bp;
    nd4j::ArrayList<double>*  resultsBP = bp.execute({&input, &weights, &bias, &init, &state, &inGradCt, &inGradH, &mask}, {}, {});
    ASSERT_TRUE(resultsBP->size() == 4);    

    NDArray<double>* gradX    = resultsBP->at(0);
    NDArray<double>* gradW    = resultsBP->at(1);
    NDArray<double>* gradB    = resultsBP->at(2); 
    NDArray<double>* gradInit = resultsBP->at(3);    

    ASSERT_TRUE(expGradX.equalsTo(gradX)); 
    ASSERT_TRUE(expGradW.equalsTo(gradW));
    ASSERT_TRUE(expGradB.equalsTo(gradB));
    ASSERT_TRUE(expGradInit.equalsTo(gradInit));
    
    delete resultsBP;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, Maxpool2d_bp2) {
    
    int bS=2, iD=1, iH=4,iW=4, oD=3, kH=2,kW=2, sH=1,sW=1, pH=0,pW=0, dH=1,dW=1;
    int oH = (iH - kH - (kH-1)*(dH-1) + 2*pH)/sH + 1;     
    int oW = (iW - kW - (kW-1)*(dW-1) + 2*pW)/sW + 1;    

    double epsilonBuff[]  = {6., 7., 8., 10., 11., 12., 14., 15., 16., 22., 23., 24., 26., 27., 28., 30., 31., 32.};
    double expectedBuff[] = {0., 0., 0., 0.,0., 6., 7., 8.,0.,10.,11.,12.,0.,14.,15.,16.,0., 0., 0., 0.,0.,22.,23.,24.,0.,26.,27.,28.,0.,30.,31.,32.};

    NDArray<double> input   ('c', {bS,iD,iH,iW});
    NDArray<double> epsilon ('c', {bS,iD,oH,oW});
    NDArray<double> expected('c', {bS,iD,iH,iW});


    nd4j::NDArrayFactory<double>::linspace(1., input);
    epsilon.setBuffer(epsilonBuff);
    expected.setBuffer(expectedBuff);
    
    std::initializer_list<int> argI = {kH,kW, sH,sW, pH,pW, dW,dH, 0};   // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 - same mode;

    nd4j::ops::maxpool2d_bp<double> op;
    nd4j::ArrayList<double>*  results = op.execute({&input, &epsilon}, {}, argI);
    NDArray<double>* output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, Avgpool2d_bp2) {
    
    int bS=2, iD=1, iH=4,iW=4, oD=3, kH=2,kW=2, sH=1,sW=1, pH=0,pW=0, dH=1,dW=1;
    int oH = (iH - kH - (kH-1)*(dH-1) + 2*pH)/sH + 1;     
    int oW = (iW - kW - (kW-1)*(dW-1) + 2*pW)/sW + 1;    

    double epsilonBuff[] = {3.5 , 4.5 , 5.5, 7.5 , 8.5 , 9.5, 11.5, 12.5, 13.5, 19.5, 20.5, 21.5, 23.5, 24.5, 25.5, 27.5, 28.5, 29.5};
    double expectedBuff[] = {0.875, 2., 2.5,  1.375, 2.75 , 6., 7.,  3.75, 4.75 ,10., 11., 5.75, 2.875, 6., 6.5, 3.375, 4.875, 10.,10.5, 5.375, 10.75, 22.,23., 11.75, 12.75, 26.,27., 13.75, 6.875, 14.,14.5, 7.375};

    NDArray<double> input   ('c', {bS,iD,iH,iW});
    NDArray<double> epsilon ('c', {bS,iD,oH,oW});
    NDArray<double> expected('c', {bS,iD,iH,iW});


    nd4j::NDArrayFactory<double>::linspace(1., input);
    epsilon.setBuffer(epsilonBuff);
    expected.setBuffer(expectedBuff);
    
    std::initializer_list<int> argI = {kH,kW, sH,sW, pH,pW, dW,dH, 0};   // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 - same mode;

    nd4j::ops::avgpool2d_bp<double> op;
    nd4j::ArrayList<double>*  results = op.execute({&input, &epsilon}, {}, argI);
    NDArray<double>* output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));
}

TEST_F(DeclarableOpsTests, ArgMax1) {
    NDArray<float> x('c', {3, 5});
    NDArrayFactory<float>::linspace(1, x);
    NDArray<float> exp('c', {3, 1});
    exp.assign(4.0f);

    nd4j::ops::argmax<float> op;

    auto result = op.execute({&x}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests, ArgMax2) {
    NDArray<float> x('c', {3, 5});
    NDArrayFactory<float>::linspace(1, x);
    NDArray<float> exp('c', {1, 5});
    exp.assign(2.0f);

    nd4j::ops::argmax<float> op;

    auto result = op.execute({&x}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests, ArgMin1) {
    NDArray<float> x('c', {3, 5});
    NDArrayFactory<float>::linspace(1, x);
    NDArray<float> exp('c', {3, 1});
    exp.assign(0.0f);

    nd4j::ops::argmin<float> op;

    auto result = op.execute({&x}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests, SquareTests1) {
    NDArray<float> x('c', {3, 5});
    NDArrayFactory<float>::linspace(1, x);

    NDArray<float> exp('c', {3, 5});
    NDArrayFactory<float>::linspace(1, exp);
    exp *= exp;

    nd4j::ops::square<float> op;

    auto result = op.execute({&x}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests, OneHotTests_1) {
    NDArray<float> indices('c', {1, 4});
    indices.putScalar(0, 0.0);
    indices.putScalar(1, 2.0);
    indices.putScalar(2, -1.0);
    indices.putScalar(3, 1.0);

    float _expB[] = {1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.};
    NDArray<float> exp('c', {4, 3});
    exp.setBuffer(_expB);

    nd4j::ops::onehot<float> op;

    auto result = op.execute({&indices}, {1.0f, 0.0f}, {3, -1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests, OneHotTests_2) {
    NDArray<float> indices('c', {2, 2});
    indices.putScalar(0, 0.0);
    indices.putScalar(1, 2.0);
    indices.putScalar(2, 1.0);
    indices.putScalar(3, -1.0);

    float _expB[] = {1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0.};
    NDArray<float> exp('c', {2, 2, 3});
    exp.setBuffer(_expB);

    nd4j::ops::onehot<float> op;
    auto result = op.execute({&indices}, {1.0f, 0.0f}, {3, -1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests, FillAs_1) {
    NDArray<float> x('c', {2, 2});
    x.assign(117);

    float scalar = 119.f;

    nd4j::ops::fill_as<float> op;
    auto result = op.execute({&x}, {scalar}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(x.isSameShape(result->at(0)));

    ASSERT_NEAR(scalar, result->at(0)->meanNumber(), 1e-5f);

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, LRN1) {
    nd4j::ops::lrn<double> lrn;

    lrn.getOpName();
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, Stack_1) {

    float buff1[]   = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12};
    float buff2[]   = {13,14,16,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,16,16,17,18,19,20,21,22,23,24};
    int shape1[]    = {2, 3, 4, 4, 1, 0, 1, 99};
    int shape2[]    = {2, 3, 4, 4, 1, 0, 1, 99};
    int expShape[]  = {3, 2, 3, 4, 12, 4, 1, 0, 1, 99};

    NDArray<float> input1(buff1, shape1);
    NDArray<float> input2(buff2, shape2);
    NDArray<float> expected(expBuff, expShape);

    nd4j::ops::stack<float> op;
    nd4j::ArrayList<float>*  results = op.execute({&input1, &input2}, {}, {0});
    NDArray<float>* output = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, Stack_2) {

    float buff1[]   = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12};
    float buff2[]   = {13,14,16,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {1,  2,  3,  4, 13, 14, 16, 16, 5,  6,  7,  8, 17, 18, 19, 20, 9, 10, 11, 12, 21, 22, 23, 24};
    int shape1[]    = {2, 3, 4, 4, 1, 0, 1, 99};
    int shape2[]    = {2, 3, 4, 4, 1, 0, 1, 99};
    int expShape[]  = {3, 3, 2, 4, 8, 4, 1, 0, 1, 99};

    NDArray<float> input1(buff1, shape1);
    NDArray<float> input2(buff2, shape2);
    NDArray<float> expected(expBuff, expShape);

    nd4j::ops::stack<float> op;
    nd4j::ArrayList<float>*  results = op.execute({&input1, &input2}, {}, {1});
    NDArray<float>* output = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, Stack_3) {

    float buff1[]   = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12};
    float buff2[]   = {13,14,16,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,16,16,17,18,19,20,21,22,23,24};
    int shape1[]    = {2, 1, 12, 12, 1, 0, 1, 99};
    int shape2[]    = {2, 1, 12, 12, 1, 0, 1, 99};
    int expShape[]  = {2, 2, 12, 12, 1, 0, 1, 99};

    NDArray<float> input1(buff1, shape1);
    NDArray<float> input2(buff2, shape2);
    NDArray<float> expected(expBuff, expShape);

    nd4j::ops::stack<float> op;
    nd4j::ArrayList<float>*  results = op.execute({&input1, &input2}, {}, {0});
    NDArray<float>* output = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, Stack_4) {

    float buff1[]   = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12};
    float buff2[]   = {13,14,16,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,16,16,17,18,19,20,21,22,23,24};
    int shape1[]    = {2, 1, 12, 12, 1, 0, 1, 99};
    int shape2[]    = {2, 1, 12, 12, 1, 0, 1, 99};
    int expShape[]  = {2, 2, 12, 12, 1, 0, 1, 99};

    NDArray<float> input1(buff1, shape1);
    NDArray<float> input2(buff2, shape2);
    NDArray<float> expected(expBuff, expShape);

    nd4j::ops::stack<float> op;
    nd4j::ArrayList<float>*  results = op.execute({&input1, &input2}, {}, {1});
    NDArray<float>* output = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, Stack_5) {

    float buff1[]   = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12};
    float buff2[]   = {13,14,16,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,16,16,17,18,19,20,21,22,23,24};
    int shape1[]    = {2, 12, 1, 1,  1, 0, 1, 99};
    int shape2[]    = {2, 12, 1, 1,  1, 0, 1, 99};
    int expShape[]  = {2, 2, 12, 12, 1, 0, 1, 99};

    NDArray<float> input1(buff1, shape1);
    NDArray<float> input2(buff2, shape2);
    NDArray<float> expected(expBuff, expShape);

    nd4j::ops::stack<float> op;
    nd4j::ArrayList<float>*  results = op.execute({&input1, &input2}, {}, {0});
    NDArray<float>* output = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, Stack_6) {

    float buff1[]   = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12};
    float buff2[]   = {13,14,16,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {1 ,13 ,2 ,14 ,3 ,16 ,4 ,16 ,5 ,17 ,6 ,18 ,7 ,19 ,8 ,20 ,9 ,21 ,10 ,22 ,11 ,23 ,12 ,24};
    int shape1[]    = {2, 12, 1, 1, 12, 0, 1, 99};
    int shape2[]    = {2, 12, 1, 1, 12, 0, 1, 99};
    int expShape[]  = {2, 12, 2, 2,  1, 0, 1, 99};

    NDArray<float> input1(buff1, shape1);
    NDArray<float> input2(buff2, shape2);
    NDArray<float> expected(expBuff, expShape);

    nd4j::ops::stack<float> op;
    nd4j::ArrayList<float>*  results = op.execute({&input1, &input2}, {}, {1});
    NDArray<float>* output = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, Reverse_1 ) {

    float inBuff[]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {24., 23., 22., 21., 20., 19., 18., 17., 16., 15., 14., 13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1.};
    int shapeInfo[] = {3, 2, 3, 4, 12, 4, 1, 0, 1, 99};   
    // int shapeInfo[] = {2, 2, 12, 12, 1, 0, 1, 99};   

    NDArray<float> input(inBuff, shapeInfo);
    NDArray<float> expected(expBuff, shapeInfo);
    NDArray<float> output(shapeInfo);   

    nd4j::ops::reverse<float> op;
    auto results = op.execute({&input}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);    

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, Reverse_2 ) {

    float inBuff[]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {24., 23., 22., 21., 20., 19., 18., 17., 16., 15., 14., 13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1.};
    int shapeInfo[] = {3, 2, 3, 4, 12, 4, 1, 0, 1, 99};   
    // int shapeInfo[] = {2, 2, 12, 12, 1, 0, 1, 99};   

    NDArray<float> input(inBuff, shapeInfo);
    NDArray<float> expected(expBuff, shapeInfo);
    NDArray<float> output(shapeInfo);   

    nd4j::ops::reverse<float> op;
    auto results = op.execute({&input}, {}, {}, true);

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);    

    ASSERT_TRUE(expected.isSameShapeStrict(&input));
    ASSERT_TRUE(expected.equalsTo(&input));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, Reverse_3 ) {

    float inBuff[]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1., 24., 23., 22., 21., 20., 19., 18., 17., 16., 15., 14., 13.};
    int shapeInfo[] = {3, 2, 3, 4, 12, 4, 1, 0, 1, 99};    

    NDArray<float> input(inBuff, shapeInfo);
    NDArray<float> expected(expBuff, shapeInfo);
    NDArray<float> output(shapeInfo);   

    nd4j::ops::reverse<float> op;
    auto results = op.execute({&input}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float>* result = results->at(0);    
    // result->printBuffer();    

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, Reverse_4 ) {

    float inBuff[]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {16,15,14,13,    20,19,18,17,       24,23,22,21,    4,3,2,1,    8,7,6,5,      12,11,10,9,};
    int shapeInfo[] = {3, 2, 3, 4, 12, 4, 1, 0, 1, 99};    

    NDArray<float> input(inBuff, shapeInfo);
    NDArray<float> expected(expBuff, shapeInfo);
    NDArray<float> output(shapeInfo);   

    nd4j::ops::reverse<float> op;
    auto results = op.execute({&input}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float>* result = results->at(0);    
    // result->printBuffer();    

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, Reverse_5 ) {

    float inBuff[]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {21., 22., 23., 24., 17., 18., 19., 20., 13., 14., 15., 16., 9., 10., 11., 12., 5., 6., 7., 8., 1., 2., 3., 4.};
    int shapeInfo[] = {3, 2, 3, 4, 12, 4, 1, 0, 1, 99};    

    NDArray<float> input(inBuff, shapeInfo);
    NDArray<float> expected(expBuff, shapeInfo);
    NDArray<float> output(shapeInfo);   

    nd4j::ops::reverse<float> op;
    auto results = op.execute({&input}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float>* result = results->at(0);        

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, Reverse_6 ) {

    float inBuff[]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {4., 3., 2., 1., 8., 7., 6., 5., 12., 11., 10., 9., 16., 15., 14., 13., 20., 19., 18., 17., 24., 23., 22., 21.};
    int shapeInfo[] = {3, 2, 3, 4, 12, 4, 1, 0, 1, 99};    

    NDArray<float> input(inBuff, shapeInfo);
    NDArray<float> expected(expBuff, shapeInfo);
    NDArray<float> output(shapeInfo);   

    nd4j::ops::reverse<float> op;
    auto results = op.execute({&input}, {}, {0,1}, true);

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float>* result = results->at(0);        
    // result->printBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(&input));
    ASSERT_TRUE(expected.equalsTo(&input));

    delete results;
}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, Reverse_7 ) {

    float inBuff[]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {9., 10., 11., 12., 5., 6., 7., 8., 1., 2., 3., 4., 21., 22., 23., 24., 17., 18., 19., 20., 13., 14., 15., 16.};
    int shapeInfo[] = {3, 2, 3, 4, 12, 4, 1, 0, 1, 99};    

    NDArray<float> input(inBuff, shapeInfo);
    NDArray<float> expected(expBuff, shapeInfo);
    NDArray<float> output(shapeInfo);   

    nd4j::ops::reverse<float> op;
    auto results = op.execute({&input}, {}, {0,2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float>* result = results->at(0);        
    // result->printBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, Reverse_8 ) {

    float inBuff[]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {9., 10., 11., 12., 5., 6., 7., 8., 1., 2., 3., 4., 21., 22., 23., 24., 17., 18., 19., 20., 13., 14., 15., 16.};
    int shapeInfo[] = {3, 2, 3, 4, 12, 4, 1, 0, 1, 99};    

    NDArray<float> input(inBuff, shapeInfo);
    NDArray<float> expected(expBuff, shapeInfo);
    NDArray<float> output(shapeInfo);   

    nd4j::ops::reverse<float> op;
    auto results = op.execute({&input}, {}, {2,0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float>* result = results->at(0);        
    // result->printBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests, Reverse_9 ) {

    float inBuff[]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.};
    int shapeInfo[] = {3, 2, 3, 4, 12, 4, 1, 0, 1, 99};    

    NDArray<float> input(inBuff, shapeInfo);
    NDArray<float> expected(expBuff, shapeInfo);
    NDArray<float> output(shapeInfo);   

    nd4j::ops::reverse<float> op;
    auto results = op.execute({&input}, {}, {1,2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float>* result = results->at(0);    

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}


////////////////////////////////////////////////////////////////////
// CONSTANT mode 2D
TEST_F(DeclarableOpsTests, Pad_1) {

    float inBuff[]  = {1,2,3,4,5,6};
    float padBuff[] = {1,1,2,2};
    float expBuff[] = {0,0,0,0,0,0,0, 0,0,1,2,3,0,0, 0,0,4,5,6,0,0, 0,0,0,0,0,0,0};    

    NDArray<float> input   (inBuff,  'c', {2,3});
    NDArray<float> paddings(padBuff, 'c', {2,2});
    NDArray<float> expected(expBuff, 'c', {4,7});

    nd4j::ops::pad<float> op;
    auto results = op.execute({&input, &paddings}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float>* result = results->at(0);    
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}


////////////////////////////////////////////////////////////////////
// REFLECT mode 2D
TEST_F(DeclarableOpsTests, Pad_2) {

    float inBuff[]  = {1,2,3,4,5,6};
    float padBuff[] = {1,1,2,2};
    float expBuff[] = {6,5,4,5,6,5,4, 3,2,1,2,3,2,1, 6,5,4,5,6,5,4, 3,2,1,2,3,2,1};    

    NDArray<float> input   (inBuff,  'c', {2,3});
    NDArray<float> paddings(padBuff, 'c', {2,2});
    NDArray<float> expected(expBuff, 'c', {4,7});

    nd4j::ops::pad<float> op;
    auto results = op.execute({&input, &paddings}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float>* result = results->at(0);    
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}


////////////////////////////////////////////////////////////////////
// SYMMETRIC mode 2D
TEST_F(DeclarableOpsTests, Pad_3) {

    float inBuff[]  = {1,2,3,4,5,6};
    float padBuff[] = {1,1,2,2};
    float expBuff[] = {2,1,1,2,3,3,2, 2,1,1,2,3,3,2, 5,4,4,5,6,6,5, 5,4,4,5,6,6,5};    

    NDArray<float> input   (inBuff,  'c', {2,3});
    NDArray<float> paddings(padBuff, 'c', {2,2});
    NDArray<float> expected(expBuff, 'c', {4,7});

    nd4j::ops::pad<float> op;
    auto results = op.execute({&input, &paddings}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float>* result = results->at(0);    
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}


////////////////////////////////////////////////////////////////////
// CONSTANT mode 3D
TEST_F(DeclarableOpsTests, Pad_4) {

    float inBuff[]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};
    float padBuff[] = {1,1,2,2,2,2};
    float expBuff[] = {0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 1, 2, 3,0,0,0,0, 4, 5, 6,0,0,0,0, 7, 8, 9,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0,10,11,12,0,0,0,0,13,14,15,0,0,0,0,16,17,18,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0};

    NDArray<float> input   (inBuff,  'c', {2,3,3});
    NDArray<float> paddings(padBuff, 'c', {3,2});
    NDArray<float> expected(expBuff, 'c', {4,7,7});

    nd4j::ops::pad<float> op;
    auto results = op.execute({&input, &paddings}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float>* result = results->at(0);    
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}



////////////////////////////////////////////////////////////////////
// REFLECT mode 3D
TEST_F(DeclarableOpsTests, Pad_5) {

    float inBuff[]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};
    float padBuff[] = {1,1,2,2,2,2};
    float expBuff[] = {18,17,16,17,18,17,16, 15,14,13,14,15,14,13, 12,11,10,11,12,11,10, 15,14,13,14,15,14,13, 18,17,16,17,18,17,16, 15,14,13,14,15,14,13, 12,11,10,11,12,11,10, 9, 8, 7, 8, 9, 8, 7, 6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1, 6, 5, 4, 5, 6, 5, 4, 9, 8, 7, 8, 9, 8, 7, 6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1, 18,17,16,17,18,17,16, 15,14,13,14,15,14,13, 12,11,10,11,12,11,10, 15,14,13,14,15,14,13, 18,17,16,17,18,17,16, 15,14,13,14,15,14,13, 12,11,10,11,12,11,10, 9, 8, 7, 8, 9, 8, 7, 6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1, 6, 5, 4, 5, 6, 5, 4, 9, 8, 7, 8, 9, 8, 7, 6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1};                      
    NDArray<float> input   (inBuff,  'c', {2,3,3});
    NDArray<float> paddings(padBuff, 'c', {3,2});
    NDArray<float> expected(expBuff, 'c', {4,7,7});

    nd4j::ops::pad<float> op;
    auto results = op.execute({&input, &paddings}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float>* result = results->at(0);    
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}


////////////////////////////////////////////////////////////////////
// SYMMETRIC mode 3D
TEST_F(DeclarableOpsTests, Pad_6) {

    float inBuff[]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};
    float padBuff[] = {1,1,2,2,2,2};
    float expBuff[] = {5, 4, 4, 5, 6, 6, 5, 2, 1, 1, 2, 3, 3, 2, 2, 1, 1, 2, 3, 3, 2, 5, 4, 4, 5, 6, 6, 5, 8, 7, 7, 8, 9, 9, 8, 8, 7, 7, 8, 9, 9, 8, 5, 4, 4, 5, 6, 6, 5, 5, 4, 4, 5, 6, 6, 5, 2, 1, 1, 2, 3, 3, 2, 2, 1, 1, 2, 3, 3, 2, 5, 4, 4, 5, 6, 6, 5, 8, 7, 7, 8, 9, 9, 8, 8, 7, 7, 8, 9, 9, 8, 5, 4, 4, 5, 6, 6, 5, 14,13,13,14,15,15,14, 11,10,10,11,12,12,11, 11,10,10,11,12,12,11, 14,13,13,14,15,15,14, 17,16,16,17,18,18,17, 17,16,16,17,18,18,17, 14,13,13,14,15,15,14, 14,13,13,14,15,15,14, 11,10,10,11,12,12,11, 11,10,10,11,12,12,11, 14,13,13,14,15,15,14, 17,16,16,17,18,18,17, 17,16,16,17,18,18,17, 14,13,13,14,15,15,14};

    NDArray<float> input   (inBuff,  'c', {2,3,3});
    NDArray<float> paddings(padBuff, 'c', {3,2});
    NDArray<float> expected(expBuff, 'c', {4,7,7});

    nd4j::ops::pad<float> op;
    auto results = op.execute({&input, &paddings}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float>* result = results->at(0);    
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
// CONSTANT mode 4D
TEST_F(DeclarableOpsTests, Pad_7)
{

    float inBuff[] =  {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float padBuff[] = {1, 1, 1, 1, 1, 1, 1, 1};
    float expBuff[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6, 0, 0, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 10, 0, 0, 11, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 14, 0, 0, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    NDArray<float> input(inBuff, 'c', {2, 2, 2, 2});
    NDArray<float> paddings(padBuff, 'c', {4, 2});
    NDArray<float> expected(expBuff, 'c', {4, 4, 4, 4});

    nd4j::ops::pad<float> op;
    auto results = op.execute({&input, &paddings}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
// REFLECT mode 4D
TEST_F(DeclarableOpsTests, Pad_8)
{

    float inBuff[] =  {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float padBuff[] = {1, 1, 1, 1, 1, 1, 1, 1};
    float expBuff[] = {16, 15, 16, 15, 14, 13, 14, 13, 16, 15, 16, 15, 14, 13, 14, 13, 12, 11, 12, 11, 10, 9, 10, 9, 12, 11, 12, 11, 10, 9, 10, 9, 16, 15, 16, 15, 14, 13, 14, 13, 16, 15, 16, 15, 14, 13, 14, 13, 12, 11, 12, 11, 10, 9, 10, 9, 12, 11, 12, 11, 10, 9, 10, 9, 8, 7, 8, 7, 6, 5, 6, 5, 8, 7, 8, 7, 6, 5, 6, 5, 4, 3, 4, 3, 2, 1, 2, 1, 4, 3, 4, 3, 2, 1, 2, 1, 8, 7, 8, 7, 6, 5, 6, 5, 8, 7, 8, 7, 6, 5, 6, 5, 4, 3, 4, 3, 2, 1, 2, 1, 4, 3, 4, 3, 2, 1, 2, 1, 16, 15, 16, 15, 14, 13, 14, 13, 16, 15, 16, 15, 14, 13, 14, 13, 12, 11, 12, 11, 10, 9, 10, 9, 12, 11, 12, 11, 10, 9, 10, 9, 16, 15, 16, 15, 14, 13, 14, 13, 16, 15, 16, 15, 14, 13, 14, 13, 12, 11, 12, 11, 10, 9, 10, 9, 12, 11, 12, 11, 10, 9, 10, 9, 8, 7, 8, 7, 6, 5, 6, 5, 8, 7, 8, 7, 6, 5, 6, 5, 4, 3, 4, 3, 2, 1, 2, 1, 4, 3, 4, 3, 2, 1, 2, 1, 8, 7, 8, 7, 6, 5, 6, 5, 8, 7, 8, 7, 6, 5, 6, 5, 4, 3, 4, 3, 2, 1, 2, 1, 4, 3, 4, 3, 2, 1, 2, 1};    
    NDArray<float> input(inBuff, 'c', {2, 2, 2, 2});
    NDArray<float> paddings(padBuff, 'c', {4, 2});
    NDArray<float> expected(expBuff, 'c', {4, 4, 4, 4});

    nd4j::ops::pad<float> op;
    auto results = op.execute({&input, &paddings}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

//////////////////////////////////////////////////////////////////
// SYMMETRIC mode 4D 
TEST_F(DeclarableOpsTests, Pad_9)
{

    float inBuff[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float padBuff[] = {1, 1, 1, 1, 1, 1, 1, 1};
    float expBuff[] = {1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4, 1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4, 5, 5, 6, 6, 5, 5, 6, 6, 7, 7, 8, 8, 7, 7, 8, 8, 5, 5, 6, 6, 5, 5, 6, 6, 7, 7, 8, 8, 7, 7, 8, 8, 1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4, 1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4, 5, 5, 6, 6, 5, 5, 6, 6, 7, 7, 8, 8, 7, 7, 8, 8, 5, 5, 6, 6, 5, 5, 6, 6, 7, 7, 8, 8, 7, 7, 8, 8, 9, 9, 10, 10, 9, 9, 10, 10, 11, 11, 12, 12, 11, 11, 12, 12, 9, 9, 10, 10, 9, 9, 10, 10, 11, 11, 12, 12, 11, 11, 12, 12, 13, 13, 14, 14, 13, 13, 14, 14, 15, 15, 16, 16, 15, 15, 16, 16, 13, 13, 14, 14, 13, 13, 14, 14, 15, 15, 16, 16, 15, 15, 16, 16, 9, 9, 10, 10, 9, 9, 10, 10, 11, 11, 12, 12, 11, 11, 12, 12, 9, 9, 10, 10, 9, 9, 10, 10, 11, 11, 12, 12, 11, 11, 12, 12, 13, 13, 14, 14, 13, 13, 14, 14, 15, 15, 16, 16, 15, 15, 16, 16, 13, 13, 14, 14, 13, 13, 14, 14, 15, 15, 16, 16, 15, 15, 16, 16};
    NDArray<float> input(inBuff, 'c', {2, 2, 2, 2});
    NDArray<float> paddings(padBuff, 'c', {4, 2});
    NDArray<float> expected(expBuff, 'c', {4, 4, 4, 4});

    nd4j::ops::pad<float> op;
    auto results = op.execute({&input, &paddings}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}


