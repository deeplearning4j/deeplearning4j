//
// @author raver119@gmail.com
//

#include "testlayers.h"
#include <Block.h>
#include <Variable.h>
#include <VariableSpace.h>
#include <ops/declarable/declarable_ops.h>

using namespace nd4j::graph;

class DeclarableOpsTests : public testing::Test {
public:
    int *cShape = new int[8]{2, 2, 2, 2, 1, 0, 1, 99};
    int *fShape = new int[8]{2, 2, 2, 1, 2, 0, 1, 102};
};

TEST_F(DeclarableOpsTests, BasicInitialization1) {
    auto concat = new nd4j::ops::Concat<float>();
    std::string expName("Concat");
    ASSERT_EQ(expName, *(concat->getOpName()));

    NDArray<float> x0(1, 5, 'c');
    NDArray<float> x1(1, 5, 'c');
    NDArray<float> x2(1, 5, 'c');
    NDArray<float> x3(1, 5, 'c');
    NDArray<float> x4(1, 5, 'c');

    auto variableSpace = new VariableSpace<float>();

    variableSpace->putVariable(-1, &x0);
    variableSpace->putVariable(-2, &x1);
    variableSpace->putVariable(-3, &x2);
    variableSpace->putVariable(-4, &x3);
    variableSpace->putVariable(-5, &x4);

    Block<float> block(variableSpace);

    block.fillInputs({-1, -2, -3, -4, -5});

    Nd4jStatus result = concat->validateAndExecute(&block);


    ASSERT_EQ(ND4J_STATUS_OK, result);
}