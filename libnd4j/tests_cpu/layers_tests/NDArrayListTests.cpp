//
// @author raver119@gmail.com
//

#include <NDArray.h>
#include <NDArrayList.h>
#include "testlayers.h"

using namespace nd4j;

class NDArrayListTests : public testing::Test {
public:

};


TEST_F(NDArrayListTests, BasicTests_1) {
    NDArrayList<float> list(false);

    NDArray<float> x('c', {1, 10});
    NDArray<float> y('c', {1, 10});

    ASSERT_EQ(ND4J_STATUS_OK, list.write(1, x.dup()));

    //ASSERT_EQ(ND4J_STATUS_DOUBLE_WRITE, list.write(1, &y));
}

TEST_F(NDArrayListTests, BasicTests_2) {
    NDArrayList<float> list(false);

    NDArray<float> x('c', {1, 10});
    NDArray<float> y('c', {1, 7});

    ASSERT_EQ(ND4J_STATUS_OK, list.write(1, x.dup()));

    ASSERT_EQ(ND4J_STATUS_BAD_DIMENSIONS, list.write(0, &y));
}


TEST_F(NDArrayListTests, Test_Stack_UnStack_1) {
    NDArray<float> input('c', {10, 10});
    input.linspace(1);

    NDArrayList<float> list(false);

    list.unstack(&input, 0);

    ASSERT_EQ(10, list.elements());

    auto array = list.stack();

    ASSERT_TRUE(input.isSameShape(array));

    ASSERT_TRUE(input.equalsTo(array));

    delete array;
}