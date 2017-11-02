//
// Created by raver119 on 01.11.2017.
//

#include "testlayers.h"
#include <helpers/ShapeUtils.h>


using namespace nd4j;
using namespace nd4j::graph;

class ShapeUtilsTests : public testing::Test {
public:

};

TEST_F(ShapeUtilsTests, BasicInject1) {
    std::vector<int> shape({1, 4});

    ShapeUtils<float>::insertDimension(2, shape.data(), -1, 3);
    ASSERT_EQ(4, shape.at(0));
    ASSERT_EQ(3, shape.at(1));
}


TEST_F(ShapeUtilsTests, BasicInject2) {
    std::vector<int> shape({1, 4});

    ShapeUtils<float>::insertDimension(2, shape.data(), 0, 3);
    ASSERT_EQ(3, shape.at(0));
    ASSERT_EQ(4, shape.at(1));
}