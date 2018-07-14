#include "testlayers.h"
#include <graph/RandomGenerator.h>
#include <array/DataTypeUtils.h>

using namespace nd4j;
using namespace nd4j::graph;

class GraphRandomGeneratorTests : public testing::Test {
public:

};

TEST_F(GraphRandomGeneratorTests, Reproducibility_Test_1) {
    nd4j::graph::RandomGenerator g0(119);
    nd4j::graph::RandomGenerator g1(119);

    auto i0 = g0.relativeT<int>(15, 0, DataTypeUtils::max<int>());
    auto i1 = g1.relativeT<int>(15, 0, DataTypeUtils::max<int>());

    ASSERT_EQ(i0, i1);
}

TEST_F(GraphRandomGeneratorTests, Reproducibility_Test_2) {
    nd4j::graph::RandomGenerator g0(119);
    nd4j::graph::RandomGenerator g1(117);

    auto i0 = g0.relativeT<int>(15, 0, DataTypeUtils::max<int>());
    auto i1 = g1.relativeT<int>(15, 0, DataTypeUtils::max<int>());

    ASSERT_NE(i0, i1);
}

TEST_F(GraphRandomGeneratorTests, Reproducibility_Test_3) {
    nd4j::graph::RandomGenerator g0(119, 5);
    nd4j::graph::RandomGenerator g1(119, 10);

    auto i0 = g0.relativeT<int>(15, 0, DataTypeUtils::max<int>());
    auto i1 = g1.relativeT<int>(15, 0, DataTypeUtils::max<int>());

    ASSERT_NE(i0, i1);
}

TEST_F(GraphRandomGeneratorTests, Reproducibility_Test_4) {
    nd4j::graph::RandomGenerator g0(119, 5);
    nd4j::graph::RandomGenerator g1(117, 5);

    auto i0 = g0.relativeT<int>(15, 0, DataTypeUtils::max<int>());
    auto i1 = g1.relativeT<int>(15, 0, DataTypeUtils::max<int>());

    ASSERT_NE(i0, i1);
}