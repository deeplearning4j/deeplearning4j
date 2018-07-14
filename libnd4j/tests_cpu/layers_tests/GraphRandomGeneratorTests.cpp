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

TEST_F(GraphRandomGeneratorTests, Sequential_Test_1) {
    nd4j::graph::RandomGenerator g0(119, 5);
    nd4j::graph::RandomGenerator g1(119, 5);

    auto v0 = g0.relativeT<int>(15, 0, DataTypeUtils::max<int>());
    auto v1 = g1.relativeT<int>(15, 0, DataTypeUtils::max<int>());
    g0.rewindH(200);
    auto r0 = g0.relativeT<int>(15, 0, DataTypeUtils::max<int>());
    auto r1 = g1.relativeT<int>(15, 0, DataTypeUtils::max<int>());

    // values after rewind aren't equal
    ASSERT_NE(r0, v0);

    // two generators must give the same output
    ASSERT_EQ(v0, v1);

    // but not after one of them was rewinded
    ASSERT_NE(r1, r0);
}

TEST_F(GraphRandomGeneratorTests, Sequential_Test_2) {
    nd4j::graph::RandomGenerator g0(119, 5);
    nd4j::graph::RandomGenerator g1(119, 5);

    auto v0 = g0.relativeT<int>(15, 0, DataTypeUtils::max<int>());
    auto v1 = g1.relativeT<int>(15, 0, DataTypeUtils::max<int>());
    g0.rewindH(200);
    g1.rewindH(199);
    auto r0 = g0.relativeT<int>(15, 0, DataTypeUtils::max<int>());
    auto r1 = g1.relativeT<int>(15, 0, DataTypeUtils::max<int>());

    // values after rewind aren't equal
    ASSERT_NE(r0, v0);

    // two generators must give the same output
    ASSERT_EQ(v0, v1);

    // but not after they was rewinded with different number of elements
    ASSERT_NE(r1, r0);
}

TEST_F(GraphRandomGeneratorTests, Sequential_Test_3) {
    nd4j::graph::RandomGenerator g0(119, 5);
    nd4j::graph::RandomGenerator g1(119, 5);

    auto v0 = g0.relativeT<int>(15, 0, DataTypeUtils::max<int>());
    auto v1 = g1.relativeT<int>(15, 0, DataTypeUtils::max<int>());
    g0.rewindH(200);
    g1.rewindH(200);
    auto r0 = g0.relativeT<int>(15, 0, DataTypeUtils::max<int>());
    auto r1 = g1.relativeT<int>(15, 0, DataTypeUtils::max<int>());

    // values after rewind aren't equal
    ASSERT_NE(r0, v0);

    // two generators must give the same output
    ASSERT_EQ(v0, v1);

    // and here output must be equal as well
    ASSERT_EQ(r1, r0);
}