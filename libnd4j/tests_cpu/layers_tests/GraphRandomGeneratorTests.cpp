#include "testlayers.h"
#include <graph/RandomGenerator.h>
#include <array/DataTypeUtils.h>
#include <graph/Graph.h>

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


TEST_F(GraphRandomGeneratorTests, FloatingPoint_Test_1) {
    nd4j::graph::RandomGenerator g0(119, 5);
    nd4j::graph::RandomGenerator g1(119, 5);

    std::array<double, 100> z0, z1, z2, z3;

    for (int e = 0; e < z0.size(); e++) {
        z0[e] = g0.relativeT<double>(e, -1.0, 1.0);
        z1[e] = g1.relativeT<double>(e, -1.0, 1.0);
    }

    g0.rewindH(z0.size());
    g1.rewindH(z0.size());

    for (int e = 0; e < z0.size(); e++) {
        z2[e] = g0.relativeT<double>(e, -1.0, 1.0);
        z3[e] = g1.relativeT<double>(e, -1.0, 1.0);
    }

    // these sequences should be equal
    ASSERT_EQ(z0, z1);
    ASSERT_EQ(z2, z3);

    // these sequences should be different due to rewind
    ASSERT_NE(z0, z3);

    // we'll count negatives as well
    int negs = 0;

    // make sure every value stays within distribution borders
    for (int e = 0; e < z0.size(); e++) {
        auto v = z0[e];
        if (!(v >= -1.0 && v <= 1.0)) {
            nd4j_printf("Failed at idx [%i]: %f\n", e, (float) v);
            ASSERT_TRUE(v >= -1.0 && v <= 1.0);
        }

        if (v < 0.0)
            negs++;
    }

    ASSERT_TRUE(negs > 0);
}