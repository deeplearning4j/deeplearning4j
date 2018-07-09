//
// Created by raver119 on 02/07/18.
//

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <loops/type_conversions.h>

using namespace nd4j;
using namespace nd4j::ops;
using namespace nd4j::graph;

class TypeCastTests : public testing::Test {
public:

};

TEST_F(TypeCastTests, Test_Cast_1) {
    const int limit = 100;
    auto src = new double[limit];
    auto z = new float[limit];
    auto exp = new float[limit];

    for (int e = 0; e < limit; e++) {
        src[e] = static_cast<double>(e);
        exp[e] = static_cast<float>(e);
    }

    TypeCast::convertGeneric<double, float>(nullptr, reinterpret_cast<void *>(src), limit, reinterpret_cast<void *>(z));

    for (int e = 0; e < limit; e++) {
        ASSERT_NEAR(exp[e], z[e], 1e-5f);
    }

    delete[] src;
    delete[] z;
    delete[] exp;
}

TEST_F(TypeCastTests, Test_ConvertDtype_1) {
    float src[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float16 dst[5];
    float16 exp[] = {(float16) 1.0f, (float16) 2.0f, (float16) 3.0f, (float16) 4.0f, (float16) 5.0f};

    NativeOps ops;
    ops.convertTypes(nullptr, ND4J_FLOAT32, src, 5, ND4J_FLOAT16, dst);

    for (int e = 0; e < 5; e++)
        ASSERT_NEAR(exp[e], dst[e], (float16) 0.01f);
}