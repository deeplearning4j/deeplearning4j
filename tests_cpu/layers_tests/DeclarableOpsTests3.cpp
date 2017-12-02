#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <helpers/helper_hash.h>
#include <NDArray.h>
#include <array/NDArrayList.h>


using namespace nd4j;
using namespace nd4j::graph;

class DeclarableOpsTests3 : public testing::Test {
public:
    
    DeclarableOpsTests3() {
        printf("\n");
        fflush(stdout);
    }
};


TEST_F(DeclarableOpsTests3, Test_Permute_1) {
    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> permute('c', {1, 3}, {0, 2, 1});
    NDArray<float> exp('c', {2, 4, 3});

    nd4j::ops::permute<float> op;
    auto result = op.execute({&x, &permute}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));

    delete result;
}

TEST_F(DeclarableOpsTests3, Test_Permute_2) {
    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {4, 3, 2});

    nd4j::ops::permute<float> op;
    auto result = op.execute({&x}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));

    delete result;
}
