#include "testlayers.h"
#include <Context.h>
#include <iomanip>
#include <Variable.h>
#include <VariableSpace.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/CustomOperations.h>
#include <helpers/helper_hash.h>
#include <NDArray.h>
#include <array/NDArrayList.h>


using namespace nd4j;
using namespace nd4j::graph;

class DeclarableOpsTests2 : public testing::Test {
public:
    
    DeclarableOpsTests2() {
        printf("\n");
    }
};

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, Gather_test_1) {
    
    NDArray<float> input   ('c', {2,3,4}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24});
    NDArray<float> indices ('c', {1,6},   {0,1, 2,2, 1,2});
    NDArray<float> expected('c', {2,6,4}, {1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12, 9,10,11,12, 5, 6, 7, 8, 9,10,11,12, 13,14,15,16, 17,18,19,20, 21,22,23,24, 21,22,23,24, 17,18,19,20, 21,22,23,24});

    nd4j::ops::gather<float> op;

    ResultSet<float>* result = op.execute({&input, &indices}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    NDArray<float>* output = result->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete result;
}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, Gather_test_2) {
    
    NDArray<float> input   ('c', {2,3,4}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24});
    NDArray<float> indices ('c', {1,1},   {2});
    NDArray<float> expected('c', {2,4}, {9,10,11,12,21,22,23,24});

    nd4j::ops::gather<float> op;

    ResultSet<float>* result = op.execute({&input, &indices}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    NDArray<float>* output = result->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, Gather_test_3) {
    
    NDArray<float> input   ('c', {2,3,4},   {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24});
    NDArray<float> indices ('c', {2,3},     {0, 1, 2, 2, 1,2} );
    NDArray<float> expected('c', {2,2,3,4}, {1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,  9,10,11,12, 5, 6, 7, 8, 9,10,11,12, 13,14,15,16,17,18,19,20,21,22,23,24, 21,22,23,24,17,18,19,20,21,22,23,24});

    nd4j::ops::gather<float> op;

    ResultSet<float>* result = op.execute({&input, &indices}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    NDArray<float>* output = result->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete result;
}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, Gather_test_4) {
    
    NDArray<float> input   ('c', {3,3,4},   {1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12, 13,14,15,16,17,18,19,20,21,22,23,24, 25,26,27,28,29,30,31,32,33,34,35,36});
    NDArray<float> indices ('c', {2,3},     {0, 1, 2, 2, 1,2} );
    NDArray<float> expected('c', {2,3,3,4}, {1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12, 13,14,15,16,17,18,19,20,21,22,23,24, 25,26,27,28,29,30,31,32,33,34,35,36, 25,26,27,28,29,30,31,32,33,34,35,36, 13,14,15,16,17,18,19,20,21,22,23,24, 25,26,27,28,29,30,31,32,33,34,35,36});

    nd4j::ops::gather<float> op;

    ResultSet<float>* result = op.execute({&input, &indices}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    NDArray<float>* output = result->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete result;
}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, Gather_test_5) {
    
    NDArray<float> input   ('c', {2,3,4},   {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24});
    NDArray<float> indices ('c', {2,3},     {0, 1, 2, 2, 1,2} );
    NDArray<float> expected('c', {2,3,2,3}, {1, 2, 3, 3, 2, 3, 5, 6, 7, 7, 6, 7, 9,10,11,11,10,11, 13,14,15,15,14,15, 17,18,19,19,18,19, 21,22,23,23,22,23});

    nd4j::ops::gather<float> op;

    ResultSet<float>* result = op.execute({&input, &indices}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    NDArray<float>* output = result->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete result;
}

 