//
// @author raver119@gmail.com
//

#include <helpers/shape.h>
#include "testlayers.h"

using namespace nd4j;
using namespace nd4j::graph;

class ShapeTests : public testing::Test {
public:

};


TEST_F(ShapeTests, Test_Basics_1) {
    int shape[] = {2, 5, 3, 3, 1, 0, 1, 99};

    ASSERT_EQ(2, shape::rank(shape));
    ASSERT_EQ(1, shape::elementWiseStride(shape));
    ASSERT_EQ(5, shape::sizeAt(shape, 0));
    ASSERT_EQ(3, shape::sizeAt(shape, 1));
    ASSERT_EQ('c', shape::order(shape));
}


TEST_F(ShapeTests, Test_Basics_2) {
    int shape[] = {4, 2, 3, 4, 5, 60, 20, 5, 1, 0, -1, 102};

    ASSERT_EQ(4, shape::rank(shape));
    ASSERT_EQ(-1, shape::elementWiseStride(shape));
    ASSERT_EQ(2, shape::sizeAt(shape, 0));
    ASSERT_EQ(3, shape::sizeAt(shape, 1));
    ASSERT_EQ(4, shape::sizeAt(shape, 2));
    ASSERT_EQ(5, shape::sizeAt(shape, 3));
    ASSERT_EQ('f', shape::order(shape));
}


TEST_F(ShapeTests, Test_tadLength_1) {
    int shape[] = {4, 2, 3, 4, 5, 60, 20, 5, 1, 0, -1, 102};
    int axis[] = {2, 3};

    ASSERT_EQ(20, shape::tadLength(shape, axis, 2));
}


TEST_F(ShapeTests, Test_ShapeEquality_1) {
    int shape[] = {4, 2, 3, 4, 5, 60, 20, 5, 1, 0, -1, 102};
    int shape_GOOD[] = {4, 2, 3, 4, 5, 60, 20, 5, 1, 0, 1, 99};
    int shape_BAD[] = {4, 3, 3, 4, 5, 60, 20, 5, 1, 0, -1, 102};
    

    ASSERT_TRUE(shape::equalsSoft(shape, shape_GOOD));
    ASSERT_FALSE(shape::equalsSoft(shape, shape_BAD));
}

TEST_F(ShapeTests, Test_ShapeEquality_2) {
    int shape[] = {4, 2, 3, 4, 5, 60, 20, 5, 1, 0, -1, 102};
    int shape_GOOD[] = {4, 2, 3, 4, 5, 60, 20, 5, 1, 0, -1, 102};
    int shape_BAD[] = {4, 2, 3, 4, 5, 60, 20, 5, 1, 0, -1, 99};
    

    ASSERT_TRUE(shape::equalsStrict(shape, shape_GOOD));
    ASSERT_FALSE(shape::equalsStrict(shape, shape_BAD));
}

TEST_F(ShapeTests, Test_Ind2SubC_1) {
    int shape[] = {3, 5};
    auto c0 = shape::ind2subC(2, shape, 0);

    ASSERT_EQ(0, c0[0]);
    ASSERT_EQ(0, c0[1]);

    auto c1 = shape::ind2subC(2, shape, 1);

    ASSERT_EQ(0, c1[0]);
    ASSERT_EQ(1, c1[1]);

    auto c6 = shape::ind2subC(2, shape, 5);

    ASSERT_EQ(1, c6[0]);
    ASSERT_EQ(0, c6[1]);

    delete[] c0;
    delete[] c1;
    delete[] c6;
}

TEST_F(ShapeTests, Test_Ind2Sub_1) {
    int shape[] = {3, 5};
    auto c0 = shape::ind2sub(2, shape, 0);

    ASSERT_EQ(0, c0[0]);
    ASSERT_EQ(0, c0[1]);

    auto c1 = shape::ind2sub(2, shape, 1);

    ASSERT_EQ(1, c1[0]);
    ASSERT_EQ(0, c1[1]);

    auto c6 = shape::ind2sub(2, shape, 5);

    ASSERT_EQ(2, c6[0]);
    ASSERT_EQ(1, c6[1]);

    delete[] c0;
    delete[] c1;
    delete[] c6;
}

TEST_F(ShapeTests, Test_ShapeDetector_1) {
    int shape[] = {2, 5, 3, 3, 1, 0, 1, 99};

    ASSERT_TRUE(shape::isMatrix(shape));
}

TEST_F(ShapeTests, Test_ShapeDetector_2) {
    int shape[] = {3, 2, 5, 3, 15, 3, 1, 0, 1, 99};

    ASSERT_FALSE(shape::isMatrix(shape));
}

TEST_F(ShapeTests, Test_ShapeDetector_3) {
    int shape[] = {2, 1, 3, 3, 1, 0, 1, 99};

    ASSERT_FALSE(shape::isColumnVector(shape));
    ASSERT_TRUE(shape::isVector(shape));
    ASSERT_TRUE(shape::isRowVector(shape));
    ASSERT_FALSE(shape::isMatrix(shape));
}


TEST_F(ShapeTests, Test_ShapeDetector_4) {
    int shape[] = {2, 3, 1, 1, 1, 0, 1, 99};

    ASSERT_TRUE(shape::isColumnVector(shape));
    ASSERT_TRUE(shape::isVector(shape));
    ASSERT_FALSE(shape::isRowVector(shape));
    ASSERT_FALSE(shape::isMatrix(shape));
}

TEST_F(ShapeTests, Test_ShapeDetector_5) {
    int shape[] = {2, 1, 1, 1, 1, 0, 1, 99};

    ASSERT_TRUE(shape::isScalar(shape));
    ASSERT_FALSE(shape::isMatrix(shape));

    // edge case here. Technicaly it's still a vector with length of 1
    ASSERT_TRUE(shape::isVector(shape));
}

TEST_F(ShapeTests, Test_ShapeDetector_6) {
    int shape[] = {2, 1, 1, 1, 1, 0, 1, 99};

    ASSERT_EQ(8, shape::shapeInfoLength(shape));
    ASSERT_EQ(32, shape::shapeInfoByteLength(shape));
}

TEST_F(ShapeTests, Test_ShapeDetector_7) {
    int shape[] = {3, 1, 1, 1, 1, 1, 1, 0, 1, 99};

    ASSERT_EQ(10, shape::shapeInfoLength(shape));
    ASSERT_EQ(40, shape::shapeInfoByteLength(shape));
}

TEST_F(ShapeTests, Test_Transpose_1) {
    int shape[] = {3, 2, 5, 3, 15, 3, 1, 0, 1, 99};
    int exp[] = {3, 3, 5, 2, 1, 3, 15, 0, 1, 102};

    shape::transposeInplace(shape);

    ASSERT_TRUE(shape::equalsStrict(exp, shape));
}

TEST_F(ShapeTests, Test_Transpose_2) {
    int shape[] = {2, 5, 3, 3, 1, 0, 1, 99};
    int exp[] = {2, 3, 5, 1, 3, 0, 1, 102};

    shape::transposeInplace(shape);

    ASSERT_TRUE(shape::equalsStrict(exp, shape));
}

TEST_F(ShapeTests, Test_Transpose_3) {
    int shape[] = {2, 1, 3, 3, 1, 0, 1, 99};
    int exp[] = {2, 3, 1, 1, 3, 0, 1, 102};

    shape::transposeInplace(shape);

    ASSERT_TRUE(shape::equalsStrict(exp, shape));
}


TEST_F(ShapeTests, Test_Transpose_4) {
    int shape[] = {4, 2, 3, 4, 5, 5, 4, 3, 2, 0, 1, 99};
    int exp[] = {4, 5, 4, 3, 2, 2, 3, 4, 5, 0, 1, 102};

    shape::transposeInplace(shape);

    ASSERT_TRUE(shape::equalsStrict(exp, shape));
}
