//
// Created by raver119 on 04.08.17.
//

#include "testlayers.h"

class NDArrayTest : public testing::Test {
public:
    int alpha = 0;

    int *cShape = new int[8]{2, 2, 2, 2, 1, 0, 1, 99};
    int *fShape = new int[8]{2, 2, 2, 1, 2, 0, 1, 102};
};


TEST_F(NDArrayTest, AssignScalar1) {
    auto *array = new NDArray<float>(10, 'c');

    array->assign(2.0f);

    for (int i = 0; i < array->lengthOf(); i++) {
        ASSERT_EQ(2.0f, array->getScalar(i));
    }
}



TEST_F(NDArrayTest, NDArrayOrder1) {
    // original part
    float *c = new float[4] {1, 2, 3, 4};

    // expected part
    float *f = new float[4] {1, 3, 2, 4};

    auto *arrayC = new NDArray<float>(c, cShape);
    auto *arrayF = arrayC->dup('f');
    auto *arrayC2 = arrayF->dup('c');

    ASSERT_EQ('c', arrayC->ordering());
    ASSERT_EQ('f', arrayF->ordering());
    ASSERT_EQ('c', arrayC2->ordering());

    for (int i = 0; i < 4; i++) {
        ASSERT_EQ(f[i], arrayF->buffer[i]);
    }

    for (int i = 0; i < 8; i++) {
        ASSERT_EQ(fShape[i], arrayF->shapeInfo[i]);
    }

    for (int i = 0; i < 4; i++) {
        ASSERT_EQ(c[i], arrayC2->buffer[i]);
    }

    for (int i = 0; i < 8; i++) {
        ASSERT_EQ(cShape[i], arrayC2->shapeInfo[i]);
    }
}

TEST_F(NDArrayTest, TestGetScalar1) {
    float *c = new float[4] {1, 2, 3, 4};
    int *cShape = new int[8]{2, 2, 2, 2, 1, 0, 1, 99};

    auto *arrayC = new NDArray<float>(c, cShape);

    ASSERT_EQ(3.0f, arrayC->getScalar(1, 0));
    ASSERT_EQ(4.0f, arrayC->getScalar(1, 1));

    auto *arrayF = arrayC->dup('f');

    ASSERT_EQ(3.0f, arrayF->getScalar(1, 0));
    ASSERT_EQ(4.0f, arrayF->getScalar(1, 1));


    arrayF->putScalar(1, 0, 7.0f);
    ASSERT_EQ(7.0f, arrayF->getScalar(1, 0));


    arrayC->putScalar(1, 1, 9.0f);
    ASSERT_EQ(9.0f, arrayC->getScalar(1, 1));

}


TEST_F(NDArrayTest, EqualityTest1) {
    auto *arrayA = new NDArray<float>(3, 5, 'f');
    auto *arrayB = new NDArray<float>(3, 5, 'f');
    auto *arrayC = new NDArray<float>(3, 5, 'f');

    auto *arrayD = new NDArray<float>(2, 4, 'f');
    auto *arrayE = new NDArray<float>(15, 'f');

    for (int i = 0; i < arrayA->rows(); i++) {
        for (int k = 0; k < arrayA->columns(); k++) {
            arrayA->putScalar(i, k, (float) i);
        }
    }

    for (int i = 0; i < arrayB->rows(); i++) {
        for (int k = 0; k < arrayB->columns(); k++) {
            arrayB->putScalar(i, k, (float) i);
        }
    }

    for (int i = 0; i < arrayC->rows(); i++) {
        for (int k = 0; k < arrayC->columns(); k++) {
            arrayC->putScalar(i, k, (float) i+1);
        }
    }



    ASSERT_TRUE(arrayA->equalsTo(arrayB, 1e-5));

    ASSERT_FALSE(arrayC->equalsTo(arrayB, 1e-5));

    ASSERT_FALSE(arrayD->equalsTo(arrayB, 1e-5));

    ASSERT_FALSE(arrayE->equalsTo(arrayB, 1e-5));
}



TEST_F(NDArrayTest, TestSum1) {
    float *c = new float[4] {1, 2, 3, 4};

    auto *array = new NDArray<float>(c, cShape);

    ASSERT_EQ(10.0f, array->sumNumber());
    ASSERT_EQ(2.5f, array->meanNumber());
}

TEST_F(NDArrayTest, TestAddiRowVector) {
    float *c = new float[4] {1, 2, 3, 4};
    float *e = new float[4] {2, 3, 4, 5};

    auto *array = new NDArray<float>(c, cShape);
    auto *row = new NDArray<float>(2, 'c');
    auto *exp = new NDArray<float>(e, cShape);
    row->assign(1.0f);

    array->addiRowVector(row);

    ASSERT_TRUE(exp->equalsTo(array));
}