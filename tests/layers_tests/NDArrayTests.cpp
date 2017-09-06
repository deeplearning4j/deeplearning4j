//
// Created by raver119 on 04.08.17.
//

#include "testlayers.h"

//////////////////////////////////////////////////////////////////////
class NDArrayTest : public testing::Test {
public:
    int alpha = 0;

    int *cShape = new int[8]{2, 2, 2, 2, 1, 0, 1, 99};
    int *fShape = new int[8]{2, 2, 2, 1, 2, 0, 1, 102};

	float arr1[6] = {1,2,3,4,5,6};
	int shape1[8] = {2,2,3,3,1,0,1,99};
	float arr2[48] = {1,2,3,1,2,3,4,5,6,4,5,6,1,2,3,1,2,3,4,5,6,4,5,6,1,2,3,1,2,3,4,5,6,4,5,6,1,2,3,1,2,3,4,5,6,4,5,6};
	int shape2[10] = {3,2,4,6,24,6,1,0,1,99};
	const std::vector<int> tileShape1 = {2,2,2};
};

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, AssignScalar1) {
    auto *array = new NDArray<float>(10, 'c');

    array->assign(2.0f);

    for (int i = 0; i < array->lengthOf(); i++) {
        ASSERT_EQ(2.0f, array->getScalar(i));
    }
}

//////////////////////////////////////////////////////////////////////
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
        ASSERT_EQ(f[i], arrayF->_buffer[i]);
    }

    for (int i = 0; i < 8; i++) {
        ASSERT_EQ(fShape[i], arrayF->_shapeInfo[i]);
    }

    for (int i = 0; i < 4; i++) {
        ASSERT_EQ(c[i], arrayC2->_buffer[i]);
    }

    for (int i = 0; i < 8; i++) {
        ASSERT_EQ(cShape[i], arrayC2->_shapeInfo[i]);
    }
}

//////////////////////////////////////////////////////////////////////
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

//////////////////////////////////////////////////////////////////////
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

TEST_F(NDArrayTest, TestTad1) {
    auto array = new NDArray<float>(3, 3, 'c');

    auto row2 = array->tensorAlongDimension(1, {1});

    ASSERT_TRUE(row2->isView());
    ASSERT_EQ(3, row2->lengthOf());

    row2->assign(1.0);

    row2->printBuffer();

    ASSERT_NEAR(3.0f, array->sumNumber(), 1e-5);

    array->printBuffer();

    delete row2;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestTad2) {
    auto array = new NDArray<float>(3, 3, 'c');

    ASSERT_EQ(3, array->tensorsAlongDimension({1}));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestTad3) {
    auto array = new NDArray<float>(4, 3, 'c');

    auto row2 = array->tensorAlongDimension(1, {1});

    ASSERT_TRUE(row2->isView());
    ASSERT_EQ(3, row2->lengthOf());

    row2->putScalar(1, 1.0);

    array->printBuffer();

    row2->putIndexedScalar(2, 1.0);

    array->printBuffer();

    delete row2;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestRepeat1) {
    auto eBuffer = new float[8] {1.0,2.0,1.0,2.0,3.0,4.0,3.0,4.0};
    auto eShape = new int[8]{2, 4, 2, 2, 1, 0, 1, 99};
    auto array = new NDArray<float>(2, 2, 'c');
    auto exp = new NDArray<float>(eBuffer, eShape);
    for (int e = 0; e < array->lengthOf(); e++)
        array->putScalar(e, e + 1);

    array->printBuffer();

    auto rep = array->repeat(0, {2});

    ASSERT_EQ(4, rep->sizeAt(0));
    ASSERT_EQ(2, rep->sizeAt(1));

    rep->printBuffer();

    ASSERT_TRUE(exp->equalsTo(rep));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestIndexedPut1) {
    auto array = new NDArray<float>(3, 3, 'f');

    array->putIndexedScalar(4, 1.0f);
    ASSERT_EQ(1.0f, array->getIndexedScalar(4));
    array->printBuffer();
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestSum1) {
    float *c = new float[4] {1, 2, 3, 4};

    auto array = new NDArray<float>(c, cShape);

    ASSERT_EQ(10.0f, array->sumNumber());
    ASSERT_EQ(2.5f, array->meanNumber());
}

//////////////////////////////////////////////////////////////////////
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

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test3D_1) {
    auto arrayC = new NDArray<double>('c', {2, 5, 10});
    auto arrayF = new NDArray<double>('f', {2, 5, 10});

    ASSERT_EQ(100, arrayC->lengthOf());
    ASSERT_EQ(100, arrayF->lengthOf());

    ASSERT_EQ('c', arrayC->ordering());
    ASSERT_EQ('f', arrayF->ordering());
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestTranspose1) {
    auto *arrayC = new NDArray<double>('c', {2, 5, 10});

    int *expC = new int[10] {3, 2, 5, 10, 50, 10, 1, 0, 1, 99};
    int *expT = new int[10] {3, 10, 5, 2, 1, 10, 50, 0, 1, 102};

    auto *arrayT = arrayC->transpose();


    for (int e = 0; e < arrayC->rankOf() * 2 + 4; e++) {
        ASSERT_EQ(expC[e], arrayC->_shapeInfo[e]);
        ASSERT_EQ(expT[e], arrayT->_shapeInfo[e]);
    }

}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestTranspose2) {
    auto *arrayC = new NDArray<double>('c', {2, 5, 10});

    int *expC = new int[10] {3, 2, 5, 10, 50, 10, 1, 0, 1, 99};
    int *expT = new int[10] {3, 10, 5, 2, 1, 10, 50, 0, 1, 102};

    arrayC->transposei();


    for (int e = 0; e < arrayC->rankOf() * 2 + 4; e++) {
        ASSERT_EQ(expT[e], arrayC->_shapeInfo[e]);
    }

}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestSumAlongDimension1) {
    float *c = new float[4] {1, 2, 3, 4};
    auto *array = new NDArray<float>(c, cShape);

    auto *res = array->sum({0});

    ASSERT_EQ(2, res->lengthOf());

    ASSERT_EQ(4.0f, res->getScalar(0));
    ASSERT_EQ(6.0f, res->getScalar(1));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestSumAlongDimension2) {
    float *c = new float[4] {1, 2, 3, 4};
    auto *array = new NDArray<float>(c, cShape);

    auto *res = array->sum({1});

    ASSERT_EQ(2, res->lengthOf());

    ASSERT_EQ(3.0f, res->getScalar(0));
    ASSERT_EQ(7.0f, res->getScalar(1));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestReduceAlongDimension1) {
    float *c = new float[4] {1, 2, 3, 4};
    auto *array = new NDArray<float>(c, cShape);

    auto *exp = array->sum({1});
    auto *res = array->reduceAlongDimension<simdOps::Sum<float>>({1});



    ASSERT_EQ(2, res->lengthOf());

    ASSERT_EQ(3.0f, res->getScalar(0));
    ASSERT_EQ(7.0f, res->getScalar(1));

}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestTransform1) {
    float *c = new float[4] {-1, -2, -3, -4};
    auto *array = new NDArray<float>(c, cShape);

    float *e = new float[4] {1, 2, 3, 4};
    auto *exp = new NDArray<float>(e, cShape);

    array->applyTransform<simdOps::Abs<float>>();

    ASSERT_TRUE(exp->equalsTo(array));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestReduceScalar1) {
    float *c = new float[4] {-1, -2, -3, -4};
    auto *array = new NDArray<float>(c, cShape);

    ASSERT_EQ(-4, array->reduceNumber<simdOps::Min<float>>(nullptr));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestApplyTransform1) {
    float *c = new float[4] {-1, -2, -3, -4};
    auto *array = new NDArray<float>(c, cShape);

    float *e = new float[4] {1, 2, 3, 4};
    auto *exp = new NDArray<float>(e, cShape);

    array->applyTransform<simdOps::Abs<float>>();


    ASSERT_TRUE(exp->equalsTo(array));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestVectors1) {
    float *c = new float[4]{-1, -2, -3, -4};
    auto *array = new NDArray<float>(c, cShape);


    auto vecShape = array->getShapeAsVector();
    auto vecBuffer = array->getBufferAsVector();

    ASSERT_EQ(8, vecShape.size());
    ASSERT_EQ(4, vecBuffer.size());

    for (int e = 0; e < vecBuffer.size(); e++) {
        ASSERT_NEAR(c[e], vecBuffer.at(e), 1e-5);
    }

    for (int e = 0; e < vecShape.size(); e++) {
        ASSERT_EQ(cShape[e], vecShape.at(e));
    }
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestChecks1) {
    NDArray<float> array(1, 5, 'c');

    ASSERT_FALSE(array.isMatrix());
    ASSERT_FALSE(array.isScalar());
    ASSERT_TRUE(array.isVector());
    ASSERT_FALSE(array.isColumnVector());
    ASSERT_TRUE(array.isRowVector());
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestChecks2) {
    NDArray<float> array(5, 5, 'c');

    ASSERT_TRUE(array.isMatrix());
    ASSERT_FALSE(array.isScalar());
    ASSERT_FALSE(array.isVector());
    ASSERT_FALSE(array.isColumnVector());
    ASSERT_FALSE(array.isRowVector());
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestChecks3) {
    NDArray<float> array(5, 1, 'c');

    ASSERT_FALSE(array.isMatrix());
    ASSERT_FALSE(array.isScalar());
    ASSERT_TRUE(array.isVector());
    ASSERT_TRUE(array.isColumnVector());
    ASSERT_FALSE(array.isRowVector());
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestChecks4) {
    NDArray<float> array(1, 1, 'c');

    ASSERT_FALSE(array.isMatrix());
    ASSERT_FALSE(array.isVector());
    ASSERT_FALSE(array.isColumnVector());
    ASSERT_FALSE(array.isRowVector());
    ASSERT_TRUE(array.isScalar());
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestChecks5) {
    NDArray<float> array('c', {5, 5, 5});

    ASSERT_FALSE(array.isMatrix());
    ASSERT_FALSE(array.isVector());
    ASSERT_FALSE(array.isColumnVector());
    ASSERT_FALSE(array.isRowVector());
    ASSERT_FALSE(array.isScalar());
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestTile1) {

	NDArray<float> array1(arr1,shape1);
	NDArray<float> array2(arr2,shape2);
    auto expA = array1.dup('c');

    auto tiled = array1.tile(tileShape1);

    tiled->printShapeInfo();
    tiled->printBuffer();

	ASSERT_TRUE(tiled->isSameShapeStrict(&array2));
	ASSERT_TRUE(tiled->equalsTo(&array2));

    ASSERT_TRUE(expA->isSameShapeStrict(&array1));
    ASSERT_TRUE(expA->equalsTo(&array1));
	
	delete tiled;
	delete expA;
}

TEST_F(NDArrayTest, TestTile2) {

	NDArray<float> array1(arr1,shape1);
	NDArray<float> array2(arr2,shape2);

    NDArray<float>* tiled = array1.tile(tileShape1);

	ASSERT_TRUE(tiled->isSameShapeStrict(&array2));
	ASSERT_TRUE(tiled->equalsTo(&array2));
	delete tiled;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestTile3) {

	NDArray<float> array1(arr1,shape1);
	NDArray<float> array2(arr2,shape2);

    array1.tilei(tileShape1);

	ASSERT_TRUE(array1.isSameShapeStrict(&array2));
	ASSERT_TRUE(array1.equalsTo(&array2));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestMmulHelper1) {
    auto xBuffer = new float[3]{1.f, 2.f, 3.f};
    auto xShape = new int[8] {2, 1, 3, 1, 1, 0, 1, 99};
    auto x = new NDArray<float>(xBuffer, xShape);

    auto yBuffer = new float[3]{2.f, 4.f, 6.f};
    auto yShape = new int[8] {2, 1, 3, 1, 1, 0, 1, 99};
    auto y = new NDArray<float>(yBuffer, yShape);

    auto z = NDArray<float>::mmulHelper(x, y);

    ASSERT_EQ(1, z->lengthOf());
    ASSERT_NEAR(28, z->getScalar(0), 1e-5);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestMmulHelper2) {
    auto xBuffer = new float[15]{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f};
    auto xShape = new int[8] {2, 5, 3, 3, 1, 0, 1, 99};
    auto x = new NDArray<float>(xBuffer, xShape);

    auto yBuffer = new float[3]{2.f, 4.f, 6.f};
    auto yShape = new int[8] {2, 3, 1, 1, 1, 0, 1, 99};
    auto y = new NDArray<float>(yBuffer, yShape);

    auto z = new NDArray<float>(5, 1, 'f');

    auto expBuffer = new float[5]{28.00,  64.00,  100.00,  136.00,  172.00};
    auto exp = new NDArray<float>(expBuffer, z->_shapeInfo);

    //nd4j::blas::GEMV<float>::op('f',  x->rows(), x->columns(), 1.0f, x->_buffer, y->rows(), y->_buffer, 1, 0.0, z->_buffer, 1);

    NDArray<float>::mmulHelper(x, y, z);

    z->printBuffer();

    ASSERT_TRUE(z->equalsTo(exp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestMmulHelper3) {
    auto xBuffer = new float[15]{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f};
    auto xShape = new int[8] {2, 5, 3, 1, 5, 0, 1, 102};
    auto x = new NDArray<float>(xBuffer, xShape);

    auto yBuffer = new float[3]{2.f, 4.f, 6.f};
    auto yShape = new int[8] {2, 3, 1, 1, 1, 0, 1, 99};
    auto y = new NDArray<float>(yBuffer, yShape);

    auto z = new NDArray<float>(5, 1, 'f');

    auto expBuffer = new float[5]{92.00,  104.00,  116.00,  128.00,  140.00};
    auto exp = new NDArray<float>(expBuffer, z->_shapeInfo);

    //nd4j::blas::GEMV<float>::op('f',  x->rows(), x->columns(), 1.0f, x->_buffer, y->rows(), y->_buffer, 1, 0.0, z->_buffer, 1);

    NDArray<float>::mmulHelper(x, y, z);

    z->printBuffer();

    ASSERT_TRUE(z->equalsTo(exp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestMmulHelper4) {
    auto xBuffer = new float[6]{1, 2, 3, 4, 5, 6};
    auto xShape = new int[8] {2, 3, 2, 2, 1, 0, 1, 99};
    auto x = new NDArray<float>(xBuffer, xShape);

    auto yBuffer = new float[6]{7, 8, 9, 0, 1, 2};
    auto yShape = new int[8] {2, 2, 3, 3, 1, 0, 1, 99};
    auto y = new NDArray<float>(yBuffer, yShape);

    auto z = new NDArray<float>(3, 3, 'f');

    auto expBuffer = new float[9]{7.0, 21.0, 35.0, 10.0, 28.0, 46.0, 13.0, 35.0, 57.0};
    auto exp = new NDArray<float>(expBuffer, z->_shapeInfo);

    NDArray<float>::mmulHelper(x, y, z);
    ASSERT_TRUE(z->equalsTo(exp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestMmulHelper5) {
    auto xBuffer = new float[6]{1, 2, 3, 4, 5, 6};
    auto xShape = new int[8] {2, 3, 2, 1, 3, 0, 1, 102};
    auto x = new NDArray<float>(xBuffer, xShape);

    auto yBuffer = new float[6]{7, 8, 9, 0, 1, 2};
    auto yShape = new int[8] {2, 2, 3, 3, 1, 0, 1, 99};
    auto y = new NDArray<float>(yBuffer, yShape);

    auto z = new NDArray<float>(3, 3, 'f');

    auto expBuffer = new float[9]{7.0, 14.0, 21.0, 12.0, 21.0, 30.0, 17.0, 28.0, 39.0};
    auto exp = new NDArray<float>(expBuffer, z->_shapeInfo);

    NDArray<float>::mmulHelper(x, y, z);
    ASSERT_TRUE(z->equalsTo(exp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestMmulHelper6) {
    auto xBuffer = new float[6]{1, 2, 3, 4, 5, 6};
    auto xShape = new int[8] {2, 3, 2, 1, 3, 0, 1, 102};
    auto x = new NDArray<float>(xBuffer, xShape);

    auto yBuffer = new float[6]{7, 8, 9, 0, 1, 2};
    auto yShape = new int[8] {2, 2, 3, 1, 2, 0, 1, 102};
    auto y = new NDArray<float>(yBuffer, yShape);

    auto z = new NDArray<float>(3, 3, 'f');

    auto expBuffer = new float[9]{39.0, 54.0, 69.0, 9.0, 18.0, 27.0, 9.0, 12.0, 15.0};
    auto exp = new NDArray<float>(expBuffer, z->_shapeInfo);

    NDArray<float>::mmulHelper(x, y, z);
    ASSERT_TRUE(z->equalsTo(exp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestMmulHelper7) {
    auto xBuffer = new float[15]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    auto xShape = new int[8] {2, 5, 3, 1, 5, 0, 1, 102};
    auto x = new NDArray<float>(xBuffer, xShape);

    auto yBuffer = new float[5]{2, 4, 6, 8, 10};
    auto yShape = new int[8] {2, 1, 5, 1, 1, 0, 1, 99};
    auto y = new NDArray<float>(yBuffer, yShape);

    auto z = new NDArray<float>(1, 3, 'f');

    auto expBuffer = new float[9]{110.00,  260.00,  410.00};
    auto exp = new NDArray<float>(expBuffer, z->_shapeInfo);

    NDArray<float>::mmulHelper(y, x, z);

    z->printBuffer();
    ASSERT_TRUE(z->equalsTo(exp));
}

//////////////////////////////////////////////////////////////////////
// not-in-place
TEST_F(NDArrayTest, Permute1) {  
    
    const int shape1[] = {3, 5, 10, 15, 150, 15, 1, 0, 1, 99};
	const int shape2[] = {3, 15, 5, 10, 1, 150, 15, 0, -1, 99};
    const std::initializer_list<int> perm = {2, 0, 1};    
    
    NDArray<float> arr1(shape1);
    NDArray<float> arr2(shape2);    

	NDArray<float>* result = arr1.permute(perm);        	
	ASSERT_TRUE(result->isSameShapeStrict(&arr2));

	delete result;
}

//////////////////////////////////////////////////////////////////////
// in-place
TEST_F(NDArrayTest, Permute2) {
    
    const int shape1[] = {3, 5, 10, 15, 150, 15, 1, 0, 1, 99};
	const int shape2[] = {3, 15, 5, 10, 1, 150, 15, 0, -1, 99};
    const std::initializer_list<int> perm = {2, 0, 1};    
    
    NDArray<float> arr1(shape1);
    NDArray<float> arr2(shape2);    

	ASSERT_TRUE(arr1.permutei(perm));
	ASSERT_TRUE(arr1.isSameShapeStrict(&arr2));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Broadcast1) {
    
    const int shape1[10] = {3, 5, 1, 10, 10, 10, 1, 0, 1, 99};
	const int shape2[8]  = {2,    7, 10, 10, 1, 0, 1, 99};
	const int shape3[10] = {3, 5, 7, 10, 70, 10, 1, 0, 1, 99};    
    
	NDArray<float> arr1(shape1);
    NDArray<float> arr2(shape2);    
	NDArray<float> arr3(shape3);    

	NDArray<float>* result = arr1.broadcast(arr2);		
	ASSERT_TRUE(result->isSameShapeStrict(&arr3));
	delete result;
}
