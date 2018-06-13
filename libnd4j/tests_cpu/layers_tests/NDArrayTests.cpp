//
// Created by raver119 on 04.08.17.
//

#include "testlayers.h"
#include <memory>
#include <NDArray.h>
#include <NDArrayFactory.h>

using namespace nd4j;

//////////////////////////////////////////////////////////////////////
class NDArrayTest : public testing::Test {
public:
    int alpha = 0;

    Nd4jLong *cShape = new Nd4jLong[8]{2, 2, 2, 2, 1, 0, 1, 99};
    Nd4jLong *fShape = new Nd4jLong[8]{2, 2, 2, 1, 2, 0, 1, 102};

	float arr1[6] = {1,2,3,4,5,6};
	Nd4jLong shape1[8] = {2,2,3,3,1,0,1,99};
	float arr2[48] = {1,2,3,1,2,3,4,5,6,4,5,6,1,2,3,1,2,3,4,5,6,4,5,6,1,2,3,1,2,3,4,5,6,4,5,6,1,2,3,1,2,3,4,5,6,4,5,6};
	Nd4jLong shape2[10] = {3,2,4,6,24,6,1,0,1,99};
	const std::vector<Nd4jLong> tileShape1 = {2,2,2};


    ~NDArrayTest() {
        delete[] cShape;
        delete[] fShape;
    }
};


//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestDup1) {
    NDArray<float> array(arr1, shape1);

    auto arrC = array.dup('c');
    auto arrF = array.dup('f');

    //arrC->printShapeInfo("C shape");
    //arrF->printShapeInfo("F shape");

    ASSERT_TRUE(array.equalsTo(arrF));
    ASSERT_TRUE(array.equalsTo(arrC));

    ASSERT_TRUE(arrF->equalsTo(arrC));

    delete arrC;
    delete arrF;
}


//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, AssignScalar1) {
    auto *array = new NDArray<float>('c', {1, 10});

    array->assign(2.0f);

    for (int i = 0; i < array->lengthOf(); i++) {
        ASSERT_EQ(2.0f, array->getScalar(i));
    }

    delete array;
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
        ASSERT_EQ(f[i], arrayF->getBuffer()[i]);
    }

    for (int i = 0; i < 8; i++) {
        ASSERT_EQ(fShape[i], arrayF->getShapeInfo()[i]);
    }

    for (int i = 0; i < 4; i++) {
        ASSERT_EQ(c[i], arrayC2->getBuffer()[i]);
    }

    for (int i = 0; i < 8; i++) {
        ASSERT_EQ(cShape[i], arrayC2->getShapeInfo()[i]);
    }

    delete[] c;
    delete[] f;
    delete arrayC;
    delete arrayF;
    delete arrayC2;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestGetScalar1) {
    float *c = new float[4] {1, 2, 3, 4};
    auto cShape = new Nd4jLong[8]{2, 2, 2, 2, 1, 0, 1, 99};

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

    delete[] c;
    delete[] cShape;

    delete arrayC;
    delete arrayF;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, EqualityTest1) {
    auto *arrayA = new NDArray<float>('f', {3, 5});
    auto *arrayB = new NDArray<float>('f', {3, 5});
    auto *arrayC = new NDArray<float>('f', {3, 5});

    auto *arrayD = new NDArray<float>('f', {2, 4});
    auto *arrayE = new NDArray<float>('f', {1, 15});

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

    delete arrayA;
    delete arrayB;
    delete arrayC;
    delete arrayD;
    delete arrayE;
}

TEST_F(NDArrayTest, TestTad1) {
    auto array = new NDArray<float>('c', {3, 3});

    auto row2 = array->tensorAlongDimension(1, {1});

    ASSERT_TRUE(row2->isView());
    ASSERT_EQ(3, row2->lengthOf());

    row2->assign(1.0);

    //row2->printBuffer();

    ASSERT_NEAR(3.0f, array->sumNumber(), 1e-5);

    //array->printBuffer();

    delete row2;
    delete array;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestTad2) {
    auto array = new NDArray<float>('c', {3, 3});

    ASSERT_EQ(3, array->tensorsAlongDimension({1}));

    delete array;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestTad3) {
    auto array = new NDArray<float>('c', {4, 3});

    auto row2 = array->tensorAlongDimension(1, {1});

    ASSERT_TRUE(row2->isView());
    ASSERT_EQ(3, row2->lengthOf());

    row2->putScalar(1, 1.0);

    //array->printBuffer();

    row2->putIndexedScalar(2, 1.0);

    //array->printBuffer();

    delete row2;
    delete array;
}


TEST_F(NDArrayTest, TestPermuteReshape1) {
    NDArray<float> array('c', {2, 2, 5, 5});
    int pShape[] = {4, 2, 5, 5, 2, 25, 5, 1, 50, 0, -1, 99};
    int rShape[] = {3, 2, 25, 2, 25, 1, 50, 0, -1, 99};

    array.permutei({1, 2, 3, 0});

    for (int e = 0; e < shape::shapeInfoLength(array.getShapeInfo()); e++)
        ASSERT_EQ(pShape[e], array.getShapeInfo()[e]);


    array.reshapei('c', {2, 25, 2});

    for (int e = 0; e < shape::shapeInfoLength(array.getShapeInfo()); e++)
        ASSERT_EQ(rShape[e], array.getShapeInfo()[e]);
}


TEST_F(NDArrayTest, TestPermuteReshape2) {
    NDArray<float> array('c', {2, 2, 5, 5, 6, 6});
    int pShape[] = {6, 2, 2, 6, 6, 5, 5, 900, 1800, 6, 1, 180, 36, 0, -1, 99};
    int rShape[] = {3, 2, 72, 25, 1800, 25, 1, 0, 1, 99};


    // array.printShapeInfo("before");

    array.permutei({1, 0, 4, 5, 2, 3});

    // array.printShapeInfo("after ");

    for (int e = 0; e < shape::shapeInfoLength(array.getShapeInfo()); e++)
        ASSERT_EQ(pShape[e], array.getShapeInfo()[e]);

    array.reshapei('c', {2, 72, 25});

    for (int e = 0; e < shape::shapeInfoLength(array.getShapeInfo()); e++)
        ASSERT_EQ(rShape[e], array.getShapeInfo()[e]);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestRepeat1) {
    auto eBuffer = new float[8] {1.0,2.0,1.0,2.0,3.0,4.0,3.0,4.0};
    auto eShape = new Nd4jLong[8]{2, 4, 2, 2, 1, 0, 1, 99};
    auto array = new NDArray<float>('c', {2, 2});
    auto exp = new NDArray<float>(eBuffer, eShape);
    for (int e = 0; e < array->lengthOf(); e++)
        array->putScalar(e, e + 1);

    //array->printBuffer();

    auto rep = array->repeat(0, {2});

    ASSERT_EQ(4, rep->sizeAt(0));
    ASSERT_EQ(2, rep->sizeAt(1));

    //rep->printBuffer();

    ASSERT_TRUE(exp->equalsTo(rep));

    delete[] eBuffer;
    delete[] eShape;
    delete array;
    delete exp;
    delete rep;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestIndexedPut1) {
    auto array = new NDArray<float>('f', {3, 3});

    array->putIndexedScalar(4, 1.0f);
    ASSERT_EQ(1.0f, array->getIndexedScalar(4));
    //array->printBuffer();

    delete array;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestSum1) {
    float *c = new float[4] {1, 2, 3, 4};

    auto array = new NDArray<float>(c, cShape);

    ASSERT_EQ(10.0f, array->sumNumber());
    ASSERT_EQ(2.5f, array->meanNumber());

    delete[] c;
    delete array;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestAddiRowVector) {
    float *c = new float[4] {1, 2, 3, 4};
    float *e = new float[4] {2, 3, 4, 5};

    auto *array = new NDArray<float>(c, cShape);
    auto *row = new NDArray<float>('c', {1, 2});
    auto *exp = new NDArray<float>(e, cShape);
    row->assign(1.0f);

    array->addiRowVector(row);

    ASSERT_TRUE(exp->equalsTo(array));

    delete[] c;
    delete[] e;

    delete array;
    delete row;
    delete exp;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestAddiColumnVector) {
    float arr1[] = {1, 2, 3, 4};
    float arr2[] = {5, 6};
	float arr3[] = {6, 7, 9, 10};
	Nd4jLong shape1[] = {2,2,2,2,1,0,1,99};
	Nd4jLong shape2[] = {2,2,1,1,1,0,1,99};
	NDArray<float> matrix(arr1, shape1);
	NDArray<float> column(arr2, shape2);
	NDArray<float> exp(arr3, shape1);
	
    matrix.addiColumnVector(&column);	
	ASSERT_TRUE(exp.isSameShapeStrict(&matrix));		
    ASSERT_TRUE(exp.equalsTo(&matrix));
}


//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestMuliColumnVector) {
    float arr1[] = {1, 2, 3, 4};
    float arr2[] = {5, 6};
	float arr3[] = {5, 10, 18, 24};
	Nd4jLong shape1[] = {2,2,2,2,1,0,1,99};
	Nd4jLong shape2[] = {2,2,1,1,1,0,1,99};
	NDArray<float> matrix(arr1, shape1);
	NDArray<float> column(arr2, shape2);
	NDArray<float> exp(arr3, shape1);
	
    matrix.muliColumnVector(&column);

	ASSERT_TRUE(exp.isSameShapeStrict(&matrix));	
    ASSERT_TRUE(exp.equalsTo(&matrix));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test3D_1) {
    auto arrayC = new NDArray<double>('c', {2, 5, 10});
    auto arrayF = new NDArray<double>('f', {2, 5, 10});

    ASSERT_EQ(100, arrayC->lengthOf());
    ASSERT_EQ(100, arrayF->lengthOf());

    ASSERT_EQ('c', arrayC->ordering());
    ASSERT_EQ('f', arrayF->ordering());

    delete arrayC;
    delete arrayF;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestTranspose1) {
    auto *arrayC = new NDArray<double>('c', {2, 5, 10});

    auto expC = new Nd4jLong[10] {3, 2, 5, 10, 50, 10, 1, 0, 1, 99};
    auto expT = new Nd4jLong[10] {3, 10, 5, 2, 1, 10, 50, 0, 1, 102};

    auto *arrayT = arrayC->transpose();

    for (int e = 0; e < arrayC->rankOf(); e++) {
        ASSERT_EQ(shape::shapeOf(expC)[e], arrayC->sizeAt(e));
        ASSERT_EQ(shape::shapeOf(expT)[e], arrayT->sizeAt(e));
    }

    delete arrayC;
    delete[] expC;
    delete[] expT;
    delete arrayT;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestTranspose2) {
    auto *arrayC = new NDArray<double>('c', {2, 5, 10});

    auto expC = new Nd4jLong[10] {3, 2, 5, 10, 50, 10, 1, 0, 1, 99};
    auto expT = new Nd4jLong[10] {3, 10, 5, 2, 1, 10, 50, 0, 1, 102};

    arrayC->transposei();


    for (int e = 0; e < arrayC->rankOf(); e++) {
        ASSERT_EQ(shape::shapeOf(expT)[e], arrayC->sizeAt(e));
    }

    delete arrayC;
    delete[] expC;
    delete[] expT;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestSumAlongDimension1) {
    float *c = new float[4] {1, 2, 3, 4};
    auto *array = new NDArray<float>(c, cShape);

    auto *res = array->sum({0});

    ASSERT_EQ(2, res->lengthOf());

    ASSERT_EQ(4.0f, res->getScalar(0));
    ASSERT_EQ(6.0f, res->getScalar(1));

    delete[] c;
    delete array;
    delete res;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestSumAlongDimension2) {
    /*float *c = new float[4] {1, 2, 3, 4};
    auto *array = new NDArray<float>(c, cShape);

    auto *res = array->sum({1});

    ASSERT_EQ(2, res->lengthOf());

    ASSERT_EQ(3.0f, res->getScalar(0));
    ASSERT_EQ(7.0f, res->getScalar(1));

    delete[] c;
    delete array;
    delete res;
    */
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

    delete[] c;
    delete array;
    delete exp;
    delete res;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestTransform1) {
    float *c = new float[4] {-1, -2, -3, -4};
    auto *array = new NDArray<float>(c, cShape);

    float *e = new float[4] {1, 2, 3, 4};
    auto *exp = new NDArray<float>(e, cShape);

    array->applyTransform<simdOps::Abs<float>>();

    ASSERT_TRUE(exp->equalsTo(array));

    delete[] c;
    delete array;
    delete[] e;
    delete exp;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestReduceScalar1) {
    float *c = new float[4] {-1, -2, -3, -4};
    auto *array = new NDArray<float>(c, cShape);

    ASSERT_EQ(-4, array->reduceNumber<simdOps::Min<float>>(nullptr));

    delete[] c;
    delete array;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestReduceScalar2) {
    float *c = new float[4] {-1, -2, -3, -4};
    auto *array = new NDArray<float>(c, cShape);

    ASSERT_EQ(-10, array->reduceNumber<simdOps::Sum<float>>(nullptr));

    delete[] c;
    delete array;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestReduceScalar3) {
    auto *array = new NDArray<float>(arr1, shape1);

    ASSERT_EQ(21, array->reduceNumber<simdOps::Sum<float>>(nullptr));

    delete array;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestApplyTransform1) {
    float *c = new float[4] {-1, -2, -3, -4};
    auto *array = new NDArray<float>(c, cShape);

    float *e = new float[4] {1, 2, 3, 4};
    auto *exp = new NDArray<float>(e, cShape);

    array->applyTransform<simdOps::Abs<float>>();


    ASSERT_TRUE(exp->equalsTo(array));

    delete[] c;
    delete array;

    delete[] e;
    delete exp;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestVectors1) {
    float *c = new float[4]{-1, -2, -3, -4};
    auto *array = new NDArray<float>(c, cShape);


    auto vecShape = array->getShapeInfoAsVector();
    auto vecBuffer = array->getBufferAsVector();

    ASSERT_EQ(8, vecShape.size());
    ASSERT_EQ(4, vecBuffer.size());

    for (int e = 0; e < vecBuffer.size(); e++) {
        ASSERT_NEAR(c[e], vecBuffer.at(e), 1e-5);
    }

    for (int e = 0; e < vecShape.size(); e++) {
        ASSERT_EQ(cShape[e], vecShape.at(e));
    }

    delete[] c;
    delete array;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestChecks1) {
    NDArray<float> array('c', {1, 5});

    ASSERT_FALSE(array.isMatrix());
    ASSERT_FALSE(array.isScalar());
    ASSERT_TRUE(array.isVector());
    ASSERT_FALSE(array.isColumnVector());
    ASSERT_TRUE(array.isRowVector());
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestChecks2) {
    NDArray<float> array('c', {5, 5});

    ASSERT_TRUE(array.isMatrix());
    ASSERT_FALSE(array.isScalar());
    ASSERT_FALSE(array.isVector());
    ASSERT_FALSE(array.isColumnVector());
    ASSERT_FALSE(array.isRowVector());
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestChecks3) {
    NDArray<float> array('c', {5, 1});

    ASSERT_FALSE(array.isMatrix());
    ASSERT_FALSE(array.isScalar());
    ASSERT_TRUE(array.isVector());
    ASSERT_TRUE(array.isColumnVector());
    ASSERT_FALSE(array.isRowVector());
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestChecks4) {
    NDArray<float> array('c', {1, 1});

    ASSERT_FALSE(array.isMatrix());
    ASSERT_FALSE(array.isVector());
    ASSERT_FALSE(array.isColumnVector());
    ASSERT_FALSE(array.isRowVector());
    ASSERT_TRUE(array.isScalar());
}

TEST_F(NDArrayTest, TestReductionAny1) {
    NDArray<float> array('c', {2, 2});
    array.putScalar(0, 1.0f);
    array.putScalar(1, 1.0f);
    array.putScalar(2, 0.0f);
    array.putScalar(3, 0.0f);

    auto result0 = array.template reduceAlongDimension<simdOps::Any<float>>({0});

    ASSERT_EQ(2, result0->lengthOf());

    ASSERT_NEAR(1.0f, result0->getScalar(0), 1e-5f);
    ASSERT_NEAR(1.0f, result0->getScalar(1), 1e-5f);

    auto result1 = array.template reduceAlongDimension<simdOps::Any<float>>({1});

    ASSERT_EQ(2, result1->lengthOf());

    ASSERT_NEAR(1.0f, result1->getScalar(0), 1e-5f);
    ASSERT_NEAR(0.0f, result1->getScalar(1), 1e-5f);

    delete result0;
    delete result1;
}

TEST_F(NDArrayTest, TestReductionAll1) {
    NDArray<float> array('c', {2, 2});
    array.putScalar(0, 1.0f);
    array.putScalar(1, 1.0f);
    array.putScalar(2, 0.0f);
    array.putScalar(3, 0.0f);

    auto result0 = array.template reduceAlongDimension<simdOps::All<float>>({0});

    ASSERT_EQ(2, result0->lengthOf());

    ASSERT_NEAR(0.0f, result0->getScalar(0), 1e-5f);
    ASSERT_NEAR(0.0f, result0->getScalar(1), 1e-5f);

    auto result1 = array.template reduceAlongDimension<simdOps::All<float>>({1});

    ASSERT_EQ(2, result1->lengthOf());

    ASSERT_NEAR(1.0f, result1->getScalar(0), 1e-5f);
    ASSERT_NEAR(0.0f, result1->getScalar(1), 1e-5f);

    delete result0;
    delete result1;
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

    //array2.printShapeInfo("Expct shape");
    //tiled.printShapeInfo("Tiled shape");
    //tiled.printBuffer();

	ASSERT_TRUE(tiled.isSameShape(&array2));
	ASSERT_TRUE(tiled.equalsTo(&array2));

    ASSERT_TRUE(expA->isSameShape(&array1));
    ASSERT_TRUE(expA->equalsTo(&array1));
	
	// delete tiled;
	delete expA;
}

TEST_F(NDArrayTest, TestTile2) {

	NDArray<float> array1(arr1,shape1);
	NDArray<float> array2(arr2,shape2);

    NDArray<float> tiled = array1.tile(tileShape1);

	ASSERT_TRUE(tiled.isSameShape(&array2));
	ASSERT_TRUE(tiled.equalsTo(&array2));
	// delete tiled;
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
TEST_F(NDArrayTest, TestTile4) {

    float xBuff[]   = {1,2,3,4,5,6};
    float expBuff[] = {1.,2., 1.,2., 3.,4., 3.,4., 5.,6., 5.,6.};

    NDArray<float> x(xBuff, 'c', {3,1,2});
    NDArray<float> exp(expBuff, 'c', {3,2,2});    

    NDArray<float> result = x.tile({2,1});

    ASSERT_TRUE(result.isSameShapeStrict(&exp));
    ASSERT_TRUE(result.equalsTo(&exp));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestTile5) {

    float xBuff[]   = {1,2,3,4,5,6,7,8,9,10,11,12};
    float expBuff[] = {1., 2., 3., 4., 1., 2., 3., 4., 5., 6., 7., 8., 5., 6., 7., 8., 9.,10., 11.,12., 9.,10., 11.,12.};

    NDArray<float> x(xBuff, 'c', {3,2,2});
    NDArray<float> exp(expBuff, 'c', {3,4,2});    

    NDArray<float> result = x.tile({2,1});

    ASSERT_TRUE(result.isSameShapeStrict(&exp));
    ASSERT_TRUE(result.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestTile6)
{
    double expBuff[] = {10.,11., 10.,11., 10.,11., 10.,11., 12.,13., 12.,13., 12.,13., 12.,13., 14.,15., 14.,15., 14.,15., 14.,15.};

    NDArray<double> x('c', {3, 1, 2});    
    NDArray<double> expected(expBuff, 'c', {3, 4, 2});

    NDArrayFactory<double>::linspace(10, x);    

    NDArray<double> result = x.tile({1,4,1});
//    result.printBuffer();

    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));
}


//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestMmulHelper1) {
    auto xBuffer = new float[3]{1.f, 2.f, 3.f};
    auto xShape = new Nd4jLong[8] {2, 1, 3, 1, 1, 0, 1, 99};
    auto x = new NDArray<float>(xBuffer, xShape);

    auto yBuffer = new float[3]{2.f, 4.f, 6.f};
    auto yShape = new Nd4jLong[8] {2, 1, 3, 1, 1, 0, 1, 99};
    auto y = new NDArray<float>(yBuffer, yShape);

    auto z = NDArrayFactory<float>::mmulHelper(x, y);

    ASSERT_EQ(1, z->lengthOf());
    ASSERT_NEAR(28, z->getScalar(0), 1e-5);

    delete z;
    delete[] xBuffer;
    delete[] xShape;
    delete[] yBuffer;
    delete[] yShape;
    delete y;
    delete x;
}


TEST_F(NDArrayTest, TestPermuteReshapeMmul1) {
    NDArray<float> x('c', {6, 3});
    NDArray<float> y('c', {3, 6});

    Nd4jLong _expS[] = {2, 3, 3, 1, 3, 0, 1, 102};
    float _expB[] = {231.0, 252.0, 273.0, 537.0, 594.0, 651.0, 843.0, 936.0, 1029.0};
    NDArray<float> exp(_expB, _expS);
    exp.triggerAllocationFlag(false, false);

    for (int e = 0; e < x.lengthOf(); e++)
        x.putScalar(e, e+1);

    for (int e = 0; e < y.lengthOf(); e++)
        y.putScalar(e, e+1);

    x.permutei({1, 0});
    y.permutei({1, 0});

    auto z = NDArrayFactory<float>::mmulHelper(&x, &y);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete z;
}

TEST_F(NDArrayTest, TestPermuteReshapeMmul2) {
    NDArray<float> x('c', {6, 3});
    NDArray<float> y('c', {3, 6});

    Nd4jLong _expS[] = {2, 3, 3, 1, 3, 0, 1, 102};
    float _expB[] = {231.0, 252.0, 273.0, 537.0, 594.0, 651.0, 843.0, 936.0, 1029.0};
    NDArray<float> exp(_expB, _expS);
    exp.triggerAllocationFlag(false, false);

    for (int e = 0; e < x.lengthOf(); e++)
        x.putScalar(e, e+1);

    for (int e = 0; e < y.lengthOf(); e++)
        y.putScalar(e, e+1);

    auto x_ = x.dup('f');
    auto y_ = y.dup('f');

    x_->permutei({1, 0});
    y_->permutei({1, 0});

    auto z = NDArrayFactory<float>::mmulHelper(x_, y_);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete z;
    delete x_;
    delete y_;
}


TEST_F(NDArrayTest, TestPermuteReshapeMmul3) {
    NDArray<float> x('c', {2, 2, 2, 3, 2, 2});
    NDArray<float> y('c', {2, 3, 2 ,2});

    Nd4jLong _expS[] = {2, 8, 2, 1, 8, 0, 1, 102};
    float _expB[] = {1624.0, 1858.0, 2092.0, 2326.0, 5368.0, 5602.0, 5836.0, 6070.0, 4504.0, 5170.0, 5836.0, 6502.0, 15160.0, 15826.0, 16492.0, 17158.0};
    NDArray<float> exp(_expB, _expS);
    exp.triggerAllocationFlag(false, false);

    for (int e = 0; e < x.lengthOf(); e++)
        x.putScalar(e, e+1);

    for (int e = 0; e < y.lengthOf(); e++)
        y.putScalar(e, e+1);

    x.permutei({0, 3, 4, 5, 1, 2});
    y.permutei({3, 2, 1, 0});

    x.reshapei('c', {2 * 2 * 2, 3 * 2 * 2});
    y.reshapei('c', {2 * 2 * 3, 2});

    auto z = NDArrayFactory<float>::mmulHelper(&x, &y);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete z;
}

TEST_F(NDArrayTest, TestPermuteReshapeMmul4) {
    NDArray<float> x('c', {2, 2, 2, 3, 2, 2});
    NDArray<float> y('c', {2, 3, 2 ,2});

    Nd4jLong _expS[] = {2, 8, 2, 1, 8, 0, 1, 102};
    float _expB[] = {1624.0, 1858.0, 2092.0, 2326.0, 5368.0, 5602.0, 5836.0, 6070.0, 4504.0, 5170.0, 5836.0, 6502.0, 15160.0, 15826.0, 16492.0, 17158.0};
    NDArray<float> exp(_expB, _expS);
    exp.triggerAllocationFlag(false, false);

    for (int e = 0; e < x.lengthOf(); e++)
        x.putScalar(e, e+1);

    for (int e = 0; e < y.lengthOf(); e++)
        y.putScalar(e, e+1);

    auto y_ = y.dup('f');

    x.permutei({0, 3, 4, 5, 1, 2});
    y_->permutei({3, 2, 1, 0});

    x.reshapei('c', {2 * 2 * 2, 3 * 2 * 2});
    y_->reshapei('c', {2 * 2 * 3, 2});

    auto z = NDArrayFactory<float>::mmulHelper(&x, y_);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete z;
    delete y_;
}


//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestMmulHelper2) {
    auto xBuffer = new float[15]{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f};
    auto xShape = new Nd4jLong[8] {2, 5, 3, 3, 1, 0, 1, 99};
    auto x = new NDArray<float>(xBuffer, xShape);
    x->triggerAllocationFlag(true, true);

    auto yBuffer = new float[3]{2.f, 4.f, 6.f};
    auto yShape = new Nd4jLong[8] {2, 3, 1, 1, 1, 0, 1, 99};
    auto y = new NDArray<float>(yBuffer, yShape);
    y->triggerAllocationFlag(true, true);

    auto z = new NDArray<float>('f', {5, 1});

    auto expBuffer = new float[5]{28.00,  64.00,  100.00,  136.00,  172.00};
    auto exp = new NDArray<float>(expBuffer, z->getShapeInfo());
    exp->triggerAllocationFlag(true, false);

    //nd4j::blas::GEMV<float>::op('f',  x->rows(), x->columns(), 1.0f, x->getBuffer(), y->rows(), y->getBuffer(), 1, 0.0, z->getBuffer(), 1);

    NDArrayFactory<float>::mmulHelper(x, y, z);

    //z->printBuffer();

    ASSERT_TRUE(z->equalsTo(exp));

    delete x;
    delete y;
    delete z;
    delete exp;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestMmulHelper3) {
    auto xBuffer = new float[15]{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f};
    auto xShape = new Nd4jLong[8] {2, 5, 3, 1, 5, 0, 1, 102};
    auto x = new NDArray<float>(xBuffer, xShape);

    auto yBuffer = new float[3]{2.f, 4.f, 6.f};
    auto yShape = new Nd4jLong[8] {2, 3, 1, 1, 1, 0, 1, 99};
    auto y = new NDArray<float>(yBuffer, yShape);

    auto z = new NDArray<float>('f', {5, 1});

    auto expBuffer = new float[5]{92.00,  104.00,  116.00,  128.00,  140.00};
    auto exp = new NDArray<float>(expBuffer, z->getShapeInfo());

    //nd4j::blas::GEMV<float>::op('f',  x->rows(), x->columns(), 1.0f, x->getBuffer(), y->rows(), y->getBuffer(), 1, 0.0, z->getBuffer(), 1);

    NDArrayFactory<float>::mmulHelper(x, y, z);

    //z->printBuffer();

    ASSERT_TRUE(z->equalsTo(exp));

    delete[] expBuffer;
    delete[] xBuffer;
    delete[] yBuffer;
    delete[] xShape;
    delete[] yShape;

    delete x;
    delete y;
    delete z;
    delete exp;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestMmulHelper4) {
    auto xBuffer = new float[6]{1, 2, 3, 4, 5, 6};
    auto xShape = new Nd4jLong[8] {2, 3, 2, 2, 1, 0, 1, 99};
    auto x = new NDArray<float>(xBuffer, xShape);

    auto yBuffer = new float[6]{7, 8, 9, 0, 1, 2};
    auto yShape = new Nd4jLong[8] {2, 2, 3, 3, 1, 0, 1, 99};
    auto y = new NDArray<float>(yBuffer, yShape);

    auto z = new NDArray<float>('f', {3, 3});

    auto expBuffer = new float[9]{7.0, 21.0, 35.0, 10.0, 28.0, 46.0, 13.0, 35.0, 57.0};
    auto exp = new NDArray<float>(expBuffer, z->getShapeInfo());

    NDArrayFactory<float>::mmulHelper(x, y, z);
    ASSERT_TRUE(z->equalsTo(exp));

    delete[] expBuffer;
    delete[] xBuffer;
    delete[] yBuffer;
    delete[] xShape;
    delete[] yShape;

    delete x;
    delete y;
    delete z;
    delete exp;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestMmulHelper5) {
    auto xBuffer = new float[6]{1, 2, 3, 4, 5, 6};
    auto xShape = new Nd4jLong[8] {2, 3, 2, 1, 3, 0, 1, 102};
    auto x = new NDArray<float>(xBuffer, xShape);

    auto yBuffer = new float[6]{7, 8, 9, 0, 1, 2};
    auto yShape = new Nd4jLong[8] {2, 2, 3, 3, 1, 0, 1, 99};
    auto y = new NDArray<float>(yBuffer, yShape);

    auto z = new NDArray<float>('f', {3, 3});

    auto expBuffer = new float[9]{7.0, 14.0, 21.0, 12.0, 21.0, 30.0, 17.0, 28.0, 39.0};
    auto exp = new NDArray<float>(expBuffer, z->getShapeInfo());

    NDArrayFactory<float>::mmulHelper(x, y, z);
    ASSERT_TRUE(z->equalsTo(exp));

    delete[] expBuffer;
    delete[] xBuffer;
    delete[] yBuffer;
    delete[] xShape;
    delete[] yShape;

    delete x;
    delete y;
    delete z;
    delete exp;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestMmulHelper6) {
    auto xBuffer = new float[6]{1, 2, 3, 4, 5, 6};
    auto xShape = new Nd4jLong[8] {2, 3, 2, 1, 3, 0, 1, 102};
    auto x = new NDArray<float>(xBuffer, xShape);

    auto yBuffer = new float[6]{7, 8, 9, 0, 1, 2};
    auto yShape = new Nd4jLong[8] {2, 2, 3, 1, 2, 0, 1, 102};
    auto y = new NDArray<float>(yBuffer, yShape);

    auto z = new NDArray<float>('f', {3, 3});

    auto expBuffer = new float[9]{39.0, 54.0, 69.0, 9.0, 18.0, 27.0, 9.0, 12.0, 15.0};
    auto exp = new NDArray<float>(expBuffer, z->getShapeInfo());

    NDArrayFactory<float>::mmulHelper(x, y, z);
    ASSERT_TRUE(z->equalsTo(exp));


    delete[] expBuffer;
    delete[] xBuffer;
    delete[] yBuffer;
    delete[] xShape;
    delete[] yShape;

    delete x;
    delete y;
    delete z;
    delete exp;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestMmulHelper7) {
    auto xBuffer = new float[15]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    auto xShape = new Nd4jLong[8] {2, 5, 3, 1, 5, 0, 1, 102};
    auto x = new NDArray<float>(xBuffer, xShape);

    auto yBuffer = new float[5]{2, 4, 6, 8, 10};
    auto yShape = new Nd4jLong[8] {2, 1, 5, 1, 1, 0, 1, 99};
    auto y = new NDArray<float>(yBuffer, yShape);

    auto z = new NDArray<float>('f', {1, 3});

    auto expBuffer = new float[9]{110.00,  260.00,  410.00};
    auto exp = new NDArray<float>(expBuffer, z->getShapeInfo());

    NDArrayFactory<float>::mmulHelper(y, x, z);

    //z->printBuffer();
    ASSERT_TRUE(z->equalsTo(exp));

    delete[] expBuffer;
    delete[] xBuffer;
    delete[] yBuffer;
    delete[] xShape;
    delete[] yShape;

    delete x;
    delete y;
    delete z;
    delete exp;
}


TEST_F(NDArrayTest, TestMmulHelper_ND_1) {
    Nd4jLong _expS[] = {3, 2, 3, 3, 9, 3, 1, 0, 1, 99};
    float _expB[] = {70.f, 80.f, 90.f, 158.f, 184.f, 210.f, 246.f, 288.f, 330.f, 1030.f, 1088.f, 1146.f, 1310.f, 1384.f, 1458.f, 1590.f, 1680.f, 1770.f};

    NDArray<float> a('c', {2, 3, 4});
    for (int e = 0; e < a.lengthOf(); e++)
        a.putScalar(e, e+1);

    NDArray<float> b('c', {2, 4, 3});
    for (int e = 0; e < b.lengthOf(); e++)
        b.putScalar(e, e+1);

    NDArray<float> exp(_expB, _expS);
    exp.triggerAllocationFlag(false, false);

    auto c = NDArrayFactory<float>::mmulHelper(&a, &b);

    ASSERT_TRUE(exp.isSameShapeStrict(c));
    //c->printShapeInfo("Result shape");
    //c->printBuffer("Result buffer");

    ASSERT_TRUE(exp.equalsTo(c));

    delete c;
}


TEST_F(NDArrayTest, TestMmulHelper_ND_2) {
    Nd4jLong _expS[] = {3, 2, 72, 2, 144, 2, 1, 0, 1, 99};
    float _expB[] = {1.07250000e+04,   1.10500000e+04,   2.63500000e+04,   2.73000000e+04,  4.19750000e+04,   4.35500000e+04,   5.76000000e+04,   5.98000000e+04,    7.32250000e+04,   7.60500000e+04,   8.88500000e+04,   9.23000000e+04,    1.04475000e+05,   1.08550000e+05,   1.20100000e+05,   1.24800000e+05,    1.35725000e+05,   1.41050000e+05,   1.51350000e+05,   1.57300000e+05,    1.66975000e+05,   1.73550000e+05,   1.82600000e+05,   1.89800000e+05,    1.98225000e+05,   2.06050000e+05,   2.13850000e+05,   2.22300000e+05,    2.29475000e+05,   2.38550000e+05,   2.45100000e+05,   2.54800000e+05,    2.60725000e+05,   2.71050000e+05,   2.76350000e+05,   2.87300000e+05,    2.91975000e+05,   3.03550000e+05,   3.07600000e+05,   3.19800000e+05,    3.23225000e+05,   3.36050000e+05,   3.38850000e+05,   3.52300000e+05,    3.54475000e+05,   3.68550000e+05,   3.70100000e+05,   3.84800000e+05,    3.85725000e+05,   4.01050000e+05,   4.01350000e+05,   4.17300000e+05,    4.16975000e+05,   4.33550000e+05,   4.32600000e+05,   4.49800000e+05,    4.48225000e+05,   4.66050000e+05,   4.63850000e+05,   4.82300000e+05,    4.79475000e+05,   4.98550000e+05,   4.95100000e+05,   5.14800000e+05,    5.10725000e+05,   5.31050000e+05,   5.26350000e+05,   5.47300000e+05,    5.41975000e+05,   5.63550000e+05,   5.57600000e+05,   5.79800000e+05,    5.73225000e+05,   5.96050000e+05,   5.88850000e+05,   6.12300000e+05,    6.04475000e+05,   6.28550000e+05,   6.20100000e+05,   6.44800000e+05,    6.35725000e+05,   6.61050000e+05,   6.51350000e+05,   6.77300000e+05,    6.66975000e+05,   6.93550000e+05,   6.82600000e+05,   7.09800000e+05,    6.98225000e+05,   7.26050000e+05,   7.13850000e+05,   7.42300000e+05,    7.29475000e+05,   7.58550000e+05,   7.45100000e+05,   7.74800000e+05,    7.60725000e+05,   7.91050000e+05,   7.76350000e+05,   8.07300000e+05,    7.91975000e+05,   8.23550000e+05,   8.07600000e+05,   8.39800000e+05,    8.23225000e+05,   8.56050000e+05,   8.38850000e+05,   8.72300000e+05,    8.54475000e+05,   8.88550000e+05,   8.70100000e+05,   9.04800000e+05,   8.85725000e+05,   9.21050000e+05,   9.01350000e+05,   9.37300000e+05, 9.16975000e+05,   9.53550000e+05,   9.32600000e+05,   9.69800000e+05, 9.48225000e+05,   9.86050000e+05,   9.63850000e+05,   1.00230000e+06, 9.79475000e+05,   1.01855000e+06,   9.95100000e+05,   1.03480000e+06, 1.01072500e+06,   1.05105000e+06,   1.02635000e+06,   1.06730000e+06, 1.04197500e+06,   1.08355000e+06,   1.05760000e+06,   1.09980000e+06, 1.07322500e+06,   1.11605000e+06,   1.08885000e+06,   1.13230000e+06, 1.10447500e+06,   1.14855000e+06,   1.12010000e+06,   1.16480000e+06, 1.13572500e+06,   1.18105000e+06,   1.15135000e+06,   1.19730000e+06, 1.16697500e+06,   1.21355000e+06,   3.54260000e+06,   3.58980000e+06, 3.58947500e+06,   3.63730000e+06,   3.63635000e+06,   3.68480000e+06, 3.68322500e+06,   3.73230000e+06,   3.73010000e+06,   3.77980000e+06,   3.77697500e+06,   3.82730000e+06,   3.82385000e+06,   3.87480000e+06, 3.87072500e+06,   3.92230000e+06,   3.91760000e+06,   3.96980000e+06,  3.96447500e+06,   4.01730000e+06,   4.01135000e+06,   4.06480000e+06, 4.05822500e+06,   4.11230000e+06,   4.10510000e+06,   4.15980000e+06,  4.15197500e+06,   4.20730000e+06,   4.19885000e+06,   4.25480000e+06, 4.24572500e+06,   4.30230000e+06,   4.29260000e+06,   4.34980000e+06,  4.33947500e+06,   4.39730000e+06,   4.38635000e+06,   4.44480000e+06, 4.43322500e+06,   4.49230000e+06,   4.48010000e+06,   4.53980000e+06, 4.52697500e+06,   4.58730000e+06,   4.57385000e+06,   4.63480000e+06,  4.62072500e+06,   4.68230000e+06,   4.66760000e+06,   4.72980000e+06,  4.71447500e+06,   4.77730000e+06,   4.76135000e+06,   4.82480000e+06,  4.80822500e+06,   4.87230000e+06,   4.85510000e+06,   4.91980000e+06,  4.90197500e+06,   4.96730000e+06,   4.94885000e+06,   5.01480000e+06,   4.99572500e+06,   5.06230000e+06,   5.04260000e+06,   5.10980000e+06,   5.08947500e+06,   5.15730000e+06,   5.13635000e+06,   5.20480000e+06,   5.18322500e+06,   5.25230000e+06,   5.23010000e+06,   5.29980000e+06,  5.27697500e+06,   5.34730000e+06,   5.32385000e+06,   5.39480000e+06,       5.37072500e+06,   5.44230000e+06,   5.41760000e+06,   5.48980000e+06,       5.46447500e+06,   5.53730000e+06,   5.51135000e+06,   5.58480000e+06,       5.55822500e+06,   5.63230000e+06,   5.60510000e+06,   5.67980000e+06,       5.65197500e+06,   5.72730000e+06,   5.69885000e+06,   5.77480000e+06,       5.74572500e+06,   5.82230000e+06,   5.79260000e+06,   5.86980000e+06,       5.83947500e+06,   5.91730000e+06,   5.88635000e+06,   5.96480000e+06,       5.93322500e+06,   6.01230000e+06,   5.98010000e+06,   6.05980000e+06,       6.02697500e+06,   6.10730000e+06,   6.07385000e+06,   6.15480000e+06,       6.12072500e+06,   6.20230000e+06,   6.16760000e+06,   6.24980000e+06,       6.21447500e+06,   6.29730000e+06,   6.26135000e+06,   6.34480000e+06,       6.30822500e+06,   6.39230000e+06,   6.35510000e+06,   6.43980000e+06,       6.40197500e+06,   6.48730000e+06,   6.44885000e+06,   6.53480000e+06,       6.49572500e+06,   6.58230000e+06,   6.54260000e+06,   6.62980000e+06,       6.58947500e+06,   6.67730000e+06,   6.63635000e+06,   6.72480000e+06,       6.68322500e+06,   6.77230000e+06,   6.73010000e+06,   6.81980000e+06,       6.77697500e+06,   6.86730000e+06,   6.82385000e+06,   6.91480000e+06,       6.87072500e+06,   6.96230000e+06,   6.91760000e+06,   7.00980000e+06,       6.96447500e+06,   7.05730000e+06,   7.01135000e+06,   7.10480000e+06,       1.17619750e+07,   1.18560500e+07,   1.18401000e+07,   1.19348000e+07,       1.19182250e+07,   1.20135500e+07,   1.19963500e+07,   1.20923000e+07,       1.20744750e+07,   1.21710500e+07,   1.21526000e+07,   1.22498000e+07,       1.22307250e+07,   1.23285500e+07,   1.23088500e+07,   1.24073000e+07,       1.23869750e+07,   1.24860500e+07,   1.24651000e+07,   1.25648000e+07,       1.25432250e+07,   1.26435500e+07,   1.26213500e+07,   1.27223000e+07,       1.26994750e+07,   1.28010500e+07,   1.27776000e+07,   1.28798000e+07,       1.28557250e+07,   1.29585500e+07,   1.29338500e+07,   1.30373000e+07,       1.30119750e+07,   1.31160500e+07,   1.30901000e+07,   1.31948000e+07,       1.31682250e+07,   1.32735500e+07,   1.32463500e+07,   1.33523000e+07,       1.33244750e+07,   1.34310500e+07,   1.34026000e+07,   1.35098000e+07,       1.34807250e+07,   1.35885500e+07,   1.35588500e+07,   1.36673000e+07,       1.36369750e+07,   1.37460500e+07,   1.37151000e+07,   1.38248000e+07,       1.37932250e+07,   1.39035500e+07,   1.38713500e+07,   1.39823000e+07,       1.39494750e+07,   1.40610500e+07,   1.40276000e+07,   1.41398000e+07,       1.41057250e+07,   1.42185500e+07,   1.41838500e+07,   1.42973000e+07,       1.42619750e+07,   1.43760500e+07,   1.43401000e+07,   1.44548000e+07,       1.44182250e+07,   1.45335500e+07,   1.44963500e+07,   1.46123000e+07,       1.45744750e+07,   1.46910500e+07,   1.46526000e+07,   1.47698000e+07,       1.47307250e+07,   1.48485500e+07,   1.48088500e+07,   1.49273000e+07,       1.48869750e+07,   1.50060500e+07,   1.49651000e+07,   1.50848000e+07,       1.50432250e+07,   1.51635500e+07,   1.51213500e+07,   1.52423000e+07,       1.51994750e+07,   1.53210500e+07,   1.52776000e+07,   1.53998000e+07,       1.53557250e+07,   1.54785500e+07,   1.54338500e+07,   1.55573000e+07,       1.55119750e+07,   1.56360500e+07,   1.55901000e+07,   1.57148000e+07,       1.56682250e+07,   1.57935500e+07,   1.57463500e+07,   1.58723000e+07,       1.58244750e+07,   1.59510500e+07,   1.59026000e+07,   1.60298000e+07,       1.59807250e+07,   1.61085500e+07,   1.60588500e+07,   1.61873000e+07,       1.61369750e+07,   1.62660500e+07,   1.62151000e+07,   1.63448000e+07,       1.62932250e+07,   1.64235500e+07,   1.63713500e+07,   1.65023000e+07,       1.64494750e+07,   1.65810500e+07,   1.65276000e+07,   1.66598000e+07,       1.66057250e+07,   1.67385500e+07,   1.66838500e+07,   1.68173000e+07,       1.67619750e+07,   1.68960500e+07,   1.68401000e+07,   1.69748000e+07,       1.69182250e+07,   1.70535500e+07,   1.69963500e+07,   1.71323000e+07,       1.70744750e+07,   1.72110500e+07,   1.71526000e+07,   1.72898000e+07,       1.72307250e+07,   1.73685500e+07,   1.73088500e+07,   1.74473000e+07,       1.73869750e+07,   1.75260500e+07,   1.74651000e+07,   1.76048000e+07,       1.75432250e+07,   1.76835500e+07,   2.46688500e+07,   2.48098000e+07,       2.47782250e+07,   2.49198000e+07,   2.48876000e+07,   2.50298000e+07,       2.49969750e+07,   2.51398000e+07,   2.51063500e+07,   2.52498000e+07,       2.52157250e+07,   2.53598000e+07,   2.53251000e+07,   2.54698000e+07,       2.54344750e+07,   2.55798000e+07,   2.55438500e+07,   2.56898000e+07,       2.56532250e+07,   2.57998000e+07,   2.57626000e+07,   2.59098000e+07,       2.58719750e+07,   2.60198000e+07,   2.59813500e+07,   2.61298000e+07,       2.60907250e+07,   2.62398000e+07,   2.62001000e+07,   2.63498000e+07,       2.63094750e+07,   2.64598000e+07,   2.64188500e+07,   2.65698000e+07,       2.65282250e+07,   2.66798000e+07,   2.66376000e+07,   2.67898000e+07,       2.67469750e+07,   2.68998000e+07,   2.68563500e+07,   2.70098000e+07,       2.69657250e+07,   2.71198000e+07,   2.70751000e+07,   2.72298000e+07,       2.71844750e+07,   2.73398000e+07,   2.72938500e+07,   2.74498000e+07,       2.74032250e+07,   2.75598000e+07,   2.75126000e+07,   2.76698000e+07,       2.76219750e+07,   2.77798000e+07,   2.77313500e+07,   2.78898000e+07,       2.78407250e+07,   2.79998000e+07,   2.79501000e+07,   2.81098000e+07,       2.80594750e+07,   2.82198000e+07,   2.81688500e+07,   2.83298000e+07,       2.82782250e+07,   2.84398000e+07,   2.83876000e+07,   2.85498000e+07,       2.84969750e+07,   2.86598000e+07,   2.86063500e+07,   2.87698000e+07,       2.87157250e+07,   2.88798000e+07,   2.88251000e+07,   2.89898000e+07,       2.89344750e+07,   2.90998000e+07,   2.90438500e+07,   2.92098000e+07,       2.91532250e+07,   2.93198000e+07,   2.92626000e+07,   2.94298000e+07,       2.93719750e+07,   2.95398000e+07,   2.94813500e+07,   2.96498000e+07,       2.95907250e+07,   2.97598000e+07,   2.97001000e+07,   2.98698000e+07,       2.98094750e+07,   2.99798000e+07,   2.99188500e+07,   3.00898000e+07,    3.00282250e+07,   3.01998000e+07,   3.01376000e+07,   3.03098000e+07,    3.02469750e+07,   3.04198000e+07,   3.03563500e+07,   3.05298000e+07,  3.04657250e+07,   3.06398000e+07,   3.05751000e+07,   3.07498000e+07, 3.06844750e+07,   3.08598000e+07,   3.07938500e+07,   3.09698000e+07,    3.09032250e+07,   3.10798000e+07,   3.10126000e+07,   3.11898000e+07,    3.11219750e+07,   3.12998000e+07,   3.12313500e+07,   3.14098000e+07,    3.13407250e+07,   3.15198000e+07,   3.14501000e+07,   3.16298000e+07,    3.15594750e+07,   3.17398000e+07,   3.16688500e+07,   3.18498000e+07,    3.17782250e+07,   3.19598000e+07,   3.18876000e+07,   3.20698000e+07,    3.19969750e+07,   3.21798000e+07,   3.21063500e+07,   3.22898000e+07,    3.22157250e+07,   3.23998000e+07,   3.23251000e+07,   3.25098000e+07,    3.24344750e+07,   3.26198000e+07,   3.25438500e+07,   3.27298000e+07, 3.26532250e+07,   3.28398000e+07,   3.27626000e+07,   3.29498000e+07};

    NDArray<float> a('c', {2, 72, 25});
    for (int e = 0; e < a.lengthOf(); e++)
        a.putScalar(e, e+1);

    NDArray<float> b('c', {2, 25, 2});
    for (int e = 0; e < b.lengthOf(); e++)
        b.putScalar(e, e+1);

    NDArray<float> exp(_expB, _expS);
    exp.triggerAllocationFlag(false, false);

    auto c = NDArrayFactory<float>::mmulHelper(&a, &b);

    ASSERT_TRUE(exp.isSameShapeStrict(c));
    //c->printShapeInfo("Result shape");
    //c->printBuffer("Result buffer");
    //exp.printBuffer("Expctd buffer");
    ASSERT_TRUE(exp.equalsTo(c, 1e1));

    delete c;
}


TEST_F(NDArrayTest, TestNegSize1) {
    NDArray<float> array('c', {2, 5, 7});

    ASSERT_EQ(7, array.sizeAt(-1));
    ASSERT_EQ(5, array.sizeAt(-2));
    ASSERT_EQ(2, array.sizeAt(-3));
}

//////////////////////////////////////////////////////////////////////
// not-in-place
TEST_F(NDArrayTest, Permute1) {  
    
    const Nd4jLong shape1[] = {3, 5, 10, 15, 150, 15, 1, 0, 1, 99};
	const Nd4jLong shape2[] = {3, 15, 5, 10, 1, 150, 15, 0, -1, 99};
    const std::initializer_list<int> perm = {2, 0, 1};    
    
    NDArray<float> arr1(shape1,true);
    NDArray<float> arr2(shape2,true);    

	NDArray<float>* result = arr1.permute(perm);        	
	ASSERT_TRUE(result->isSameShapeStrict(&arr2));

	delete result;
}

//////////////////////////////////////////////////////////////////////
// in-place
TEST_F(NDArrayTest, Permute2) {
    
    const Nd4jLong shape1[] = {3, 5, 10, 15, 150, 15, 1, 0, 1, 99};
	const Nd4jLong shape2[] = {3, 15, 5, 10, 1, 150, 15, 0, -1, 99};
    const std::initializer_list<int> perm = {2, 0, 1};    
    
    NDArray<float> arr1(shape1,true);
    NDArray<float> arr2(shape2,true);    

	ASSERT_TRUE(arr1.permutei(perm));
	ASSERT_TRUE(arr1.isSameShapeStrict(&arr2));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Broadcast1) {
    
    const Nd4jLong shape1[10] = {3, 5, 1, 10, 10, 10, 1, 0, 1, 99};
	const Nd4jLong shape2[8]  = {2,    7, 10, 10, 1, 0, 1, 99};
	const Nd4jLong shape3[10] = {3, 5, 7, 10, 70, 10, 1, 0, 1, 99};    
    
	NDArray<float> arr1(shape1);
    NDArray<float> arr2(shape2);    
	NDArray<float> arr3(shape3);    

	NDArray<float>* result = arr1.broadcast(arr2);		
	ASSERT_TRUE(result->isSameShapeStrict(&arr3));
	delete result;
}


TEST_F(NDArrayTest, RSubScalarTest1) {
    NDArray<double> array('c', {1, 4});
    array.assign(2.0);

    NDArray<double> result('c', {1, 4});

    array.applyScalar<simdOps::ReverseSubtract<double>>(1.0, &result);

    ASSERT_NEAR(-1.0, result.meanNumber(), 1e-5);
}

TEST_F(NDArrayTest, BroadcastOpsTest1) {

    NDArray<float> x('c', {5, 5});
    auto row = nd4j::NDArrayFactory<float>::linspace(1.0f, 5.0f, 5);
    float *brow = new float[5]{1,2,3,4,5};
    auto bshape = new Nd4jLong[8]{2, 1, 5, 1, 1, 0, 1, 99};
    float *ebuf = new float[25] {1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5};
    auto eshape = new Nd4jLong[8] {2, 5, 5, 5, 1, 0, 1, 99};
    NDArray<float> expRow(brow, bshape);
    NDArray<float> exp(ebuf, eshape);

    ASSERT_TRUE(row->equalsTo(&expRow));


    x.applyBroadcast<simdOps::Add<float>>({1}, row);

    //x.printBuffer("Result");

    ASSERT_TRUE(x.equalsTo(&exp));

    delete[] brow;
    delete[] bshape;
    delete[] ebuf;
    delete[] eshape;
    delete row;
}

TEST_F(NDArrayTest, TestIndexedPut2) {
    NDArray<float> x('f', {2, 2});
    //x.printShapeInfo("x shape");
    x.putIndexedScalar(1, 1.0f);

    //x.printBuffer("after");
    ASSERT_NEAR(x.getBuffer()[2], 1.0, 1e-5);
}

TEST_F(NDArrayTest, TestIndexedPut3) {
    NDArray<float> x('c', {2, 2});
    x.putIndexedScalar(1, 1.0f);

    //x.printBuffer("after");
    ASSERT_NEAR(x.getBuffer()[1], 1.0, 1e-5);
}

TEST_F(NDArrayTest, TestIndexedPut4) {
    NDArray<float> x('f', {2, 2});
    x.putScalar(0, 1, 1.0f);

    //x.printBuffer("after");
    ASSERT_NEAR(x.getBuffer()[2], 1.0, 1e-5);
}


TEST_F(NDArrayTest, TestIndexedPut5) {
    NDArray<float> x('c', {2, 2});
    x.putScalar(0, 1, 1.0f);

    //x.printBuffer("after");
    ASSERT_NEAR(x.getBuffer()[1], 1.0, 1e-5);
}

TEST_F(NDArrayTest, TestAllTensors1) {
    NDArray<float> matrix('c', {3, 5});

    std::unique_ptr<ResultSet<float>> rows(NDArrayFactory<float>::allTensorsAlongDimension(&matrix, {1}));

    ASSERT_EQ(3, rows->size());
}


TEST_F(NDArrayTest, TestIndexing1) {
    NDArray<float> matrix('c', {5, 5});
    for (int e = 0; e < matrix.lengthOf(); e++)
        matrix.putScalar(e, (float) e);

    IndicesList idx({NDIndex::interval(2,4), NDIndex::all()});
    auto sub = matrix.subarray(idx);

    ASSERT_TRUE(sub != nullptr);

    ASSERT_EQ(2, sub->rows());
    ASSERT_EQ(5, sub->columns());

    ASSERT_NEAR(10, sub->getScalar(0), 1e-5);

    delete sub;
}


TEST_F(NDArrayTest, TestIndexing2) {
    NDArray<float> matrix('c', {2, 5, 4, 4});
    for (int e = 0; e < matrix.lengthOf(); e++)
        matrix.putScalar(e, (float) e);

    IndicesList idx({ NDIndex::all(), NDIndex::interval(2,4), NDIndex::all(),  NDIndex::all()});
    auto sub = matrix.subarray(idx);

    ASSERT_TRUE(sub != nullptr);

    ASSERT_EQ(2, sub->sizeAt(0));
    ASSERT_EQ(2, sub->sizeAt(1));
    ASSERT_EQ(4, sub->sizeAt(2));
    ASSERT_EQ(4, sub->sizeAt(3));


    ASSERT_EQ(64, sub->lengthOf());
    ASSERT_NEAR(32, sub->getScalar(0), 1e-5);
    ASSERT_NEAR(112, sub->getIndexedScalar(32), 1e-5);

    delete sub;
}

TEST_F(NDArrayTest, TestIndexing3) {
    NDArray<float> matrix('c', {5, 5});
    for (int e = 0; e < matrix.lengthOf(); e++)
        matrix.putScalar(e, (float) e);

    std::initializer_list<std::vector<Nd4jLong>> idx = {{2,4}, {}};    
    auto sub = matrix(idx);

    ASSERT_EQ(2, sub.rows());
    ASSERT_EQ(5, sub.columns());

    ASSERT_NEAR(10, sub.getScalar(0), 1e-5);
}


TEST_F(NDArrayTest, TestIndexing4) {
    NDArray<float> matrix('c', {2, 5, 4, 4});
    for (int e = 0; e < matrix.lengthOf(); e++)
        matrix.putScalar(e, (float) e);

    std::initializer_list<std::vector<Nd4jLong>> idx = {{}, {2,4}, {}, {}};    
    auto sub = matrix(idx);    

    ASSERT_EQ(2, sub.sizeAt(0));
    ASSERT_EQ(2, sub.sizeAt(1));
    ASSERT_EQ(4, sub.sizeAt(2));
    ASSERT_EQ(4, sub.sizeAt(3));


    ASSERT_EQ(64, sub.lengthOf());
    ASSERT_NEAR(32, sub.getScalar(0), 1e-5);
    ASSERT_NEAR(112, sub.getIndexedScalar(32), 1e-5);
}

TEST_F(NDArrayTest, TestReshapeNegative1) {
    std::unique_ptr<NDArray<float>> array(new NDArray<float>('c', {2, 3, 4, 64}));

    array->reshapei('c', {-1, 64});

    ASSERT_EQ(24, array->sizeAt(0));
    ASSERT_EQ(64, array->sizeAt(1));
}

TEST_F(NDArrayTest, TestReshapeNegative2) {
    std::unique_ptr<NDArray<float>> array(new NDArray<float>('c', {2, 3, 4, 64}));

    std::unique_ptr<NDArray<float>> reshaped(array->reshape('c', {-1, 64}));

    ASSERT_EQ(24, reshaped->sizeAt(0));
    ASSERT_EQ(64, reshaped->sizeAt(1));
}

//////////////////////////////////////////////////////////////////////
// TEST_F(NDArrayTest, SVD1) {
    
//     double arrA[8]  = {1, 2, 3, 4, 5, 6, 7, 8};
// 	double arrU[8]  = {-0.822647, 0.152483, -0.421375, 0.349918, -0.020103, 0.547354, 0.381169, 0.744789};
// 	double arrS[2]  = {0.626828, 14.269095};
// 	double arrVt[4] = {0.767187,-0.641423, 0.641423, 0.767187};
		
// 	int shapeA[8]  = {2, 4, 2, 2, 1, 0, 1, 99};
// 	int shapeS[8]  = {2, 1, 2, 2, 1, 0, 1, 99};
// 	int shapeVt[8] = {2, 2, 2, 2, 1, 0, 1, 99};
   
// 	NDArray<double> a(arrA,   shapeA);
//     NDArray<double> u(arrU,   shapeA);    
// 	NDArray<double> s(arrS,   shapeS);    
// 	NDArray<double> vt(arrVt, shapeVt);    
// 	NDArray<double> expU, expS(shapeS), expVt(shapeVt);
	
// 	a.svd(expU, expS, expVt);
// 	ASSERT_TRUE(u.equalsTo(&expU));
// 	ASSERT_TRUE(s.equalsTo(&expS));
// 	ASSERT_TRUE(vt.equalsTo(&expVt));
	
// }

// //////////////////////////////////////////////////////////////////////
// TEST_F(NDArrayTest, SVD2) {
    
//     double arrA[6]  = {1, 2, 3, 4, 5, 6};
// 	double arrU[6]  = {-0.386318, -0.922366, 0.000000, -0.922366, 0.386318, 0.000000};
// 	double arrS[3]  = {9.508032, 0.77287, 0.000};
// 	double arrVt[9] = {-0.428667, -0.566307, -0.703947, 0.805964, 0.112382,  -0.581199, 0.408248, -0.816497, 0.408248};

// 	int shapeA[8]  = {2, 2, 3, 3, 1, 0, 1, 99};
// 	int shapeS[8]  = {2, 1, 3, 3, 1, 0, 1, 99};
// 	int shapeVt[8] = {2, 3, 3, 3, 1, 0, 1, 99};
    
// 	NDArray<double> a(arrA,   shapeA);
//     NDArray<double> u(arrU,   shapeA);    
// 	NDArray<double> s(arrS,   shapeS);    
// 	NDArray<double> vt(arrVt, shapeVt);    
// 	NDArray<double> expU, expS(shapeS), expVt(shapeVt);
	
// 	a.svd(expU, expS, expVt);
// 	ASSERT_TRUE(u.equalsTo	(&expU));
// 	ASSERT_TRUE(s.equalsTo(&expS));
// 	ASSERT_TRUE(vt.equalsTo(&expVt));
	
// }

// //////////////////////////////////////////////////////////////////////
// TEST_F(NDArrayTest, SVD3) {
   
//    double arrA[8]  = {1, 2, 3, 4, 5, 6, 7, 8};
// 	double arrU[8]  = {-0.822647, 0.152483, -0.421375, 0.349918, -0.020103, 0.547354, 0.381169, 0.744789};
// 	double arrS[2]  = {0.626828, 14.269095};
// 	double arrVt[4] = {0.767187,-0.641423, 0.641423, 0.767187};
		
// 	int shapeA[8]  = {2, 4, 2, 2, 1, 0, 1, 99};
// 	int shapeS[8]  = {2, 1, 2, 2, 1, 0, 1, 99};
// 	int shapeVt[8] = {2, 2, 2, 2, 1, 0, 1, 99};
  
// 	NDArray<double> a(arrA,   shapeA);
//    NDArray<double> u(arrU,   shapeA);    
// 	NDArray<double> s(arrS,   shapeS);    
// 	NDArray<double> vt(arrVt, shapeVt);    
// 	NDArray<double> expU, expS(shapeS), expVt(shapeVt);
	
// 	a.svd(expU, expS, expVt);
// 	ASSERT_TRUE(expU.hasOrthonormalBasis(1));
// 	ASSERT_TRUE(expVt.hasOrthonormalBasis(0));
// 	ASSERT_TRUE(expVt.hasOrthonormalBasis(1));
// 	ASSERT_TRUE(expVt.isUnitary());
// }

// //////////////////////////////////////////////////////////////////////
// TEST_F(NDArrayTest, SVD4) {
    
//     double arrA[6]  = {1, 2, 3, 4, 5, 6};
// 	double arrU[6]  = {-0.386318, -0.922366, 0.000000, -0.922366, 0.386318, 0.000000};
// 	double arrS[3]  = {9.508032, 0.77287, 0.000};
// 	double arrVt[9] = {-0.428667, -0.566307, -0.703947, 0.805964, 0.112382,  -0.581199, 0.408248, -0.816497, 0.408248};

// 	int shapeA[8]  = {2, 2, 3, 3, 1, 0, 1, 99};
// 	int shapeS[8]  = {2, 1, 3, 3, 1, 0, 1, 99};
// 	int shapeVt[8] = {2, 3, 3, 3, 1, 0, 1, 99};
    
// 	NDArray<double> a(arrA,   shapeA);
//     NDArray<double> u(arrU,   shapeA);    
// 	NDArray<double> s(arrS,   shapeS);    
// 	NDArray<double> vt(arrVt, shapeVt);    
// 	NDArray<double> expU, expS(shapeS), expVt(shapeVt);
	
// 	a.svd(expU, expS, expVt);
// 	ASSERT_TRUE(expU.hasOrthonormalBasis(1));
// 	ASSERT_TRUE(expVt.hasOrthonormalBasis(0));
// 	ASSERT_TRUE(expVt.hasOrthonormalBasis(1));
// 	ASSERT_TRUE(expVt.isUnitary());	
// }

////////////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestStdDev1) {
    NDArray<double> array('c', {1, 5});
    for (int e = 0; e < array.lengthOf(); e++)
        array.putScalar(e, e+1);

    auto std = array.varianceNumber<simdOps::SummaryStatsStandardDeviation<double>>(true);
    ASSERT_NEAR(std, 1.58109, 1e-4);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestStdDev2) {
    NDArray<double> array('c', {5, 6});
    auto tad = array.tensorAlongDimension(0, {0});

    ASSERT_EQ(5, tad->lengthOf());

    for (int e = 0; e < tad->lengthOf(); e++)
        tad->putIndexedScalar(e, e+1);

    ASSERT_NEAR(15, tad->sumNumber(), 1e-5);

    auto std = tad->varianceNumber<simdOps::SummaryStatsStandardDeviation<double>>(true);
    ASSERT_NEAR(std, 1.58109, 1e-4);

    delete tad;
}

TEST_F(NDArrayTest, TestStdDev3) {
    NDArray<float> array('c', {1, 5000000});
    for (int e = 0; e < array.lengthOf(); e++)
        array(e) = 1.f + (e%2?0.5f:-0.5f);

    auto std = array.varianceNumber<simdOps::SummaryStatsStandardDeviation<float>>(true);
    // nd4j_printf("Variance is %f\n", std);
    ASSERT_NEAR(std, 0.5f, 1.0e-5f);
}

TEST_F(NDArrayTest, TestStdDev4) {
    NDArray<float> array('c', {1, 2000000});
    float const ethalon = 1 / 3.f;
    float x = ethalon;
    int total = array.lengthOf();
    for (int e = 0; e < total; e++) {
        array(e) = 1.0f + (e % 2?ethalon:-ethalon);
        x *= (e % 2? 2.f: 0.5f);
    }
    x = 0.f;
    for (int e = 0; e < total; ++e) {
        x += array(e);
    }
    x /= array.lengthOf();
    float y = 0;
    double M2 = 0;
    for (int e = 0; e < total; ++e) {
    //    y += nd4j::math::nd4j_abs(array(e) - x);
        M2 += (array(e) - x) * (array(e) - x);
    }
    //y /= total;
    M2 /= total;
    
    y = M2;
    auto std = array.varianceNumber<simdOps::SummaryStatsStandardDeviation<float>>(false);
//    float bY = array.varianceNumber();
    float bY = 0.3333333f;
    // nd4j_printf("Variance is %f, res is %f, internal is %f\n, deviance is %f(%f)\n", std, x, bY, y, nd4j::math::nd4j_sqrt<double>(M2));
    ASSERT_NEAR(std, 0.3333333f, 1.0e-5f);
}

TEST_F(NDArrayTest, TestStdDev5) {
    NDArray<float> array('c', {1, 1000000}); //00000});
    NDArray<double> arrayD('c', {1, 1000000}); //00000});
    for (int e = 0; e < array.lengthOf(); e++) {
        array(e) = 1.f + (e%2?1/5.f:-1/5.f);
        arrayD(e) = 1.0 + (e%2?1/5.:-1/5.);
    }
    float stdF = array.varianceNumber<simdOps::SummaryStatsStandardDeviation<float>>(false);
    double stdD = arrayD.varianceNumber<simdOps::SummaryStatsStandardDeviation<double>>(false);
    // nd4j_printf("Variance is %f(%f)\n", stdF, stdD);
    ASSERT_NEAR(stdD, 0.2, 1.0e-8); // 1/5 = 0.2
    ASSERT_NEAR(stdF, 0.2f, 1.0e-5f); // 1/5 = 0.2
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestApplyIndexReduce1) {
    float xBuff[] = {1, 5, 2, 12, 9, 3, 10, 7, 4, 11, 6, 8};    
    Nd4jLong xShapeInfo[] = {3, 2, 3, 2, 6, 2, 1, 0, 1, 99};        
    float expBuff[] = {3,1}; 
    Nd4jLong expShapeInfo[] = {1, 2, 1, 0, 1, 99};
    std::vector<int> dim = {0,1};
    
    NDArray<float> x(xBuff, xShapeInfo);
    NDArray<float> exp(expBuff, expShapeInfo);
    
    NDArray<float>* result = x.applyIndexReduce<simdOps::IndexMax<float>>(dim);
    ASSERT_TRUE(exp.isSameShapeStrict(result));
    ASSERT_TRUE(exp.equalsTo(result));
    
    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, applyReduce3Dot) {
    float xBuff[] = {1, 2, 3, 4, 5, 6};    
    float yBuff[] = {2, 2, 2, 2, 2, 2};    
    Nd4jLong xShapeInfo[] = {2, 2, 3, 3, 1, 0, 1, 99};        
        
    NDArray<float> x(xBuff, xShapeInfo);
    NDArray<float> y(yBuff, xShapeInfo);
    
    NDArray<float>* result = x.applyReduce3<simdOps::Dot<float>>(&y);
    ASSERT_TRUE(result->lengthOf() == 1);
    ASSERT_NEAR(42, result->getScalar(0), 1e-5);
    
    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, applyAllReduce3EuclideanDistance) {
    float xBuff[] =   {1, 2, 3, 4, 5, 6};    
    float yBuff[] =   {2, 2, 2, 2, 2, 2};
    float expBuff[] = {1.414214, 1.414214, 5.385165, 5.385165};
    Nd4jLong expShapeInfo[] = {2, 2, 2, 2, 1, 0, 1, 99};        
    Nd4jLong xShapeInfo[] =   {2, 2, 3, 3, 1, 0, 1, 99};        
        
    NDArray<float> x(xBuff, xShapeInfo);
    NDArray<float> y(yBuff, xShapeInfo);
    NDArray<float> exp(expBuff, expShapeInfo);
    
    NDArray<float>* result = x.applyAllReduce3<simdOps::EuclideanDistance<float>>(&y,{1});
    ASSERT_TRUE(exp.isSameShapeStrict(result));
    ASSERT_TRUE(exp.equalsTo(result));
    
    delete result;
}


//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, applyReduce3EuclideanDistance) {
    float xBuff[] =   {1, 2, 3, 4, 5, 6};    
    float yBuff[] =   {2, 2, 2, 2, 2, 2};
    float expBuff[] = {1.414214, 1.414214, 5.385165, 5.385165};
    Nd4jLong expShapeInfo[] = {2, 2, 2, 2, 1, 0, 1, 99};        
    Nd4jLong xShapeInfo[] =   {2, 2, 3, 3, 1, 0, 1, 99};        
        
    NDArray<float> x(xBuff, xShapeInfo);
    NDArray<float> y(yBuff, xShapeInfo);
    NDArray<float> exp(expBuff, expShapeInfo);
    
    NDArray<float>* result = x.applyAllReduce3<simdOps::EuclideanDistance<float>>(&y,{1});

    ASSERT_TRUE(exp.isSameShapeStrict(result));
    ASSERT_TRUE(exp.equalsTo(result));
    
    delete result;
}



//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestVarianceAlongDimension1) {
    float xBuff[] =   {1, 2, 3, 4, 5, 6};        
    float expBuff[] = {0.666667, 0.666667};
    Nd4jLong xShapeInfo[] =   {2, 2, 3, 3, 1, 0, 1, 99};        
    Nd4jLong expShapeInfo[] = {1, 2, 1, 0, 1, 99};        
    
        
    NDArray<float> x(xBuff, xShapeInfo);    
    NDArray<float> exp(expBuff, expShapeInfo);
    
    NDArray<float>* result = x.varianceAlongDimension<simdOps::SummaryStatsVariance<float>>(false, {1});

    ASSERT_TRUE(exp.isSameShapeStrict(result));
    ASSERT_TRUE(exp.equalsTo(result));
    
    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestSubRowVector1) {    
    float xBuff[] = {6, 7, 8, 9};
    float yBuff[] = {1, 2};
    float expBuff[] =  {5, 5, 7, 7};
    Nd4jLong xShapeInfo[] = {2, 2, 2, 2, 1, 0, 1, 99};        
    Nd4jLong yShapeInfo[] = {2, 1, 2, 2, 1, 0, 1, 99};        
    
    NDArray<float> x(xBuff, xShapeInfo);
    NDArray<float> y(yBuff, yShapeInfo);
    NDArray<float> target(x);
    NDArray<float> exp(expBuff, xShapeInfo);

    x.subRowVector(&y,&target);    

    ASSERT_TRUE(exp.isSameShapeStrict(&target));
    ASSERT_TRUE(exp.equalsTo(&target));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestDivRowVector1) {    
    float xBuff[] = {6, 8, 10, 12};
    float yBuff[] = {2, 4};
    float expBuff[] =  {3, 2, 5, 3};
    Nd4jLong xShapeInfo[] = {2, 2, 2, 2, 1, 0, 1, 99};        
    Nd4jLong yShapeInfo[] = {2, 1, 2, 2, 1, 0, 1, 99};        
    
    NDArray<float> x(xBuff, xShapeInfo);
    NDArray<float> y(yBuff, yShapeInfo);
    NDArray<float> target(x);
    NDArray<float> exp(expBuff, xShapeInfo);

    x.divRowVector(&y,&target);    

    ASSERT_TRUE(exp.isSameShapeStrict(&target));
    ASSERT_TRUE(exp.equalsTo(&target));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestMulRowVector1) {    
    float xBuff[] = {6, 8, 10, 12};
    float yBuff[] = {2, 4};
    float expBuff[] =  {12, 32, 20, 48};
    Nd4jLong xShapeInfo[] = {2, 2, 2, 2, 1, 0, 1, 99};        
    Nd4jLong yShapeInfo[] = {2, 1, 2, 2, 1, 0, 1, 99};        
    
    NDArray<float> x(xBuff, xShapeInfo);
    NDArray<float> y(yBuff, yShapeInfo);
    NDArray<float> target(x);
    NDArray<float> exp(expBuff, xShapeInfo);

    x.mulRowVector(&y,&target);    
    
    ASSERT_TRUE(exp.isSameShapeStrict(&target));
    ASSERT_TRUE(exp.equalsTo(&target));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestTensorDotAgain_1) {
    int sY = 1;
    int sX = 1;
    int pY = 0;
    int pX = 0;
    int iC = 2;
    int oC = 2;
    int kY = 3;
    int kX = 3;
    int iY = 2;
    int iX = 2;
    int oY = 6;
    int oX = 6;
    int eD = iC * oC;
    int B = 2;

    /*
    input = np.linspace(1, B * iC * iY * iX, B * iC * iY * iX).reshape(B, iC, iY, iX)
    weights = np.linspace(1, iC * oC * kY * kX, iC * oC * kY * kX).reshape(iC, oC, kY, kX)
     */
    double _expB[] = {96.0,  116.0,  136.0,  156.0,  256.0,  276.0,  296.0,  316.0,  102.0,  124.0,  146.0,  168.0,    278.0,  300.0,  322.0,  344.0,  108.0,  132.0,  156.0,  180.0,  300.0,  324.0,  348.0,  372.0,    114.0,  140.0,  166.0,  192.0,  322.0,  348.0,  374.0,  400.0,  120.0,  148.0,  176.0,  204.0,    344.0,  372.0,  400.0,  428.0,  126.0,  156.0,  186.0,  216.0,  366.0,  396.0,  426.0,  456.0,    132.0,  164.0,  196.0,  228.0,  388.0,  420.0,  452.0,  484.0,  138.0,  172.0,  206.0,  240.0,    410.0,  444.0,  478.0,  512.0,  144.0,  180.0,  216.0,  252.0,  432.0,  468.0,  504.0,  540.0,    150.0,  188.0,  226.0,  264.0,  454.0,  492.0,  530.0,  568.0,  156.0,  196.0,  236.0,  276.0,    476.0,  516.0,  556.0,  596.0,  162.0,  204.0,  246.0,  288.0,  498.0,  540.0,  582.0,  624.0,    168.0,  212.0,  256.0,  300.0,  520.0,  564.0,  608.0,  652.0,  174.0,  220.0,  266.0,  312.0,    542.0,  588.0,  634.0,  680.0,  180.0,  228.0,  276.0,  324.0,  564.0,  612.0,  660.0,  708.0,    186.0,  236.0,  286.0,  336.0,  586.0,  636.0,  686.0,  736.0,  192.0,  244.0,  296.0,  348.0,    608.0,  660.0,  712.0,  764.0,  198.0,  252.0,  306.0,  360.0,  630.0,  684.0,  738.0,  792.0};

    Nd4jLong _expS[] = {6, 2, 3, 3, 2, 2, 2, 72, 24, 8, 4, 2, 1, 0, 1, 99};
    NDArray<double> exp(_expB, _expS);
    exp.triggerAllocationFlag(false, false);

    NDArray<double> input('c', {B, iC, iY, iX});
    NDArray<double> weights('c', {iC, oC, kY, kX});

    NDArrayFactory<double>::linspace(1, input);
    NDArrayFactory<double>::linspace(1, weights);

    auto result = NDArrayFactory<double>::tensorDot(&weights, &input, {0}, {1});

    //result->printShapeInfo("result shape");
    ASSERT_TRUE(exp.isSameShape(result));

//        exp.printBuffer("Expctd buffer");
//    result->printBuffer("Result buffer");
    ASSERT_TRUE(exp.equalsTo(result));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestBroadcast_1) {
    double _expB[] = {1.000000, 1.000000, 1.000000, 1.000000, 2.000000, 2.000000, 2.000000, 2.000000, 3.000000, 3.000000, 3.000000, 3.000000, 1.000000, 1.000000, 1.000000, 1.000000, 2.000000, 2.000000, 2.000000, 2.000000, 3.000000, 3.000000, 3.000000, 3.000000};
    Nd4jLong _expS[] = {4, 2, 3, 2, 2, 12, 4, 2, 1, 0, 1, 99};
    NDArray<double> exp(_expB, _expS);
    exp.triggerAllocationFlag(false, false);

    NDArray<double> input('c',{ 2, 3, 2, 2});
    NDArray<double> bias('c', {1, 3});

    NDArrayFactory<double>::linspace(1, bias);

    input.template applyBroadcast<simdOps::Add<double>>({1}, &bias);

    //input.printBuffer("result");
    ASSERT_TRUE(exp.equalsTo(&input));
}

TEST_F(NDArrayTest, TestTranspose_11) {
    NDArray<float> x('c', {2, 3, 4});
    x.transposei();

    ASSERT_EQ(4, x.sizeAt(0));
    ASSERT_EQ(3, x.sizeAt(1));
    ASSERT_EQ(2, x.sizeAt(2));
}


TEST_F(NDArrayTest, TestTranspose_12) {
    NDArray<float> x('c', {2, 3, 4});
    auto y = x.transpose();

    ASSERT_EQ(4, y->sizeAt(0));
    ASSERT_EQ(3, y->sizeAt(1));
    ASSERT_EQ(2, y->sizeAt(2));

    ASSERT_EQ(2, x.sizeAt(0));
    ASSERT_EQ(3, x.sizeAt(1));
    ASSERT_EQ(4, x.sizeAt(2));

    delete y;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestMMulMultiDim) {
    const int bS=2;
    const int K=3;
    const int N=4;
    Nd4jLong expShape[] = {3, 2, 9, 4, 36, 4, 1, 0, 1, 99};
    double expBuff[] = { 38,   44,   50,   56, 83,   98,  113,  128, 128,  152,  176,  200, 173,  206,  239,  272, 218,  260,  302,  344, 263,  314,  365,  416, 308,  368,  428,  488, 353,  422,  491,  560, 398,  476,  554,  632, 110,  116,  122,  128, 263,  278,  293,  308, 416,  440,  464,  488, 569,  602,  635,  668, 722,  764,  806,  848, 875,  926,  977, 1028, 1028, 1088, 1148, 1208, 1181, 1250, 1319, 1388, 1334, 1412, 1490, 1568};

    NDArray<double> input  ('c', {bS,  K, N});
    NDArray<double> weights('c', {3*K, K});
    //NDArray<double>* result(nullptr);
    NDArray<double> expected ('c', {bS,  3*K, N});
    expected.setBuffer(expBuff);
    expected.triggerAllocationFlag(false, true);

    NDArrayFactory<double>::linspace(1, input);
    NDArrayFactory<double>::linspace(1, weights);

    auto result = NDArrayFactory<double>::mmulHelper(&weights, &input, nullptr, 1., 0.);
    //  result must have such shape   [bS x 3K x N]

    ASSERT_TRUE(result->isSameShape(&expected));

    //result->printShapeInfo("result shape");
    //result->printBuffer("result buffer");
    ASSERT_TRUE(result->equalsTo(&expected));
    delete result;
}


TEST_F(NDArrayTest, AdditionOperator1) {

    NDArray<double> input1('c', {2,2});
    NDArray<double> input2('c', {2,2});
    NDArray<double> expected('c', {2,2});

    input1.assign(1.5);
    input2.assign(2.);
    expected.assign(3.5);

    input2 = input1 + input2;

    ASSERT_TRUE(input2.equalsTo(&expected));

}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestMatmMul_Again_1) {
    NDArray<double> a('c', {3, 4, 1});
    NDArray<double> b('c', {3, 1, 5});

    NDArrayFactory<double>::linspace(1, a);
    NDArrayFactory<double>::linspace(1, b);

    double _expB[] = {1.f,    2.f,    3.f,    4.f,    5.f,    2.f,    4.f,    6.f,    8.f,   10.f,    3.f,    6.f,    9.f,   12.f,   15.f,    4.f,    8.f,   12.f,   16.f,   20.f,   30.f,   35.f,   40.f,   45.f,    50.f,   36.f,   42.f,   48.f,   54.f,   60.f,   42.f,   49.f,   56.f,   63.f,   70.f,   48.f,    56.f,   64.f,   72.f,   80.f,   99.f,  108.f,  117.f,  126.f,  135.f,  110.f,  120.f,  130.f,    140.f,  150.f,  121.f,  132.f,  143.f,  154.f,  165.f,  132.f,  144.f,  156.f,  168.f,  180.f};
    Nd4jLong _expS[] = {3, 3, 4, 5, 20, 5, 1, 0, 1, 99};
    NDArray<double> c(_expB, _expS);
    c.triggerAllocationFlag(false, false);

    auto c_ = NDArrayFactory<double>::mmulHelper(&a, &b);

    ASSERT_TRUE(c.isSameShape(c_));
    ASSERT_TRUE(c.equalsTo(c_));

    delete c_;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestMatmMul_Again_2) {
    NDArray<double> a('c', {2, 5, 4});
    NDArray<double> b('c', {2, 4, 1});

    NDArrayFactory<double>::linspace(1, a);
    NDArrayFactory<double>::linspace(1, b);

    double _expB[] = {30.f,    70.f,   110.f,   150.f,   190.f,   590.f,   694.f,   798.f,   902.f,  1006.f};
    Nd4jLong _expS[] = {3, 2, 5, 1, 5, 1, 1, 0, 1, 99};
    NDArray<double> c(_expB, _expS);
    c.triggerAllocationFlag(false, false);

    auto c_ = NDArrayFactory<double>::mmulHelper(&a, &b);

    ASSERT_TRUE(c.isSameShape(c_));

    ASSERT_TRUE(c.equalsTo(c_));

    delete c_;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Plus_Test_1)
{
    double expBuff[] = {2., 3, 3., 4., 4., 5, 5., 6., 6., 7, 7., 8.};

    NDArray<double> x('c', {3, 1, 2});
    NDArray<double> y('c',    {2, 1});
    NDArray<double> expected(expBuff, 'c', {3, 2, 2});

    NDArrayFactory<double>::linspace(1, x);
    NDArrayFactory<double>::linspace(1, y);

    NDArray<double> result = x + y;

    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));
}


//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Plus_Test_2)
{
    double expBuff[] = {2., 3, 3., 4., 4., 5, 5., 6., 6., 7, 7., 8.};

    NDArray<double> x('c', {3, 2, 1});
    NDArray<double> y('c',    {1, 2});
    NDArray<double> expected(expBuff, 'c', {3, 2, 2});

    NDArrayFactory<double>::linspace(1, x);
    NDArrayFactory<double>::linspace(1, y);

    NDArray<double> result = x + y;
    // result.printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));
}


//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Plus_Test_3)
{
    double expBuff[] = {2., 3, 3., 4., 4., 5, 5., 6., 6., 7, 7., 8.};

    NDArray<double> x('c', {3, 2, 1});
    NDArray<double> y('c',    {1, 2});
    NDArray<double> expected(expBuff, 'c', {3, 2, 2});

    NDArrayFactory<double>::linspace(1, x);
    NDArrayFactory<double>::linspace(1, y);

    NDArray<double> result = x + y;
    // result.printIndexedBuffer();
    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Plus_Test_4)
{
    double expBuff[] = {11.,12., 12.,13., 13.,14., 14.,15., 13.,14., 14.,15., 15.,16., 16.,17., 15.,16., 16.,17., 17.,18., 18.,19.};

    NDArray<double> x('c', {3, 1, 2});
    NDArray<double> y('c',    {4, 1});
    NDArray<double> expected(expBuff, 'c', {3, 4, 2});

    NDArrayFactory<double>::linspace(10, x);
    NDArrayFactory<double>::linspace(1, y);

    NDArray<double> result = x + y;
    
    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));
}
 

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Minus_Test_1)
{
    double expBuff[] = {9. ,10., 10.,11., 11.,12., 12.,13., 17.,18., 18.,19., 19.,20., 20.,21., 25.,26., 26.,27., 27.,28., 28.,29.};

    NDArray<double> x('c', {3, 4, 2});
    NDArray<double> y('c',    {4, 1});
    NDArray<double> expected(expBuff, 'c', {3, 4, 2});

    NDArrayFactory<double>::linspace(10, x);
    NDArrayFactory<double>::linspace(1, y);

    NDArray<double> result = x - y;

    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));
}


//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Minus_Test_2)
{
    double expBuff[] = {9., 8., 7., 6., 6., 5., 4., 3., 11.,10., 9., 8., 8., 7., 6., 5., 13.,12.,11.,10., 10., 9., 8., 7.};

    NDArray<double> x('c', {3, 2, 1});
    NDArray<double> y('c', {1, 2, 4});
    NDArray<double> expected(expBuff, 'c', {3, 2, 4});

    NDArrayFactory<double>::linspace(10, x);
    NDArrayFactory<double>::linspace(1, y);

    NDArray<double> result = x - y;
    
    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Minus_Test_3)
{
    double expBuff[] = {9., 8., 7., 6., 6., 5., 4., 3., 11.,10., 9., 8., 8., 7., 6., 5., 13.,12.,11.,10., 10., 9., 8., 7.};

    NDArray<double> x('c', {3, 2, 1});
    NDArray<double> y('c', {2, 4});
    NDArray<double> expected(expBuff, 'c', {3, 2, 4});

    NDArrayFactory<double>::linspace(10, x);
    NDArrayFactory<double>::linspace(1, y);

    NDArray<double> result = x - y;
    
    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Minus_Test_4)
{
    double expBuff[] = {9.,10., 8., 9., 11.,12.,10.,11., 13.,14.,12.,13.};

    NDArray<double> x('c', {3, 1, 2});
    NDArray<double> y('c',    {2, 1});
    NDArray<double> expected(expBuff, 'c', {3, 2, 2});

    NDArrayFactory<double>::linspace(10, x);
    NDArrayFactory<double>::linspace(1, y);

    NDArray<double> result = x - y;
    
    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));
}

 
//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Minus_Test_5)
{
    double expBuff[] = {9. ,8 ,10., 9., 11.,10, 12.,11., 13.,12, 14.,13.};

    NDArray<double> x('c', {3, 2, 1});
    NDArray<double> y('c',    {1, 2});
    NDArray<double> expected(expBuff, 'c', {3, 2, 2});

    NDArrayFactory<double>::linspace(10, x);
    NDArrayFactory<double>::linspace(1, y);

    NDArray<double> result = x - y;    

    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));
}


//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Minus_Test_6)
{
    double expBuff[] = {9., 8, 10., 9, 11.,10, 12.,11., 13.,12, 14.,13, 15.,14, 16.,15., 17.,16, 18.,17, 19.,18, 20.,19.};

    NDArray<double> x('c', {3, 4, 1});
    NDArray<double> y('c', {1, 1, 2});
    NDArray<double> expected(expBuff, 'c', {3, 4, 2});

    NDArrayFactory<double>::linspace(10, x);
    NDArrayFactory<double>::linspace(1, y);

    NDArray<double> result = x - y;    

    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Multiply_Test_1)
{
    double expBuff[] = {10., 11., 24., 26., 42., 45., 64., 68., 18., 19., 40., 42., 66., 69., 96.,100., 26., 27., 56., 58., 90., 93., 128.,132.};

    NDArray<double> x('c', {3, 4, 2});
    NDArray<double> y('c',    {4, 1});
    NDArray<double> expected(expBuff, 'c', {3, 4, 2});

    NDArrayFactory<double>::linspace(10, x);
    NDArrayFactory<double>::linspace(1, y);

    NDArray<double> result = x * y;

    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Multiply_Test_2)
{
    double expBuff[] = {10.,20., 30., 40., 55.,66., 77., 88., 12.,24., 36., 48., 65.,78., 91.,104., 14.,28., 42., 56., 75.,90.,105.,120.};

    NDArray<double> x('c', {3, 2, 1});
    NDArray<double> y('c', {1, 2, 4});
    NDArray<double> expected(expBuff, 'c', {3, 2, 4});

    NDArrayFactory<double>::linspace(10, x);
    NDArrayFactory<double>::linspace(1, y);

    NDArray<double> result = x * y;
    
    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));
}


//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Multiply_Test_3)
{
    double expBuff[] = {10.,20., 30., 40.,55.,66., 77., 88., 12.,24., 36., 48.,65.,78., 91.,104., 14.,28., 42., 56.,75.,90.,105.,120.};

    NDArray<double> x('c', {3, 2, 1});
    NDArray<double> y('c', {2, 4});
    NDArray<double> expected(expBuff, 'c', {3, 2, 4});

    NDArrayFactory<double>::linspace(10, x);
    NDArrayFactory<double>::linspace(1, y);

    NDArray<double> result = x * y;
    
    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));
}


//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Multiply_Test_4)
{
    double expBuff[] = {10.,11.,20.,22., 12.,13.,24.,26., 14.,15.,28.,30.};

    NDArray<double> x('c', {3, 1, 2});
    NDArray<double> y('c',    {2, 1});
    NDArray<double> expected(expBuff, 'c', {3, 2, 2});

    NDArrayFactory<double>::linspace(10, x);
    NDArrayFactory<double>::linspace(1, y);

    NDArray<double> result = x * y;
    
    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));
}


 
//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Multiply_Test_5)
{
    double expBuff[] = {10.,20.,11.,22., 12.,24.,13.,26., 14.,28.,15.,30.};

    NDArray<double> x('c', {3, 2, 1});
    NDArray<double> y('c',    {1, 2});
    NDArray<double> expected(expBuff, 'c', {3, 2, 2});

    NDArrayFactory<double>::linspace(10, x);
    NDArrayFactory<double>::linspace(1, y);

    NDArray<double> result = x * y;    

    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));
}



//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Multiply_Test_6)
{
    double expBuff[] = {10,11.,12.,13.,28.,30.,32.,34.,54.,57.,60.,63.};

    NDArray<double> x('c', {3, 4, 1});
    NDArray<double> y('c', {3, 1, 1});
    NDArray<double> expected(expBuff, 'c', {3, 4, 1});

    NDArrayFactory<double>::linspace(10, x);
    NDArrayFactory<double>::linspace(1, y);

    NDArray<double> result = x * y;    

    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));
}


//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Divide_Test_1)
{
    double expBuff[] = {10.    ,11.    , 6.    , 6.5   , 4.6666, 5.    , 4.    , 4.25  , 18.    ,19.    , 10.    ,10.5   , 7.3333, 7.6666, 6.    , 6.25  , 26.    ,27.    , 14.    ,14.5   , 10.    ,10.3333, 8.    , 8.25};

    NDArray<double> x('c', {3, 4, 2});
    NDArray<double> y('c',    {4, 1});
    NDArray<double> expected(expBuff, 'c', {3, 4, 2});

    NDArrayFactory<double>::linspace(10, x);
    NDArrayFactory<double>::linspace(1, y);

    NDArray<double> result = x / y;

    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result,1e-4));
}


//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Divide_Test_2)
{
    double expBuff[] = {10. ,5.     ,3.333333,2.5  , 2.2,1.83333,1.571428,1.375, 12. ,6.     ,4.      ,3.   , 2.6,2.16666,1.857142,1.625, 14. ,7.     ,4.666666,3.5  , 3. ,2.5    ,2.142857,1.875};

    NDArray<double> x('c', {3, 2, 1});
    NDArray<double> y('c', {1, 2, 4});
    NDArray<double> expected(expBuff, 'c', {3, 2, 4});

    NDArrayFactory<double>::linspace(10, x);
    NDArrayFactory<double>::linspace(1, y);

    NDArray<double> result = x / y;
    
    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));
}


//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Divide_Test_3)
{
    double expBuff[] = {10. ,5.      ,3.333333,2.5  , 2.2,1.833333,1.571428,1.375, 12. ,6.      ,4.      ,3.   , 2.6,2.166666,1.857142,1.625, 14. ,7.      ,4.666666,3.5  , 3. ,2.5     ,2.142857,1.875};

    NDArray<double> x('c', {3, 2, 1});
    NDArray<double> y('c', {2, 4});
    NDArray<double> expected(expBuff, 'c', {3, 2, 4});

    NDArrayFactory<double>::linspace(10, x);
    NDArrayFactory<double>::linspace(1, y);

    NDArray<double> result = x / y;
    
    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));
}


//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Divide_Test_4)
{
    double expBuff[] = {10.,11., 5., 5.5, 12.,13., 6., 6.5, 14.,15., 7., 7.5};

    NDArray<double> x('c', {3, 1, 2});
    NDArray<double> y('c',    {2, 1});
    NDArray<double> expected(expBuff, 'c', {3, 2, 2});

    NDArrayFactory<double>::linspace(10, x);
    NDArrayFactory<double>::linspace(1, y);

    NDArray<double> result = x / y;
    
    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));
}


//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Divide_Test_5)
{
    double expBuff[] = {10.,5., 11.,5.5, 12.,6., 13.,6.5, 14.,7., 15.,7.5};

    NDArray<double> x('c', {3, 2, 1});
    NDArray<double> y('c',    {1, 2});
    NDArray<double> expected(expBuff, 'c', {3, 2, 2});

    NDArrayFactory<double>::linspace(10, x);
    NDArrayFactory<double>::linspace(1, y);

    NDArray<double> result = x / y;    

    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));
}


//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Divide_Test_6)
{
    double expBuff[] = {10.      , 5.5     , 4.      , 3.25    ,14.      , 7.5     , 5.333333, 4.25    ,18.      , 9.5     , 6.666666, 5.25};

    NDArray<double> x('c', {3, 4, 1});
    NDArray<double> y('c', {1, 4, 1});
    NDArray<double> expected(expBuff, 'c', {3, 4, 1});

    NDArrayFactory<double>::linspace(10, x);
    NDArrayFactory<double>::linspace(1, y);

    NDArray<double> result = x / y;    

    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));
}


//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Divide_Test_7)
{
    double expBuff[] = {10., 5. ,3.333333,2.5 ,11., 5.5,3.666666,2.75,12., 6. ,4.      ,3.  ,13., 6.5,4.333333,3.25, 14., 7. ,4.666666,3.5 ,15., 7.5,5.      ,3.75,16., 8. ,5.333333,4.  ,17., 8.5,5.666666,4.25, 18., 9. ,6.      ,4.5 ,19., 9.5,6.333333,4.75,20.,10. ,6.666666,5.  ,21.,10.5,7.      ,5.25};

    NDArray<double> x('c', {3, 4, 1});
    NDArray<double> y('c', {1, 1, 4});
    NDArray<double> expected(expBuff, 'c', {3, 4, 4});

    NDArrayFactory<double>::linspace(10, x);
    NDArrayFactory<double>::linspace(1, y);

    NDArray<double> result = x / y;    

    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_Lambda_1) {
    NDArray<float> x('c', {1, 5}, {1, 2, 3, 4, 5});
    NDArray<float> exp('c', {1, 5}, {4, 5, 6, 7, 8});

    auto lambda = LAMBDA_F(_val) {
        return _val + 3.0f;
    };

    x.applyLambda(lambda);

    ASSERT_TRUE(exp.equalsTo(&x));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_Lambda_2) {
    NDArray<float> x('c', {1, 5}, {1, 2, 1, 2, 1});
    NDArray<float> y('c', {1, 5}, {1, 2, 1, 2, 1});
    NDArray<float> exp('c', {1, 5}, {3, 5, 3, 5, 3});

    auto lambda = LAMBDA_FF(_x, _y) {
        return _x + _y + 1.0f;
    };

    x.applyPairwiseLambda(&y, lambda);

    ASSERT_TRUE(exp.equalsTo(&x));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_Lambda_3) {
    NDArray<double> x('c', {1, 5}, {1, 2, 1, 2, 1});
    NDArray<double> y('c', {1, 5}, {1, 2, 1, 2, 1});
    NDArray<double> exp('c', {1, 5}, {4, 8, 4, 8, 4});

    auto lambda = LAMBDA_DD(_x, _y) {
        return (_x + _y) * 2;
    };

    x.applyPairwiseLambda(&y, lambda);

    ASSERT_TRUE(exp.equalsTo(&x));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_swapUnsafe_1) {
    
    NDArray<double> x('c', {2, 2}, {1, 2, 3, 4});
    NDArray<double> y('c', {1, 4}, {5, 6, 7, 8});
    NDArray<double> expX('c', {2, 2}, {5, 6, 7, 8});
    NDArray<double> expY('c', {1, 4}, {1, 2, 3, 4});

    x.swapUnsafe(y);

    ASSERT_TRUE(expX.equalsTo(&x));
    ASSERT_TRUE(expY.equalsTo(&y));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_1) {
    
    NDArray<double> x('c', {2, 3}, {1, 2, 3, 4, 5, 6});
    NDArray<double> exp('c', {2, 1}, {1, 5});    

    NDArray<double>* diag = x.diagonal('c');
    
    ASSERT_TRUE(exp.isSameShape(diag));
    ASSERT_TRUE(exp.equalsTo(diag));
    
    delete diag;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_2) {
    
    NDArray<double> x('f', {2, 3});
    NDArray<double> exp('f', {2, 1}, {1, 5});    
    NDArrayFactory<double>::linspace(1, x);

    NDArray<double>* diag = x.diagonal('c');            

    ASSERT_TRUE(exp.isSameShape(diag));
    ASSERT_TRUE(exp.equalsTo(diag));
    
    delete diag;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_3) {
    
    NDArray<double> x('c', {2, 2});
    NDArrayFactory<double>::linspace(1, x);
    NDArray<double> exp('c', {1, 2}, {1, 4});    

    NDArray<double>* diag = x.diagonal('r');    

    ASSERT_TRUE(exp.isSameShape(diag));
    ASSERT_TRUE(exp.equalsTo(diag));
    
    delete diag;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_4) {
    
    NDArray<double> x('f', {2, 2});
    NDArrayFactory<double>::linspace(1, x);
    NDArray<double> exp('f', {1, 2}, {1, 4});    

    NDArray<double>* diag = x.diagonal('r');    

    ASSERT_TRUE(exp.isSameShape(diag));
    ASSERT_TRUE(exp.equalsTo(diag));
    
    delete diag;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_5) {
    
    NDArray<double> x('c', {2, 2, 2});
    NDArrayFactory<double>::linspace(1, x);
    NDArray<double> exp('c', {1, 2}, {1, 8});    

    NDArray<double>* diag = x.diagonal('r');    

    ASSERT_TRUE(exp.isSameShape(diag));
    ASSERT_TRUE(exp.equalsTo(diag));
    
    delete diag;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_6) {
    
    NDArray<double> x('f', {2, 2, 2});
    NDArrayFactory<double>::linspace(1, x);
    NDArray<double> exp('f', {1, 2}, {1, 8});    

    NDArray<double>* diag = x.diagonal('r');    

    ASSERT_TRUE(exp.isSameShape(diag));
    ASSERT_TRUE(exp.equalsTo(diag));
    
    delete diag;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_7) {
    
    NDArray<double> x('f', {2, 2, 2});
    NDArrayFactory<double>::linspace(1, x);
    NDArray<double> exp('f', {2, 1}, {1, 8});    

    NDArray<double>* diag = x.diagonal('c');    

    ASSERT_TRUE(exp.isSameShape(diag));
    ASSERT_TRUE(exp.equalsTo(diag));
    
    delete diag;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_8) {
    
    NDArray<double> x('c', {2, 3});
    NDArrayFactory<double>::linspace(1, x);
    NDArray<double> exp('c', {1, 2}, {1, 5});    

    NDArray<double>* diag = x.diagonal('r');
    
    ASSERT_TRUE(exp.isSameShape(diag));
    ASSERT_TRUE(exp.equalsTo(diag));
    
    delete diag;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_9) {
    
    NDArray<double> x('c', {2, 2});
    NDArrayFactory<double>::linspace(1, x);
    NDArray<double> exp('c', {2, 1}, {1, 4});    

    NDArray<double>* diag = x.diagonal('c');    

    ASSERT_TRUE(exp.isSameShape(diag));
    ASSERT_TRUE(exp.equalsTo(diag));
    
    delete diag;
}


//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_10) {
    
    NDArray<double> x('f', {2, 2});
    NDArrayFactory<double>::linspace(1, x);    
    NDArray<double> exp('f', {2, 1}, {1, 4});    

    NDArray<double>* diag = x.diagonal('c');    

    ASSERT_TRUE(exp.isSameShape(diag));
    ASSERT_TRUE(exp.equalsTo(diag));
    
    delete diag;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_11) {
    
    NDArray<double> x('f', {3, 3});
    NDArrayFactory<double>::linspace(1, x);
    NDArray<double> exp('f', {3, 1}, {1, 5, 9});    

    NDArray<double>* diag = x.diagonal('c');        

    ASSERT_TRUE(exp.isSameShape(diag));
    ASSERT_TRUE(exp.equalsTo(diag));
    
    delete diag;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_12) {
    
    NDArray<double> x('c', {3, 3});
    NDArrayFactory<double>::linspace(1, x);
    NDArray<double> exp('c', {1, 3}, {1, 5, 9});    

    NDArray<double>* diag = x.diagonal('r');        

    ASSERT_TRUE(exp.isSameShape(diag));
    ASSERT_TRUE(exp.equalsTo(diag));
    
    delete diag;
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_13) {
    
    NDArray<double> x('c', {3, 3, 4});
    NDArrayFactory<double>::linspace(1, x);
    NDArray<double> exp('c', {3, 1}, {1,18,35});        
    
    NDArray<double>* diag = x.diagonal('c');
        
    ASSERT_TRUE(exp.isSameShape(diag));
    ASSERT_TRUE(exp.equalsTo(diag));
    
    delete diag;
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_14) {
    
    NDArray<double> x('c', {3, 3, 4});
    NDArrayFactory<double>::linspace(1, x);
    NDArray<double> exp('c', {1, 3}, {1,18,35});        
    
    NDArray<double>* diag = x.diagonal('r');
        
    ASSERT_TRUE(exp.isSameShape(diag));
    ASSERT_TRUE(exp.equalsTo(diag));
    
    delete diag;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_15) {
    
    NDArray<double> x('f', {3, 3, 4});
    NDArrayFactory<double>::linspace(1, x);
    NDArray<double> exp('f', {1, 3}, {1,18,35});        
    
    NDArray<double>* diag = x.diagonal('r');
        
    ASSERT_TRUE(exp.isSameShape(diag));
    ASSERT_TRUE(exp.equalsTo(diag));
    
    delete diag;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_16) {
    
    NDArray<double> x('f', {1, 5});
    NDArrayFactory<double>::linspace(1, x);
    NDArray<double> exp('f', {1, 1}, {1});        
    
    NDArray<double>* diag = x.diagonal('c');
        
    ASSERT_TRUE(exp.isSameShape(diag));
    ASSERT_TRUE(exp.equalsTo(diag));
    
    delete diag;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_17) {
    
    NDArray<double> x('c', {5, 1});
    NDArrayFactory<double>::linspace(1, x);
    NDArray<double> exp('c', {1, 1}, {1});        
    
    NDArray<double>* diag = x.diagonal('r');
        
    ASSERT_TRUE(exp.isSameShape(diag));
    ASSERT_TRUE(exp.equalsTo(diag));
    
    delete diag;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_18) {
    
    NDArray<double> x('f', {1, 1});
    NDArrayFactory<double>::linspace(1, x);
    NDArray<double> exp('f', {1, 1}, {1});        
    
    NDArray<double>* diag = x.diagonal('r');
        
    ASSERT_TRUE(exp.isSameShape(diag));
    ASSERT_TRUE(exp.equalsTo(diag));
    
    delete diag;
}


