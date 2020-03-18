/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author raver119@gmail.com
//

#ifndef LIBND4J_TADTESTS_H
#define LIBND4J_TADTESTS_H

#include "testlayers.h"
#include <array/NDArray.h>
#include <helpers/TAD.h>
#include <array>
#include <helpers/ConstantTadHelper.h>

using namespace sd;

class TadTests : public testing::Test {
public:
    int numLoops = 100000000;

    int extLoops = 1000;
    int intLoops = 1000;
};

TEST_F(TadTests, Test4DTad1) {

    NDArray*  arraySource = sd::NDArrayFactory::linspace(1.0f, 10000.0f, 10000);

    Nd4jLong badShape[]  = {4, 2, 1, 4, 4, 80, 16, 4, 1, 8192, -1, 99};
    Nd4jLong goodShape[] = {4, 2, 1, 4, 4, 16, 16, 4, 1, 8192,  1, 99};

    std::vector<float> buff = arraySource->getBufferAsVector<float>();

    NDArray* arrayExp = new NDArray(buff.data(), goodShape);
    NDArray* arrayBad = new NDArray(buff.data(), badShape);

    int dim = 1;
    shape::TAD tad;
    tad.init(arrayBad->getShapeInfo(), &dim, 1);
    tad.createTadOnlyShapeInfo();
    tad.createOffsets();

    int exp[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95 };
    for (int e = 0; e < 32; e++)
        ASSERT_EQ((int) tad.tadOffsets[e],  exp[e]);

    delete arrayExp;
    delete arrayBad;
    delete arraySource;
}

TEST_F(TadTests, TestNumTads1) {
    auto x = NDArrayFactory::create<float>('c', {2, 3});
    auto y = NDArrayFactory::create<float>('c', {2, 2});

    std::vector<int> dim({0});

    Nd4jLong tadLengthX = shape::tadLength(x.getShapeInfo(), dim.data(), dim.size());
    Nd4jLong numTadsX = x.lengthOf() / tadLengthX;

    Nd4jLong tadLengthY = shape::tadLength(y.getShapeInfo(), dim.data(), dim.size());
    Nd4jLong numTadsY = y.lengthOf() / tadLengthY;

    ASSERT_EQ(2, tadLengthX);
    ASSERT_EQ(3, numTadsX);

    ASSERT_EQ(2, tadLengthY);
    ASSERT_EQ(2, numTadsY);
}

TEST_F(TadTests, TestShapeTad_1) {

    float buff[]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,16,17,18,19,20,21,22,23,24};
    Nd4jLong shapeInfo[] = {3, 2, 3, 4, 12, 4, 1, 8192, 1, 99};

    NDArray input(buff, shapeInfo);

    std::vector<int> dimensions = {0,1,2};
    Nd4jLong tadLength = shape::tadLength(input.getShapeInfo(), dimensions.data(), dimensions.size());
    Nd4jLong numTads = input.lengthOf() / tadLength;

    shape::TAD tad;
    tad.init(input.getShapeInfo(), dimensions.data(), dimensions.size());
    tad.createTadOnlyShapeInfo();
    tad.createOffsets();

    auto tadShapeInfo = new Nd4jLong[shape::shapeInfoLength(tad.tadOnlyShapeInfo[0])];
    std::memcpy(tadShapeInfo, tad.tadOnlyShapeInfo, shape::shapeInfoByteLength(tad.tadOnlyShapeInfo));

    float* tadBuff = reinterpret_cast<float*>(input.getBuffer()) + tad.tadOffsets[0];
    NDArray tadArr(tadBuff, tadShapeInfo);

    ASSERT_TRUE(numTads==1);
    ASSERT_TRUE(input.isSameShapeStrict(tadArr));
    ASSERT_TRUE(input.equalsTo(&tadArr));

	delete[] tadShapeInfo;
}

TEST_F(TadTests, TadNoAxis_1) {
    auto array = NDArrayFactory::create<float>('c', {2, 3});

    shape::TAD tad;
    tad.init(array.shapeInfo(), nullptr, 0);
    tad.createTadOnlyShapeInfo();
    tad.createOffsets();

    ASSERT_TRUE(tad.wholeThing);

    ASSERT_TRUE(shape::equalsStrict(tad.tadOnlyShapeInfo, array.shapeInfo()));
}

TEST_F(TadTests, TadEdgeCase_1) {
    auto array = NDArrayFactory::create<float>('c', {5, 4, 1});
    auto exp = NDArrayFactory::create<float>('c', {5, 4});
    array.linspace(1);

    auto tad = array(0, {2});

    ASSERT_TRUE(exp.isSameShape(tad));
}

TEST_F(TadTests, TestEdgeCase_2) {

    auto array = NDArrayFactory::create<float>('f', {2, 3, 1}, {1, 4, 2, 5, 3, 6});

    for (int e = 0 ; e < array.lengthOf(); e++) {
        auto tad = array(e, {0,1});
        ASSERT_NEAR(tad.e<float>(0), array.e<float>(e), 1e-5);
    }
}

TEST_F(TadTests, TadEdgeCase_2) {
    auto array = NDArrayFactory::create<float>('c', {2, 3, 4});

    auto tad = array(0, {0,2});

    ASSERT_EQ(3, tad.lengthOf());
}


TEST_F(TadTests, test_Tad_Ews_optimization_1) {
    shape::TAD xTad;

    std::array<int,2> array = {1,2};
    ASSERT_TRUE(xTad.dimensionsDescending(3, array.data(), array.size()));
}

TEST_F(TadTests, test_Tad_Ews_optimization_2) {
    shape::TAD xTad;

    std::array<int,2> array = {0,2};
    ASSERT_FALSE(xTad.dimensionsDescending(3, array.data(), array.size()));
}

TEST_F(TadTests, test_Tad_Ews_optimization_3) {
    shape::TAD xTad;

    std::array<int,1> array = {1};
    ASSERT_TRUE(xTad.dimensionsDescending(2, array.data(), array.size()));
}

TEST_F(TadTests, test_Tad_Ews_optimization_4) {
    shape::TAD xTad;

    std::array<int,1> array = {0};
    ASSERT_TRUE(xTad.dimensionsDescending(1, array.data(), array.size()));
}

TEST_F(TadTests, test_Tad_Ews_optimization_5) {
    shape::TAD xTad;

    std::array<int,2> array = {2,3};
    ASSERT_TRUE(xTad.dimensionsDescending(4, array.data(), array.size()));
}

TEST_F(TadTests, test_TAD_empty_dims_1) {
    Nd4jLong xShape[8] = {2, 150, 1, 3, 1, 16384, 3, 99};
    shape::TAD xTad;
    xTad.init(xShape, reinterpret_cast<int*>(112L), 0);
    xTad.createTadOnlyShapeInfo();
    xTad.createOffsets();
}

TEST_F(TadTests, test_tad_order_1) {
    Nd4jLong xShape[8] = {2, 150, 10, 10, 1, 8192, 1, 99};
    Nd4jLong tShape[8] = {2, 1, 10, 1, 1, 8192, 1, 99};
    shape::TAD xTad;
    int dim = 1;
    xTad.init(xShape, &dim, 1);
    xTad.createTadOnlyShapeInfo();

    ASSERT_TRUE(shape::equalsStrict(tShape, xTad.tadOnlyShapeInfo));
}

TEST_F(TadTests, test_tad_order_2) {
    Nd4jLong xShape[8] = {2, 150, 10, 10, 1, 8192, 1, 99};
    Nd4jLong tShape[8] = {2, 1, 150, 1, 10, 8192, 10, 99};
    shape::TAD xTad;
    int dim = 0;
    xTad.init(xShape, &dim, 1);
    xTad.createTadOnlyShapeInfo();

    ASSERT_TRUE(shape::equalsStrict(tShape, xTad.tadOnlyShapeInfo));
}


TEST_F(TadTests, test_tad_order_3) {
    Nd4jLong xShape[10] = {3, 10, 20, 30, 600 ,30, 1, 8192, 1, 99};
    Nd4jLong tShape[8] = {2, 1, 30, 1, 1, 8192, 1, 99};
    shape::TAD xTad;
    int dim = 2;
    xTad.init(xShape, &dim, 1);
    xTad.createTadOnlyShapeInfo();

    ASSERT_TRUE(shape::equalsStrict(tShape, xTad.tadOnlyShapeInfo));
}


TEST_F(TadTests, test_tad_order_4) {
    Nd4jLong xShape[10] = {3, 10, 20, 30, 600 ,30, 1, 8192, 1, 99};
    Nd4jLong tShape[8] = {2, 20, 30, 30, 1, 8192, 1, 99};
    shape::TAD xTad;
    int dim[2] = {1, 2};
    xTad.init(xShape, dim, 2);
    xTad.createTadOnlyShapeInfo();

    ASSERT_TRUE(shape::equalsStrict(tShape, xTad.tadOnlyShapeInfo));
}

TEST_F(TadTests, test_column_1) {
    auto x = NDArrayFactory::create<float>('c', {5, 2});
    auto tadPack = sd::ConstantTadHelper::getInstance()->tadForDimensions(x.shapeInfo(), 0);

    ASSERT_EQ(1, shape::rank(tadPack.primaryShapeInfo()));
    ASSERT_EQ(5, shape::length(tadPack.primaryShapeInfo()));
    ASSERT_TRUE(shape::isVector(tadPack.primaryShapeInfo()));

    auto scalarViewPack = sd::ConstantTadHelper::getInstance()->tadForDimensions(tadPack.primaryShapeInfo(), 0);

    ASSERT_TRUE(shape::equalsStrict(tadPack.primaryShapeInfo(), scalarViewPack.primaryShapeInfo()));
}

///////////////////////////////////////////////////////////////////
TEST_F(TadTests, calcOffsets_1) {

    Nd4jLong shapeInfoF[10]  = {3, 2,3,4,  1,2,6,   8192, 1, 102};
    Nd4jLong shapeInfoC[10]  = {3, 2,3,4,  12,4,1,  8192, 1, 99};
    Nd4jLong shapeInfoFC[10] = {3, 2,3,4,  1,2,6,   8192, 1, 99};;

    Nd4jLong expOffsetsF[24] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23};
    Nd4jLong expOffsetsC[24] = {0,12,4,16,8,20,1,13,5,17,9,21,2,14,6,18,10,22,3,15,7,19,11,23};

    Nd4jLong offsets[24];

    shape::calcOffsets(shapeInfoF, offsets, 'f');

    for (int e = 0; e < 24; e++)
        ASSERT_TRUE(offsets[e] == expOffsetsF[e]);

    shape::calcOffsets(shapeInfoC, offsets, 'f');

    for (int e = 0; e < 24; e++)
        ASSERT_TRUE(offsets[e] == expOffsetsC[e]);

    shape::calcOffsets(shapeInfoFC, offsets, 'f');

    for (int e = 0; e < 24; e++)
        ASSERT_TRUE(offsets[e] == expOffsetsF[e]);
}


/////////////////////////////////////////////////////////////////
TEST_F(TadTests, outerArrayIndexes_1) {

    NDArray x('c', {2,3,4,5}, sd::DataType::FLOAT32);
    int maxIdxs[120];

    NDArray y1('c', {3,5}, sd::DataType::FLOAT32);
    const std::vector<int> dimsToExclude1 = {0,2};
    const int n1[] = {20,25,30,35,  80,85,90,95};
    int minIdx = 5;

    int N = shape::outerArrayIndexes(maxIdxs, minIdx, x.getShapeInfo(), y1.getShapeInfo(), dimsToExclude1.data());
    ASSERT_TRUE(N == x.lengthOf()/y1.lengthOf());
    for(int i = 0; i < N; ++i)
        ASSERT_TRUE(n1[i] == maxIdxs[i]);

    NDArray y2('c', {4,5}, sd::DataType::FLOAT32);
    const std::vector<int> dimsToExclude2 = {0,1};
    const int n2[] = {12,32,52,  72,92,112};
    minIdx = 12;

    N = shape::outerArrayIndexes(maxIdxs, minIdx, x.getShapeInfo(), y2.getShapeInfo(), dimsToExclude2.data());
    ASSERT_TRUE(N == x.lengthOf()/y2.lengthOf());
    for(int i = 0; i < N; ++i)
        ASSERT_TRUE(n2[i] == maxIdxs[i]);

    NDArray y3('c', {2,5}, sd::DataType::FLOAT32);
    const std::vector<int> dimsToExclude3 = {1,2};
    const int n3[] = {64,69,74,79,84,89,94,99,104,109,114,119};
    minIdx = 9;

    N = shape::outerArrayIndexes(maxIdxs, minIdx, x.getShapeInfo(), y3.getShapeInfo(), dimsToExclude3.data());
    ASSERT_TRUE(N == x.lengthOf()/y3.lengthOf());
    for(int i = 0; i < N; ++i)
        ASSERT_TRUE(n3[i] == maxIdxs[i]);

    NDArray y4('c', {2,3}, sd::DataType::FLOAT32);
    const std::vector<int> dimsToExclude4 = {2,3};
    const int n4[] = {20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39};
    minIdx = 1;

    N = shape::outerArrayIndexes(maxIdxs, minIdx, x.getShapeInfo(), y4.getShapeInfo(), dimsToExclude4.data());
    ASSERT_TRUE(N == x.lengthOf()/y4.lengthOf());
    for(int i = 0; i < N; ++i)
        ASSERT_TRUE(n4[i] == maxIdxs[i]);

    NDArray y5('c', {2,4}, sd::DataType::FLOAT32);
    const std::vector<int> dimsToExclude5 = {1,3};
    const int n5[] = {65,66,67,68,69, 85,86,87,88,89, 105,106,107,108,109};
    minIdx = 5;

    N = shape::outerArrayIndexes(maxIdxs, minIdx, x.getShapeInfo(), y5.getShapeInfo(), dimsToExclude5.data());
    ASSERT_TRUE(N == x.lengthOf()/y5.lengthOf());
    for(int i = 0; i < N; ++i)
        ASSERT_TRUE(n5[i] == maxIdxs[i]);

    NDArray y6('c', {2,3,4}, sd::DataType::FLOAT32);
    const std::vector<int> dimsToExclude6 = {3};
    const int n6[] = {65,66,67,68,69};
    minIdx = 13;

    N = shape::outerArrayIndexes(maxIdxs, minIdx, x.getShapeInfo(), y6.getShapeInfo(), dimsToExclude6.data());
    ASSERT_TRUE(N == x.lengthOf()/y6.lengthOf());
    for(int i = 0; i < N; ++i)
        ASSERT_TRUE(n6[i] == maxIdxs[i]);

    NDArray y7('c', {4}, sd::DataType::FLOAT32);
    const std::vector<int> dimsToExclude7 = {0,1,3};
    const int n7[] = {15,16,17,18,19, 35,36,37,38,39, 55,56,57,58,59, 75,76,77,78,79, 95,96,97,98,99, 115,116,117,118,119};
    minIdx = 3;

    N = shape::outerArrayIndexes(maxIdxs, minIdx, x.getShapeInfo(), y7.getShapeInfo(), dimsToExclude7.data());
    ASSERT_TRUE(N == x.lengthOf()/y7.lengthOf());
    for(int i = 0; i < N; ++i)
        ASSERT_TRUE(n7[i] == maxIdxs[i]);

    NDArray y8('c', {5}, sd::DataType::FLOAT32);
    const std::vector<int> dimsToExclude8 = {0,1,2};
    const int n8[] = {0,5,10,15,  20,25,30,35, 40,45,50,55, 60,65,70,75, 80,85,90,95, 100,105,110,115};
    minIdx = 0;

    N = shape::outerArrayIndexes(maxIdxs, minIdx, x.getShapeInfo(), y8.getShapeInfo(), dimsToExclude8.data());
    ASSERT_TRUE(N == x.lengthOf()/y8.lengthOf());
    for(int i = 0; i < N; ++i)
        ASSERT_TRUE(n8[i] == maxIdxs[i]);

    NDArray y9('c', {2}, sd::DataType::FLOAT32);
    const std::vector<int> dimsToExclude9 = {1,2,3};
    const int n9[] = {60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119};
    minIdx = 1;

    N = shape::outerArrayIndexes(maxIdxs, minIdx, x.getShapeInfo(), y9.getShapeInfo(), dimsToExclude9.data());
    ASSERT_TRUE(N == x.lengthOf()/y9.lengthOf());
    for(int i = 0; i < N; ++i)
        ASSERT_TRUE(n9[i] == maxIdxs[i]);

    NDArray y10('c', {3,4,5}, sd::DataType::FLOAT32);
    const std::vector<int> dimsToExclude10 = {0};
    const int n10[] = {11, 71};
    minIdx = 11;

    N = shape::outerArrayIndexes(maxIdxs, minIdx, x.getShapeInfo(), y10.getShapeInfo(), dimsToExclude10.data());
    ASSERT_TRUE(N == x.lengthOf()/y10.lengthOf());
    for(int i = 0; i < N; ++i)
        ASSERT_TRUE(n10[i] == maxIdxs[i]);

    NDArray y11('c', {2,4,5}, sd::DataType::FLOAT32);
    const std::vector<int> dimsToExclude11 = {1};
    const int n11[] = {66, 86, 106};
    minIdx = 26;

    N = shape::outerArrayIndexes(maxIdxs, minIdx, x.getShapeInfo(), y11.getShapeInfo(), dimsToExclude11.data());
    ASSERT_TRUE(N == x.lengthOf()/y11.lengthOf());
    for(int i = 0; i < N; ++i)
        ASSERT_TRUE(n11[i] == maxIdxs[i]);

    NDArray y12('c', {3,2}, sd::DataType::FLOAT32);
    const std::vector<int> dimsToExclude12 = {0,2};
    const int n12[] = {0,2,4,5,7,9,10,12,14,15,17,19,60,62,64,65,67,69,70,72,74,75,77,79};
    minIdx = 0;

    N = shape::outerArrayIndexes(maxIdxs, minIdx, x.getShapeInfo(), y12.getShapeInfo(), dimsToExclude12.data());
    for(int i = 0; i < N; ++i)
        ASSERT_TRUE(n12[i] == maxIdxs[i]);

    NDArray y13('c', {3,2}, sd::DataType::FLOAT32);
    const std::vector<int> dimsToExclude13 = {0,2};
    const int n13[] = {1,3,6,8,11,13,16,18,61,63,66,68,71,73,76,78};
    minIdx = 1;

    N = shape::outerArrayIndexes(maxIdxs, minIdx, x.getShapeInfo(), y13.getShapeInfo(), dimsToExclude13.data());
    for(int i = 0; i < N; ++i)
        ASSERT_TRUE(n13[i] == maxIdxs[i]);

    NDArray y14('c', {4,5}, sd::DataType::FLOAT32);
    const int n14[] = {12,32,52,  72,92,112};
    minIdx = 12;

    N = shape::outerArrayIndexes(maxIdxs, minIdx, x.getShapeInfo(), y14.getShapeInfo(), nullptr);
    ASSERT_TRUE(N == x.lengthOf()/y14.lengthOf());
    for(int i = 0; i < N; ++i)
        ASSERT_TRUE(n14[i] == maxIdxs[i]);

    NDArray y15('c', {3,4,5}, sd::DataType::FLOAT32);
    const int n15[] = {11, 71};
    minIdx = 11;

    N = shape::outerArrayIndexes(maxIdxs, minIdx, x.getShapeInfo(), y15.getShapeInfo(), nullptr);
    ASSERT_TRUE(N == x.lengthOf()/y15.lengthOf());
    for(int i = 0; i < N; ++i)
        ASSERT_TRUE(n15[i] == maxIdxs[i]);
}



#endif //LIBND4J_TADTESTS_H
