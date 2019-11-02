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

#include "testlayers.h"
#include <householder.h>
#include <biDiagonalUp.h>
#include <hhSequence.h>
#include <svd.h>
#include <hhColPivQR.h>
#include <jacobiSVD.h>
#include <ops/declarable/helpers/reverse.h>
#include <ops/declarable/helpers/activations.h>
#include <ops/declarable/helpers/rnn.h>
#include <ops/declarable/helpers/sg_cb.h>
#include <MmulHelper.h>
#include <GradCheck.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/lstmLayer.h>


using namespace nd4j;

class HelpersTests1 : public testing::Test {
public:

    HelpersTests1() {

        std::cout<<std::endl<<std::flush;
    }

};

#ifndef __CUDABLAS__

TEST_F(HelpersTests1, test_binary_search_1) {
    std::array<int, 10> array({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    auto idx = nd4j::ops::helpers::binarySearch(array.data(), 2, 10);
    ASSERT_EQ(2, idx);
}

TEST_F(HelpersTests1, test_binary_search_2) {
    std::array<int, 10> array({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    auto idx = nd4j::ops::helpers::binarySearch(array.data(), 18, 10);
    ASSERT_EQ(-1, idx);
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, evalHHmatrix_test1) {


    auto x = NDArrayFactory::create<double>('c', {1,4}, {14,17,3,1});
    auto exp = NDArrayFactory::create<double>('c', {4,4}, {-0.629253, -0.764093,   -0.13484, -0.0449467, -0.764093,  0.641653, -0.0632377, -0.0210792, -0.13484,-0.0632377,    0.98884,-0.00371987, -0.0449467,-0.0210792,-0.00371987,    0.99876});

    auto result = ops::helpers::Householder<double>::evalHHmatrix(x);

    ASSERT_TRUE(result.isSameShapeStrict(&exp));
    ASSERT_TRUE(result.equalsTo(&exp));

}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, evalHHmatrix_test2) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto x = NDArrayFactory::create<double>('c', {1,3}, {14,-4,3});
    auto exp = NDArrayFactory::create<double>('c', {3,3}, {-0.941742, 0.269069,-0.201802, 0.269069, 0.962715,0.0279639, -0.201802,0.0279639, 0.979027});

    auto result = ops::helpers::Householder<double>::evalHHmatrix(x);

    ASSERT_TRUE(result.isSameShapeStrict(&exp));
    ASSERT_TRUE(result.equalsTo(&exp));

}


/////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, evalHHmatrixData_test1) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto x = NDArrayFactory::create<double>('c', {1,4}, {14,17,3,1});
    auto tail = NDArrayFactory::create<double>('c', {1,3});
    auto expTail = NDArrayFactory::create<double>('c', {1,3}, {0.468984, 0.0827618, 0.0275873});
    const double normXExpected = -22.2486;
    const double coeffExpected = 1.62925;

    double normX, coeff;
    ops::helpers::Householder<double>::evalHHmatrixData(x, tail, coeff, normX);

    ASSERT_NEAR(normX, normXExpected, 1e-5);
    ASSERT_NEAR(coeff, coeffExpected, 1e-5);
    ASSERT_TRUE(tail.isSameShapeStrict(&expTail));
    ASSERT_TRUE(tail.equalsTo(&expTail));

}


/////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, Householder_mulLeft_test1) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto x = NDArrayFactory::create<double>('c', {4,4}, {12 ,19 ,14 ,3 ,10 ,4 ,17 ,19 ,19 ,18 ,5 ,3 ,6 ,4 ,2 ,16});
    auto tail = NDArrayFactory::create<double>('c', {1,3}, {0.5,0.5,0.5});
    auto exp = NDArrayFactory::create<double>('c', {4,4}, {9.05,15.8,11.4, 0.8, 8.525, 2.4,15.7,17.9, 17.525,16.4, 3.7, 1.9, 4.525, 2.4, 0.7,14.9});

    ops::helpers::Householder<double>::mulLeft(x, tail, 0.1);
    // expTail.printShapeInfo();

    ASSERT_TRUE(x.isSameShapeStrict(&exp));
    ASSERT_TRUE(x.equalsTo(&exp));

}

/////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, Householder_mulLeft_test2) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto x = NDArrayFactory::create<double>('c', {4,4}, {12 ,19 ,14 ,3 ,10 ,4 ,17 ,19 ,19 ,18 ,5 ,3 ,6 ,4 ,2 ,16});
    auto tail = NDArrayFactory::create<double>('c', {3,1}, {0.5,0.5,0.5});
    auto exp = NDArrayFactory::create<double>('c', {4,4}, {9.05,15.8,11.4, 0.8, 8.525, 2.4,15.7,17.9, 17.525,16.4, 3.7, 1.9, 4.525, 2.4, 0.7,14.9});

    ops::helpers::Householder<double>::mulLeft(x, tail, 0.1);

    ASSERT_TRUE(x.isSameShapeStrict(&exp));
    ASSERT_TRUE(x.equalsTo(&exp));

}

/////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, Householder_mulRight_test1) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto x = NDArrayFactory::create<double>('c', {4,4}, {12 ,19 ,14 ,3 ,10 ,4 ,17 ,19 ,19 ,18 ,5 ,3 ,6 ,4 ,2 ,16});
    auto tail = NDArrayFactory::create<double>('c', {1,3}, {0.5,0.5,0.5});
    auto exp = NDArrayFactory::create<double>('c', {4,4}, {9,17.5,12.5,  1.5, 7, 2.5,15.5, 17.5, 15.8,16.4, 3.4,  1.4, 4.3,3.15,1.15,15.15});

    ops::helpers::Householder<double>::mulRight(x, tail, 0.1);

    ASSERT_TRUE(x.isSameShapeStrict(&exp));
    ASSERT_TRUE(x.equalsTo(&exp));

}


/////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, BiDiagonalizeUp_test1) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix = NDArrayFactory::create<double>('c', {4,4}, {9,13,3,6,13,11,7,6,3,7,4,7,6,6,7,10});
    auto hhMatrixExp = NDArrayFactory::create<double>('c', {4,4}, {1.524000,  1.75682,0.233741,0.289458, 0.496646,   1.5655, 1.02929,0.971124, 0.114611,-0.451039, 1.06367,0, 0.229221,-0.272237,0.938237,0});
    auto hhBidiagExp = NDArrayFactory::create<double>('c', {4,4}, {-17.1756, 24.3869,       0,      0, 0,-8.61985,-3.89823,      0, 0,       0, 4.03047,4.13018, 0,       0,       0,1.21666});

    ops::helpers::BiDiagonalUp object(matrix);
    // object._HHmatrix.printBuffer();

    ASSERT_TRUE(hhMatrixExp.isSameShapeStrict(&object._HHmatrix));
    ASSERT_TRUE(hhMatrixExp.equalsTo(&object._HHmatrix));
    ASSERT_TRUE(hhBidiagExp.isSameShapeStrict(&object._HHbidiag));
    ASSERT_TRUE(hhBidiagExp.equalsTo(&object._HHbidiag));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, BiDiagonalizeUp_test2) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix = NDArrayFactory::create<double>('c', {5,4}, {9,-13,3,6, 13,11,7,-6, 3,7,4,7, -6,6,7,10, 2,17,9,12});
    auto hhMatrixExp = NDArrayFactory::create<double>('c', {5,4}, {1.52048, 1.37012, 0.636326, -0.23412, 0.494454, 1.66025,  1.66979,-0.444696, 0.114105,0.130601, 1.58392,        0, -0.22821, 0.215638,0.0524781,  1.99303, 0.0760699,0.375605, 0.509835,0.0591568});
    auto hhBidiagExp = NDArrayFactory::create<double>('c', {4,4}, {-17.2916,7.03123,       0,       0, 0, 16.145,-22.9275,       0, 0,      0, -9.9264,-11.5516, 0,      0,       0,-12.8554});

    ops::helpers::BiDiagonalUp object(matrix);
    // object._HHmatrix.printBuffer();

    ASSERT_TRUE(hhMatrixExp.isSameShapeStrict(&object._HHmatrix));
    ASSERT_TRUE(hhMatrixExp.equalsTo(&object._HHmatrix));
    ASSERT_TRUE(hhBidiagExp.isSameShapeStrict(&object._HHbidiag));
    ASSERT_TRUE(hhBidiagExp.equalsTo(&object._HHbidiag));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, BiDiagonalizeUp_test3) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix = NDArrayFactory::create<double>('c', {6,4}, {9,-13,3,6, 13,11,7,-6, 3,7,4,7, -6,6,7,10, 2,17,9,12, 0,-15,10,2});
    auto hhMatrixExp = NDArrayFactory::create<double>('c', {6,4}, {1.52048,  1.37012, 0.636326, -0.23412, 0.494454,  1.65232,  1.59666,-0.502606, 0.114105, 0.129651,  1.35075,        0, -0.22821, 0.214071, 0.103749,  1.61136, 0.0760699, 0.372875, 0.389936,   0.2398, 0,0.0935171,-0.563777, 0.428587});
    auto hhBidiagExp = NDArrayFactory::create<double>('c', {4,4}, {-17.2916,7.03123,       0,      0, 0,16.3413,-20.7828,      0, 0,      0,-18.4892,4.13261, 0,      0,       0,-21.323});

    ops::helpers::BiDiagonalUp object(matrix);
    // object._HHmatrix.printBuffer();

    ASSERT_TRUE(hhMatrixExp.isSameShapeStrict(&object._HHmatrix));
    ASSERT_TRUE(hhMatrixExp.equalsTo(&object._HHmatrix));
    ASSERT_TRUE(hhBidiagExp.isSameShapeStrict(&object._HHbidiag));
    ASSERT_TRUE(hhBidiagExp.equalsTo(&object._HHbidiag));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test1) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix = NDArrayFactory::create<double>('c', {5,4}, {9,-13,3,6, 13,11,7,-6, 3,7,4,7, -6,6,7,10, 2,17,9,12});
    auto vectorsUseqExp = NDArrayFactory::create<double>('c', {5,4}, {1.52048, 1.37012, 0.636326, -0.23412, 0.494454, 1.66025,  1.66979,-0.444696, 0.114105,0.130601, 1.58392, 0, -0.22821,0.215638,0.0524781,  1.99303, 0.0760699,0.375605, 0.509835,0.0591568});
    auto vectorsVseqExp = NDArrayFactory::create<double>('c', {5,4}, {1.52048, 1.37012, 0.636326, -0.23412, 0.494454, 1.66025,  1.66979,-0.444696, 0.114105,0.130601, 1.58392, 0, -0.22821,0.215638,0.0524781,  1.99303, 0.0760699,0.375605, 0.509835,0.0591568});
    auto coeffsUseqExp = NDArrayFactory::create<double>('c', {4,1}, {1.52048,1.66025,1.58392,1.99303});
    auto coeffsVseqExp = NDArrayFactory::create<double>('c', {3,1}, {1.37012,1.66979,0});

    ops::helpers::BiDiagonalUp object(matrix);
    ops::helpers::HHsequence uSeq = object.makeHHsequence('u');
    ops::helpers::HHsequence vSeq = object.makeHHsequence('v');

    ASSERT_TRUE(uSeq._vectors.isSameShapeStrict(&vectorsUseqExp));
    ASSERT_TRUE(vSeq._vectors.isSameShapeStrict(&vectorsVseqExp));
    ASSERT_TRUE(uSeq._vectors.equalsTo(&vectorsUseqExp));
    ASSERT_TRUE(vSeq._vectors.equalsTo(&vectorsVseqExp));

    ASSERT_TRUE(vSeq._diagSize == uSeq._diagSize - 1);
    ASSERT_TRUE(vSeq._shift == 1);
    ASSERT_TRUE(uSeq._shift == 0);

}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test2) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix = NDArrayFactory::create<double>('c', {6,4}, {9,-13,3,6, 13,11,7,-6, 3,7,4,7, -6,6,7,10, 2,17,9,12 ,0,-15,10,2});
    auto vectorsUseqExp = NDArrayFactory::create<double>('c', {6,4}, {1.52048,  1.37012, 0.636326, -0.23412, 0.494454,  1.65232,  1.59666,-0.502606, 0.114105, 0.129651,  1.35075,        0, -0.22821, 0.214071, 0.103749,  1.61136, 0.0760699, 0.372875, 0.389936,   0.2398, 0,0.0935171,-0.563777, 0.428587});
    auto vectorsVseqExp = NDArrayFactory::create<double>('c', {6,4}, {1.52048,  1.37012, 0.636326, -0.23412, 0.494454,  1.65232,  1.59666,-0.502606, 0.114105, 0.129651,  1.35075,        0, -0.22821, 0.214071, 0.103749,  1.61136, 0.0760699, 0.372875, 0.389936,   0.2398, 0,0.0935171,-0.563777, 0.428587});
    auto coeffsUseqExp = NDArrayFactory::create<double>('c', {4,1}, {1.52048,1.65232,1.35075,1.61136});
    auto coeffsVseqExp = NDArrayFactory::create<double>('c', {3,1}, {1.37012,1.59666,0});

    ops::helpers::BiDiagonalUp object(matrix);
    ops::helpers::HHsequence uSeq = object.makeHHsequence('u');
    ops::helpers::HHsequence vSeq = object.makeHHsequence('v');

    ASSERT_TRUE(uSeq._vectors.isSameShapeStrict(&vectorsUseqExp));
    ASSERT_TRUE(vSeq._vectors.isSameShapeStrict(&vectorsVseqExp));
    ASSERT_TRUE(uSeq._vectors.equalsTo(&vectorsUseqExp));
    ASSERT_TRUE(vSeq._vectors.equalsTo(&vectorsVseqExp));

    ASSERT_TRUE(vSeq._diagSize == uSeq._diagSize - 1);
    ASSERT_TRUE(vSeq._shift == 1);
    ASSERT_TRUE(uSeq._shift == 0);

}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test3) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix = NDArrayFactory::create<double>('c', {4,4}, {9,13,3,6, 13,11,7,6, 3,7,4,7, 6,6,7,10});
    auto vectorsUseqExp = NDArrayFactory::create<double>('c', {4,4}, {1.524,  1.75682,0.233741,0.289458, 0.496646,   1.5655, 1.02929,0.971124, 0.114611,-0.451039, 1.06367,       0, 0.229221,-0.272237,0.938237, 0});
    auto vectorsVseqExp = NDArrayFactory::create<double>('c', {4,4}, {1.524,  1.75682,0.233741,0.289458, 0.496646,   1.5655, 1.02929,0.971124, 0.114611,-0.451039, 1.06367,       0, 0.229221,-0.272237,0.938237, 0});
    auto coeffsUseqExp = NDArrayFactory::create<double>('c', {4,1}, { 1.524, 1.5655,1.06367,0});
    auto coeffsVseqExp = NDArrayFactory::create<double>('c', {3,1}, {1.75682,1.02929, 0});

    ops::helpers::BiDiagonalUp object(matrix);
    ops::helpers::HHsequence uSeq = object.makeHHsequence('u');
    ops::helpers::HHsequence vSeq = object.makeHHsequence('v');

    ASSERT_TRUE(uSeq._vectors.isSameShapeStrict(&vectorsUseqExp));
    ASSERT_TRUE(vSeq._vectors.isSameShapeStrict(&vectorsVseqExp));
    ASSERT_TRUE(uSeq._vectors.equalsTo(&vectorsUseqExp));
    ASSERT_TRUE(vSeq._vectors.equalsTo(&vectorsVseqExp));

    ASSERT_TRUE(vSeq._diagSize == uSeq._diagSize - 1);
    ASSERT_TRUE(vSeq._shift == 1);
    ASSERT_TRUE(uSeq._shift == 0);

}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test4) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix = NDArrayFactory::create<double>('c', {4,4}, {9,13,3,6, 13,11,7,6, 3,7,4,7, 6,6,7,10});
    auto exp    = NDArrayFactory::create<double>('c', {4,4}, {2.49369, 2.62176, 5.88386, 7.69905, -16.0588,-18.7319,-9.15007,-12.6164, 4.7247, 3.46252, 1.02038, -1.4533, 2.9279,-2.29178, 1.90139,-0.66187});

    ops::helpers::BiDiagonalUp object(matrix);
    ops::helpers::HHsequence uSeq = object.makeHHsequence('u');
    uSeq.mulLeft(matrix);

    ASSERT_TRUE(matrix.equalsTo(&exp));

}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test5) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix = NDArrayFactory::create<double>('c', {5,4}, {9,-13,3,6, 13,11,7,-6, 3,7,4,7, -6,6,7,10, 2,17,9,12});
    auto exp    = NDArrayFactory::create<double>('c', {5,4}, {4.52891, 8.09473,-2.73704,-13.0302, -11.0752, 7.41549,-3.75125,0.815252, -7.76818,-15.9102,-9.90869,-11.8677, 1.63942,-17.0312,-9.05102,-4.49088, -9.63311,0.540226,-1.52764, 5.79111});

    ops::helpers::BiDiagonalUp object(matrix);
    ops::helpers::HHsequence uSeq = object.makeHHsequence('u');
    uSeq.mulLeft(matrix);

    ASSERT_TRUE(matrix.equalsTo(&exp));

}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test6) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix = NDArrayFactory::create<double>('c', {5,4}, {9,-13,3,6, 13,11,7,-6, 3,7,4,7, -6,6,7,10, 2,17,9,12});
    auto matrix2 = NDArrayFactory::create<double>('c',{6,4}, {9,-1,3,9, 10,11,-7,-5, 3,2,4,7, -1,6,7,19, 2,17,9,15, 2,17,-9,15});
    auto exp    = NDArrayFactory::create<double>('c', {6,4}, {9,-1,3,9, -4.43019,-15.1713, -3.2854,-7.65743, -9.39162,-7.03599, 8.03827, 9.48453, -2.97785, -16.424, 5.35265,-20.1171, -0.0436177, -13.118,-8.37287,-17.3012, -1.14074, 4.18282,-10.0914,-5.69014});

    ops::helpers::BiDiagonalUp object(matrix);
    ops::helpers::HHsequence uSeq = object.makeHHsequence('u');
    uSeq.mulLeft(matrix2);

    ASSERT_TRUE(matrix2.equalsTo(&exp));

}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test7) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix = NDArrayFactory::create<double>('c', {4,4}, {9,13,3,6, 13,11,7,6, 3,7,4,7, 6,6,7,10});
    auto exp    = NDArrayFactory::create<double>('c', {4,4}, {9,13,3,6,-5.90424,-2.30926,-0.447417, 3.05712, -10.504,-9.31339, -8.85493,-10.8886, -8.29494,-10.6737, -5.94895,-7.55591});

    ops::helpers::BiDiagonalUp object(matrix);
    ops::helpers::HHsequence vSeq = object.makeHHsequence('v');
    vSeq.mulLeft(matrix);

    ASSERT_TRUE(matrix.equalsTo(&exp));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test8) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix = NDArrayFactory::create<double>('c', {5,4}, {9,-13,3,6, 13,11,7,-6, 3,7,4,7, -6,6,7,10, 2,17,9,12});
    auto exp    = NDArrayFactory::create<double>('c', {5,4}, {9,     -13,        3,       6, 13,      11,        7,      -6, -6.90831,-5.01113, 0.381677,0.440128, -0.80107,0.961605,-0.308019,-1.96153, -0.795985, 18.6538,  12.0731, 16.9988});

    ops::helpers::BiDiagonalUp object(matrix);
    ops::helpers::HHsequence vSeq = object.makeHHsequence('v');
    vSeq.mulLeft(matrix);

    ASSERT_TRUE(matrix.equalsTo(&exp));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test9) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix = NDArrayFactory::create<double>('c', {6,4}, {9,-13,3,6, 13,11,7,-6, 3,7,4,7, -6,6,7,10, 2,17,9,12 ,0,-15,10,2});
    auto exp    = NDArrayFactory::create<double>('c', {6,4}, {9,     -13,        3,       6, 13,      11,        7,      -6, 3,       7,        4,       7, 3.77597, 18.6226,-0.674868, 4.61365, 5.02738,-14.1486, -2.22877,-8.98245, -0.683766, 1.73722,  14.9859, 12.0843});

    ops::helpers::BiDiagonalUp object(matrix);
    ops::helpers::HHsequence vSeq = object.makeHHsequence('v');
    vSeq.mulLeft(matrix);

    ASSERT_TRUE(matrix.equalsTo(&exp));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test10) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix = NDArrayFactory::create<double>('c', {4,4}, {9,13,3,6, 13,11,7,6, 3,7,4,7, 6,6,7,10});
    auto matrix2 = NDArrayFactory::create<double>('c',{6,4}, {9,-1,3,9, 10,11,-7,-5, 3,2,4,7, -1,6,7,19, 2,17,9,15, 2,17,-9,15});
    auto exp    = NDArrayFactory::create<double>('c', {6,4}, {9,      -1,       3,        9, 10,      11,      -7,       -5, 3,       2,       4,        7, 2.58863, 11.0295,-4.17483,-0.641012, -1.21892,-16.3151, 6.12049, -20.0239, -0.901799,-15.0389,-12.4944, -20.2394});

    ops::helpers::BiDiagonalUp object(matrix);
    ops::helpers::HHsequence vSeq = object.makeHHsequence('v');
    vSeq.mulLeft(matrix2);

    ASSERT_TRUE(matrix2.equalsTo(&exp));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test11) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix = NDArrayFactory::create<double>('c', {5,4}, {9,-13,3,6, 13,11,7,-6, 3,7,4,7, -6,6,7,10, 2,17,9,12});
    auto matrix2 = NDArrayFactory::create<double>('c',{6,4}, {9,-1,3,9, 10,11,-7,-5, 3,2,4,7, -1,6,7,19, 2,17,9,15, 2,17,-9,15});
    auto exp    = NDArrayFactory::create<double>('c', {6,4}, {9,      -1,       3,       9, 10,      11,      -7,      -5, 3,       2,       4,       7, 1.14934, 4.40257, 8.70127,-1.18824, 1.5132,0.220419,-11.6285,-11.7549, 2.32148, 24.3838,0.256531, 25.9116});

    ops::helpers::BiDiagonalUp object(matrix);
    ops::helpers::HHsequence vSeq = object.makeHHsequence('v');
    vSeq.mulLeft(matrix2);

    ASSERT_TRUE(matrix2.equalsTo(&exp));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test12) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix = NDArrayFactory::create<double>('c', {5,3}, {9,-13,3, 13,11,7, 3,7,4, -6,6,7, 2,17,9});
    auto matrix2 = NDArrayFactory::create<double>('c',{6,4}, {9,-1,3,9, 10,11,-7,-5, 3,2,4,7, -1,6,7,19, 2,17,9,15, 2,17,-9,15});
    auto exp    = NDArrayFactory::create<double>('c', {6,4}, {9,      -1,       3,       9, 10,      11,      -7,      -5, 3,       2,       4,       7, -1,       6,       7,      19, -2.62252,-22.2914, 4.76743,-19.6689, -1.05943,-9.00514,-11.8013,-7.94571});

    ops::helpers::BiDiagonalUp object(matrix);
    ops::helpers::HHsequence vSeq = object.makeHHsequence('v');
    vSeq.mulLeft(matrix2);

    ASSERT_TRUE(matrix2.equalsTo(&exp));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test13) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix = NDArrayFactory::create<double>('c', {5,3}, {9,-13,3, 13,11,7, 3,7,4, -6,6,7, 2,17,9});
    auto matrix2 = NDArrayFactory::create<double>('c',{6,4}, {9,-1,3,9, 10,11,-7,-5, 3,2,4,7, -1,6,7,19, 2,17,9,15, 2,17,-9,15});
    auto exp    = NDArrayFactory::create<double>('c', {6,4}, {9 ,     -1 ,      3 ,      9, -4.65167, 3.44652, 7.83593, 22.6899, -9.48514, -21.902, 5.66559,-13.0533, -0.343184, 15.2895,  7.2888, 14.0489, 0.289638,-1.87752,   3.944,-1.49707, -2.48845, 3.18285,-10.6685,0.406502});

    ops::helpers::BiDiagonalUp object(matrix);
    ops::helpers::HHsequence uSeq = object.makeHHsequence('u');
    uSeq.mulLeft(matrix2);

    ASSERT_TRUE(matrix2.equalsTo(&exp));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test14) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix = NDArrayFactory::create<double>('c', {5,3}, {9,-13,3, 13,11,7, 3,7,4, -6,6,7, 2,17,9});
    auto matrix2 = NDArrayFactory::create<double>('c',{5,5}, {9,-1,3,9,10,  11,-7,-5,3, 2,  4,7,-1,6,7,  19,2,17,9,15, 2,17,-9,15,2});
    auto exp    = NDArrayFactory::create<double>('c', {5,5}, {1.78958,  8.06962,-6.13687, 4.36267, 1.06472, -14.9578,  -8.1522, 1.30442,-18.3343,-13.2578, 13.5536,  5.50764, 15.7859, 7.60831, 11.7871, -1.3626,-0.634986, 7.60934, -2.1841, 5.62694, -13.0577,  15.1554, -7.6511, 3.76365,-5.87368});

    ops::helpers::BiDiagonalUp object(matrix);
    ops::helpers::HHsequence uSeq = object.makeHHsequence('u');
    uSeq.mulLeft(matrix2);

    ASSERT_TRUE(matrix2.equalsTo(&exp));
}


///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test15) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix = NDArrayFactory::create<double>('c', {5,3}, {9,-13,3, 13,11,7, 3,7,4, -6,6,7, 2,17,9});
    auto matrix2 = NDArrayFactory::create<double>('c',{5,5}, {9,-1,3,9,10,  11,-7,-5,3, 2,  4,7,-1,6,7,  19,2,17,9,15, 2,17,-9,15,2});
    auto exp    = NDArrayFactory::create<double>('c', {5,5}, {9,      -1,       3,       9,      10, 11,      -7,      -5,       3,       2, 4,       7,      -1,       6,       7, -9.26566,-16.4298, 1.64125,-17.3243,-7.70257, -16.7077, 4.80216,-19.1652,-2.42279,-13.0258});

    ops::helpers::BiDiagonalUp object(matrix);
    ops::helpers::HHsequence vSeq = object.makeHHsequence('v');
    vSeq.mulLeft(matrix2);

    ASSERT_TRUE(matrix2.equalsTo(&exp));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test16) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix = NDArrayFactory::create<double>('c', {5,5}, {9,-1,3,9,10,  11,-7,-5,3, 2,  4,7,-1,6,7,  19,2,17,9,15, 2,17,-9,15,2});
    auto matrix2 = NDArrayFactory::create<double>('c', {10,10});
    matrix2 = 100.;
    auto exp = NDArrayFactory::create<double>('c',{5,5}, {-0.372742, 0.295145, 0.325359, 0.790947,   0.20615, -0.455573,-0.824221,-0.239444, 0.216163,-0.0951492, -0.165663, 0.285319, -0.18501, 0.130431, -0.916465, -0.7869, 0.245393, 0.116952,-0.541267,  0.117997, -0.0828315, 0.303191,-0.888202, 0.133021,    0.3076});

    ops::helpers::BiDiagonalUp object(matrix);
    ops::helpers::HHsequence uSeq = object.makeHHsequence('u');
    uSeq.applyTo(matrix2);

    ASSERT_TRUE(matrix2.equalsTo(&exp));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test17) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix = NDArrayFactory::create<double>('c', {5,5}, {9,-1,3,9,10,  11,-7,-5,3, 2,  4,7,-1,6,7,  19,2,17,9,15, 2,17,-9,15,2});
    auto matrix2 = NDArrayFactory::create<double>('c', {10,10});
    matrix2 = 100.;
    auto exp = NDArrayFactory::create<double>('c',{5,5}, {1,        0,        0,         0,        0, 0,-0.022902, 0.986163, 0.0411914, 0.158935, 0, -0.44659, 0.021539,  0.797676,-0.404731, 0,-0.554556, 0.103511, -0.600701, -0.56649, 0,-0.701784,-0.127684,-0.0342758, 0.700015});

    ops::helpers::BiDiagonalUp object(matrix);
    ops::helpers::HHsequence vSeq = object.makeHHsequence('v');
    vSeq.applyTo(matrix2);

    ASSERT_TRUE(matrix2.equalsTo(&exp));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test18) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix  = NDArrayFactory::create<double>('c',{6,4}, {9,-1,3,9, 10,11,-7,-5, 3,2,4,7, -1,6,7,19, 2,17,9,15, 2,17,-9,15});
    auto matrix2 = NDArrayFactory::create<double>('c', {10,10});
    matrix2 = 100.;
    auto exp = NDArrayFactory::create<double>('c',{6,6}, {-0.637993,  0.190621,-0.524821,-0.312287, 0.407189, 0.133659, -0.708881, 0.0450803,  0.47462, 0.232701,-0.204602,-0.417348, -0.212664,-0.0405892,-0.297123,0.0240276,-0.821557, 0.435099, 0.0708881, -0.432466, -0.49252,-0.145004,-0.199312,-0.710367, -0.141776,  -0.56468,-0.180549, 0.706094, 0.274317, 0.233707, -0.141776, -0.673865, 0.368567,-0.572848,0.0490246, 0.243733});

    ops::helpers::BiDiagonalUp object(matrix);
    ops::helpers::HHsequence uSeq = object.makeHHsequence('u');
    uSeq.applyTo(matrix2);

    ASSERT_TRUE(matrix2.equalsTo(&exp));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, HHsequence_test19) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix  = NDArrayFactory::create<double>('c',{6,4}, {9,-1,3,9, 10,11,-7,-5, 3,2,4,7, -1,6,7,19, 2,17,9,15, 2,17,-9,15});
    auto matrix2 = NDArrayFactory::create<double>('c', {10,10});
    matrix2 = 100.;
    auto exp = NDArrayFactory::create<double>('c',{4,4}, {1,        0,        0,        0, 0,-0.859586,  0.28601, -0.42345, 0,  0.19328,-0.585133,-0.787567, 0,-0.473027,-0.758826, 0.447693});

    ops::helpers::BiDiagonalUp object(matrix);
    ops::helpers::HHsequence vSeq = object.makeHHsequence('v');
    vSeq.applyTo(matrix2);

    ASSERT_TRUE(matrix2.equalsTo(&exp));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, SVD_test1) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix  = NDArrayFactory::create<double>('c', {5,5}, {-17 ,14 ,9 ,-12 ,-12 ,5 ,-4 ,-19 ,-7 ,-12 ,15 ,16 ,17 ,-6 ,8 ,-10 ,14 ,-15 ,6 ,-10 ,-14 ,12 ,-1 ,-16 ,3});
    auto matrix2 = NDArrayFactory::create<double>('c', {5,5}, {18 ,3 ,2 ,7 ,-11 ,7 ,7 ,10 ,-13 ,-8 ,13 ,20 ,-4 ,-16 ,-9 ,-17 ,-5 ,-7 ,-19 ,-8 ,-9 ,9 ,6 ,14 ,-11});
    auto expM  = NDArrayFactory::create<double>('c', {5,5}, {-17,14,9,-12,-12, 5,-4,    -19, -7,-12, 15,16,17.0294, -6,  8, -10,14,    -15,  6,-10, -14,12,      0,-16,  0});
    auto expU  = NDArrayFactory::create<double>('c', {5,5}, {18,3, 2,7,-11, 7, 7.75131,10,-12.5665, -8, 13,  20.905,-4,-14.7979, -9, -17,-3.87565,-7,-19.2608, -8, -9,       9, 6,      14,-11});

    ops::helpers::SVD<double> svd(matrix, 4, true, true, true, 't');
    svd._m = matrix;
    svd._u = matrix2;
    svd.deflation1(1,1,2,2);

    ASSERT_TRUE(expM.equalsTo(&svd._m));
    ASSERT_TRUE(expU.equalsTo(&svd._u));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, SVD_test2) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix= NDArrayFactory::create<double>('c', {5,5}, {-17 ,14 ,9 ,-12 ,-12 ,5 ,-4 ,-19 ,-7 ,-12 ,15 ,16 ,17 ,-6 ,8 ,-10 ,14 ,-15 ,6 ,-10 ,-14 ,12 ,-1 ,-16 ,3});
    auto matrix2 = NDArrayFactory::create<double>('c', {5,5}, {18 ,3 ,2 ,7 ,-11 ,7 ,7 ,10 ,-13 ,-8 ,13 ,20 ,-4 ,-16 ,-9 ,-17 ,-5 ,-7 ,-19 ,-8 ,-9 ,9 ,6 ,14 ,-11});
    auto expM  = NDArrayFactory::create<double>('c', {5,5}, {22.6716,14,  9,-12,-12, 5,-4,-19, -7,-12, 0,16,  0, -6,  8, -10,14,-15,  6,-10, -14,12, -1,-16,  3});
    auto expU  = NDArrayFactory::create<double>('c', {5,5}, {-12.1738, 3, -13.4089,  7,-11, 1.36735, 7, -12.1297,-13, -8, -12.3944,20, -5.60173,-16, -9, -17,-5,-7,-19, -8, -9, 9, 6, 14,-11});

    ops::helpers::SVD<double> svd(matrix, 4, true, true, true);
    svd._m = matrix;
    svd._u = matrix2;
    svd.deflation1(0,0,2,2);

    ASSERT_TRUE(expM.equalsTo(&svd._m));
    ASSERT_TRUE(expU.equalsTo(&svd._u));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, SVD_test3) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix= NDArrayFactory::create<double>('c', {5,5}, {-17 ,14 ,9 ,-12 ,-12 ,5 ,-4 ,-19 ,-7 ,-12 ,15 ,16 ,17 ,-6 ,8 ,-10 ,14 ,-15 ,6 ,-10 ,-14 ,12 ,-1 ,-16 ,3});
    auto matrix2 = NDArrayFactory::create<double>('c', {2,6}, {18 ,3 ,2 ,7 ,-11 ,7 ,7 ,10 ,-13 ,-8 ,13 ,20});
    auto expM  = NDArrayFactory::create<double>('c', {5,5}, {-17,14,9,-12,-12, 5,-4,    -19, -7,-12, 15,16,17.0294, -6,  8, -10,14,    -15,  6,-10, -14,12,      0,-16,  0});
    auto expU  = NDArrayFactory::create<double>('c', {2,6}, {18, 2.58377,   2,  7.16409,-11,  7, 7 ,10.4525 ,-13, -7.39897 ,13 ,20});

    ops::helpers::SVD<double> svd(matrix, 4, false, true, true, 't');
    svd._m = matrix;
    svd._u = matrix2;
    svd.deflation1(1,1,2,2);

    ASSERT_TRUE(expM.equalsTo(&svd._m));
    ASSERT_TRUE(expU.equalsTo(&svd._u));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, SVD_test4) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix1 = NDArrayFactory::create<double>('c', {6,5}, {12 ,20 ,19 ,-18 ,-6 ,3 ,6 ,2 ,-7 ,-7 ,14 ,8 ,18 ,-17 ,18 ,-14 ,-15 ,1 ,2 ,2 ,-3 ,-18 ,8 ,-17 ,-19 ,12 ,18 ,6 ,-2 ,-17});
    auto matrix2 = NDArrayFactory::create<double>('c', {6,6}, {-10 ,-16 ,-20 ,13 ,20 ,-10 ,-9 ,-1 ,-7 ,-20 ,-4 ,20 ,-11 ,19 ,-5 ,-18 ,12 ,-19 ,18 ,-18 ,17 ,-10 ,-19 ,14 ,-2 ,-7 ,-17 ,-14 ,-4 ,-16 ,18 ,-6 ,-18 ,1 ,-15 ,-12});
    auto matrix3 = NDArrayFactory::create<double>('c', {5,5}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0});
    auto expM  = NDArrayFactory::create<double>('c', {6,5}, {12, 20,     19,-18, -6, 3,  6,      2, -7, -7, 14,  8,     18,-17, 18, -14,-15,8.06226,  2,  2, -3,-18,      0,-17,  2, 12, 18,      6, -2,-17});
    auto expU  = NDArrayFactory::create<double>('c', {6,6}, {-10,-16,     -20,     13, 20,-10, -9, -1,-20.7138,4.46525, -4, 20, -11, 19,-18.4812,2.72876, 12,-19, 18,-18,      17,    -10,-19, 14, -2, -7,     -17,    -14, -4,-16, 18, -6,     -18,      1,-15,-12});
    auto expV  = NDArrayFactory::create<double>('c', {5,5}, {-18,  1,     19,      -7, 1, 2,-18,    -13,      14, 2, -2,-11,2.97683,-7.69015,-6, -3, -8,      8,      -2, 7, 16, 15,     -3,       7, 0});

    ops::helpers::SVD<double> svd(matrix3, 4, true, true, true, 't');
    svd._m = matrix1;
    svd._u = matrix2;
    svd._v = matrix3;
    svd.deflation2(1, 2, 2, 1, 1, 2, 1);

    ASSERT_TRUE(expM.equalsTo(&svd._m));
    ASSERT_TRUE(expU.equalsTo(&svd._u));
    ASSERT_TRUE(expV.equalsTo(&svd._v));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, SVD_test5) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix1 = NDArrayFactory::create<double>('c', {6,5}, {12 ,20 ,19 ,-18 ,-6 ,3 ,6 ,2 ,-7 ,-7 ,14 ,8 ,18 ,-17 ,18 ,-14 ,-15 ,1 ,2 ,2 ,-3 ,-18 ,8 ,-17 ,-19 ,12 ,18 ,6 ,-2 ,-17});
    auto matrix2 = NDArrayFactory::create<double>('c', {6,6}, {-10 ,-16 ,-20 ,13 ,20 ,-10 ,-9 ,-1 ,-7 ,-20 ,-4 ,20 ,-11 ,19 ,-5 ,-18 ,12 ,-19 ,18 ,-18 ,17 ,-10 ,-19 ,14 ,-2 ,-7 ,-17 ,-14 ,-4 ,-16 ,18 ,-6 ,-18 ,1 ,-15 ,-12});
    auto matrix3 = NDArrayFactory::create<double>('c', {5,5}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0});
    auto expM  = NDArrayFactory::create<double>('c', {6,5}, {18.4391, 20,     19,-18, -6, 3,  6,      2, -7, -7, 0,  8,18.4391,-17, 18, -14,-15,      1,  2,  2, -3,-18,      8,-17,-19, 12, 18,      6, -2,-17});
    auto expU  = NDArrayFactory::create<double>('c', {6,6}, {-10,-16,-20,13, 20,-10, -9,-15.8359, -7,-12.2566, -4, 20, -11,-1.30158, -5,-26.1401, 12,-19, 18,-19.3068, 17, 7.15871,-19, 14, -2,      -7,-17,     -14, -4,-16, 18,      -6,-18,       1,-15,-12});
    auto expV  = NDArrayFactory::create<double>('c', {5,5}, {-18,       1, 19,     -7, 1, 2,-1.08465,-13,22.7777, 2, -2,-5.64019,  8,9.65341,-6, -3,      -8,  8,     -2, 7, 16,      15, -3,      7, 0});

    ops::helpers::SVD<double> svd(matrix3, 4, true, true, true, 't');
    svd._m = matrix1;
    svd._u = matrix2;
    svd._v = matrix3;
    svd.deflation2(1, 0, 1, 1, 0, 2, 2);

    ASSERT_TRUE(expM.equalsTo(&svd._m));
    ASSERT_TRUE(expU.equalsTo(&svd._u));
    ASSERT_TRUE(expV.equalsTo(&svd._v));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, SVD_test6) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix1 = NDArrayFactory::create<double>('c', {6,5}, {12 ,20 ,19 ,-18 ,-6 ,3 ,6 ,2 ,-7 ,-7 ,14 ,8 ,18 ,-17 ,18 ,-14 ,-15 ,1 ,2 ,2 ,-3 ,-18 ,8 ,-17 ,-19 ,12 ,18 ,6 ,-2 ,-17});
    auto matrix2 = NDArrayFactory::create<double>('c', {2,6}, {-10 ,-16 ,-20 ,13 ,20 ,-10 ,-9 ,-1 ,-7 ,-20 ,-4 ,20});
    auto matrix3 = NDArrayFactory::create<double>('c', {5,5}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0});
    auto expM  = NDArrayFactory::create<double>('c', {6,5}, {18.4391, 20,     19,-18, -6, 3,  6,      2, -7, -7, 0,  8,18.4391,-17, 18, -14,-15,      1,  2,  2, -3,-18,      8,-17,-19, 12, 18,      6, -2,-17});
    auto expU  = NDArrayFactory::create<double>('c', {2,6}, {-10, -0.542326,-20, 20.6084,20,-10, -9,  -15.8359, -7,-12.2566,-4, 20});
    auto expV  = NDArrayFactory::create<double>('c', {5,5}, {-18,       1, 19,     -7, 1, 2,-1.08465,-13,22.7777, 2, -2,-5.64019,  8,9.65341,-6, -3,      -8,  8,     -2, 7, 16,      15, -3,      7, 0});

    ops::helpers::SVD<double> svd(matrix3, 4, false, true, true, 't');
    svd._m = matrix1;
    svd._u = matrix2;
    svd._v = matrix3;
    svd.deflation2(1, 0, 1, 1, 0, 2, 2);

    ASSERT_TRUE(expM.equalsTo(&svd._m));
    ASSERT_TRUE(expU.equalsTo(&svd._u));
    ASSERT_TRUE(expV.equalsTo(&svd._v));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, SVD_test7) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix1 = NDArrayFactory::create<double>('c', {6,5}, {12 ,20 ,19 ,-18 ,-6 ,3 ,6 ,2 ,-7 ,-7 ,14 ,8 ,18 ,-17 ,18 ,-14 ,-15 ,1 ,2 ,2 ,-3 ,-18 ,8 ,-17 ,-19 ,12 ,18 ,6 ,-2 ,-17});
    auto matrix2 = NDArrayFactory::create<double>('c', {6,6}, {-10 ,-16 ,-20 ,13 ,20 ,-10 ,-9 ,-1 ,-7 ,-20 ,-4 ,20 ,-11 ,19 ,-5 ,-18 ,12 ,-19 ,18 ,-18 ,17 ,-10 ,-19 ,14 ,-2 ,-7 ,-17 ,-14 ,-4 ,-16 ,18 ,-6 ,-18 ,1 ,-15 ,-12});
    auto matrix3 = NDArrayFactory::create<double>('c', {5,5}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0});

    auto expM  = NDArrayFactory::create<double>('c', {6,5}, {12, 20,     19,-18, -6, 3,  6,      2, -7, -7, 14,  8,19.6977,-17, 18, -14,-15,      1,  2,  2, -3,-18,      0,-17,  0, 12, 18,      6, -2,-17});
    auto expU  = NDArrayFactory::create<double>('c', {6,6}, {-10,     -16,-20,      13, 20,-10, -9,-9.03658, -7,-17.8701, -4, 20, -11, 10.0519, -5,-24.1652, 12,-19, 18,  -20.51, 17,-1.82762,-19, 14, -2,-12.0826,-17,-9.95039, -4,-16, 18,      -6,-18,       1,-15,-12});
    auto expV  = NDArrayFactory::create<double>('c', {5,5}, {-18,  1, 19,-7, 1, 2,-18,-13,14, 2, -2,-11,  8, 2,-6, -3, -8,  8,-2, 7, 16, 15, -3, 7, 0});

    ops::helpers::SVD<double> svd(matrix3, 4, true, true, true, 't');
    svd._m = matrix1;
    svd._u = matrix2;
    svd._v = matrix3;
    svd.deflation(1, 3, 1, 1, 2, 1);

    ASSERT_TRUE(expM.equalsTo(&svd._m));
    ASSERT_TRUE(expU.equalsTo(&svd._u));
    ASSERT_TRUE(expV.equalsTo(&svd._v));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, SVD_test8) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix1 = NDArrayFactory::create<double>('c', {6,5}, {12 ,20 ,19 ,-18 ,-6 ,3 ,6 ,2 ,-7 ,-7 ,14 ,8 ,18 ,-17 ,18 ,-14 ,-15 ,1 ,2 ,2 ,-3 ,-18 ,8 ,-17 ,-19 ,12 ,18 ,6 ,-2 ,-17});
    auto matrix2 = NDArrayFactory::create<double>('c', {6,6}, {-10 ,-16 ,-20 ,13 ,20 ,-10 ,-9 ,-1 ,-7 ,-20 ,-4 ,20 ,-11 ,19 ,-5 ,-18 ,12 ,-19 ,18 ,-18 ,17 ,-10 ,-19 ,14 ,-2 ,-7 ,-17 ,-14 ,-4 ,-16 ,18 ,-6 ,-18 ,1 ,-15 ,-12});
    auto matrix3 = NDArrayFactory::create<double>('c', {5,5}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0});

    auto expM  = NDArrayFactory::create<double>('c', {6,5}, {12, 20,19,-18, -6, 3,  6, 2, -7, -7, 14,-15, 2,-17, 18, -14,  8, 1, 18,  2, -3,-18, 8,-17,-19, 12, 18, 6, -2,-17});
    auto expU  = NDArrayFactory::create<double>('c', {6,6}, {-10,-20,-16, 13, 20,-10, -9, -7, -1,-20, -4, 20, -11, -5, 19,-18, 12,-19, 18, 17,-18,-10,-19, 14, -2, -7,-17,-14, -4,-16, 18, -6,-18,  1,-15,-12});
    auto expV  = NDArrayFactory::create<double>('c', {5,5}, {-18,  1, 19,-7, 1, 2,-18,-13, 2,14, -2,-11,  8,-6, 2, -3, -8,  8, 7,-2, 16, 15, -3, 7, 0});

    ops::helpers::SVD<double> svd(matrix3, 4, true, true, true, 't');
    svd._m = matrix1;
    svd._u = matrix2;
    svd._v = matrix3;
    svd.deflation(0, 2, 2, 1, 2, 1);

    ASSERT_TRUE(expM.equalsTo(&svd._m));
    ASSERT_TRUE(expU.equalsTo(&svd._u));
    ASSERT_TRUE(expV.equalsTo(&svd._v));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, SVD_test9) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto col0 = NDArrayFactory::create<double>('c', {10,1}, {12 ,20 ,19 ,-18 ,-6 ,3 ,6 ,2 ,-7 ,14});
    auto diag = NDArrayFactory::create<double>('c', {10,1}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2});
    auto permut = NDArrayFactory::create<double>('c', {1,10}, {8 ,1 ,4 ,0, 5 ,2 ,9 ,3 ,7 ,6});
    auto matrix3 = NDArrayFactory::create<double>('c', {5,5}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0});

    auto expSingVals = NDArrayFactory::create<double>('c', {10,1}, {-2, 15.304323, 11.2, -1, 1.73489, -12, -15.3043, -12.862, 5.6, 41.4039});
    auto expShifts = NDArrayFactory::create<double>('c', {10,1}, {1, 19, 19, 1, 2, -18, -18, -13, 2, 2});
    auto expMus    = NDArrayFactory::create<double>('c', {10,1}, {-3, -3.695677, -7.8, -2, -0.265108, 6, 2.69568, 0.138048, 3.6, 39.4039});

    auto singVals = NDArrayFactory::create<double>('c', {10,1});
    auto shifts = NDArrayFactory::create<double>('c', {10,1});
    auto mus    = NDArrayFactory::create<double>('c', {10,1});

    ops::helpers::SVD<double> svd(matrix3, 4, true, true, true, 't');
    svd.calcSingVals(col0, diag, permut, singVals, shifts, mus);

    ASSERT_TRUE(expSingVals.equalsTo(&singVals));
    ASSERT_TRUE(expShifts.equalsTo(&shifts));
    ASSERT_TRUE(expMus.equalsTo(&mus));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, SVD_test10) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto singVals = NDArrayFactory::create<double>('c', {4,1}, {1 ,1 ,1 ,1});
    auto col0 = NDArrayFactory::create<double>('c', {4,1}, {1 ,1 ,1 ,1});
    auto diag = NDArrayFactory::create<double>('c', {4,1}, {5 ,7 ,-13 ,14});
    auto permut = NDArrayFactory::create<double>('c', {1,4}, {0 ,2 ,3 ,1 });
    auto mus  = NDArrayFactory::create<double>('c', {4,1}, {4,1,4,6});
    auto shifts = NDArrayFactory::create<double>('c', {4,1}, {4,2,5,6});
    auto matrix3 = NDArrayFactory::create<double>('c', {5,5}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0});

    auto expZhat = NDArrayFactory::create<double>('c', {4,1}, {0, 0.278208, 72.501953, 0});

    auto zhat = NDArrayFactory::create<double>('c', {4,1});

    ops::helpers::SVD<double> svd(matrix3, 4, true, true, true, 't');
    svd.perturb(col0, diag, permut, singVals, shifts,  mus, zhat);

    ASSERT_NEAR(expZhat.e<double>(1), zhat.e<double>(1), EPS);
    ASSERT_NEAR(expZhat.e<double>(2), zhat.e<double>(2), EPS);
}


///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, SVD_test11) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto singVals = NDArrayFactory::create<double>('c', {4,1}, {1 ,1 ,1 ,1});
    auto zhat   = NDArrayFactory::create<double>('c', {4,1}, {2 ,1 ,2 ,1});
    auto diag = NDArrayFactory::create<double>('c', {4,1}, {5 ,7 ,-13 ,14});
    auto permut = NDArrayFactory::create<double>('c', {1,4}, {0 ,2 ,3 ,1 });
    auto mus  = NDArrayFactory::create<double>('c', {4,1}, {4,1,4,6});
    auto shifts = NDArrayFactory::create<double>('c', {4,1}, {4,2,5,6});
    auto matrix3 = NDArrayFactory::create<double>('c', {5,5}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0});

    auto expU = NDArrayFactory::create<double>('c', {5,5}, {-0.662161, 0.980399,-0.791469,-0.748434, 0, -0.744931, 0.183825,-0.593602,-0.392928, 0, 0.0472972, 0.061275,0.0719517, 0.104781, 0, 0.0662161,0.0356509, 0.126635, 0.523904, 0, 0,        0,        0,        0, 1});
    auto expV = NDArrayFactory::create<double>('c', {4,4}, {-0.745259,-0.965209, -0.899497, -0.892319, -0.652102,  0.21114,  -0.39353, -0.156156, -0.0768918,-0.130705,-0.0885868,-0.0773343, 0.115929,0.0818966,  0.167906,  0.416415});
    auto U = NDArrayFactory::create<double>('c', {5,5});
    auto V = NDArrayFactory::create<double>('c', {4,4});


    ops::helpers::SVD<double> svd(matrix3, 4, true, true, true, 't');
    svd.calcSingVecs(zhat, diag,permut, singVals, shifts, mus, U, V);

    ASSERT_TRUE(expU.equalsTo(&U));
    ASSERT_TRUE(expV.equalsTo(&V));

}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, SVD_test12) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix1 = NDArrayFactory::create<double>('c', {6,5}, {-2 ,-3 ,2 ,1 ,0 ,0 ,-4 ,5 ,-2 ,-3 ,-4 ,0 ,5 ,-1 ,-5 ,-3 ,-5 ,3 ,3 ,3 ,-5 ,5 ,-5 ,0 ,2 ,-2 ,-3 ,-4 ,-5 ,-3});
    auto matrix2 = NDArrayFactory::create<double>('c', {6,6}, {-10 ,-16 ,-20 ,13 ,20 ,-10 ,-9 ,-1 ,-7 ,-20 ,-4 ,20 ,-11 ,19 ,-5 ,-18 ,12 ,-19 ,18 ,-18 ,17 ,-10 ,-19 ,14 ,-2 ,-7 ,-17 ,-14 ,-4 ,-16 ,18 ,-6 ,-18 ,1 ,-15 ,-12});
    auto matrix3 = NDArrayFactory::create<double>('c', {5,5}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0});
    auto matrix4 = NDArrayFactory::create<double>('c', {5,5}, {3 ,-8 ,5 ,7 ,-8 ,4 ,-19 ,-12 ,-4 ,-5 ,-11 ,19 ,-2 ,-7 ,1 ,16 ,-5 ,10 ,19 ,-19 ,0 ,-20 ,0 ,-8 ,-13});

    auto expSingVals = NDArrayFactory::create<double>('c', {4,1}, {8.43282, 5, 2.3, 1.10167});
    auto expU  = NDArrayFactory::create<double>('c', {5,5}, {0.401972,0, 0.206791, 0.891995,0, 0,1,        0,        0,0, 0.816018,0,-0.522818,-0.246529,0, -0.415371,0,-0.826982, 0.378904,0, 0,0,        0,        0,1});
    auto expV  = NDArrayFactory::create<double>('c', {4,4}, {-0.951851,0,-0.133555,-0.275939, 0,1,        0,        0, 0.290301,0,-0.681937,-0.671333, -0.098513,0,-0.719114, 0.687873});

    ops::helpers::SVD<double> svd(matrix4, 4, true, true, true, 't');
    svd._m = matrix1;
    svd._u = matrix2;
    svd._v = matrix3;
    NDArray U, singVals, V;
    svd.calcBlockSVD(1, 4, U, singVals, V);

    ASSERT_TRUE(expSingVals.equalsTo(&singVals));
    ASSERT_TRUE(expU.equalsTo(&U));
    ASSERT_TRUE(expV.equalsTo(&V));

    ASSERT_TRUE(expSingVals.isSameShapeStrict(&singVals));
    ASSERT_TRUE(expU.isSameShapeStrict(&U));
    ASSERT_TRUE(expV.isSameShapeStrict(&V));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, SVD_test13) {

    #ifdef __CUDABLAS__
    return;
    #endif
    NDArray matrix1('c', {6,5}, {12 ,20 ,19 ,-18 ,-6 ,3 ,6 ,2 ,-7 ,-7 ,14 ,8 ,18 ,-17 ,18 ,-14 ,-15 ,1 ,2 ,2 ,-3 ,-18 ,8 ,-17 ,-19 ,12 ,18 ,6 ,-2 ,-17});

    auto expQR = NDArrayFactory::create<double>('c', {6,5}, {-37.054 ,  0.323852 , 8.04231 , -22.9395 ,-13.089, 0.105164,    32.6021,  6.42277, -0.262898,-1.58766, 0.140218,  -0.485058,  29.2073,  -9.92301,-23.7111, -0.262909,-0.00866538, 0.103467,   8.55831,-1.86455, -0.315491,   0.539207,  0.40754,-0.0374124,-7.10401, 0.315491,   0.385363,-0.216459, -0.340008,0.628595});
    auto expCoeffs = NDArrayFactory::create<double>('c', {1,5}, {1.53975, 1.19431, 1.63446, 1.7905, 1.43356});
    auto expPermut = NDArrayFactory::create<double>('c', {5,5}, {0,0,0,1,0, 1,0,0,0,0, 0,0,0,0,1, 0,0,1,0,0, 0,1,0,0,0});

    ops::helpers::HHcolPivQR qr(matrix1);

    ASSERT_TRUE(expQR.equalsTo(&qr._qr));
    ASSERT_TRUE(expCoeffs.equalsTo(&qr._coeffs));
    ASSERT_TRUE(expPermut.equalsTo(&qr._permut));

    ASSERT_TRUE(expQR.isSameShapeStrict(&qr._qr));
    ASSERT_TRUE(expCoeffs.isSameShapeStrict(&qr._coeffs));
    ASSERT_TRUE(expPermut.isSameShapeStrict(&qr._permut));

}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, SVD_test14) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix1 = NDArrayFactory::create<double>('c', {5,6}, {12 ,20 ,19 ,-18 ,-6 ,3 ,6 ,2 ,-7 ,-7 ,14 ,8 ,18 ,-17 ,18 ,-14 ,-15 ,1 ,2 ,2 ,-3 ,-18 ,8 ,-17 ,-19 ,12 ,18 ,6 ,-2 ,-17});

    auto expQR = NDArrayFactory::create<double>('c', {5,6}, {-32.665, -4.95944,  -8.26574,  7.22487, 16.5927, 11.7251, -0.135488, -29.0586,   10.9776, -14.6886, 4.18841, 20.7116, 0.348399, 0.323675,   25.5376,  1.64324, 9.63959, -9.0238, -0.0580664,0.0798999,-0.0799029,  19.5281,-4.97736, 16.0969, 0.348399,-0.666783, 0.0252425,0.0159188, 10.6978,-4.69198});
    auto expCoeffs = NDArrayFactory::create<double>('c', {1,5}, {1.58166, 1.28555, 1.98605, 1.99949, 0});
    auto expPermut = NDArrayFactory::create<double>('c', {6,6}, {0,1,0,0,0,0, 0,0,1,0,0,0, 1,0,0,0,0,0, 0,0,0,0,0,1, 0,0,0,0,1,0, 0,0,0,1,0,0});

    ops::helpers::HHcolPivQR qr(matrix1);

    ASSERT_TRUE(expQR.equalsTo(&qr._qr));
    ASSERT_TRUE(expCoeffs.equalsTo(&qr._coeffs));
    ASSERT_TRUE(expPermut.equalsTo(&qr._permut));

    ASSERT_TRUE(expQR.isSameShapeStrict(&qr._qr));
    ASSERT_TRUE(expCoeffs.isSameShapeStrict(&qr._coeffs));
    ASSERT_TRUE(expPermut.isSameShapeStrict(&qr._permut));
}


///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, SVD_test15) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix1 = NDArrayFactory::create<double>('c', {6,6}, {-10 ,-16 ,-20 ,13 ,20 ,-10 ,-9 ,-1 ,-7 ,-20 ,-4 ,20 ,-11 ,19 ,-5 ,-18 ,12 ,-19 ,18 ,-18 ,17 ,-10 ,-19 ,14 ,-2 ,-7 ,-17 ,-14 ,-4 ,-16 ,18 ,-6 ,-18 ,1 ,-15 ,-12});

    auto expQR = NDArrayFactory::create<double>('c', {6,6}, {38.1707, -3.03898, 5.16103,  23.0805, -7.57126, -13.885, -0.41519,  34.3623, 3.77403,  2.62327, -8.17784, 9.10312, 0.394431, 0.509952,-30.2179, -6.78341,  12.8421, 28.5491, -0.290633, 0.111912,0.450367,  28.1139,  15.5195, 2.60562, 0.332152, 0.405161,0.308163,0.0468127,   22.294,-2.94931, 0.249114,0.0627956,0.657873,  0.76767,-0.752594,-7.46986});
    auto expCoeffs = NDArrayFactory::create<double>('c', {1,6}, {1.26198, 1.38824, 1.15567, 1.25667, 1.27682, 0});
    auto expPermut = NDArrayFactory::create<double>('c', {6,6}, {0,0,1,0,0,0, 0,0,0,0,1,0, 0,0,0,1,0,0, 0,1,0,0,0,0, 0,0,0,0,0,1, 1,0,0,0,0,0});

    ops::helpers::HHcolPivQR qr(matrix1);

    ASSERT_TRUE(expQR.equalsTo(&qr._qr));
    ASSERT_TRUE(expCoeffs.equalsTo(&qr._coeffs));
    ASSERT_TRUE(expPermut.equalsTo(&qr._permut));

    ASSERT_TRUE(expQR.isSameShapeStrict(&qr._qr));
    ASSERT_TRUE(expCoeffs.isSameShapeStrict(&qr._coeffs));
    ASSERT_TRUE(expPermut.isSameShapeStrict(&qr._permut));
}


///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, JacobiSVD_test1) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix3 = NDArrayFactory::create<double>('c', {5,5}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0});
    auto left = NDArrayFactory::create<double>('c',  {2,2});
    auto right = NDArrayFactory::create<double>('c', {2,2});

    auto expLeft = NDArrayFactory::create<double>('c', {2,2}, {0.972022, 0.23489, -0.23489, 0.972022});
    auto expRight = NDArrayFactory::create<double>('c', {2,2}, {0.827657, 0.561234, -0.561234, 0.827657});

    ops::helpers::JacobiSVD<double>::svd2x2(matrix3, 1, 3, left, right);

    ASSERT_TRUE(expLeft.equalsTo(&left));
    ASSERT_TRUE(expRight.equalsTo(&right));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, JacobiSVD_test2) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix3 = NDArrayFactory::create<double>('c', {5,5}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0});
    auto matrix4 = NDArrayFactory::create<double>('c', {5,5}, {12 ,20 ,19 ,-18 ,-6 ,3 ,6 ,2 ,-7 ,-7 ,14 ,8 ,18 ,-17 ,18 ,-14 ,-15 ,1 ,2 ,2 ,-3 ,-18 ,8 ,-17 ,-19});
    auto matrix5 = NDArrayFactory::create<double>('c', {5,5}, {3 ,-8 ,5 ,7 ,-8 ,4 ,-19 ,-12 ,-4 ,-5 ,-11 ,19 ,-2 ,-7 ,1 ,16 ,-5 ,10 ,19 ,-19 ,0 ,-20 ,0 ,-8 ,-13});

    auto exp3 = NDArrayFactory::create<double>('c', {5,5}, {-18,      1,      19,      -7,       1, -0.609208,19.6977, 8.63044,-11.9811,-4.67059, -2,    -11,       8,       2,      -6, 3.55371,      0,-12.5903, 7.51356, -5.5844, 16,     15,      -3,       7,       0});
    auto exp4 = NDArrayFactory::create<double>('c', {5,5}, {12, -10.9657,19,24.5714, -6, 3,  -2.6399, 2,8.83351, -7, 14,-0.406138,18,18.7839, 18, -14,  12.8949, 1,-7.9197,  2, -3,   23.353, 8, 8.2243,-19});
    auto exp5 = NDArrayFactory::create<double>('c', {5,5}, {3 ,-8 ,5 ,7 ,-8 ,4 ,-19 ,-12 ,-4 ,-5 ,-11 ,19 ,-2 ,-7 ,1 ,16 ,-5 ,10 ,19 ,-19 ,0 ,-20 ,0 ,-8 ,-13});

    ops::helpers::JacobiSVD<double> jac(matrix3, true, true, true);
    jac._m = matrix3;
    jac._u = matrix4;
    jac._v = matrix5;

    double maxElem;
    bool result = jac.isBlock2x2NotDiag(matrix3, 1, 3, maxElem);

    // ASSERT_NEAR(maxElem, 19.69772, 1e-5);
    ASSERT_TRUE(exp3.equalsTo(&matrix3));
    ASSERT_TRUE(exp4.equalsTo(&jac._u));
    ASSERT_TRUE(exp5.equalsTo(&jac._v));

    ASSERT_TRUE(result);
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, JacobiSVD_test3) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix = NDArrayFactory::create<double>('c', {5,5}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0});
    auto rotation = NDArrayFactory::create<double>('c', {2,2}, {0.2, math::nd4j_sqrt<double, double>(0.6), -math::nd4j_sqrt<double, double>(0.6), 0.2});

    auto expected = NDArrayFactory::create<double>('c', {5,5}, {-18,       1,     19,      -7,       1, -1.14919,-12.1206,3.59677, 4.34919,-4.24758, -1.94919, 11.7427,11.6698,-10.4444,-2.74919, -3,      -8,      8,      -2,       7, 16,      15,     -3,       7,       0});

    ops::helpers::JacobiSVD<double>::mulRotationOnLeft(1, 2, matrix, rotation);

    ASSERT_TRUE(expected.equalsTo(&matrix));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, JacobiSVD_test4) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix = NDArrayFactory::create<double>('c', {5,5}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0});
    auto rotation = NDArrayFactory::create<double>('c', {2,2}, {0.2, math::nd4j_sqrt<double, double>(0.6), -math::nd4j_sqrt<double, double>(0.6), 0.2});

    auto expected = NDArrayFactory::create<double>('c', {5,5}, {-18,       1,      19,     -7,       1, 1.94919, 4.92056,-8.79677,1.25081, 5.04758, 1.14919,-16.1427,-8.46976,11.2444,0.349193, -3,      -8,       8,     -2,       7, 16,      15,      -3,      7,       0});

    ops::helpers::JacobiSVD<double>::mulRotationOnLeft(2, 1, matrix, rotation);

    ASSERT_TRUE(expected.equalsTo(&matrix));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, JacobiSVD_test5) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix = NDArrayFactory::create<double>('c', {5,5}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0});
    auto rotation = NDArrayFactory::create<double>('c', {2,2}, {0.2, math::nd4j_sqrt<double, double>(0.6), -math::nd4j_sqrt<double, double>(0.6), 0.2});

    auto expected = NDArrayFactory::create<double>('c', {5,5}, {-18,      1,      19,      -7,       1, 2,    -18,     -13,      14,       2, 1.14919,6.32056,-4.59677,-1.14919, 3.44758, -3,     -8,       8,      -2,       7, 16,     15,      -3,       7,       0});

    ops::helpers::JacobiSVD<double>::mulRotationOnLeft(2, 2, matrix, rotation);

    ASSERT_TRUE(expected.equalsTo(&matrix));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, JacobiSVD_test6) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix = NDArrayFactory::create<double>('c', {5,5}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0});
    auto rotation = NDArrayFactory::create<double>('c', {2,2}, {0.2, math::nd4j_sqrt<double, double>(0.6), -math::nd4j_sqrt<double, double>(0.6), 0.2});

    auto expected = NDArrayFactory::create<double>('c', {5,5}, {-18,-14.5173,  4.5746,-7, 1, 2, 6.46976,-16.5427,14, 2, -2,-8.39677,-6.92056, 2,-6, -3,-7.79677,-4.59677,-2, 7, 16, 5.32379,  11.019, 7, 0});

    ops::helpers::JacobiSVD<double>::mulRotationOnRight(1, 2, matrix, rotation);

    ASSERT_TRUE(expected.equalsTo(&matrix));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, JacobiSVD_test7) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix = NDArrayFactory::create<double>('c', {5,5}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0});
    auto rotation = NDArrayFactory::create<double>('c', {2,2}, {0.2, math::nd4j_sqrt<double, double>(0.6), -math::nd4j_sqrt<double, double>(0.6), 0.2});

    auto expected = NDArrayFactory::create<double>('c', {5,5}, {-18, 14.9173, 3.0254,-7, 1, 2,-13.6698,11.3427,14, 2, -2, 3.99677,10.1206, 2,-6, -3, 4.59677,7.79677,-2, 7, 16, 0.67621,-12.219, 7, 0});

    ops::helpers::JacobiSVD<double>::mulRotationOnRight(2, 1, matrix, rotation);

    ASSERT_TRUE(expected.equalsTo(&matrix));
}

//////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, JacobiSVD_test8) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix = NDArrayFactory::create<double>('c', {5,5}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0});
    auto rotation = NDArrayFactory::create<double>('c', {2,2}, {0.2, math::nd4j_sqrt<double, double>(0.6), -math::nd4j_sqrt<double,double>(0.6), 0.2});

    auto expected = NDArrayFactory::create<double>('c', {5,5}, {-18,  1, 18.5173,-7, 1, 2,-18,-12.6698,14, 2, -2,-11, 7.79677, 2,-6, -3, -8, 7.79677,-2, 7, 16, 15,-2.92379, 7, 0});

    ops::helpers::JacobiSVD<double>::mulRotationOnRight(2, 2, matrix, rotation);

    ASSERT_TRUE(expected.equalsTo(&matrix));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, JacobiSVD_test9) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix = NDArrayFactory::create<double>('c', {5,5}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0});

    auto expS = NDArrayFactory::create<double>('c', {5,1}, {35.7975, 29.1924, 11.1935, 9.2846, 6.77071});
    auto expU = NDArrayFactory::create<double>('c', {5,5}, {0.744855,0.0686476,  0.079663,0.0889877,  0.65285, -0.386297,-0.760021,0.00624688, 0.156774, 0.498522, 0.186491,-0.322427,  0.773083,-0.468826,-0.209299, 0.246053,-0.215594,  0.240942, 0.821793,-0.399475, -0.447933, 0.516928,  0.581295, 0.269001, 0.349106});
    auto expV = NDArrayFactory::create<double>('c', {5,5}, {-0.627363,   0.23317, 0.501211,  0.160272,  -0.524545, -0.0849394,  0.917171,-0.155876,-0.0124053,   0.356555, 0.66983,  0.182569, 0.696897,  0.179807,0.000864568, -0.387647, -0.264316, 0.416597, 0.0941014,   0.772955, 0.0160818,-0.0351459,-0.255484,  0.965905,  0.0161524});

    ops::helpers::JacobiSVD<double> jac(matrix, true, true, true);

    ASSERT_TRUE(expS.equalsTo(&jac._s));
    ASSERT_TRUE(expU.equalsTo(&jac._u));
    ASSERT_TRUE(expV.equalsTo(&jac._v));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, JacobiSVD_test10) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix = NDArrayFactory::create<double>('c', {5,5}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0});

    auto expS = NDArrayFactory::create<double>('c', {5,1}, {35.7975, 29.1924, 11.1935, 9.2846, 6.77071});
    auto expU = NDArrayFactory::create<double>('c', {5,5}, {0.744855,0.0686476,  0.079663,0.0889877,  0.65285, -0.386297,-0.760021,0.00624688, 0.156774, 0.498522, 0.186491,-0.322427,  0.773083,-0.468826,-0.209299, 0.246053,-0.215594,  0.240942, 0.821793,-0.399475, -0.447933, 0.516928,  0.581295, 0.269001, 0.349106});
    auto expV = NDArrayFactory::create<double>('c', {5,5}, {-0.627363,   0.23317, 0.501211,  0.160272,  -0.524545, -0.0849394,  0.917171,-0.155876,-0.0124053,   0.356555, 0.66983,  0.182569, 0.696897,  0.179807,0.000864568, -0.387647, -0.264316, 0.416597, 0.0941014,   0.772955, 0.0160818,-0.0351459,-0.255484,  0.965905,  0.0161524});

    ops::helpers::JacobiSVD<double> jac(matrix, true, true, false);

    ASSERT_TRUE(expS.equalsTo(&jac._s));
    ASSERT_TRUE(expU.equalsTo(&jac._u));
    ASSERT_TRUE(expV.equalsTo(&jac._v));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, JacobiSVD_test11) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix = NDArrayFactory::create<double>('c', {6,5}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0, 3, -11, 2, 12, 10});

    auto expS = NDArrayFactory::create<double>('c', {5,1}, {36.27, 32.1997, 15.9624, 10.6407, 6.9747});
    auto expU = NDArrayFactory::create<double>('c', {6,5}, {0.720125,-0.149734,  0.227784,-0.0288531,  0.595353, -0.509487,-0.567298, -0.237169,-0.0469077,   0.38648, 0.120912, -0.32916,-0.0202265,  0.921633, -0.153994, 0.180033,-0.294831,  0.357867, -0.194106, -0.646595, -0.354033, 0.521937,  0.556566,  0.305582,  0.211013, -0.222425,-0.433662,  0.673515, -0.128465,  0.099309});
    auto expV = NDArrayFactory::create<double>('c', {5,5}, {-0.581609,  0.315327,0.333158,  0.34476, -0.576582, 0.117364,  0.889461,0.175174,-0.166603,  0.369651, 0.643246,-0.0899117,0.613288, 0.442462,-0.0790943, -0.480818, -0.264384,0.395122, 0.223126,  0.702145, -0.0548207, -0.177325,0.571031,-0.779632,   -0.1779});

    ops::helpers::JacobiSVD<double> jac(matrix, true, true, false);

    ASSERT_TRUE(expS.equalsTo(&jac._s));
    ASSERT_TRUE(expU.equalsTo(&jac._u));
    ASSERT_TRUE(expV.equalsTo(&jac._v));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, JacobiSVD_test12) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix = NDArrayFactory::create<double>('c', {6,5}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0, 3, -11, 2, 12, 10});

    auto expS = NDArrayFactory::create<double>('c', {5,1}, {36.27, 32.1997, 15.9624, 10.6407, 6.9747});
    auto expU = NDArrayFactory::create<double>('c', {6,6}, {0.720125,-0.149734,  0.227784,-0.0288531, 0.595353,-0.227676, -0.509487,-0.567298, -0.237169,-0.0469077,  0.38648,-0.459108, 0.120912, -0.32916,-0.0202265,  0.921633,-0.153994,0.0591992, 0.180033,-0.294831,  0.357867, -0.194106,-0.646595,-0.544823, -0.354033, 0.521937,  0.556566,  0.305582, 0.211013,-0.393155, -0.222425,-0.433662,  0.673515, -0.128465, 0.099309, 0.531485});
    auto expV = NDArrayFactory::create<double>('c', {5,5}, {-0.581609,  0.315327,0.333158,  0.34476, -0.576582, 0.117364,  0.889461,0.175174,-0.166603,  0.369651, 0.643246,-0.0899117,0.613288, 0.442462,-0.0790943, -0.480818, -0.264384,0.395122, 0.223126,  0.702145, -0.0548207, -0.177325,0.571031,-0.779632,   -0.1779});

    ops::helpers::JacobiSVD<double> jac(matrix, true, true, true);

    ASSERT_TRUE(expS.equalsTo(&jac._s));
    ASSERT_TRUE(expU.equalsTo(&jac._u));
    ASSERT_TRUE(expV.equalsTo(&jac._v));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, JacobiSVD_test13) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix = NDArrayFactory::create<double>('c', {5,6}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0, 3, -11, 2, 12, 10});

    auto expS = NDArrayFactory::create<double>('c', {5,1}, {40.499, 23.5079, 17.8139, 14.4484, 7.07957});
    auto expU = NDArrayFactory::create<double>('c', {5,5}, {0.592324,-0.121832,-0.484064,-0.624878,-0.0975619, 0.651331, 0.367418, 0.117429, 0.370792,  0.538048, -0.272693,-0.138725, 0.249336,-0.540587,  0.742962, 0.263619,-0.903996, 0.179714, 0.276206, 0.0686237, -0.284717,-0.117079,-0.810818, 0.321741,  0.379848});
    auto expV = NDArrayFactory::create<double>('c', {6,6}, {-0.619634,-0.158345, 0.462262,-0.021009,-0.299779,  0.53571, -0.183441,-0.504296,-0.150804,-0.251078,-0.563079,-0.556052, 0.724925,-0.404744, 0.154104,-0.177039,-0.262604, 0.431988, 0.0335645,-0.501546, 0.221702, 0.797602, 0.186339,-0.165176, -0.0675636,0.0663677,-0.728788, 0.414614,-0.390566, 0.368038, -0.226262, -0.54849,-0.399426,-0.311613, 0.580387, 0.233392});

    ops::helpers::JacobiSVD<double> jac(matrix, true, true, true);

    ASSERT_TRUE(expS.equalsTo(&jac._s));
    ASSERT_TRUE(expU.equalsTo(&jac._u));
    ASSERT_TRUE(expV.equalsTo(&jac._v));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, JacobiSVD_test14) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix = NDArrayFactory::create<double>('c', {5,6}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0, 3, -11, 2, 12, 10});

    auto expS = NDArrayFactory::create<double>('c', {5,1}, {40.499, 23.5079, 17.8139, 14.4484, 7.07957});
    auto expU = NDArrayFactory::create<double>('c', {5,5}, {0.592324,-0.121832,-0.484064,-0.624878,-0.0975619, 0.651331, 0.367418, 0.117429, 0.370792,  0.538048, -0.272693,-0.138725, 0.249336,-0.540587,  0.742962, 0.263619,-0.903996, 0.179714, 0.276206, 0.0686237, -0.284717,-0.117079,-0.810818, 0.321741,  0.379848});
    auto expV = NDArrayFactory::create<double>('c', {6,5}, {-0.619634,-0.158345, 0.462262,-0.021009,-0.299779, -0.183441,-0.504296,-0.150804,-0.251078,-0.563079, 0.724925,-0.404744, 0.154104,-0.177039,-0.262604, 0.0335645,-0.501546, 0.221702, 0.797602, 0.186339, -0.0675636,0.0663677,-0.728788, 0.414614,-0.390566, -0.226262, -0.54849,-0.399426,-0.311613, 0.580387});

    ops::helpers::JacobiSVD<double> jac(matrix, true, true, false);

    ASSERT_TRUE(expS.equalsTo(&jac._s));
    ASSERT_TRUE(expU.equalsTo(&jac._u));
    ASSERT_TRUE(expV.equalsTo(&jac._v));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, JacobiSVD_test15) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix = NDArrayFactory::create<double>('c', {5,6}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0, 3, -11, 2, 12, 10});

    auto expS = NDArrayFactory::create<double>('c', {5,1}, {40.499, 23.5079, 17.8139, 14.4484, 7.07957});
    auto expU = NDArrayFactory::create<double>('c', {5,5}, {0.592324,-0.121832,-0.484064,-0.624878,-0.0975619, 0.651331, 0.367418, 0.117429, 0.370792,  0.538048, -0.272693,-0.138725, 0.249336,-0.540587,  0.742962, 0.263619,-0.903996, 0.179714, 0.276206, 0.0686237, -0.284717,-0.117079,-0.810818, 0.321741,  0.379848});
    auto expV = NDArrayFactory::create<double>('c', {6,5}, {-0.619634,-0.158345, 0.462262,-0.021009,-0.299779, -0.183441,-0.504296,-0.150804,-0.251078,-0.563079, 0.724925,-0.404744, 0.154104,-0.177039,-0.262604, 0.0335645,-0.501546, 0.221702, 0.797602, 0.186339, -0.0675636,0.0663677,-0.728788, 0.414614,-0.390566, -0.226262, -0.54849,-0.399426,-0.311613, 0.580387});

    ops::helpers::JacobiSVD<double> jac(matrix, false, false, false);

    ASSERT_TRUE(expS.equalsTo(&jac._s));
}


///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, SVD_test16) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix1 = NDArrayFactory::create<double>('c', {6,5}, {-2 ,-3 ,2 ,1 ,0 ,0 ,-4 ,5 ,-2 ,-3 ,-4 ,0 ,5 ,-1 ,-5 ,-3 ,-5 ,3 ,3 ,3 ,-5 ,5 ,-5 ,0 ,2 ,-2 ,-3 ,-4 ,-5 ,-3});
    auto matrix2 = NDArrayFactory::create<double>('c', {6,6}, {-10 ,-16 ,-20 ,13 ,20 ,-10 ,-9 ,-1 ,-7 ,-20 ,-4 ,20 ,-11 ,19 ,-5 ,-18 ,12 ,-19 ,18 ,-18 ,17 ,-10 ,-19 ,14 ,-2 ,-7 ,-17 ,-14 ,-4 ,-16 ,18 ,-6 ,-18 ,1 ,-15 ,-12});
    auto matrix3 = NDArrayFactory::create<double>('c', {5,5}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0});
    auto matrix4 = NDArrayFactory::create<double>('c', {5,5}, {3 ,-8 ,5 ,7 ,-8 ,4 ,-19 ,-12 ,-4 ,-5 ,-11 ,19 ,-2 ,-7 ,1 ,16 ,-5 ,10 ,19 ,-19 ,0 ,-20 ,0 ,-8 ,-13});

    auto expM = NDArrayFactory::create<double>('c', {6,5}, {-2,     -3,      2,      1,      0, 0,7.07022,      0,      0,      0, -4,      0,5.09585,      0,      0, -3,      0,      0,3.32256,      0, -5,      0,      0,      0,1.00244, -2,     -3,     -4,     -5,      0});
    auto expU = NDArrayFactory::create<double>('c', {6,6}, {-5.58884,-2.18397,-11.0944, 3.30292,  0,-10, 8.19094, 5.05917, 16.9641,-4.53112,  0, 20, 6.55878, 3.76734, 15.9255,-3.76399,  0,-19, 1.36021, 23.3551,-8.01165, -1.5816,  0, 14, -15.6318,-2.85386, 8.83051, 2.74286,  1,-16, 18,      -6,     -18,       1,-15,-12});
    auto expV = NDArrayFactory::create<double>('c', {5,5}, {-18,       1,      19,      -7,       1, 2, 14.5866, 3.90133, 1.06593, 9.99376, -2, 9.97311, 2.44445, 6.85159, 2.37014, -3, 0.56907,-8.93313,-5.31596, 3.10096, 16,-10.6859, 1.70708,-7.24295,-10.6975});

    ops::helpers::SVD<double> svd(matrix4, 4, true, true, true, 't');
    svd._m = matrix1;
    svd._u = matrix2;
    svd._v = matrix3;

    svd.DivideAndConquer(0, 3, 1, 1, 1);
    // svd._m.printIndexedBuffer();
    ASSERT_TRUE(expM.isSameShapeStrict(&svd._m));
    ASSERT_TRUE(expU.isSameShapeStrict(&svd._u));
    ASSERT_TRUE(expV.isSameShapeStrict(&svd._v));

    ASSERT_TRUE(expM.equalsTo(&svd._m));
    ASSERT_TRUE(expU.equalsTo(&svd._u));
    ASSERT_TRUE(expV.equalsTo(&svd._v));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, SVD_test17) {

    #ifdef __CUDABLAS__
    return;
    #endif
    auto matrix1 = NDArrayFactory::create<double>('c', {6,5}, {-2 ,-3 ,2 ,1 ,0 ,0 ,-4 ,5 ,-2 ,-3 ,-4 ,0 ,5 ,-1 ,-5 ,-3 ,-5 ,3 ,3 ,3 ,-5 ,5 ,-5 ,0 ,2 ,-2 ,-3 ,-4 ,-5 ,-3});
    auto matrix2 = NDArrayFactory::create<double>('c', {6,6}, {-10 ,-16 ,-20 ,13 ,20 ,-10 ,-9 ,-1 ,-7 ,-20 ,-4 ,20 ,-11 ,19 ,-5 ,-18 ,12 ,-19 ,18 ,-18 ,17 ,-10 ,-19 ,14 ,-2 ,-7 ,-17 ,-14 ,-4 ,-16 ,18 ,-6 ,-18 ,1 ,-15 ,-12});
    auto matrix3 = NDArrayFactory::create<double>('c', {5,5}, {-18 ,1 ,19 ,-7 ,1 ,2 ,-18 ,-13 ,14 ,2 ,-2 ,-11 ,8 ,2 ,-6 ,-3 ,-8 ,8 ,-2 ,7 ,16 ,15 ,-3 ,7 ,0});
    auto matrix4 = NDArrayFactory::create<double>('c', {5,5}, {3 ,-8 ,5 ,7 ,-8 ,4 ,-19 ,-12 ,-4 ,-5 ,-11 ,19 ,-2 ,-7 ,1 ,16 ,-5 ,10 ,19 ,-19 ,0 ,-20 ,0 ,-8 ,-13});

    auto expM = NDArrayFactory::create<double>('c', {6,5}, {-2,     -3,      2,      1,       0, 0,12.1676,      0,      0,       0, -4,      0,7.49514,      0,       0, -3,      0,      0,5.00951,       0, -5,      0,      0,      0, 1.63594, -2,      0,      0,      0,       0});
    auto expU = NDArrayFactory::create<double>('c', {6,6}, {0.295543,-0.238695, 0.262095,-0.231772,  -0.85631,-10, 0.519708,0.0571492,-0.368706,-0.727615,  0.247527, 20, 0.313717,-0.561567,-0.602941, 0.469567,-0.0468295,-19, 0.474589,-0.372165, 0.656962, 0.124776,  0.434845, 14, -0.564717,-0.697061,0.0150082,  -0.4252,  0.119081,-16, 18,       -6,      -18,        1,       -15,-12});
    auto expV = NDArrayFactory::create<double>('c', {5,5}, {-18,         1,        19,        -7,       1, 2,-0.0366659,  0.977361,-0.0316106,0.205967, -2, -0.670795, -0.151697, -0.503288,0.523185, -3,  0.740124,-0.0841435, -0.486714,0.456339, 16, 0.0300945, -0.121135,   0.71331,0.689645});

    ops::helpers::SVD<double> svd(matrix4, 10, true, true, true, 't');
    svd._m = matrix1;
    svd._u = matrix2;
    svd._v = matrix3;

    svd.DivideAndConquer(0, 3, 1, 1, 1);

    ASSERT_TRUE(expM.equalsTo(&svd._m));
    ASSERT_TRUE(expU.equalsTo(&svd._u));
    ASSERT_TRUE(expV.equalsTo(&svd._v));

    ASSERT_TRUE(expM.isSameShapeStrict(&svd._m));
    ASSERT_TRUE(expU.isSameShapeStrict(&svd._u));
    ASSERT_TRUE(expV.isSameShapeStrict(&svd._v));
}

// ///////////////////////////////////////////////////////////////////
// TEST_F(HelpersTests1, SVD_test18) {

//     auto matrix('c', {10,10}, {10 ,7 ,5 ,2 ,17 ,18 ,-18 ,10 ,18 ,1 ,4 ,2 ,-7 ,-18 ,20 ,14 ,
//                                           -3 ,-10 ,-4 ,2 ,-17 ,-17 ,1 ,2 ,-9 ,-6 ,-13 ,16 ,-18 ,-13 ,
//                                           -10 ,16 ,-10 ,-13 ,-11 ,-6 ,-19 ,17 ,-12 ,3 ,-14 ,7 ,7 ,-9 ,
//                                           5 ,-16 ,7 ,16 ,13 ,12 ,2 ,18 ,6 ,3 ,-8 ,11 ,-1 ,5 ,16 ,-16 ,
//                                           -9 ,8 ,10 ,-7 ,-4 ,1 ,-10 ,0 ,20 ,7 ,-11 ,-13 ,-3 ,20 ,-6 ,
//                                           9 ,10 ,8 ,-20 ,1 ,19 ,19 ,-12 ,-20 ,-2 ,17 ,-18 ,-5 ,-14 ,0
//                                           ,9 ,-16 ,9 ,-15 ,7 ,18 ,-10 ,8 ,-11 ,-4});

//     auto expS('c', {10, 1}, {65.0394, 56.1583, 48.9987, 39.2841, 35.7296, 22.8439, 17.474, 15.2708, 15.0768, 0.846648});

//     auto expU('c', {10,10}, {0.413187, 0.159572,0.0238453, 0.601154,-0.0428558, -0.461779,   0.41787, -0.221153, 0.0206268, 0.0532219,
//                                         0.364377,-0.154281, 0.199857,-0.0943331,  0.415653, -0.139834, -0.258458,   0.10677,   0.72003,-0.0749772,
//                                        -0.315063,-0.418079,-0.377499,  0.37031, 0.0123835,  0.300036,  0.153702, -0.129223,  0.390675,  0.403962,
//                                         0.102001,-0.216667, -0.74093,-0.166164,-0.0269665, -0.240065, 0.0549761,-0.0178001, 0.0197525,  -0.55134,
//                                        -0.107298, 0.386899,-0.377536, 0.033214,  0.486739, -0.245438,  -0.43788, -0.208875, -0.170449,  0.365491,
//                                          0.18026, 0.240482,-0.115801, 0.237399, -0.643413,  0.139274, -0.582963, -0.116222,  0.224524,-0.0525887,
//                                         0.141172, 0.340505,-0.261653, 0.186411, 0.0625811,   0.19585,  0.128195,  0.832893, 0.0319884, 0.0864513,
//                                        -0.385777,-0.330504, 0.128342, 0.156083, -0.200883, -0.648548, -0.256507,   0.40519,-0.0434365, 0.0909978,
//                                         0.574478,-0.371028,-0.136672,-0.328417, -0.190226,-0.0476664,-0.0399815, 0.0687528, -0.242039,  0.549918,
//                                         0.209886,-0.398294,0.0919207, 0.490454,  0.305228,  0.280486, -0.341358, 0.0540678, -0.432618, -0.264332});

//     auto expV('c', {10,10}, {0.423823,-0.0845148,  0.389647, -0.10717,-0.168732, 0.123783, 0.159237, -0.450407, -0.611513,-0.0629076,
//                                          0.412121,  0.317493, -0.355665,-0.383203,-0.382616,-0.309073, -0.21869,-0.0746378, 0.0829771,  0.392186,
//                                        -0.0603483,  0.232234, 0.0383737, 0.435441,0.0829318, 0.327822,-0.206101,  0.184083,  -0.34018,  0.667018,
//                                         -0.453935,  0.119616,  0.288392, 0.184366,-0.524289, -0.42264,  0.41005,-0.0505891,0.00333608,  0.195602,
//                                          0.247802, 0.0776165,   0.33026, 0.190986, 0.526809,-0.345006,0.0651023, -0.386472,  0.395169,  0.284091,
//                                          0.426355, -0.269507,  0.304685, 0.386708,-0.257916,-0.287742,-0.329622,  0.463719, 0.0613767,  -0.16261,
//                                         -0.384582,  0.241486,  0.425935,-0.292636,0.0465594,-0.125018,-0.685871, -0.112806,-0.0977978, -0.127356,
//                                         -0.121678,  -0.06796, -0.501443, 0.473165,0.0422977,-0.369324,-0.248758, -0.408769, -0.305785, -0.211138,
//                                          0.186099,  0.809997, 0.0338281, 0.268965, -0.04829, 0.141617,  0.12121, 0.0362537, 0.0831986, -0.436428,
//                                         0.0174496,  0.161638,-0.0334757,-0.224027, 0.439364,-0.478697, 0.237318,  0.457809, -0.483235,-0.0253522});

//     ops::helpers::SVD<double> svd(matrix, 8, true, true, true);
//     // svd._u.printShapeInfo();
//     // svd._u.printIndexedBuffer();

//     ASSERT_TRUE(expS.equalsTo(&svd._s));
//     ASSERT_TRUE(expU.equalsTo(&svd._u));
//     ASSERT_TRUE(expV.equalsTo(&svd._v));

//     ASSERT_TRUE(expS.isSameShapeStrict(&svd._s));
//     ASSERT_TRUE(expU.isSameShapeStrict(&svd._u));
//     ASSERT_TRUE(expV.isSameShapeStrict(&svd._v));
// }


// ///////////////////////////////////////////////////////////////////
// TEST_F(HelpersTests1, SVD_test19) {

//     auto matrix('c', {11,10}, {10 ,7 ,5 ,2 ,17 ,18 ,-18 ,10 ,18 ,1 ,4 ,2 ,-7 ,-18 ,20 ,14 ,
//                                           -3 ,-10 ,-4 ,2 ,-17 ,-17 ,1 ,2 ,-9 ,-6 ,-13 ,16 ,-18 ,-13 ,
//                                           -10 ,16 ,-10 ,-13 ,-11 ,-6 ,-19 ,17 ,-12 ,3 ,-14 ,7 ,7 ,-9 ,
//                                           5 ,-16 ,7 ,16 ,13 ,12 ,2 ,18 ,6 ,3 ,-8 ,11 ,-1 ,5 ,16 ,-16 ,
//                                           -9 ,8 ,10 ,-7 ,-4 ,1 ,-10 ,0 ,20 ,7 ,-11 ,-13 ,-3 ,20 ,-6 ,
//                                           9 ,10 ,8 ,-20 ,1 ,19 ,19 ,-12 ,-20 ,-2 ,17 ,-18 ,-5 ,-14 ,0
//                                           ,9 ,-16 ,9 ,-15 ,7 ,18 ,-10 ,8 ,-11 ,-4,
//                                           -7,  1, -2,  15, 0,  4,  -9,19,  -3, 10 });

//     auto expS('c', {10, 1}, {65.5187, 56.305, 50.9808, 41.6565, 35.8698, 29.3898, 17.9743, 15.3568, 15.2223, 0.846847});

//     auto expU('c', {11,11},   {-0.387999,-0.117659,  0.162976,  0.641067,-0.0174306, -0.181469,-0.218643,  -0.308042, 0.0670776,-0.0632539, -0.462228,
//                                            -0.37021,  0.14822, -0.195157,-0.0467394, -0.381275, -0.183363, 0.326599,  -0.370579,  -0.56626, 0.0798798,  0.225133,
//                                            0.339692, 0.433146,   0.30841,  0.134184, -0.108725,  0.466056,-0.153546,  -0.359783, -0.189621, -0.402737, 0.0605675,
//                                          -0.0650167, 0.268868,  0.662416, -0.327524, 0.0339198,-0.0916729,0.0415428, -0.0765093,-0.0288338,  0.546108, -0.247418,
//                                            0.114029,-0.361828,  0.379255,-0.0935836, -0.488912, -0.125232, 0.480666,-0.00544881,  0.280747,  -0.36698,-0.0648559,
//                                           -0.174798, -0.21859,  0.178313,  0.212153,  0.579101,  0.369942, 0.551063,  -0.139813,-0.0296135, 0.0572204,  0.212783,
//                                           -0.133981,-0.311817,  0.304673, 0.0865395, -0.104221,  0.196295,-0.191271,   0.571084, -0.603697,-0.0868996,-0.0196788,
//                                            0.398676, 0.319697, -0.112145,  0.235089,  0.201666, -0.337134,  0.43406,   0.261686, -0.283102,-0.0999458, -0.411893,
//                                           -0.559998, 0.392802, 0.0996997, -0.281135,   0.24017, -0.136769,0.0121463,   0.218664,  0.127577, -0.550001,0.00227476,
//                                           -0.197522, 0.403875,-0.0647804,  0.383315, -0.388502,  0.335719,  0.20912,   0.404926,  0.309087,  0.266437, 0.0942471,
//                                            0.140425,0.0934688,  0.325994,  0.345081, 0.0825574, -0.521239,-0.129018,  0.0806886, 0.0442647,  0.014397,  0.665103});

//     auto expV('c', {10,10},  {-0.4428, 0.0661762,-0.361903, 0.0307317,   0.19574,-0.0356551,-0.241991, 0.0866805,    0.74701, 0.062837,
//                                           -0.400091, -0.277277, 0.375095, -0.323052,  0.443668, -0.264809, 0.292881, -0.106586,-0.00623963,-0.392226,
//                                           0.0536693, -0.232105,0.0106246,  0.332557, -0.167406,  0.400872,0.0835708,  0.414598,   0.141906,-0.666936,
//                                            0.473793, -0.121962,-0.147941,  0.414665,  0.538964, -0.372149,-0.285458, -0.132952, -0.0166319,-0.195945,
//                                           -0.251722,-0.0813691,-0.233887,  0.280439, -0.512597, -0.328782, 0.074277, -0.581806, -0.0327555,-0.284121,
//                                           -0.406324,  0.284462,-0.168731,  0.518021,  0.226396, -0.109282, 0.381083,  0.305342,  -0.359301, 0.162524,
//                                            0.335857, -0.302206,-0.484806, -0.196382,0.00286755, -0.111789, 0.672115, 0.0705632,   0.191787, 0.127533,
//                                            0.185896,  0.134279, 0.608397,  0.382412,-0.0997649, -0.117987, 0.326934,-0.0941208,   0.496913, 0.210914,
//                                           -0.201675, -0.795446,0.0916484,  0.267237,0.00604554,  0.167517, -0.13914,-0.0355323, -0.0869256, 0.436465,
//                                          0.00123325, -0.142684,0.0978458,-0.0945446, -0.349755, -0.674457,-0.196126,  0.587134,-0.00964182,0.0249317});

//     ops::helpers::SVD<double> svd(matrix, 8, true, true, true);

//     ASSERT_TRUE(expS.equalsTo(&svd._s));
//     ASSERT_TRUE(expU.equalsTo(&svd._u));
//     ASSERT_TRUE(expV.equalsTo(&svd._v));

//     ASSERT_TRUE(expS.isSameShapeStrict(&svd._s));
//     ASSERT_TRUE(expU.isSameShapeStrict(&svd._u));
//     ASSERT_TRUE(expV.isSameShapeStrict(&svd._v));
// }


// ///////////////////////////////////////////////////////////////////
// TEST_F(HelpersTests1, SVD_test20) {

//     auto matrix('c', {10,11}, {10 ,7 ,5 ,2 ,17 ,18 ,-18 ,10 ,18 ,1 ,4 ,2 ,-7 ,-18 ,20 ,14 ,
//                                           -3 ,-10 ,-4 ,2 ,-17 ,-17 ,1 ,2 ,-9 ,-6 ,-13 ,16 ,-18 ,-13 ,
//                                           -10 ,16 ,-10 ,-13 ,-11 ,-6 ,-19 ,17 ,-12 ,3 ,-14 ,7 ,7 ,-9 ,
//                                           5 ,-16 ,7 ,16 ,13 ,12 ,2 ,18 ,6 ,3 ,-8 ,11 ,-1 ,5 ,16 ,-16 ,
//                                           -9 ,8 ,10 ,-7 ,-4 ,1 ,-10 ,0 ,20 ,7 ,-11 ,-13 ,-3 ,20 ,-6 ,
//                                           9 ,10 ,8 ,-20 ,1 ,19 ,19 ,-12 ,-20 ,-2 ,17 ,-18 ,-5 ,-14 ,0
//                                           ,9 ,-16 ,9 ,-15 ,7 ,18 ,-10 ,8 ,-11 ,-4,
//                                           -7,  1, -2,  15, 0,  4,  -9,19,  -3, 10 });

//     auto expS('c', {10, 1}, {68.9437, 54.8773, 50.7858, 42.4898, 35.1984, 26.6285, 21.376, 12.2334, 5.9112, 0.38292});

//     auto expU('c', {10,10}, {0.30332,-0.0677785,  0.155514, -0.722623,-0.0843687,-0.0712535,  0.414936,  -0.15422, -0.381536,-0.057561,
//                                         0.473286, 0.0231518, 0.0878106,   0.45493, -0.311654,  0.138957,  0.311305,  0.509971, -0.288207,0.0656506,
//                                        -0.131548,   0.32051,  0.489848,-0.0539042, -0.521328, -0.363728, -0.328685,-0.0329672,-0.0726502, 0.344431,
//                                         0.072974,  0.522632, -0.477056, 0.0618953,-0.0980883, -0.095653,  -0.26596,  -0.15453, -0.475107,-0.388594,
//                                         0.267569, -0.336154,-0.0930604, -0.261336,  -0.39945,  0.480346, -0.568317, 0.0593335,  0.102036,-0.106029,
//                                       -0.0919782, -0.460136,  0.106434,  0.327722, 0.0952523, 0.0915698, -0.129052, -0.460878,  -0.59722, 0.240608,
//                                        -0.248827,  -0.48834, -0.243788, -0.106636,-0.0803772, -0.567457,  -0.12005,  0.480504, -0.188409,-0.139802,
//                                         0.643408,  -0.16245, -0.152596,   0.16849,-0.0120438,  -0.51616,-0.0694232,  -0.36172,  0.322169,0.0440701,
//                                        -0.229467,-0.0227008, -0.588303,-0.0327104, -0.482264, 0.0794715,  0.340158, -0.175969,  0.108784, 0.449731,
//                                         0.229718,  0.169979, -0.227516,  -0.21815,  0.454459,  0.017476, -0.278516,  0.287333, -0.148844, 0.655637});

//     auto expV('c', {11,11},  {0.190806, -0.193628,  0.383793,-0.0266376,   0.113035,  0.158361, 0.0297803,  -0.793229,  -0.13761,-0.260666, -0.152503,
//                                         -0.303449, 0.0392386,  0.250627, -0.165231,   0.141567, 0.0479565,   0.72763,    0.14053, -0.339907, 0.224366, -0.280806,
//                                         -0.159724,  -0.38984, -0.256355, -0.337861,   0.075089, -0.237427, -0.153718,  -0.217747,  0.320899, 0.455058, -0.446697,
//                                          0.376823, -0.560303,  0.269135,  0.265416,-0.00742902, 0.0263377, -0.192808,   0.435842, -0.275365,0.0511804,  -0.30799,
//                                          0.522537,  0.209791,  -0.44191, -0.282323,   -0.12139,  0.226382,  0.221075,  0.0844301, 0.0285412,-0.297578, -0.443394,
//                                         0.0588008,  0.115035,   0.54835,  -0.52266,  -0.141345,  0.411122, -0.182423,   0.213721,  0.353022, 0.119504, 0.0508673,
//                                         -0.299021,-0.0424794, -0.285618,  0.177961,    0.35831,  0.769783, -0.215983,-0.00423939, -0.110575,0.0928082,-0.0841152,
//                                        -0.0977062, -0.624782, -0.240391, -0.276154,  -0.342018,  0.199695,  0.268881, 0.00402219,-0.0536164, -0.17679,  0.450283,
//                                          0.428931, 0.0748696, -0.120853, -0.360103,    0.37093,-0.0611563, -0.100263, -0.0604207, -0.432926, 0.412875,   0.39142,
//                                          -0.35553,  0.127463,-0.0199906, -0.343149,  -0.315968, -0.115698, -0.442585,  0.0126156, -0.584161,-0.219242,  -0.20156,
//                                         -0.134753, -0.154272,  0.037343, -0.281348,   0.666324, -0.213813,-0.0427932,   0.238783,  0.132347,-0.557478, 0.0253325});

//     ops::helpers::SVD<double> svd(matrix, 8, true, true, true);

//     ASSERT_TRUE(expS.equalsTo(&svd._s));
//     ASSERT_TRUE(expU.equalsTo(&svd._u));
//     ASSERT_TRUE(expV.equalsTo(&svd._v));

//     ASSERT_TRUE(expS.isSameShapeStrict(&svd._s));
//     ASSERT_TRUE(expU.isSameShapeStrict(&svd._u));
//     ASSERT_TRUE(expV.isSameShapeStrict(&svd._v));
// }


/////////////////////////////////////////////////////////////////////
//TEST_F(HelpersTests1, reverseArray_test1) {
//
//    auto inArr = NDArrayFactory::create<float>('c', {2,5}, {1,2,3,4,5,6,7,8,9,10});
//    auto exp = NDArrayFactory::create<float>('c', {2,5}, {10,9,8,7,6,5,4,3,2,1});
//    auto outArr = NDArrayFactory::create<float>('c', {2,5});
//
//
//    ops::helpers::reverseArray<float>(nd4j::LaunchContext ::defaultContext(), inArr.getBuffer(), inArr.getShapeInfo(), outArr.getBuffer(), outArr.getShapeInfo());
//
//    ASSERT_TRUE(outArr.equalsTo(&exp));
//    ASSERT_TRUE(outArr.isSameShapeStrict(&exp));
//}
//
//
/////////////////////////////////////////////////////////////////////
//TEST_F(HelpersTests1, reverseArray_test2) {
//
//    auto inArr = NDArrayFactory::create<float>('c', {2,5}, {1,2,3,4,5,6,7,8,9,10});
//    auto exp = NDArrayFactory::create<float>('c', {2,5}, {10,9,8,7,6,5,4,3,2,1});
//
//
//    ops::helpers::reverseArray<float>(nd4j::LaunchContext ::defaultContext(), inArr.getBuffer(), inArr.getShapeInfo(), inArr.getBuffer(), inArr.getShapeInfo());
//
//    ASSERT_TRUE(inArr.equalsTo(&exp));
//    ASSERT_TRUE(inArr.isSameShapeStrict(&exp));
//}
//
//
/////////////////////////////////////////////////////////////////////
//TEST_F(HelpersTests1, reverseArray_test3) {
//
//    auto inArr = NDArrayFactory::create<float>('c', {2,5}, {1,2,3,4,5,6,7,8,9,10});
//    auto exp = NDArrayFactory::create<float>('c', {2,5}, {5,4,3,2,1,6,7,8,9,10});
//    auto outArr = NDArrayFactory::create<float>('c', {2,5});
//
//    ops::helpers::reverseArray<float>(nd4j::LaunchContext ::defaultContext(), inArr.getBuffer(), inArr.getShapeInfo(), outArr.getBuffer(), outArr.getShapeInfo(), 5);
//
//    ASSERT_TRUE(outArr.equalsTo(&exp));
//    ASSERT_TRUE(outArr.isSameShapeStrict(&exp));
//}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, rnnCell_test1) {

    const int bS = 2;
    const int inSize   = 4;
    const int numUnits = 4;

    NDArray xt('c', {bS, inSize}, nd4j::DataType::DOUBLE);
    NDArray ht_1('c', {bS, numUnits}, nd4j::DataType::DOUBLE);
    NDArray Wx('c', {inSize, numUnits}, nd4j::DataType::DOUBLE);
    NDArray Wh('c', {numUnits, numUnits}, nd4j::DataType::DOUBLE);
    NDArray b ('c', {2*numUnits}, {0.0,0.0,0.0,0.0,  0.1,0.2,0.3,0.4});
    NDArray ht('c', {bS, numUnits}, nd4j::DataType::DOUBLE);

    xt.assign(0.1);
    ht_1.assign(0.2);
    Wx.assign(0.3);
    Wh.assign(0.4);

    NDArray expHt('c', {bS, numUnits}, {0.492988, 0.56489956, 0.6291452 , 0.6858091,0.492988, 0.56489956, 0.6291452 , 0.6858091});

    ops::helpers::rnnCell(nd4j::LaunchContext ::defaultContext(), &xt, &Wx, &Wh, &b, &ht_1, &ht);

    ASSERT_TRUE(expHt.isSameShape(ht));
    ASSERT_TRUE(expHt.equalsTo(ht));
}


///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, rnnCell_test2) {

    const int bS = 2;
    const int inSize   = 10;
    const int numUnits = 4;

    auto xt = NDArrayFactory::create<double>('c', {bS, inSize});
    auto ht_1 = NDArrayFactory::create<double>('c', {bS, numUnits});
    auto Wx = NDArrayFactory::create<double>('c', {inSize, numUnits});
    auto Wh = NDArrayFactory::create<double>('c', {numUnits, numUnits});
    auto b  = NDArrayFactory::create<double>('c', {2*numUnits}, {0.0,0.0,0.0,0.0,  0.1,0.2,0.3,0.4});

    auto ht = NDArrayFactory::create<double>('c', {bS, numUnits});

    xt.assign(0.1);
    ht_1.assign(0.2);
    Wx.assign(0.3);
    Wh.assign(0.4);

    auto expHt = NDArrayFactory::create<double>('c', {bS, numUnits}, {0.6169093,0.67506987,0.72589741,0.76986654,0.6169093,0.67506987,0.72589741,0.76986654});

    ops::helpers::rnnCell(nd4j::LaunchContext ::defaultContext(), &xt, &Wx, &Wh, &b, &ht_1, &ht);

    ASSERT_TRUE(expHt.isSameShape(ht));
    ASSERT_TRUE(expHt.equalsTo(ht));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, rnnCell_test3) {

    const int bS = 2;
    const int inSize   = 10;
    const int numUnits = 4;

    auto xt = NDArrayFactory::create<double>('c', {bS, inSize});
    auto ht_1 = NDArrayFactory::create<double>('c', {bS, numUnits});
    auto Wx = NDArrayFactory::create<double>('c', {inSize, numUnits});
    auto Wh = NDArrayFactory::create<double>('c', {numUnits, numUnits});
    auto b  = NDArrayFactory::create<double>('c', {2*numUnits}, {0.01,0.02,0.03,0.04,  0.05,0.06,0.07,0.08});

    auto ht = NDArrayFactory::create<double>('c', {bS, numUnits});

    xt.assign(0.1);
    ht_1.assign(0.2);
    Wx.assign(0.3);
    Wh.assign(0.4);

    auto expHt = NDArrayFactory::create<double>('c', {bS, numUnits}, {0.5915195, 0.6043678, 0.6169093, 0.6291452,0.5915195, 0.6043678, 0.6169093, 0.6291452});

    ops::helpers::rnnCell(nd4j::LaunchContext ::defaultContext(), &xt, &Wx, &Wh, &b, &ht_1, &ht);

    ASSERT_TRUE(expHt.isSameShape(ht));
    ASSERT_TRUE(expHt.equalsTo(ht));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, rnnCell_test4) {

    const int bS = 2;
    const int inSize   = 3;
    const int numUnits = 4;

    auto xt = NDArrayFactory::create<double>('c', {bS, inSize});
    auto ht_1 = NDArrayFactory::create<double>('c', {bS, numUnits});
    auto Wx = NDArrayFactory::create<double>('c', {inSize, numUnits});
    auto Wh = NDArrayFactory::create<double>('c', {numUnits, numUnits});
    auto b  = NDArrayFactory::create<double>('c', {2*numUnits});

    auto ht = NDArrayFactory::create<double>('c', {bS, numUnits});

    xt.linspace(0.01, 0.01);
    ht_1 = 0.2;
    Wx   = 0.3;
    Wh   = 0.4;
    b    = 0.25;

    auto expHt = NDArrayFactory::create<double>('c', {bS, numUnits}, {0.68474828, 0.68474828, 0.68474828, 0.68474828,0.69882484, 0.69882484, 0.69882484, 0.69882484});

    ops::helpers::rnnCell(nd4j::LaunchContext ::defaultContext(), &xt, &Wx, &Wh, &b, &ht_1, &ht);

    ASSERT_TRUE(expHt.isSameShape(ht));
    ASSERT_TRUE(expHt.equalsTo(ht));
}

#endif
////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, mmulHelper_test_1) {

    auto x = NDArrayFactory::create<double>('c', {3,3}, {10,11,12,13,14,15,16,17,18});
    auto y = NDArrayFactory::create<double>('c', {3,3}, {1,2,3,4,5,6,7,8,9});
    auto expected = NDArrayFactory::create<double>('c', {3,3}, {138.,171.,204. ,174.,216.,258. ,210.,261.,312.});

    auto result = MmulHelper::mmul(&x, &y, nullptr, 1., 0.);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete result;

}

////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, mmulHelper_test_2) {

    auto x = NDArrayFactory::create<double>('c', {3,3}, {10,11,12,13,14,15,16,17,18});
    auto y = NDArrayFactory::create<double>('c', {3,3}, {1,2,3,4,5,6,7,8,9});
    auto expected = NDArrayFactory::create<double>('c', {3,3}, {138.,171.,204. ,174.,216.,258. ,210.,261.,312.});
    auto result = NDArrayFactory::create<double>('c', {3,3});

    MmulHelper::mmul(&x, &y, &result, 1., 0.);

    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));

}

////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, mmulHelper_test_3) {

    auto x = NDArrayFactory::create<float>('c', {3,4});  x.linspace(1);
    auto y = NDArrayFactory::create<float>('c', {4,5});  y.linspace(1);
    auto expected = NDArrayFactory::create<float>('c', {3,5}, {110.,120.,130.,140.,150.,246.,272.,298.,324.,350.,382.,424.,466.,508.,550.});

    auto result = MmulHelper::mmul(&x, &y, nullptr, 1., 0.);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, mmulHelper_test_4) {

    auto x = NDArrayFactory::create<float>('c', {3,4});  x.linspace(1);
    auto y = NDArrayFactory::create<float>('c', {4,5});  y.linspace(1);
    auto expected = NDArrayFactory::create<float>('c', {3,5}, {110.,120.,130.,140.,150.,246.,272.,298.,324.,350.,382.,424.,466.,508.,550.});
    auto result = NDArrayFactory::create<float>('c', {3,5});

    MmulHelper::mmul(&x, &y, &result, 1., 0.);

    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));
}


////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, mmulHelper_test_5) {

    auto x = NDArrayFactory::create<float>('c', {4,3});  x.linspace(1);
    auto y = NDArrayFactory::create<float>('c', {3,5});  y.linspace(1);
    auto expected = NDArrayFactory::create<float>('c', {4,5}, {46., 52., 58., 64., 70.,100.,115.,130.,145.,160.,154.,178.,202.,226.,250.,208.,241.,274.,307.,340.});

    auto result = MmulHelper::mmul(&x, &y, nullptr, 1., 0.);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, mmulHelper_test_6) {

    auto x = NDArrayFactory::create<float>('c', {4,3});  x.linspace(1);
    auto y = NDArrayFactory::create<float>('c', {3,5});  y.linspace(1);
    auto expected = NDArrayFactory::create<float>('c', {4,5}, {46., 52., 58., 64., 70.,100.,115.,130.,145.,160.,154.,178.,202.,226.,250.,208.,241.,274.,307.,340.});
    auto result = NDArrayFactory::create<float>('c', {4,5});

    MmulHelper::mmul(&x, &y, &result, 1., 0.);

    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));

}

////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, mmulHelper_test_7) {

    auto x = NDArrayFactory::create<float>('c', {4, 1}, {1, 2, 3, 4});
    auto y = NDArrayFactory::create<float>('c', {1, 4}, {1, 2, 3, 4});
    auto exp = NDArrayFactory::create<float>('c', {4, 4}, {1,2, 3, 4,2,4, 6, 8,3,6, 9,12,4,8,12,16});
    auto result = NDArrayFactory::create<float>('c', {4,4});

    MmulHelper::mmul(&x, &y, &result, 1., 0.);

    ASSERT_TRUE(exp.isSameShape(&result));
    ASSERT_TRUE(exp.equalsTo(&result));

}

////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, tensordot_test_1) {

    auto a = NDArrayFactory::create<float>('c', {2, 3, 4});
    auto b = NDArrayFactory::create<float>('c', {2, 5, 3});

    auto c =  MmulHelper::tensorDot(&a, &b, {1}, {2});

    ASSERT_TRUE(c->isSameShape({2,4,2,5}));
    delete c;
}

////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, tensordot_test_2) {

    auto a = NDArrayFactory::create<float>('c', {7, 3, 4, 6});
    auto b = NDArrayFactory::create<float>('c', {2, 5, 3, 8, 4});

    auto c =  MmulHelper::tensorDot(&a, &b, {2,1}, {4,2});

    ASSERT_TRUE(c->isSameShape({7,6,2,5,8}));
    delete c;
}

////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, tensordot_test_3) {

    auto a = NDArrayFactory::create<float>('c', {7, 3, 4, 6});
    auto b = NDArrayFactory::create<float>('c', {2, 5, 3, 8, 4});
    auto c = NDArrayFactory::create<float>('f', {7,6,2,8,5});

    MmulHelper::tensorDot(&a, &b, &c, {2,1}, {4,2}, {0,1,2,4,3});

    ASSERT_TRUE(c.isSameShape({7,6,2,8,5}));
}

////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, tensordot_test_4) {

    auto a = NDArrayFactory::create<float>('c', {7, 3, 4, 3});
    auto b = NDArrayFactory::create<float>('c', {2, 5, 3, 2, 4});
    auto c = NDArrayFactory::create<float>('f', {7,3,2,2,5});
    auto expected = NDArrayFactory::create<float>('c', {7,3,2,2,5}, {  754.5,  2014.5,  3274.5,  4534.5 ,  5794.5,  964.5,  2224.5,  3484.5,  4744.5,  6004.5, 7054.5,  8314.5,  9574.5, 10834.5, 12094.5, 7264.5,  8524.5,  9784.5, 11044.5, 12304.5,  786. ,  2118. ,  3450. ,  4782. ,  6114. , 1008. ,  2340. ,  3672. ,  5004. ,  6336. ,
                                                 7446. ,  8778. , 10110. , 11442. , 12774. , 7668. ,  9000. , 10332. , 11664. , 12996. ,  817.5,  2221.5,  3625.5,  5029.5,  6433.5, 1051.5,  2455.5,  3859.5,  5263.5,  6667.5, 7837.5,  9241.5, 10645.5, 12049.5, 13453.5, 8071.5,  9475.5, 10879.5, 12283.5, 13687.5,
                                                 1888.5,  5740.5,  9592.5, 13444.5, 17296.5, 2530.5,  6382.5, 10234.5, 14086.5, 17938.5,21148.5, 25000.5, 28852.5, 32704.5, 36556.5,21790.5, 25642.5, 29494.5, 33346.5, 37198.5, 1920. ,  5844. ,  9768. , 13692. , 17616. , 2574. ,  6498. , 10422. , 14346. , 18270. ,
                                                21540. , 25464. , 29388. , 33312. , 37236. ,22194. , 26118. , 30042. , 33966. , 37890. , 1951.5,  5947.5,  9943.5, 13939.5, 17935.5, 2617.5,  6613.5, 10609.5, 14605.5, 18601.5,21931.5, 25927.5, 29923.5, 33919.5, 37915.5,22597.5, 26593.5, 30589.5, 34585.5, 38581.5,
                                                 3022.5,  9466.5, 15910.5, 22354.5, 28798.5, 4096.5, 10540.5, 16984.5, 23428.5, 29872.5,35242.5, 41686.5, 48130.5, 54574.5, 61018.5,36316.5, 42760.5, 49204.5, 55648.5, 62092.5, 3054. ,  9570. , 16086. , 22602. , 29118. , 4140. , 10656. , 17172. , 23688. , 30204. ,
                                                35634. , 42150. , 48666. , 55182. , 61698. ,36720. , 43236. , 49752. , 56268. , 62784. , 3085.5,  9673.5, 16261.5, 22849.5, 29437.5, 4183.5, 10771.5, 17359.5, 23947.5, 30535.5,36025.5, 42613.5, 49201.5, 55789.5, 62377.5,37123.5, 43711.5, 50299.5, 56887.5, 63475.5,
                                                 4156.5, 13192.5, 22228.5, 31264.5, 40300.5, 5662.5, 14698.5, 23734.5, 32770.5, 41806.5,49336.5, 58372.5, 67408.5, 76444.5, 85480.5,50842.5, 59878.5, 68914.5, 77950.5, 86986.5, 4188. , 13296. , 22404. , 31512. , 40620. , 5706. , 14814. , 23922. , 33030. , 42138. ,
                                                49728. , 58836. , 67944. , 77052. , 86160. ,51246. , 60354. , 69462. , 78570. , 87678. , 4219.5, 13399.5, 22579.5, 31759.5, 40939.5, 5749.5, 14929.5, 24109.5, 33289.5, 42469.5,50119.5, 59299.5, 68479.5, 77659.5, 86839.5,51649.5, 60829.5, 70009.5, 79189.5, 88369.5,
                                                 5290.5, 16918.5, 28546.5, 40174.5, 51802.5, 7228.5, 18856.5, 30484.5, 42112.5, 53740.5,63430.5, 75058.5, 86686.5, 98314.5,109942.5,65368.5, 76996.5, 88624.5,100252.5,111880.5, 5322. , 17022. , 28722. , 40422. , 52122. , 7272. , 18972. , 30672. , 42372. , 54072. ,
                                                63822. , 75522. , 87222. , 98922. ,110622. ,65772. , 77472. , 89172. ,100872. ,112572. , 5353.5, 17125.5, 28897.5, 40669.5, 52441.5, 7315.5, 19087.5, 30859.5, 42631.5, 54403.5,64213.5, 75985.5, 87757.5, 99529.5,111301.5,66175.5, 77947.5, 89719.5,101491.5,113263.5,
                                                 6424.5, 20644.5, 34864.5, 49084.5, 63304.5, 8794.5, 23014.5, 37234.5, 51454.5, 65674.5,77524.5, 91744.5,105964.5,120184.5,134404.5,79894.5, 94114.5,108334.5,122554.5,136774.5, 6456. , 20748. , 35040. , 49332. , 63624. , 8838. , 23130. , 37422. , 51714. , 66006. ,
                                                77916. , 92208. ,106500. ,120792. ,135084. ,80298. , 94590. ,108882. ,123174. ,137466. , 6487.5, 20851.5, 35215.5, 49579.5, 63943.5, 8881.5, 23245.5, 37609.5, 51973.5, 66337.5,78307.5, 92671.5,107035.5,121399.5,135763.5,80701.5, 95065.5,109429.5,123793.5,138157.5,
                                                 7558.5, 24370.5, 41182.5, 57994.5, 74806.5,10360.5, 27172.5, 43984.5, 60796.5, 77608.5,91618.5,108430.5,125242.5,142054.5,158866.5,94420.5,111232.5,128044.5,144856.5,161668.5, 7590. , 24474. , 41358. , 58242. , 75126. ,10404. , 27288. , 44172. , 61056. , 77940. ,
                                                92010. ,108894. ,125778. ,142662. ,159546. ,94824. ,111708. ,128592. ,145476. ,162360. , 7621.5, 24577.5, 41533.5, 58489.5, 75445.5,10447.5, 27403.5, 44359.5, 61315.5, 78271.5,92401.5,109357.5,126313.5,143269.5,160225.5,95227.5,112183.5,129139.5,146095.5,163051.5});

    a.linspace(0.5, 0.5);
    b.linspace(0.5, 0.5);

    MmulHelper::tensorDot(&a, &b, &c, {2,1}, {4,2}, {0,1,2,4,3});

    ASSERT_TRUE(c.isSameShape(expected));
    ASSERT_TRUE(c.equalsTo(expected));
}

////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, tensordot_test_5) {

    auto a = NDArrayFactory::create<float>('c', {2, 3});
    auto b = NDArrayFactory::create<float>('c', {3, 4});
    auto c = NDArrayFactory::create<float>('f', {2, 4});
    auto expected = NDArrayFactory::create<float>('c', {2, 4}, {9.5,11.,12.5 ,14.,20.75 ,24.5,28.25,32.});

    a.linspace(0.5, 0.5);
    b.linspace(0.5, 0.5);

    MmulHelper::tensorDot(&a, &b, &c, {1}, {0});
    // c.printIndexedBuffer();

    ASSERT_TRUE(c.isSameShape(expected));
    ASSERT_TRUE(c.equalsTo(expected));
}

////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, tensordot_test_6) {

    int bS=2, iH=3,iW=2,  iC=2,mC=2,  kH=2,kW=2;
    int       oC=iC*mC;
    int       oH=3,oW=2;

    auto a = NDArrayFactory::create<float>('c', {bS, iC, kH, kW, oH, oW});
    auto b = NDArrayFactory::create<float>('c', {kH, kW, iC, mC});
    auto c = NDArrayFactory::create<float>('c', {bS, oH, oW, iC*mC});
    auto expected = NDArrayFactory::create<float>('c', {bS, oH, oW, iC*mC}, {100.,110.,336.,370.,107.,118.,345.,380.,114.,126.,354.,390.,121.,134.,363.,400.,128.,142.,372.,410.,135.,150.,381.,420.,
                                                       436.,494.,768.,850.,443.,502.,777.,860.,450.,510.,786.,870.,457.,518.,795.,880.,464.,526.,804.,890.,471.,534.,813.,900.});

    a.linspace(0.5, 0.5);
    b.linspace(0.5, 0.5);

    auto cR = c.reshape(a.ordering(), {bS, oH, oW, iC, mC});

    // [iC, bS*oH*oW, kW*kH] x [iC, kH*kW, mC] = [iC, bS*oH*oW, mC]
    MmulHelper::tensorDot(&a, &b, &cR, {{1,0,4,5,2,3}, {iC,bS*oH*oW,kW*kH}},  {{2,0,1,3},{iC,kH*kW,mC}},  {{3,0,1,2,4},{iC, bS*oH*oW, mC}});

    ASSERT_TRUE(c.isSameShape(expected));
    ASSERT_TRUE(c.equalsTo(expected));
}

////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, mmmulHelperAgain) {
    auto x = NDArrayFactory::create<float>('c', {128, 156});
    auto y = NDArrayFactory::create<float>('c', {156, 256});
    auto z = NDArrayFactory::create<float>('c', {128, 256});
    auto e = NDArrayFactory::create<float>('c', {128, 256});

    x.assign(1.0f);
    y.assign(1.0f);
    e.assign(156.0f);

    MmulHelper::mmul(&x, &y, &z);

    ASSERT_TRUE(e.isSameShape(z));
    ASSERT_TRUE(e.equalsTo(z));
}

////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, OpArgsHolder_test1) {

    auto x1 = NDArrayFactory::create<float>('c', {1, 1});
    auto x2 = NDArrayFactory::create<float>('c', {2, 2});
    auto x3 = NDArrayFactory::create<double>('c', {3, 3});

    OpArgsHolder holder1({&x1});
    OpArgsHolder holder2({&x1,&x2,&x3}, {4.f, 5.f}, {6});

    ASSERT_TRUE(holder1.getNumInArrs() == 1);
    ASSERT_TRUE(holder1.getNumTArgs()  == 0);
    ASSERT_TRUE(holder1.getNumIArgs()  == 0);

    ASSERT_TRUE(holder2.getNumInArrs() == 3);
    ASSERT_TRUE(holder2.getNumTArgs()  == 2);
    ASSERT_TRUE(holder2.getNumIArgs()  == 1);

    const std::vector<bool>& isArrAlloc1 = holder1.getAllocInfo();
    ASSERT_TRUE(isArrAlloc1.size() == 0);

    const std::vector<bool>& isArrAlloc2 = holder2.getAllocInfo();
    ASSERT_TRUE(isArrAlloc2.size() == 0);
}

////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, OpArgsHolder_test2) {

    auto x1 = NDArrayFactory::create<float>('c', {1, 1});
    auto x2 = NDArrayFactory::create<float>('c', {2, 2});
    auto x3 = NDArrayFactory::create<double>('c', {3, 3});
    auto grad = NDArrayFactory::create<float>('c', {2, 3});

    OpArgsHolder holderFF({&x1,&x2,&x3}, {4.f, 5.f}, {6});
    OpArgsHolder holderBP1 = holderFF.createArgsHolderForBP({&grad});
    OpArgsHolder holderBP2 = holderFF.createArgsHolderForBP({&grad}, true);

    ASSERT_TRUE(holderBP1.getNumInArrs() == 4);
    ASSERT_TRUE(holderBP1.getNumTArgs()  == 2);
    ASSERT_TRUE(holderBP1.getNumIArgs()  == 1);
    ASSERT_TRUE(holderBP2.getNumInArrs() == 4);
    ASSERT_TRUE(holderBP2.getNumTArgs()  == 2);
    ASSERT_TRUE(holderBP2.getNumIArgs()  == 1);

    const std::vector<bool>& isArrAllocBP1 = holderBP1.getAllocInfo();
    ASSERT_TRUE(isArrAllocBP1.size() == 0);

    const std::vector<bool>& isArrAllocBP2 = holderBP2.getAllocInfo();
    for(int i = 0; i < holderFF.getNumInArrs(); ++i) {
        ASSERT_TRUE(static_cast<bool>(isArrAllocBP2[i]) == true);
    }

    ASSERT_TRUE(static_cast<bool>(isArrAllocBP2[holderFF.getNumInArrs()+1]) == false);
}

//////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, OpArgsHolder_test3) {

    auto input  = NDArrayFactory::create<double>('c', {2, 3}, {1.,2.,3.,4.,5.,6.});
    auto gradO  = NDArrayFactory::create<double>('c', {4, 9});
    auto exp    = NDArrayFactory::create<double>('c', {4, 9}, {1, 2, 3, 1, 2, 3, 1, 2, 3,4, 5, 6, 4, 5, 6, 4, 5, 6,1, 2, 3, 1, 2, 3, 1, 2, 3,4, 5, 6, 4, 5, 6, 4, 5, 6});
    auto gradIExp = NDArrayFactory::create<double>('c', {2, 3}, {0.78, 0.84, 0.9,1.32, 1.38, 1.44});

    gradO.linspace(0.01, 0.01);

    OpArgsHolder holderFF({&input}, {}, {2, 3});
    nd4j::ops::tile opFF;                                              // the kind of op doesn't matter, we simply check here whether op.execute() works with OpArgsHolder correctly
    auto results = opFF.execute(holderFF);
    auto tiled = results->at(0);
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(tiled));
    ASSERT_TRUE(exp.equalsTo(tiled));
    delete results;

    OpArgsHolder holderBP = holderFF.createArgsHolderForBP({&gradO}, true);
    nd4j::ops::tile_bp opBP;
    results = opBP.execute(holderBP);
    auto gradI = results->at(0);
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(gradIExp.isSameShape(gradI));
    ASSERT_TRUE(gradIExp.equalsTo(gradI));
    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, checkGrad_test1) {

    auto     x = NDArrayFactory::create<double>('c', {2, 3}, {0.1, 0.2, 0.3, 0.4, 0.5 ,0.6});
    auto gradO = NDArrayFactory::create<double>('c', {2, 3});

    const OpArgsHolder argsHolderFF({&x}, {}, {});
    const OpArgsHolder argsHolderBP({&x, &gradO}, {}, {});

    nd4j::ops::sigmoid opFF;
    nd4j::ops::sigmoid_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

//////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, checkGrad_test2) {

    auto       x = NDArrayFactory::create<double>('c', {1, 1, 3, 3});
    auto weights = NDArrayFactory::create<double>('c', {2, 1, 2, 2});
    auto   gradO = NDArrayFactory::create<double>('c', {1, 2, 3, 3});

    x.linspace(1);
    weights.linspace(0.1, 0.1);
    weights.permutei({2,3,1,0});

    const OpArgsHolder argsHolderFF({&x, &weights},         {}, {2, 2, 1, 1, 0, 0, 1, 1, 1});
    const OpArgsHolder argsHolderBP({&x, &weights, &gradO}, {}, {2, 2, 1, 1, 0, 0, 1, 1, 1});

    nd4j::ops::conv2d opFF;
    nd4j::ops::conv2d_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

//////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, checkGrad_test3) {

    auto       x = NDArrayFactory::create<double>('c', {1, 1, 3, 3});
    auto weights = NDArrayFactory::create<double>('c', {2, 1, 2, 2});
    auto    bias = NDArrayFactory::create<double>('c', {2, 1});
    auto   gradO = NDArrayFactory::create<double>('c', {1, 2, 3, 3});

    x.linspace(1);
    weights.linspace(0.1, 0.1);
    bias = 0.5;
    weights.permutei({2,3,1,0});

    const OpArgsHolder argsHolderFF({&x, &weights, &bias},         {}, {2, 2, 1, 1, 0, 0, 1, 1, 1});
    const OpArgsHolder argsHolderBP({&x, &weights, &bias, &gradO}, {}, {2, 2, 1, 1, 0, 0, 1, 1, 1});

    nd4j::ops::conv2d opFF;
    nd4j::ops::conv2d_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

//////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, checkGrad_test4) {

    auto       x = NDArrayFactory::create<double>('c', {1, 1, 3, 3});
    auto weights = NDArrayFactory::create<double>('c', {2, 1, 2, 2});
    auto    bias = NDArrayFactory::create<double>('c', {2, 1});
    auto   gradO = NDArrayFactory::create<double>('c', {1, 2, 3, 3});

    x.linspace(1);
    weights.linspace(0.1, 0.1);
    bias = 0.5;
    weights.permutei({2,3,1,0});

    const OpArgsHolder argsHolderFF({&x, &weights, &bias},         {}, {2, 2, 1, 1, 0, 0, 1, 1, 1});
    const OpArgsHolder argsHolderBP({&x, &weights, &bias, &gradO}, {}, {2, 2, 1, 1, 0, 0, 1, 1, 1});

    nd4j::ops::conv2d opFF;
    nd4j::ops::conv2d_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP, {1, 0, 1});

    ASSERT_TRUE(isGradCorrect);
}

//////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, checkGrad_test5) {

    auto       x = NDArrayFactory::create<double>('c', {1, 1, 3, 3});
    auto weights = NDArrayFactory::create<double>('c', {2, 1, 2, 2});
    auto    bias = NDArrayFactory::create<double>('c', {2, 1});
    auto   gradO = NDArrayFactory::create<double>('c', {1, 2, 3, 3});

    x.linspace(1);
    weights.linspace(0.1, 0.1);
    bias = 0.5;
    weights.permutei({2,3,1,0});

    const OpArgsHolder argsHolderFF({&x, &weights, &bias},         {}, {2, 2, 1, 1, 0, 0, 1, 1, 1});
    const OpArgsHolder argsHolderBP({&x, &weights, &bias, &gradO}, {}, {2, 2, 1, 1, 0, 0, 1, 1, 1});

    nd4j::ops::conv2d opFF;
    nd4j::ops::conv2d_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP, {1, 1, 1}, {0.5, 1});

    ASSERT_TRUE(isGradCorrect);
}

//////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, checkGrad_test6) {

    auto       x = NDArrayFactory::create<double>('c', {1, 1, 3, 3});
    auto weights = NDArrayFactory::create<double>('c', {2, 1, 2, 2});
    auto    bias = NDArrayFactory::create<double>('c', {2, 1});
    auto   gradO = NDArrayFactory::create<double>('c', {1, 2, 3, 3});

    x.linspace(1);
    weights.linspace(0.1, 0.1);
    bias = 0.5;
    weights.permutei({2,3,1,0});

    const OpArgsHolder argsHolderFF({&x, &weights, &bias},         {}, {2, 2, 1, 1, 0, 0, 1, 1, 1});
    const OpArgsHolder argsHolderBP({&x, &weights, &bias, &gradO}, {}, {2, 2, 1, 1, 0, 0, 1, 1, 1});

    nd4j::ops::conv2d opFF;
    nd4j::ops::conv2d_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP, {1, 0, 1}, {0.5, 1}, GradCheck::MEAN);

    ASSERT_TRUE(isGradCorrect);
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, softMaxForVector_test1) {

    auto input = NDArrayFactory::create<double>('c', {1,5}, {1,2,3,4,5});
    auto output = NDArrayFactory::create<double>('c', {1,5});
    auto expOutput = NDArrayFactory::create<double>('c', {1,5});
    expOutput = 1;

    ops::helpers::softmax(nd4j::LaunchContext ::defaultContext(), input, output, 0);

    ASSERT_TRUE(output.equalsTo(&expOutput));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, softMaxForVector_test2) {

    auto input = NDArrayFactory::create<double>('c', {5,1}, {1,2,3,4,5});
    auto output = NDArrayFactory::create<double>('c', {5,1});
    auto expOutput = NDArrayFactory::create<double>('c', {5,1}, {0.01165623,  0.03168492,  0.08612854,  0.23412166,  0.63640865});

    ops::helpers::softmax(nd4j::LaunchContext ::defaultContext(), input, output, 0);

    ASSERT_TRUE(output.equalsTo(&expOutput));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, softMaxForVector_test3) {

    auto input= NDArrayFactory::create<double>('c', {5}, {1,2,3,4,5});
    auto output = NDArrayFactory::create<double>('c', {5});
    auto expOutput = NDArrayFactory::create<double>('c', {5}, {0.01165623,  0.03168492,  0.08612854,  0.23412166,  0.63640865});

    ops::helpers::softmax(nd4j::LaunchContext ::defaultContext(), input, output, 0);

    ASSERT_TRUE(output.equalsTo(&expOutput));
}

//////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, softMaxForVector_test4) {

    NDArray input('c', {1500}, nd4j::DataType::DOUBLE);
    NDArray output('c', {1500}, nd4j::DataType::DOUBLE);
    NDArray expOutput('c', {1500}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.00001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001,
0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001,0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001,
0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002,0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002,
0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000002, 0.000003, 0.000003, 0.000003, 0.000003, 0.000003, 0.000003, 0.000003, 0.000003, 0.000003, 0.000003, 0.000003, 0.000003, 0.000003, 0.000003, 0.000003, 0.000003, 0.000003, 0.000003, 0.000003, 0.000003,0.000003, 0.000003, 0.000003, 0.000003, 0.000003, 0.000003, 0.000003, 0.000003, 0.000003, 0.000003, 0.000003, 0.000003, 0.000003, 0.000004, 0.000004, 0.000004, 0.000004, 0.000004, 0.000004, 0.000004, 0.000004, 0.000004, 0.000004, 0.000004, 0.000004, 0.000004, 0.000004, 0.000004,
0.000004, 0.000004, 0.000004, 0.000004, 0.000004, 0.000004, 0.000004, 0.000004, 0.000004, 0.000004, 0.000005, 0.000005, 0.000005, 0.000005, 0.000005, 0.000005, 0.000005, 0.000005, 0.000005, 0.000005, 0.000005, 0.000005, 0.000005, 0.000005, 0.000005, 0.000005, 0.000005, 0.000005,0.000005, 0.000005, 0.000006, 0.000006, 0.000006, 0.000006, 0.000006, 0.000006, 0.000006, 0.000006, 0.000006, 0.000006, 0.000006, 0.000006, 0.000006, 0.000006, 0.000006, 0.000006, 0.000006, 0.000007, 0.000007, 0.000007, 0.000007, 0.000007, 0.000007, 0.000007, 0.000007, 0.000007,
0.000007, 0.000007, 0.000007, 0.000007, 0.000007, 0.000008, 0.000008, 0.000008, 0.000008, 0.000008, 0.000008, 0.000008, 0.000008, 0.000008, 0.000008, 0.000008, 0.000008, 0.000008, 0.000009, 0.000009, 0.000009, 0.000009, 0.000009, 0.000009, 0.000009, 0.000009, 0.000009, 0.000009,0.000009, 0.000010, 0.000010, 0.000010, 0.000010, 0.000010, 0.000010, 0.000010, 0.000010, 0.000010, 0.000010, 0.000011, 0.000011, 0.000011, 0.000011, 0.000011, 0.000011, 0.000011, 0.000011, 0.000011, 0.000012, 0.000012, 0.000012, 0.000012, 0.000012, 0.000012, 0.000012, 0.000012,
0.000012, 0.000013, 0.000013, 0.000013, 0.000013, 0.000013, 0.000013, 0.000013, 0.000014, 0.000014, 0.000014, 0.000014, 0.000014, 0.000014, 0.000014, 0.000015, 0.000015, 0.000015, 0.000015, 0.000015, 0.000015, 0.000015, 0.000016, 0.000016, 0.000016, 0.000016, 0.000016, 0.000016,0.000017, 0.000017, 0.000017, 0.000017, 0.000017, 0.000017, 0.000018, 0.000018, 0.000018, 0.000018, 0.000018, 0.000018, 0.000019, 0.000019, 0.000019, 0.000019, 0.000019, 0.000020, 0.000020, 0.000020, 0.000020, 0.000020, 0.000021, 0.000021, 0.000021, 0.000021, 0.000021, 0.000022,
0.000022, 0.000022, 0.000022, 0.000023, 0.000023, 0.000023, 0.000023, 0.000023, 0.000024, 0.000024, 0.000024, 0.000024, 0.000025, 0.000025, 0.000025, 0.000025, 0.000026, 0.000026, 0.000026, 0.000026, 0.000027, 0.000027, 0.000027, 0.000028, 0.000028, 0.000028, 0.000028, 0.000029,0.000029, 0.000029, 0.000030, 0.000030, 0.000030, 0.000030, 0.000031, 0.000031, 0.000031, 0.000032, 0.000032, 0.000032, 0.000033, 0.000033, 0.000033, 0.000034, 0.000034, 0.000034, 0.000035, 0.000035, 0.000035, 0.000036, 0.000036, 0.000036, 0.000037, 0.000037, 0.000038, 0.000038,
0.000038, 0.000039, 0.000039, 0.000039, 0.000040, 0.000040, 0.000041, 0.000041, 0.000041, 0.000042, 0.000042, 0.000043, 0.000043, 0.000044, 0.000044, 0.000044, 0.000045, 0.000045, 0.000046, 0.000046, 0.000047, 0.000047, 0.000048, 0.000048, 0.000049, 0.000049, 0.000050, 0.000050,0.000051, 0.000051, 0.000052, 0.000052, 0.000053, 0.000053, 0.000054, 0.000054, 0.000055, 0.000055, 0.000056, 0.000057, 0.000057, 0.000058, 0.000058, 0.000059, 0.000059, 0.000060, 0.000061, 0.000061, 0.000062, 0.000063, 0.000063, 0.000064, 0.000064, 0.000065, 0.000066, 0.000066,
0.000067, 0.000068, 0.000068, 0.000069, 0.000070, 0.000070, 0.000071, 0.000072, 0.000073, 0.000073, 0.000074, 0.000075, 0.000076, 0.000076, 0.000077, 0.000078, 0.000079, 0.000079, 0.000080, 0.000081, 0.000082, 0.000083, 0.000084, 0.000084, 0.000085, 0.000086, 0.000087, 0.000088,0.000089, 0.000090, 0.000090, 0.000091, 0.000092, 0.000093, 0.000094, 0.000095, 0.000096, 0.000097, 0.000098, 0.000099, 0.000100, 0.000101, 0.000102, 0.000103, 0.000104, 0.000105, 0.000106, 0.000107, 0.000108, 0.000109, 0.000111, 0.000112, 0.000113, 0.000114, 0.000115, 0.000116,
0.000117, 0.000119, 0.000120, 0.000121, 0.000122, 0.000123, 0.000125, 0.000126, 0.000127, 0.000128, 0.000130, 0.000131, 0.000132, 0.000134, 0.000135, 0.000136, 0.000138, 0.000139, 0.000141, 0.000142, 0.000143, 0.000145, 0.000146, 0.000148, 0.000149, 0.000151, 0.000152, 0.000154,0.000155, 0.000157, 0.000158, 0.000160, 0.000162, 0.000163, 0.000165, 0.000167, 0.000168, 0.000170, 0.000172, 0.000173, 0.000175, 0.000177, 0.000179, 0.000180, 0.000182, 0.000184, 0.000186, 0.000188, 0.000190, 0.000192, 0.000194, 0.000195, 0.000197, 0.000199, 0.000201, 0.000203,
0.000205, 0.000208, 0.000210, 0.000212, 0.000214, 0.000216, 0.000218, 0.000220, 0.000223, 0.000225, 0.000227, 0.000229, 0.000232, 0.000234, 0.000236, 0.000239, 0.000241, 0.000244, 0.000246, 0.000248, 0.000251, 0.000253, 0.000256, 0.000259, 0.000261, 0.000264, 0.000266, 0.000269,0.000272, 0.000275, 0.000277, 0.000280, 0.000283, 0.000286, 0.000289, 0.000292, 0.000295, 0.000297, 0.000300, 0.000303, 0.000307, 0.000310, 0.000313, 0.000316, 0.000319, 0.000322, 0.000325, 0.000329, 0.000332, 0.000335, 0.000339, 0.000342, 0.000346, 0.000349, 0.000353, 0.000356,
0.000360, 0.000363, 0.000367, 0.000371, 0.000374, 0.000378, 0.000382, 0.000386, 0.000390, 0.000394, 0.000398, 0.000402, 0.000406, 0.000410, 0.000414, 0.000418, 0.000422, 0.000426, 0.000431, 0.000435, 0.000439, 0.000444, 0.000448, 0.000453, 0.000457, 0.000462, 0.000467, 0.000471,0.000476, 0.000481, 0.000486, 0.000490, 0.000495, 0.000500, 0.000505, 0.000510, 0.000516, 0.000521, 0.000526, 0.000531, 0.000537, 0.000542, 0.000547, 0.000553, 0.000559, 0.000564, 0.000570, 0.000576, 0.000581, 0.000587, 0.000593, 0.000599, 0.000605, 0.000611, 0.000617, 0.000623,
0.000630, 0.000636, 0.000642, 0.000649, 0.000655, 0.000662, 0.000669, 0.000675, 0.000682, 0.000689, 0.000696, 0.000703, 0.000710, 0.000717, 0.000724, 0.000732, 0.000739, 0.000746, 0.000754, 0.000762, 0.000769, 0.000777, 0.000785, 0.000793, 0.000801, 0.000809, 0.000817, 0.000825,0.000833, 0.000842, 0.000850, 0.000859, 0.000867, 0.000876, 0.000885, 0.000894, 0.000903, 0.000912, 0.000921, 0.000930, 0.000939, 0.000949, 0.000958, 0.000968, 0.000978, 0.000988, 0.000998, 0.001008, 0.001018, 0.001028, 0.001038, 0.001049, 0.001059, 0.001070, 0.001081, 0.001092,
0.001103, 0.001114, 0.001125, 0.001136, 0.001148, 0.001159, 0.001171, 0.001182, 0.001194, 0.001206, 0.001218, 0.001231, 0.001243, 0.001256, 0.001268, 0.001281, 0.001294, 0.001307, 0.001320, 0.001333, 0.001347, 0.001360, 0.001374, 0.001388, 0.001402, 0.001416, 0.001430, 0.001444,0.001459, 0.001473, 0.001488, 0.001503, 0.001518, 0.001534, 0.001549, 0.001565, 0.001580, 0.001596, 0.001612, 0.001628, 0.001645, 0.001661, 0.001678, 0.001695, 0.001712, 0.001729, 0.001746, 0.001764, 0.001782, 0.001800, 0.001818, 0.001836, 0.001854, 0.001873, 0.001892, 0.001911,
0.001930, 0.001950, 0.001969, 0.001989, 0.002009, 0.002029, 0.002049, 0.002070, 0.002091, 0.002112, 0.002133, 0.002155, 0.002176, 0.002198, 0.002220, 0.002242, 0.002265, 0.002288, 0.002311, 0.002334, 0.002357, 0.002381, 0.002405, 0.002429, 0.002454, 0.002478, 0.002503, 0.002528,0.002554, 0.002579, 0.002605, 0.002632, 0.002658, 0.002685, 0.002712, 0.002739, 0.002767, 0.002794, 0.002822, 0.002851, 0.002879, 0.002908, 0.002938, 0.002967, 0.002997, 0.003027, 0.003057, 0.003088, 0.003119, 0.003151, 0.003182, 0.003214, 0.003247, 0.003279, 0.003312, 0.003345,
0.003379, 0.003413, 0.003447, 0.003482, 0.003517, 0.003552, 0.003588, 0.003624, 0.003660, 0.003697, 0.003734, 0.003772, 0.003810, 0.003848, 0.003887, 0.003926, 0.003965, 0.004005, 0.004045, 0.004086, 0.004127, 0.004169, 0.004211, 0.004253, 0.004296, 0.004339, 0.004382, 0.004426,0.004471, 0.004516, 0.004561, 0.004607, 0.004653, 0.004700, 0.004747, 0.004795, 0.004843, 0.004892, 0.004941, 0.004991, 0.005041, 0.005092, 0.005143, 0.005194, 0.005247, 0.005299, 0.005353, 0.005406, 0.005461, 0.005516, 0.005571, 0.005627, 0.005684, 0.005741, 0.005798, 0.005857,
0.005916, 0.005975, 0.006035, 0.006096, 0.006157, 0.006219, 0.006281, 0.006345, 0.006408, 0.006473, 0.006538, 0.006603, 0.006670, 0.006737, 0.006805, 0.006873, 0.006942, 0.007012, 0.007082, 0.007153, 0.007225, 0.007298, 0.007371, 0.007445, 0.007520, 0.007596, 0.007672, 0.007749,0.007827, 0.007906, 0.007985, 0.008065, 0.008147, 0.008228, 0.008311, 0.008395, 0.008479, 0.008564, 0.008650, 0.008737, 0.008825, 0.008914, 0.009003, 0.009094, 0.009185, 0.009277, 0.009371, 0.009465, 0.009560, 0.009656, 0.009753, 0.009851, 0.009950}, nd4j::DataType::DOUBLE);
    input.linspace(0.01, 0.01);

    ops::helpers::softmax(nd4j::LaunchContext ::defaultContext(), input, output, 0);

    ASSERT_TRUE(output.equalsTo(&expOutput));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, logSoftMaxForVector_test1) {

    auto input = NDArrayFactory::create<double>('c', {1,5}, {1,2,3,4,5});
    auto output = NDArrayFactory::create<double>('c', {1,5});
    auto expOutput = NDArrayFactory::create<double>('c', {1,5});
    expOutput = 0;

    ops::helpers::logSoftmax(nd4j::LaunchContext ::defaultContext(), input, output, 0);

    ASSERT_TRUE(output.equalsTo(&expOutput));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, logSoftMaxForVector_test2) {

    auto input= NDArrayFactory::create<double>('c', {5,1}, {1,2,3,4,5});
    auto output = NDArrayFactory::create<double>('c', {5,1});
    auto expOutput = NDArrayFactory::create<double>('c', {5,1}, {-4.4519144, -3.4519144, -2.4519144, -1.4519144, -0.4519144});

    ops::helpers::logSoftmax(nd4j::LaunchContext ::defaultContext(), input, output, 0);

    ASSERT_TRUE(output.equalsTo(&expOutput));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, logSoftMaxForVector_test3) {

    auto input= NDArrayFactory::create<double>('c', {5}, {1,2,3,4,5});
    auto output = NDArrayFactory::create<double>('c', {5});
    auto expOutput = NDArrayFactory::create<double>('c', {5}, {-4.4519144, -3.4519144, -2.4519144, -1.4519144, -0.4519144});

    ops::helpers::logSoftmax(nd4j::LaunchContext ::defaultContext(), input, output, 0);

    ASSERT_TRUE(output.equalsTo(&expOutput));
}

//////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, logSoftMaxForVector_test4) {

    NDArray input('c', {1500}, nd4j::DataType::DOUBLE);
    NDArray output('c', {1500}, nd4j::DataType::DOUBLE);
    NDArray expOutput('c', {1500}, {-8.154773, -8.153772, -8.152773, -8.151772, -8.150773, -8.149773, -8.148773, -8.147773, -8.146772, -8.145773, -8.144773, -8.143773, -8.142773, -8.141773, -8.140773, -8.139772, -8.138773, -8.137773, -8.136773, -8.135773, -8.134773, -8.133773, -8.132772, -8.131773, -8.130773, -8.129773, -8.128773, -8.127772, -8.126773, -8.125772, -8.124773, -8.123773, -8.122773, -8.121773, -8.120772, -8.119773, -8.118773, -8.117773, -8.116773, -8.115773, -8.114773, -8.113772, -8.112773, -8.111773, -8.110773, -8.109773, -8.108773, -8.107773, -8.106772, -8.105773, -8.104773, -8.103773, -8.102773, -8.101772, -8.100773, -8.099772, -8.098773, -8.097773, -8.096773, -8.095773, -8.094772, -8.093773, -8.092772, -8.091773, -8.090773, -8.089773, -8.088773, -8.087772, -8.086773, -8.085773, -8.084773, -8.083773, -8.082773, -8.081773, -8.080772, -8.079773, -8.078773, -8.077773, -8.076773, -8.075773, -8.074773, -8.073772, -8.072773, -8.071773, -8.070773, -8.069773, -8.068772, -8.067773, -8.066772, -8.065773, -8.064773, -8.063773, -8.062773, -8.061772, -8.060773, -8.059772, -8.058773, -8.057773, -8.056773, -8.055773, -8.054772,
-8.053773, -8.052773, -8.051773, -8.050773, -8.049773, -8.048773, -8.047772, -8.046773, -8.045773, -8.044773, -8.043773, -8.042773, -8.041773, -8.040772, -8.039773, -8.038773, -8.037773, -8.036773, -8.035772, -8.034773, -8.033772, -8.032773, -8.031773, -8.030773, -8.029773, -8.028772, -8.027773, -8.026772, -8.025773, -8.024773, -8.023773, -8.022773, -8.021772, -8.020773, -8.019773, -8.018773, -8.017773, -8.016773, -8.015773, -8.014772, -8.013773, -8.012773, -8.011773, -8.010773, -8.009773, -8.008773, -8.007772, -8.006773, -8.005773, -8.004773, -8.003773, -8.002772, -8.001773, -8.000772, -7.999773, -7.998773, -7.997773, -7.996773, -7.995773, -7.994773, -7.993773, -7.992773, -7.991773, -7.990773, -7.989773, -7.988773, -7.987773, -7.986773, -7.985773, -7.984773, -7.983773, -7.982773, -7.981773, -7.980773, -7.979773, -7.978773, -7.977773, -7.976773, -7.975773, -7.974773, -7.973773, -7.972773, -7.971773, -7.970773, -7.969773, -7.968773, -7.967773, -7.966773, -7.965773, -7.964773, -7.963773, -7.962773, -7.961773, -7.960773, -7.959773, -7.958773, -7.957773, -7.956773, -7.955773, -7.954773, -7.953773, -7.952773,
-7.951773, -7.950773, -7.949773, -7.948773, -7.947773, -7.946773, -7.945773, -7.944773, -7.943773, -7.942773, -7.941773, -7.940773, -7.939773, -7.938773, -7.937773, -7.936773, -7.935773, -7.934773, -7.933773, -7.932773, -7.931773, -7.930773, -7.929773, -7.928773, -7.927773, -7.926773, -7.925773, -7.924773, -7.923773, -7.922773, -7.921773, -7.920773, -7.919773, -7.918773, -7.917773, -7.916773, -7.915773, -7.914773, -7.913773, -7.912773, -7.911773, -7.910773, -7.909773, -7.908773, -7.907773, -7.906773, -7.905773, -7.904773, -7.903773, -7.902773, -7.901773, -7.900773, -7.899773, -7.898773, -7.897773, -7.896773, -7.895773, -7.894773, -7.893773, -7.892773, -7.891773, -7.890773, -7.889773, -7.888773, -7.887773, -7.886773, -7.885773, -7.884773, -7.883773, -7.882773, -7.881773, -7.880773, -7.879773, -7.878773, -7.877773, -7.876773, -7.875773, -7.874773, -7.873773, -7.872773, -7.871773, -7.870773, -7.869773, -7.868773, -7.867773, -7.866773, -7.865773, -7.864773, -7.863773, -7.862773, -7.861773, -7.860773, -7.859773, -7.858773, -7.857773, -7.856773, -7.855773, -7.854773, -7.853773, -7.852773, -7.851773, -7.850773, -7.849773,
-7.848773, -7.847773, -7.846773, -7.845773, -7.844773, -7.843773, -7.842773, -7.841773, -7.840773, -7.839773, -7.838773, -7.837773, -7.836773, -7.835773, -7.834773, -7.833773, -7.832773, -7.831773, -7.830773, -7.829773, -7.828773, -7.827773, -7.826773, -7.825773, -7.824773, -7.823773, -7.822773, -7.821773, -7.820773, -7.819773, -7.818773, -7.817773, -7.816773, -7.815773, -7.814773, -7.813773, -7.812773, -7.811773, -7.810773, -7.809773, -7.808773, -7.807773, -7.806773, -7.805773, -7.804773, -7.803773, -7.802773, -7.801773, -7.800773, -7.799773, -7.798773, -7.797773, -7.796773, -7.795773, -7.794773, -7.793773, -7.792773, -7.791773, -7.790773, -7.789773, -7.788773, -7.787773, -7.786773, -7.785773, -7.784773, -7.783773, -7.782773, -7.781773, -7.780773, -7.779773, -7.778773, -7.777773, -7.776773, -7.775773, -7.774773, -7.773773, -7.772773, -7.771773, -7.770773, -7.769773, -7.768773, -7.767773, -7.766773, -7.765773, -7.764773, -7.763773, -7.762773, -7.761773, -7.760773, -7.759773, -7.758773, -7.757773, -7.756773, -7.755773, -7.754773, -7.753773, -7.752773, -7.751773, -7.750773, -7.749773, -7.748773, -7.747773, -7.746773,
-7.745773, -7.744773, -7.743773, -7.742773, -7.741773, -7.740773, -7.739773, -7.738773, -7.737773, -7.736773, -7.735773, -7.734773, -7.733773, -7.732773, -7.731773, -7.730773, -7.729773, -7.728773, -7.727773, -7.726773, -7.725773, -7.724773, -7.723773, -7.722773, -7.721773, -7.720773, -7.719773, -7.718773, -7.717773, -7.716773, -7.715773, -7.714773, -7.713773, -7.712773, -7.711773, -7.710773, -7.709773, -7.708773, -7.707773, -7.706773, -7.705773, -7.704773, -7.703773, -7.702773, -7.701773, -7.700773, -7.699773, -7.698773, -7.697773, -7.696773, -7.695773, -7.694773, -7.693773, -7.692773, -7.691773, -7.690773, -7.689773, -7.688773, -7.687773, -7.686773, -7.685773, -7.684773, -7.683773, -7.682773, -7.681773, -7.680773, -7.679773, -7.678773, -7.677773, -7.676773, -7.675773, -7.674773, -7.673773, -7.672773, -7.671773, -7.670773, -7.669773, -7.668773, -7.667773, -7.666773, -7.665773, -7.664773, -7.663773, -7.662773, -7.661773, -7.660773, -7.659773, -7.658773, -7.657773, -7.656773, -7.655773, -7.654773, -7.653773, -7.652773, -7.651773, -7.650773, -7.649773, -7.648773, -7.647773, -7.646773, -7.645773, -7.644773, -7.643773,
-7.642773, -7.641773, -7.640773, -7.639773, -7.638773, -7.637773, -7.636773, -7.635773, -7.634773, -7.633773, -7.632773, -7.631773, -7.630773, -7.629773, -7.628773, -7.627773, -7.626773, -7.625773, -7.624773, -7.623773, -7.622773, -7.621773, -7.620773, -7.619773, -7.618773, -7.617773, -7.616773, -7.615773, -7.614773, -7.613773, -7.612773, -7.611773, -7.610773, -7.609773, -7.608773, -7.607773, -7.606773, -7.605773, -7.604773, -7.603773, -7.602773, -7.601773, -7.600773, -7.599773, -7.598773, -7.597773, -7.596773, -7.595773, -7.594773, -7.593773, -7.592773, -7.591773, -7.590773, -7.589773, -7.588773, -7.587773, -7.586773, -7.585773, -7.584773, -7.583773, -7.582773, -7.581773, -7.580773, -7.579773, -7.578773, -7.577773, -7.576773, -7.575773, -7.574773, -7.573773, -7.572773, -7.571773, -7.570773, -7.569773, -7.568773, -7.567773, -7.566773, -7.565773, -7.564773, -7.563773, -7.562773, -7.561773, -7.560773, -7.559773, -7.558773, -7.557773, -7.556773, -7.555773, -7.554773, -7.553773, -7.552773, -7.551773, -7.550773, -7.549773, -7.548773, -7.547773, -7.546773, -7.545773, -7.544773, -7.543773, -7.542773, -7.541773, -7.540773,
-7.539773, -7.538773, -7.537773, -7.536773, -7.535773, -7.534773, -7.533773, -7.532773, -7.531773, -7.530773, -7.529773, -7.528773, -7.527773, -7.526773, -7.525773, -7.524773, -7.523773, -7.522773, -7.521773, -7.520773, -7.519773, -7.518773, -7.517773, -7.516773, -7.515773, -7.514773, -7.513773, -7.512773, -7.511773, -7.510773, -7.509773, -7.508773, -7.507773, -7.506773, -7.505773, -7.504773, -7.503773, -7.502773, -7.501773, -7.500773, -7.499773, -7.498773, -7.497773, -7.496773, -7.495773, -7.494773, -7.493773, -7.492773, -7.491773, -7.490773, -7.489773, -7.488773, -7.487773, -7.486773, -7.485773, -7.484773, -7.483773, -7.482773, -7.481773, -7.480773, -7.479773, -7.478773, -7.477773, -7.476773, -7.475773, -7.474773, -7.473773, -7.472773, -7.471773, -7.470773, -7.469773, -7.468773, -7.467773, -7.466773, -7.465773, -7.464773, -7.463773, -7.462773, -7.461773, -7.460773, -7.459773, -7.458773, -7.457773, -7.456773, -7.455773, -7.454773, -7.453773, -7.452773, -7.451773, -7.450773, -7.449773, -7.448773, -7.447773, -7.446773, -7.445773, -7.444773, -7.443773, -7.442773, -7.441773, -7.440773, -7.439773, -7.438773, -7.437773,
-7.436773, -7.435773, -7.434773, -7.433773, -7.432773, -7.431773, -7.430773, -7.429773, -7.428773, -7.427773, -7.426773, -7.425773, -7.424773, -7.423773, -7.422773, -7.421773, -7.420773, -7.419773, -7.418773, -7.417773, -7.416773, -7.415773, -7.414773, -7.413773, -7.412773, -7.411773, -7.410773, -7.409773, -7.408773, -7.407773, -7.406773, -7.405773, -7.404773, -7.403773, -7.402773, -7.401773, -7.400773, -7.399773, -7.398773, -7.397773, -7.396773, -7.395773, -7.394773, -7.393773, -7.392773, -7.391773, -7.390773, -7.389773, -7.388773, -7.387773, -7.386773, -7.385773, -7.384773, -7.383773, -7.382773, -7.381773, -7.380773, -7.379773, -7.378773, -7.377773, -7.376773, -7.375773, -7.374773, -7.373773, -7.372773, -7.371773, -7.370773, -7.369773, -7.368773, -7.367773, -7.366773, -7.365773, -7.364773, -7.363773, -7.362773, -7.361773, -7.360773, -7.359773, -7.358773, -7.357773, -7.356773, -7.355773, -7.354773, -7.353773, -7.352773, -7.351773, -7.350773, -7.349773, -7.348773, -7.347773, -7.346773, -7.345773, -7.344773, -7.343773, -7.342773, -7.341773, -7.340773, -7.339773, -7.338773, -7.337773, -7.336773, -7.335773, -7.334773,
-7.333773, -7.332773, -7.331773, -7.330773, -7.329773, -7.328773, -7.327773, -7.326773, -7.325773, -7.324773, -7.323773, -7.322773, -7.321773, -7.320773, -7.319773, -7.318773, -7.317773, -7.316773, -7.315773, -7.314773, -7.313773, -7.312773, -7.311773, -7.310773, -7.309773, -7.308773, -7.307773, -7.306773, -7.305773, -7.304773, -7.303773, -7.302773, -7.301773, -7.300773, -7.299773, -7.298773, -7.297773, -7.296773, -7.295773, -7.294773, -7.293773, -7.292773, -7.291773, -7.290773, -7.289773, -7.288773, -7.287773, -7.286773, -7.285773, -7.284773, -7.283773, -7.282773, -7.281773, -7.280773, -7.279773, -7.278773, -7.277773, -7.276773, -7.275773, -7.274773, -7.273773, -7.272773, -7.271773, -7.270773, -7.269773, -7.268773, -7.267773, -7.266773, -7.265773, -7.264773, -7.263773, -7.262773, -7.261773, -7.260773, -7.259773, -7.258773, -7.257773, -7.256773, -7.255773, -7.254773, -7.253773, -7.252773, -7.251773, -7.250773, -7.249773, -7.248773, -7.247773, -7.246773, -7.245773, -7.244773, -7.243773, -7.242773, -7.241773, -7.240773, -7.239773, -7.238773, -7.237773, -7.236773, -7.235773, -7.234773, -7.233773, -7.232773, -7.231773,
-7.230773, -7.229773, -7.228773, -7.227773, -7.226773, -7.225773, -7.224773, -7.223773, -7.222773, -7.221773, -7.220773, -7.219773, -7.218773, -7.217773, -7.216773, -7.215773, -7.214773, -7.213773, -7.212773, -7.211773, -7.210773, -7.209773, -7.208773, -7.207773, -7.206773, -7.205773, -7.204773, -7.203773, -7.202773, -7.201773, -7.200773, -7.199773, -7.198773, -7.197773, -7.196773, -7.195773, -7.194773, -7.193773, -7.192773, -7.191773, -7.190773, -7.189773, -7.188773, -7.187773, -7.186773, -7.185773, -7.184773, -7.183773, -7.182773, -7.181773, -7.180773, -7.179773, -7.178773, -7.177773, -7.176773, -7.175773, -7.174773, -7.173773, -7.172773, -7.171773, -7.170773, -7.169773, -7.168773, -7.167773, -7.166773, -7.165773, -7.164773, -7.163773, -7.162773, -7.161773, -7.160773, -7.159773, -7.158773, -7.157773, -7.156773, -7.155773, -7.154773, -7.153773, -7.152773, -7.151773, -7.150773, -7.149773, -7.148773, -7.147773, -7.146773, -7.145773, -7.144773, -7.143773, -7.142773, -7.141773, -7.140773, -7.139773, -7.138773, -7.137773, -7.136773, -7.135773, -7.134773, -7.133773, -7.132773, -7.131773, -7.130773, -7.129773, -7.128773,
-7.127773, -7.126773, -7.125773, -7.124773, -7.123773, -7.122773, -7.121773, -7.120773, -7.119773, -7.118773, -7.117773, -7.116773, -7.115773, -7.114773, -7.113773, -7.112773, -7.111773, -7.110773, -7.109773, -7.108773, -7.107773, -7.106773, -7.105773, -7.104773, -7.103773, -7.102773, -7.101773, -7.100773, -7.099773, -7.098773, -7.097773, -7.096773, -7.095773, -7.094773, -7.093773, -7.092773, -7.091773, -7.090773, -7.089773, -7.088773, -7.087773, -7.086773, -7.085773, -7.084773, -7.083773, -7.082773, -7.081773, -7.080773, -7.079773, -7.078773, -7.077773, -7.076773, -7.075773, -7.074773, -7.073773, -7.072773, -7.071773, -7.070773, -7.069773, -7.068773, -7.067773, -7.066773, -7.065773, -7.064773, -7.063773, -7.062773, -7.061773, -7.060773, -7.059773, -7.058773, -7.057773, -7.056773, -7.055773, -7.054773, -7.053773, -7.052773, -7.051773, -7.050773, -7.049773, -7.048773, -7.047773, -7.046773, -7.045773, -7.044773, -7.043773, -7.042773, -7.041773, -7.040773, -7.039773, -7.038773, -7.037773, -7.036773, -7.035773, -7.034773, -7.033773, -7.032773, -7.031773, -7.030773, -7.029773, -7.028773, -7.027773, -7.026773, -7.025773,
-7.024773, -7.023773, -7.022773, -7.021773, -7.020773, -7.019773, -7.018773, -7.017773, -7.016773, -7.015773, -7.014773, -7.013773, -7.012773, -7.011773, -7.010773, -7.009773, -7.008773, -7.007773, -7.006773, -7.005773, -7.004773, -7.003773, -7.002773, -7.001773, -7.000773, -6.999773, -6.998773, -6.997773, -6.996773, -6.995773, -6.994773, -6.993773, -6.992773, -6.991773, -6.990773, -6.989773, -6.988773, -6.987773, -6.986773, -6.985773, -6.984773, -6.983773, -6.982773, -6.981773, -6.980773, -6.979773, -6.978773, -6.977773, -6.976773, -6.975773, -6.974773, -6.973773, -6.972773, -6.971773, -6.970773, -6.969773, -6.968773, -6.967773, -6.966773, -6.965773, -6.964773, -6.963773, -6.962773, -6.961773, -6.960773, -6.959773, -6.958773, -6.957773, -6.956773, -6.955773, -6.954773, -6.953773, -6.952773, -6.951773, -6.950773, -6.949773, -6.948773, -6.947773, -6.946773, -6.945773, -6.944773, -6.943773, -6.942773, -6.941773, -6.940773, -6.939773, -6.938773, -6.937773, -6.936773, -6.935773, -6.934773, -6.933773, -6.932773, -6.931773, -6.930773, -6.929773, -6.928773, -6.927773, -6.926773, -6.925773, -6.924773, -6.923773, -6.922773,
-6.921773, -6.920773, -6.919773, -6.918773, -6.917773, -6.916773, -6.915773, -6.914773, -6.913773, -6.912773, -6.911773, -6.910773, -6.909773, -6.908773, -6.907773, -6.906773, -6.905773, -6.904773, -6.903773, -6.902773, -6.901773, -6.900773, -6.899773, -6.898773, -6.897773, -6.896773, -6.895773, -6.894773, -6.893773, -6.892773, -6.891773, -6.890773, -6.889773, -6.888773, -6.887773, -6.886773, -6.885773, -6.884773, -6.883773, -6.882773, -6.881773, -6.880773, -6.879773, -6.878773, -6.877773, -6.876773, -6.875773, -6.874773, -6.873773, -6.872773, -6.871773, -6.870773, -6.869773, -6.868773, -6.867773, -6.866773, -6.865773, -6.864773, -6.863773, -6.862773, -6.861773, -6.860773, -6.859773, -6.858773, -6.857773, -6.856773, -6.855773, -6.854773, -6.853773, -6.852773, -6.851773, -6.850773, -6.849773, -6.848773, -6.847773, -6.846773, -6.845773, -6.844773, -6.843773, -6.842773, -6.841773, -6.840773, -6.839773, -6.838773, -6.837773, -6.836773, -6.835773, -6.834773, -6.833773, -6.832773, -6.831773, -6.830773, -6.829773, -6.828773, -6.827773, -6.826773, -6.825773, -6.824773, -6.823773, -6.822773, -6.821773, -6.820773, -6.819773,
-6.818773, -6.817773, -6.816773, -6.815773, -6.814773, -6.813773, -6.812773, -6.811773, -6.810773, -6.809773, -6.808773, -6.807773, -6.806773, -6.805773, -6.804773, -6.803773, -6.802773, -6.801773, -6.800773, -6.799773, -6.798773, -6.797773, -6.796773, -6.795773, -6.794773, -6.793773, -6.792773, -6.791773, -6.790773, -6.789773, -6.788773, -6.787773, -6.786773, -6.785773, -6.784773, -6.783773, -6.782773, -6.781773, -6.780773, -6.779773, -6.778773, -6.777773, -6.776773, -6.775773, -6.774773, -6.773773, -6.772773, -6.771773, -6.770773, -6.769773, -6.768773, -6.767773, -6.766773, -6.765773, -6.764773, -6.763773, -6.762773, -6.761773, -6.760773, -6.759773, -6.758773, -6.757773, -6.756773, -6.755773, -6.754773, -6.753773, -6.752773, -6.751773, -6.750773, -6.749773, -6.748773, -6.747773, -6.746773, -6.745773, -6.744773, -6.743773, -6.742773, -6.741773, -6.740773, -6.739773, -6.738773, -6.737773, -6.736773, -6.735773, -6.734773, -6.733773, -6.732773, -6.731773, -6.730773, -6.729773, -6.728773, -6.727773, -6.726773, -6.725773, -6.724773, -6.723773, -6.722773, -6.721773, -6.720773, -6.719773, -6.718773, -6.717773, -6.716773, -6.715773,
-6.714773, -6.713773, -6.712773, -6.711773, -6.710773, -6.709773, -6.708773, -6.707773, -6.706773, -6.705773, -6.704773, -6.703773, -6.702773, -6.701773, -6.700773, -6.699773, -6.698773, -6.697773, -6.696773, -6.695773, -6.694773, -6.693773, -6.692773, -6.691773, -6.690773, -6.689773, -6.688773, -6.687773, -6.686773, -6.685773, -6.684773, -6.683773, -6.682773, -6.681773, -6.680773, -6.679773, -6.678773, -6.677773, -6.676773, -6.675773, -6.674773, -6.673773, -6.672773, -6.671773, -6.670773, -6.669773, -6.668773, -6.667773, -6.666773, -6.665773, -6.664773, -6.663773, -6.662773, -6.661773, -6.660773, -6.659773, -6.658773, -6.657773, -6.656773, -6.655773}, nd4j::DataType::DOUBLE);
    input.linspace(0.01, 0.001);

    ops::helpers::logSoftmax(nd4j::LaunchContext ::defaultContext(), input, output, 0);

    ASSERT_TRUE(output.equalsTo(&expOutput));
}


//////////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, mmulMxV_1) {

    const Nd4jLong M = 3;
    const Nd4jLong N = 4;

    NDArray a('f', {M,N}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, nd4j::DataType::DOUBLE);
    NDArray temp('f', {M,N,5}, {16,2,-6,7,2,-2,4,-7,6,4,4,6,-3,1,3,9,1,4,9,10,-10,-3,-8,7,-7,-7,6,9,7,-6,8,7,-3,-3,4,-2,5,-3,-3,4,6,-5,-1,7,-5,4,-10,-1,8,0,-7,4,-10,-7,-8,-9,2,9,7,9}, nd4j::DataType::DOUBLE);
    NDArray x = temp(6, {0,2});
    NDArray y('f', {M}, nd4j::DataType::DOUBLE);

    NDArray exp('f', {M}, {5.5, 5.1, 4.7}, nd4j::DataType::DOUBLE);

    nd4j::MmulHelper::mmul(&a, &x, &y, 1., 0.);
    ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, mmulMxV_2) {

    const Nd4jLong M = 3;
    const Nd4jLong N = 4;

    NDArray a('f', {N,M}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, nd4j::DataType::DOUBLE);
    a.permutei({1,0});
    NDArray temp('f', {M,N,5}, {16,2,-6,7,2,-2,4,-7,6,4,4,6,-3,1,3,9,1,4,9,10,-10,-3,-8,7,-7,-7,6,9,7,-6,8,7,-3,-3,4,-2,5,-3,-3,4,6,-5,-1,7,-5,4,-10,-1,8,0,-7,4,-10,-7,-8,-9,2,9,7,9}, nd4j::DataType::DOUBLE);
    NDArray x = temp(6, {0,2});
    NDArray y('f', {M}, nd4j::DataType::DOUBLE);

    NDArray exp('f', {M}, {5.1, 3.3, 1.5}, nd4j::DataType::DOUBLE);

    nd4j::MmulHelper::mmul(&a, &x, &y, 1., 0.);
    ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, mmulMxV_3) {

    const Nd4jLong M = 3;
    const Nd4jLong N = 4;

    NDArray a('f', {N,M}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, nd4j::DataType::DOUBLE);
    a.permutei({1,0});
    NDArray temp('f', {N,M,5}, {16,2,-6,7,2,-2,4,-7,6,4,4,6,-3,1,3,9,1,4,9,10,-10,-3,-8,7,-7,-7,6,9,7,-6,8,7,-3,-3,4,-2,5,-3,-3,4,6,-5,-1,7,-5,4,-10,-1,8,0,-7,4,-10,-7,-8,-9,2,9,7,9}, nd4j::DataType::DOUBLE);
    NDArray x = temp(4, {1,2});
    NDArray y('f', {M}, nd4j::DataType::DOUBLE);

    NDArray exp('f', {M}, {6.2, 4.5, 1.7}, nd4j::DataType::DOUBLE);

    nd4j::MmulHelper::mmul(&a, &x, &y, 1., 0.);
    ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, mmulMxV_4) {

    const Nd4jLong M = 3;
    const Nd4jLong N = 4;

    NDArray a('f', {N,M}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, nd4j::DataType::DOUBLE);
    a.permutei({1,0});
    NDArray temp('f', {5,M,N}, {16,2,-6,7,2,-2,4,-7,6,4,4,6,-3,1,3,9,1,4,9,10,-10,-3,-8,7,-7,-7,6,9,7,-6,8,7,-3,-3,4,-2,5,-3,-3,4,6,-5,-1,7,-5,4,-10,-1,8,0,-7,4,-10,-7,-8,-9,2,9,7,9}, nd4j::DataType::DOUBLE);
    NDArray x = temp(3, {0,1});
    NDArray y('f', {M}, nd4j::DataType::DOUBLE);

    NDArray exp('f', {M}, {1.5, 1.8, 1.5}, nd4j::DataType::DOUBLE);

    nd4j::MmulHelper::mmul(&a, &x, &y, 1., 0.);
    ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, mmulMxV_5) {

    const Nd4jLong M = 3;
    const Nd4jLong N = 4;

    NDArray a('c', {N,M}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, nd4j::DataType::DOUBLE);
    a.permutei({1,0});
    NDArray temp('f', {5,M,N}, {16,2,-6,7,2,-2,4,-7,6,4,4,6,-3,1,3,9,1,4,9,10,-10,-3,-8,7,-7,-7,6,9,7,-6,8,7,-3,-3,4,-2,5,-3,-3,4,6,-5,-1,7,-5,4,-10,-1,8,0,-7,4,-10,-7,-8,-9,2,9,7,9}, nd4j::DataType::DOUBLE);
    NDArray x = temp(2, {0,1});
    NDArray y('f', {M}, nd4j::DataType::DOUBLE);

    NDArray exp('f', {M}, {-0.3, 0.3, 0.9}, nd4j::DataType::DOUBLE);

    nd4j::MmulHelper::mmul(&a, &x, &y, 1., 0.);
    ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, mmulMxV_6) {

    const Nd4jLong M = 3;
    const Nd4jLong N = 4;

    NDArray a('c', {N,M}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, nd4j::DataType::DOUBLE);
    a.permutei({1,0});
    NDArray temp('c', {5,N,M}, {16,2,-6,7,2,-2,4,-7,6,4,4,6,-3,1,3,9,1,4,9,10,-10,-3,-8,7,-7,-7,6,9,7,-6,8,7,-3,-3,4,-2,5,-3,-3,4,6,-5,-1,7,-5,4,-10,-1,8,0,-7,4,-10,-7,-8,-9,2,9,7,9}, nd4j::DataType::DOUBLE);
    NDArray x = temp(13, {0,2});
    NDArray y('f', {M}, nd4j::DataType::DOUBLE);

    NDArray exp('f', {M}, {-12.1, -10.9, -9.7}, nd4j::DataType::DOUBLE);

    nd4j::MmulHelper::mmul(&a, &x, &y, 1., 0.);
    ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, mmulMxV_7) {

    const Nd4jLong M = 3;
    const Nd4jLong N = 4;

    NDArray a('c', {N,M}, {1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0}, nd4j::DataType::DOUBLE);
    a.permutei({1,0});
    NDArray temp('c', {5,N,M}, {16,2,-6,7,2,-2,4,-7,6,4,4,6,-3,1,3,9,1,4,9,10,-10,-3,-8,7,-7,-7,6,9,7,-6,8,7,-3,-3,4,-2,5,-3,-3,4,6,-5,-1,7,-5,4,-10,-1,8,0,-7,4,-10,-7,-8,-9,2,9,7,9}, nd4j::DataType::DOUBLE);
    NDArray x = temp(10, {0,2});
    NDArray y('c', {M}, nd4j::DataType::DOUBLE);

    NDArray exp('c', {M}, {3.3, 3.3, 3.3}, nd4j::DataType::DOUBLE);

    nd4j::MmulHelper::mmul(&a, &x, &y, 1., 0.);
    ASSERT_TRUE(y.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, softmaxDerivative_1) {

    NDArray input('c', {3,3}, {-1, 1, -2, 2, -3, 3, -4, 4, 5.}, nd4j::DataType::DOUBLE);
    NDArray expOutput('c', {3,3}, {0.04508, 0.04514, 0.0008 , 0.0472 , 0.00087, 0.10492, 0.00235, 0.04592, 0.10553}, nd4j::DataType::DOUBLE);
    NDArray output('c', {3,3}, nd4j::DataType::DOUBLE);

    // input.applyTransform(nd4j::transform::SoftMaxDerivative, &output);

    nd4j::ops::helpers::softmaxDerivative(input.getContext(), input, output, 0);
    ASSERT_TRUE(expOutput.isSameShape(output));
    ASSERT_TRUE(expOutput.equalsTo(output));
}

//////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, softmaxDerivative_2) {

    NDArray input('c', {3,3,3}, {-1, 1, -2, 2, -3, 3, -4, 4, -5,5 ,-6,6, -7,7, -8,8, -9,9, -10,10, -11,11, -12,12, -13,13, 14.}, nd4j::DataType::DOUBLE);
    NDArray expOutput('c', {3,3,3}, {4.50755e-02, 4.51394e-02, 6.64586e-03,4.72027e-02, 8.67128e-04, 6.97440e-03,2.35008e-03, 4.59243e-02, 3.32995e-04,
                                    4.51766e-02, 2.26032e-06, 4.51767e-02,2.91394e-07, 2.37285e-06, 3.94360e-08,4.51769e-02, 1.12535e-07, 4.51767e-02,
                                    7.58256e-10, 4.51767e-02, 1.22325e-11,7.96007e-10, 1.32293e-11, 1.04994e-01,3.77513e-11, 4.51767e-02, 1.04994e-01}, nd4j::DataType::DOUBLE);
    NDArray output('c', {3,3,3}, nd4j::DataType::DOUBLE);

    // input.applyTransform(nd4j::transform::SoftMaxDerivative, &output);

    nd4j::ops::helpers::softmaxDerivative(input.getContext(), input, output, 1);
    ASSERT_TRUE(expOutput.isSameShape(output));
    ASSERT_TRUE(expOutput.equalsTo(output));
}

//////////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, softmaxDerivative_3) {

    NDArray input('c', {5}, {-1., 1, -2, 2, 3}, nd4j::DataType::DOUBLE);
    NDArray expOutput('c', {5}, {0.01184, 0.08071, 0.00439, 0.18277, 0.22618}, nd4j::DataType::DOUBLE);
    NDArray output('c', {5}, nd4j::DataType::DOUBLE);

    // input.applyTransform(nd4j::transform::SoftMaxDerivative, &output);

    nd4j::ops::helpers::softmaxDerivative(input.getContext(), input, output, 0);
    ASSERT_TRUE(expOutput.isSameShape(output));
    ASSERT_TRUE(expOutput.equalsTo(output));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, lstmLayerCell_1) {

    const int bS   = 2;
    const int nIn  = 10;
    const int nOut = 4;

    const float dataFormat = 0;     // is ignored in cell op
    const float cellClip = 5;       // clipping value
    const float gateAct = 2;        // sigmoid activation for input (i), forget (f) and output (o) gates
    const float gateAlpha = 0;      // alpha value for activation for gates, not required for sigmoid
    const float gateBeta = 0;       // beta value for activation for gates, not required for sigmoid
    const float cellAct = 0;        // tanh activation for cell state
    const float cellAlpha = 0;      // alpha value for cell state activation, not required for tanh
    const float cellBeta = 0;       // beta value for cell state activation, not required for tanh
    const float outAct = 0;         // tanh activation for output
    const float outAlpha = 0;       // alpha value for output activation, not required for tanh
    const float outBeta = 0;        // beta value for output activation, not required for tanh

    NDArray x ('c', {bS, nIn}, nd4j::DataType::FLOAT32);
    NDArray Wx('c', {nIn, 4*nOut}, nd4j::DataType::FLOAT32);
    NDArray Wr('c', {nOut, 4*nOut}, nd4j::DataType::FLOAT32);
    NDArray b ('c', {4*nOut}, nd4j::DataType::FLOAT32);
    NDArray hI('c', {bS, nOut}, nd4j::DataType::FLOAT32);
    NDArray cI('c', {bS, nOut}, nd4j::DataType::FLOAT32);
    NDArray Wp('c', {3*nOut}, nd4j::DataType::FLOAT32);

    NDArray h('c', {bS, nOut}, nd4j::DataType::FLOAT32);
    NDArray c('c', {bS, nOut}, nd4j::DataType::FLOAT32);

    NDArray expH('c', {bS, nOut}, {0.999288, 0.999288, 0.999288, 0.999288, 0.999288, 0.999288, 0.999288, 0.999288}, nd4j::DataType::FLOAT32);
    NDArray expC('c', {bS, nOut}, {3.999778, 3.999778, 3.999778, 3.999778, 3.999778, 3.999778, 3.999778, 3.999778}, nd4j::DataType::FLOAT32);

    std::vector<float> params = {dataFormat, 0, cellClip, gateAct, gateAlpha, gateBeta, cellAct, cellAlpha, cellBeta, outAct, outAlpha, outBeta};

    x = 1.;
    hI = 2.;
    cI = 3.;
    Wx = 0.5;
    Wr = 0.4;
    Wp = 0.3;
    b = 0.7;

    nd4j::ops::helpers::lstmLayerCell(&x, &Wx, &Wr, &b, &hI, &cI, &Wp, params, &h, &c);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));
    ASSERT_TRUE(expC.isSameShape(c));
    ASSERT_TRUE(expC.equalsTo(c));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, lstmLayerCell_2) {

    const int bS   = 2;
    const int nIn  = 10;
    const int nOut = 4;

    const float dataFormat = 0;     // is ignored in cell op
    const float cellClip = 3;       // clipping value
    const float gateAct = 2;        // sigmoid activation for input (i), forget (f) and output (o) gates
    const float gateAlpha = 0;      // alpha value for activation for gates, not required for sigmoid
    const float gateBeta = 0;       // beta value for activation for gates, not required for sigmoid
    const float cellAct = 0;        // tanh activation for cell state
    const float cellAlpha = 0;      // alpha value for cell state activation, not required for tanh
    const float cellBeta = 0;       // beta value for cell state activation, not required for tanh
    const float outAct = 0;         // tanh activation for output
    const float outAlpha = 0;       // alpha value for output activation, not required for tanh
    const float outBeta = 0;        // beta value for output activation, not required for tanh

    NDArray x ('c', {bS, nIn}, nd4j::DataType::FLOAT32);
    NDArray Wx('c', {nIn, 4*nOut}, nd4j::DataType::FLOAT32);
    NDArray Wr('c', {nOut, 4*nOut}, nd4j::DataType::FLOAT32);
    NDArray b ('c', {4*nOut}, nd4j::DataType::FLOAT32);
    NDArray hI('c', {bS, nOut}, nd4j::DataType::FLOAT32);
    NDArray cI('c', {bS, nOut}, nd4j::DataType::FLOAT32);
    NDArray Wp('c', {3*nOut}, nd4j::DataType::FLOAT32);

    NDArray h('c', {bS, nOut}, nd4j::DataType::FLOAT32);
    NDArray c('c', {bS, nOut}, nd4j::DataType::FLOAT32);

    NDArray expH('c', {bS, nOut}, {0.995, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995}, nd4j::DataType::FLOAT32);
    NDArray expC('c', {bS, nOut}, {3., 3., 3., 3., 3., 3., 3., 3.}, nd4j::DataType::FLOAT32);

    std::vector<float> params = {dataFormat, 0, cellClip, gateAct, gateAlpha, gateBeta, cellAct, cellAlpha, cellBeta, outAct, outAlpha, outBeta};

    x = 1.;
    hI = 2.;
    cI = 3.;
    Wx = 0.5;
    Wr = 0.4;
    Wp = 0.3;
    b = 0.7;

    nd4j::ops::helpers::lstmLayerCell(&x, &Wx, &Wr, &b, &hI, &cI, &Wp, params, &h, &c);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));
    ASSERT_TRUE(expC.isSameShape(c));
    ASSERT_TRUE(expC.equalsTo(c));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests1, lstmLayerCell_3) {

    const int nIn  = 10;
    const int nOut = 4;

    const float dataFormat = 0;     // is ignored in cell op
    const float cellClip = 5;       // clipping value
    const float gateAct = 2;        // sigmoid activation for input (i), forget (f) and output (o) gates
    const float gateAlpha = 0;      // alpha value for activation for gates, not required for sigmoid
    const float gateBeta = 0;       // beta value for activation for gates, not required for sigmoid
    const float cellAct = 0;        // tanh activation for cell state
    const float cellAlpha = 0;      // alpha value for cell state activation, not required for tanh
    const float cellBeta = 0;       // beta value for cell state activation, not required for tanh
    const float outAct = 0;         // tanh activation for output
    const float outAlpha = 0;       // alpha value for output activation, not required for tanh
    const float outBeta = 0;        // beta value for output activation, not required for tanh

    NDArray x ('c', {nIn}, nd4j::DataType::FLOAT32);
    NDArray Wx('c', {nIn, 4*nOut}, nd4j::DataType::FLOAT32);
    NDArray Wr('c', {nOut, 4*nOut}, nd4j::DataType::FLOAT32);
    NDArray b ('c', {4*nOut}, nd4j::DataType::FLOAT32);
    NDArray hI('c', {nOut}, nd4j::DataType::FLOAT32);
    NDArray cI('c', {nOut}, nd4j::DataType::FLOAT32);
    NDArray Wp('c', {3*nOut}, nd4j::DataType::FLOAT32);

    NDArray h('c', {nOut}, nd4j::DataType::FLOAT32);
    NDArray c('c', {nOut}, nd4j::DataType::FLOAT32);

    NDArray expH('c', {nOut}, {0.999288, 0.999288, 0.999288, 0.999288}, nd4j::DataType::FLOAT32);
    NDArray expC('c', {nOut}, {3.999778, 3.999778, 3.999778, 3.999778}, nd4j::DataType::FLOAT32);

    std::vector<float> params = {dataFormat, 0, cellClip, gateAct, gateAlpha, gateBeta, cellAct, cellAlpha, cellBeta, outAct, outAlpha, outBeta};

    x = 1.;
    hI = 2.;
    cI = 3.;
    Wx = 0.5;
    Wr = 0.4;
    Wp = 0.3;
    b = 0.7;

    nd4j::ops::helpers::lstmLayerCell(&x, &Wx, &Wr, &b, &hI, &cI, &Wp, params, &h, &c);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));
    ASSERT_TRUE(expC.isSameShape(c));
    ASSERT_TRUE(expC.equalsTo(c));
}

