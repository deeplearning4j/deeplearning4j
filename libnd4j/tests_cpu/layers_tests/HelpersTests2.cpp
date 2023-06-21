/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
#include <helpers/EigenValsAndVecs.h>
#include <helpers/FullPivLU.h>
#include <helpers/HessenbergAndSchur.h>
#include <helpers/Sqrtm.h>
#include <ops/declarable/helpers/triangular_solve.h>

#include "testlayers.h"

using namespace sd;

class HelpersTests2 : public testing::Test {
 public:
  HelpersTests2() { std::cout << std::endl << std::flush; }
};

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, Hessenberg_1) {
  NDArray x1('c', {1, 4}, {14, 17, 3, 1}, sd::DataType::DOUBLE);
  NDArray x2('c', {1, 1}, {14}, sd::DataType::DOUBLE);
  NDArray expQ('c', {1, 1}, {1}, sd::DataType::DOUBLE);

  ops::helpers::Hessenberg<double> hess1(x1);
  ASSERT_TRUE(hess1._H.isSameShape(&x1));
  ASSERT_TRUE(hess1._H.equalsTo(&x1));
  ASSERT_TRUE(hess1._Q.isSameShape(&expQ));
  ASSERT_TRUE(hess1._Q.equalsTo(&expQ));

  ops::helpers::Hessenberg<double> hess2(x2);
  ASSERT_TRUE(hess2._H.isSameShape(&x2));
  ASSERT_TRUE(hess2._H.equalsTo(&x2));
  ASSERT_TRUE(hess2._Q.isSameShape(&expQ));
  ASSERT_TRUE(hess2._Q.equalsTo(&expQ));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, Hessenberg_2) {
  NDArray x('c', {2, 2}, {1.5, -2, 17, 5}, sd::DataType::DOUBLE);
  NDArray expQ('c', {2, 2}, {1, 0, 0, 1}, sd::DataType::DOUBLE);
  ops::helpers::Hessenberg<double> hess(x);



  x.printIndexedBuffer("expected x");
  hess._H.printIndexedBuffer("output h");


  expQ.printIndexedBuffer("expected q");
  hess._Q.printIndexedBuffer("output q");


  ASSERT_TRUE(hess._H.isSameShape(&x));
  ASSERT_TRUE(hess._H.equalsTo(&x));

  ASSERT_TRUE(hess._Q.isSameShape(&expQ));
  ASSERT_TRUE(hess._Q.equalsTo(&expQ));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, Hessenberg_3) {
  NDArray x('c', {3, 3}, {33, 24, -48, 57, 12.5, -3, 1.1, 10, -5.2}, sd::DataType::DOUBLE);
  NDArray expH('c', {3, 3}, {33, -23.06939, -48.45414, -57.01061, 12.62845, 3.344058, 0, -9.655942, -5.328448},
               sd::DataType::DOUBLE);
  NDArray expQ('c', {3, 3}, {1, 0, 0, 0, -0.99981, -0.019295, 0, -0.019295, 0.99981}, sd::DataType::DOUBLE);

  ops::helpers::Hessenberg<double> hess(x);

  ASSERT_TRUE(hess._H.isSameShape(&expH));
  ASSERT_TRUE(hess._H.equalsTo(&expH));

  ASSERT_TRUE(hess._Q.isSameShape(&expQ));
  ASSERT_TRUE(hess._Q.equalsTo(&expQ));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, Hessenberg_4) {
  NDArray x('c', {4, 4},
            {0.33, -7.25, 1.71, 6.20, 1.34, 5.38, -2.76, -8.51, 7.59, 3.44, 2.24, -6.82, -1.15, 4.80, -4.67, 2.14},
            sd::DataType::DOUBLE);
  NDArray expH('c', {4, 4},
               {0.33, 0.4961181, 3.51599, 9.017665, -7.792702, 4.190221, 6.500328, 5.438888, 0, 3.646734, 0.4641911,
                -7.635502, 0, 0, 5.873535, 5.105588},
               sd::DataType::DOUBLE);
  NDArray expQ(
      'c', {4, 4},
      {1, 0, 0, 0, 0, -0.171956, 0.336675, -0.925787, 0, -0.973988, 0.0826795, 0.210976, 0, 0.147574, 0.937984, 0.3137},
      sd::DataType::DOUBLE);

  ops::helpers::Hessenberg<double> hess(x);

  ASSERT_TRUE(hess._H.isSameShape(&expH));
  ASSERT_TRUE(hess._H.equalsTo(&expH));

  ASSERT_TRUE(hess._Q.isSameShape(&expQ));
  ASSERT_TRUE(hess._Q.equalsTo(&expQ));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, Hessenberg_5) {
  NDArray x('c', {10, 10},
            {6.9,  4.8,  9.5,  3.1,  6.5,  5.8,  -0.9, -7.3, -8.1, 3.0,  0.1,  9.9,  -3.2, 6.4,  6.2,  -7.0, 5.5,
             -2.2, -4.0, 3.7,  -3.6, 9.0,  -1.4, -2.4, 1.7,  -6.1, -4.2, -2.5, -5.6, -0.4, 0.4,  9.1,  -2.1, -5.4,
             7.3,  3.6,  -1.7, -5.7, -8.0, 8.8,  -3.0, -0.5, 1.1,  10.0, 8.0,  0.8,  1.0,  7.5,  3.5,  -1.8, 0.3,
             -0.6, -6.3, -4.5, -1.1, 1.8,  0.6,  9.6,  9.2,  9.7,  -2.6, 4.3,  -3.4, 0.0,  -6.7, 5.0,  10.5, 1.5,
             -7.8, -4.1, -5.3, -5.0, 2.0,  -4.4, -8.4, 6.0,  -9.4, -4.8, 8.2,  7.8,  5.2,  -9.5, -3.9, 0.2,  6.8,
             5.7,  -8.5, -1.9, -0.3, 7.4,  -8.7, 7.2,  1.3,  6.3,  -3.7, 3.9,  3.3,  -6.0, -9.1, 5.9},
            sd::DataType::DOUBLE);
  NDArray expH(
      'c', {10, 10},
      {
          6.9,      6.125208,  -8.070945, 7.219828, -9.363308,  2.181236,  5.995414,  3.892612,  4.982657,   -2.088574,
          -12.6412, 1.212547,  -6.449684, 5.162879, 0.4341714,  -5.278079, -2.624011, -2.03615,  11.39619,   -3.034842,
          0,        -12.71931, 10.1146,   6.494434, -1.062934,  5.668906,  -4.672953, -9.319893, -2.023392,  6.090341,
          0,        0,         7.800521,  -1.46286, 1.484626,   -10.58252, -3.492978, 2.42187,   5.470045,   1.877265,
          0,        0,         0,         14.78259, -0.3147726, -5.74874,  -0.377823, 3.310056,  2.242614,   -5.111574,
          0,        0,         0,         0,        -9.709131,  3.885072,  6.762626,  4.509144,  2.390195,   -4.991013,
          0,        0,         0,         0,        0,          8.126269,  -12.32529, 9.030151,  1.390931,   0.8634045,
          0,        0,         0,         0,        0,          0,         -12.99477, 9.574299,  -0.3098022, 4.910835,
          0,        0,         0,         0,        0,          0,         0,         14.75256,  18.95723,   -5.054717,
          0,        0,         0,         0,        0,          0,         0,         0,         -4.577715,  -5.440827,
      },
      sd::DataType::DOUBLE);
  NDArray expQ('c', {10, 10},
               {1, 0,          0,         0,        0,        0,         0,          0,         0,         0,
                0, -0.0079106, -0.38175,  -0.39287, -0.26002, -0.44102,  -0.071516,  0.12118,   0.64392,   0.057562,
                0, 0.28478,    0.0058784, 0.3837,   -0.47888, 0.39477,   0.0036847,  -0.24678,  0.3229,    0.47042,
                0, -0.031643,  -0.61277,  0.087648, 0.12014,  0.47648,   -0.5288,    0.060599,  0.021434,  -0.30102,
                0, 0.23732,    -0.17801,  -0.31809, -0.31267, 0.27595,   0.30134,    0.64555,   -0.33392,  0.13363,
                0, -0.023732,  -0.40236,  0.43089,  -0.38692, -0.5178,   -0.03957,   -0.081667, -0.47515,  -0.0077949,
                0, 0.20568,    -0.0169,   0.36962,  0.49669,  -0.22475,  -0.22199,   0.50075,   0.10454,   0.46112,
                0, 0.41926,    0.30243,   -0.3714,  -0.16795, -0.12969,  -0.67572,   -0.1205,   -0.26047,  0.10407,
                0, -0.41135,   -0.28357,  -0.33858, 0.18836,  0.083822,  -0.0068213, -0.30161,  -0.24956,  0.66327,
                0, 0.68823,    -0.33616,  -0.12129, 0.36163,  -0.063256, 0.34198,    -0.37564,  -0.048196, -0.058948},
               sd::DataType::DOUBLE);

  ops::helpers::Hessenberg<double> hess(x);

  ASSERT_TRUE(hess._H.isSameShape(&expH));
  ASSERT_TRUE(hess._H.equalsTo(&expH));

  ASSERT_TRUE(hess._Q.isSameShape(&expQ));
  ASSERT_TRUE(hess._Q.equalsTo(&expQ));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, Schur_1) {
  NDArray x('c', {3, 3}, sd::DataType::DOUBLE);

  NDArray expT('c', {3, 3}, {-2.5, -2, 1, 0, 1.5, -2, 3, 4, 5}, sd::DataType::DOUBLE);
  NDArray expU('c', {3, 3}, {0.3, 0.2, -0.1, 0, -0.1, 0.2, -0.3, -0.4, 0.5}, sd::DataType::DOUBLE);

  ops::helpers::Schur<double> schur(x);
  schur.t.linspace(-3, 1);
  schur.u.linspace(-0.3, 0.1);

  schur.splitTwoRows(1, 0.5);

  ASSERT_TRUE(schur.t.isSameShape(&expT));
  ASSERT_TRUE(schur.t.equalsTo(&expT));

  ASSERT_TRUE(schur.u.isSameShape(&expU));
  ASSERT_TRUE(schur.u.equalsTo(&expU));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, Schur_2) {
  NDArray x('c', {3, 3}, sd::DataType::DOUBLE);

  NDArray shift('c', {3}, sd::DataType::DOUBLE);
  NDArray exp1('c', {3}, {1, -3, 0}, sd::DataType::DOUBLE);
  NDArray exp2('c', {3}, {3, 3, -7}, sd::DataType::DOUBLE);
  NDArray exp3('c', {3}, {0.964, 0.964, 0.964}, sd::DataType::DOUBLE);
  NDArray exp1T('c', {3, 3}, {-3, -2, -1, 0, 1, 2, 3, 4, 5}, sd::DataType::DOUBLE);
  NDArray exp2T('c', {3, 3}, {-8, -2, -1, 0, -4, 2, 3, 4, 0}, sd::DataType::DOUBLE);
  NDArray exp3T('c', {3, 3},
                {
                    -9.464102,
                    -2,
                    -1,
                    0,
                    -5.464102,
                    2,
                    3,
                    4,
                    -1.464102,
                },
                sd::DataType::DOUBLE);

  ops::helpers::Schur<double> schur(x);

  schur.t.linspace(-3, 1);
  double expShift = 0;
  schur.calcShift(1, 5, expShift, shift);
  ASSERT_TRUE(schur.t.equalsTo(&exp1T));
  ASSERT_TRUE(shift.isSameShape(&exp1));
  ASSERT_TRUE(shift.equalsTo(&exp1));
  ASSERT_TRUE(expShift == 0);

  schur.t.linspace(-3, 1);
  expShift = 0;
  schur.calcShift(2, 10, expShift, shift);
  ASSERT_TRUE(schur.t.equalsTo(&exp2T));
  ASSERT_TRUE(shift.isSameShape(&exp2));
  ASSERT_TRUE(shift.equalsTo(&exp2));
  ASSERT_TRUE(expShift == 5);

  schur.t.linspace(-3, 1);
  expShift = 0;
  schur.calcShift(2, 30, expShift, shift);
  ASSERT_TRUE(schur.t.equalsTo(&exp3T));
  ASSERT_TRUE(shift.isSameShape(&exp3));
  ASSERT_TRUE(shift.equalsTo(&exp3));
  ASSERT_TRUE((6.4641 - 0.00001) < expShift && expShift < (6.4641 + 0.00001));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, Schur_3) {
  NDArray x('c', {2, 2}, {1.5, -2, 17, 5}, sd::DataType::DOUBLE);
  NDArray expU('c', {2, 2}, {1, 0, 0, 1}, sd::DataType::DOUBLE);

  ops::helpers::Schur<double> schur(x);

  ASSERT_TRUE(schur.t.isSameShape(&x));
  ASSERT_TRUE(schur.t.equalsTo(&x));

  ASSERT_TRUE(schur.u.isSameShape(&expU));
  ASSERT_TRUE(schur.u.equalsTo(&expU));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, Schur_4) {
  NDArray x('c', {3, 3}, {33, 24, -48, 57, 12.5, -3, 1.1, 10, -5.2}, sd::DataType::DOUBLE);
  NDArray expT('c', {3, 3}, {53.73337, -20.21406, -50.44809, 0, -27.51557, 26.74307, 0, 0, 14.0822},
               sd::DataType::DOUBLE);
  NDArray expU(
      'c', {3, 3},
      {-0.5848506, 0.7185352, 0.3763734, -0.7978391, -0.5932709, -0.1071558, -0.1462962, 0.3629555, -0.9202504},
      sd::DataType::DOUBLE);

  ops::helpers::Schur<double> schur(x);

  ASSERT_TRUE(schur.t.isSameShape(&expT));
  ASSERT_TRUE(schur.t.equalsTo(&expT));

  ASSERT_TRUE(schur.u.isSameShape(&expU));
  ASSERT_TRUE(schur.u.equalsTo(&expU));
}


///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, EigenValsAndVecs_1) {
  NDArray x('c', {2, 2}, {1.5, -2, 17, 5}, sd::DataType::DOUBLE);
  NDArray expVals('c', {2, 2}, {3.25, 5.562149, 3.25, -5.562149}, sd::DataType::DOUBLE);
  NDArray expVecs('c', {2, 2, 2}, {-0.3094862, -0.0973726, -0.3094862, 0.0973726, 0, 0.9459053, 0, -0.9459053},
                  sd::DataType::DOUBLE);

  ops::helpers::EigenValsAndVecs<double> eig(x);

  ASSERT_TRUE(eig._Vals.isSameShape(&expVals));
  ASSERT_TRUE(eig._Vals.equalsTo(&expVals));

  ASSERT_TRUE(eig._Vecs.isSameShape(&expVecs));
  ASSERT_TRUE(eig._Vecs.equalsTo(&expVecs));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, EigenValsAndVecs_2) {
  NDArray x('c', {3, 3}, {33, 24, -48, 57, 12.5, -3, 1.1, 10, -5.2}, sd::DataType::DOUBLE);
  NDArray expVals('c', {3, 2}, {53.73337, 0, -27.51557, 0, 14.0822, 0}, sd::DataType::DOUBLE);
  NDArray expVecs('c', {3, 3, 2},
                  {-0.5848506, 0, 0.5560778, 0, -0.04889745, 0, -0.7978391, 0, -0.7683444, 0, -0.8855156, 0, -0.1462962,
                   0, 0.3168979, 0, -0.4620293, 0},
                  sd::DataType::DOUBLE);

  ops::helpers::EigenValsAndVecs<double> eig(x);

  ASSERT_TRUE(eig._Vals.isSameShape(&expVals));
  ASSERT_TRUE(eig._Vals.equalsTo(&expVals));

  ASSERT_TRUE(eig._Vecs.isSameShape(&expVecs));
  ASSERT_TRUE(eig._Vecs.equalsTo(&expVecs));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, EigenValsAndVecs_3) {
  NDArray x('c', {4, 4},
            {0.33, -7.25, 1.71, 6.20, 1.34, 5.38, -2.76, -8.51, 7.59, 3.44, 2.24, -6.82, -1.15, 4.80, -4.67, 2.14},
            sd::DataType::DOUBLE);
  NDArray expVals('c', {4, 2}, {6.114896, 4.659591, 6.114896, -4.659591, -1.069896, 4.45631, -1.069896, -4.45631},
                  sd::DataType::DOUBLE);
  NDArray expVecs('c', {4, 4, 2},
                  {-0.2141303, 0.4815241,  -0.2141303, -0.4815241,  0.1035092,  -0.4270603, 0.1035092,  0.4270603,
                   0.2703519,  -0.2892722, 0.2703519,  0.2892722,   -0.5256817, 0.044061,   -0.5256817, -0.044061,
                   0.6202137,  0.05521234, 0.6202137,  -0.05521234, -0.5756007, 0.3932209,  -0.5756007, -0.3932209,
                   -0.4166034, -0.0651337, -0.4166034, 0.0651337,   -0.1723716, 0.1138941,  -0.1723716, -0.1138941},
                  sd::DataType::DOUBLE);

  ops::helpers::EigenValsAndVecs<double> eig(x);

  ASSERT_TRUE(eig._Vals.isSameShape(&expVals));
  ASSERT_TRUE(eig._Vals.equalsTo(&expVals));

  ASSERT_TRUE(eig._Vecs.isSameShape(&expVecs));
  ASSERT_TRUE(eig._Vecs.equalsTo(&expVecs));
}



///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, fullPivLU_1) {
  NDArray a('c', {4, 4},
            {0.33, -7.25, 1.71, 6.20, 1.34, 5.38, -2.76, -8.51, 7.59, 3.44, 2.24, -6.82, -1.15, 4.80, -4.67, 2.14},
            sd::DataType::DOUBLE);
  NDArray b('c', {4, 1}, {-5., 10, 9, 1}, sd::DataType::DOUBLE);

  NDArray x = b.ulike();

  NDArray expX('c', {4, 1}, {0.8527251, -0.2545784, -1.076495, -0.8526268}, sd::DataType::DOUBLE);

  ops::helpers::FullPivLU<double>::solve(a, b, x);

  ASSERT_TRUE(x.equalsTo(&expX));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, fullPivLU_2) {
  NDArray a('c', {4, 4},
            {0.33, -7.25, 1.71, 6.20, 1.34, 5.38, -2.76, -8.51, 7.59, 3.44, 2.24, -6.82, -1.15, 4.80, -4.67, 2.14},
            sd::DataType::DOUBLE);
  NDArray b('c', {4, 2}, {-5., 10, 9, 1, 1.5, -2, 17, 5}, sd::DataType::DOUBLE);

  NDArray x = b.ulike();

  NDArray expX('c', {4, 2}, {1.462913, 1.835338, 0.4083664, -2.163816, -3.344481, -3.739225, 0.5156383, 0.01624954},
               sd::DataType::DOUBLE);

  ops::helpers::FullPivLU<double>::solve(a, b, x);

  ASSERT_TRUE(x.equalsTo(&expX));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, fullPivLU_3) {
  NDArray a1('c', {4, 3}, {0.33, 1.71, 6.20, 1.34, 5.38, -2.76, -8.51, 2.24, -6.82, 4.80, -4.67, 2.14},
             sd::DataType::DOUBLE);
  NDArray a2('c', {3, 4}, {0.33, 1.71, 6.20, 1.34, 5.38, -2.76, -8.51, 2.24, -6.82, 4.80, -4.67, 2.14},
             sd::DataType::DOUBLE);
  NDArray b1('c', {4, 2}, {-5., 10, 9, 1, 1.5, -2, 17, 5}, sd::DataType::DOUBLE);
  NDArray b2('c', {3, 2}, {-5., 10, 9, 1, 1.5, -2}, sd::DataType::DOUBLE);

  NDArray expX1('c', {3, 2}, {0.9344955, -0.5841325, 0.8768102, 1.029137, -1.098021, 1.360152}, sd::DataType::DOUBLE);
  NDArray expX2('c', {4, 2}, {0.3536033, 0.5270184, 0, 0, -0.8292221, 0.967515, 0.01827441, 2.856337},
                sd::DataType::DOUBLE);

  NDArray x1 = expX1.ulike();
  ops::helpers::FullPivLU<double>::solve(a1, b1, x1);
  ASSERT_TRUE(x1.equalsTo(&expX1));

  NDArray x2 = expX2.ulike();
  ops::helpers::FullPivLU<double>::solve(a2, b2, x2);
  ASSERT_TRUE(x2.equalsTo(&expX2));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, fullPivLU_4) {
  NDArray a('c', {10, 10},
            {6.9,  4.8,  9.5,  3.1,  6.5,  5.8,  -0.9, -7.3, -8.1, 3.0,  0.1,  9.9,  -3.2, 6.4,  6.2,  -7.0, 5.5,
             -2.2, -4.0, 3.7,  -3.6, 9.0,  -1.4, -2.4, 1.7,  -6.1, -4.2, -2.5, -5.6, -0.4, 0.4,  9.1,  -2.1, -5.4,
             7.3,  3.6,  -1.7, -5.7, -8.0, 8.8,  -3.0, -0.5, 1.1,  10.0, 8.0,  0.8,  1.0,  7.5,  3.5,  -1.8, 0.3,
             -0.6, -6.3, -4.5, -1.1, 1.8,  0.6,  9.6,  9.2,  9.7,  -2.6, 4.3,  -3.4, 0.0,  -6.7, 5.0,  10.5, 1.5,
             -7.8, -4.1, -5.3, -5.0, 2.0,  -4.4, -8.4, 6.0,  -9.4, -4.8, 8.2,  7.8,  5.2,  -9.5, -3.9, 0.2,  6.8,
             5.7,  -8.5, -1.9, -0.3, 7.4,  -8.7, 7.2,  1.3,  6.3,  -3.7, 3.9,  3.3,  -6.0, -9.1, 5.9},
            sd::DataType::DOUBLE);
  NDArray b('c', {10, 2}, {-5., 10, 9, 1, 1.5, -2, 17, 5, 3.6, 0.12, -3.1, 2.27, -0.5, 27.3, 8.9, 5, -7, 8, -9, 10},
            sd::DataType::DOUBLE);

  NDArray x = b.ulike();

  NDArray expX('c', {10, 2}, {-0.697127, 2.58257,    2.109721,  3.160622,  -2.217796, -3.275736, -0.5752479,
                              2.475356,  1.996841,   -1.928947, 2.213154,  3.541014,  0.7104885, -1.981451,
                              -3.297972, -0.4720612, 3.672657,  0.9161028, -2.322383, -1.784493},
               sd::DataType::DOUBLE);

  ops::helpers::FullPivLU<double>::solve(a, b, x);

  ASSERT_TRUE(x.equalsTo(&expX));
}
