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
// Created by raver on 8/4/2018.
//

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <NDArray.h>
#include <ops/ops.h>
#include <GradCheck.h>
#include <array>


using namespace nd4j;


class DeclarableOpsTests15 : public testing::Test {
public:

    DeclarableOpsTests15() {
        printf("\n");
        fflush(stdout);
    }
};

TEST_F(DeclarableOpsTests15, Test_NormalizeMoments_1) {
    auto d = NDArrayFactory::create<double>('c', {10, 10});
    auto w = NDArrayFactory::create<double>(10);
    auto x = NDArrayFactory::create<double>('c', {10});
    auto y = NDArrayFactory::create<double>('c', {10});

    auto z0 = NDArrayFactory::create<double>('c', {10});
    auto z1 = NDArrayFactory::create<double>('c', {10});

    nd4j::ops::normalize_moments op;
    auto result = op.execute({&w, &x, &y}, {&z0, &z1}, {1e-4}, {}, {});
    ASSERT_EQ(Status::OK(), result);
}

TEST_F(DeclarableOpsTests15, Test_Add_1) {
    auto x = NDArrayFactory::create<int>('c', {5}, {1, 1, 1, 1, 1});
    auto y = NDArrayFactory::create<int>('c', {5}, {1, 1, 1, 1, 1});
    auto e = NDArrayFactory::create<int>('c', {5}, {2, 2, 2, 2, 2});

    nd4j::ops::add op;
    auto result = op.execute({&x, &y}, {&x}, {}, {}, {});
    ASSERT_EQ(Status::OK(), result);
    ASSERT_EQ(e, x);
}

TEST_F(DeclarableOpsTests15, Test_Half_assign_1) {
    auto x = NDArrayFactory::create<float16>('c', {2, 5});
    int y = 1;
    x.assign(y);

    ASSERT_EQ(10, x.sumNumber().e<int>(0));
}

TEST_F(DeclarableOpsTests15, test_avgpooling_edge_1) {
    int inOutH = 5;// 35;
    int inOutW = 5;// 35;
    int inOutC = 10;// 192;

    auto x = NDArrayFactory::create<double>('c', {1, inOutH, inOutW, inOutC});
    x.linspace(1.0);

    nd4j::ops::avgpool2d op;
    auto result = op.execute({&x}, {}, {3,3, 1,1, 0,0, 1,1, 1, 0, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    int totalPadHeight = (inOutH - 1) * 1 + 3 - inOutH;
    int padTop = totalPadHeight / 2;
    int padBottom = totalPadHeight - totalPadHeight / 2;

    int k = 3;

    auto m = NDArrayFactory::create<double>('c', {1, inOutH, inOutW, inOutC});
    auto c = NDArrayFactory::create<double>('c', {1, inOutH, inOutW, inOutC});

    for (int h = 0; h < inOutH; h++) {
        for (int w = 0; w < inOutW; w++) {
            int hFrom = h - padTop;
            int wFrom = w - padBottom;

            int hTo = hFrom + k;
            int wTo = wFrom + k;

            hFrom = nd4j::math::nd4j_max<int>(0, hFrom);
            wFrom = nd4j::math::nd4j_max<int>(0, wFrom);

            hTo = nd4j::math::nd4j_min<int>(inOutH, hTo);
            wTo = nd4j::math::nd4j_min<int>(inOutW, wTo);

            int idxOut[4];
            int idxIn[4];
            for (int ch = 0; ch < inOutC; ch++) {
                idxOut[1] = h;
                idxOut[2] = w;
                idxOut[3] = ch;
                idxIn[3] = ch;

                for (int kh = hFrom; kh < hTo; kh++) {
                    for (int kw = wFrom; kw < wTo; kw++) {
                        idxIn[1] = kh;
                        idxIn[2] = kw;

                        auto inVal = x.e<double>(0, kh, kw, ch);
                        m.p(0, h, w, ch, inVal + m.e<double>(0, h, w, ch));
                        c.p(0, h, w, ch, 1 + c.e<int>(0, h, w, ch));
                    }
                }
            }
        }
    }
    m /= c;

    ASSERT_EQ(m, *z);

    delete result;
}

TEST_F(DeclarableOpsTests15, Test_standarize_1) {
    auto x = NDArrayFactory::create<float>('c', {5}, {1, 1, 1, 1, 1});
    auto e = NDArrayFactory::create<float>('c', {5}, {0, 0, 0, 0, 0});

    nd4j::ops::standardize op;
    auto result = op.execute({&x}, {&x}, {}, {0}, {});
    ASSERT_EQ(Status::OK(), result);
    ASSERT_EQ(e, x);
}

TEST_F(DeclarableOpsTests15, Test_standarize_bp_1) {
    auto x = NDArrayFactory::create<float>('c', {5}, {1.f, 1.f, 1.f, 1.f, 1.f});
    auto eps = NDArrayFactory::create<float>('c', {5}, {0.f, 0.f, 0.f, 0.f, 0.f});

    nd4j::ops::standardize_bp op;
    auto result = op.execute({&x, &eps}, {}, {0}, {});
    ASSERT_EQ(Status::OK(), result->status());
    delete result;
}

TEST_F(DeclarableOpsTests15, Test_AdjustContrast_1) {
    auto x = NDArrayFactory::create<double>('c', {4,4,3});
    NDArray factor = NDArrayFactory::create<double>(2.);
    auto e = NDArrayFactory::create<double>('c', {4,4,3}, {-21.5, -20.5, -19.5,  -15.5, -14.5, -13.5,  -9.5,  -8.5,  -7.5,  -3.5,  -2.5,  -1.5,
                                     2.5,   3.5,   4.5,    8.5,   9.5,  10.5,  14.5,  15.5,  16.5,  20.5,  21.5,  22.5,
                                    26.5,  27.5,  28.5,   32.5,  33.5,  34.5,  38.5,  39.5,  40.5,  44.5,  45.5,  46.5,
                                    50.5,  51.5,  52.5,   56.5,  57.5,  58.5,  62.5,  63.5,  64.5,  68.5,  69.5,  70.5});


    x.linspace(1.);
    nd4j::ops::adjust_contrast op;
    auto result = op.execute({&x, &factor}, {}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());
    auto out = result->at(0);

    ASSERT_TRUE(e.equalsTo(out));
    delete result;
}

TEST_F(DeclarableOpsTests15, Test_AdjustContrast_2) {
    auto x = NDArrayFactory::create<float>('c', {1, 4,4,3});
    auto e = NDArrayFactory::create<float>('c', {1, 4,4,3}, {
            -21.5f, -20.5f, -19.5f,  -15.5f, -14.5f, -13.5f,  -9.5f,  -8.5f,  -7.5f,  -3.5f,  -2.5f,  -1.5f,
            2.5f,   3.5f,   4.5f,    8.5f,   9.5f,  10.5f,  14.5f,  15.5f,  16.5f,  20.5f,  21.5f,  22.5f,
            26.5f,  27.5f,  28.5f,   32.5f,  33.5f,  34.5f,  38.5f,  39.5f,  40.5f,  44.5f,  45.5f,  46.5f,
            50.5f,  51.5f,  52.5f,   56.5f,  57.5f,  58.5f,  62.5f,  63.5f,  64.5f,  68.5f,  69.5f,  70.5f
    });
    x.linspace(1.);
    nd4j::ops::adjust_contrast op;
    auto result = op.execute({&x}, {2.}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());
    auto out = result->at(0);
//    out->printIndexedBuffer("Adjusted Constrast");
    ASSERT_TRUE(e.equalsTo(out));
    delete result;
}

TEST_F(DeclarableOpsTests15, Test_AdjustContrast_3) {
    auto x = NDArrayFactory::create<float>('c', {1, 4,4,3});
    auto e = NDArrayFactory::create<float>('c', {1, 4,4,3}, {
            -21.5f, -20.5f, -19.5f,  -15.5f, -14.5f, -13.5f,  -9.5f,  -8.5f,  -7.5f,  -3.5f,  -2.5f,  -1.5f,
            2.5f,   3.5f,   4.5f,    8.5f,   9.5f,  10.5f,  14.5f,  15.5f,  16.5f,  20.5f,  21.5f,  22.5f,
            26.5f,  27.5f,  28.5f,   32.5f,  33.5f,  34.5f,  38.5f,  39.5f,  40.5f,  44.5f,  45.5f,  46.5f,
            50.5f,  51.5f,  52.5f,   56.5f,  57.5f,  58.5f,  62.5f,  63.5f,  64.5f,  68.5f,  69.5f,  70.5f
    });
    x.linspace(1.);
    nd4j::ops::adjust_contrast_v2 op;
    auto result = op.execute({&x}, {2.}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());
    auto out = result->at(0);
//    out->printIndexedBuffer("Adjusted Constrast");
    ASSERT_TRUE(e.equalsTo(out));
    delete result;
}

TEST_F(DeclarableOpsTests15, Test_AdjustContrast_4) {
    auto x = NDArrayFactory::create<double>('c', {4, 4, 3});
    auto e = NDArrayFactory::create<double>('c', {4, 4, 3}, {
            -21.5, -20.5, -19.5,  -15.5, -14.5, -13.5,  -9.5,  -8.5,  -7.5,  -3.5,  -2.5,  -1.5,
            2.5,   3.5,   4.5,    8.5,   9.5,  10.5,  14.5,  15.5,  16.5,  20.5,  21.5,  22.5,
            26.5,  27.5,  28.5,   32.5,  33.5,  34.5,  38.5,  39.5,  40.5,  44.5,  45.5,  46.5,
            50.5,  51.5,  52.5,   56.5,  57.5,  58.5,  62.5,  63.5,  64.5,  68.5,  69.5,  70.5
    });
    x.linspace(1.);
    nd4j::ops::adjust_contrast_v2 op;
    auto result = op.execute({&x}, {2.}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());
    auto out = result->at(0);
//    out->printIndexedBuffer("Adjusted Constrast");
    ASSERT_TRUE(e.equalsTo(out));
    delete result;
}

TEST_F(DeclarableOpsTests15, Test_AdjustContrast_5) {
    auto x = NDArrayFactory::create<double>('c', {1, 3, 4});
    auto e = NDArrayFactory::create<double>('c', {1, 3, 4}, {
        -3., -2., -1.,  0.,      5.,  6.,  7.,  8.,     13., 14., 15., 16.
    });
    x.linspace(1.);
    nd4j::ops::adjust_contrast_v2 op;
    auto result = op.execute({&x}, {2.}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());
    auto out = result->at(0);
//    out->printIndexedBuffer("Adjusted Constrast");
    ASSERT_TRUE(e.equalsTo(out));
    delete result;
}

/*
 * public void testAdjustContrast1() {
        INDArray in = Nd4j.createFromArray(new float[]{0.7788f,0.8012f,0.7244f,0.2309f,0.7271f,0.1804f,
            0.5056f,0.8925f,0.5461f,0.9234f,0.0856f,0.7938f,0.6591f,0.5555f,0.1596f,0.3087f,0.1548f,0.4695f,
            0.9939f,0.6113f,0.6765f,0.1800f,0.6750f,0.2246f,0.0509f,0.4601f,0.8284f,0.2354f,0.9752f,0.8361f,
            0.2585f,0.4189f,0.7028f,0.7679f,0.5373f,0.7234f,0.2690f,0.0062f,0.0327f,0.0644f,0.8428f,0.7494f,
            0.0755f,0.6245f,0.3491f,0.5793f,0.5730f,0.1822f,0.6420f,0.9143f,0.3019f,
            0.3574f,0.1704f,0.8395f,0.5468f,0.0744f,0.9011f,0.6574f,0.4124f,0.2445f,0.4248f,0.5219f,
            0.6952f,0.4900f,0.2158f,0.9549f,0.1386f,0.1544f,0.5365f,0.0134f,0.4163f,0.1456f,0.4109f,
                0.2484f, 0.3330f,0.2974f,0.6636f,0.3808f,0.8664f, 0.1896f, 0.7530f, 0.7215f, 0.6612f, 0.7270f,
            0.5704f,0.2666f,0.7453f,0.0444f,0.3024f,0.4850f,0.7982f,0.0965f,0.7843f,0.5075f,
            0.0844f,0.8370f,0.6103f,0.4604f,0.6087f, 0.8594f, 0.4599f, 0.6714f, 0.2744f, 0.1981f, 0.4143f,
            0.7821f,0.3505f,0.5040f,0.1180f,0.8307f,0.1817f,0.8442f,0.5074f,0.4471f,0.5105f,0.6666f,
            0.2576f,0.2341f,0.6801f,0.2652f,0.5394f,0.4690f,0.6146f,0.1210f,0.2576f,0.0769f,0.4643f,
            0.1628f,0.2026f,0.3774f,0.0506f,0.3462f,0.5720f,0.0838f,0.4228f,0.0588f,0.5362f,0.4756f,
            0.2530f,0.1778f,0.0751f,0.8977f,0.3648f,0.3065f,0.4739f,0.7014f,0.4473f,0.5171f,0.1744f,
            0.3487f,0.7759f,0.9491f,0.2072f,0.2182f,0.6520f,0.3092f,0.9545f,0.1881f,0.9579f,0.1785f,
            0.9636f,0.4830f,0.6569f,0.3353f,0.9997f,0.5869f,0.5747f,0.0238f,0.2943f,0.5248f,0.5879f,
            .7266f,0.1965f,0.9167f,0.9726f,0.9206f,0.0519f,0.2997f,0.0039f,0.7652f,0.5498f,
            0.3794f,0.3791f,0.3528f,0.2873f,0.8082f,0.4732f,0.4399f,0.6606f,0.5991f,0.0034f,0.4874f
        }).reshape(8,8,3,1);
        INDArray out = Nd4j.create(DataType.FLOAT, in.shape());
        INDArray[] res = Nd4j.exec(new AdjustContrast(in, 2.0, out));
        assertArrayEquals(out.shape(), in.shape());
        //assertEquals(expected, out);
    }
 * */

TEST_F(DeclarableOpsTests15, Test_AdjustContrast_6) {
    auto x = NDArrayFactory::create<float>('c', {8,8, 3, 1}, {0.7788f,0.8012f,0.7244f,0.2309f,0.7271f,0.1804f,
                                                              0.5056f,0.8925f,0.5461f,0.9234f,0.0856f,0.7938f,0.6591f,0.5555f,0.1596f,0.3087f,0.1548f,0.4695f,
                                                              0.9939f,0.6113f,0.6765f,0.1800f,0.6750f,0.2246f,0.0509f,0.4601f,0.8284f,0.2354f,0.9752f,0.8361f,
                                                              0.2585f,0.4189f,0.7028f,0.7679f,0.5373f,0.7234f,0.2690f,0.0062f,0.0327f,0.0644f,0.8428f,0.7494f,
                                                              0.0755f,0.6245f,0.3491f,0.5793f,0.5730f,0.1822f,0.6420f,0.9143f,0.3019f,
                                                              0.3574f,0.1704f,0.8395f,0.5468f,0.0744f,0.9011f,0.6574f,0.4124f,0.2445f,0.4248f,0.5219f,
                                                              0.6952f,0.4900f,0.2158f,0.9549f,0.1386f,0.1544f,0.5365f,0.0134f,0.4163f,0.1456f,0.4109f,
                                                              0.2484f, 0.3330f,0.2974f,0.6636f,0.3808f,0.8664f, 0.1896f, 0.7530f, 0.7215f, 0.6612f, 0.7270f,
                                                              0.5704f,0.2666f,0.7453f,0.0444f,0.3024f,0.4850f,0.7982f,0.0965f,0.7843f,0.5075f,
                                                              0.0844f,0.8370f,0.6103f,0.4604f,0.6087f, 0.8594f, 0.4599f, 0.6714f, 0.2744f, 0.1981f, 0.4143f,
                                                              0.7821f,0.3505f,0.5040f,0.1180f,0.8307f,0.1817f,0.8442f,0.5074f,0.4471f,0.5105f,0.6666f,
                                                              0.2576f,0.2341f,0.6801f,0.2652f,0.5394f,0.4690f,0.6146f,0.1210f,0.2576f,0.0769f,0.4643f,
                                                              0.1628f,0.2026f,0.3774f,0.0506f,0.3462f,0.5720f,0.0838f,0.4228f,0.0588f,0.5362f,0.4756f,
                                                              0.2530f,0.1778f,0.0751f,0.8977f,0.3648f,0.3065f,0.4739f,0.7014f,0.4473f,0.5171f,0.1744f,
                                                              0.3487f,0.7759f,0.9491f,0.2072f,0.2182f,0.6520f,0.3092f,0.9545f,0.1881f,0.9579f,0.1785f,
                                                              0.9636f,0.4830f,0.6569f,0.3353f,0.9997f,0.5869f,0.5747f,0.0238f,0.2943f,0.5248f,0.5879f,
                                                              .7266f,0.1965f,0.9167f,0.9726f,0.9206f,0.0519f,0.2997f,0.0039f,0.7652f,0.5498f,
                                                              0.3794f,0.3791f,0.3528f,0.2873f,0.8082f,0.4732f,0.4399f,0.6606f,0.5991f,0.0034f,0.4874f});
    auto e = NDArrayFactory::create<float>('c', {8, 8, 3, 1}, {
            1.0218375f,
            1.0666375f,
            0.9130375f,

            -0.07396251f,
             0.91843754f,
            -0.17496246f,

            0.47543746f,
             1.2492375f,
            0.55643755f,

              1.3110375f,
            -0.36456245f,
              1.0518374f,

              0.7824375f,
             0.57523745f,
            -0.21656245f,

             0.0816375f,
            -0.2261625f,
            0.40323752f,

             1.4520376f,
             0.6868375f,
            0.81723756f,

            -0.17576247f,
             0.81423753f,
            -0.08656245f,


            -0.36249164f,
             0.45590833f,
              1.1925083f,

            0.00650835f,
             1.4861084f,
             1.2079083f,

            0.05270836f,
            0.37350836f,
            0.94130826f,

            1.0715083f,
            0.6103083f,
            0.9825083f,

             0.07370833f,
             -0.4518917f,
            -0.39889166f,

            -0.3354917f,
             1.2213084f,
             1.0345083f,

            -0.3132917f,
            0.78470826f,
            0.23390833f,

              0.6943083f,
             0.68170834f,
            -0.09989169f,


             0.8352709f,
             1.3798709f,
            0.15507084f,

             0.26607084f,
            -0.10792917f,
              1.2302709f,

              0.6448709f,
            -0.29992914f,
              1.3534708f,

            0.86607087f,
            0.37607086f,
            0.04027084f,

            0.40087086f,
            0.59507084f,
             0.9416709f,

             0.53127086f,
            -0.01712915f,
              1.4610709f,

            -0.17152917f,
            -0.13992918f,
              0.6242708f,

            -0.42192918f,
             0.38387084f,
            -0.15752912f,


             0.3311833f,
            0.00618333f,
            0.17538333f,

            0.10418332f,
             0.8365834f,
            0.27098334f,

             1.2421833f,
            -0.1114167f,
             1.0153834f,

            0.9523833f,
            0.8317833f,
            0.9633833f,

             0.6501833f,
            0.04258335f,
             0.9999833f,

            -0.40181667f,
             0.11418331f,
             0.47938335f,

              1.1057833f,
            -0.29761666f,
              1.0779834f,

              0.5243833f,
            -0.32181668f,
              1.1833833f,


            0.73157084f,
             0.4317708f,
             0.7283708f,

             1.2297708f,
             0.4307708f,
            0.85377085f,

             0.05977082f,
            -0.09282917f,
             0.33957082f,

             1.0751709f,
             0.2119708f,
            0.51897085f,

            -0.25302917f,
              1.1723708f,
            -0.12562919f,

             1.1993709f,
             0.5257708f,
            0.40517086f,

            0.53197086f,
             0.8441708f,
            0.02617085f,

            -0.0208292f,
             0.8711709f,
            0.04137081f,


            0.74936247f,
             0.6085625f,
             0.8997625f,

            -0.08743751f,
             0.18576252f,
            -0.17563748f,

             0.5991625f,
            -0.0038375f,
            0.07576251f,

             0.42536253f,
            -0.22823751f,
             0.36296248f,

             0.81456256f,
            -0.16183749f,
              0.5161625f,

            -0.21183747f,
              0.7429625f,
              0.6217625f,

             0.17656249f,
             0.02616251f,
            -0.17923748f,

             1.4659625f,
            0.40016252f,
            0.28356248f,


             0.4195791f,
             0.8745791f,
            0.36637908f,

             0.50597906f,
            -0.17942089f,
             0.16917908f,

              1.0235791f,
              1.3699791f,
            -0.11382091f,

            -0.0918209f,
             0.7757791f,
            0.09017909f,

              1.3807791f,
            -0.15202093f,
              1.3875791f,

            -0.1712209f,
             1.3989791f,
            0.43777913f,

            0.7855791f,
            0.1423791f,
            1.4711791f,

              0.6455791f,
              0.6211791f,
            -0.48062086f,


            0.10189578f,
             0.5628958f,
            0.68909574f,

             0.96649575f,
            -0.09370419f,
              1.3466958f,

             1.4584957f,
             1.3544958f,
            -0.3829042f,

             0.11269578f,
            -0.47890422f,
              1.0436958f,

             0.6128957f,
            0.27209583f,
             0.2714958f,

            0.21889582f,
            0.08789578f,
             1.1296958f,

             0.4596958f,
            0.39309582f,
             0.8344958f,

            0.71149576f,
            -0.4799042f,
             0.4880958f
    });

    nd4j::ops::adjust_contrast op;
    auto result = op.execute({&x}, {2.}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());
    auto out = result->at(0);
//    out->printBuffer("Adjusted Constrast6");
//    e.printBuffer("Adjusted Expected 6");
//    ASSERT_TRUE(e.equalsTo(out));
    delete result;
}

TEST_F(DeclarableOpsTests15, Test_AdjustContrast_7) {
    auto x = NDArrayFactory::create<double>('c', {8,8, 3, 1}, {0.7788f,0.8012f,0.7244f,0.2309f,0.7271f,0.1804f,
                                                              0.5056f,0.8925f,0.5461f,0.9234f,0.0856f,0.7938f,0.6591f,0.5555f,0.1596f,0.3087f,0.1548f,0.4695f,
                                                              0.9939f,0.6113f,0.6765f,0.1800f,0.6750f,0.2246f,0.0509f,0.4601f,0.8284f,0.2354f,0.9752f,0.8361f,
                                                              0.2585f,0.4189f,0.7028f,0.7679f,0.5373f,0.7234f,0.2690f,0.0062f,0.0327f,0.0644f,0.8428f,0.7494f,
                                                              0.0755f,0.6245f,0.3491f,0.5793f,0.5730f,0.1822f,0.6420f,0.9143f,0.3019f,
                                                              0.3574f,0.1704f,0.8395f,0.5468f,0.0744f,0.9011f,0.6574f,0.4124f,0.2445f,0.4248f,0.5219f,
                                                              0.6952f,0.4900f,0.2158f,0.9549f,0.1386f,0.1544f,0.5365f,0.0134f,0.4163f,0.1456f,0.4109f,
                                                              0.2484f, 0.3330f,0.2974f,0.6636f,0.3808f,0.8664f, 0.1896f, 0.7530f, 0.7215f, 0.6612f, 0.7270f,
                                                              0.5704f,0.2666f,0.7453f,0.0444f,0.3024f,0.4850f,0.7982f,0.0965f,0.7843f,0.5075f,
                                                              0.0844f,0.8370f,0.6103f,0.4604f,0.6087f, 0.8594f, 0.4599f, 0.6714f, 0.2744f, 0.1981f, 0.4143f,
                                                              0.7821f,0.3505f,0.5040f,0.1180f,0.8307f,0.1817f,0.8442f,0.5074f,0.4471f,0.5105f,0.6666f,
                                                              0.2576f,0.2341f,0.6801f,0.2652f,0.5394f,0.4690f,0.6146f,0.1210f,0.2576f,0.0769f,0.4643f,
                                                              0.1628f,0.2026f,0.3774f,0.0506f,0.3462f,0.5720f,0.0838f,0.4228f,0.0588f,0.5362f,0.4756f,
                                                              0.2530f,0.1778f,0.0751f,0.8977f,0.3648f,0.3065f,0.4739f,0.7014f,0.4473f,0.5171f,0.1744f,
                                                              0.3487f,0.7759f,0.9491f,0.2072f,0.2182f,0.6520f,0.3092f,0.9545f,0.1881f,0.9579f,0.1785f,
                                                              0.9636f,0.4830f,0.6569f,0.3353f,0.9997f,0.5869f,0.5747f,0.0238f,0.2943f,0.5248f,0.5879f,
                                                              .7266f,0.1965f,0.9167f,0.9726f,0.9206f,0.0519f,0.2997f,0.0039f,0.7652f,0.5498f,
                                                              0.3794f,0.3791f,0.3528f,0.2873f,0.8082f,0.4732f,0.4399f,0.6606f,0.5991f,0.0034f,0.4874f});
    auto e = NDArrayFactory::create<double>('c', {8, 8, 3, 1}, {
            1.0218375 ,
             1.0666375 ,
             0.9130375 ,

            -0.07396251,
             0.91843754,
            -0.17496246,

             0.47543746,
             1.2492375 ,
             0.55643755,

             1.3110375 ,
            -0.36456245,
             1.0518374 ,

             0.7824375 ,
             0.57523745,
            -0.21656245,

             0.0816375 ,
            -0.2261625 ,
             0.40323752,

             1.4520376 ,
             0.6868375 ,
             0.81723756,

            -0.17576247,
             0.81423753,
            -0.08656245,


            -0.36249164,
             0.45590833,
             1.1925083 ,

             0.00650835,
             1.4861084 ,
             1.2079083 ,

             0.05270836,
             0.37350836,
             0.94130826,

             1.0715083 ,
             0.6103083 ,
             0.9825083 ,

             0.07370833,
            -0.4518917 ,
            -0.39889166,

            -0.3354917 ,
             1.2213084 ,
             1.0345083 ,

            -0.3132917 ,
             0.78470826,
             0.23390833,

             0.6943083 ,
             0.68170834,
            -0.09989169,


             0.8352709 ,
             1.3798709 ,
             0.15507084,

             0.26607084,
            -0.10792917,
             1.2302709 ,

             0.6448709 ,
            -0.29992914,
             1.3534708 ,

             0.86607087,
             0.37607086,
             0.04027084,

             0.40087086,
             0.59507084,
             0.9416709 ,

             0.53127086,
            -0.01712915,
             1.4610709 ,

            -0.17152917,
            -0.13992918,
             0.6242708 ,

            -0.42192918,
             0.38387084,
            -0.15752912,


             0.3311833 ,
             0.00618333,
             0.17538333,

             0.10418332,
             0.8365834 ,
             0.27098334,

             1.2421833 ,
            -0.1114167 ,
             1.0153834 ,

             0.9523833 ,
             0.8317833 ,
             0.9633833 ,

             0.6501833 ,
             0.04258335,
             0.9999833 ,

            -0.40181667,
             0.11418331,
             0.47938335,

             1.1057833 ,
            -0.29761666,
             1.0779834 ,

             0.5243833 ,
            -0.32181668,
             1.1833833 ,


             0.73157084,
             0.4317708 ,
             0.7283708 ,

             1.2297708 ,
             0.4307708 ,
             0.85377085,

             0.05977082,
            -0.09282917,
             0.33957082,

             1.0751709 ,
             0.2119708 ,
             0.51897085,

            -0.25302917,
             1.1723708 ,
            -0.12562919,

             1.1993709 ,
             0.5257708 ,
             0.40517086,

             0.53197086,
             0.8441708 ,
             0.02617085,

            -0.0208292 ,
             0.8711709 ,
             0.04137081,


             0.74936247,
             0.6085625 ,
             0.8997625 ,

            -0.08743751,
             0.18576252,
            -0.17563748,

             0.5991625 ,
            -0.0038375 ,
             0.07576251,

             0.42536253,
            -0.22823751,
             0.36296248,

             0.81456256,
            -0.16183749,
             0.5161625 ,

            -0.21183747,
             0.7429625 ,
             0.6217625 ,

             0.17656249,
             0.02616251,
            -0.17923748,

             1.4659625 ,
             0.40016252,
             0.28356248,


             0.4195791 ,
             0.8745791 ,
             0.36637908,

             0.50597906,
            -0.17942089,
             0.16917908,

             1.0235791 ,
             1.3699791 ,
            -0.11382091,

            -0.0918209 ,
             0.7757791 ,
             0.09017909,

             1.3807791 ,
            -0.15202093,
             1.3875791 ,

            -0.1712209 ,
             1.3989791 ,
             0.43777913,

             0.7855791 ,
             0.1423791 ,
             1.4711791 ,

             0.6455791 ,
             0.6211791 ,
            -0.48062086,


             0.10189578,
             0.5628958 ,
             0.68909574,

             0.96649575,
            -0.09370419,
             1.3466958 ,

             1.4584957 ,
             1.3544958 ,
            -0.3829042 ,

             0.11269578,
            -0.47890422,
             1.0436958 ,

             0.6128957 ,
             0.27209583,
             0.2714958 ,

             0.21889582,
             0.08789578,
             1.1296958 ,

             0.4596958 ,
             0.39309582,
             0.8344958 ,

             0.71149576,
            -0.4799042,
             0.4880958
    });
//    x.linspace(1.);
    nd4j::ops::adjust_contrast_v2 op;
    auto result = op.execute({&x}, {2.}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());
    auto out = result->at(0);
//    out->printBuffer("Adjusted Constrast7");
//    e.printBuffer("Adjusted expected 7");
    auto diff = e - *out;
//    diff.printBuffer("Adjusted subtract 7");
    ASSERT_TRUE(e.equalsTo(out));
    delete result;
}

TEST_F(DeclarableOpsTests15, Test_BitCast_1) {
    auto x = NDArrayFactory::create<float>('c', {2, 2, 2});
    auto e = NDArrayFactory::create<double>('c', {2, 2}, {2., 512., 8192., 131072.032 });
    x.linspace(1.);
    nd4j::ops::bitcast op;
    auto result = op.execute({&x}, {}, {nd4j::DataType::DOUBLE}, {});
    ASSERT_EQ(Status::OK(), result->status());
    auto out = result->at(0);
//    out->printIndexedBuffer("Casted result");
    ASSERT_TRUE(e.equalsTo(out));
    delete result;
}

TEST_F(DeclarableOpsTests15, Test_BitCast_2) {
    auto x = NDArrayFactory::create<float>('c', {2, 4});
    auto e = NDArrayFactory::create<float16>('c', {2, 4, 2}, {0.f, 1.875f, 0.f, 2.f,    0.f, 2.125f, 0.f,  2.25f,
                                                                              0.f, 2.312f, 0.f, 2.375f, 0.f, 2.438f, 0.f, 2.5f});
    x.linspace(1.);
    nd4j::ops::bitcast op;
    auto result = op.execute({&x}, {}, {nd4j::DataType::HALF}, {});
    ASSERT_EQ(Status::OK(), result->status());
    auto out = result->at(0);
    ASSERT_TRUE(e.equalsTo(out));
    delete result;
}

TEST_F(DeclarableOpsTests15, Test_BitCast_3) {
    auto x = NDArrayFactory::create<float>('c', {1, 4});

    x.linspace(1.);
    nd4j::ops::bitcast op;
    try {
        auto result = op.execute({&x}, {}, {nd4j::DataType::INT64}, {});
        ASSERT_NE(Status::OK(), result->status());
        delete result;
    } catch (std::exception& e) {
        nd4j_printf("Error should be here `%s'. It's OK.\n", e.what());
    }
}

TEST_F(DeclarableOpsTests15, Test_BitCast_4) {
    auto x = NDArrayFactory::create<float>('c', {1, 4});
    auto e = NDArrayFactory::create<Nd4jLong>('c', {1, 2}, {1234567890LL, 2468013579LL});
    x.linspace(1.);
    nd4j::ops::bitcast op;
    try {
        auto result = op.execute({&x}, {&e}, {}, {nd4j::DataType::INT64}, {});
        ASSERT_NE(Status::OK(), result);
    } catch(std::exception& e) {
        nd4j_printf("Error `%s' should be here. It's OK.\n",e.what());
    }

}


TEST_F(DeclarableOpsTests15, Test_BitCast_5) {
    auto x = NDArrayFactory::create<float16>('c', {4, 4}, {
        0.4922f,    0.2969f,    0.6172f,    0.8906f,
        0.9297f,    0.0859f,    0.2344f,    0.3828f,
        0.5781f,    0.7969f,    0.0391f,    0.1719f,
        0.8359f,    0.9297f,    0.3438f,    0.0938f});

    auto e = NDArrayFactory::create<Nd4jLong>('c', {4}, {4260467851820808160LL, 3900173902914993008LL, 3566895990128523424LL,
                                                         3314989625590692528LL});
    nd4j::ops::bitcast op;
    auto result = op.execute({&x}, {}, {nd4j::DataType::INT64}, {});
    ASSERT_EQ(Status::OK(), result->status());
    auto res = result->at(0);
//    res->printIndexedBuffer("BITCAST5");
    ASSERT_TRUE(e.equalsTo(res));
    delete result;
}

TEST_F(DeclarableOpsTests15, Test_BitCast_6) {
    auto x = NDArrayFactory::create<float16>('c', {4, 4}, {
            1.f,    2.f,    3.f,    4.f,
            5.f,    6.f,    7.f,    8.f,
            9.f,   10.f,   11.f,   12.f,
           13.f,   14.f,   15.f,   16.f});

    auto e = NDArrayFactory::create<Nd4jLong>('c', {4}, {4899988963420290048LL, 5188224837230806272LL, 5332342774136064128LL,
                                                         5476460161268730496LL});
    nd4j::ops::bitcast op;
    auto result = op.execute({&x}, {}, {nd4j::DataType::INT64}, {});
    ASSERT_EQ(Status::OK(), result->status());
    auto res = result->at(0);
//    res->printIndexedBuffer("BITCAST6");
    ASSERT_TRUE(e.equalsTo(res));
    delete result;
}
TEST_F(DeclarableOpsTests15, Test_BitCast_7) {
    auto x = NDArrayFactory::create<float16>('c', {4, 4}, {
            1.1f,    2.2f,    3.3f,    4.4f,
            5.1f,    6.2f,    7.3f,    8.4f,
            9.1f,   10.2f,   11.3f,   12.4f,
            13.f,   14.2f,   15.3f,   16.4f});

    auto e = NDArrayFactory::create<Nd4jLong>('c', {4}, {
        4928700072476425318LL, 5202580391758873882LL, 5346698272827918477LL,  5483778673873668736LL});
    nd4j::ops::bitcast op;
    auto result = op.execute({&x}, {}, {nd4j::DataType::INT64}, {});
    ASSERT_EQ(Status::OK(), result->status());
    auto res = result->at(0);
//    res->printIndexedBuffer("BITCAST7");
    ASSERT_TRUE(e.equalsTo(res));
    delete result;
}

TEST_F(DeclarableOpsTests15, Test_depthwise_bp_1) {
    auto in = NDArrayFactory::create<float>('c', {4, 8, 64, 64});
    auto w = NDArrayFactory::create<float>('c', {2, 2, 8, 2});
    auto b = NDArrayFactory::create<float>('c', {1, 16});
    auto grad = NDArrayFactory::create<float>('c', {4, 16, 64, 64});

    auto gradI = in.like();
    auto gradW = w.like();
    auto gradB = b.like();

    nd4j:ops::depthwise_conv2d_bp op;
    auto status = op.execute({&in, &w, &b, &grad}, {&gradI, &gradW, &gradB}, {}, {2, 2, 1, 1, 0, 0, 1, 1, 1, 0}, {});
    ASSERT_EQ(Status::OK(), status);
}

TEST_F(DeclarableOpsTests15, test_matmul_bp_1) {
    auto a = NDArrayFactory::create<double>('c', {1, 3});
    auto b = NDArrayFactory::create<double>('c', {1, 4});
    auto gI = NDArrayFactory::create<double>('c', {3, 4});

    auto gA = NDArrayFactory::create<double>('c', {1, 3});
    auto gB = NDArrayFactory::create<double>('c', {1, 4});

    nd4j::ops::matmul_bp op;
    auto status = op.execute({&a, &b, &gI}, {&gA, &gB}, {}, {1, 0, 0}, {});
    ASSERT_EQ(Status::OK(), status);
}

TEST_F(DeclarableOpsTests15, test_non_decreasing_1) {
    auto x = NDArrayFactory::create<double>(1.0);
    auto z = NDArrayFactory::create<bool>(false);
    auto e = NDArrayFactory::create<bool>(true);

    nd4j::ops::is_non_decreasing op;
    Context ctx(1);
    ctx.setInputArray(0, &x);
    ctx.setOutputArray(0, &z);

    auto status = op.execute(&ctx);
    ASSERT_EQ(Status::OK(), status);
    ASSERT_EQ(e, z);
}

TEST_F(DeclarableOpsTests15, test_check_numeric_1) {
    auto x = NDArrayFactory::create<float>('c', {3},{1.f, 2.f, 3.f});
    auto y = NDArrayFactory::string("shouldn't ever trigger");

    nd4j::ops::check_numerics op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(x, *z);

    delete result;
}

TEST_F(DeclarableOpsTests15, test_check_numeric_2) {
    auto x = NDArrayFactory::create<float>('c', {3},{1.f, 2.f, std::numeric_limits<float>::infinity()});
    auto y = NDArrayFactory::string("should trigger");
    auto z = NDArrayFactory::create<float>('c', {3} );

    nd4j::ops::check_numerics op;
    try {
        auto status = op.execute({&x, &y}, {&z}, {}, {}, {});
        ASSERT_TRUE(false);
    } catch (std::invalid_argument &e) {
        //
    }
}

TEST_F(DeclarableOpsTests15, test_check_numeric_3) {
    auto x = NDArrayFactory::create<float>('c', {3},{1.f, 2.f, std::numeric_limits<float>::quiet_NaN()});
    auto y = NDArrayFactory::string("should trigger");
    auto z = NDArrayFactory::create<float>('c', {3} );

    nd4j::ops::check_numerics op;
    try {
        auto status = op.execute({&x, &y}, {&z}, {}, {}, {});
        ASSERT_TRUE(false);
    } catch (std::invalid_argument &e) {
        //
    }
}

TEST_F(DeclarableOpsTests15, Test_layer_norm_1) {
    auto x = NDArrayFactory::create<float>('c', {1, 5}, {1.f, 2.f, 3.f, 4.f, 5.f});
    auto g = NDArrayFactory::create<float>('c', {5}, {1.f, 2.f, 3.f, 4.f, 5.f});
    auto b = NDArrayFactory::create<float>('c', {5}, {1.f, 2.f, 3.f, 4.f, 5.f});

    nd4j::ops::layer_norm op;
    auto result = op.execute({&x, &g, &b}, {}, {0}, {false});
    ASSERT_EQ(Status::OK(), result->status());
    delete result;
}

TEST_F(DeclarableOpsTests15, Test_layer_norm_bp_1) {
    auto x = NDArrayFactory::create<float>('c', {1, 5}, {1.f, 2.f, 3.f, 4.f, 5.f});
    auto g = NDArrayFactory::create<float>('c', {5}, {1.f, 2.f, 3.f, 4.f, 5.f});
    auto b = NDArrayFactory::create<float>('c', {5}, {1.f, 2.f, 3.f, 4.f, 5.f});
    auto eps = NDArrayFactory::create<float>('c', {1, 5}, {0.f, 0.f, 0.f, 0.f, 0.f});

    nd4j::ops::layer_norm_bp op;
    auto result = op.execute({&x, &g, &b, &eps}, {}, {0}, {false});
    ASSERT_EQ(Status::OK(), result->status());
    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, Test_layer_norm_bp_2) {

    NDArray x('c', {3, 4, 8, 8}, nd4j::DataType::FLOAT32);
    NDArray gain('c', {4}, {-0.1, 0.1, -0.2, 0.2}, nd4j::DataType::FLOAT32);
    NDArray bias('c', {4}, {-0.05, 0.05, -1.05, 1.05}, nd4j::DataType::FLOAT32);
    NDArray gradO('c', {3, 4, 8, 8}, nd4j::DataType::FLOAT32);

    NDArray gradI('c', {3, 4, 8, 8}, nd4j::DataType::FLOAT32);
    NDArray gradG('c', {4}, nd4j::DataType::FLOAT32);
    NDArray gradB('c', {4}, nd4j::DataType::FLOAT32);

    x.linspace(-20, 0.5);
    gradO.linspace(-4, 0.05);

    nd4j::ops::layer_norm_bp op;
    auto status = op.execute({&x, &gain, &bias, &gradO}, {&gradI, &gradG, &gradB}, {}, {1,2,3}, {true});
    ASSERT_EQ(Status::OK(), status);
}

TEST_F(DeclarableOpsTests15, test_hashCode_1) {
    auto x = NDArrayFactory::create<int>('c', {10});
    auto y = NDArrayFactory::create<int>('c', {10});

    x.linspace(1.);
    y.linspace(2.);

    nd4j::ops::hashcode op;
    auto resultA0 = op.execute({&x}, {}, {}, {}, false, nd4j::DataType::INT64);
    auto resultA1 = op.execute({&x}, {}, {}, {}, false, nd4j::DataType::INT64);
    auto resultB0 = op.execute({&y}, {}, {}, {}, false, nd4j::DataType::INT64);
//    resultA0->at(0)->printIndexedBuffer("A0");
//    resultA1->at(0)->printIndexedBuffer("A1");
//    resultB0->at(0)->printIndexedBuffer("B0");
    ASSERT_EQ(*resultA0->at(0), *resultA1->at(0));
    ASSERT_NE(*resultA0->at(0), *resultB0->at(0));

    delete resultA0;
    delete resultA1;
    delete resultB0;
}

TEST_F(DeclarableOpsTests15, test_hashCode_2) {
    auto x = NDArrayFactory::create<int>('c', {1027});
    auto y = NDArrayFactory::create<int>('c', {1027});

    x.linspace(1.);
    y.linspace(2.);

    nd4j::ops::hashcode op;
    auto resultA0 = op.execute({&x}, {}, {}, {}, false, nd4j::DataType::INT64);
    auto resultA1 = op.execute({&x}, {}, {}, {}, false, nd4j::DataType::INT64);
    auto resultB0 = op.execute({&y}, {}, {}, {}, false, nd4j::DataType::INT64);

//    resultA0->at(0)->printIndexedBuffer("A0");
//    resultA1->at(0)->printIndexedBuffer("A1");
//    resultB0->at(0)->printIndexedBuffer("B0");

    ASSERT_EQ(*resultA0->at(0), *resultA1->at(0));
    ASSERT_NE(*resultA0->at(0), *resultB0->at(0));

    delete resultA0;
    delete resultA1;
    delete resultB0;
}

TEST_F(DeclarableOpsTests15, test_reshape_to_scalar_1) {
    auto array = NDArrayFactory::create<float>(119.f);
    auto e = NDArrayFactory::create<float>('c', {1, 1}, {119.f});

    nd4j::ops::reshape op;
    auto result = op.execute({&array}, {}, {1, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests15, test_reshape_to_scalar_2) {
    auto array = NDArrayFactory::create<float>(119.f);
    auto e = NDArrayFactory::create<float>('c', {1, 1}, {119.f});
    auto z = NDArrayFactory::create<float>('c', {1, 1});

    nd4j::ops::reshape op;
    auto result = op.execute({&array}, {&z}, {}, {1, 1}, {});
    ASSERT_EQ(Status::OK(), result);
    ASSERT_EQ(e, z);
}

TEST_F(DeclarableOpsTests15, test_rank_1) {
    auto array = NDArrayFactory::create<float>('c', {4, 64});
    auto e = NDArrayFactory::create<int>('c', {}, {2});
    auto z = NDArrayFactory::create<int>('c', {});

    nd4j::ops::rank op;
    auto result = op.execute({&array}, {&z}, {}, {}, {});
    ASSERT_EQ(Status::OK(), result);
    ASSERT_EQ(e, z);
}

TEST_F(DeclarableOpsTests15, test_rank_2) {
    auto array = NDArrayFactory::create<float>('c', {4, 64});
    auto e = NDArrayFactory::create<int>('c', {}, {2});

    nd4j::ops::rank op;
    auto result = op.execute({&array}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests15, test_lstmBlock_1) {
    auto x0 = NDArrayFactory::create<Nd4jLong>(5);
    auto x1 = NDArrayFactory::create<float>('c', {5, 1, 4}, {0.7787856f, 0.80119777f, 0.72437465f, 0.23089433f, 0.72714126f, 0.18039072f, 0.50563407f, 0.89252293f, 0.5461209f, 0.92336726f, 0.085571885f, 0.7937801f, 0.65908563f, 0.55552566f, 0.15962744f, 0.30874777f, 0.15476847f, 0.46954823f, 0.9938899f, 0.6112741f});
    auto x2 = NDArrayFactory::create<float>('c', {1, 3}, {0.7717289f, 0.9280778f, 0.98455656f});
    auto x3 = NDArrayFactory::create<float>('c', {1, 3}, {0.94414854f, 0.5956861f, 0.8668989f});
    auto x4 = NDArrayFactory::create<float>('c', {7, 12}, {0.460692f, 0.042572856f, 0.08420354f, -0.09538093f, -0.11416581f, -0.53166187f, 0.40133476f, -0.24381405f, 0.30778718f, 0.52713746f, 0.16253126f, -0.034891903f, 0.011679292f, -0.19076681f, 0.14710993f, -0.3704369f, 0.51872355f, 0.13536876f, -0.5568739f, -0.08727971f, 0.07601875f, -0.074174374f, -0.5345982f, -0.3581748f, -0.28263924f, -0.25141674f, 0.43328637f, -0.50227314f, -0.26641843f, -0.38241976f, -0.19636461f, -0.04020852f, -0.27312332f, 0.5207915f, -0.37247592f, -0.4713087f, -0.25670746f, -0.14942765f, -0.015806139f, -0.22531253f, 0.5582536f, 0.3093416f, 0.3221351f, -0.0964683f, 0.14318448f, 0.42279094f, -0.46992f, -0.43399644f, -0.51704615f, -0.11854091f, 0.21697259f, -0.049382925f, 0.14059627f, 0.3912331f, -0.41345632f, 0.5067368f, -0.3420229f, 0.485789f, 0.044918716f, 0.26209074f, 0.12357575f, 0.21778125f, -0.53791714f, 0.18346387f, 0.054183125f, 0.5480431f, 0.03675288f, -0.26656917f, -0.018610716f, 0.19917983f, 0.5566165f, 0.43570566f, -0.35720813f, 0.31097364f, -0.47134516f, -0.289197f, 0.091138184f, 0.13300979f, -0.36592877f, -0.17540845f, 0.21732038f, 0.4393713f, 0.42800313f, 0.5006979f});
    auto x5 = NDArrayFactory::create<float>('c', {1, 3});
    auto x6 = NDArrayFactory::create<float>('c', {1, 3});
    auto x7 = NDArrayFactory::create<float>('c', {1, 3});
    auto x8 = NDArrayFactory::create<float>('c', {12});

    nd4j::ops::lstmBlock op;
    auto result = op.execute({&x0, &x1, &x2, &x3, &x4, &x5, &x6, &x7, &x8}, {2.0, 0.3}, {0, 0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    // z->printIndexedBuffer("Z");

    delete result;
}

TEST_F(DeclarableOpsTests15, test_lstmBlock_2) {
    int seqLen = 8;
    int bS = 16;
    int nIn = 8;

    auto x0 = NDArrayFactory::create<Nd4jLong>(5);
    auto x1 = NDArrayFactory::create<float>('f', {bS, nIn, seqLen});
    auto x2 = NDArrayFactory::create<float>('f', {bS, nIn});    // nIn == nOut
    auto x3 = NDArrayFactory::create<float>('f', {bS, nIn});
    auto x4 = NDArrayFactory::create<float>('f', {2 * nIn, 4 * nIn});
    auto x5 = NDArrayFactory::create<float>('f', {nIn});
    auto x6 = NDArrayFactory::create<float>('f', {nIn});
    auto x7 = NDArrayFactory::create<float>('f', {nIn});
    auto x8 = NDArrayFactory::create<float>('f', {4 * nIn});

    nd4j::ops::lstmBlock op;
    auto result = op.execute({&x0, &x1, &x2, &x3, &x4, &x5, &x6, &x7, &x8}, {1.0, 0.0}, {0, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    delete result;
}

TEST_F(DeclarableOpsTests15, test_lstmBlock_3) {

    int seqLen = 3;
    int bS = 2;
    int nIn = 4;

    NDArray f('f', {bS, nIn, seqLen}, nd4j::DataType::FLOAT32);
    NDArray cLast('f', {bS, nIn}, nd4j::DataType::FLOAT32);

    f = 2;
    cLast = 3;

    for (int t = 0; t < seqLen; ++t) {

        //section 1
        //auto ft = f({0,0, 0,0, t,t+1});
        //auto temp = ft * cLast;


        // section 2
        auto ft = f({0,0, 0,0, t,t+1});
        auto temp1 = ft.reshape('f', {bS, nIn});
        auto temp2 = temp1 * cLast;
    }
}

TEST_F(DeclarableOpsTests15, test_empty_increasing_1) {
    auto x = NDArrayFactory::create<float>('c', {1, 0, 3});
    auto z = NDArrayFactory::create<bool>(false);

    Context ctx(1);
    ctx.setInputArray(0, &x);
    ctx.setOutputArray(0, &z);

    nd4j::ops::is_strictly_increasing op;
    auto status = op.execute(&ctx);
    ASSERT_EQ(Status::OK(), status);

    ASSERT_EQ(true, z.e<bool>(0));
}

TEST_F(DeclarableOpsTests15, test_empty_decreasing_1) {
    auto x = NDArrayFactory::create<float>('c', {1, 0, 3});
    auto z = NDArrayFactory::create<bool>(false);

    Context ctx(1);
    ctx.setInputArray(0, &x);
    ctx.setOutputArray(0, &z);

    nd4j::ops::is_non_decreasing op;
    auto status = op.execute(&ctx);
    ASSERT_EQ(Status::OK(), status);

    ASSERT_EQ(true, z.e<bool>(0));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, test_rgb_to_grs_1) {
    // rank 1
    NDArray rgbs('c', { 3 }, { 10, 50, 200 }, nd4j::DataType::INT32);
    NDArray expected('c', { 1 }, { 55 }, nd4j::DataType::INT32);
    nd4j::ops::rgb_to_grs op;
    auto result = op.execute({&rgbs}, {}, {});
    auto output = result->at(0);

    ASSERT_EQ(Status::OK(), result->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, test_rgb_to_grs_2) {
    // rank 1
    auto rgbs = NDArrayFactory::create<int>('f', { 3 }, { 1, 120, -25 });
    auto expected = NDArrayFactory::create<int>('f', { 1 }, { 67 });
    nd4j::ops::rgb_to_grs op;
    auto result = op.execute({ &rgbs }, {}, {});
    auto output = result->at(0);

    ASSERT_EQ(Status::OK(), result->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, test_rgb_to_grs_3) {
    // rank 2
    NDArray rgbs('c', { 4, 3 }, { -94,  99,  97, 90, 114, 101, 111,  96, 105, 100, 103, 102 }, nd4j::DataType::INT32);
    NDArray expected('c', { 4, 1 }, { 41, 105, 101, 101 }, nd4j::DataType::INT32);
    nd4j::ops::rgb_to_grs op;
    auto result = op.execute({ &rgbs }, {}, {});
    auto output = result->at(0);

    ASSERT_EQ(Status::OK(), result->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, test_rgb_to_grs_4) {

    NDArray rgbs('c', { 3, 2 }, {14,  99, 207, 10, 114, 201 }, nd4j::DataType::INT32);

    rgbs.permutei({1,0});
    NDArray expected('c', { 2, 1 }, { 138, 58 }, nd4j::DataType::INT32);
    nd4j::ops::rgb_to_grs op;
    auto result = op.execute({ &rgbs }, {}, {});
    auto output = result->at(0);

    ASSERT_EQ(Status::OK(), result->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, test_rgb_to_grs_5) {
    // rank 2
    NDArray rgbs('c', { 3, 4 }, { -94,  99,  97, 90, 114, 101, 111,  96, 105, 100, 103, 102 }, nd4j::DataType::INT32);
    NDArray expected('c', { 1, 4 }, { 50, 100, 105, 94 }, nd4j::DataType::INT32);
    nd4j::ops::rgb_to_grs op;
    auto result = op.execute({ &rgbs }, {}, {0});
    auto output = result->at(0);

    ASSERT_EQ(Status::OK(), result->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, test_rgb_to_grs_6) {
    // rank 3
    auto rgbs = NDArrayFactory::create<float>('c', { 5,4,3 }, {1.7750e+01f, -7.1062e+01f, -1.0019e+02f,-2.3406e+01f,  5.2094e+01f,  9.5438e+01f, -6.7461e+00f,  3.8562e+01f,  6.5078e+00f,3.3562e+01f, -5.8844e+01f,  2.2750e+01f, -1.0477e+01f,  7.7344e+00f,  9.5469e+00f,2.1391e+01f, -8.5312e+01f,  7.5830e-01f,2.3125e+01f,  1.8145e+00f,  1.4602e+01f,-4.5859e+00f,  3.9344e+01f,  1.1617e+01f,-8.6562e+01f,  1.0038e+02f,  6.7938e+01f,5.9961e+00f,  6.7812e+01f,  2.9734e+01f,2.9609e+01f, -6.1438e+01f,  1.7750e+01f,6.8562e+01f, -7.4414e+00f,  3.9656e+01f,1.1641e+01f, -2.7516e+01f,  6.7562e+01f,7.8438e+01f,  5.4883e+00f,  2.9438e+01f,-3.1344e+01f,  6.5125e+01f,  1.2695e+01f,4.0531e+01f, -6.1211e+00f,  6.2219e+01f,4.6812e+01f,  5.2250e+01f, -1.1414e+01f,1.5404e-02f,  2.9938e+01f,  5.6719e+00f,-2.0125e+01f,  2.1531e+01f,  6.2500e+01f,7.2188e+01f,  9.3750e+00f, -4.8125e+01f});
    auto expected = NDArrayFactory::create<float>('c', { 5,4,1 }, {-47.82958221f,  34.46305847f,  21.36137581f, -21.91625023f,2.49686432f, -43.59792709f,   9.64180183f,  23.04854202f,40.7946167f,  44.98754883f, -25.19047546f,  20.64586449f,-4.97033119f,   30.0226841f,  30.30688286f,  15.61459541f,43.36166f,  18.22480774f,  13.74833488f,  21.59387016f});

    nd4j::ops::rgb_to_grs op;
    auto result = op.execute({ &rgbs }, {}, {});
    auto output = result->at(0);

    ASSERT_EQ(Status::OK(), result->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, test_rgb_to_grs_7) {
    // rank 3
    auto rgbs = NDArrayFactory::create<float>('c', { 5,3,4 }, { 1.7750e+01f, -7.1062e+01f, -1.0019e+02f,-2.3406e+01f,  5.2094e+01f,  9.5438e+01f, -6.7461e+00f,  3.8562e+01f,  6.5078e+00f,3.3562e+01f, -5.8844e+01f,  2.2750e+01f, -1.0477e+01f,  7.7344e+00f,  9.5469e+00f,2.1391e+01f, -8.5312e+01f,  7.5830e-01f,2.3125e+01f,  1.8145e+00f,  1.4602e+01f,-4.5859e+00f,  3.9344e+01f,  1.1617e+01f,-8.6562e+01f,  1.0038e+02f,  6.7938e+01f,5.9961e+00f,  6.7812e+01f,  2.9734e+01f,2.9609e+01f, -6.1438e+01f,  1.7750e+01f,6.8562e+01f, -7.4414e+00f,  3.9656e+01f,1.1641e+01f, -2.7516e+01f,  6.7562e+01f,7.8438e+01f,  5.4883e+00f,  2.9438e+01f,-3.1344e+01f,  6.5125e+01f,  1.2695e+01f,4.0531e+01f, -6.1211e+00f,  6.2219e+01f,4.6812e+01f,  5.2250e+01f, -1.1414e+01f,1.5404e-02f,  2.9938e+01f,  5.6719e+00f,-2.0125e+01f,  2.1531e+01f,  6.2500e+01f,7.2188e+01f,  9.3750e+00f, -4.8125e+01f});
    auto expected = NDArrayFactory::create<float>('c', { 5,1,4 }, { 36.626545, 38.607746, -40.614971, 18.233341, -51.545094,2.234142, 20.913160, 8.783220, 15.955761, 55.273506, 36.838833, -29.751089, 8.148357, 13.676106, 1.097548, 68.766457, 38.690712, 27.176361, -14.156269, 7.157052  });

    nd4j::ops::rgb_to_grs op;
    auto result = op.execute({ &rgbs }, {}, {1});
    auto output = result->at(0);

    ASSERT_EQ(Status::OK(), result->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, test_rgb_to_grs_8) {
    // rank 3
    auto rgbs = NDArrayFactory::create<float>('c', { 3,5,4 }, {1.7750e+01f, -7.1062e+01f, -1.0019e+02f,-2.3406e+01f,  5.2094e+01f,  9.5438e+01f, -6.7461e+00f,  3.8562e+01f,  6.5078e+00f,3.3562e+01f, -5.8844e+01f,  2.2750e+01f, -1.0477e+01f,  7.7344e+00f,  9.5469e+00f,2.1391e+01f, -8.5312e+01f,  7.5830e-01f,2.3125e+01f,  1.8145e+00f,  1.4602e+01f,-4.5859e+00f,  3.9344e+01f,  1.1617e+01f,-8.6562e+01f,  1.0038e+02f,  6.7938e+01f,5.9961e+00f,  6.7812e+01f,  2.9734e+01f,2.9609e+01f, -6.1438e+01f,  1.7750e+01f,6.8562e+01f, -7.4414e+00f,  3.9656e+01f,1.1641e+01f, -2.7516e+01f,  6.7562e+01f,7.8438e+01f,  5.4883e+00f,  2.9438e+01f,-3.1344e+01f,  6.5125e+01f,  1.2695e+01f,4.0531e+01f, -6.1211e+00f,  6.2219e+01f,4.6812e+01f,  5.2250e+01f, -1.1414e+01f,1.5404e-02f,  2.9938e+01f,  5.6719e+00f,-2.0125e+01f,  2.1531e+01f,  6.2500e+01f,7.2188e+01f,  9.3750e+00f, -4.8125e+01f});
    try {
        nd4j::ops::rgb_to_grs op;
        auto result = op.execute({ &rgbs }, {}, {});
        ASSERT_EQ(Status::THROW(), result->status());
        delete result;
    } catch (std::exception& e) {
        nd4j_printf("Error should be here `%s'. It's OK.\n", e.what());
    }
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, test_rgb_to_grs_9) {
    // rank 3
    auto rgbs = NDArrayFactory::create<float>('f', { 2, 2, 3 }, { 1.7750e+01f,-7.1062e+01f, -1.0019e+02f, -2.3406e+01f,5.2094e+01f,9.5438e+01f, -6.7461e+00f,3.8562e+01f, 6.5078e+00f,      3.3562e+01f,-5.8844e+01f,2.2750e+01f});
    auto expected = NDArrayFactory::create<float>('f', { 2,2,1 }, { 36.626545f, 38.607746f, -40.614971f, 18.233341f });

    nd4j::ops::rgb_to_grs op;
    auto result = op.execute({ &rgbs }, {}, {});
    auto output = result->at(0);

    ASSERT_EQ(Status::OK(), result->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete result;
}