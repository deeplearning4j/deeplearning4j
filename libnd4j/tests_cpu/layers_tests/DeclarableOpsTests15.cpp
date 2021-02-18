/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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
#include <array/NDArray.h>
#include <ops/ops.h>
#include <helpers/GradCheck.h>
#include <array>


using namespace sd;


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

    sd::ops::normalize_moments op;
    auto result = op.execute({&w, &x, &y}, std::vector<NDArray*>{&z0, &z1}, {1e-4}, {}, {});
    ASSERT_EQ(Status::OK(), result);
}

TEST_F(DeclarableOpsTests15, Test_Add_1) {
    auto x = NDArrayFactory::create<int>('c', {5}, {1, 1, 1, 1, 1});
    auto y = NDArrayFactory::create<int>('c', {5}, {1, 1, 1, 1, 1});
    auto e = NDArrayFactory::create<int>('c', {5}, {2, 2, 2, 2, 2});

    sd::ops::add op;
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

TEST_F(DeclarableOpsTests15, Test_standarize_1) {
    auto x = NDArrayFactory::create<float>('c', {5}, {1.f, 1.f, 1.f, 1.f, 1.f});
    auto e = NDArrayFactory::create<float>('c', {5}, {0.f, 0.f, 0.f, 0.f, 0.f});

    sd::ops::standardize op;
    auto result = op.execute({&x}, {&x}, {}, {0}, {});
    ASSERT_EQ(Status::OK(), result);
    ASSERT_EQ(e, x);
}

TEST_F(DeclarableOpsTests15, Test_standarize_bp_1) {
    auto x = NDArrayFactory::create<float>('c', {5}, {1.f, 1.f, 1.f, 1.f, 1.f});
    auto eps = NDArrayFactory::create<float>('c', {5}, {0.f, 0.f, 0.f, 0.f, 0.f});

    sd::ops::standardize_bp op;
    auto result = op.evaluate({&x, &eps}, {0});
    ASSERT_EQ(Status::OK(), result.status());

}

TEST_F(DeclarableOpsTests15, Test_AdjustContrast_1) {
    auto x = NDArrayFactory::create<double>('c', {4,4,3});
    NDArray factor = NDArrayFactory::create<double>(2.);
    auto e = NDArrayFactory::create<double>('c', {4,4,3}, {-21.5, -20.5, -19.5,  -15.5, -14.5, -13.5,  -9.5,  -8.5,  -7.5,  -3.5,  -2.5,  -1.5,
                                     2.5,   3.5,   4.5,    8.5,   9.5,  10.5,  14.5,  15.5,  16.5,  20.5,  21.5,  22.5,
                                    26.5,  27.5,  28.5,   32.5,  33.5,  34.5,  38.5,  39.5,  40.5,  44.5,  45.5,  46.5,
                                    50.5,  51.5,  52.5,   56.5,  57.5,  58.5,  62.5,  63.5,  64.5,  68.5,  69.5,  70.5});


    x.linspace(1.);
    sd::ops::adjust_contrast op;
    auto result = op.evaluate({&x, &factor}, {}, {}, {});
    ASSERT_EQ(Status::OK(), result.status());
    auto out = result.at(0);

    ASSERT_TRUE(e.equalsTo(out));

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
    sd::ops::adjust_contrast op;
    auto result = op.evaluate({&x}, {2.});
    ASSERT_EQ(Status::OK(), result.status());
    auto out = result.at(0);
//    out->printIndexedBuffer("Adjusted Constrast");
    ASSERT_TRUE(e.equalsTo(out));

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
    sd::ops::adjust_contrast_v2 op;
    auto result = op.evaluate({&x}, {2.});
    ASSERT_EQ(Status::OK(), result.status());
    auto out = result.at(0);
//    out->printIndexedBuffer("Adjusted Constrast");
    ASSERT_TRUE(e.equalsTo(out));

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
    sd::ops::adjust_contrast_v2 op;
    auto result = op.evaluate({&x}, {2.}, {}, {});
    ASSERT_EQ(Status::OK(), result.status());
    auto out = result.at(0);
//    out->printIndexedBuffer("Adjusted Constrast");
    ASSERT_TRUE(e.equalsTo(out));

}

TEST_F(DeclarableOpsTests15, Test_AdjustContrast_5) {
    auto x = NDArrayFactory::create<double>('c', {1, 3, 4});
    auto e = NDArrayFactory::create<double>('c', {1, 3, 4}, {
        -3., -2., -1.,  0.,      5.,  6.,  7.,  8.,     13., 14., 15., 16.
    });
    x.linspace(1.);
    sd::ops::adjust_contrast_v2 op;
    auto result = op.evaluate({&x}, {2.}, {}, {});
    ASSERT_EQ(Status::OK(), result.status());
    auto out = result.at(0);
//    out->printIndexedBuffer("Adjusted Constrast");
    ASSERT_TRUE(e.equalsTo(out));

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
              1.0218375f,             1.0666375f,            0.9130375f,
            -0.07396251f,            0.91843754f,          -0.17496246f,
             0.47543746f,             1.2492375f,           0.55643755f,
              1.3110375f,           -0.36456245f,            1.0518374f,
              0.7824375f,            0.57523745f,          -0.21656245f,
              0.0816375f,            -0.2261625f,           0.40323752f,
              1.4520376f,             0.6868375f,           0.81723756f,
            -0.17576247f,            0.81423753f,          -0.08656245f,

           -0.36249164f,             0.45590833f,            1.1925083f,
            0.00650835f,              1.4861084f,            1.2079083f,
            0.05270836f,             0.37350836f,           0.94130826f,
             1.0715083f,              0.6103083f,            0.9825083f,
            0.07370833f,             -0.4518917f,          -0.39889166f,
            -0.3354917f,              1.2213084f,            1.0345083f,
            -0.3132917f,             0.78470826f,           0.23390833f,
             0.6943083f,             0.68170834f,          -0.09989169f,

             0.8352709f,              1.3798709f,           0.15507084f,
            0.26607084f,            -0.10792917f,            1.2302709f,
             0.6448709f,            -0.29992914f,            1.3534708f,
            0.86607087f,             0.37607086f,           0.04027084f,
            0.40087086f,             0.59507084f,            0.9416709f,
            0.53127086f,            -0.01712915f,            1.4610709f,
           -0.17152917f,            -0.13992918f,            0.6242708f,
           -0.42192918f,             0.38387084f,          -0.15752912f,

             0.3311833f,            0.00618333f,            0.17538333f,
            0.10418332f,             0.8365834f,            0.27098334f,
             1.2421833f,            -0.1114167f,             1.0153834f,
             0.9523833f,             0.8317833f,             0.9633833f,
             0.6501833f,            0.04258335f,             0.9999833f,
           -0.40181667f,            0.11418331f,            0.47938335f,
             1.1057833f,           -0.29761666f,             1.0779834f,
             0.5243833f,           -0.32181668f,             1.1833833f,

            0.73157084f,             0.4317708f,             0.7283708f,
             1.2297708f,             0.4307708f,            0.85377085f,
            0.05977082f,           -0.09282917f,            0.33957082f,
             1.0751709f,             0.2119708f,            0.51897085f,
           -0.25302917f,             1.1723708f,           -0.12562919f,
             1.1993709f,             0.5257708f,            0.40517086f,
            0.53197086f,             0.8441708f,            0.02617085f,
            -0.0208292f,             0.8711709f,            0.04137081f,

            0.74936247f,             0.6085625f,             0.8997625f,
           -0.08743751f,            0.18576252f,           -0.17563748f,
             0.5991625f,            -0.0038375f,            0.07576251f,
            0.42536253f,           -0.22823751f,            0.36296248f,
            0.81456256f,           -0.16183749f,             0.5161625f,
           -0.21183747f,             0.7429625f,             0.6217625f,
            0.17656249f,            0.02616251f,           -0.17923748f,
             1.4659625f,            0.40016252f,            0.28356248f,

             0.4195791f,             0.8745791f,            0.36637908f,
            0.50597906f,           -0.17942089f,            0.16917908f,
             1.0235791f,             1.3699791f,           -0.11382091f,
            -0.0918209f,             0.7757791f,            0.09017909f,
             1.3807791f,           -0.15202093f,             1.3875791f,
            -0.1712209f,             1.3989791f,            0.43777913f,
             0.7855791f,             0.1423791f,             1.4711791f,
             0.6455791f,             0.6211791f,           -0.48062086f,

            0.10189578f,             0.5628958f,            0.68909574f,
            0.96649575f,           -0.09370419f,             1.3466958f,
             1.4584957f,             1.3544958f,            -0.3829042f,
            0.11269578f,           -0.47890422f,             1.0436958f,
             0.6128957f,            0.27209583f,             0.2714958f,
            0.21889582f,            0.08789578f,             1.1296958f,
             0.4596958f,            0.39309582f,             0.8344958f,
            0.71149576f,            -0.4799042f,             0.4880958f
    });

    sd::ops::adjust_contrast op;
    auto result = op.evaluate({&x}, {2.}, {}, {});
    ASSERT_EQ(Status::OK(), result.status());
    auto out = result.at(0);
//    out->printBuffer("Adjusted Constrast6");
//    e.printBuffer("Adjusted Expected 6");
//    ASSERT_TRUE(e.equalsTo(out));

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
              1.0218375,             1.0666375 ,             0.9130375 ,
            -0.07396251,             0.91843754,            -0.17496246,
             0.47543746,             1.2492375 ,             0.55643755,
             1.3110375 ,            -0.36456245,             1.0518374 ,
             0.7824375 ,             0.57523745,            -0.21656245,
             0.0816375 ,            -0.2261625 ,             0.40323752,
             1.4520376 ,             0.6868375 ,             0.81723756,
            -0.17576247,             0.81423753,            -0.08656245,

            -0.36249164,             0.45590833,             1.1925083 ,
             0.00650835,             1.4861084 ,             1.2079083 ,
             0.05270836,             0.37350836,             0.94130826,
             1.0715083 ,             0.6103083 ,             0.9825083 ,
             0.07370833,            -0.4518917 ,            -0.39889166,
            -0.3354917 ,             1.2213084 ,             1.0345083 ,
            -0.3132917 ,             0.78470826,             0.23390833,
             0.6943083 ,             0.68170834,            -0.09989169,

             0.8352709 ,             1.3798709 ,             0.15507084,
             0.26607084,            -0.10792917,             1.2302709 ,
             0.6448709 ,            -0.29992914,             1.3534708 ,
             0.86607087,             0.37607086,             0.04027084,
             0.40087086,             0.59507084,             0.9416709 ,
             0.53127086,            -0.01712915,             1.4610709 ,
            -0.17152917,            -0.13992918,             0.6242708 ,
            -0.42192918,             0.38387084,            -0.15752912,


             0.3311833 ,             0.00618333,             0.17538333,
             0.10418332,             0.8365834 ,             0.27098334,
             1.2421833 ,            -0.1114167 ,             1.0153834 ,
             0.9523833 ,             0.8317833 ,             0.9633833 ,
             0.6501833 ,             0.04258335,             0.9999833 ,
            -0.40181667,             0.11418331,             0.47938335,
             1.1057833 ,            -0.29761666,             1.0779834 ,
             0.5243833 ,            -0.32181668,             1.1833833 ,

             0.73157084,             0.4317708 ,             0.7283708 ,
             1.2297708 ,             0.4307708 ,             0.85377085,
             0.05977082,            -0.09282917,             0.33957082,
             1.0751709 ,             0.2119708 ,             0.51897085,
            -0.25302917,             1.1723708 ,            -0.12562919,
             1.1993709 ,             0.5257708 ,             0.40517086,
             0.53197086,             0.8441708 ,             0.02617085,
            -0.0208292 ,             0.8711709 ,             0.04137081,

             0.74936247,             0.6085625 ,             0.8997625 ,
            -0.08743751,             0.18576252,            -0.17563748,
             0.5991625 ,            -0.0038375 ,             0.07576251,
             0.42536253,            -0.22823751,             0.36296248,
             0.81456256,            -0.16183749,             0.5161625 ,
            -0.21183747,             0.7429625 ,             0.6217625 ,
             0.17656249,             0.02616251,            -0.17923748,
             1.4659625 ,             0.40016252,             0.28356248,

             0.4195791 ,             0.8745791 ,             0.36637908,
             0.50597906,            -0.17942089,             0.16917908,
             1.0235791 ,             1.3699791 ,            -0.11382091,
            -0.0918209 ,             0.7757791 ,             0.09017909,
             1.3807791 ,            -0.15202093,             1.3875791 ,
            -0.1712209 ,             1.3989791 ,             0.43777913,
             0.7855791 ,             0.1423791 ,             1.4711791 ,
             0.6455791 ,             0.6211791 ,            -0.48062086,


             0.10189578,             0.5628958 ,             0.68909574,
             0.96649575,            -0.09370419,             1.3466958 ,
             1.4584957 ,             1.3544958 ,            -0.3829042 ,
             0.11269578,            -0.47890422,             1.0436958 ,
             0.6128957 ,             0.27209583,             0.2714958 ,
             0.21889582,             0.08789578,             1.1296958 ,
             0.4596958 ,             0.39309582,             0.8344958 ,
             0.71149576,            -0.4799042,             0.4880958
    });
//    x.linspace(1.);
    sd::ops::adjust_contrast_v2 op;
    auto result = op.evaluate({&x}, {2.}, {}, {});
    ASSERT_EQ(Status::OK(), result.status());
    auto out = result.at(0);
//    out->printBuffer("Adjusted Constrast7");
//    e.printBuffer("Adjusted expected 7");
    auto diff = e - *out;
//    diff.printBuffer("Adjusted subtract 7");
    ASSERT_TRUE(e.equalsTo(out));

}

TEST_F(DeclarableOpsTests15, Test_BitCast_1) {
    auto x = NDArrayFactory::create<float>('c', {2, 2, 2});
    auto e = NDArrayFactory::create<double>('c', {2, 2}, {2., 512., 8192., 131072.032 });
    x.linspace(1.);

    sd::ops::bitcast op;
    auto result = op.evaluate({&x}, {(int) sd::DataType::DOUBLE});
    ASSERT_EQ(Status::OK(), result.status());
    auto out = result.at(0);
//    out->printIndexedBuffer("Casted result");
    ASSERT_TRUE(e.equalsTo(out));

}

TEST_F(DeclarableOpsTests15, Test_BitCast_2) {
    auto x = NDArrayFactory::create<float>('c', {2, 4});
    auto e = NDArrayFactory::create<float16>('c', {2, 4, 2}, {0.f, 1.875f, 0.f, 2.f,    0.f, 2.125f, 0.f,  2.25f,
                                                                              0.f, 2.312f, 0.f, 2.375f, 0.f, 2.438f, 0.f, 2.5f});
    x.linspace(1.);

    sd::ops::bitcast op;
    auto result = op.evaluate({&x}, {(int) sd::DataType::HALF});
    ASSERT_EQ(Status::OK(), result.status());
    auto out = result.at(0);

    ASSERT_TRUE(e.equalsTo(out));

}

TEST_F(DeclarableOpsTests15, Test_BitCast_3) {
    auto x = NDArrayFactory::create<float>('c', {1, 4});

    x.linspace(1.);
    sd::ops::bitcast op;
    try {
        auto result = op.evaluate({&x}, {(int) sd::DataType::INT64});
        ASSERT_NE(Status::OK(), result.status());

    } catch (std::exception& e) {
        nd4j_printf("Error should be here `%s'. It's OK.\n", e.what());
    }
}

TEST_F(DeclarableOpsTests15, Test_BitCast_4) {
    auto x = NDArrayFactory::create<float>('c', {1, 4});
    auto e = NDArrayFactory::create<Nd4jLong>('c', {1, 2}, {1234567890LL, 2468013579LL});
    x.linspace(1.);
    sd::ops::bitcast op;
    try {
        auto result = op.execute({&x}, {&e}, {}, {sd::DataType::INT64}, {});
        ASSERT_NE(Status::OK(), result);
    } catch(std::exception& e) {
        nd4j_printf("Error `%s' should be here. It's OK.\n",e.what());
    }

}

TEST_F(DeclarableOpsTests15, Test_BitCast_4_1) {
    auto x = NDArrayFactory::create<double>('c', {1, 2});
    auto e = NDArrayFactory::create<Nd4jLong>('c', {1, 2}, {4607182418800017408LL, 4611686018427387904LL}); // as TF 4607182418800017408, 4611686018427387904
    x.linspace(1.);
    sd::ops::bitcast op;

    auto result = op.evaluate({&x}, {}, {sd::DataType::INT64}, {});
    ASSERT_EQ(Status::OK(), result.status());

    //    e.printIndexedBuffer("Double to int64");
    auto res = result.at(0);
    ASSERT_EQ(*res, e);

}


TEST_F(DeclarableOpsTests15, Test_BitCast_5) {
    auto x = NDArrayFactory::create<float16>('c', {4, 4}, {
        0.4922f,    0.2969f,    0.6172f,    0.8906f,
        0.9297f,    0.0859f,    0.2344f,    0.3828f,
        0.5781f,    0.7969f,    0.0391f,    0.1719f,
        0.8359f,    0.9297f,    0.3438f,    0.0938f});

    auto e = NDArrayFactory::create<Nd4jLong>('c', {4}, {4260467851820808160LL, 3900173902914993008LL, 3566895990128523424LL,
                                                         3314989625590692528LL});

    sd::ops::bitcast op;
    auto result = op.evaluate({&x}, {}, {sd::DataType::INT64}, {});
    ASSERT_EQ(Status::OK(), result.status());
    auto res = result.at(0);

//    res->printIndexedBuffer("BITCAST5");
    ASSERT_TRUE(e.equalsTo(res));

}

TEST_F(DeclarableOpsTests15, Test_BitCast_6) {
    auto x = NDArrayFactory::create<float16>('c', {4, 4}, {
            1.f,    2.f,    3.f,    4.f,
            5.f,    6.f,    7.f,    8.f,
            9.f,   10.f,   11.f,   12.f,
           13.f,   14.f,   15.f,   16.f});

    auto e = NDArrayFactory::create<Nd4jLong>('c', {4}, {4899988963420290048LL, 5188224837230806272LL, 5332342774136064128LL,
                                                         5476460161268730496LL});

    sd::ops::bitcast op;
    auto result = op.evaluate({&x}, {}, {sd::DataType::INT64}, {});
    ASSERT_EQ(Status::OK(), result.status());
    auto res = result.at(0);

//    res->printIndexedBuffer("BITCAST6");
    ASSERT_TRUE(e.equalsTo(res));

}
TEST_F(DeclarableOpsTests15, Test_BitCast_7) {
    auto x = NDArrayFactory::create<float16>('c', {4, 4}, {
            1.1f,    2.2f,    3.3f,    4.4f,
            5.1f,    6.2f,    7.3f,    8.4f,
            9.1f,   10.2f,   11.3f,   12.4f,
            13.f,   14.2f,   15.3f,   16.4f});

    auto e = NDArrayFactory::create<Nd4jLong>('c', {4}, {
        4928700072476425318LL, 5202580391758873882LL, 5346698272827918477LL,  5483778673873668736LL});

    sd::ops::bitcast op;
    auto result = op.evaluate({&x}, {}, {sd::DataType::INT64}, {});
    ASSERT_EQ(Status::OK(), result.status());
    auto res = result.at(0);

//    res->printIndexedBuffer("BITCAST7");
    ASSERT_TRUE(e.equalsTo(res));

}

TEST_F(DeclarableOpsTests15, test_matmul_bp_1) {
    auto a = NDArrayFactory::create<double>('c', {1, 3});
    auto b = NDArrayFactory::create<double>('c', {1, 4});
    auto gI = NDArrayFactory::create<double>('c', {3, 4});

    auto gA = NDArrayFactory::create<double>('c', {1, 3});
    auto gB = NDArrayFactory::create<double>('c', {1, 4});

    sd::ops::matmul_bp op;
    auto status = op.execute({&a, &b, &gI}, std::vector<NDArray*>{&gA, &gB}, {}, {1, 0, 0}, {});
    ASSERT_EQ(Status::OK(), status);
}

TEST_F(DeclarableOpsTests15, test_non_decreasing_1) {
    auto x = NDArrayFactory::create<double>(1.0);
    auto z = NDArrayFactory::create<bool>(false);
    auto e = NDArrayFactory::create<bool>(true);

    sd::ops::is_non_decreasing op;
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

    sd::ops::check_numerics op;
    auto result = op.evaluate({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result.status());

    auto z = result.at(0);

    ASSERT_EQ(x, *z);
}

TEST_F(DeclarableOpsTests15, test_check_numeric_2) {
#ifdef FFAST_MATH
    if (1 > 0)
        return;
#endif

    auto x = NDArrayFactory::create<float>('c', {3},{1.f, 2.f, std::numeric_limits<float>::infinity()});
    auto y = NDArrayFactory::string("should trigger");
    auto z = NDArrayFactory::create<float>('c', {3} );

    sd::ops::check_numerics op;
    try {
        auto status = op.execute({&x, &y}, {&z}, {}, {}, {});
        ASSERT_TRUE(false);
    } catch (std::invalid_argument &e) {
        //
    }
}

TEST_F(DeclarableOpsTests15, test_check_numeric_3) {
#ifdef FFAST_MATH
    if (1 > 0)
        return;
#endif

    auto x = NDArrayFactory::create<float>('c', {3},{1.f, 2.f, std::numeric_limits<float>::quiet_NaN()});
    auto y = NDArrayFactory::string("should trigger");
    auto z = NDArrayFactory::create<float>('c', {3} );

    sd::ops::check_numerics op;
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

    sd::ops::layer_norm op;
    auto result = op.evaluate({&x, &g, &b}, {}, {0}, {false});
    ASSERT_EQ(Status::OK(), result.status());

}

TEST_F(DeclarableOpsTests15, Test_layer_norm_bp_1) {
    auto x = NDArrayFactory::create<float>('c', {1, 5}, {1.f, 2.f, 3.f, 4.f, 5.f});
    auto g = NDArrayFactory::create<float>('c', {5}, {1.f, 2.f, 3.f, 4.f, 5.f});
    auto b = NDArrayFactory::create<float>('c', {5}, {1.f, 2.f, 3.f, 4.f, 5.f});
    auto eps = NDArrayFactory::create<float>('c', {1, 5}, {0.f, 0.f, 0.f, 0.f, 0.f});

    sd::ops::layer_norm_bp op;
    auto result = op.evaluate({&x, &g, &b, &eps}, {}, {0}, {false});
    ASSERT_EQ(Status::OK(), result.status());

}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, Test_layer_norm_bp_2) {

    NDArray x('c', {3, 4, 8, 8}, sd::DataType::FLOAT32);
    NDArray gain('c', {4}, {-0.1, 0.1, -0.2, 0.2}, sd::DataType::FLOAT32);
    NDArray bias('c', {4}, {-0.05, 0.05, -1.05, 1.05}, sd::DataType::FLOAT32);
    NDArray gradO('c', {3, 4, 8, 8}, sd::DataType::FLOAT32);

    NDArray gradI('c', {3, 4, 8, 8}, sd::DataType::FLOAT32);
    NDArray gradG('c', {4}, sd::DataType::FLOAT32);
    NDArray gradB('c', {4}, sd::DataType::FLOAT32);

    x.linspace(-20, 0.5);
    gradO.linspace(-4, 0.05);

    sd::ops::layer_norm_bp op;
    auto status = op.execute({&x, &gain, &bias, &gradO}, {&gradI, &gradG, &gradB}, {}, {1,2,3}, {true});
    ASSERT_EQ(Status::OK(), status);
}

TEST_F(DeclarableOpsTests15, test_hashCode_1) {
    auto x = NDArrayFactory::create<int>('c', {10});
    auto y = NDArrayFactory::create<int>('c', {10});

    x.linspace(1.);
    y.linspace(2.);

    sd::ops::hashcode op;
    auto resultA0 = op.evaluate({&x});
    auto resultA1 = op.evaluate({&x});
    auto resultB0 = op.evaluate({&y});
//    resultA0->at(0)->printIndexedBuffer("A0");
//    resultA1->at(0)->printIndexedBuffer("A1");
//    resultB0->at(0)->printIndexedBuffer("B0");
    ASSERT_EQ(*resultA0.at(0), *resultA1.at(0));
    ASSERT_NE(*resultA0.at(0), *resultB0.at(0));
}

TEST_F(DeclarableOpsTests15, test_hashCode_2) {
    auto x = NDArrayFactory::create<int>('c', {1027});
    auto y = NDArrayFactory::create<int>('c', {1027});

    x.linspace(1.);
    y.linspace(2.);

    sd::ops::hashcode op;
    auto resultA0 = op.evaluate({&x});
    auto resultA1 = op.evaluate({&x});
    auto resultB0 = op.evaluate({&y});

//    resultA0->at(0)->printIndexedBuffer("A0");
//    resultA1->at(0)->printIndexedBuffer("A1");
//    resultB0->at(0)->printIndexedBuffer("B0");

    ASSERT_EQ(*resultA0.at(0), *resultA1.at(0));
    ASSERT_NE(*resultA0.at(0), *resultB0.at(0));
}

TEST_F(DeclarableOpsTests15, test_rank_1) {
    auto array = NDArrayFactory::create<float>('c', {4, 64});
    auto e = NDArrayFactory::create<int>('c', {}, {2});
    auto z = NDArrayFactory::create<int>('c', {});

    sd::ops::rank op;
    auto result = op.execute({&array}, {&z}, {}, {}, {});
    ASSERT_EQ(Status::OK(), result);
    ASSERT_EQ(e, z);
}

TEST_F(DeclarableOpsTests15, test_rank_2) {
    auto array = NDArrayFactory::create<float>('c', {4, 64});
    auto e = NDArrayFactory::create<int>('c', {}, {2});

    sd::ops::rank op;
    auto result = op.evaluate({&array}, {}, {});
    ASSERT_EQ(Status::OK(), result.status());

    auto z = result.at(0);

    ASSERT_EQ(e, *z);


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

    sd::ops::lstmBlock op;
    auto result = op.evaluate({&x0, &x1, &x2, &x3, &x4, &x5, &x6, &x7, &x8}, {2.0, 0.3}, {0, 0});
    ASSERT_EQ(Status::OK(), result.status());

    auto z = result.at(0);

    // z->printIndexedBuffer("Z");
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

    sd::ops::lstmBlock op;
    auto result = op.evaluate({&x0, &x1, &x2, &x3, &x4, &x5, &x6, &x7, &x8}, {1.0, 0.0}, {0, 1});
    ASSERT_EQ(Status::OK(), result.status());

    auto z = result.at(0);

}

TEST_F(DeclarableOpsTests15, test_lstmBlock_3) {

    int seqLen = 3;
    int bS = 2;
    int nIn = 4;

    NDArray f('f', {bS, nIn, seqLen}, sd::DataType::FLOAT32);
    NDArray cLast('f', {bS, nIn}, sd::DataType::FLOAT32);

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

    sd::ops::is_strictly_increasing op;
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

    sd::ops::is_non_decreasing op;
    auto status = op.execute(&ctx);
    ASSERT_EQ(Status::OK(), status);

    ASSERT_EQ(true, z.e<bool>(0));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, test_rgb_to_grs_1) {
    // rank 1
    NDArray rgbs('c', { 3 }, { 10, 50, 200 }, sd::DataType::INT32);
    NDArray expected('c', { 1 }, std::vector<double>{ 55 }, sd::DataType::INT32);
    sd::ops::rgb_to_grs op;
    auto result = op.evaluate({&rgbs}, {}, {});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, test_rgb_to_grs_2) {
    // rank 1
    auto rgbs = NDArrayFactory::create<int>('f', { 3 }, { 1, 120, -25 });
    auto expected = NDArrayFactory::create<int>('f', { 1 }, { 67 });
    sd::ops::rgb_to_grs op;
    auto result = op.evaluate({ &rgbs }, {}, {});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, test_rgb_to_grs_3) {
    // rank 2
    NDArray rgbs('c', { 4, 3 }, { -94,  99,  97, 90, 114, 101, 111,  96, 105, 100, 103, 102 }, sd::DataType::INT32);
    NDArray expected('c', { 4, 1 }, { 41, 105, 101, 101 }, sd::DataType::INT32);
    sd::ops::rgb_to_grs op;
    auto result = op.evaluate({ &rgbs }, {}, {});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, test_rgb_to_grs_4) {

    NDArray rgbs('c', { 3, 2 }, {14,  99, 207, 10, 114, 201 }, sd::DataType::INT32);

    rgbs.permutei({1,0});
    NDArray expected('c', { 2, 1 }, { 138, 58 }, sd::DataType::INT32);
    sd::ops::rgb_to_grs op;
    auto result = op.evaluate({ &rgbs }, {}, {});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, test_rgb_to_grs_5) {
    // rank 2
    NDArray rgbs('c', { 3, 4 }, { -94,  99,  97, 90, 114, 101, 111,  96, 105, 100, 103, 102 }, sd::DataType::INT32);
    NDArray expected('c', { 1, 4 }, { 50, 100, 105, 94 }, sd::DataType::INT32);
    sd::ops::rgb_to_grs op;
    auto result = op.evaluate({ &rgbs }, {}, {0});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, test_rgb_to_grs_6) {
    // rank 3
    auto rgbs = NDArrayFactory::create<float>('c', { 5,4,3 }, {1.7750e+01f, -7.1062e+01f, -1.0019e+02f,-2.3406e+01f,  5.2094e+01f,  9.5438e+01f, -6.7461e+00f,  3.8562e+01f,  6.5078e+00f,3.3562e+01f, -5.8844e+01f,  2.2750e+01f, -1.0477e+01f,  7.7344e+00f,  9.5469e+00f,2.1391e+01f, -8.5312e+01f,  7.5830e-01f,2.3125e+01f,  1.8145e+00f,  1.4602e+01f,-4.5859e+00f,  3.9344e+01f,  1.1617e+01f,-8.6562e+01f,  1.0038e+02f,  6.7938e+01f,5.9961e+00f,  6.7812e+01f,  2.9734e+01f,2.9609e+01f, -6.1438e+01f,  1.7750e+01f,6.8562e+01f, -7.4414e+00f,  3.9656e+01f,1.1641e+01f, -2.7516e+01f,  6.7562e+01f,7.8438e+01f,  5.4883e+00f,  2.9438e+01f,-3.1344e+01f,  6.5125e+01f,  1.2695e+01f,4.0531e+01f, -6.1211e+00f,  6.2219e+01f,4.6812e+01f,  5.2250e+01f, -1.1414e+01f,1.5404e-02f,  2.9938e+01f,  5.6719e+00f,-2.0125e+01f,  2.1531e+01f,  6.2500e+01f,7.2188e+01f,  9.3750e+00f, -4.8125e+01f});
    auto expected = NDArrayFactory::create<float>('c', { 5,4,1 }, {-47.82958221f,  34.46305847f,  21.36137581f, -21.91625023f,2.49686432f, -43.59792709f,   9.64180183f,  23.04854202f,40.7946167f,  44.98754883f, -25.19047546f,  20.64586449f,-4.97033119f,   30.0226841f,  30.30688286f,  15.61459541f,43.36166f,  18.22480774f,  13.74833488f,  21.59387016f});

    sd::ops::rgb_to_grs op;
    auto result = op.evaluate({ &rgbs }, {}, {});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, test_rgb_to_grs_7) {
    // rank 3
    auto rgbs = NDArrayFactory::create<float>('c', { 5,3,4 }, { 1.7750e+01f, -7.1062e+01f, -1.0019e+02f,-2.3406e+01f,  5.2094e+01f,  9.5438e+01f, -6.7461e+00f,  3.8562e+01f,  6.5078e+00f,3.3562e+01f, -5.8844e+01f,  2.2750e+01f, -1.0477e+01f,  7.7344e+00f,  9.5469e+00f,2.1391e+01f, -8.5312e+01f,  7.5830e-01f,2.3125e+01f,  1.8145e+00f,  1.4602e+01f,-4.5859e+00f,  3.9344e+01f,  1.1617e+01f,-8.6562e+01f,  1.0038e+02f,  6.7938e+01f,5.9961e+00f,  6.7812e+01f,  2.9734e+01f,2.9609e+01f, -6.1438e+01f,  1.7750e+01f,6.8562e+01f, -7.4414e+00f,  3.9656e+01f,1.1641e+01f, -2.7516e+01f,  6.7562e+01f,7.8438e+01f,  5.4883e+00f,  2.9438e+01f,-3.1344e+01f,  6.5125e+01f,  1.2695e+01f,4.0531e+01f, -6.1211e+00f,  6.2219e+01f,4.6812e+01f,  5.2250e+01f, -1.1414e+01f,1.5404e-02f,  2.9938e+01f,  5.6719e+00f,-2.0125e+01f,  2.1531e+01f,  6.2500e+01f,7.2188e+01f,  9.3750e+00f, -4.8125e+01f});
    auto expected = NDArrayFactory::create<float>('c', { 5,1,4 }, { 36.626545f, 38.607746f, -40.614971f, 18.233341f, -51.545094f,2.234142f, 20.913160f, 8.783220f, 15.955761f, 55.273506f, 36.838833f, -29.751089f, 8.148357f, 13.676106f, 1.097548f, 68.766457f, 38.690712f, 27.176361f, -14.156269f, 7.157052f  });

    sd::ops::rgb_to_grs op;
    auto result = op.evaluate({ &rgbs }, {}, {1});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, test_rgb_to_grs_8) {
    // rank 3
    auto rgbs = NDArrayFactory::create<float>('c', { 3,5,4 }, {1.7750e+01f, -7.1062e+01f, -1.0019e+02f,-2.3406e+01f,  5.2094e+01f,  9.5438e+01f, -6.7461e+00f,  3.8562e+01f,  6.5078e+00f,3.3562e+01f, -5.8844e+01f,  2.2750e+01f, -1.0477e+01f,  7.7344e+00f,  9.5469e+00f,2.1391e+01f, -8.5312e+01f,  7.5830e-01f,2.3125e+01f,  1.8145e+00f,  1.4602e+01f,-4.5859e+00f,  3.9344e+01f,  1.1617e+01f,-8.6562e+01f,  1.0038e+02f,  6.7938e+01f,5.9961e+00f,  6.7812e+01f,  2.9734e+01f,2.9609e+01f, -6.1438e+01f,  1.7750e+01f,6.8562e+01f, -7.4414e+00f,  3.9656e+01f,1.1641e+01f, -2.7516e+01f,  6.7562e+01f,7.8438e+01f,  5.4883e+00f,  2.9438e+01f,-3.1344e+01f,  6.5125e+01f,  1.2695e+01f,4.0531e+01f, -6.1211e+00f,  6.2219e+01f,4.6812e+01f,  5.2250e+01f, -1.1414e+01f,1.5404e-02f,  2.9938e+01f,  5.6719e+00f,-2.0125e+01f,  2.1531e+01f,  6.2500e+01f,7.2188e+01f,  9.3750e+00f, -4.8125e+01f});
    try {
        sd::ops::rgb_to_grs op;
        auto result = op.evaluate({ &rgbs }, {}, {});
        ASSERT_EQ(Status::THROW(), result.status());

    } catch (std::exception& e) {
        nd4j_printf("Error should be here `%s'. It's OK.\n", e.what());
    }
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, test_rgb_to_grs_9) {
    // rank 3
    auto rgbs = NDArrayFactory::create<float>('f', { 2, 2, 3 }, { 1.7750e+01f,-7.1062e+01f, -1.0019e+02f, -2.3406e+01f,5.2094e+01f,9.5438e+01f, -6.7461e+00f,3.8562e+01f, 6.5078e+00f,      3.3562e+01f,-5.8844e+01f,2.2750e+01f});
    auto expected = NDArrayFactory::create<float>('f', { 2,2,1 }, { 36.626545f, 38.607746f, -40.614971f, 18.233341f });

    sd::ops::rgb_to_grs op;
    auto result = op.evaluate({ &rgbs }, {}, {});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, test_rgb_to_yuv_1) {
    // rank 1
    NDArray rgbs('f', { 3 }, { 10, 50, 200 }, sd::DataType::FLOAT32);
    NDArray expected('f', { 3 }, { 55.14 , 71.2872001, -39.6005542 }, sd::DataType::FLOAT32);
    sd::ops::rgb_to_yuv op;
    auto result = op.evaluate({ &rgbs }, {}, {});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, test_rgb_to_yuv_2) {

    NDArray rgbs('c', { 3, 2 }, { 14.,  99., 207., 10., 114., 201. }, sd::DataType::FLOAT32);
    rgbs.permutei({ 1,0 });

    NDArray expected('c', { 2, 3 }, { 138.691, -12.150713, -109.38929, 58.385, 70.18241, 35.63085 }, sd::DataType::FLOAT32);
    sd::ops::rgb_to_yuv op;

    auto result = op.evaluate({ &rgbs }, {}, {});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, test_rgb_to_yuv_3) {
    // rank 2
    NDArray rgbs('c', { 3, 4 }, { -9.4,  9.9, 9.7, 9.0, 1.14, 1.01, 1.11,  9.6, 1.05, 10.0, 1.03, 10.22 }, sd::DataType::FLOAT32);
    NDArray expected('c', { 3, 4 }, {  -2.021720, 4.692970, 3.669290, 9.491281, 1.511627, 2.611648, -1.298824, 0.358612, -6.472839, 4.568039, 5.290639, -0.430992 }, sd::DataType::FLOAT32);

    sd::ops::rgb_to_yuv op;
    auto result = op.evaluate({ &rgbs }, {}, { 0 });
    auto output = result.at(0);
    ASSERT_EQ(Status::OK(), result.status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, test_rgb_to_yuv_4) {
    // rank 3
    NDArray rgbs('c', { 5,4,3 }, { 1.7750e+01,  1.4602e+01,  5.4883e+00,  9.5438e+01,  1.0038e+02,  4.0531e+01,       -5.8844e+01,  2.9609e+01, -1.1414e+01,       2.1391e+01,  3.9656e+01,  2.1531e+01,       -7.1062e+01, -4.5859e+00,  2.9438e+01,       -6.7461e+00,  6.7938e+01, -6.1211e+00,       2.2750e+01, -6.1438e+01,  1.5404e-02,       -8.5312e+01,  1.1641e+01,  6.2500e+01,       -1.0019e+02,  3.9344e+01, -3.1344e+01,       3.8562e+01,  5.9961e+00,  6.2219e+01,       -1.0477e+01,  1.7750e+01,  2.9938e+01,       7.5830e-01, -2.7516e+01,  7.2188e+01,       -2.3406e+01,  1.1617e+01,  6.5125e+01,       6.5078e+00,  6.7812e+01,  4.6812e+01,       7.7344e+00,  6.8562e+01,  5.6719e+00,       2.3125e+01,  6.7562e+01,  9.3750e+00,        5.2094e+01, -8.6562e+01,  1.2695e+01,       3.3562e+01,  2.9734e+01,  5.2250e+01,       9.5469e+00, -7.4414e+00, -2.0125e+01,       1.8145e+00,  7.8438e+01, -4.8125e+01    }, sd::DataType::FLOAT32);
    NDArray expected('c', { 5,4,3 }, { 14.5042902, -4.43686799,   2.847406,  92.079556, -25.36761168,   2.94630572,  -1.515069, -4.87137291, -50.29369639,  32.128515, -5.21515376, -9.41983935,-20.5835293,   24.61614501, -44.28390394,  37.1647167, -21.30142676, -38.52221293, -29.26009994,  14.40679768,  45.62757638, -11.550021,    36.44083018, -64.71012983,-10.435098, - 10.28950082, - 78.74044941,  22.1427147,   19.72198103,  14.40435988,  10.699559,     9.46744852, - 18.5778351 ,  -7.6957283,   39.31166179,   7.41657542,  7.245035,    28.48336771, - 26.88963173,  47.0880442, - 0.13584441, - 35.60035823,  43.2050762, - 18.47048906, - 31.11782117,  47.642019, - 18.83162118, - 21.50836396,-33.788558,    22.87507047,  75.34330791,  33.445396,     9.25395257,   0.10229474,  -3.8078287, -8.02985955,  11.71587638,  41.0993915, -43.90830496, -34.46396749 }, sd::DataType::FLOAT32);

    sd::ops::rgb_to_yuv op;
    auto result = op.evaluate({ &rgbs }, {}, {});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, test_rgb_to_yuv_5) {
    // rank 3
    NDArray rgbs('c', { 5,3,4 }, { 1.7750e+01f, -7.1062e+01f, -1.0019e+02f,-2.3406e+01f,  5.2094e+01f,  9.5438e+01f, -6.7461e+00f,  3.8562e+01f,  6.5078e+00f,3.3562e+01f, -5.8844e+01f,  2.2750e+01f, -1.0477e+01f,  7.7344e+00f,  9.5469e+00f,2.1391e+01f, -8.5312e+01f,  7.5830e-01f,2.3125e+01f,  1.8145e+00f,  1.4602e+01f,-4.5859e+00f,  3.9344e+01f,  1.1617e+01f,-8.6562e+01f,  1.0038e+02f,  6.7938e+01f,5.9961e+00f,  6.7812e+01f,  2.9734e+01f,2.9609e+01f, -6.1438e+01f,  1.7750e+01f,6.8562e+01f, -7.4414e+00f,  3.9656e+01f,1.1641e+01f, -2.7516e+01f,  6.7562e+01f,7.8438e+01f,  5.4883e+00f,  2.9438e+01f,-3.1344e+01f,  6.5125e+01f,  1.2695e+01f,4.0531e+01f, -6.1211e+00f,  6.2219e+01f,4.6812e+01f,  5.2250e+01f, -1.1414e+01f,1.5404e-02f,  2.9938e+01f,  5.6719e+00f,-2.0125e+01f,  2.1531e+01f,  6.2500e+01f,7.2188e+01f,  9.3750e+00f, -4.8125e+01f }, sd::DataType::FLOAT32);
    NDArray expected('c', { 5,3,4 }, { 36.628319, 38.600643,-40.624989, 18.231001, - 14.822637, - 2.479566, - 8.965780,  2.223851, -16.561626,-96.205162,-52.255379,-36.527435,-51.546139,2.234915,  20.914114, 8.785358,  32.552223, -3.356598, 9.069552,  1.393482,36.029255, 4.824605,- 9.972263,11.058715, 15.947105, 55.283543, 36.845627, -29.750486,0.887228,  6.534475,  -21.794132,34.155693, -89.929497,39.562351, 27.276817,31.359871, 8.149521,  13.673355, 1.104303, 68.774300, 2.236881, 13.216944, - 3.555702,- 3.225931,3.063015, - 36.134724,58.302204, 8.477802, 38.695396,27.181587, - 14.157411,7.157054, 11.714512, 22.148155, 11.580557, - 27.204905,7.120562, 21.992094, 2.406748, - 6.265247,     }, sd::DataType::FLOAT32);

    sd::ops::rgb_to_yuv op;
    auto result = op.evaluate({ &rgbs }, {}, { 1 });
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, test_rgb_to_yuv_6) {
    // rank 3
    NDArray rgbs('c', { 3,5,4 }, { 1.7750e+01f, -7.1062e+01f, -1.0019e+02f,-2.3406e+01f,  5.2094e+01f,  9.5438e+01f, -6.7461e+00f,  3.8562e+01f,  6.5078e+00f,3.3562e+01f, -5.8844e+01f,  2.2750e+01f, -1.0477e+01f,  7.7344e+00f,  9.5469e+00f,2.1391e+01f, -8.5312e+01f,  7.5830e-01f,2.3125e+01f,  1.8145e+00f,  1.4602e+01f,-4.5859e+00f,  3.9344e+01f,  1.1617e+01f,-8.6562e+01f,  1.0038e+02f,  6.7938e+01f,5.9961e+00f,  6.7812e+01f,  2.9734e+01f,2.9609e+01f, -6.1438e+01f,  1.7750e+01f,6.8562e+01f, -7.4414e+00f,  3.9656e+01f,1.1641e+01f, -2.7516e+01f,  6.7562e+01f,7.8438e+01f,  5.4883e+00f,  2.9438e+01f,-3.1344e+01f,  6.5125e+01f,  1.2695e+01f,4.0531e+01f, -6.1211e+00f,  6.2219e+01f,4.6812e+01f,  5.2250e+01f, -1.1414e+01f,1.5404e-02f,  2.9938e+01f,  5.6719e+00f,-2.0125e+01f,  2.1531e+01f,  6.2500e+01f,7.2188e+01f,  9.3750e+00f, -4.8125e+01f }, sd::DataType::FLOAT32);
    try {
        sd::ops::rgb_to_yuv op;
        auto result = op.evaluate({ &rgbs }, {}, {});
        ASSERT_EQ(Status::THROW(), result.status());

    }
    catch (std::exception & e) {
        nd4j_printf("Error should be here `%s'. It's OK.\n", e.what());
    }
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, test_rgb_to_yuv_7) {
    // rank 3
    NDArray rgbs('f', { 2, 2, 3 }, { 1.7750e+01f,-7.1062e+01f, -1.0019e+02f, -2.3406e+01f,5.2094e+01f,9.5438e+01f, -6.7461e+00f,3.8562e+01f, 6.5078e+00f,      3.3562e+01f,-5.8844e+01f,2.2750e+01f }, sd::DataType::FLOAT32);
    NDArray expected('f', { 2,2,3 }, { 36.628319,38.600643, -40.624989,18.231001, -14.822637,-2.479566, -8.965780, 2.223851,  -16.561626,- 96.205162,-52.255379, -36.527435 }, sd::DataType::FLOAT32);

    sd::ops::rgb_to_yuv op;
    auto result = op.evaluate({ &rgbs }, {}, {});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, test_yuv_to_rgb_1) {
    // rank 1
    NDArray yuv('c', { 3 }, { 55.14 , 71.2872001, -39.6005542 }, sd::DataType::FLOAT32);
    NDArray expected('c', { 3 }, { 10, 50, 200 }, sd::DataType::FLOAT32);
    sd::ops::yuv_to_rgb op;
    auto result = op.evaluate({ &yuv }, {}, {});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));


}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, test_yuv_to_rgb_2) {
    // rank 1
    NDArray yuv('f', { 3 }, { 55.14, 71.2872001, -39.6005542 }, sd::DataType::FLOAT32);
    NDArray expected('f', { 3 }, { 10, 50, 200 }, sd::DataType::FLOAT32);
    sd::ops::yuv_to_rgb op;
    auto result = op.evaluate({ &yuv }, {}, {});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, test_yuv_to_rgb_3) {
    // rank 2
    NDArray expected('c', { 3, 4 }, { -9.4,  9.9, 9.7, 9.0, 1.14, 1.01, 1.11,  9.6, 1.05, 10.0, 1.03, 10.22 }, sd::DataType::FLOAT32);
    NDArray yuv('c', { 3, 4 }, { -2.021720, 4.692970, 3.669290, 9.491281, 1.511627, 2.611648, -1.298824, 0.358612, -6.472839, 4.568039, 5.290639, -0.430992 }, sd::DataType::FLOAT32);

    sd::ops::yuv_to_rgb op;
    auto result = op.evaluate({ &yuv }, {}, { 0 });
    auto output = result.at(0);
    ASSERT_EQ(Status::OK(), result.status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, test_yuv_to_rgb_4) {
    // rank 3
    NDArray expected('c', { 5,4,3 }, { 1.7750e+01,  1.4602e+01,  5.4883e+00,  9.5438e+01,  1.0038e+02,  4.0531e+01,       -5.8844e+01,  2.9609e+01, -1.1414e+01,       2.1391e+01,  3.9656e+01,  2.1531e+01,       -7.1062e+01, -4.5859e+00,  2.9438e+01,       -6.7461e+00,  6.7938e+01, -6.1211e+00,       2.2750e+01, -6.1438e+01,  1.5404e-02,       -8.5312e+01,  1.1641e+01,  6.2500e+01,       -1.0019e+02,  3.9344e+01, -3.1344e+01,       3.8562e+01,  5.9961e+00,  6.2219e+01,       -1.0477e+01,  1.7750e+01,  2.9938e+01,       7.5830e-01, -2.7516e+01,  7.2188e+01,       -2.3406e+01,  1.1617e+01,  6.5125e+01,       6.5078e+00,  6.7812e+01,  4.6812e+01,       7.7344e+00,  6.8562e+01,  5.6719e+00,       2.3125e+01,  6.7562e+01,  9.3750e+00,        5.2094e+01, -8.6562e+01,  1.2695e+01,       3.3562e+01,  2.9734e+01,  5.2250e+01,       9.5469e+00, -7.4414e+00, -2.0125e+01,       1.8145e+00,  7.8438e+01, -4.8125e+01 }, sd::DataType::FLOAT32);
    NDArray yuv('c', { 5,4,3 }, { 14.5042902, -4.43686799,   2.847406,  92.079556, -25.36761168,   2.94630572,  -1.515069, -4.87137291, -50.29369639,  32.128515, -5.21515376, -9.41983935,-20.5835293,   24.61614501, -44.28390394,  37.1647167, -21.30142676, -38.52221293, -29.26009994,  14.40679768,  45.62757638, -11.550021,    36.44083018, -64.71012983,-10.435098, -10.28950082, -78.74044941,  22.1427147,   19.72198103,  14.40435988,  10.699559,     9.46744852, -18.5778351 ,  -7.6957283,   39.31166179,   7.41657542,  7.245035,    28.48336771, -26.88963173,  47.0880442, -0.13584441, -35.60035823,  43.2050762, -18.47048906, -31.11782117,  47.642019, -18.83162118, -21.50836396,-33.788558,    22.87507047,  75.34330791,  33.445396,     9.25395257,   0.10229474,  -3.8078287, -8.02985955,  11.71587638,  41.0993915, -43.90830496, -34.46396749 }, sd::DataType::FLOAT32);

    sd::ops::yuv_to_rgb op;
    auto result = op.evaluate({ &yuv }, {}, {});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, test_yuv_to_rgb_5) {
    // rank 3
    NDArray expected('c', { 5,3,4 }, { 1.7750e+01f, -7.1062e+01f, -1.0019e+02f,-2.3406e+01f,  5.2094e+01f,  9.5438e+01f, -6.7461e+00f,  3.8562e+01f,  6.5078e+00f,3.3562e+01f, -5.8844e+01f,  2.2750e+01f, -1.0477e+01f,  7.7344e+00f,  9.5469e+00f,2.1391e+01f, -8.5312e+01f,  7.5830e-01f,2.3125e+01f,  1.8145e+00f,  1.4602e+01f,-4.5859e+00f,  3.9344e+01f,  1.1617e+01f,-8.6562e+01f,  1.0038e+02f,  6.7938e+01f,5.9961e+00f,  6.7812e+01f,  2.9734e+01f,2.9609e+01f, -6.1438e+01f,  1.7750e+01f,6.8562e+01f, -7.4414e+00f,  3.9656e+01f,1.1641e+01f, -2.7516e+01f,  6.7562e+01f,7.8438e+01f,  5.4883e+00f,  2.9438e+01f,-3.1344e+01f,  6.5125e+01f,  1.2695e+01f,4.0531e+01f, -6.1211e+00f,  6.2219e+01f,4.6812e+01f,  5.2250e+01f, -1.1414e+01f,1.5404e-02f,  2.9938e+01f,  5.6719e+00f,-2.0125e+01f,  2.1531e+01f,  6.2500e+01f,7.2188e+01f,  9.3750e+00f, -4.8125e+01f }, sd::DataType::FLOAT32);
    NDArray yuv('c', { 5,3,4 }, { 36.628319, 38.600643,-40.624989, 18.231001, -14.822637, -2.479566, -8.965780,  2.223851, -16.561626,-96.205162,-52.255379,-36.527435,-51.546139,2.234915,  20.914114, 8.785358,  32.552223, -3.356598, 9.069552,  1.393482,36.029255, 4.824605,-9.972263,11.058715, 15.947105, 55.283543, 36.845627, -29.750486,0.887228,  6.534475,  -21.794132,34.155693, -89.929497,39.562351, 27.276817,31.359871, 8.149521,  13.673355, 1.104303, 68.774300, 2.236881, 13.216944, -3.555702,-3.225931,3.063015, -36.134724,58.302204, 8.477802, 38.695396,27.181587, -14.157411,7.157054, 11.714512, 22.148155, 11.580557, -27.204905,7.120562, 21.992094, 2.406748, -6.265247, }, sd::DataType::FLOAT32);

    sd::ops::yuv_to_rgb op;
    auto result = op.evaluate({ &yuv }, {}, { 1 });
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, test_yuv_to_rgb_6) {
    // rank 3
    NDArray yuv('c', { 3,5,4 }, { 1.7750e+01f, -7.1062e+01f, -1.0019e+02f,-2.3406e+01f,  5.2094e+01f,  9.5438e+01f, -6.7461e+00f,  3.8562e+01f,  6.5078e+00f,3.3562e+01f, -5.8844e+01f,  2.2750e+01f, -1.0477e+01f,  7.7344e+00f,  9.5469e+00f,2.1391e+01f, -8.5312e+01f,  7.5830e-01f,2.3125e+01f,  1.8145e+00f,  1.4602e+01f,-4.5859e+00f,  3.9344e+01f,  1.1617e+01f,-8.6562e+01f,  1.0038e+02f,  6.7938e+01f,5.9961e+00f,  6.7812e+01f,  2.9734e+01f,2.9609e+01f, -6.1438e+01f,  1.7750e+01f,6.8562e+01f, -7.4414e+00f,  3.9656e+01f,1.1641e+01f, -2.7516e+01f,  6.7562e+01f,7.8438e+01f,  5.4883e+00f,  2.9438e+01f,-3.1344e+01f,  6.5125e+01f,  1.2695e+01f,4.0531e+01f, -6.1211e+00f,  6.2219e+01f,4.6812e+01f,  5.2250e+01f, -1.1414e+01f,1.5404e-02f,  2.9938e+01f,  5.6719e+00f,-2.0125e+01f,  2.1531e+01f,  6.2500e+01f,7.2188e+01f,  9.3750e+00f, -4.8125e+01f }, sd::DataType::FLOAT32);
    try {
        sd::ops::yuv_to_rgb op;
        auto result = op.evaluate({ &yuv }, {}, {});
        ASSERT_EQ(Status::THROW(), result.status());

    }
    catch (std::exception & e) {
        nd4j_printf("Error should be here `%s'. It's OK.\n", e.what());
    }
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, test_yuv_to_rgb_7) {
    // rank 3
    NDArray expected('f', { 2, 2, 3 }, { 1.7750e+01f,-7.1062e+01f, -1.0019e+02f, -2.3406e+01f,5.2094e+01f,9.5438e+01f, -6.7461e+00f,3.8562e+01f, 6.5078e+00f,      3.3562e+01f,-5.8844e+01f,2.2750e+01f }, sd::DataType::FLOAT32);
    NDArray yuv('f', { 2,2,3 }, { 36.628319, 38.600643, -40.624989, 18.231001, -14.822637, -2.479566, -8.965780, 2.223851, -16.561626, -96.205162, -52.255379, -36.527435 }, sd::DataType::FLOAT32);

    sd::ops::yuv_to_rgb op;
    auto result = op.evaluate({ &yuv }, {}, {});
    auto output = result.at(0);

    ASSERT_EQ(Status::OK(), result.status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////////////

TEST_F(DeclarableOpsTests15, Pow_BP_Test1) {

    // same shape
    NDArray x('c', { 2,2,2 }, { 4,3,2,5,7,8,-9,-12 }, sd::DataType::FLOAT32);
    NDArray y('c', { 2,2,2 }, { 2,3,-2,4,-1,-4,10,8 }, sd::DataType::FLOAT32);


    NDArray dLdz('c', { 2,2,2 }, sd::DataType::FLOAT32);
    NDArray dLdxExp('c', { 2,2,2 }, { 8,  27, -0.25,  500, -0.0204082, -0.000122, -3.87420e+09, -2.86654e+08 }, sd::DataType::FLOAT32);
    NDArray dLdyExp('c', { 2,2,2 }, { 22.18071, 29.66253, 0.17329, 1005.89874, 0.27799, 0.00051, 0, 0 }, sd::DataType::FLOAT32);

    dLdz.assign(1.0);

    sd::ops::Pow_bp op;
    auto results = op.evaluate({ &x, &y, &dLdz }, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto* dLdx = results.at(0);
    auto* dLdy = results.at(1);

    ASSERT_TRUE(dLdxExp.isSameShape(dLdx));
    ASSERT_TRUE(dLdxExp.equalsTo(dLdx));
    ASSERT_TRUE(dLdyExp.isSameShape(dLdy));
    ASSERT_TRUE(dLdyExp.equalsTo(dLdy));
}

TEST_F(DeclarableOpsTests15, Pow_BP_Test2) {

    NDArray x('c', { 1,2,3 }, sd::DataType::FLOAT32);
    NDArray y('c', { 3,2,1 }, sd::DataType::FLOAT32);
    NDArray dLdz('c', { 3,2,3 }, sd::DataType::FLOAT32);

    NDArray dLdxExp('c', { 1,2,3 }, { 16.8, 19.2, 21.6, 24., 26.4, 28.8 }, sd::DataType::FLOAT32);
    NDArray dLdyExp('c', { 3,2,1 }, { 13.30843, 33.27106, 53.2337, 73.19634, 93.15898, 113.12162 }, sd::DataType::FLOAT32);

    x.assign(4.0);
    y.assign(2.0);
    dLdz.linspace(0.1, 0.1);

    sd::ops::Pow_bp op;
    auto results = op.evaluate({ &x, &y, &dLdz }, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto* dLdx = results.at(0);
    auto* dLdy = results.at(1);

    ASSERT_TRUE(dLdxExp.isSameShape(dLdx));
    ASSERT_TRUE(dLdxExp.equalsTo(dLdx));
    ASSERT_TRUE(dLdyExp.isSameShape(dLdy));
    ASSERT_TRUE(dLdyExp.equalsTo(dLdy));

}

TEST_F(DeclarableOpsTests15, Pow_BP_Test3) {

    // y - same shape as dLdz
    NDArray xY('c', { 1,2,3 }, sd::DataType::FLOAT32);
    NDArray yY('c', { 3,2,3 }, sd::DataType::FLOAT32);

    NDArray dLdxExpY('c', { 1,2,3 }, { 16.8, 19.2, 21.6, 24. , 26.4, 28.8 }, sd::DataType::FLOAT32);
    NDArray dLdyExpY('c', { 3,2,3 }, { 2.21807,  4.43614,  6.65421, 8.87228, 11.09035, 13.30843, 15.5265 , 17.74457, 19.96264, 22.18071, 24.39878, 26.61685, 28.83492, 31.05299, 33.27106, 35.48914, 37.70721, 39.92528 }, sd::DataType::FLOAT32);
    NDArray dLdz('c', { 3,2,3 }, sd::DataType::FLOAT32);

    xY.assign(4.0);
    yY.assign(2.0);
    dLdz.linspace(0.1, 0.1);

    sd::ops::Pow_bp op;
    auto resultsY = op.evaluate({ &xY, &yY, &dLdz }, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, resultsY.status());

    auto* dLdxY = resultsY.at(0);
    auto* dLdyY = resultsY.at(1);

    ASSERT_TRUE(dLdxExpY.isSameShape(dLdxY));
    ASSERT_TRUE(dLdxExpY.equalsTo(dLdxY));
    ASSERT_TRUE(dLdyExpY.isSameShape(dLdyY));
    ASSERT_TRUE(dLdyExpY.equalsTo(dLdyY));
}

TEST_F(DeclarableOpsTests15, Pow_BP_Test4) {

    // x - same shape ad dLdz
    NDArray yX('c', { 1,2,3 }, sd::DataType::FLOAT32);
    NDArray xX('c', { 3,2,3 }, sd::DataType::FLOAT32);

    NDArray dLdxExpX('c', { 3,2,3 }, { 3.2,  6.4,  9.6, 12.8, 16. , 19.2, 22.4, 25.6, 28.8, 32. , 35.2, 38.4, 41.6, 44.8, 48., 51.2, 54.4, 57.6 }, sd::DataType::FLOAT32);
    NDArray dLdyExpX('c', { 1,2,3 }, { 23.28975, 26.61685, 29.94396, 33.27106, 36.59817, 39.92528 }, sd::DataType::FLOAT32);

    NDArray dLdz('c', { 3,2,3 }, sd::DataType::FLOAT32);
    dLdz.linspace(0.1, 0.1);

    sd::ops::Pow_bp op;

    xX.assign(2.0);
    yX.assign(4.0);

    auto resultsX = op.evaluate({ &xX, &yX, &dLdz }, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, resultsX.status());

    auto* dLdxX = resultsX.at(0);
    auto* dLdyX = resultsX.at(1);

    ASSERT_TRUE(dLdxExpX.isSameShape(dLdxX));
    ASSERT_TRUE(dLdxExpX.equalsTo(dLdxX));
    ASSERT_TRUE(dLdyExpX.isSameShape(dLdyX));
    ASSERT_TRUE(dLdyExpX.equalsTo(dLdyX));
}

TEST_F(DeclarableOpsTests15, Pow_BP_Test5) {

    // both single array
    NDArray xConst('c', { 1 }, sd::DataType::FLOAT32);
    NDArray yConst('c', { 1 }, sd::DataType::FLOAT32);
    NDArray dLdz('c', { 1 }, sd::DataType::FLOAT32);
    NDArray dLdxExp('c', { 1 }, sd::DataType::FLOAT32);
    NDArray dLdyExp('c', { 1 }, sd::DataType::FLOAT32);

    xConst.assign(3.0);
    yConst.assign(4.0);
    dLdz.assign(1.0);

    dLdxExp.assign(4.0 * pow(3, 3));
    dLdyExp.assign(pow(3, 4) * log(3));

    sd::ops::Pow_bp op;
    auto results = op.evaluate({ &xConst, &yConst, &dLdz }, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto* dLdx = results.at(0);
    auto* dLdy = results.at(1);

    ASSERT_TRUE(dLdxExp.isSameShape(dLdx));
    ASSERT_TRUE(dLdxExp.equalsTo(dLdx));

    ASSERT_TRUE(dLdyExp.isSameShape(dLdy));
    ASSERT_TRUE(dLdyExp.equalsTo(dLdy));
}

TEST_F(DeclarableOpsTests15, Pow_BP_Test6) {

    // x single array
    NDArray xConst('c', { 1 }, sd::DataType::FLOAT32);
    NDArray y('c', { 2, 2, 2 }, sd::DataType::FLOAT32);
    NDArray dLdzC('c', { 2, 2, 2 }, sd::DataType::FLOAT32);

    xConst.assign(2.0);
    y.assign(4.0);
    dLdzC.linspace(0.1, 0.1);

    NDArray dLdxExpXC('c', { 1 }, std::vector<double>{ 115.2 }, sd::DataType::FLOAT32);
    NDArray dLdyExpXC('c', { 2, 2, 2 }, { 1.10904, 2.21807, 3.32711, 4.43614, 5.54518, 6.65421, 7.76325, 8.87228 }, sd::DataType::FLOAT32);

    sd::ops::Pow_bp op;
    auto resultsXC = op.evaluate({ &xConst, &y, &dLdzC }, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, resultsXC.status());

    auto* dLdxXC = resultsXC.at(0);
    auto* dLdyXC = resultsXC.at(1);

    ASSERT_TRUE(dLdxExpXC.isSameShape(dLdxXC));
    ASSERT_TRUE(dLdxExpXC.equalsTo(dLdxXC));
    ASSERT_TRUE(dLdyExpXC.isSameShape(dLdyXC));
    ASSERT_TRUE(dLdyExpXC.equalsTo(dLdyXC));

}

TEST_F(DeclarableOpsTests15, Pow_BP_Test7) {

    // Y - scalar
    auto Y = NDArrayFactory::create<float>(2.f);
    NDArray x('c', { 2, 2, 2 }, sd::DataType::FLOAT32);
    NDArray dLdzC('c', { 2, 2, 2 }, sd::DataType::FLOAT32);

    dLdzC.linspace(0.1, 0.1);
    x = 4.f;

    NDArray dLdxExpYs('c', { 2, 2, 2 }, { 0.8, 1.6, 2.4, 3.2, 4., 4.8, 5.6, 6.4 }, sd::DataType::FLOAT32);

    auto dLdyExpYs = NDArrayFactory::create<float>(79.85056f);

    sd::ops::Pow_bp op;
    auto resultsYs = op.evaluate({ &x, &Y, &dLdzC }, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, resultsYs.status());

    auto* dLdxY = resultsYs.at(0);
    auto* dLdyY = resultsYs.at(1);

    ASSERT_TRUE(dLdxExpYs.isSameShape(dLdxY));
    ASSERT_TRUE(dLdxExpYs.equalsTo(dLdxY));
    ASSERT_TRUE(dLdyExpYs.isSameShape(dLdyY));
    ASSERT_TRUE(dLdyExpYs.equalsTo(dLdyY));
}

TEST_F(DeclarableOpsTests15, Pow_BP_Test8) {
    // both scalars

    auto X = NDArrayFactory::create<float>(4.f);
    auto Y = NDArrayFactory::create<float>(2.f);
    NDArray dLdz = NDArrayFactory::create<float>(0.1f);

    NDArray dLdxExp = NDArrayFactory::create<float>(2.f*4.f*0.1f);

    NDArray dLdyExp = NDArrayFactory::create<float>(pow(4.f, 2.f) * log(4.f) * 0.1f);

    sd::ops::Pow_bp op;
    auto results = op.evaluate({ &X, &Y, &dLdz }, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto* dLdx = results.at(0);
    auto* dLdy = results.at(1);

    ASSERT_TRUE(dLdxExp.isSameShape(dLdx));
    ASSERT_TRUE(dLdxExp.equalsTo(dLdx));
    ASSERT_TRUE(dLdyExp.isSameShape(dLdy));
    ASSERT_TRUE(dLdyExp.equalsTo(dLdy));

}

TEST_F(DeclarableOpsTests15, Pow_BP_Test9) {

    sd::ops::Pow_bp op;
    // diff shapes
    NDArray x('c', { 3,2,1 }, sd::DataType::FLOAT32);
    NDArray y('c', { 1,2,3 }, sd::DataType::FLOAT32);
    NDArray dLdz('c', { 3,2,3 }, sd::DataType::FLOAT32);

    NDArray dLdxExp('c', { 3,2,1 }, { 4.8, 12., 19.2, 26.4, 33.6, 40.8 }, sd::DataType::FLOAT32);
    NDArray dLdyExp('c', { 1,2,3 }, { 46.57949, 53.2337 , 59.88792, 66.54213, 73.19634, 79.85056 }, sd::DataType::FLOAT32);

    x.assign(4.0);
    y.assign(2.0);
    dLdz.linspace(0.1, 0.1);

    auto results = op.evaluate({ &x, &y, &dLdz }, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto* dLdx = results.at(0);
    auto* dLdy = results.at(1);

    ASSERT_TRUE(dLdxExp.isSameShape(dLdx));
    ASSERT_TRUE(dLdxExp.equalsTo(dLdx));
    ASSERT_TRUE(dLdyExp.isSameShape(dLdy));
    ASSERT_TRUE(dLdyExp.equalsTo(dLdy));
}

TEST_F(DeclarableOpsTests15, Pow_BP_Test10) {

    // diff shapes broadcastable
    NDArray yB('c', { 1,2,3,1 }, sd::DataType::FLOAT32);
    NDArray xB('c', { 2,3,1 }, sd::DataType::FLOAT32);

    NDArray dLdyExpB('c', { 1,2,3,1 }, { 2.21807, 4.43614, 6.65421, 8.87228, 11.09035, 13.30843 }, sd::DataType::FLOAT32);
    NDArray dLdxExpB('c', { 2,3,1 }, { 0.8, 1.6, 2.4, 3.2, 4., 4.8 }, sd::DataType::FLOAT32);
    NDArray dLdzB('c', { 1,2,3,1 }, sd::DataType::FLOAT32);

    dLdzB.linspace(0.1, 0.1);
    xB.assign(4.0);
    yB.assign(2.0);

    sd::ops::Pow_bp op;
    auto resultsB = op.evaluate({ &xB, &yB, &dLdzB }, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, resultsB.status());

    auto* dLdxB = resultsB.at(0);
    auto* dLdyB = resultsB.at(1);

    ASSERT_TRUE(dLdxExpB.isSameShape(dLdxB));
    ASSERT_TRUE(dLdxExpB.equalsTo(dLdxB));

    ASSERT_TRUE(dLdyExpB.isSameShape(dLdyB));
    ASSERT_TRUE(dLdyExpB.equalsTo(dLdyB));
}

TEST_F(DeclarableOpsTests15, Pow_BP_Test11) {
#ifdef FFAST_MATH
    if (1 > 0)
        return;
#endif

    NDArray xB('c', { 3,2,1 }, { .4, 3, 5, .8, -9, -12 }, sd::DataType::FLOAT32);
    NDArray yB('c', { 1,2,3 }, { 3, -2, .4, -4, 10, .8 }, sd::DataType::FLOAT32);

    NDArray dLdxExpB('c', { 3,2,1 }, { -5.994056, 39366.191406, 7.508829, -2.223537, -std::numeric_limits<float>::quiet_NaN(), -std::numeric_limits<float>::quiet_NaN() }, sd::DataType::FLOAT32);
    NDArray dLdyExpB('c', { 1,2,3 }, { 20.11211,  -1.119612, -std::numeric_limits<float>::quiet_NaN(), -0.1076, 12974.389648, -std::numeric_limits<float>::quiet_NaN() }, sd::DataType::FLOAT32);

    NDArray dLdzB('c', { 3,2,3 }, { .1,.2,.3, .1,.2,.3, .1,.4,.1, .2,.1,.1, .3,.1,.5, .1, .7, .1 }, sd::DataType::FLOAT32);

    sd::ops::Pow_bp op;
    auto resultsB = op.evaluate({ &xB, &yB, &dLdzB }, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, resultsB.status());
    auto* dLdxB = resultsB.at(0);
    auto* dLdyB = resultsB.at(1);

    ASSERT_TRUE(dLdxExpB.isSameShape(dLdxB));
    for (int i = 0; i < dLdxB->lengthOf(); ++i) {
        if (!sd::math::nd4j_isnan(dLdxB->e<float>(i)) && !sd::math::nd4j_isnan(dLdxExpB.e<float>(i)))
            ASSERT_NEAR(dLdxB->e<float>(i), dLdxExpB.e<float>(i), 0.00001);
    }

    ASSERT_TRUE(dLdyExpB.isSameShape(dLdyB));
    for (int i = 0; i < dLdyB->lengthOf(); ++i) {
        if (!sd::math::nd4j_isnan(dLdyB->e<float>(i)) && !sd::math::nd4j_isnan(dLdyExpB.e<float>(i)))
            ASSERT_NEAR(dLdyB->e<float>(i), dLdyExpB.e<float>(i), 0.00001);
    }


}
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, TestTensorMmul_BP1) {

    NDArray A('c', { 1, 2, 3 }, { 2.1, 2.2, 2.3, 2.4, 2.5, 2.6 }, sd::DataType::FLOAT32);
    NDArray B('c', { 1, 2, 4 }, { 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8 }, sd::DataType::FLOAT32);
    NDArray dLdC('c', { 3, 4 }, { .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 1.1, 1.1 }, sd::DataType::FLOAT32);

    NDArray dLdA('c', { 1, 2, 3 }, { 3.3,  8.5,  13.36, 3.7, 9.54, 15. }, sd::DataType::FLOAT32);
    NDArray dLdB('c', { 1, 2, 4 }, { 3.38, 4.04, 4.7, 5.13, 3.83, 4.58, 5.33, 5.82 }, sd::DataType::FLOAT32);

    sd::ops::tensormmul_bp op_bp;

    auto resultsBP = op_bp.evaluate({ &A, &B, &dLdC }, {}, { 2,0,1, 2,0,1 }, {});

    ASSERT_EQ(ND4J_STATUS_OK, resultsBP.status());

    auto* dLdAbp = resultsBP.at(0);
    auto* dLdBbp = resultsBP.at(1);

    ASSERT_TRUE(dLdA.isSameShape(*dLdAbp));
    ASSERT_TRUE(dLdA.equalsTo(*dLdAbp));

    ASSERT_TRUE(dLdB.isSameShape(*dLdBbp));
    ASSERT_TRUE(dLdB.equalsTo(*dLdBbp));

}
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, TestTensorMmul_BP2) {

    NDArray A('c', { 1, 2, 3 }, { 2,2,2, 2,2,2 }, sd::DataType::FLOAT32);
    NDArray B('c', { 1, 2, 3 }, { 3,3,3,3, 3,3 }, sd::DataType::FLOAT32);
    NDArray dLdC('c', { 1 }, { 1 }, sd::DataType::FLOAT32);

    sd::ops::tensormmul_bp op_bp;
    auto resultsBP = op_bp.evaluate({ &A, &B, &dLdC }, {}, { 2,1,2, 2,1,2 }, {});

    ASSERT_EQ(ND4J_STATUS_OK, resultsBP.status());

    auto* dLdAbp = resultsBP.at(0);
    auto* dLdBbp = resultsBP.at(1);

    ASSERT_TRUE(B.isSameShape(*dLdAbp));
    ASSERT_TRUE(B.equalsTo(*dLdAbp));

    ASSERT_TRUE(A.isSameShape(*dLdBbp));
    ASSERT_TRUE(A.equalsTo(*dLdBbp));

}
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, TestTensorMmul_BP3) {

    NDArray A('c', { 3, 2, 2 }, { 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3., 3.1, 3.2 }, sd::DataType::FLOAT32);
    NDArray B('c', { 4, 2, 2 }, { 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4., 4.1, 4.2, 4.3, 4.4, 4.5, 4.6 }, sd::DataType::FLOAT32);
    NDArray dLdC('c', { 3, 4 }, { .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 1.1, 1.2 }, sd::DataType::FLOAT32);

    NDArray dA('c', { 3, 2, 2 }, { 3.9, 4., 4.1, 4.2, 9.82, 10.08, 10.34, 10.6, 15.74, 16.16, 16.58, 17. }, sd::DataType::FLOAT32);
    NDArray dB('c', { 4, 2, 2 }, { 4.07, 4.22, 4.37, 4.52, 4.82, 5., 5.18, 5.36, 5.57, 5.78, 5.99, 6.2, 6.32, 6.56, 6.8,  7.04 }, sd::DataType::FLOAT32);

    sd::ops::tensormmul_bp op_bp;

    auto resultsBP = op_bp.evaluate({ &A, &B, &dLdC }, {}, { 2,1,2, 2,1,2 }, {});

    ASSERT_EQ(ND4J_STATUS_OK, resultsBP.status());

    auto* dLdAbp = resultsBP.at(0);
    auto* dLdBbp = resultsBP.at(1);

    ASSERT_TRUE(dA.isSameShape(*dLdAbp));
    ASSERT_TRUE(dA.equalsTo(*dLdAbp));

    ASSERT_TRUE(dB.isSameShape(*dLdBbp));
    ASSERT_TRUE(dB.equalsTo(*dLdBbp));

}
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, TestTensorMmul_BP4) {

    NDArray A('c', { 3, 4, 1 }, { 0.4, 3, 5,  9, 23, 0.12,  8, 9, 0.1,  0, 124, 3 }, sd::DataType::FLOAT32);
    NDArray B('c', { 2, 4, 1 }, { 4, 13, .5,  19, 2.3, 1.2,  18, .9 }, sd::DataType::FLOAT32);
    NDArray dLdC('c', { 3, 2 }, { 1.1, 1.2, 1.3, 1.4, 1.5, 1.6 }, sd::DataType::FLOAT32);

    NDArray dLdA('c', { 3, 4, 1 }, { 7.16, 15.74, 22.15, 21.98, 8.42, 18.58, 25.85, 25.96, 9.68, 21.42, 29.55, 29.94 }, sd::DataType::FLOAT32);
    NDArray dLdB('c', { 2, 4, 1 }, { 30.49, 3.456, 201.9, 26.1, 32.84 , 3.768, 215.6, 28.2 }, sd::DataType::FLOAT32);

    sd::ops::tensormmul_bp op_bp;

    auto resultsBP = op_bp.evaluate({ &A, &B, &dLdC }, {}, { 2,1,2, 2,1,2 }, {});

    ASSERT_EQ(ND4J_STATUS_OK, resultsBP.status());

    auto* dLdAbp = resultsBP.at(0);
    auto* dLdBbp = resultsBP.at(1);

    ASSERT_TRUE(dLdA.isSameShape(*dLdAbp));
    ASSERT_TRUE(dLdA.equalsTo(*dLdAbp));

    ASSERT_TRUE(dLdB.isSameShape(*dLdBbp));
    ASSERT_TRUE(dLdB.equalsTo(*dLdBbp));

}
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, TestTensorMmul_BP5) {

    NDArray A('c', { 3, 4, 1, 1 }, { 0.4, 3, 5,  9, 23, 0.12,  8, 9, 0.1,  0, 124, 3 }, sd::DataType::FLOAT32);
    NDArray B('c', { 2, 4, 1, 1 }, { 4, 13, .5,  19, 2.3, 1.2,  18, .9 }, sd::DataType::FLOAT32);
    NDArray dLdC('c', { 3, 1, 2, 1 }, { 1.1,1.2,1.3,1.4,1.5,1.6 }, sd::DataType::FLOAT32);

    NDArray dLdA('c', { 3, 4, 1, 1 }, { 7.16, 15.74, 22.15, 21.98, 8.42, 18.58, 25.85, 25.96, 9.68, 21.42, 29.55, 29.94 }, sd::DataType::FLOAT32);
    NDArray dLdB('c', { 2, 4, 1, 1 }, { 30.49, 3.456, 201.9,  26.1, 32.84,  3.768, 215.6, 28.2 }, sd::DataType::FLOAT32);

    sd::ops::tensormmul_bp op_bp;

    auto resultsBP = op_bp.evaluate({ &A, &B, &dLdC }, {}, { 2,1,2, 2,1,2 }, {});

    ASSERT_EQ(ND4J_STATUS_OK, resultsBP.status());

    auto* dLdAbp = resultsBP.at(0);
    auto* dLdBbp = resultsBP.at(1);

    ASSERT_TRUE(dLdA.isSameShape(*dLdAbp));
    ASSERT_TRUE(dLdA.equalsTo(*dLdAbp));

    ASSERT_TRUE(dLdB.isSameShape(*dLdBbp));
    ASSERT_TRUE(dLdB.equalsTo(*dLdBbp));

}
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, TestTensorMmul_BP6) {

    NDArray A('c', { 2, 2, 2 }, { 2,2, 2,2, 2,2, 2,2 }, sd::DataType::FLOAT32);
    NDArray B('c', { 2, 2, 2 }, { 3,3, 3,3, 3,3, 3,3  }, sd::DataType::FLOAT32);

    auto dLdC = NDArrayFactory::create<float>(1.f);

    sd::ops::tensormmul_bp op_bp;
    auto resultsBP = op_bp.evaluate({ &A, &B, &dLdC }, {}, { 3,0,1,2, 3,0,1,2 }, {});

    ASSERT_EQ(ND4J_STATUS_OK, resultsBP.status());

    auto* dLdAbp = resultsBP.at(0);
    auto* dLdBbp = resultsBP.at(1);

    ASSERT_TRUE(B.isSameShape(*dLdAbp));
    ASSERT_TRUE(B.equalsTo(*dLdAbp));

    ASSERT_TRUE(A.isSameShape(*dLdBbp));
    ASSERT_TRUE(A.equalsTo(*dLdBbp));

}
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, TestTensorMmul_BP7) {

    NDArray A('c', { 3, 4, 1 }, { 0.4, 3, 5,  9, 23, 0.12,  8, 9, 0.1,  0, 124, 3 }, sd::DataType::FLOAT32);
    NDArray B('c', { 2, 4, 1 }, { 4, 13, .5,  19, 2.3, 1.2,  18, .9 }, sd::DataType::FLOAT32);
    NDArray dLdC('c', { 3, 1, 2, 1 }, { 1.1, 1.2, 1.3, 1.4, 1.5, 1.6 }, sd::DataType::FLOAT32);

    NDArray dLdA('c', { 3, 4, 1 }, { 7.16, 15.74, 22.15, 21.98, 8.42, 18.58, 25.85, 25.96, 9.68, 21.42, 29.55, 29.94 }, sd::DataType::FLOAT32);
    NDArray dLdB('c', { 2, 4, 1 }, { 30.49, 3.456, 201.9,  26.1, 32.84,  3.768, 215.6, 28.2 }, sd::DataType::FLOAT32);

    sd::ops::tensormmul_bp op_bp;

    auto resultsBP = op_bp.evaluate({ &A, &B, &dLdC }, {}, { 1,1, 1,1 }, {});

    ASSERT_EQ(ND4J_STATUS_OK, resultsBP.status());
    auto* dLdAbp = resultsBP.at(0);
    auto* dLdBbp = resultsBP.at(1);

    ASSERT_TRUE(dLdA.isSameShape(*dLdAbp));
    ASSERT_TRUE(dLdA.equalsTo(*dLdAbp));

    ASSERT_TRUE(dLdB.isSameShape(*dLdBbp));
    ASSERT_TRUE(dLdB.equalsTo(*dLdBbp));

}
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, TestTensorMmul_BP8) {

    NDArray A('c', { 1, 1, 4, 3 }, { 0.4, 3, 5,  9, 23, 0.12,  8, 9, 0.1,  0, 124, 3 }, sd::DataType::FLOAT32);
    NDArray B('c', { 1, 1, 4, 2 }, { 4, 13, .5,  19, 2.3, 1.2,  18, .9 }, sd::DataType::FLOAT32);
    NDArray dLdC('c', { 3, 2 }, { 1.1,1.2,1.3,1.4,1.5,1.6 }, sd::DataType::FLOAT32);

    NDArray dLdA('c', { 1, 1, 4, 3 }, { 20., 23.4,  26.8, 23.35, 27.25, 31.15, 3.97,  4.67,  5.37, 20.88, 24.66, 28.44 }, sd::DataType::FLOAT32);
    NDArray dLdB('c', { 1, 1, 4, 2 }, { 11.84,   12.68,  39.98,  43.192, 20.65, 22.36, 165.7,   178.4 }, sd::DataType::FLOAT32);

    sd::ops::tensormmul_bp op_bp;

    auto resultsBP = op_bp.evaluate({ &A, &B, &dLdC }, {}, { 3,0,1,2, 3,0,1,2 }, {});

    ASSERT_EQ(ND4J_STATUS_OK, resultsBP.status());

    auto* dLdAbp = resultsBP.at(0);
    auto* dLdBbp = resultsBP.at(1);

    ASSERT_TRUE(dLdA.isSameShape(*dLdAbp));
    ASSERT_TRUE(dLdA.equalsTo(*dLdAbp));

    ASSERT_TRUE(dLdB.isSameShape(*dLdBbp));
    ASSERT_TRUE(dLdB.equalsTo(*dLdBbp));

}
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, TestTensorMmul_BP9) {

    NDArray A('c', { 3, 2, 2, 1 }, { 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3., 3.1, 3.2 }, sd::DataType::FLOAT32);
    NDArray B('c', { 4, 2, 2 ,1 }, { 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4., 4.1, 4.2, 4.3, 4.4, 4.5, 4.6 }, sd::DataType::FLOAT32);
    NDArray dLdC('c', { 3, 1, 4, 1 }, { .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 1.1, 1.2 }, sd::DataType::FLOAT32);

    NDArray dA('c', { 3, 2, 2, 1 }, { 3.9, 4., 4.1, 4.2, 9.82, 10.08, 10.34, 10.6, 15.74, 16.16, 16.58, 17. }, sd::DataType::FLOAT32);
    NDArray dB('c', { 4, 2, 2, 1 }, { 4.07, 4.22, 4.37, 4.52, 4.82, 5., 5.18, 5.36, 5.57, 5.78, 5.99, 6.2, 6.32, 6.56, 6.8,  7.04 }, sd::DataType::FLOAT32);

    sd::ops::tensormmul_bp op_bp;

    auto resultsBP = op_bp.evaluate({ &A, &B, &dLdC }, {}, { 2,1,2, 2,1,2 }, {});

    ASSERT_EQ(ND4J_STATUS_OK, resultsBP.status());

    auto* dLdAbp = resultsBP.at(0);
    auto* dLdBbp = resultsBP.at(1);

    ASSERT_TRUE(dA.isSameShape(*dLdAbp));
    ASSERT_TRUE(dA.equalsTo(*dLdAbp));

    ASSERT_TRUE(dB.isSameShape(*dLdBbp));
    ASSERT_TRUE(dB.equalsTo(*dLdBbp));

}
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, TestTensorMmul_BP10) {

    NDArray A('c', { 1, 2, 2, 3 }, { 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3., 3.1, 3.2 }, sd::DataType::FLOAT32);
    NDArray B('c', { 1, 2, 2 ,4 }, { 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4., 4.1, 4.2, 4.3, 4.4, 4.5, 4.6 }, sd::DataType::FLOAT32);
    NDArray dLdC('c', { 1, 3, 1, 4 }, { .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 1.1, 1.2 }, sd::DataType::FLOAT32);


    NDArray dA('c', { 1, 2, 2, 3 }, { 3.3, 8.5, 13.7, 3.7, 9.54, 15.38, 4.1, 10.58, 17.06, 4.5,  11.62, 18.74 }, sd::DataType::FLOAT32);
    NDArray dB('c', { 1, 2, 2, 4 }, { 3.38, 4.04, 4.7, 5.36, 3.83, 4.58, 5.33, 6.08, 4.28, 5.12, 5.96, 6.8, 4.73, 5.66, 6.59, 7.52 }, sd::DataType::FLOAT32);

    sd::ops::tensormmul_bp op_bp;

    auto resultsBP = op_bp.evaluate({ &A, &B, &dLdC }, {}, { 2,1,2, 2,1,2 }, {});

    ASSERT_EQ(ND4J_STATUS_OK, resultsBP.status());

    auto* dLdAbp = resultsBP.at(0);
    auto* dLdBbp = resultsBP.at(1);

    ASSERT_TRUE(dA.isSameShape(*dLdAbp));
    ASSERT_TRUE(dA.equalsTo(*dLdAbp));

    ASSERT_TRUE(dB.isSameShape(*dLdBbp));
    ASSERT_TRUE(dB.equalsTo(*dLdBbp));

}
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, TestTensorMmul_BP11) {

    NDArray A('c', { 2, 2, 3 }, { 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3., 3.1, 3.2 }, sd::DataType::FLOAT32);
    NDArray B('c', { 2, 2 ,4 }, { 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4., 4.1, 4.2, 4.3, 4.4, 4.5, 4.6 }, sd::DataType::FLOAT32);
    NDArray dLdC('c', { 3, 4 }, { .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 1.1, 1.2 }, sd::DataType::FLOAT32);


    NDArray dA('c', { 2, 2, 3 }, { 3.3, 8.5, 13.7, 3.7, 9.54, 15.38, 4.1, 10.58, 17.06, 4.5,  11.62, 18.74 }, sd::DataType::FLOAT32);
    NDArray dB('c', { 2, 2, 4 }, { 3.38, 4.04, 4.7, 5.36, 3.83, 4.58, 5.33, 6.08, 4.28, 5.12, 5.96, 6.8, 4.73, 5.66, 6.59, 7.52 }, sd::DataType::FLOAT32);

    sd::ops::tensormmul_bp op_bp;

    auto resultsBP = op_bp.evaluate({ &A, &B, &dLdC }, {}, { 2,0,1, 2,0,1 }, {});

    ASSERT_EQ(ND4J_STATUS_OK, resultsBP.status());

    auto* dLdAbp = resultsBP.at(0);
    auto* dLdBbp = resultsBP.at(1);

    ASSERT_TRUE(dA.isSameShape(*dLdAbp));
    ASSERT_TRUE(dA.equalsTo(*dLdAbp));

    ASSERT_TRUE(dB.isSameShape(*dLdBbp));
    ASSERT_TRUE(dB.equalsTo(*dLdBbp));

}
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, TestTensorMmul_BP12) {

    NDArray A('c', { 2, 2, 3 }, { 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3., 3.1, 3.2 }, sd::DataType::FLOAT32);
    NDArray B('c', { 2, 2 ,3 }, { 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4., 4.1, 4.2 }, sd::DataType::FLOAT32);
    NDArray dLdC('c', { 2, 3, 2, 3 }, { .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 1.1, 1.2,
                      1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.1, 2.2, 2.3, 2.4,
                      2.5, 2.6, 2.7, 2.8, 2.9, 3., 3.1, 3.2, 3.3, 3.4, 3.5, 3.6 }, sd::DataType::FLOAT32);

    NDArray dA('c', { 2, 2, 3 }, { 7.66, 20.26, 32.86, 8.29, 21.97, 35.65, 45.46, 58.06, 70.66, 49.33, 63.01, 76.69 }, sd::DataType::FLOAT32);
    NDArray dB('c', { 2, 2, 3 }, { 25.86, 27.36, 28.86, 28.74, 30.42, 32.1, 30.36, 31.86, 33.36, 33.78, 35.46, 37.14 }, sd::DataType::FLOAT32);

    sd::ops::tensormmul_bp op_bp;

    auto resultsBP = op_bp.evaluate({ &A, &B, &dLdC }, {}, { 1,1, 1,1 }, {});

    ASSERT_EQ(ND4J_STATUS_OK, resultsBP.status());

    auto* dLdAbp = resultsBP.at(0);
    auto* dLdBbp = resultsBP.at(1);

    ASSERT_TRUE(dA.isSameShape(*dLdAbp));
    ASSERT_TRUE(dA.equalsTo(*dLdAbp));

    ASSERT_TRUE(dB.isSameShape(*dLdBbp));
    ASSERT_TRUE(dB.equalsTo(*dLdBbp));

}
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, TestTensorMmul_BP13) {

    NDArray A('c', { 3, 2, 2 }, { 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3., 3.1, 3.2 }, sd::DataType::DOUBLE);
    NDArray B('c', { 3, 2, 2 }, { 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4., 4.1, 4.2 }, sd::DataType::DOUBLE);
    NDArray dLdC('c', { 3, 2, 3, 2 }, { .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 1.1, 1.2,
                      1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.1, 2.2, 2.3, 2.4,
                      2.5, 2.6, 2.7, 2.8, 2.9, 3., 3.1, 3.2, 3.3, 3.4, 3.5, 3.6 }, sd::DataType::DOUBLE);

    NDArray dA('c', { 3, 2, 2 }, { 7.79, 20.57, 8.21, 21.71, 33.35, 46.13, 35.21, 48.71, 58.91, 71.69, 62.21, 75.71 }, sd::DataType::DOUBLE);
    NDArray dB('c', { 3, 2, 2 }, { 26.49, 28.02, 28.41, 30.06, 29.55, 31.08, 31.71, 33.36, 32.61, 34.14, 35.01, 36.66 }, sd::DataType::DOUBLE);

    sd::ops::tensormmul_bp op_bp;

    auto resultsBP = op_bp.evaluate({ &A, &B, &dLdC }, {}, { 1,1, 1,1 }, {});

    ASSERT_EQ(ND4J_STATUS_OK, resultsBP.status());

    auto* dLdAbp = resultsBP.at(0);
    auto* dLdBbp = resultsBP.at(1);

    ASSERT_TRUE(dA.isSameShape(*dLdAbp));
    ASSERT_TRUE(dA.equalsTo(*dLdAbp));

    ASSERT_TRUE(dB.isSameShape(*dLdBbp));
    ASSERT_TRUE(dB.equalsTo(*dLdBbp));

}
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, TestTensorMmul_BP14) {

    NDArray A('c', { 2, 2, 2, 2 }, { 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3., 3.1, 3.2, 3.3, 3.4, 3.5, 3.6 }, sd::DataType::DOUBLE);

    NDArray B('c', { 2, 2, 2, 2 }, { 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4., 4.1, 4.2, 4.3, 4.4, 4.5, 4.6 }, sd::DataType::DOUBLE);

    NDArray dLdC('c', { 2, 2, 2, 2, 2, 2 }, { .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 1.1, 1.2,
                      1.3, 1.4, 1.5, 1.6, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 1.1, 1.2,
                      1.3, 1.4, 1.5, 1.6, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 1.1, 1.2,
                      1.3, 1.4, 1.5, 1.6, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 1.1, 1.2,
                      1.3, 1.4, 1.5, 1.6 }, sd::DataType::DOUBLE);

    NDArray dA('c', { 2, 2, 2, 2 }, { 13.88, 37.24, 13.88, 37.24, 15.32, 41.24, 15.32, 41.24, 13.88, 37.24, 13.88, 37.24, 15.32, 41.24, 15.32, 41.24 }, sd::DataType::DOUBLE);
    NDArray dB('c', { 2, 2, 2, 2 }, { 10.76, 12.88, 15., 17.12, 12.36, 14.8, 17.24, 19.68, 19.24, 21.36, 23.48, 25.6, 22.12, 24.56, 27., 29.44 }, sd::DataType::DOUBLE);

    sd::ops::tensormmul_bp op_bp;

    auto resultsBP = op_bp.evaluate({ &A, &B, &dLdC }, {}, { 1,1, 1,1 }, {});

    ASSERT_EQ(ND4J_STATUS_OK, resultsBP.status());

    auto* dLdAbp = resultsBP.at(0);
    auto* dLdBbp = resultsBP.at(1);

    ASSERT_TRUE(dA.isSameShape(*dLdAbp));
    ASSERT_TRUE(dA.equalsTo(*dLdAbp));

    ASSERT_TRUE(dB.isSameShape(*dLdBbp));
    ASSERT_TRUE(dB.equalsTo(*dLdBbp));

}
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, TestTensorMmul_BP15) {

    NDArray A('c', { 2, 2, 3 }, { 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12. }, sd::DataType::FLOAT32);
    NDArray B('f', { 2, 2, 3 }, { 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12. }, sd::DataType::FLOAT32);

    NDArray dLdC('f', { 2, 2 }, { 23.0, 24.44, 2.0, 26. }, sd::DataType::FLOAT32);

    NDArray dA('c', { 2, 2, 3 }, { 27., 127., 227., 77., 177., 277., 76.44, 278.20001, 479.96002, 177.32, 379.08001, 580.839966 }, sd::DataType::FLOAT32);
    NDArray dB('f', { 2, 2, 3 }, { 194.08, 184., 336.4, 268., 241.52, 212., 383.839996, 296., 288.96002, 240., 431.27999, 324. }, sd::DataType::FLOAT32);

    sd::ops::tensormmul_bp op;
    auto results = op.evaluate({ &A, &B, &dLdC }, {}, { 2,1,2,2,1,2 });

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto* dLdA = results.at(0);
    auto* dLdB = results.at(1);

    ASSERT_TRUE(dA.isSameShape(*dLdA));
    ASSERT_TRUE(dA.equalsTo(*dLdA));

    ASSERT_TRUE(dB.isSameShape(*dLdB));
    ASSERT_TRUE(dB.equalsTo(*dLdB));

}
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, TestTensorMmul_BP16) {

    NDArray A('f', { 2, 2, 3 }, { 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12. }, sd::DataType::DOUBLE);
    NDArray B('c', { 2, 2, 3 }, { 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12. }, sd::DataType::DOUBLE);

    NDArray dLdC('c', { 2, 2 }, sd::DataType::DOUBLE);

    const OpArgsHolder argsHolderFF({ &A, &B }, {}, { 2,1,2, 2,1,2 });
    const OpArgsHolder argsHolderBP({ &A, &B, &dLdC }, {}, { 2,1,2, 2,1,2 });

    sd::ops::tensormmul op;
    sd::ops::tensormmul_bp op_bp;

    const bool isGradCorrect = GradCheck::checkGrad(op, op_bp, argsHolderFF, argsHolderBP, {1,0});
    ASSERT_TRUE(isGradCorrect);
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, TestTensorMmul_BP17) {

    NDArray A('f', { 2, 2, 3 }, { 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12. }, sd::DataType::DOUBLE);
    NDArray B('f', { 2, 2, 3 }, { 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12. }, sd::DataType::DOUBLE);

    NDArray dLdC('c', { 2, 2 }, sd::DataType::DOUBLE);

    const OpArgsHolder argsHolderFF({ &A, &B }, {}, { 2,1,2, 2,1,2 });
    const OpArgsHolder argsHolderBP({ &A, &B, &dLdC }, {}, { 2,1,2, 2,1,2 });

    sd::ops::tensormmul op;
    sd::ops::tensormmul_bp op_bp;

    const bool isGradCorrect = GradCheck::checkGrad(op, op_bp, argsHolderFF, argsHolderBP, { 1,0 });
    ASSERT_TRUE(isGradCorrect);
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, gru_1) {

    const int sL = 3;
    const int bS = 2;
    const int nIn = 5;
    const int nOut = 4;


    NDArray x('c', {sL, bS, nIn}, {0.5,  1. ,  1.5,  2. ,  2.5, 3. ,  3.5,  4. ,  4.5,  5. ,  5.5,  6. ,  6.5,  7. ,  7.5, 8. ,  8.5,  9. ,  9.5, 10. ,  10.5, 11. , 11.5, 12. , 12.5, 13. , 13.5, 14. , 14.5, 15.}, sd::DataType::FLOAT32);
    NDArray hI('c', {bS, nOut}, {-3,-2,-1,0,1,2,3,4}, sd::DataType::FLOAT32);
    NDArray Wx('c', {nIn, 3*nOut}, sd::DataType::FLOAT32);
    NDArray Wh('c', {nOut, 3*nOut}, sd::DataType::FLOAT32);
    NDArray b('c', {3*nOut}, sd::DataType::FLOAT32);

    NDArray expH('c', {sL, bS, nOut}, {-1.681847, -1.062565, -0.443283,  0.175998,0.837823,  1.488041,  2.13826 ,  2.788478, -0.888747, -0.491826, -0.094907,  0.302014,
                  0.751355,  1.182715,  1.614075,  2.045434, -0.388876, -0.126716,  0.135444,  0.397604,0.710558,  1.002922,  1.295287,  1.587651}, sd::DataType::FLOAT32);

    Wx = 0.003;
    Wh = 0.006;
    b  = 0.5;

    NDArray dLdC('c', { 2, 2 }, sd::DataType::DOUBLE);

    sd::ops::gru op;
    auto results = op.evaluate({&x, &hI, &Wx, &Wh, &b}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto* h = results.at(0);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, sqrtm_1) {

    NDArray x1('c', {1,1}, {4.}, sd::DataType::DOUBLE);
    NDArray x2('c', {2,2}, {1.3,2,0.3,.5}, sd::DataType::DOUBLE);
    NDArray x3('c', {3,3}, {0.5 ,-0.4 ,1.2 ,-2.8 ,-0.2 ,-2.1 ,-2.4 ,-2.0 ,1.1}, sd::DataType::DOUBLE);
    NDArray x4('c', {4,4}, {0.33 ,-7.25 ,1.71 ,6.20 ,1.34 ,5.38 ,-2.76 ,-8.51 ,7.59 ,3.44 ,2.24 ,-6.82 ,-1.15 ,4.80 ,-4.67 ,2.14}, sd::DataType::DOUBLE);
    NDArray x5('c', {5,5}, {2.4 ,0.3 ,0.0 ,1.1 ,1.8 ,0.1 ,1.7 ,2.7 ,1.5 ,2.6 ,0.6 ,2.1 ,2.2 ,1.0 ,0.2 ,1.2 ,2.8 ,1.9 ,0.8 ,2.0 ,0.5 ,1.6 ,0.9 ,1.4 ,2.5}, sd::DataType::DOUBLE);

    NDArray exp1('c', {1,1}, {2.}, sd::DataType::DOUBLE);
    NDArray exp2('c', {2,2}, {1.0163674, 1.3341597,0.200124, 0.4827035}, sd::DataType::DOUBLE);
    NDArray exp3('c', {3,3}, {6.5692188, 2.6273616,-0.1387864,-16.8404762,-7.0296495, 0.9204148,-11.4664296,-5.834273 , 2.2087478}, sd::DataType::DOUBLE);
    NDArray exp4('c', {4,4}, {1.161387 ,-1.9343154, 0.230372 , 0.8660897,0.80588  , 3.4045446,-1.0152824,-2.0369467,2.2589629, 1.9674252, 1.5109997,-1.4283141,0.0226356, 1.3032279,-1.00396  , 1.8278487}, sd::DataType::DOUBLE);
    NDArray exp5('c', {5,5}, {1.4175046,-0.4425298, 0.1846149, 0.3166522, 0.9140631,-0.1929139, 0.2889113, 1.4045273, 0.2600026, 1.552021 , 0.1372758, 0.5703854, 1.3336126, 0.3869317,-0.082492 ,
                                0.8607272, 3.1792474,-0.9499947, 0.8541668,-1.4243879, 0.0081136,-0.0622248, 0.4534325, 0.4641865, 1.8132138}, sd::DataType::DOUBLE);

    sd::ops::sqrtm op;

    auto results = op.evaluate({&x1}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, results.status());
    ASSERT_TRUE(exp1.isSameShape(results.at(0)));
    ASSERT_TRUE(exp1.equalsTo(results.at(0)));

    results = op.evaluate({&x2}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, results.status());
    ASSERT_TRUE(exp2.isSameShape(results.at(0)));
    ASSERT_TRUE(exp2.equalsTo(results.at(0)));

    results = op.evaluate({&x3}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, results.status());
    ASSERT_TRUE(exp3.isSameShape(results.at(0)));
    ASSERT_TRUE(exp3.equalsTo(results.at(0)));

    results = op.evaluate({&x4}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, results.status());
    ASSERT_TRUE(exp4.isSameShape(results.at(0)));
    ASSERT_TRUE(exp4.equalsTo(results.at(0)));

    results = op.evaluate({&x5}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, results.status());
    ASSERT_TRUE(exp5.isSameShape(results.at(0)));
    ASSERT_TRUE(exp5.equalsTo(results.at(0)));
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests15, sqrtm_2) {

    NDArray x('c', {10,10}, {-0.3 ,2.7 ,4.9 ,7.0 ,7.3 ,-1.3 ,0.5 ,9.9 ,-9.4 ,8.4 ,2.2 ,5.2 ,7.6 ,1.2 ,2.0 ,-3.8 ,2.1 ,6.1 ,1.6 ,6.9 ,5.1 ,5.3 ,6.4 ,8.7 ,0.1 ,8.5 ,
                               3.3 ,1.0 ,6.8 ,0.4 ,0.7 ,3.2 ,7.4 ,6.7 ,1.1 ,7.2 ,6.0 ,7.5 ,9.7 ,5.4 ,9.0 ,6.3 ,0.0 ,4.5 ,8.3 ,7.9 ,3.0 ,6.5 ,0.6 ,8.0 ,9.5 ,3.6 ,1.9 ,6.2 ,0.9 ,4.0 ,4.1 ,
                               8.1 ,3.9 ,4.3 ,4.7 ,3.7 ,3.4 ,5.8 ,10.0 ,8.6 ,9.3 ,9.1 ,4.6 ,1.4 ,7.8 ,1.5 ,7.7 ,4.2 ,9.6 ,8.2 ,-7.1 ,5.7 ,5.5 ,2.6 ,8.8 ,2.9 ,0.2 ,5.6 ,-2.5 ,8.9 ,2.8 ,0.8 ,1.5 ,3.1 ,3.5 ,4.4 ,2.4 ,9.2 ,-4.8 ,1.7 ,6.6 ,9.8 ,1.8 ,5.9}, sd::DataType::DOUBLE);

    NDArray expZ('c', {10,10}, {1.2779038,  0.0333321,  0.8215617,  0.5736392,  1.3973911, -1.1757741,0.1990005,  1.5893778, -3.0159568,  2.5829108,0.5692253,  2.219431 ,  1.022612 , -0.3131795, -0.1957848, -1.7805065,
                                0.6668489,  1.1968921,  0.9781974,  1.2007764,0.7028634,  0.7496937,  2.2511438,  2.1945378,  0.2559353,  2.8948612,-0.4306994, -0.9922216,  0.3884369, -1.4174481,
                                -1.6060233,  0.1571057,  1.432471 ,  0.4508346,  0.0618069, -2.4511742,2.0641709,  2.4751085,  1.84787  ,  3.4146313,0.7774219,  0.768369 , -0.1417226, -0.3970577,  2.9512879,  0.5474537,
                                0.4991412,  0.7604095,  0.4523091,  1.7813704,2.5998339,  0.9402402, -0.82775  ,  2.3637147, -0.6394584,  4.6181937,-0.1762181, -0.2820475,  0.9280713, -2.1876918,
                                0.1576249,  0.336376 ,  0.2017592,  0.851786 ,  1.3542577,  1.2752901,2.9718476,  1.1102557,  0.0067319, -0.2652283,0.8839235, -0.2637131,  1.5687876,  0.5156139,  1.9015886,  0.9087172,
                                -1.5607482,  2.4216275,  1.0399745, -0.4930439,1.3044354,  0.1690006,  0.2106909, -0.2683631, -0.4193939,  1.0233265,0.4571777, -0.2024148,  2.3564855,  1.0442339,
                                1.1073322,  1.0728525, -0.5917566,  2.2267418, -1.6096582,  2.0685315,0.6800798,  0.4451858, -0.4048465,  1.2347676}, sd::DataType::DOUBLE);
    sd::ops::sqrtm op;

    auto results = op.evaluate({&x}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, results.status());
    ASSERT_TRUE(expZ.isSameShape(results.at(0)));
    ASSERT_TRUE(expZ.equalsTo(results.at(0)));
}
