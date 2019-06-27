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
    int inOutH = 35;
    int inOutW = 35;
    int inOutC = 192;

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
    auto x = NDArrayFactory::create<float>('c', {5}, {1., 1., 1., 1., 1.});
    auto eps = NDArrayFactory::create<float>('c', {5}, {0., 0., 0., 0., 0.});

    nd4j::ops::standardize_bp op;
    auto result = op.execute({&x, &eps}, {}, {0}, {});
    ASSERT_EQ(Status::OK(), result->status());
    delete result;
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
    auto x = NDArrayFactory::create<float>('c', {1, 5}, {1., 2., 3., 4., 5.});
    auto g = NDArrayFactory::create<float>('c', {1, 5}, {1., 2., 3., 4., 5.});
    auto b = NDArrayFactory::create<float>('c', {1, 5}, {1., 2., 3., 4., 5.});

    nd4j::ops::layer_norm op;
    auto result = op.execute({&x, &g, &b}, {}, {0}, {});
    ASSERT_EQ(Status::OK(), result->status());
    delete result;
}

TEST_F(DeclarableOpsTests15, Test_layer_norm_bp_1) {
    auto x = NDArrayFactory::create<float>('c', {1, 5}, {1., 2., 3., 4., 5.});
    auto g = NDArrayFactory::create<float>('c', {1, 5}, {1., 2., 3., 4., 5.});
    auto b = NDArrayFactory::create<float>('c', {1, 5}, {1., 2., 3., 4., 5.});
    auto eps = NDArrayFactory::create<float>('c', {1, 5}, {0., 0., 0., 0., 0.});

    nd4j::ops::layer_norm_bp op;
    auto result = op.execute({&x, &g, &b, &eps}, {}, {0}, {});
    ASSERT_EQ(Status::OK(), result->status());
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

    z->printIndexedBuffer("Z");

    delete result;
}
