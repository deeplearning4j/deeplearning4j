/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
    auto w = NDArrayFactory::create<int>(10);
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
