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
// Created by raver119 on 16.10.2017.
//

#include "testlayers.h"
#include <NDArray.h>
#include <ops/declarable/LegacyTransformOp.h>
#include <ops/declarable/LegacyPairwiseTransformOp.h>
#include <ops/declarable/LegacyScalarOp.h>
#include <ops/declarable/LegacyReduceOp.h>
#include <ops/declarable/LegacyIndexReduceOp.h>
#include <ops/declarable/LegacyBroadcastOp.h>

using namespace nd4j;
using namespace nd4j::ops;

class LegacyOpsTests : public testing::Test {

};


TEST_F(LegacyOpsTests, TransformTests_1) {
    auto x = NDArrayFactory::_create<float>('c', {5, 5});
    x.assign(1.0);

    auto exp = NDArrayFactory::_create<float>('c', {5, 5});
    exp.assign(-1.0);

    nd4j::ops::LegacyTransformOp op(6);
    op.execute({&x}, {&x}, {}, {});

    ASSERT_TRUE(x.equalsTo(&exp));
}

TEST_F(LegacyOpsTests, TransformTests_2) {
    auto x = NDArrayFactory::_create<float>('c', {5, 5});
    x.assign(1.0);

    auto exp = NDArrayFactory::_create<float>('c', {5, 5});
    exp.assign(-1.0);

    nd4j::ops::LegacyTransformOp op(6);
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(1, result->size());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(LegacyOpsTests,  Reciprocal_1) {
    auto x = NDArrayFactory::_create<float>('c', {5, 5});
    x.assign(2.0f);

    auto ethalon = NDArrayFactory::_create<float>('c', {5, 5});
    ethalon.assign(0.5f);

    nd4j::ops::LegacyTransformOp op(94);
    Nd4jStatus status = op.execute({&x}, {&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(ethalon.equalsTo(&x));
    
}

TEST_F(LegacyOpsTests,  PWT_Tests_1) {
    auto x = NDArrayFactory::_create<float>('c', {5, 5});
    x.assign(2.0);

    auto y = NDArrayFactory::_create<float>('c', {5, 5});
    y.assign(3.0);

    auto exp = NDArrayFactory::_create<float>('c', {5, 5});
    exp.assign(6.0);

    nd4j::ops::LegacyPairwiseTransformOp op(6);
    Nd4jStatus status = op.execute({&x, &y}, {&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, status);

    ASSERT_TRUE(exp.equalsTo(&x));


}

TEST_F(LegacyOpsTests,  PWT_Tests_2) {
    auto x = NDArrayFactory::_create<float>('c', {5, 5});
    x.assign(2.0);

    auto y = NDArrayFactory::_create<float>('c', {5, 5});
    y.assign(3.0);

    auto exp = NDArrayFactory::_create<float>('c', {5, 5});
    exp.assign(6.0);

    nd4j::ops::LegacyPairwiseTransformOp op(6);
    auto result = op.execute({&x, &y}, {}, {});

    auto z = result->at(0);

    //z->printBuffer("Z");
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(LegacyOpsTests, Scalar_Test_1) {
    auto x = NDArrayFactory::_create<float>('c', {5, 5});
    x.assign(2.0);

    auto exp = NDArrayFactory::_create<float>('c', {5, 5});
    exp.assign(7.0);

    nd4j::ops::LegacyScalarOp op(0);
    op.execute({&x}, {&x}, {5.0}, {});

    ASSERT_TRUE(exp.equalsTo(&x));
}

TEST_F(LegacyOpsTests, Scalar_Test_2) {
    auto x = NDArrayFactory::_create<float>('c', {5, 5});
    x.assign(2.0);

    auto exp = NDArrayFactory::_create<float>('c', {5, 5});
    exp.assign(7.0);

    nd4j::ops::LegacyScalarOp op(0, 5.0f);
    auto result = op.execute({&x}, {}, {});

    auto z = result->at(0);
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(LegacyOpsTests, ReduceTests_1) {
    auto x = NDArrayFactory::_create<float>('c', {5, 5});
    x.assign(1.0);

    nd4j::ops::LegacyReduceOp op(1);

    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(1, result->size());

    auto z = result->at(0);

    ASSERT_TRUE(z->isScalar());
    ASSERT_NEAR(x.sumNumber().e<float>(0), z->e<float>(0), 1e-5f);

    delete result;
}


TEST_F(LegacyOpsTests, ReduceTests_2) {
    auto x = NDArrayFactory::_create<float>('c', {5, 5});
    x.assign(1.0);

    nd4j::ops::LegacyReduceOp op(1);

    auto result = op.execute({&x}, {}, {1});

    ASSERT_EQ(1, result->size());

    auto z = result->at(0);

    auto exp = x.reduceAlongDimension(reduce::Sum, {1});

    ASSERT_TRUE(exp->isSameShape(z));
    ASSERT_TRUE(exp->equalsTo(z));

    delete result;
    delete exp;
}


TEST_F(LegacyOpsTests, ReduceTests_3) {
    auto x = NDArrayFactory::_create<float>('c', {3, 5});
    x.linspace(1);
    auto indices = NDArrayFactory::_create<float>('c', {1,1}, {1});


    nd4j::ops::LegacyReduceOp op(1);
    auto result = op.execute({&x, &indices}, {}, {});
    auto z = result->at(0);
    auto exp = x.reduceAlongDims(reduce::Sum,{1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));
    
    delete result;
}


TEST_F(LegacyOpsTests, ReduceTests_4) {
    auto x = NDArrayFactory::_create<float>('c', {2, 3, 5});
    x.linspace(1);
    auto indices = NDArrayFactory::_create<float>('c', {1,1}, {1});


    nd4j::ops::LegacyReduceOp op(1);
    auto result = op.execute({&x, &indices}, {}, {1});
    auto z = result->at(0);
    auto exp = x.reduceAlongDims(reduce::Sum, {1}, true);

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(LegacyOpsTests, IndexReduceTests_1) {
    auto x = NDArrayFactory::_create<float>('c', {5, 5});
    x.linspace(1);

    nd4j::ops::LegacyIndexReduceOp op(0);

    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(1, result->size());

    auto z = result->at(0);

    ASSERT_TRUE(z->isScalar());
    ASSERT_EQ(24, z->e<int>(0));

    delete result;
}


TEST_F(LegacyOpsTests, IndexReduceTests_2) {
    auto x = NDArrayFactory::_create<float>('c', {5, 5});
    x.linspace(1);

    nd4j::ops::LegacyIndexReduceOp op(0);

    auto result = op.execute({&x}, {}, {1});

    ASSERT_EQ(1, result->size());

    auto z = result->at(0);

    ASSERT_EQ(4, z->e<int>(0));
    ASSERT_EQ(4, z->e<int>(1));
    ASSERT_EQ(4, z->e<int>(2));
    ASSERT_EQ(4, z->e<int>(3));
    ASSERT_EQ(4, z->e<int>(4));

    delete result;
}


TEST_F(LegacyOpsTests, BroadcastingTests_1) {
    auto x = NDArrayFactory::_create<double>('c', {5, 5});
    x.assign(0.0f);

    auto row = NDArrayFactory::_create<double>('c', {1, 5});
    row.linspace(1);

    nd4j::ops::LegacyBroadcastOp op(0);
    Nd4jStatus status = op.execute({&x, &row}, {&x}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, status);

    auto list = x.allTensorsAlongDimension({1});

    for (int e = 0; e < list->size(); e++)
        ASSERT_TRUE(row.equalsTo(list->at(e)));

    delete list;
}

TEST_F(LegacyOpsTests, PowDerivative_1) {
    auto x = NDArrayFactory::_create<float>('c', {5, 5});
    auto exp = NDArrayFactory::_create<float>('c', {5, 5});
    x.assign(3.f);
    exp.assign(6.f);

    float p = 2.0f;

    x.applyScalar(scalar::PowDerivative, p);

    ASSERT_TRUE(exp.equalsTo(&x));
}