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
// @author raver119@gmail.com
//

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <NDArray.h>
#include <ops/ops.h>
#include <GradCheck.h>
#include <array>


using namespace nd4j;


class DeclarableOpsTests16 : public testing::Test {
public:

    DeclarableOpsTests16() {
        printf("\n");
        fflush(stdout);
    }
};

TEST_F(DeclarableOpsTests16, scatter_upd_1) {
    auto x = NDArrayFactory::create<float>('c', {3}, {1, 1, 1});
    auto y = NDArrayFactory::create<int>(0);
    auto w = NDArrayFactory::create<float>(3.0f);
    auto e = NDArrayFactory::create<float>('c', {3}, {3.f, 1.f, 1.f});

    nd4j::ops::scatter_upd op;
    auto result = op.execute({&x, &y, &w}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests16, scatter_upd_2) {

    NDArray x('c', {10, 3}, nd4j::DataType::FLOAT32);
    NDArray indices('c', {2}, {2,5}, nd4j::DataType::INT32);
    NDArray updates('c', {2, 3}, {100,101,102,  200,201,202}, nd4j::DataType::FLOAT32);
    NDArray e('c', {10, 3}, {1,2,3, 4,5,6, 100,101,102, 10,11,12, 13,14,15, 200,201,202, 19,20,21, 22,23,24, 25,26,27, 28,29,30}, nd4j::DataType::FLOAT32);

    x.linspace(1);

    nd4j::ops::scatter_upd op;
    auto result = op.execute({&x, &indices, &updates}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests16, scatter_upd_3) {

    NDArray x('c', {10, 3}, nd4j::DataType::FLOAT32);
    NDArray indices('c', {2}, {20,5}, nd4j::DataType::INT32);
    NDArray updates('c', {2, 3}, {100,101,102,  200,201,202}, nd4j::DataType::FLOAT32);
    NDArray output('c', {10, 3}, nd4j::DataType::FLOAT32);

    nd4j::ops::scatter_upd op;
    ASSERT_ANY_THROW(op.execute({&x, &indices, &updates}, {&output}, {}, {}, {true, true}));
}

TEST_F(DeclarableOpsTests16, test_size_dtype_1) {
    auto x = NDArrayFactory::create<float>('c', {3}, {1, 1, 1});
    auto z = NDArrayFactory::create<float>(0.0f);
    auto e = NDArrayFactory::create<float>(3.0f);

    nd4j::ops::size op;
    auto status = op.execute({&x}, {&z}, {}, {}, {});
    ASSERT_EQ(Status::OK(), status);

    ASSERT_EQ(e, z);
}

TEST_F(DeclarableOpsTests16, test_empty_noop_1) {
    auto z = NDArrayFactory::empty<Nd4jLong>();

    nd4j::ops::noop op;
    auto status = op.execute({}, {&z}, {}, {}, {});
    ASSERT_EQ(Status::OK(), status);
}

TEST_F(DeclarableOpsTests16, test_empty_noop_2) {
    auto z = NDArrayFactory::empty<Nd4jLong>();

    Context ctx(1);
    ctx.setOutputArray(0, z.buffer(), z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo());

    nd4j::ops::noop op;
    auto status = op.execute(&ctx);

    ASSERT_EQ(Status::OK(), status);
}

TEST_F(DeclarableOpsTests16, test_svd_1) {
    auto x = NDArrayFactory::create<float>('c', {3, 3}, {0.7787856f, 0.80119777f, 0.72437465f, 0.23089433f, 0.72714126f, 0.18039072f,0.50563407f, 0.89252293f, 0.5461209f});
    auto z = NDArrayFactory::create<float>('c', {3});

    nd4j::ops::svd op;
    auto status = op.execute({&x}, {&z}, {}, {0, 0, 16}, {});

    ASSERT_EQ(Status::OK(), status);
}

TEST_F(DeclarableOpsTests16, test_hamming_distance_1) {
    auto x = NDArrayFactory::create<Nd4jLong>({37, 37, 37});
    auto y = NDArrayFactory::create<Nd4jLong>({8723, 8723, 8723});
    auto e = NDArrayFactory::create<Nd4jLong>(18);

    nd4j::ops::bits_hamming_distance op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests16, test_knn_mindistance_1) {
    auto input = NDArrayFactory::create<float>('c', {512});
    auto low = NDArrayFactory::create<float>('c', {512});
    auto high = NDArrayFactory::create<float>('c', {512});

    auto output = NDArrayFactory::create<float>(0.0f);

    input.linspace(1.0);
    low.linspace(1.0);
    high.linspace(1.0);

    nd4j::ops::knn_mindistance op;
    auto result = op.execute({&input, &low, &high}, {&output}, {}, {}, {});
    ASSERT_EQ(Status::OK(), result);
}

TEST_F(DeclarableOpsTests16, test_empty_cast_1) {
    auto x = NDArrayFactory::create<bool>('c', {1, 0, 2});
    auto e = NDArrayFactory::create<Nd4jLong>('c', {1, 0, 2});

    nd4j::ops::cast op;
    auto result = op.execute({&x}, {}, {10});
    ASSERT_EQ(Status::OK(), result->status());
    ASSERT_EQ(e, *result->at(0));

    delete result;
}

TEST_F(DeclarableOpsTests16, test_range_1) {
    nd4j::ops::range op;
    auto z = NDArrayFactory::create<float>('c', {200});

    Context ctx(1);
    ctx.setTArguments({-1.0, 1.0, 0.01});
    ctx.setOutputArray(0, &z);

    auto status = op.execute(&ctx);
    ASSERT_EQ(Status::OK(), status);
}

TEST_F(DeclarableOpsTests16, test_range_2) {
    nd4j::ops::range op;
    auto z = NDArrayFactory::create<float>('c', {200});

    double tArgs[] = {-1.0, 1.0, 0.01};

    auto shapes = ::calculateOutputShapes2(nullptr, op.getOpHash(), nullptr, nullptr, 0, tArgs, 3, nullptr, 0, nullptr, 0);
    shape::printShapeInfoLinear("Result", shapes->at(0));
    ASSERT_TRUE(shape::shapeEquals(z.shapeInfo(), shapes->at(0)));

    delete shapes;
}