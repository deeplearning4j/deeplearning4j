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

#include "testlayers.h"
#include <NDArray.h>
#include <NDArrayFactory.h>
#include <Context.h>
#include <Node.h>
#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <graph/LaunchContext.h>
#include <specials_cuda.h>
#include <TAD.h>

#include <cuda.h>
#include <cuda_launch_config.h>

using namespace nd4j;
using namespace nd4j::graph;

class NDArrayConstructorsTests : public testing::Test {
public:

};

TEST_F(NDArrayConstructorsTests, test_constructor_1) {
    auto x = NDArrayFactory::empty<float>();

    ASSERT_TRUE(x->buffer() == nullptr);
    ASSERT_TRUE(x->specialBuffer() == nullptr);

    ASSERT_FALSE(x->shapeInfo() == nullptr);
    ASSERT_FALSE(x->specialShapeInfo() == nullptr);

    delete x;
}

TEST_F(NDArrayConstructorsTests, test_constructor_2) {
    auto x = NDArrayFactory::vector<float>(5, 1.0f);


    ASSERT_FALSE(x->buffer() == nullptr);
    ASSERT_FALSE(x->specialBuffer() == nullptr);

    ASSERT_FALSE(x->shapeInfo() == nullptr);
    ASSERT_FALSE(x->specialShapeInfo() == nullptr);

    delete x;
}

TEST_F(NDArrayConstructorsTests, test_constructor_3) {
    auto x = NDArrayFactory::create<float>('c',{5, 5});

    ASSERT_TRUE(x.buffer() == nullptr);
    ASSERT_FALSE(x.specialBuffer() == nullptr);

    ASSERT_FALSE(x.shapeInfo() == nullptr);
    ASSERT_FALSE(x.specialShapeInfo() == nullptr);
}

TEST_F(NDArrayConstructorsTests, test_constructor_4) {
    auto x = NDArrayFactory::create(nd4j::DataType::FLOAT32, 1.0f);

    ASSERT_FALSE(x.buffer() == nullptr);
    ASSERT_FALSE(x.specialBuffer() == nullptr);

    ASSERT_FALSE(x.shapeInfo() == nullptr);
    ASSERT_FALSE(x.specialShapeInfo() == nullptr);
}

TEST_F(NDArrayConstructorsTests, test_constructor_5) {
    auto x = NDArrayFactory::create<double>('c',{2, 2}, {1, 2, 3, 4});

    ASSERT_FALSE(x.buffer() == nullptr);
    ASSERT_FALSE(x.specialBuffer() == nullptr);

    ASSERT_FALSE(x.shapeInfo() == nullptr);
    ASSERT_FALSE(x.specialShapeInfo() == nullptr);
}

TEST_F(NDArrayConstructorsTests, test_constructor_6) {
    auto x = NDArrayFactory::create<double>('c', {2, 2}, {1, 2, 3, 4});
    NDArray y(x);

    ASSERT_TRUE(y.buffer() == nullptr);
    ASSERT_FALSE(y.specialBuffer() == nullptr);

    ASSERT_FALSE(y.shapeInfo() == nullptr);
    ASSERT_FALSE(y.specialShapeInfo() == nullptr);
}