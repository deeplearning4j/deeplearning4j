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

using namespace nd4j;
using namespace nd4j::graph;

class NDArrayConstructorsTests : public testing::Test {
public:

};

TEST_F(NDArrayConstructorsTests, test_constructor_1) {
    auto x = NDArrayFactory::empty_<float>();

    ASSERT_TRUE(x->buffer() == nullptr);
    ASSERT_TRUE(x->specialBuffer() == nullptr);

    ASSERT_FALSE(x->shapeInfo() == nullptr);
    ASSERT_FALSE(x->specialShapeInfo() == nullptr);

    ASSERT_TRUE(x->isActualOnDeviceSide());
    ASSERT_TRUE(x->isActualOnHostSide());

    delete x;
}

TEST_F(NDArrayConstructorsTests, test_constructor_2) {
    auto x = NDArrayFactory::vector<float>(5, 1.0f);


    ASSERT_FALSE(x->buffer() == nullptr);
    ASSERT_FALSE(x->specialBuffer() == nullptr);

    ASSERT_FALSE(x->shapeInfo() == nullptr);
    ASSERT_FALSE(x->specialShapeInfo() == nullptr);

    ASSERT_TRUE(x->isActualOnDeviceSide());
    ASSERT_FALSE(x->isActualOnHostSide());

    delete x;
}

TEST_F(NDArrayConstructorsTests, test_constructor_3) {
    auto x = NDArrayFactory::create<float>('c',{5, 5});

    ASSERT_TRUE(x.buffer() == nullptr);
    ASSERT_FALSE(x.specialBuffer() == nullptr);

    ASSERT_FALSE(x.shapeInfo() == nullptr);
    ASSERT_FALSE(x.specialShapeInfo() == nullptr);

    ASSERT_TRUE(x.isActualOnDeviceSide());
    ASSERT_FALSE(x.isActualOnHostSide());
}

TEST_F(NDArrayConstructorsTests, test_constructor_4) {
    auto x = NDArrayFactory::create(nd4j::DataType::FLOAT32, 1.0f);

    ASSERT_FALSE(x.buffer() == nullptr);
    ASSERT_FALSE(x.specialBuffer() == nullptr);

    ASSERT_FALSE(x.shapeInfo() == nullptr);
    ASSERT_FALSE(x.specialShapeInfo() == nullptr);

    ASSERT_TRUE(x.isActualOnDeviceSide());
    ASSERT_TRUE(x.isActualOnHostSide());
}

TEST_F(NDArrayConstructorsTests, test_constructor_5) {
    auto x = NDArrayFactory::create<double>('c',{2, 2}, {1, 2, 3, 4});

    ASSERT_FALSE(x.buffer() == nullptr);
    ASSERT_FALSE(x.specialBuffer() == nullptr);

    ASSERT_FALSE(x.shapeInfo() == nullptr);
    ASSERT_FALSE(x.specialShapeInfo() == nullptr);

    ASSERT_TRUE(x.isActualOnDeviceSide());
    ASSERT_TRUE(x.isActualOnHostSide());
}

TEST_F(NDArrayConstructorsTests, test_constructor_6) {
    auto x = NDArrayFactory::create<double>('c', {2, 2}, {1, 2, 3, 4});
    NDArray y(x);

    ASSERT_TRUE(y.buffer() == nullptr);
    ASSERT_FALSE(y.specialBuffer() == nullptr);

    ASSERT_FALSE(y.shapeInfo() == nullptr);
    ASSERT_FALSE(y.specialShapeInfo() == nullptr);

    ASSERT_TRUE(y.isActualOnDeviceSide());
    ASSERT_FALSE(y.isActualOnHostSide());
}

TEST_F(NDArrayConstructorsTests, test_constructor_7) {
    auto x = NDArrayFactory::create<float>(1.0f);

    ASSERT_FALSE(x.buffer() == nullptr);
    ASSERT_FALSE(x.specialBuffer() == nullptr);

    ASSERT_FALSE(x.shapeInfo() == nullptr);
    ASSERT_FALSE(x.specialShapeInfo() == nullptr);

    ASSERT_TRUE(x.isActualOnDeviceSide());
    ASSERT_TRUE(x.isActualOnHostSide());
}

TEST_F(NDArrayConstructorsTests, test_constructor_8) {
    auto x = NDArrayFactory::create_<double>('c',{2, 2}, {1, 2, 3, 4});

    ASSERT_FALSE(x->buffer() == nullptr);
    ASSERT_FALSE(x->specialBuffer() == nullptr);

    ASSERT_FALSE(x->shapeInfo() == nullptr);
    ASSERT_FALSE(x->specialShapeInfo() == nullptr);

    ASSERT_TRUE(x->isActualOnDeviceSide());
    ASSERT_TRUE(x->isActualOnHostSide());

    delete x;
}

TEST_F(NDArrayConstructorsTests, test_constructor_9) {
    auto x = NDArrayFactory::create_<double>('c',{2, 2});

    ASSERT_TRUE(x->buffer() == nullptr);
    ASSERT_FALSE(x->specialBuffer() == nullptr);

    ASSERT_FALSE(x->shapeInfo() == nullptr);
    ASSERT_FALSE(x->specialShapeInfo() == nullptr);

    ASSERT_TRUE(x->isActualOnDeviceSide());
    ASSERT_FALSE(x->isActualOnHostSide());

    delete x;
}

TEST_F(NDArrayConstructorsTests, test_linspace_1) {
    auto x = NDArrayFactory::linspace<float>(1.0f, 10.0f, 20);

    ASSERT_FALSE(x->buffer() == nullptr);
    ASSERT_FALSE(x->specialBuffer() == nullptr);

    ASSERT_FALSE(x->shapeInfo() == nullptr);
    ASSERT_FALSE(x->specialShapeInfo() == nullptr);

    ASSERT_TRUE(x->isActualOnDeviceSide());
    ASSERT_TRUE(x->isActualOnHostSide());

    delete x;
}

TEST_F(NDArrayConstructorsTests, test_constructor_10) {

    NDArray scalar1(nd4j::DataType::DOUBLE); // scalar1 = 0
    NDArray scalar2('c', {0}, {0});

    ASSERT_TRUE(scalar1.isActualOnDeviceSide());
    ASSERT_TRUE(!scalar1.isActualOnHostSide());
    ASSERT_TRUE(scalar2.isActualOnDeviceSide());
    ASSERT_TRUE(scalar2.isActualOnHostSide());
    
    ASSERT_TRUE(scalar2.equalsTo(scalar1));
    
    ASSERT_TRUE(scalar1.isActualOnDeviceSide());
    ASSERT_TRUE(!scalar1.isActualOnHostSide());
    ASSERT_TRUE(scalar2.isActualOnDeviceSide());
    ASSERT_TRUE(scalar2.isActualOnHostSide());

    ASSERT_TRUE(scalar1.getBuffer() == nullptr);        
    ASSERT_TRUE(scalar1.getSpecialBuffer() != nullptr);
    ASSERT_TRUE(scalar1.getShapeInfo() != nullptr);
    ASSERT_TRUE(scalar1.getSpecialShapeInfo() != nullptr);
    ASSERT_TRUE(scalar1.lengthOf() == 1);
}