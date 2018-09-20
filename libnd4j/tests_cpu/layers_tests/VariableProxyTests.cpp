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
#include <graph/VariableProxy.h>

using namespace nd4j;
using namespace nd4j::graph;

class VariableProxyTests : public testing::Test {
public:

};


TEST_F(VariableProxyTests, Test_Simple_1) {
    auto x = NDArrayFactory::create<float>('c', {2, 2}, {1, 2, 3, 4});
    VariableSpace ref;

    ref.putVariable(119, x);

    ASSERT_TRUE(ref.hasVariable(119));

    VariableProxy proxy(&ref);

    ASSERT_TRUE(proxy.hasVariable(119));
}


TEST_F(VariableProxyTests, Test_Simple_2) {
    auto x = NDArrayFactory::create<float>('c', {2, 2}, {1, 2, 3, 4});
    VariableSpace ref;

    ASSERT_FALSE(ref.hasVariable(119));

    VariableProxy proxy(&ref);

    ASSERT_FALSE(proxy.hasVariable(119));

    proxy.putVariable(119, x);

    ASSERT_FALSE(ref.hasVariable(119));

    ASSERT_TRUE(proxy.hasVariable(119));
}


TEST_F(VariableProxyTests, Test_Simple_3) {
    auto x = NDArrayFactory::create<float>('c', {2, 2}, {1, 2, 3, 4});
    auto y = NDArrayFactory::create<float>('c', {2, 2}, {4, 2, 3, 1});
    VariableSpace ref;

    ref.putVariable(119, x);

    ASSERT_TRUE(ref.hasVariable(119));

    VariableProxy proxy(&ref);

    ASSERT_TRUE(proxy.hasVariable(119));

    proxy.putVariable(119, y);

    ASSERT_TRUE(ref.hasVariable(119));

    ASSERT_TRUE(proxy.hasVariable(119));

    auto z0 = ref.getVariable(119)->getNDArray();
    auto z1 = proxy.getVariable(119)->getNDArray();

    ASSERT_FALSE(z0 == z1);
    ASSERT_TRUE(y == z1);
    ASSERT_TRUE(x == z0);
}

TEST_F(VariableProxyTests, Test_Simple_4) {
    auto x = NDArrayFactory::create<float>('c', {2, 2}, {1, 2, 3, 4});
    auto y = NDArrayFactory::create<float>('c', {2, 2}, {4, 2, 3, 1});
    auto z = NDArrayFactory::create<float>('c', {2, 2}, {4, 1, 3, 2});
    VariableSpace ref;

    ref.putVariable(119, x);
    ref.putVariable(118, z);

    ASSERT_TRUE(ref.hasVariable(119));

    VariableProxy proxy(&ref);

    ASSERT_TRUE(proxy.hasVariable(119));

    proxy.putVariable(119, y);

    ASSERT_TRUE(ref.hasVariable(119));
    ASSERT_TRUE(ref.hasVariable(118));

    ASSERT_TRUE(proxy.hasVariable(119));
    ASSERT_TRUE(proxy.hasVariable(118));

    auto z0 = ref.getVariable(119)->getNDArray();
    auto z1 = proxy.getVariable(119)->getNDArray();

    ASSERT_FALSE(z0 == z1);
    ASSERT_TRUE(y == z1);
    ASSERT_TRUE(x == z0);
}


TEST_F(VariableProxyTests, Test_Cast_1) {
    auto x = NDArrayFactory::create<float>('c', {2, 2}, {1, 2, 3, 4});
    auto y = NDArrayFactory::create<float>('c', {2, 2}, {4, 2, 3, 1});
    VariableSpace ref;

    ref.putVariable(-119, x);

    ASSERT_TRUE(ref.hasVariable(-119));

    VariableProxy proxy(&ref);
    auto cast = (VariableSpace *) &proxy;

    ASSERT_TRUE(cast->hasVariable(-119));

    cast->putVariable(-119, y);

    ASSERT_TRUE(ref.hasVariable(-119));

    ASSERT_TRUE(cast->hasVariable(-119));

    auto z0 = ref.getVariable(-119)->getNDArray();
    auto z1 = cast->getVariable(-119)->getNDArray();

    ASSERT_FALSE(z0 == z1);
    ASSERT_TRUE(y == z1);
    ASSERT_TRUE(x == z0);
}


TEST_F(VariableProxyTests, Test_Clone_1) {
    auto x = NDArrayFactory::create<float>('c', {2, 2}, {1, 2, 3, 4});
    auto y = NDArrayFactory::create<float>('c', {2, 2}, {4, 2, 3, 1});
    VariableSpace ref;

    ref.putVariable(118, x);

    VariableProxy proxy(&ref);

    proxy.putVariable(119, y);

    ASSERT_TRUE(proxy.hasVariable(118));
    ASSERT_TRUE(proxy.hasVariable(119));

    auto clone = proxy.clone();

    ASSERT_TRUE(clone->hasVariable(118));
    ASSERT_TRUE(clone->hasVariable(119));

    delete clone;
}