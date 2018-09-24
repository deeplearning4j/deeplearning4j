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

#ifndef LIBND4J_STASHTESTS_H
#define LIBND4J_STASHTESTS_H

#include <NDArray.h>
#include "testlayers.h"
#include <graph/Stash.h>

using namespace nd4j;
using namespace nd4j::graph;

class StashTests : public testing::Test {
public:

};

TEST_F(StashTests, BasicTests_1) {
    Stash stash;

    auto alpha = NDArrayFactory::create_<float>('c',{5, 5});
    alpha->assign(1.0);

    auto beta = NDArrayFactory::create_<float>('c',{5, 5});
    beta->assign(2.0);

    auto cappa = NDArrayFactory::create_<float>('c',{5, 5});
    cappa->assign(3.0);

    stash.storeArray(1, "alpha", alpha);
    stash.storeArray(2, "alpha", beta);
    stash.storeArray(3, "cappa", cappa);

    ASSERT_TRUE(stash.checkStash(1, "alpha"));
    ASSERT_TRUE(stash.checkStash(2, "alpha"));
    ASSERT_TRUE(stash.checkStash(3, "cappa"));

    ASSERT_FALSE(stash.checkStash(3, "alpha"));
    ASSERT_FALSE(stash.checkStash(2, "beta"));
    ASSERT_FALSE(stash.checkStash(1, "cappa"));
}


TEST_F(StashTests, BasicTests_2) {
    Stash stash;

    auto alpha = NDArrayFactory::create_<float>('c',{5, 5});
    alpha->assign(1.0);

    auto beta = NDArrayFactory::create_<float>('c',{5, 5});
    beta->assign(2.0);

    auto cappa = NDArrayFactory::create_<float>('c',{5, 5});
    cappa->assign(3.0);

    stash.storeArray(1, "alpha1", alpha);
    stash.storeArray(1, "alpha2", beta);
    stash.storeArray(1, "alpha3", cappa);

    ASSERT_FALSE(stash.checkStash(2, "alpha1"));
    ASSERT_FALSE(stash.checkStash(2, "alpha2"));
    ASSERT_FALSE(stash.checkStash(2, "alpha3"));

    ASSERT_TRUE(alpha == stash.extractArray(1, "alpha1"));
    ASSERT_TRUE(beta == stash.extractArray(1, "alpha2"));
    ASSERT_TRUE(cappa == stash.extractArray(1, "alpha3"));

}

#endif //LIBND4J_STASHTESTS_H
