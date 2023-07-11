/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author raver119@gmail.com
//
#include <array/NDArray.h>
#include <helpers/GradCheck.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/ops.h>

#include <array>

#include "testlayers.h"

using namespace sd;

class DeclarableOpsTests17 : public NDArrayTests {
 public:
  DeclarableOpsTests17() {
    printf("\n");
    fflush(stdout);
  }
};

TEST_F(DeclarableOpsTests17, test_sparse_to_dense_1) {
  auto values = NDArrayFactory::create<float>({1.f, 2.f, 3.f});
  auto shape = NDArrayFactory::create<sd::LongType>({3, 3});
  auto ranges = NDArrayFactory::create<sd::LongType>({0, 0, 1, 1, 2, 2});
  auto def = NDArrayFactory::create<float>(0.f);
  auto exp = NDArrayFactory::create<float>('c', {3, 3}, {1.f, 0.f, 0.f, 0.f, 2.f, 0.f, 0.f, 0.f, 3.f});

  sd::ops::compat_sparse_to_dense op;
  auto result = op.evaluate({&ranges, &shape, &values, &def});
  ASSERT_EQ(sd::Status::OK, result.status());
}

TEST_F(DeclarableOpsTests17, test_sparse_to_dense_2) {
  std::vector<std::string> data = {"alpha", "beta", "gamma"};
  auto values = NDArrayFactory::string({3}, data);
  auto shape = NDArrayFactory::create<sd::LongType>({3, 3});
  auto ranges = NDArrayFactory::create<sd::LongType>({0, 0, 1, 1, 2, 2});
  auto def = NDArrayFactory::string("d");
  std::vector<std::string> data2 =  {"alpha", "d", "d", "d", "beta", "d", "d", "d", "gamma"};
  auto exp = NDArrayFactory::string({3, 3},data2);

  sd::ops::compat_sparse_to_dense op;
  auto result = op.evaluate({&ranges, &shape, &values, &def});
  ASSERT_EQ(sd::Status::OK, result.status());
}

