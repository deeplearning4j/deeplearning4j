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
// @author raver119@gmail.com
//
#pragma  once
#if !defined(LIBND4J_TESTLAYERS_H)
#define LIBND4J_TESTLAYERS_H
#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <cnpy/cnpy.h>
#include <graph/GraphExecutioner.h>
#include <graph/Node.h>
#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <gtest/gtest.h>
#include <helpers/shape.h>
#include <legacy/NativeOps.h>
#include <memory/MemoryTracker.h>
#include <ops/gemm.h>
#include <ops/ops.h>
#include <system/common.h>

#include <array>
class NDArrayTests : public testing::Test {
  inline static std::map<std::string, std::vector<sd::NDArray*>> arrays;

 protected:
  sd::NDArray* registerArr(sd::NDArray arr) {
    auto ret = new sd::NDArray(arr);
    auto const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    NDArrayTests::arrays[std::string(test_info->name())].push_back(ret);
    return ret;
  }
  void SetUp() override {
    Test::SetUp();
    auto const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    arrays[std::string(test_info->name())] = std::vector<sd::NDArray*>();
  }

  void TearDown() override {
    Test::TearDown();
    auto const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    // delete any existing memory not found in the current test
    // this is to avoid deleting any memory that may or may not be asynchronously used
    // by cuda and prevents issues when running only 1 test
    std::vector<std::string> keysToDelete;
    for (auto it = arrays.begin(); it != arrays.end(); it++) {
      if (std::string(test_info->name()) != std::string(it->first)) {
        sd_printf("Deleting for test name %s\n", test_info->name());
        for (auto arr : it->second) {
          delete arr;
        }

        keysToDelete.push_back(it->first);
      }
    }

    for (auto key : keysToDelete) {
      arrays.erase(key);
    }
  }
};
#endif  // LIBND4J_TESTLAYERS_H
