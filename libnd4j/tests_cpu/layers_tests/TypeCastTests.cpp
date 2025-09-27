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
// Created by raver119 on 02/07/18.
//
#include <loops/type_conversions.h>
#include <ops/declarable/CustomOperations.h>

#include "testlayers.h"

using namespace sd;
using namespace sd::ops;
using namespace sd::graph;

class TypeCastTests : public NDArrayTests {
 public:
};

TEST_F(TypeCastTests, Test_Cast_1) {
#ifndef SD_CUDA
  const int limit = 100;
  auto src = new double[limit];
  auto z = new float[limit];
  auto exp = new float[limit];

  for (int e = 0; e < limit; e++) {
    src[e] = static_cast<double>(e);
    exp[e] = static_cast<float>(e);
  }

  TypeCast::convertGeneric<double, float>(nullptr, reinterpret_cast<void *>(src), limit, reinterpret_cast<void *>(z));

  for (int e = 0; e < limit; e++) {
    ASSERT_NEAR(exp[e], z[e], 1e-5f);
  }

  delete[] src;
  delete[] z;
  delete[] exp;
#endif
}

TEST_F(TypeCastTests, Test_ConvertDtype_1) {
#ifndef SD_CUDA

  float src[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  float16 dst[5];
  float16 exp[] = {(float16)1.0f, (float16)2.0f, (float16)3.0f, (float16)4.0f, (float16)5.0f};

  convertTypes(nullptr, ND4J_FLOAT32, src, 5, ND4J_FLOAT16, dst);

  for (int e = 0; e < 5; e++) ASSERT_NEAR(exp[e], dst[e], (float16)0.01f);

#endif
}
