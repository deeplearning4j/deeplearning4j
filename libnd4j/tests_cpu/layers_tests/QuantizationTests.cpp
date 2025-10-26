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
// @author raver119@protonmail.com
//

#include <array/NDArray.h>
#include <loops/type_conversions.h>

#include "testlayers.h"

using namespace sd;

class QuantizationTests : public NDArrayTests {};

TEST_F(QuantizationTests, Basic_Test_1) {
#ifndef SD_CUDA
  auto s = TypeCast::estimateQuantizedSize(10);
  ASSERT_EQ(18, s);
#endif
}

TEST_F(QuantizationTests, Basic_Test_2) {
#ifndef SD_CUDA
  auto s = TypeCast::estimateQuantizedSize(1);
  ASSERT_EQ(9, s);
#endif
}

TEST_F(QuantizationTests, Compression_Test_1) {
#ifndef SD_CUDA

  auto x = NDArrayFactory::create<float>('c', {10});
  auto z = NDArrayFactory::create<float>('c', {10});
  x.linspace(1.0f);

  auto q = new char[TypeCast::estimateQuantizedSize(x.lengthOf())];

  TypeCast::convertToQuantized<float>(nullptr, x.buffer(), x.lengthOf(), q);
  TypeCast::convertFromQuantized<float>(nullptr, q, x.lengthOf(), z.buffer());

  ASSERT_TRUE(x.equalsTo(z, 0.1));

  auto fq = reinterpret_cast<float *>(q);

  ASSERT_NEAR(1.0f, fq[0], 1e-5);
  ASSERT_NEAR(10.0f, fq[1], 1e-5);

  delete[] q;

#endif
}
