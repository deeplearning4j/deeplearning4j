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
//  @author raver119@gmail.com
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_set_seed)

#include <legacy/NativeOps.h>
#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {
CUSTOM_OP_IMPL(set_seed, -2, 1, false, 0, -2) {
  auto rng = block.getRng();  //.getRNG();

  LongType seed = 0;
  if (block.getIArguments()->size() > 0) {
    seed = INT_ARG(0);
  } else if (block.width() > 0) {
    auto input = INPUT_VARIABLE(0);
    REQUIRE_TRUE(input->isScalar(), 0, "SetSeed: Seed operand should be scalar");
    seed = input->e<LongType>(0);
  } else {
    REQUIRE_TRUE(false, 0, "SetSeed: either IArg or scalr input should be provided");
  }

  // FIXME: this approach isn't really good for cuda, since it'll assume that CUDA might get nullptr instead of stream
  // refreshBuffer(nullptr, seed, (sd::Pointer) rng);
  rng.setSeed((int)seed);
  return Status::OK;
}

DECLARE_SHAPE_FN(set_seed) {
  auto newshape = ConstantShapeHelper::getInstance().scalarShapeInfo(block.dataType());
  return SHAPELIST(newshape);
}

DECLARE_TYPES(set_seed) { getOpDescriptor()->setAllowedInputTypes({ALL_INTS})->setAllowedOutputTypes({ALL_FLOATS}); }
}  // namespace ops
}  // namespace sd

#endif
