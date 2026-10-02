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
#include <ops/declarable/headers/random.h>
#include <ops/declarable/helpers/random.h>

namespace sd {
namespace ops {
CUSTOM_OP_IMPL(set_seed, -2, 1, false, 0, -2) {
  auto& rng = block.randomGenerator();

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

  // Seeds the context's generator the way a random op's own seed argument does (0 leaves it as it
  // is); the caller hands the state on to later random ops (SameDiff to Nd4j.getRandom()).
  helpers::applySeedArgument(rng, seed);
  double written = static_cast<double>(seed);
  OUTPUT_VARIABLE(0)->assign(written);
  return Status::OK;
}

DECLARE_SHAPE_FN(set_seed) {
  // The output is the seed as a float scalar. The seed may come from an IArg alone, so the
  // type is the DataType argument's, not an input's.
  const DataType dtype = block.numD() > 0 ? D_ARG(0) : FLOAT32;
  return SHAPELIST(ConstantShapeHelper::getInstance().scalarShapeInfo(dtype));
}

DECLARE_TYPES(set_seed) {
  getOpDescriptor()->setAllowedInputTypes({ALL_INTS})->setAllowedOutputTypes({ALL_FLOATS});
  getOpDescriptor()->addTraits(OP_TRAIT_FULLY_WRITING | OP_TRAIT_STATEFUL);
}
}  // namespace ops
}  // namespace sd

#endif
