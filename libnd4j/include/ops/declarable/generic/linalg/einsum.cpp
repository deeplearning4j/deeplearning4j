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

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_einsum)

#include <helpers/ConstantShapeHelper.h>
#include <ops/declarable/headers/transforms.h>
#include <ops/declarable/helpers/einsum.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(einsum, -2, 1, false, 0, 0) {
  auto sArgs = block.getSArguments();
  REQUIRE_TRUE(sArgs != nullptr && !sArgs->empty(), 0,
               "EINSUM: equation string must be provided as a string argument");

  const std::string& equation = sArgs->at(0);

  std::vector<NDArray*> inputs;
  for (int i = 0; i < block.width(); i++) {
    inputs.push_back(INPUT_VARIABLE(i));
  }

  auto output = OUTPUT_VARIABLE(0);

  helpers::einsum(block.launchContext(), equation, inputs, *output);

  return Status::OK;
}

DECLARE_TYPES(einsum) {

  getOpDescriptor()
      ->setAllowedInputTypes({ALL_FLOATS, ALL_INTS})
      ->setAllowedOutputTypes({ALL_FLOATS, ALL_INTS});
  getOpDescriptor()->addTraits(OP_TRAIT_FULLY_WRITING | OP_TRAIT_DATA_DEPENDENT);
}

DECLARE_SHAPE_FN(einsum) {
  auto sArgs = block.getSArguments();
  REQUIRE_TRUE(sArgs != nullptr && !sArgs->empty(), 0,
               "EINSUM: equation string must be provided as a string argument");

  const std::string& equation = sArgs->at(0);

  std::vector<const LongType*> inputShapeInfos;
  for (int i = 0; i < block.width(); i++) {
    inputShapeInfos.push_back(inputShape->at(i));
  }

  auto outShape = helpers::einsumOutputShape(equation, inputShapeInfos);
  auto dtype = ArrayOptions::dataType(inputShape->at(0));

  if (outShape.empty()) {
    // Scalar output
    return SHAPELIST(ConstantShapeHelper::getInstance().scalarShapeInfo(dtype));
  }

  return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c', outShape));
}

}  // namespace ops
}  // namespace sd

#endif
