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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_clipbyavgnorm)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/transforms.h>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(clipbyavgnorm, -1, 1, true, -2, 0) {
  if (block.inputs()->size() > 1) {
    auto input = INPUT_VARIABLE(0);
    auto clipNorm = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);
    const bool isInplace = block.isInplace();
    helpers::clipByNorm(block.launchContext(), input, output, *block.getIArguments(), clipNorm, isInplace, true);
  } else {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    const bool isInplace = block.isInplace();
    auto clipNorm = NDArrayFactory::create(T_ARG(0), block.launchContext());

    helpers::clipByNorm(block.launchContext(), input, output, *block.getIArguments(), &clipNorm, isInplace, true);
  }

  return sd::Status::OK;
}

DECLARE_TYPES(clipbyavgnorm) {
  getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(clipbyavgnorm_bp, -2, 1, false, -1, 0) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);

  auto gradI = OUTPUT_VARIABLE(0);
  if (block.inputs()->size() > 2) {
    const auto clipNorm = INPUT_VARIABLE(2);
    helpers::clipByNormBp(block.launchContext(), input, gradO, gradI, *block.getIArguments(), clipNorm, true);
  } else {
    auto clipNorm = NDArrayFactory::create(gradI->dataType(), T_ARG(0), block.launchContext());
    helpers::clipByNormBp(block.launchContext(), input, gradO, gradI, *block.getIArguments(), &clipNorm, true);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(clipbyavgnorm_bp) {
  return SHAPELIST(CONSTANT(inputShape->at(1)));
}

DECLARE_TYPES(clipbyavgnorm_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, DataType::ANY)
      ->setAllowedInputTypes(1, {ALL_FLOATS})
      ->setAllowedOutputTypes(0, {ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif
