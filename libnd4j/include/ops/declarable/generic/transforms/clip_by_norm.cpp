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
#if NOT_EXCLUDED(OP_clipbynorm)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/transforms.h>

namespace sd {
namespace ops {

CONFIGURABLE_OP_IMPL(clipbynorm, 1, 1, true, 1, 0) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);
  if (block.numT() > 0) {
    auto clipNorm = NDArrayFactory::create(output->dataType(), T_ARG(0), block.launchContext());
    const bool isInplace = block.isInplace();
    helpers::clipByNorm(block.launchContext(), input, output, *block.getIArguments(), &clipNorm, isInplace, false);
  } else {
    auto clipNorm = INPUT_VARIABLE(1);
    const bool isInplace = block.isInplace();
    helpers::clipByNorm(block.launchContext(), input, output, *block.getIArguments(), clipNorm, isInplace, false);
  }

  return Status::OK;
}

CUSTOM_OP_IMPL(clipbynorm_bp, 2, 1, false, 1, 0) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);

  auto gradI = OUTPUT_VARIABLE(0);
  if (block.numT() > 0) {
    auto clipNorm = NDArrayFactory::create(gradI->dataType(), T_ARG(0), block.launchContext());
    helpers::clipByNormBp(block.launchContext(), input, gradO, gradI, *block.getIArguments(), &clipNorm, false);
  } else {
    const auto clipNorm = INPUT_VARIABLE(1);
    helpers::clipByNormBp(block.launchContext(), input, gradO, gradI, *block.getIArguments(), clipNorm, false);
  }

  return Status::OK;
}

DECLARE_SHAPE_FN(clipbynorm_bp) {
  auto inShapeInfo = inputShape->at(1);

  LongType *newShape = nullptr;
  COPY_SHAPE(inShapeInfo, newShape);

  return SHAPELIST(CONSTANT(newShape));
}

DECLARE_TYPES(clipbynorm) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_TYPES(clipbynorm_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, ANY)
      ->setAllowedInputTypes(1, {ALL_FLOATS})
      ->setAllowedOutputTypes(0, {ALL_FLOATS});
}
}  // namespace ops
}  // namespace sd

#endif
