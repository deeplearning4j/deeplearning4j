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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 02.11.2017
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_reverse)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/reverse.h>

namespace sd {
namespace ops {

CONFIGURABLE_OP_IMPL(reverse, 1, 1, true, 0, -2) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  if (output->isEmpty()) {
    // No-op
    return Status::OK;
  }

  std::vector<LongType> axis;

  if (block.width() > 1)
    axis = INPUT_VARIABLE(1)->template asVectorT<LongType>();
  else if (block.numI() > 0)
    axis = *block.getIArguments();

  if (axis.empty()) {  // do not perform reversion
    if (!block.isInplace()) output->assign(input);
  } else {
    // check the consistency of input dimensions to reverse along
    shape::checkDimensions(input->rankOf(), &axis);
    helpers::reverse(block.launchContext(), input, output, &axis);
  }

  return Status::OK;
}

DECLARE_SYN(reverse_v2, reverse);

DECLARE_TYPES(reverse) {
  getOpDescriptor()->setAllowedInputTypes(0, ANY);
  getOpDescriptor()->setAllowedInputTypes(1, {INT32, INT64});
  getOpDescriptor()->setAllowedOutputTypes(0, INHERIT);
}

CUSTOM_OP_IMPL(reverse_bp, 2, 1, false, 0, -2) {
  auto input = INPUT_VARIABLE(0);
  auto eps = block.width() == 3 ? INPUT_VARIABLE(2) : INPUT_VARIABLE(1);

  auto output = OUTPUT_VARIABLE(0);
  std::vector<LongType> axis;

  if (block.width() == 3)
    axis = INPUT_VARIABLE(1)->template asVectorT<LongType>();
  else if (block.numI() > 0)
    axis = *block.getIArguments();

  if (axis.empty()) {  // reversion is not performed in this case
    output->assign(eps);
  } else {
    // check the consistency of input dimensions to reverse along
    shape::checkDimensions(input->rankOf(), &axis);
    // we just reverse back original array
    helpers::reverse(block.launchContext(), eps, output, &axis);
  }

  return Status::OK;
}

DECLARE_TYPES(reverse_bp) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(reverse_bp) {
  auto in = inputShape->at(0);
  LongType *out;
  COPY_SHAPE(in, out);

  return SHAPELIST(CONSTANT(out));
}

}  // namespace ops
}  // namespace sd

#endif
