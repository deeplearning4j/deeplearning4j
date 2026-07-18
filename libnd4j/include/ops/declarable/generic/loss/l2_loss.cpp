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
// Created by GS <sgazeos@gmail.com> 31.01.2018
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_l2_loss)

#include <ops/declarable/headers/loss.h>
#include <helpers/logger.h>

namespace sd {
namespace ops {
CUSTOM_OP_IMPL(l2_loss, 1, 1, false, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(output->isScalar(), 0, "Rank output should be scalar");

  // l2_loss = sum(x^2) / 2
  input->reduceNumber(reduce::SquaredNorm, output);
  if (sd::env_isDebug())
    sd_printf("[L2LOSS_DIAG] input.rank=%i input.len=%lld after reduceNumber output=%.8f\n",
              input->rankOf(), (long long)input->lengthOf(), output->e<double>(0));
  double two = 2.0;
  output->applyScalar(scalar::Divide, two, output);
  if (sd::env_isDebug()) sd_printf("[L2LOSS_DIAG] after divide-by-2 output=%.8f\n", output->e<double>(0));

  return Status::OK;
}
DECLARE_SHAPE_FN(l2_loss) {
  return SHAPELIST(ConstantShapeHelper::getInstance().scalarShapeInfo(ArrayOptions::dataType(inputShape->at(0))));
}

DECLARE_TYPES(l2_loss) {
  getOpDescriptor()->addTraits(OP_TRAIT_REDUCTION | OP_TRAIT_FULLY_WRITING);
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}
}  // namespace ops
}  // namespace sd

#endif
